"""
Ingesta y fragmentación de documentos contables + construcción del índice FAISS.

Flujo:
    data/raw/*.{pdf,txt} -> loaders -> splitter (con metadatos) -> embeddings
    -> índice FAISS persistido en vectorstore/

El chunking está pensado para NO truncar tablas financieras a la mitad: se
usa una lista de separadores jerárquica que prioriza cortar en saltos de
párrafo/línea antes que en medio de una fila "Concepto ... Monto".
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    DATA_RAW_DIR,
    FAISS_INDEX_NAME,
    VECTORSTORE_DIR,
    get_embeddings,
    settings,
)

# Palabras clave -> tipo_documento. Se evalúan sobre el nombre de archivo en
# minúsculas; el primer match gana. Es una heurística simple y explícita,
# preferible a "adivinar" con el propio LLM (evita costo/latencia extra en
# ingesta y mantiene la trazabilidad determinista de la metadata).
_TIPO_DOCUMENTO_KEYWORDS: dict[str, str] = {
    "factura": "factura_electronica",
    "nota_credito": "nota_credito",
    "notacredito": "nota_credito",
    "estado_resultado": "estado_de_resultados",
    "balance": "balance_trimestral",
    "riesgo": "informe_de_riesgo",
}

# Separadores en orden de preferencia: primero intenta cortar entre
# secciones/párrafos, luego líneas, luego frases. Como último recurso corta
# por espacio, nunca a mitad de un token numérico.
_FINANCIAL_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "; ", ", ", " "]


def _infer_tipo_documento(filename: str) -> str:
    """Infiera el tipo de documento contable a partir del nombre de archivo."""
    lowered = filename.lower()
    for keyword, tipo in _TIPO_DOCUMENTO_KEYWORDS.items():
        if keyword in lowered:
            return tipo
    return "documento_general"


def load_raw_documents(raw_dir: Path = DATA_RAW_DIR) -> list[Document]:
    """Carga todos los .pdf y .txt de `raw_dir` como `Document` de LangChain.

    Cada documento recibe metadata base (`source`, `tipo_documento`,
    `fecha_ingesta`) que luego se propaga a cada chunk. `page` la añaden los
    loaders de PDF automáticamente; para .txt se fija en 1.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de documentos: {raw_dir}")

    fecha_ingesta = _dt.date.today().isoformat()
    documents: list[Document] = []

    paths: Iterable[Path] = sorted(raw_dir.glob("*"))
    for path in paths:
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif path.suffix.lower() in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
            for doc in loaded:
                doc.metadata.setdefault("page", 1)
        else:
            continue

        tipo_documento = _infer_tipo_documento(path.name)
        for doc in loaded:
            doc.metadata["source"] = path.name
            doc.metadata["tipo_documento"] = tipo_documento
            doc.metadata["fecha_ingesta"] = fecha_ingesta
            documents.append(doc)

    if not documents:
        raise FileNotFoundError(
            f"No se encontraron documentos .pdf/.txt en {raw_dir}. "
            "Agrega al menos un archivo antes de construir el índice."
        )
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Fragmenta documentos preservando metadatos y cuidando tablas financieras.

    Se usa `RecursiveCharacterTextSplitter` con separadores jerárquicos
    (ver `_FINANCIAL_SEPARATORS`) y overlap suficiente (default 200 chars)
    para que una fila de tabla "Concepto | Monto" partida entre dos chunks
    siempre aparezca completa en, al menos, uno de ellos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=_FINANCIAL_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Metadata adicional de trazabilidad a nivel de chunk (requisito IE3/IE5:
    # cada dato debe poder rastrearse hasta su fuente exacta).
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def build_faiss_index(
    chunks: list[Document],
    persist_dir: Path = VECTORSTORE_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """Genera embeddings para `chunks` y persiste el índice FAISS en disco."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir), index_name=index_name)
    return vectorstore


def load_faiss_index(
    persist_dir: Path = VECTORSTORE_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """Carga un índice FAISS previamente persistido."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(persist_dir),
        embeddings,
        index_name=index_name,
        # Seguro en este proyecto: el índice lo genera nuestro propio pipeline,
        # nunca se carga un .faiss de origen externo/no confiable.
        allow_dangerous_deserialization=True,
    )


def ingest_and_index(
    raw_dir: Path = DATA_RAW_DIR,
    persist_dir: Path = VECTORSTORE_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """Pipeline de ingesta completo: cargar -> fragmentar -> indexar."""
    documents = load_raw_documents(raw_dir)
    chunks = split_documents(documents)
    return build_faiss_index(chunks, persist_dir=persist_dir, index_name=index_name)


if __name__ == "__main__":
    vs = ingest_and_index()
    print(f"Índice FAISS construido con {vs.index.ntotal} vectores en {VECTORSTORE_DIR}")
