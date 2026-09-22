"""
Pruebas de ingesta y chunking (src/loaders.py) que NO requieren credenciales
de LLM: `load_raw_documents` y `split_documents` no llaman a ningún proveedor
de embeddings. Solo `build_faiss_index` / `ingest_and_index` lo hacen, y por
eso quedan fuera de este archivo (ver README.md, sección "Tests").
"""

from __future__ import annotations

from src.config import DATA_RAW_DIR
from src.loaders import _infer_tipo_documento, load_raw_documents, split_documents


def test_infer_tipo_documento_factura() -> None:
    assert _infer_tipo_documento("factura_electronica_F1023.txt") == "factura_electronica"


def test_infer_tipo_documento_nota_credito() -> None:
    assert _infer_tipo_documento("nota_credito_NC045.txt") == "nota_credito"


def test_infer_tipo_documento_riesgo() -> None:
    assert _infer_tipo_documento("informe_riesgo_Q2_2025.txt") == "informe_de_riesgo"


def test_infer_tipo_documento_fallback() -> None:
    assert _infer_tipo_documento("archivo_random.txt") == "documento_general"


def test_load_raw_documents_returns_metadata() -> None:
    docs = load_raw_documents(DATA_RAW_DIR)
    assert len(docs) >= 3
    for doc in docs:
        assert "source" in doc.metadata
        assert "tipo_documento" in doc.metadata
        assert "fecha_ingesta" in doc.metadata
        assert "page" in doc.metadata


def test_split_documents_preserves_metadata_and_adds_chunk_id() -> None:
    docs = load_raw_documents(DATA_RAW_DIR)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    assert len(chunks) >= len(docs)  # al menos un chunk por documento
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "source" in chunk.metadata
        assert "tipo_documento" in chunk.metadata


def test_split_documents_does_not_exceed_chunk_size_by_much() -> None:
    """El splitter debe respetar aproximadamente chunk_size; permite margen
    para separadores largos, pero no debería duplicar el tamaño configurado.
    """
    docs = load_raw_documents(DATA_RAW_DIR)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    for chunk in chunks:
        assert len(chunk.page_content) <= 300 * 1.5
