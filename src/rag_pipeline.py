"""
Ensamblado del pipeline RAG completo con LCEL.

Flujo objetivo (retriever | prompt | model | parser) enriquecido con pasos
adicionales exigidos por el caso de auditoría financiera:

    pregunta -> retriever (FAISS, fuente INTERNA)
             -> LLM-as-a-judge (filtra chunks)
             -> + indicadores del día (mindicador.cl, fuente EXTERNA)
             -> prompt (few-shot + CoT + guardrails XML)
             -> Groq -> parser Pydantic -> InformeAuditoria

El LLM-as-a-judge existe porque en auditoría un solo chunk irrelevante
"colado" en el contexto puede inducir al modelo a mezclar cifras de
documentos distintos (p.ej. sumar el monto de una factura no relacionada).
Se prefiere sacrificar algo de recall por precisión en este dominio.

La fuente EXTERNA (`src/external_data.py`) se combina con la interna en el
mismo bloque `<contexto_auditoria>`, pero con su propia etiqueta
("[FUENTE EXTERNA: mindicador.cl | ...]") para que quede claro cuál dato
viene de los documentos auditados y cuál de un servicio externo — requisito
IL1.2/IE3 de "combinar fuentes de datos internas y externas".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import get_chat_model, settings
from src.external_data import (
    IndicadoresExternos,
    fetch_indicadores_economicos,
    format_external_context,
)
from src.loaders import load_faiss_index
from src.metrics import AuditCallbackHandler, InformeAuditoria
from src.prompts import AUDIT_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE

_parser = PydanticOutputParser(pydantic_object=InformeAuditoria)


def format_docs(docs: list[Document]) -> str:
    """Concatena chunks en un bloque de contexto citable.

    Cada fragmento se antecede de su fuente y chunk_id, de modo que las
    "citas_textuales" que exige el esquema Pydantic sean rastreables hasta
    el documento y fragmento exacto (requisito de trazabilidad, IE1/IE5).
    """
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "desconocido")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[fuente={source} | chunk_id={chunk_id}]\n{doc.page_content}")
    return "\n\n".join(parts)


def merge_context(internal_context: str, external_context: str) -> str:
    """Combina el contexto INTERNO (chunks de FAISS) con el bloque de la
    fuente EXTERNA (mindicador.cl) en un solo texto citable para el prompt.

    Función separada (en vez de un f-string inline) para poder testearla de
    forma aislada y determinista, sin necesitar LLM ni embeddings — ver
    `tests/test_external_data.py`.
    """
    return f"{internal_context}\n\n{external_context}"


def _parse_score(raw_text: str) -> float:
    """Extrae el primer número entero/decimal de la respuesta del juez."""
    match = re.search(r"-?\d+(\.\d+)?", raw_text)
    return float(match.group()) if match else 0.0


@dataclass
class RagRunResult:
    """Resultado enriquecido de una ejecución de `AuditRagPipeline.run`."""

    informe: InformeAuditoria
    retrieved_chunk_ids: list[int]
    kept_chunk_ids: list[int]
    context_text: str
    callback: AuditCallbackHandler
    external_data_available: bool


class AuditRagPipeline:
    """Orquesta el flujo RAG completo: retriever -> LLM-judge -> prompt -> LLM -> Pydantic.

    Se implementa como clase (en vez de una única cadena LCEL plana) porque
    el paso de LLM-as-a-judge necesita examinar CADA chunk recuperado de
    forma independiente antes de decidir cuáles pasan al prompt final, y
    porque se requiere exponer `retrieved_chunk_ids` / `kept_chunk_ids` para
    las métricas de `context_precision` / `context_recall` (`src/metrics.py`).
    Internamente, cada paso individual (retriever, judge, prompt|llm|parser)
    SÍ se construye y ejecuta como Runnables LCEL; ver también
    `build_rag_chain()` para la variante plana sin juez.
    """

    def __init__(self, k: int | None = None, judge_min_score: float | None = None) -> None:
        self.vectorstore = load_faiss_index()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k or settings.retriever_k})
        self.llm = get_chat_model()
        self.judge_chain = JUDGE_PROMPT_TEMPLATE | self.llm
        self.answer_chain = (
            AUDIT_PROMPT_TEMPLATE.partial(format_instructions=_parser.get_format_instructions())
            | self.llm
            | _parser
        )
        self.judge_min_score = judge_min_score if judge_min_score is not None else settings.judge_min_score
        # Los indicadores económicos (fuente EXTERNA) cambian a lo sumo una
        # vez al día, así que se cachean por instancia del pipeline en vez
        # de golpear mindicador.cl en cada pregunta. `None` = aún no
        # consultado; se resuelve de forma perezosa en el primer `run()`.
        self._external_cache: IndicadoresExternos | None = None

    def _get_external_context(self) -> tuple[str, bool]:
        """Consulta (con caché) la fuente externa y retorna su bloque de contexto."""
        if self._external_cache is None:
            self._external_cache = fetch_indicadores_economicos()
        return format_external_context(self._external_cache), self._external_cache.disponible

    def _judge_filter(self, question: str, docs: list[Document], config: dict) -> list[Document]:
        """Filtra chunks recuperados usando el LLM como juez de relevancia (0-10)."""
        kept: list[Document] = []
        for doc in docs:
            response = self.judge_chain.invoke({"question": question, "chunk": doc.page_content}, config=config)
            score = _parse_score(getattr(response, "content", str(response)))
            if score >= self.judge_min_score:
                kept.append(doc)
        return kept

    def run(self, question: str) -> RagRunResult:
        callback = AuditCallbackHandler(model_name=self._model_name())
        config = {"callbacks": [callback]}

        retrieved_docs = self.retriever.invoke(question, config=config)
        retrieved_ids = [d.metadata.get("chunk_id") for d in retrieved_docs]

        kept_docs = self._judge_filter(question, retrieved_docs, config)
        if not kept_docs:
            # Si el juez descarta todo, se preserva el top-k original para no
            # dejar al generador sin contexto: es preferible que responda
            # "No dispongo de esta información..." a que la cadena falle.
            kept_docs = retrieved_docs
        kept_ids = [d.metadata.get("chunk_id") for d in kept_docs]

        internal_context = format_docs(kept_docs)
        external_context, external_available = self._get_external_context()
        context_text = merge_context(internal_context, external_context)

        informe = self.answer_chain.invoke({"context": context_text, "question": question}, config=config)

        return RagRunResult(
            informe=informe,
            retrieved_chunk_ids=retrieved_ids,
            kept_chunk_ids=kept_ids,
            context_text=context_text,
            callback=callback,
            external_data_available=external_available,
        )

    def _model_name(self) -> str:
        return getattr(self.llm, "model", getattr(self.llm, "model_name", "desconocido"))


def build_rag_chain():
    """Variante LCEL plana: `retriever | prompt | model | parser`, sin el
    filtro LLM-judge, expuesta por completitud respecto del patrón canónico
    enseñado en el curso. Para el flujo completo con juez y métricas, usar
    `AuditRagPipeline`.

    También combina fuente interna (FAISS) + externa (mindicador.cl), igual
    que `AuditRagPipeline`, para que ambas variantes sean consistentes.
    """
    vectorstore = load_faiss_index()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
    llm = get_chat_model()

    def build_context(question: str) -> str:
        internal_context = format_docs(retriever.invoke(question))
        external_context = format_external_context(fetch_indicadores_economicos())
        return merge_context(internal_context, external_context)

    chain = (
        {
            "context": RunnableLambda(build_context),
            "question": RunnablePassthrough(),
        }
        | AUDIT_PROMPT_TEMPLATE.partial(format_instructions=_parser.get_format_instructions())
        | llm
        | _parser
    )
    return chain


if __name__ == "__main__":
    pipeline = AuditRagPipeline()
    result = pipeline.run(
        "¿Cuál es el monto total y el nivel de riesgo detectado en los documentos auditados?"
    )
    print(result.informe.model_dump_json(indent=2))
    print("Métricas de la llamada:", result.callback.summary())
    print("Fuente externa disponible:", result.external_data_available)
