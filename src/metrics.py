"""
Esquemas Pydantic de salida, callback de observabilidad (latencia/tokens/costo)
y métricas de evaluación RAG deterministas (IE5, IE6).

Las 4 métricas oficiales de RAG vistas en el curso se calculan aquí de forma
determinista y sin dependencias externas (no se integra RAGAS/LangSmith en
este proyecto), pero se usan los MISMOS NOMBRES y una firma de datos
compatible con esos frameworks, para que migrar a una integración real sea
un cambio de implementación interna, no de interfaz:

- context_precision: de los chunks recuperados, ¿cuántos eran relevantes?
- context_recall: de los chunks relevantes que existían, ¿cuántos se
  recuperaron?
- faithfulness: de las afirmaciones de la respuesta, ¿cuántas están
  respaldadas por el contexto (según fact_checking.py)?
- answer_relevancy: ¿la respuesta aborda directamente la pregunta? (proxy
  léxico determinista: similitud coseno TF entre pregunta y respuesta;
  ver nota de diseño en la función).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Esquema Pydantic estricto del reporte de auditoría (salida forzada del LLM)
# --------------------------------------------------------------------------- #
NivelRiesgo = Literal["bajo", "medio", "alto", "no_determinable"]


class InformeAuditoria(BaseModel):
    """Salida estructurada y estricta que debe producir la cadena principal."""

    monto_total: float | None = Field(
        default=None, description="Monto total identificado en el contexto, en unidades de `divisa`."
    )
    divisa: str | None = Field(default=None, description="Código de divisa, p.ej. CLP, USD.")
    rut_emisor: str | None = Field(default=None, description="RUT del emisor/receptor relevante, si aplica.")
    nivel_riesgo_fraude: NivelRiesgo = Field(
        description="Veredicto de riesgo de fraude tras el razonamiento paso a paso."
    )
    alertas_detectadas: list[str] = Field(
        default_factory=list, description="Señales de riesgo o inconsistencias detectadas."
    )
    citas_textuales: list[str] = Field(
        default_factory=list,
        description="Fragmentos textuales exactos del contexto que respaldan la respuesta.",
    )


class ClaimVerification(BaseModel):
    """Una afirmación individual verificada por la cadena de fact-checking."""

    afirmacion: str
    veredicto: Literal["Soportado", "Contradictorio", "No Mencionado"]
    evidencia: str = Field(description='Cita textual exacta, o "sin evidencia" si No Mencionado.')


class FactCheckReport(BaseModel):
    """Salida completa de la cadena de fact-checking para una respuesta."""

    afirmaciones: list[ClaimVerification] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Callback de observabilidad: latencia, tokens y costo estimado por llamada
# --------------------------------------------------------------------------- #

# Tabla de referencia de precios (USD por 1K tokens). Son valores de
# referencia para ILUSTRAR la metodología de costeo, no una fuente de verdad
# de pricing vigente: en producción esto debería leerse de una config externa
# actualizable sin tocar código.
PRICING_USD_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.00030},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.00500},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "_default": {"input": 0.0005, "output": 0.0015},
}


@dataclass
class CallRecord:
    """Registro de una única llamada al LLM."""

    model: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    timestamp: float = field(default_factory=time.time)


class AuditCallbackHandler(BaseCallbackHandler):
    """Callback de LangChain que registra latencia, tokens y costo por llamada.

    Uso:
        handler = AuditCallbackHandler(model_name=settings.google_chat_model)
        chain.invoke(payload, config={"callbacks": [handler]})
        print(handler.summary())
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.records: list[CallRecord] = []
        self._start_times: dict[str, float] = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id"))
        self._start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id"))
        start = self._start_times.pop(run_id, time.perf_counter())
        latency = time.perf_counter() - start

        input_tokens, output_tokens = self._extract_token_usage(response)
        cost = self._estimate_cost(input_tokens, output_tokens)

        self.records.append(
            CallRecord(
                model=self.model_name,
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            )
        )

    def _extract_token_usage(self, response: LLMResult) -> tuple[int, int]:
        """Extrae tokens de entrada/salida desde `llm_output` o `usage_metadata`.

        Distintos proveedores exponen el conteo en lugares distintos
        (`llm_output['token_usage']` estilo OpenAI, o `usage_metadata` en las
        generaciones estilo Gemini). Se intentan ambos y se cae a 0 si no hay
        información disponible (mejor un costo subestimado y visible que una
        excepción que tumbe el pipeline).
        """
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
        if usage:
            return (
                int(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
                int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
            )

        for generation_list in response.generations:
            for generation in generation_list:
                info = getattr(generation, "generation_info", None) or {}
                gen_usage = info.get("usage_metadata")
                if gen_usage:
                    return (
                        int(gen_usage.get("prompt_token_count", 0)),
                        int(gen_usage.get("candidates_token_count", 0)),
                    )
        return (0, 0)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING_USD_PER_1K_TOKENS.get(self.model_name, PRICING_USD_PER_1K_TOKENS["_default"])
        return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

    def summary(self) -> dict[str, float]:
        """Resumen agregado de todas las llamadas registradas hasta ahora."""
        if not self.records:
            return {
                "num_llamadas": 0,
                "latencia_promedio_s": 0.0,
                "tokens_entrada_total": 0,
                "tokens_salida_total": 0,
                "costo_estimado_total_usd": 0.0,
            }
        return {
            "num_llamadas": len(self.records),
            "latencia_promedio_s": sum(r.latency_seconds for r in self.records) / len(self.records),
            "tokens_entrada_total": sum(r.input_tokens for r in self.records),
            "tokens_salida_total": sum(r.output_tokens for r in self.records),
            "costo_estimado_total_usd": sum(r.estimated_cost_usd for r in self.records),
        }


# --------------------------------------------------------------------------- #
# Métricas RAG deterministas
# --------------------------------------------------------------------------- #

@dataclass
class RagEvaluationResult:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
        }


def context_precision(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Proporción de chunks recuperados que efectivamente eran relevantes.

    precision = |recuperados ∩ relevantes| / |recuperados|
    """
    if not retrieved_ids:
        return 0.0
    relevant_retrieved = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return relevant_retrieved / len(retrieved_ids)


def context_recall(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Proporción de chunks relevantes (ground truth) que se lograron recuperar.

    recall = |recuperados ∩ relevantes| / |relevantes|
    """
    if not relevant_ids:
        return 1.0  # No hay nada relevante que recuperar: recall vacío-verdadero.
    relevant_retrieved = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def faithfulness(claims: Sequence[ClaimVerification]) -> float:
    """Proporción de afirmaciones [Soportado] sobre el total evaluado.

    Basado en la salida de `fact_checking.py`: una respuesta 100% fiel es
    aquella donde todas sus afirmaciones verificables están [Soportado] por
    el contexto (ninguna [Contradictorio] ni inventada).
    """
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.veredicto == "Soportado")
    return supported / len(claims)


_TOKEN_RE = re.compile(r"[a-záéíóúñ0-9]+", re.IGNORECASE)


def _tf_vector(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def answer_relevancy(answer: str, question: str) -> float:
    """Proxy léxico determinista de relevancia respuesta-pregunta.

    Nota de diseño: la métrica "oficial" de RAGAS genera preguntas sintéticas
    a partir de la respuesta y mide similitud de EMBEDDINGS contra la
    pregunta original (requiere llamadas a LLM/embeddings, nada determinista).
    Para mantener este proyecto ejecutable sin credenciales y con resultados
    reproducibles en `tests/`, se aproxima con similitud coseno sobre
    vectores TF (bag-of-words) de pregunta y respuesta. La firma
    (`answer: str, question: str -> float en [0, 1]`) es intercambiable por
    una implementación basada en embeddings reales sin tocar el resto del
    pipeline.
    """
    return _cosine_similarity(_tf_vector(answer), _tf_vector(question))


def evaluate_rag_response(
    *,
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    claims: Sequence[ClaimVerification],
    answer: str,
    question: str,
) -> RagEvaluationResult:
    """Calcula las 4 métricas oficiales para una respuesta del pipeline RAG."""
    return RagEvaluationResult(
        context_precision=context_precision(retrieved_ids, relevant_ids),
        context_recall=context_recall(retrieved_ids, relevant_ids),
        faithfulness=faithfulness(claims),
        answer_relevancy=answer_relevancy(answer, question),
    )
