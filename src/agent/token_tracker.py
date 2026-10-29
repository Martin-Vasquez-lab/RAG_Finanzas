"""Registro de tokens/costo/latencia por llamada al LLM durante UNA
ejecución del agente (EP2), para que las corridas `live` dejen cifras
MEDIDAS (no estimadas) en `docs/evidencia_pruebas.md`.

No se implementó threadeando el `RunnableConfig`/callbacks de LangChain
(como hace `AuditCallbackHandler` en la EP1) a través de LangGraph +
`StructuredTool` + las cadenas internas de cada herramienta: habría que
reenviar `config` manualmente en cada sitio de todas formas para
garantizar que llegue (el grafo no lo propaga solo), así que en vez de
asumir una propagación automática no verificada, cada sitio que ya tiene
el objeto `response` en la mano (`tools.py`, `graph.py`) llama a
`record()` directamente. Reutiliza la MISMA tabla de precios que
`src.metrics.AuditCallbackHandler` (EP1), para no duplicar criterios de
costeo entre ambos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.metrics import PRICING_USD_PER_1K_TOKENS


@dataclass(frozen=True)
class LlmCallRecord:
    label: str  # p.ej. "llm_node", "planner", "juzgar_relevancia", "generar_informe"
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_seconds: float
    timestamp: float = field(default_factory=time.time)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_USD_PER_1K_TOKENS.get(model, PRICING_USD_PER_1K_TOKENS["_default"])
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]


@dataclass
class TokenTracker:
    """Una instancia por ejecución del agente (mismo criterio que
    `ChunkStore`/`InformeStore`). Opcional en `AgentDependencies`: si es
    `None`, nada registra nada (compatibilidad con los 100 tests
    deterministas existentes, que no necesitan esto)."""

    model_name: str
    records: list[LlmCallRecord] = field(default_factory=list)

    def record(self, label: str, response: Any, latency_seconds: float) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self.records.append(
            LlmCallRecord(
                label=label,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=_estimate_cost(self.model_name, input_tokens, output_tokens),
                latency_seconds=latency_seconds,
            )
        )

    def summary(self) -> dict:
        if not self.records:
            return {
                "num_llamadas": 0,
                "tokens_entrada_total": 0,
                "tokens_salida_total": 0,
                "costo_estimado_total_usd": 0.0,
                "latencia_total_s": 0.0,
            }
        return {
            "num_llamadas": len(self.records),
            "tokens_entrada_total": sum(r.input_tokens for r in self.records),
            "tokens_salida_total": sum(r.output_tokens for r in self.records),
            "costo_estimado_total_usd": sum(r.estimated_cost_usd for r in self.records),
            "latencia_total_s": sum(r.latency_seconds for r in self.records),
        }
