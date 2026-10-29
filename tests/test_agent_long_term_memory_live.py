"""Test LIVE del escenario de memoria de largo plazo (EP2, IE4/IE7):
llama a Groq real. 3 corridas, cada una con sesión A (hallazgo de riesgo
alto + aprobación) + sesión B relacionada (nueva, tras reconstruir
FindingsIndex/CaseLog desde disco) + control negativo en otra sesión.

Qué prueba (distinto del test determinista): que el MODELO REAL decide
por sí mismo llamar a buscar_hallazgos_previos ante una pregunta
relacionada con un hallazgo de una sesión anterior -y que se abstiene
ante una pregunta sin relación-, no solo que la mecánica de persistencia
funciona (eso ya lo prueba test_agent_long_term_memory.py).
"""

from __future__ import annotations

import sys
import time

import pytest
from groq import RateLimitError

# La consola de Windows (cp1252) no puede imprimir todo lo que un modelo
# real puede generar (p.ej. guiones Unicode "‑" U+2011); sin esto, un
# print() de la respuesta del modelo puede hacer fallar el test por un
# problema de encoding, no de lógica del agente.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent.approval import aprobar_hallazgo
from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.graph import build_agent_graph, extract_tool_call_sequence, run_agent_turn
from src.agent.informe_store import InformeStore
from src.agent.token_tracker import TokenTracker
from src.agent.tools import AgentDependencies
from src.config import get_chat_model, get_embeddings, settings
from src.external_data import IndicadoresExternos
from tests.agent_test_helpers import make_chunk

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not settings.groq_api_key, reason="Requiere GROQ_API_KEY real (prueba live, no determinista)."
    ),
]

N_CORRIDAS = 1  # listo para la siguiente tanda (ajustar junto con CORRIDA_OFFSET)
# Ver test_agent_graph_live.py: permite correr por tandas sin confundir
# la numeración de corridas en los prints (el aislamiento de datos entre
# corridas ya lo da tmp_path, que es nuevo en cada invocación de pytest).
CORRIDA_OFFSET = 1  # corrida 0 ya hecha el 2026-10-29, ver docs/evidencia_pruebas.md


class _FakeRetrieverForLive:
    """Ver test_agent_graph_live.py: registra las consultas recibidas
    (`.calls`) para poder inspeccionar, tras una corrida, con qué
    `pregunta` exacta el agente llamó recuperar_documentos cada vez."""

    def __init__(self, docs):
        self._docs = docs
        self.calls: list[str] = []

    def invoke(self, query, config=None):
        self.calls.append(query)
        return self._docs


def _run_with_rate_limit_backoff(graph, deps, pregunta, thread_id, max_attempts=4, backoff_seconds=65.0):
    """Ver tests/test_agent_graph_live.py: la cuenta de Groq de este
    proyecto tiene un límite de 8000 tokens/minuto."""
    last_exc: RateLimitError | None = None
    for attempt in range(max_attempts):
        try:
            return run_agent_turn(graph, deps, pregunta, thread_id=thread_id)
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                print(
                    f"  [rate limit] intento {attempt + 1}/{max_attempts} para thread_id={thread_id}; "
                    f"esperando {backoff_seconds:.0f}s..."
                )
                time.sleep(backoff_seconds)
    raise last_exc


def _deps_live(docs, case_log_path, findings_dir):
    return AgentDependencies(
        retriever=_FakeRetrieverForLive(docs),
        llm=get_chat_model(),
        chunk_store=ChunkStore(),
        informe_store=InformeStore(),
        case_log=CaseLog(case_log_path),
        findings_index=FindingsIndex(get_embeddings(), persist_dir=findings_dir),
        external_fetch=lambda: IndicadoresExternos(
            fecha_consulta="2026-10-28", disponible=False, error="no usado en este test"
        ),
        token_tracker=TokenTracker(model_name=settings.groq_chat_model),
    )


def _print_token_summary(label: str, deps: AgentDependencies) -> None:
    resumen = deps.token_tracker.summary()
    print(
        f"  [{label}] tokens: {resumen['tokens_entrada_total']} in / "
        f"{resumen['tokens_salida_total']} out | costo ~${resumen['costo_estimado_total_usd']:.6f} USD | "
        f"latencia total {resumen['latencia_total_s']:.1f}s | {resumen['num_llamadas']} llamadas"
    )
    for tipo, r in deps.token_tracker.summary_by_label().items():
        print(
            f"    - {tipo}: {r['num_llamadas']} llamadas | {r['tokens_entrada_total']} in / "
            f"{r['tokens_salida_total']} out | ~${r['costo_estimado_total_usd']:.6f} USD"
        )
    if getattr(deps.retriever, "calls", None):
        for idx, consulta in enumerate(deps.retriever.calls, start=1):
            print(f"    - recuperar_documentos llamada {idx}: pregunta={consulta!r}")


def test_live_memoria_de_largo_plazo_entre_sesiones(tmp_path):
    for i in range(CORRIDA_OFFSET, CORRIDA_OFFSET + N_CORRIDAS):
        case_log_path = tmp_path / f"corrida-{i}" / "hallazgos.jsonl"
        findings_dir = tmp_path / f"corrida-{i}" / "vectorstore_hallazgos"

        # --- Sesión A: hallazgo de riesgo alto -----------------------
        docs_a = [
            make_chunk(
                0,
                "nota_credito_NC045.txt",
                "Nota de crédito NC-045 emitida 3 veces por el mismo monto ($5.000.000) "
                "en 24 horas, mismo RUT receptor 11.222.333-4.",
            )
        ]
        deps_a = _deps_live(docs_a, case_log_path, findings_dir)
        graph_a = build_agent_graph(deps_a)
        resultado_a = _run_with_rate_limit_backoff(
            graph_a,
            deps_a,
            "¿Existen señales de riesgo de fraude en la nota de crédito NC-045?",
            thread_id=f"live-A-{i}",
        )
        secuencia_a = extract_tool_call_sequence(resultado_a["messages"])
        print(f"\n[memoria largo plazo | corrida {i + 1} | sesión A] secuencia = {secuencia_a}")
        _print_token_summary("sesión A", deps_a)

        pendientes = [e for e in deps_a.case_log._read_all() if e["estado"] == "pendiente_de_aprobación"]
        assert pendientes, f"Corrida {i + 1}: el modelo no registró ningún hallazgo pendiente en la sesión A."
        hallazgo_id = pendientes[0]["id"]
        aprobar_hallazgo(hallazgo_id, deps_a.case_log, deps_a.findings_index)

        # --- "Reinicio": instancias nuevas sobre el mismo disco --------
        # --- Sesión B relacionada ---------------------------------------
        deps_b = _deps_live([], case_log_path, findings_dir)
        graph_b = build_agent_graph(deps_b)
        resultado_b = _run_with_rate_limit_backoff(
            graph_b,
            deps_b,
            "¿Qué se sabe de la nota de crédito NC-045? ¿Hay algún hallazgo registrado?",
            thread_id=f"live-B-relacionada-{i}",
        )
        secuencia_b = extract_tool_call_sequence(resultado_b["messages"])
        respuesta_b = resultado_b["messages"][-1].content
        print(f"[memoria largo plazo | corrida {i + 1} | sesión B relacionada] secuencia = {secuencia_b}")
        print(f"  respuesta final: {respuesta_b!r}")
        _print_token_summary("sesión B relacionada", deps_b)

        assert "buscar_hallazgos_previos" in secuencia_b, (i, secuencia_b)

        # El hallazgo_id REAL debe aparecer en la OBSERVACIÓN de la
        # herramienta (prueba que recuperó el mismo registrado en la
        # sesión A, no uno inventado). No exigimos el UUID crudo en la
        # respuesta final en prosa al auditor -eso sería mala UX-, pero sí
        # que el contenido del hallazgo (no inventado) se refleje en ella.
        tool_messages_b = [m for m in resultado_b["messages"] if type(m).__name__ == "ToolMessage"]
        contenidos_tool_b = " ".join(m.content for m in tool_messages_b)
        assert hallazgo_id in contenidos_tool_b, (
            f"Corrida {i + 1}: la observación de buscar_hallazgos_previos no incluyó el hallazgo_id real "
            f"({hallazgo_id}); pudo haber recuperado otra cosa o nada."
        )
        assert "NC-045" in respuesta_b or "NC‑045" in respuesta_b, (
            f"Corrida {i + 1}: la respuesta final no menciona NC-045; no se ve que haya usado el hallazgo recuperado."
        )
        assert "no se" not in respuesta_b.lower()[:80] and "no hay" not in respuesta_b.lower()[:80], (
            f"Corrida {i + 1}: la respuesta parece negar tener información, pero sí recuperó el hallazgo: {respuesta_b!r}"
        )

        # --- Control negativo: sesión B, pregunta NO relacionada -------
        docs_control = [
            make_chunk(1, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP | RUT: 76.123.456-7")
        ]
        deps_b2 = _deps_live(docs_control, case_log_path, findings_dir)
        graph_b2 = build_agent_graph(deps_b2)
        resultado_b2 = _run_with_rate_limit_backoff(
            graph_b2,
            deps_b2,
            "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?",
            thread_id=f"live-B-control-{i}",
        )
        secuencia_b2 = extract_tool_call_sequence(resultado_b2["messages"])
        print(f"[memoria largo plazo | corrida {i + 1} | control negativo] secuencia = {secuencia_b2}")
        _print_token_summary("control negativo", deps_b2)

        assert "buscar_hallazgos_previos" not in secuencia_b2, (i, secuencia_b2)
