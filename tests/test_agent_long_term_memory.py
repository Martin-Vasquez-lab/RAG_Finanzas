"""Test DETERMINISTA del escenario de memoria de largo plazo (EP2,
IE3/IE4). Sin API: LLM con guion fijo, embeddings fake.

Qué prueba (y qué NO): que la mecánica -persistencia en disco,
aislamiento entre sesiones, recuperación por ID- funciona correctamente.
NO prueba que un modelo real decida por sí mismo llamar a
buscar_hallazgos_previos en una pregunta relacionada y se abstenga en una
no relacionada -eso solo lo puede demostrar una corrida contra Groq real,
ver tests/test_agent_long_term_memory_live.py-.

Por qué el escenario NO es en la misma sesión (acordado antes de
implementar): si la pregunta de la sesión B usara el mismo thread_id que
la sesión A, el historial de CORTO plazo (los mensajes del propio turno
anterior, vía el checkpointer) ya contendría el hallazgo, y el agente
podría responder desde ahí sin tocar buscar_hallazgos_previos -el
escenario no probaría nada de IE4 (memoria de LARGO plazo), solo de IE3
(memoria de sesión, ya cubierta en otros tests-. Por eso la sesión B usa
un thread_id nuevo, y además reconstruye FindingsIndex/CaseLog con
instancias Python nuevas sobre el mismo disco -simulando un reinicio del
proceso-, no reutiliza los objetos de la sesión A.
"""

from __future__ import annotations

from src.agent.approval import aprobar_hallazgo
from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.graph import build_agent_graph, extract_tool_call_sequence, run_agent_turn
from src.agent.informe_store import InformeStore
from src.agent.tools import AgentDependencies
from src.external_data import IndicadoresExternos
from tests.agent_test_helpers import (
    PLAN_JSON_OK,
    FakeChatModel,
    FakeEmbeddings,
    FakeRetriever,
    ScriptedToolCallingLLM,
    ai_final,
    ai_tool_call,
    make_chunk,
)

_SIN_FUENTE_EXTERNA = lambda: IndicadoresExternos(  # noqa: E731
    fecha_consulta="2026-10-28", disponible=False, error="no usado en este test"
)


def _build_deps(llm, case_log_path, findings_dir, retriever_docs=None):
    return AgentDependencies(
        retriever=FakeRetriever(retriever_docs or []),
        llm=llm,
        planner_llm=FakeChatModel([PLAN_JSON_OK] * 5),
        chunk_store=ChunkStore(),
        informe_store=InformeStore(),
        case_log=CaseLog(case_log_path),
        findings_index=FindingsIndex(FakeEmbeddings(), persist_dir=findings_dir),
        external_fetch=_SIN_FUENTE_EXTERNA,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )


def test_hallazgo_aprobado_en_sesion_a_se_recupera_en_sesion_b_tras_reinicio(tmp_path):
    case_log_path = tmp_path / "hallazgos.jsonl"
    findings_dir = tmp_path / "vectorstore_hallazgos"

    # --- Sesión A: se detecta y registra un hallazgo de riesgo alto ----
    script_a = [
        ai_tool_call("recuperar_documentos", {"pregunta": "riesgo de NC-045"}, "call-1"),
        ai_tool_call(
            "registrar_hallazgo",
            {
                "descripcion": "Emisión duplicada del mismo monto en 24 horas en NC-045",
                "nivel_riesgo": "alto",
                "chunk_ids_relacionados": [0],
            },
            "call-2",
        ),
        ai_final("Hallazgo registrado, pendiente de aprobación."),
    ]
    deps_a = _build_deps(
        ScriptedToolCallingLLM(script_a),
        case_log_path,
        findings_dir,
        retriever_docs=[
            make_chunk(0, "nota_credito_NC045.txt", "NC-045 emitida 3 veces por el mismo monto en 24 horas")
        ],
    )
    run_agent_turn(build_agent_graph(deps_a), deps_a, "¿hay riesgo de fraude en NC-045?", thread_id="sesion-A")

    pendientes = [e for e in deps_a.case_log._read_all() if e["estado"] == "pendiente_de_aprobación"]
    assert len(pendientes) == 1, "La sesión A debió registrar exactamente un hallazgo pendiente."
    hallazgo_id = pendientes[0]["id"]

    # Aprobación (humana o vía settings.auto_approve_hallazgos): NUNCA una
    # herramienta del LLM, ver src/agent/approval.py.
    aprobar_hallazgo(hallazgo_id, deps_a.case_log, deps_a.findings_index)

    # --- "Reinicio del proceso": instancias NUEVAS sobre el mismo disco,
    # no se reutiliza ningún objeto Python de la sesión A --------------
    script_b_relacionada = [
        ai_tool_call("buscar_hallazgos_previos", {"pregunta": "NC-045 emisión duplicada"}, "call-1"),
        ai_final(f"Según el hallazgo {hallazgo_id}, ya se detectó emisión duplicada en NC-045."),
    ]
    deps_b = _build_deps(ScriptedToolCallingLLM(script_b_relacionada), case_log_path, findings_dir)

    resultado_b = run_agent_turn(
        build_agent_graph(deps_b), deps_b, "¿Qué se sabe de la nota de crédito NC-045?", thread_id="sesion-B"
    )

    secuencia_b = extract_tool_call_sequence(resultado_b["messages"])
    assert secuencia_b == ["buscar_hallazgos_previos"]

    tool_messages_b = [m for m in resultado_b["messages"] if type(m).__name__ == "ToolMessage"]
    contenido_busqueda = next(m.content for m in tool_messages_b)
    assert hallazgo_id in contenido_busqueda, "El hallazgo recuperado debe ser EL MISMO aprobado en la sesión A."

    respuesta_final = resultado_b["messages"][-1].content
    assert hallazgo_id in respuesta_final, "La respuesta final debe citar el ID real del hallazgo, no inventar uno."


def test_pregunta_no_relacionada_en_sesion_b_no_llama_a_buscar_hallazgos_previos(tmp_path):
    """Control negativo: con la MISMA memoria de largo plazo poblada
    (sesión A ya registrada y aprobada), una pregunta sin relación en una
    sesión B distinta no debe llamar a buscar_hallazgos_previos -si el
    agente la llamara siempre "por si acaso", el escenario positivo no
    demostraría una decisión real, solo una herramienta cableada."""
    case_log_path = tmp_path / "hallazgos.jsonl"
    findings_dir = tmp_path / "vectorstore_hallazgos"

    script_a = [
        ai_tool_call("recuperar_documentos", {"pregunta": "riesgo de NC-045"}, "call-1"),
        ai_tool_call(
            "registrar_hallazgo",
            {
                "descripcion": "Emisión duplicada del mismo monto en 24 horas en NC-045",
                "nivel_riesgo": "alto",
                "chunk_ids_relacionados": [0],
            },
            "call-2",
        ),
        ai_final("Hallazgo registrado, pendiente de aprobación."),
    ]
    deps_a = _build_deps(
        ScriptedToolCallingLLM(script_a),
        case_log_path,
        findings_dir,
        retriever_docs=[make_chunk(0, "nota_credito_NC045.txt", "NC-045 emitida 3 veces")],
    )
    run_agent_turn(build_agent_graph(deps_a), deps_a, "¿hay riesgo de fraude en NC-045?", thread_id="sesion-A")
    hallazgo_id = next(e["id"] for e in deps_a.case_log._read_all() if e["estado"] == "pendiente_de_aprobación")
    aprobar_hallazgo(hallazgo_id, deps_a.case_log, deps_a.findings_index)

    script_b_no_relacionada = [
        ai_tool_call("recuperar_documentos", {"pregunta": "monto de F-1023"}, "call-1"),
        ai_final("El monto de F-1023 es $1.245.000 CLP, sin relación con hallazgos previos."),
    ]
    deps_b = _build_deps(
        ScriptedToolCallingLLM(script_b_no_relacionada),
        case_log_path,
        findings_dir,
        retriever_docs=[make_chunk(1, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP")],
    )

    resultado_b = run_agent_turn(
        build_agent_graph(deps_b),
        deps_b,
        "¿Cuál es el monto total de la factura F-1023?",
        thread_id="sesion-B-control",
    )

    secuencia_b = extract_tool_call_sequence(resultado_b["messages"])
    assert "buscar_hallazgos_previos" not in secuencia_b
