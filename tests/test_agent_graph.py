"""Tests DETERMINISTAS del grafo del agente (EP2, IE1/IE2/IE5). Sin API:
el LLM es un guion fijo (`ScriptedToolCallingLLM`).

Qué prueba este archivo (distinto de `test_agent_graph_live.py`): la
MECÁNICA del grafo -obedece lo que el "modelo" decide en cada paso, sin
ningún orden cableado, respeta el tope de iteraciones, el tope de tokens
y la guarda contra repetir una llamada idéntica- independientemente de
si un modelo real razona bien. Lo que NO prueba: si un LLM real elige
bien las herramientas según el contenido de la pregunta -eso solo lo
puede demostrar una corrida contra Groq real, ver el archivo `_live`.
"""

from __future__ import annotations

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.graph import (
    MAX_ITERATIONS,
    MAX_TOKENS_PER_TURN,
    build_agent_graph,
    extract_tool_call_sequence,
    run_agent_turn,
)
from src.agent.informe_store import InformeStore
from src.agent.tools import AgentDependencies
from src.external_data import IndicadoresExternos
from tests.agent_test_helpers import (
    FakeEmbeddings,
    FakeRetriever,
    ScriptedToolCallingLLM,
    ai_final,
    ai_tool_call,
    make_chunk,
)


def _deps(tmp_path, llm, retriever_docs=None):
    return AgentDependencies(
        retriever=FakeRetriever(
            retriever_docs
            or [make_chunk(0, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP")]
        ),
        llm=llm,
        chunk_store=ChunkStore(),
        informe_store=InformeStore(),
        case_log=CaseLog(tmp_path / "hallazgos.jsonl"),
        findings_index=FindingsIndex(FakeEmbeddings()),
        external_fetch=lambda: IndicadoresExternos(
            fecha_consulta="2026-10-28", disponible=False, error="no usado en este test"
        ),
        max_retries=1,
        retry_backoff_seconds=0.0,
    )


def test_grafo_ejecuta_la_secuencia_que_decide_el_llm_no_un_orden_cableado(tmp_path):
    """Dos guiones distintos -> dos secuencias de herramientas EJECUTADAS
    distintas. Si el grafo tuviera un orden cableado, ambas corridas
    terminarían ejecutando lo mismo sin importar qué pida el guion."""
    script_a = [
        ai_tool_call("recuperar_documentos", {"pregunta": "monto de F-1023"}, "call-1"),
        ai_tool_call("consultar_tipo_cambio", {"moneda": "UF"}, "call-2"),
        ai_final("Listo, sin hallazgos que registrar."),
    ]
    deps_a = _deps(tmp_path / "a", ScriptedToolCallingLLM(script_a))
    result_a = run_agent_turn(build_agent_graph(deps_a), deps_a, "monto de F-1023", thread_id="sesion-a")

    script_b = [
        ai_tool_call("recuperar_documentos", {"pregunta": "riesgo de NC-045"}, "call-1"),
        ai_tool_call(
            "registrar_hallazgo",
            {
                "descripcion": "Emisión duplicada del mismo monto en 24 horas",
                "nivel_riesgo": "alto",
                "chunk_ids_relacionados": [0],
            },
            "call-2",
        ),
        ai_tool_call("buscar_hallazgos_previos", {"pregunta": "emisión duplicada"}, "call-3"),
        ai_final("Listo, hallazgo registrado y pendiente de aprobación."),
    ]
    deps_b = _deps(tmp_path / "b", ScriptedToolCallingLLM(script_b))
    result_b = run_agent_turn(build_agent_graph(deps_b), deps_b, "riesgo de NC-045", thread_id="sesion-b")

    secuencia_a = extract_tool_call_sequence(result_a["messages"])
    secuencia_b = extract_tool_call_sequence(result_b["messages"])

    assert secuencia_a == ["recuperar_documentos", "consultar_tipo_cambio"]
    assert secuencia_b == ["recuperar_documentos", "registrar_hallazgo", "buscar_hallazgos_previos"]
    assert secuencia_a != secuencia_b


def test_tope_de_iteraciones_detiene_el_bucle_sin_colgarse(tmp_path):
    # El guion SIEMPRE pide una herramienta con argumentos distintos (para
    # no toparse con la guarda de repetición antes que con el tope de
    # iteraciones), más veces de las que el tope debería permitir.
    script = [
        ai_tool_call("buscar_hallazgos_previos", {"pregunta": f"tema {i}"}, f"call-{i}")
        for i in range(MAX_ITERATIONS + 3)
    ]
    llm = ScriptedToolCallingLLM(script)
    deps = _deps(tmp_path, llm)

    result = run_agent_turn(build_agent_graph(deps), deps, "pregunta que nunca concluye", thread_id="sesion-loop")

    assert result["iteration"] == MAX_ITERATIONS
    assert llm.calls == MAX_ITERATIONS  # el grafo deja de llamar al LLM al llegar al tope, no sigue de largo


def test_tope_de_tokens_detiene_el_bucle_aunque_la_iteracion_sea_baja(tmp_path):
    script = [
        ai_tool_call(
            "buscar_hallazgos_previos",
            {"pregunta": "algo"},
            "call-1",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": MAX_TOKENS_PER_TURN + 1},
        ),
    ]
    deps = _deps(tmp_path, ScriptedToolCallingLLM(script))

    result = run_agent_turn(build_agent_graph(deps), deps, "pregunta cara", thread_id="sesion-tokens")

    # Se detiene tras la primera llamada por TOKENS, no por iteración.
    assert result["iteration"] == 1
    assert result["tokens_used"] >= MAX_TOKENS_PER_TURN


def test_guarda_contra_llamada_identica_consecutiva_no_la_ejecuta_de_nuevo(tmp_path):
    script = [
        ai_tool_call("recuperar_documentos", {"pregunta": "monto de F-1023"}, "call-1"),
        ai_tool_call("recuperar_documentos", {"pregunta": "monto de F-1023"}, "call-2"),  # idéntica
        ai_final("Ya lo tengo."),
    ]
    deps = _deps(tmp_path, ScriptedToolCallingLLM(script))

    result = run_agent_turn(build_agent_graph(deps), deps, "monto de F-1023", thread_id="sesion-repite")

    # El retriever real solo debió invocarse UNA vez: la segunda llamada
    # idéntica se bloquea antes de ejecutar la herramienta.
    assert deps.retriever.calls == ["monto de F-1023"]
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("idéntica" in m.content for m in tool_messages)
