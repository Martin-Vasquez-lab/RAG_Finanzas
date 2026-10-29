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

from langchain_core.runnables import Runnable

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.graph import (
    MAX_ITERATIONS,
    MAX_TOKENS_PER_TURN,
    build_agent_graph,
    extract_tool_call_sequence,
    run_agent_turn,
    run_demo_cli,
)
from src.agent.informe_store import InformeStore
from src.agent.token_tracker import TokenTracker
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


def _deps(tmp_path, llm, retriever_docs=None, planner_llm=None):
    return AgentDependencies(
        retriever=FakeRetriever(
            retriever_docs
            or [make_chunk(0, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP")]
        ),
        llm=llm,
        # El planificador usa un LLM SEPARADO del guion de decisiones de
        # herramientas (ScriptedToolCallingLLM): si compartieran la misma
        # cola, la llamada de planificación consumiría por error el
        # siguiente tool_call que el test esperaba para el bucle principal.
        planner_llm=planner_llm or FakeChatModel([PLAN_JSON_OK] * 10),
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


# --------------------------------------------------------------------------- #
# Plan priorizado y replanificación (IE1/IE5): disparador concreto y prueba
# de que el plan no es decorativo, efectivamente influye en la decisión.
# --------------------------------------------------------------------------- #


def test_hallazgo_de_riesgo_alto_dispara_replanificacion_antes_de_seguir(tmp_path):
    """Disparador CONCRETO: registrar_hallazgo con nivel_riesgo='alto'
    (estado pendiente_de_aprobación) fuerza una replanificación antes de
    la siguiente decisión del LLM. El plan actualizado queda como mensaje
    en la conversación -no solo en un campo interno del estado- y
    aparece ANTES del cierre del turno."""
    script = [
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
        ai_final("Hallazgo registrado y plan actualizado."),
    ]
    deps = _deps(tmp_path, ScriptedToolCallingLLM(script))

    result = run_agent_turn(build_agent_graph(deps), deps, "riesgo de NC-045", thread_id="sesion-replan")

    assert result["plan_version"] == 2  # v1 del planner inicial + v2 de la replanificación
    contenidos = [getattr(m, "content", "") for m in result["messages"]]
    assert any("[PLAN v1]" in c for c in contenidos)
    indice_plan_actualizado = next(i for i, c in enumerate(contenidos) if "[PLAN ACTUALIZADO v2]" in c)
    assert indice_plan_actualizado < len(contenidos) - 1  # llega ANTES del mensaje final


def test_fuente_externa_caida_dispara_replanificacion(tmp_path):
    """Disparador CONCRETO #2: una herramienta que señala una condición
    externa caída (consultar_tipo_cambio con disponible=False, el default
    de `_deps` en este archivo) también fuerza una replanificación, no
    solo el riesgo alto de un hallazgo."""
    script = [
        ai_tool_call("consultar_tipo_cambio", {"moneda": "UF"}, "call-1"),
        ai_final("Sigo sin el dato externo, pero completé el informe igual."),
    ]
    deps = _deps(tmp_path, ScriptedToolCallingLLM(script))

    result = run_agent_turn(build_agent_graph(deps), deps, "contrato en UF", thread_id="sesion-api-caida")

    assert result["plan_version"] == 2
    contenidos = [getattr(m, "content", "") for m in result["messages"]]
    assert any("[PLAN ACTUALIZADO v2]" in c for c in contenidos)


def test_mensaje_de_seguimiento_en_sesion_continua_dispara_replanificacion(tmp_path):
    """Disparador CONCRETO #3: una segunda pregunta en la MISMA sesión
    (mismo thread_id) -no la primera- entra directo a `replanner`, no a
    `planner`: route_start detecta que la sesión ya tenía un plan."""
    script = [
        ai_final("Respuesta a la primera pregunta."),
        ai_final("Respuesta a la pregunta de seguimiento."),
    ]
    deps = _deps(tmp_path, ScriptedToolCallingLLM(script))
    graph = build_agent_graph(deps)

    resultado_1 = run_agent_turn(graph, deps, "primera pregunta", thread_id="sesion-seguimiento")
    assert resultado_1["plan_version"] == 1

    resultado_2 = run_agent_turn(
        graph, deps, "ahora lo urgente es la factura F-2000", thread_id="sesion-seguimiento"
    )

    assert resultado_2["plan_version"] == 2
    contenidos = [getattr(m, "content", "") for m in resultado_2["messages"]]
    assert any("[PLAN ACTUALIZADO v2]" in c for c in contenidos)
    # El System Message del agente no debe repetirse: solo aparece una vez por sesión.
    system_messages = [m for m in resultado_2["messages"] if type(m).__name__ == "SystemMessage"]
    assert len(system_messages) == 1


class _PlanAwareFakeLLM(Runnable):
    """Fake LLM local a este test: decide su TERCER tool_call mirando si
    ya ve un mensaje "[PLAN ACTUALIZADO" en el historial que recibe como
    input. No es razonamiento real -es un guion condicional-, pero prueba
    el WIRING: que el texto del plan replanificado llega al input del LLM
    a tiempo para pesar en la siguiente decisión. Si el mensaje llegara
    tarde (p.ej. después de la llamada al LLM en vez de antes), este fake
    devolvería la rama de "FALLO" y el test lo detectaría."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.calls += 1
        texto = " ".join(getattr(m, "content", "") or "" for m in messages)
        if self.calls == 1:
            return ai_tool_call("recuperar_documentos", {"pregunta": "riesgo de NC-045"}, "call-1")
        if self.calls == 2:
            return ai_tool_call(
                "registrar_hallazgo",
                {
                    "descripcion": "Emisión duplicada del mismo monto en 24 horas",
                    "nivel_riesgo": "alto",
                    "chunk_ids_relacionados": [0],
                },
                "call-2",
            )
        if self.calls == 3:
            if "[PLAN ACTUALIZADO" in texto:
                return ai_tool_call(
                    "buscar_hallazgos_previos", {"pregunta": "emisión duplicada del mismo monto"}, "call-3"
                )
            return ai_final("FALLO: el plan actualizado no llegó a tiempo al input del LLM.")
        return ai_final("Listo.")


def test_el_plan_actualizado_influye_causalmente_en_la_siguiente_herramienta_elegida(tmp_path):
    """Más exigente que el test anterior: no basta con que el mensaje de
    replanificación exista en algún lado, tiene que llegar a tiempo para
    que la SIGUIENTE decisión pueda basarse en él. Si el wiring estuviera
    roto, la tercera llamada del fake caería en la rama de "FALLO" y la
    secuencia no incluiría buscar_hallazgos_previos."""
    llm = _PlanAwareFakeLLM()
    deps = _deps(tmp_path, llm)

    result = run_agent_turn(build_agent_graph(deps), deps, "riesgo de NC-045", thread_id="sesion-causal")

    secuencia = extract_tool_call_sequence(result["messages"])
    assert secuencia == ["recuperar_documentos", "registrar_hallazgo", "buscar_hallazgos_previos"]


# --------------------------------------------------------------------------- #
# Punto de entrada de la demo (`python -m src.agent.graph`): arranca con un
# LLM simulado, sin gastar API. No prueba la construcción de dependencias
# REALES (build_demo_dependencies, que sí requiere GROQ_API_KEY y el índice
# FAISS) -eso se verifica manualmente al correr la demo de verdad-, solo el
# bucle interactivo (run_demo_cli) en sí.
# --------------------------------------------------------------------------- #


def test_run_demo_cli_arranca_y_corre_un_turno_sin_api(tmp_path):
    deps = _deps(tmp_path, ScriptedToolCallingLLM([ai_final("Respuesta de prueba.")]))
    deps.token_tracker = TokenTracker(model_name="modelo-de-prueba")

    entradas = iter(["sesion-prueba", "1", "salir"])
    salidas: list[str] = []

    run_demo_cli(deps, input_fn=lambda _mensaje: next(entradas), print_fn=salidas.append)

    texto = "\n".join(salidas)
    assert "Sesión: sesion-prueba" in texto
    assert "Escenarios documentados" in texto
    assert "[Pensamiento/Respuesta] Respuesta de prueba." in texto
    assert "Plan vigente v1" in texto
    assert "Resumen de la sesión" in texto


def test_run_demo_cli_entrada_vacia_termina_sin_correr_ningun_turno(tmp_path):
    """Si no se elige ningún escenario ni se escribe una pregunta, el
    bucle debe terminar limpio sin invocar al LLM en absoluto (ID de
    sesión y luego Enter directo)."""
    llm = ScriptedToolCallingLLM([])  # cualquier .invoke() aquí haría fallar el test
    deps = _deps(tmp_path, llm)

    entradas = iter(["sesion-vacia", ""])
    salidas: list[str] = []

    run_demo_cli(deps, input_fn=lambda _mensaje: next(entradas), print_fn=salidas.append)

    assert llm.calls == 0
