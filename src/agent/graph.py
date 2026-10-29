"""Grafo del agente (EP2, IE1/IE2/IE5): el LLM decide qué herramienta usar
y en qué orden; el grafo solo ejecuta esa decisión, no fija el orden.

Bucle (LangGraph):

    START -> planner -> llm -> (¿tool_calls?) -> tools -> llm -> ... -> END
                                                     |
                                                     v (hallazgo de riesgo alto)
                                                 replanner -> llm

`llm` invoca el modelo con las 7 herramientas de `tools.py` vía
`bind_tools`; mientras la última respuesta traiga `tool_calls`, el grafo
pasa a `tools`, las ejecuta, y vuelve a `llm` con las observaciones. Si la
respuesta no trae `tool_calls`, o se alcanza el tope de iteraciones/tokens
del turno, el grafo termina.

`planner` genera, ANTES de la primera decisión del LLM, un plan priorizado
(`src/agent/plan.py`) que se inyecta como mensaje en la conversación -no
solo en un campo interno del estado- para que realmente pese en la
elección de herramientas, no sea decorativo. `replanner` se dispara con
TRES disparadores CONCRETOS, todos observables en código (ninguno
depende de que el LLM "decida" replanificar por su cuenta):
  1. Hallazgo de riesgo alto: `registrar_hallazgo` devuelve
     `estado="pendiente_de_aprobación"` (ver `route_after_tools`).
  2. Condición externa cambiante: una herramienta devuelve
     `disponible=False` (p.ej. `consultar_tipo_cambio` cuando
     mindicador.cl está caído) (también en `route_after_tools`).
  3. Mensaje de seguimiento a mitad de tarea: una pregunta NUEVA en una
     sesión (`thread_id`) que YA tiene un plan -no la primera pregunta de
     esa sesión- entra directo a `replanner` en vez de `planner`
     (ver `route_start`/`run_agent_turn`).

Fuente ÚNICA de historial conversacional: el canal `messages` del estado
del grafo (acumulado por el checkpointer entre turnos de la misma
`thread_id`). Deliberadamente NO se usa `src.memory.BufferMemory`/
`SummaryMemory` aquí -esas siguen siendo exclusivas de
`AuditRagPipeline` (EP1)-, para no duplicar el historial en el prompt ni
pagar el doble de tokens.

Controles anti-descontrol (se afinan en un paso posterior, pero deben
existir desde ya porque las corridas `live` gastan dinero real):
  - Tope de iteraciones (`MAX_ITERATIONS`): cada vuelta por el nodo `llm`
    cuenta una iteración.
  - Tope de tokens aproximado (`MAX_TOKENS_PER_TURN`): acumulado desde
    `usage_metadata` de cada respuesta del LLM en el turno.
  - Guarda contra repetir la MISMA llamada (mismo nombre + mismos
    argumentos) dos veces seguidas: la segunda no se ejecuta, se devuelve
    una observación de error para que el modelo cambie de curso.

`chunk_store`/`informe_store` (de `AgentDependencies`) se reflejan también
en el estado del grafo, en los campos reservados para pasos futuros
(ver `AgentState`), aunque la ejecución real de las herramientas sigue
resolviéndose por clausura sobre la misma instancia de `AgentDependencies`
-no leyendo el store desde el estado en cada nodo-, porque el checkpointer
usado aquí (`MemorySaver`) mantiene los objetos vivos en memoria del
proceso sin necesidad de serializarlos. Si más adelante se migra a un
checkpointer persistente (SQLite), estos campos tendrán que
serializar/reconstruirse en vez de guardar la referencia directa.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from groq import RateLimitError  # excluido del manejo de errores de generación de llm_node, ver docstring
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.informe_store import InformeStore
from src.agent.llm_utils import invoke_with_retry
from src.agent.plan import PlanAuditoria, PlanPasos
from src.agent.token_tracker import TokenTracker
from src.agent.tools import AgentDependencies, build_agent_tools
from src.config import CASE_LOG_PATH, FINDINGS_INDEX_DIR, FINDINGS_INDEX_NAME, get_chat_model, get_embeddings, settings
from src.external_data import IndicadoresExternos, fetch_indicadores_economicos
from src.loaders import load_faiss_index
from src.prompts import NO_DATA_RULE

MAX_ITERATIONS = 8
MAX_TOKENS_PER_TURN = 20_000
# Reintentos ante un error de GENERACIÓN del LLM (p.ej. tool_call a una
# herramienta inexistente/mal formada), no de ejecución de herramienta.
MAX_LLM_GENERATION_RETRIES = 2

AGENT_SYSTEM_PROMPT = f"""
Eres el agente de auditoría del Departamento de Auditoría Interna y
Cumplimiento Tributario de Banco Andino S.A. Resuelves la pregunta del
área de auditoría DECIDIENDO TÚ, en cada paso, qué herramienta usar y en
qué orden -no sigas siempre la misma secuencia; úsalas solo cuando la
pregunta concreta lo justifique-.

Reglas:
- {NO_DATA_RULE}
- No repitas una llamada idéntica (misma herramienta, mismos argumentos)
  dos veces seguidas: si una herramienta te devuelve un error, cambia de
  argumentos, de herramienta, o concluye con lo que ya tengas.
- generar_informe produce la respuesta final para el usuario: solo una
  vez por pregunta. Si necesitas revisarla, usa verificar_afirmaciones
  sobre el informe_id que te devolvió, no generes un informe nuevo.
- Cuando ya tengas una respuesta completa y verificada, responde en
  texto plano sin llamar a ninguna herramienta más: así termina el turno.
- Verás bloques [PLAN]/[PLAN ACTUALIZADO] en la conversación: son una
  guía priorizada, no una orden ciega -síguelos salvo que la pregunta
  concreta amerite desviarte, y dilo si te desvías-.
- Al usar registrar_hallazgo, califica nivel_riesgo="alto" ante señales
  CLARAS de fraude: duplicidad de montos/documentos en una ventana de
  24 horas, RUT no coincidentes, montos atípicos sin respaldo, o
  inconsistencias graves entre documentos. Usa "medio" solo ante
  inconsistencias menores, y "bajo" cuando el hallazgo es solo
  informativo. Si ya generaste un informe con generar_informe, el
  nivel_riesgo de registrar_hallazgo debe ser COHERENTE con el
  nivel_riesgo_fraude de ese informe -no lo contradigas sin una razón
  explícita que puedas justificar.
""".strip()

NOMBRES_HERRAMIENTAS = (
    "recuperar_documentos, juzgar_relevancia, consultar_tipo_cambio, generar_informe, "
    "verificar_afirmaciones, buscar_hallazgos_previos, registrar_hallazgo"
)

PLAN_SYSTEM_PROMPT = f"""
Eres el módulo de planificación del agente de auditoría de Banco Andino
S.A. NO ejecutas nada tú mismo: dado el contexto, propones una lista de
pasos PRIORIZADOS (prioridad alta/media/baja) que el agente debería
seguir para resolver la tarea. Cada paso debe usar, como "accion", el
nombre EXACTO de una de estas herramientas reales: {NOMBRES_HERRAMIENTAS}.
""".strip()

PLAN_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", PLAN_SYSTEM_PROMPT),
        ("human", "{contexto}\n\n{format_instructions}"),
    ]
)

_plan_parser = PydanticOutputParser(pydantic_object=PlanPasos)


# --------------------------------------------------------------------------- #
# Estado del grafo
# --------------------------------------------------------------------------- #


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    chunk_store: ChunkStore
    informe_store: InformeStore
    plan: dict | None  # dict de PlanAuditoria; None hasta que corre el nodo "planner"
    plan_version: int
    iteration: int
    last_tool_call: dict | None
    tokens_used: int




def _extract_tokens(response: AIMessage) -> int:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return int(usage.get("total_tokens", 0))
    return 0


def _tool_call_signature(tool_call: dict) -> dict:
    return {"name": tool_call["name"], "args": tool_call["args"]}


def _first_human_message_content(messages: list[AnyMessage]) -> str:
    """La pregunta ORIGINAL del turno (primer HumanMessage): tanto
    `planner` como `replanner` la necesitan como contexto, y los mensajes
    de plan/replan inyectados también son HumanMessage (ver `plan.py`),
    así que no basta con tomar "el primero" sin filtrar por tipo."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _is_empty_text_response(response: Any) -> bool:
    return not getattr(response, "content", "").strip()


# --------------------------------------------------------------------------- #
# Nodos
# --------------------------------------------------------------------------- #


@dataclass
class AgentGraphBuilder:
    """Junta las herramientas (ligadas a `deps`) con el LLM para construir
    los dos nodos del grafo. Separado en una clase pequeña, en vez de
    funciones sueltas con `deps`/`tools` en variables de módulo, para que
    `build_agent_graph()` pueda construir grafos independientes con
    distintas dependencias (p.ej. en tests) sin pisarse entre sí."""

    deps: AgentDependencies
    tools: list[StructuredTool]

    def __post_init__(self) -> None:
        self._tools_by_name = {t.name: t for t in self.tools}
        self._llm_with_tools = self.deps.llm.bind_tools(self.tools)
        self._plan_chain = (
            PLAN_PROMPT_TEMPLATE.partial(format_instructions=_plan_parser.get_format_instructions())
            | self.deps.planner_llm
        )

    def _generar_pasos_plan(self, contexto: str, label: str) -> list:
        start = time.perf_counter()
        response = invoke_with_retry(
            lambda: self._plan_chain.invoke({"contexto": contexto}),
            is_empty=_is_empty_text_response,
            max_retries=self.deps.max_retries,
            backoff_seconds=self.deps.retry_backoff_seconds,
        )
        if self.deps.token_tracker is not None:
            self.deps.token_tracker.record(label, response, time.perf_counter() - start)
        try:
            return _plan_parser.parse(response.content).pasos
        except Exception:  # noqa: BLE001 - seguir sin plan es preferible a tumbar el turno
            return []

    def planner_node(self, state: AgentState) -> dict:
        pregunta = _first_human_message_content(state["messages"])
        pasos = self._generar_pasos_plan(f"Pregunta del área de auditoría:\n{pregunta}", label="planner")
        plan = PlanAuditoria(pasos=pasos, version=1)
        return {
            "plan": plan.model_dump(),
            "plan_version": plan.version,
            "messages": [HumanMessage(content=plan.render())],
        }

    def _motivo_replanificacion(self, state: AgentState) -> str:
        """Determina CUÁL de los 3 disparadores causó esta replanificación,
        inspeccionando el último mensaje (no le pregunta al LLM por qué):
        un ToolMessage con `estado="pendiente_de_aprobación"` (hallazgo de
        riesgo alto), uno con `disponible=False` (fuente externa caída), o
        -si el último mensaje es un HumanMessage nuevo, no un ToolMessage-
        un mensaje de seguimiento a mitad de tarea."""
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            try:
                payload = json.loads(last.content)
            except (TypeError, ValueError):
                payload = {}
            if payload.get("estado") == "pendiente_de_aprobación":
                return (
                    "Se acaba de registrar un hallazgo de riesgo ALTO (pendiente de aprobación), "
                    f"resultado de la herramienta: {last.content}\n"
                    "Sube la prioridad de los pasos relacionados con ese hallazgo (p.ej. "
                    "buscar_hallazgos_previos, o verificar_afirmaciones si ya hay un informe)."
                )
            if payload.get("disponible") is False:
                return (
                    "Una fuente externa dejó de estar disponible a mitad de tarea "
                    f"(resultado de la herramienta: {last.content}).\n"
                    "Ajusta el plan para continuar sin ese dato -indícalo explícitamente en el "
                    "informe en vez de bloquear la tarea esperando que vuelva a estar disponible."
                )
        return (
            "Llegó un mensaje de seguimiento del área de auditoría a mitad de tarea "
            f'("{getattr(last, "content", "")}"). Ajusta las prioridades del plan considerando '
            "esta nueva indicación, sin perder de vista lo que ya se había avanzado."
        )

    def replanner_node(self, state: AgentState) -> dict:
        pregunta = _first_human_message_content(state["messages"])
        motivo = self._motivo_replanificacion(state)
        contexto = f"Pregunta original del área de auditoría:\n{pregunta}\n\n{motivo}"
        pasos_previos = (state.get("plan") or {}).get("pasos", [])
        pasos = self._generar_pasos_plan(contexto, label="replanner") or pasos_previos
        plan = PlanAuditoria(pasos=pasos, version=state["plan_version"] + 1)
        return {
            "plan": plan.model_dump(),
            "plan_version": plan.version,
            "messages": [HumanMessage(content=plan.render(es_replanificacion=True))],
        }

    def llm_node(self, state: AgentState) -> dict:
        """Invoca al LLM con las herramientas disponibles. Si el modelo
        genera una llamada a una herramienta inexistente o mal formada
        (observado en vivo: el modelo escribió "judzar_relevancia" en vez
        de "juzgar_relevancia"), el proveedor rechaza la petición ANTES de
        que llegue un tool_call ejecutable -no es un error de ejecución de
        herramienta, es un error de generación-, así que no lo captura
        `tools.py`. Se reintenta acotado con una corrección explícita; si
        se agotan los reintentos, el turno termina con un mensaje de error
        en vez de propagar la excepción y tumbar el grafo.

        Los errores de RATE LIMIT (`RateLimitError`) se EXCLUYEN de este
        manejo y se dejan propagar: observado en vivo, capturarlos aquí
        los convertía en "no se pudo completar este paso" (un turno vacío
        en segundos) en vez de dejar que quien orquesta el turno (p.ej.
        `_run_with_rate_limit_backoff` en los tests `live`) espere a que
        se reabra la ventana del minuto y reintente. Es un acoplamiento
        específico de Groq (el proveedor de chat activo); habría que
        generalizarlo si el proveedor por defecto cambia."""
        messages = state["messages"]
        response: AIMessage | None = None
        last_exc: Exception | None = None
        start = time.perf_counter()

        for attempt in range(MAX_LLM_GENERATION_RETRIES + 1):
            try:
                response = self._llm_with_tools.invoke(messages)
                break
            except RateLimitError:
                raise
            except Exception as exc:  # noqa: BLE001 - error de generación del proveedor, no de una herramienta
                last_exc = exc
                if attempt < MAX_LLM_GENERATION_RETRIES:
                    messages = messages + [
                        HumanMessage(
                            content=(
                                "Tu intento anterior de usar una herramienta no fue válido "
                                f"({exc}). Usa EXACTAMENTE uno de los nombres de herramienta "
                                "disponibles, sin inventar ni modificar el nombre."
                            )
                        )
                    ]

        if response is None:
            response = AIMessage(
                content=f"No se pudo completar este paso tras {MAX_LLM_GENERATION_RETRIES + 1} intentos: {last_exc}"
            )
        elif self.deps.token_tracker is not None:
            self.deps.token_tracker.record("llm_node", response, time.perf_counter() - start)

        return {
            "messages": [response],
            "iteration": state["iteration"] + 1,
            "tokens_used": state["tokens_used"] + _extract_tokens(response),
        }

    def tools_node(self, state: AgentState) -> dict:
        last_ai = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        last_call = state.get("last_tool_call")

        for tool_call in last_ai.tool_calls:
            signature = _tool_call_signature(tool_call)
            if signature == last_call:
                payload = {
                    "error": "Llamada idéntica (misma herramienta, mismos argumentos) a la "
                    "inmediatamente anterior; no se vuelve a ejecutar. Cambia de herramienta, "
                    "de argumentos, o concluye con la respuesta disponible."
                }
            else:
                tool = self._tools_by_name[tool_call["name"]]
                payload = tool.invoke(tool_call["args"])
                last_call = signature
            tool_messages.append(
                ToolMessage(content=json.dumps(payload, ensure_ascii=False, default=str), tool_call_id=tool_call["id"])
            )

        return {"messages": tool_messages, "last_tool_call": last_call}

    def route_after_llm(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state["iteration"] >= MAX_ITERATIONS:
            return END
        if state["tokens_used"] >= MAX_TOKENS_PER_TURN:
            return END
        return "tools"

    def route_after_tools(self, state: AgentState) -> str:
        """Dos de los tres disparadores de replanificación (el tercero,
        mensaje de seguimiento, se detecta en `route_start`): el resultado
        de la ÚLTIMA herramienta ejecutada fue un hallazgo de riesgo alto
        (`estado="pendiente_de_aprobación"`) o una fuente externa caída
        (`disponible=False`). Revisa solo el último ToolMessage -cubre el
        caso común de una herramienta a la vez; si el LLM pidiera varias
        herramientas en paralelo en el mismo paso, una señal que no sea la
        última no dispara el replan, limitación conocida y no el caso
        típico aquí."""
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            try:
                payload = json.loads(last.content)
            except (TypeError, ValueError):
                payload = {}
            if payload.get("estado") == "pendiente_de_aprobación" or payload.get("disponible") is False:
                return "replanner"
        return "llm"

    def route_start(self, state: AgentState) -> str:
        """Tercer disparador: si la sesión (`thread_id`) YA tiene un plan
        -esta no es la primera pregunta- la nueva pregunta entra directo a
        `replanner` (mensaje de seguimiento a mitad de tarea), no a
        `planner`. Una sesión nueva siempre entra a `planner`."""
        if state.get("plan"):
            return "replanner"
        return "planner"

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self.planner_node)
        graph.add_node("llm", self.llm_node)
        graph.add_node("tools", self.tools_node)
        graph.add_node("replanner", self.replanner_node)
        graph.add_conditional_edges(START, self.route_start, {"planner": "planner", "replanner": "replanner"})
        graph.add_edge("planner", "llm")
        graph.add_conditional_edges("llm", self.route_after_llm, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", self.route_after_tools, {"replanner": "replanner", "llm": "llm"})
        graph.add_edge("replanner", "llm")
        return graph.compile(checkpointer=MemorySaver())


def build_agent_graph(deps: AgentDependencies):
    """Construye y compila el grafo del agente para una instancia de
    `AgentDependencies` (una por ejecución/turno, ver `tools.py`)."""
    tools = build_agent_tools(deps)
    return AgentGraphBuilder(deps=deps, tools=tools).build()


def run_agent_turn(graph: Any, deps: AgentDependencies, pregunta: str, thread_id: str) -> AgentState:
    """Ejecuta un turno (una pregunta) sobre `graph`, ya compilado con
    `build_agent_graph(deps)`.

    `iteration`/`last_tool_call`/`tokens_used` SIEMPRE se reinician aquí
    (0/None/0): el tope de iteraciones/tokens y la guarda de repetición
    son por turno, no acumulados a lo largo de toda la sesión. `chunk_store`/
    `informe_store` también se renuevan cada turno (ver `tools.py`:
    `generar_informe` se valida como máximo una vez por ejecución).

    `plan`/`plan_version` NO se tocan aquí si la sesión ya tenía un plan
    (se consulta `graph.get_state` antes de invocar): así persisten entre
    preguntas de la MISMA sesión, lo que es justo lo que permite que
    `route_start` detecte "esta es una pregunta de seguimiento, no la
    primera" y fuerce una replanificación en vez de un plan desde cero.
    Una sesión nueva (sin checkpoint previo) sí arranca con
    `plan=None, plan_version=0`, y con el SystemMessage del agente -que
    solo debe aparecer una vez por sesión, no repetirse en cada turno."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    es_sesion_nueva = not snapshot.values.get("plan")

    update: dict = {
        "messages": [HumanMessage(content=pregunta)],
        "chunk_store": deps.chunk_store,
        "informe_store": deps.informe_store,
        "iteration": 0,
        "last_tool_call": None,
        "tokens_used": 0,
    }
    if es_sesion_nueva:
        update["messages"] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + update["messages"]
        update["plan"] = None
        update["plan_version"] = 0

    return graph.invoke(update, config=config)


def extract_tool_call_sequence(messages: list[AnyMessage]) -> list[str]:
    """Secuencia, en orden, de los NOMBRES de herramienta que el LLM pidió
    ejecutar a lo largo de `messages` (incluye llamadas bloqueadas por la
    guarda de repetición, que sí aparecen como tool_calls del LLM aunque
    no se hayan ejecutado de nuevo)."""
    sequence: list[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            sequence.extend(tc["name"] for tc in tool_calls)
    return sequence


# --------------------------------------------------------------------------- #
# Demo / CLI interactiva: `python -m src.agent.graph`
#
# Usa los componentes REALES del proyecto (índice FAISS de data/raw/,
# Groq, memoria de largo plazo persistida en disco) — no los fakes de
# tests/agent_test_helpers.py. Por eso requiere `python -m src.loaders`
# corrido antes (construye el índice) y `GROQ_API_KEY` en `.env`.
# --------------------------------------------------------------------------- #

ESCENARIOS_DEMO: dict[str, dict[str, str]] = {
    "1": {
        "nombre": "Consulta simple (factura F-1023, CLP)",
        "pregunta": "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?",
    },
    "2": {
        "nombre": "Factura en UF (contrato CA-078)",
        "pregunta": (
            "¿Cuál es el canon de arriendo del contrato CA-078 en UF, y a cuánto "
            "equivale en CLP con el valor de la UF de hoy?"
        ),
    },
    "3": {
        "nombre": "Riesgo alto (nota de crédito NC-045)",
        "pregunta": "¿Existen señales de riesgo de fraude en la nota de crédito NC-045?",
    },
    "4": {
        "nombre": "Documento inexistente (guardrail anti-alucinación)",
        "pregunta": "¿Cuál es el RUT del receptor de la factura F-9999?",
    },
    "5": {
        "nombre": "Fuente externa caída (forzada a propósito para la demo, ver nota abajo)",
        "pregunta": (
            "¿Cuál es el canon de arriendo del contrato CA-078 en UF, y a cuánto "
            "equivale en CLP con el valor de la UF de hoy?"
        ),
        "fuente_externa_forzada_no_disponible": "si",
    },
    "6": {
        "nombre": "Memoria de largo plazo (hallazgos previos de NC-045)",
        "pregunta": "¿Qué se sabe de la nota de crédito NC-045? ¿Hay algún hallazgo registrado?",
    },
}
# Nota: "evidencia contradictoria" e "inyección de instrucciones" no están
# en este menú porque requieren documentos sintéticos que NO forman parte
# del corpus real de data/raw/ (no se agregaron ahí para no alterar el
# corpus de la EP1/EP2 sin aprobación). Esos dos escenarios viven en
# tests/test_agent_graph_live.py con sus propios documentos de prueba.


def build_demo_dependencies(forzar_fuente_externa_no_disponible: bool = False) -> AgentDependencies:
    """Construye `AgentDependencies` con los componentes REALES del
    proyecto: índice FAISS de `data/raw/` (requiere haber corrido antes
    `python -m src.loaders`), el modelo de chat configurado (requiere
    `GROQ_API_KEY`), embeddings locales, y memoria de largo plazo
    persistida en disco (`settings.CASE_LOG_PATH` /
    `settings.FINDINGS_INDEX_DIR`, ambos ignorados por git)."""
    settings.validate_chat()
    vectorstore = load_faiss_index()

    def external_fetch() -> IndicadoresExternos:
        if forzar_fuente_externa_no_disponible:
            return IndicadoresExternos(
                fecha_consulta="(forzado por la demo)",
                disponible=False,
                error="Forzado a no disponible por el escenario 'fuente externa caída' de la demo.",
            )
        return fetch_indicadores_economicos()

    return AgentDependencies(
        retriever=vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k}),
        llm=get_chat_model(),
        chunk_store=ChunkStore(),
        informe_store=InformeStore(),
        case_log=CaseLog(CASE_LOG_PATH),
        findings_index=FindingsIndex(get_embeddings(), persist_dir=FINDINGS_INDEX_DIR, index_name=FINDINGS_INDEX_NAME),
        external_fetch=external_fetch,
        token_tracker=TokenTracker(model_name=settings.groq_chat_model),
    )


def _pedir_entrada(mensaje: str, input_fn) -> str:
    try:
        return input_fn(mensaje)
    except (EOFError, StopIteration):
        return ""


def _imprimir_traza(mensajes_nuevos: list[AnyMessage], print_fn) -> None:
    """Pensamiento -> Acción -> Observación, a partir de los mensajes
    NUEVOS de este turno (no todo el historial de la sesión)."""
    for msg in mensajes_nuevos:
        tipo = type(msg).__name__
        contenido = getattr(msg, "content", "") or ""
        if tipo == "AIMessage":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    print_fn(f"  [Acción] {tc['name']}({tc['args']})")
            elif contenido:
                print_fn(f"  [Pensamiento/Respuesta] {contenido}")
        elif tipo == "ToolMessage":
            recorte = contenido if len(contenido) <= 300 else contenido[:300] + "…"
            print_fn(f"  [Observación] {recorte}")
        elif tipo == "HumanMessage" and contenido.startswith("[PLAN"):
            print_fn(f"  [Plan] {contenido}")


def run_demo_cli(deps: AgentDependencies, input_fn=input, print_fn=print) -> None:
    """Bucle interactivo de la demo (`python -m src.agent.graph`). No
    requiere editar código: el `thread_id`, el escenario/pregunta, y
    cuándo terminar se piden todos por `input_fn` (por defecto `input()`
    real). `deps`/`input_fn`/`print_fn` son inyectables para poder
    probar este bucle de forma determinista sin gastar API -ver
    `tests/test_agent_graph.py::test_run_demo_cli_...`-."""
    print_fn("=== Demo del agente de auditoría (EP2) ===")
    thread_id = (
        _pedir_entrada(
            "ID de sesión (Enter para generar uno nuevo; reutiliza uno existente para continuar esa sesión): ",
            input_fn,
        ).strip()
        or f"demo-{int(time.time())}"
    )
    print_fn(f"Sesión: {thread_id}\n")

    graph = build_agent_graph(deps)

    while True:
        print_fn("Escenarios documentados:")
        for clave, escenario in ESCENARIOS_DEMO.items():
            print_fn(f"  {clave}) {escenario['nombre']}")
        print_fn("  (o escribe directamente tu propia pregunta)")
        entrada = _pedir_entrada("\nElige un número, escribe una pregunta, o 'salir': ", input_fn).strip()

        if not entrada or entrada.lower() in {"salir", "exit", "quit"}:
            break

        escenario = ESCENARIOS_DEMO.get(entrada)
        pregunta = escenario["pregunta"] if escenario else entrada

        external_fetch_original = deps.external_fetch
        if escenario and escenario.get("fuente_externa_forzada_no_disponible"):
            deps.external_fetch = lambda: IndicadoresExternos(
                fecha_consulta="(forzado por la demo)",
                disponible=False,
                error="Forzado a no disponible por el escenario 'fuente externa caída' de la demo.",
            )

        snapshot_previo = graph.get_state({"configurable": {"thread_id": thread_id}})
        mensajes_antes = len(snapshot_previo.values.get("messages", [])) if snapshot_previo.values else 0
        registros_antes = len(deps.token_tracker.records) if deps.token_tracker is not None else 0

        print_fn(f"\n>>> {pregunta}\n")
        resultado = run_agent_turn(graph, deps, pregunta, thread_id=thread_id)

        deps.external_fetch = external_fetch_original

        _imprimir_traza(resultado["messages"][mensajes_antes:], print_fn)

        plan = resultado.get("plan") or {}
        print_fn(f"\n  [Plan vigente v{resultado.get('plan_version', 0)}] {len(plan.get('pasos', []))} paso(s)")

        if deps.token_tracker is not None:
            registros_turno = deps.token_tracker.records[registros_antes:]
            tokens_in = sum(r.input_tokens for r in registros_turno)
            tokens_out = sum(r.output_tokens for r in registros_turno)
            costo = sum(r.estimated_cost_usd for r in registros_turno)
            latencia = sum(r.latency_seconds for r in registros_turno)
            print_fn(
                f"  [Este turno] {len(registros_turno)} llamadas | {tokens_in} in / {tokens_out} out | "
                f"~${costo:.6f} USD | {latencia:.1f}s"
            )
        print_fn("")

    if deps.token_tracker is not None:
        resumen = deps.token_tracker.summary()
        print_fn(
            f"=== Resumen de la sesión: {resumen['num_llamadas']} llamadas | "
            f"{resumen['tokens_entrada_total']} in / {resumen['tokens_salida_total']} out | "
            f"~${resumen['costo_estimado_total_usd']:.6f} USD ==="
        )


if __name__ == "__main__":
    run_demo_cli(build_demo_dependencies())
