"""Grafo del agente (EP2, IE1/IE2/IE5): el LLM decide qué herramienta usar
y en qué orden; el grafo solo ejecuta esa decisión, no fija el orden.

Bucle mínimo (LangGraph):

    START -> llm -> (¿tool_calls?) -> tools -> llm -> ... -> END

`llm` invoca el modelo con las 7 herramientas de `tools.py` vía
`bind_tools`; mientras la última respuesta traiga `tool_calls`, el grafo
pasa a `tools`, las ejecuta, y vuelve a `llm` con las observaciones. Si la
respuesta no trae `tool_calls`, o se alcanza el tope de iteraciones/tokens
del turno, el grafo termina.

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
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.chunk_store import ChunkStore
from src.agent.informe_store import InformeStore
from src.agent.tools import AgentDependencies, build_agent_tools
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
""".strip()


# --------------------------------------------------------------------------- #
# Estado del grafo
# --------------------------------------------------------------------------- #


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    chunk_store: ChunkStore
    informe_store: InformeStore
    # Reservados para el paso de planificación priorizada (aún no se usan).
    plan: dict | None
    plan_version: int
    iteration: int
    last_tool_call: dict | None
    tokens_used: int


def initial_state(pregunta: str, deps: AgentDependencies) -> AgentState:
    """Estado inicial de UN turno (una pregunta). `iteration`/
    `last_tool_call`/`tokens_used` SIEMPRE arrancan en 0/None/0 aquí: el
    tope de iteraciones/tokens y la guarda de repetición son por turno,
    no acumulados a lo largo de toda la sesión."""
    return AgentState(
        messages=[SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=pregunta)],
        chunk_store=deps.chunk_store,
        informe_store=deps.informe_store,
        plan=None,
        plan_version=0,
        iteration=0,
        last_tool_call=None,
        tokens_used=0,
    )


def _extract_tokens(response: AIMessage) -> int:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return int(usage.get("total_tokens", 0))
    return 0


def _tool_call_signature(tool_call: dict) -> dict:
    return {"name": tool_call["name"], "args": tool_call["args"]}


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

    def llm_node(self, state: AgentState) -> dict:
        """Invoca al LLM con las herramientas disponibles. Si el modelo
        genera una llamada a una herramienta inexistente o mal formada
        (observado en vivo: el modelo escribió "judzar_relevancia" en vez
        de "juzgar_relevancia"), el proveedor rechaza la petición ANTES de
        que llegue un tool_call ejecutable -no es un error de ejecución de
        herramienta, es un error de generación-, así que no lo captura
        `tools.py`. Se reintenta acotado con una corrección explícita; si
        se agotan los reintentos, el turno termina con un mensaje de error
        en vez de propagar la excepción y tumbar el grafo."""
        messages = state["messages"]
        response: AIMessage | None = None
        last_exc: Exception | None = None

        for attempt in range(MAX_LLM_GENERATION_RETRIES + 1):
            try:
                response = self._llm_with_tools.invoke(messages)
                break
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

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.llm_node)
        graph.add_node("tools", self.tools_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", self.route_after_llm, {"tools": "tools", END: END})
        graph.add_edge("tools", "llm")
        return graph.compile(checkpointer=MemorySaver())


def build_agent_graph(deps: AgentDependencies):
    """Construye y compila el grafo del agente para una instancia de
    `AgentDependencies` (una por ejecución/turno, ver `tools.py`)."""
    tools = build_agent_tools(deps)
    return AgentGraphBuilder(deps=deps, tools=tools).build()


def run_agent_turn(graph: Any, deps: AgentDependencies, pregunta: str, thread_id: str) -> AgentState:
    """Ejecuta un turno completo (una pregunta) sobre `graph`, ya
    compilado con `build_agent_graph(deps)`. Reinicia iteration/
    last_tool_call/tokens_used (ver `initial_state`); el checkpointer
    conserva igualmente `messages` de turnos previos con el mismo
    `thread_id`."""
    state = initial_state(pregunta, deps)
    return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})


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
