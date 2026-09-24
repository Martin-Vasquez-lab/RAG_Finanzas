"""
Gestión de memoria conversacional para el pipeline de auditoría (IE4).

Dos estrategias intercambiables por configuración (`MEMORY_STRATEGY` en
`.env` / `src/config.py`), evaluadas empíricamente en
`docs/comparacion_memoria.md`:

- **BufferMemory**: guarda cada turno (pregunta, respuesta) tal cual, sin
  modificar, en orden. Simple y sin costo adicional de LLM, pero el
  contexto crece linealmente con la conversación — más tokens por pregunta
  a medida que avanza la sesión, y preserva las cifras exactas (montos,
  RUT) sin ningún riesgo de que un resumen las distorsione.
- **SummaryMemory**: mantiene un resumen progresivo de la conversación en
  vez del historial completo, actualizado con una llamada EXTRA al LLM
  (mismo proveedor de chat configurado, Groq) cada vez que se agrega un
  turno. El contexto se mantiene casi constante en tamaño sin importar
  cuántos turnos lleve la sesión, pero cada turno cuesta una llamada extra
  y el resumen, al ser generado por LLM, no tiene garantía matemática de
  preservar cifras exactas al comprimir.

Ambas exponen la MISMA interfaz (`ConversationMemory`) para que
`src/rag_pipeline.py` sea agnóstico a cuál está activa — ver
`create_memory()`.

Las instancias de memoria viven en memoria del PROCESO, indexadas por
`session_id` (sin persistencia en disco): alcanzan para el ciclo de vida
de un script, notebook o sesión de servidor; no sobreviven un reinicio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.prompts import ChatPromptTemplate

from src.config import get_chat_model, settings

# --------------------------------------------------------------------------- #
# Interfaz común
# --------------------------------------------------------------------------- #


class ConversationMemory(Protocol):
    """Interfaz que `rag_pipeline.py` espera de cualquier estrategia de memoria.

    `config` (opcional) permite propagar un `AuditCallbackHandler` a
    cualquier llamada LLM interna de la estrategia (solo lo usa
    `SummaryMemory`, para que su llamada extra de resumen quede contabilizada
    en las mismas métricas de latencia/tokens/costo del turno).
    """

    def add_turn(self, question: str, answer: str, config: dict | None = None) -> None: ...

    def get_context_for_prompt(self) -> str: ...

    def turn_count(self) -> int: ...


@dataclass
class Turn:
    """Un turno de conversación: la pregunta original y la respuesta (resumida
    a texto legible, ver `rag_pipeline._render_answer_for_memory`)."""

    question: str
    answer: str


# --------------------------------------------------------------------------- #
# Estrategia 1: BufferMemory (historial completo)
# --------------------------------------------------------------------------- #


@dataclass
class BufferMemory:
    """Guarda el historial completo de la sesión, turno por turno, sin modificar.

    Ventaja: cero pérdida de información, cero llamadas extra al LLM.
    Desventaja: el bloque de contexto crece sin límite con la conversación.
    """

    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str, config: dict | None = None) -> None:
        # `config` no se usa: BufferMemory no hace ninguna llamada LLM.
        self.turns.append(Turn(question=question, answer=answer))

    def get_context_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lineas = ["[HISTORIAL_CONVERSACION - buffer completo]"]
        for i, turn in enumerate(self.turns, start=1):
            lineas.append(f"Turno {i} - Pregunta: {turn.question}")
            lineas.append(f"Turno {i} - Respuesta: {turn.answer}")
        lineas.append("[/HISTORIAL_CONVERSACION]")
        return "\n".join(lineas)

    def turn_count(self) -> int:
        return len(self.turns)


# --------------------------------------------------------------------------- #
# Estrategia 2: SummaryMemory (resumen progresivo)
# --------------------------------------------------------------------------- #

# El prompt de actualización instruye explícitamente a preservar cifras
# exactas (montos, RUT, folios, fechas): es la mitigación de diseño contra
# la pérdida de precisión al resumir, pero al depender de un LLM no hay
# garantía matemática — por eso se mide empíricamente en
# docs/comparacion_memoria.md si esa instrucción alcanza en la práctica.
_SUMMARY_UPDATE_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente que mantiene un resumen conciso y fiel de una "
            "conversación de auditoría financiera. Actualiza el resumen "
            "existente incorporando el nuevo turno (pregunta + respuesta), "
            "PRESERVANDO EXPLÍCITAMENTE todo dato numérico exacto (montos, "
            "RUT, folios, fechas) que aparezca en el nuevo turno o que ya "
            "estuviera en el resumen anterior. No omitas cifras aunque el "
            "resumen deba ser breve. Responde SOLO con el nuevo resumen, en "
            "prosa breve, sin explicaciones ni saludos adicionales.",
        ),
        (
            "human",
            "Resumen anterior:\n{resumen_previo}\n\n"
            "Nuevo turno:\nPregunta: {question}\nRespuesta: {answer}\n\n"
            "Nuevo resumen actualizado:",
        ),
    ]
)


@dataclass
class SummaryMemory:
    """Mantiene un resumen progresivo de la conversación en vez del historial completo.

    Actualiza el resumen con una llamada EXTRA al LLM (mismo proveedor de
    chat configurado vía `src/config.get_chat_model()`) cada vez que se
    agrega un turno.
    """

    summary: str = ""
    _turn_count: int = field(default=0, repr=False)
    _llm: object = field(default=None, repr=False, compare=False)

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def add_turn(self, question: str, answer: str, config: dict | None = None) -> None:
        llm = self._get_llm()
        chain = _SUMMARY_UPDATE_TEMPLATE | llm
        response = chain.invoke(
            {
                "resumen_previo": self.summary or "(vacío, es la primera pregunta de la sesión)",
                "question": question,
                "answer": answer,
            },
            config=config or {},
        )
        self.summary = getattr(response, "content", str(response)).strip()
        self._turn_count += 1

    def get_context_for_prompt(self) -> str:
        if not self.summary:
            return ""
        return f"[HISTORIAL_CONVERSACION - resumen progresivo]\n{self.summary}\n[/HISTORIAL_CONVERSACION]"

    def turn_count(self) -> int:
        return self._turn_count


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def create_memory(strategy: str | None = None) -> ConversationMemory:
    """Instancia la estrategia de memoria solicitada (o la de `settings.memory_strategy`)."""
    strategy = (strategy or settings.memory_strategy).lower()
    if strategy == "buffer":
        return BufferMemory()
    if strategy == "summary":
        return SummaryMemory()
    raise ValueError(f"Estrategia de memoria no soportada: {strategy!r} (usa 'buffer' o 'summary')")
