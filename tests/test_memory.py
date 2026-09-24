"""
Pruebas deterministas de la memoria conversacional (src/memory.py, IE4).

NO gastan cuota real de Groq: la llamada LLM interna de `SummaryMemory`
(actualización del resumen) se mockea por completo con `unittest.mock`.
Cubren:
  1) BufferMemory preserva el historial completo sin pérdida.
  2) SummaryMemory efectivamente comprime frente a Buffer al crecer la
     conversación (y usa el LLM configurado para actualizarse).
  3) Ambas exponen la misma interfaz que espera `rag_pipeline.py`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory import BufferMemory, SummaryMemory, Turn, create_memory

# --------------------------------------------------------------------------- #
# 1) BufferMemory: preserva el historial completo, sin pérdida
# --------------------------------------------------------------------------- #


def test_buffer_memory_empty_context_when_no_turns() -> None:
    memory = BufferMemory()
    assert memory.get_context_for_prompt() == ""
    assert memory.turn_count() == 0


def test_buffer_memory_preserves_all_turns_in_order_without_modification() -> None:
    memory = BufferMemory()
    memory.add_turn("¿Cuál es el monto total de F-1023?", "monto_total=1245000 CLP")
    memory.add_turn("¿Y el RUT del emisor?", "rut_emisor=76.123.456-7")
    memory.add_turn("¿Y el nivel de riesgo?", "nivel_riesgo_fraude=bajo")

    assert memory.turn_count() == 3
    assert memory.turns == [
        Turn(question="¿Cuál es el monto total de F-1023?", answer="monto_total=1245000 CLP"),
        Turn(question="¿Y el RUT del emisor?", answer="rut_emisor=76.123.456-7"),
        Turn(question="¿Y el nivel de riesgo?", answer="nivel_riesgo_fraude=bajo"),
    ]

    contexto = memory.get_context_for_prompt()
    # Cada pregunta y respuesta debe aparecer TEXTUALMENTE, sin resumir ni truncar.
    assert "¿Cuál es el monto total de F-1023?" in contexto
    assert "monto_total=1245000 CLP" in contexto
    assert "¿Y el RUT del emisor?" in contexto
    assert "rut_emisor=76.123.456-7" in contexto
    assert "¿Y el nivel de riesgo?" in contexto
    assert "nivel_riesgo_fraude=bajo" in contexto
    assert "[HISTORIAL_CONVERSACION - buffer completo]" in contexto


def test_buffer_memory_ignores_config_param_no_llm_call() -> None:
    """BufferMemory nunca debe intentar usar `config` (no hace llamadas LLM)."""
    memory = BufferMemory()
    # No debe lanzar ninguna excepción ni requerir credenciales.
    memory.add_turn("pregunta", "respuesta", config={"callbacks": ["cualquier-cosa"]})
    assert memory.turn_count() == 1


# --------------------------------------------------------------------------- #
# 2) SummaryMemory: comprime frente a Buffer, usa el LLM configurado (mockeado)
# --------------------------------------------------------------------------- #


def _fake_llm_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = text
    return response


def test_summary_memory_calls_configured_chat_model() -> None:
    """SummaryMemory debe actualizarse invocando get_chat_model() de src.config
    (el mismo proveedor configurado, Groq por defecto), no un cliente propio.
    """
    fake_llm = MagicMock()
    fake_llm.__or__ = MagicMock()  # soporta el operador `|` de LCEL si se usa directo

    with patch("src.memory.get_chat_model", return_value=fake_llm) as mock_get_chat_model:
        memory = SummaryMemory()
        # Parcheamos también el chain resultante de `_SUMMARY_UPDATE_TEMPLATE | llm`
        with patch("src.memory._SUMMARY_UPDATE_TEMPLATE") as mock_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = _fake_llm_response("Resumen: factura F-1023 por 1245000 CLP.")
            mock_template.__or__.return_value = mock_chain

            memory.add_turn("¿Cuál es el monto de F-1023?", "monto_total=1245000 CLP")

    mock_get_chat_model.assert_called_once()
    mock_chain.invoke.assert_called_once()
    assert memory.summary == "Resumen: factura F-1023 por 1245000 CLP."
    assert memory.turn_count() == 1


def test_summary_memory_compresses_context_vs_buffer_after_several_turns() -> None:
    """Tras varios turnos, el contexto de SummaryMemory debe ser
    significativamente más chico que el de BufferMemory para la MISMA
    conversación — esa es la propiedad central que se compara en
    docs/comparacion_memoria.md.
    """
    turns = [
        ("¿Cuál es el monto total y el RUT del emisor en la factura F-1023?", "monto_total=1245000 CLP; rut_emisor=76.123.456-7"),
        ("¿Existen señales de riesgo de fraude en la nota de crédito NC-045?", "monto_total=5000000 CLP; nivel_riesgo_fraude=alto; alertas=Emisión múltiple del mismo monto en 24 horas"),
        ("¿Cuál es el resultado operacional del Q2 2025?", "monto_total=368000000 CLP; nivel_riesgo_fraude=bajo"),
        ("¿Y cuál era el RUT de esa misma factura F-1023 otra vez?", "rut_emisor=76.123.456-7"),
    ]
    # El resumen mockeado se mantiene deliberadamente corto y constante en
    # tamaño (como debería comportarse un resumen real "progresivo pero
    # acotado"), a diferencia del buffer que crece con cada turno.
    resumen_fijo = "Resumen: F-1023 (RUT 76.123.456-7, $1.245.000), NC-045 riesgo alto, Q2 2025 resultado $368.000.000."

    buffer_memory = BufferMemory()
    summary_memory = SummaryMemory()
    summary_memory._llm = MagicMock()  # evita que _get_llm() intente resolver credenciales reales

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [_fake_llm_response(resumen_fijo) for _ in turns]

    with patch("src.memory._SUMMARY_UPDATE_TEMPLATE") as mock_template:
        mock_template.__or__.return_value = mock_chain
        for question, answer in turns:
            buffer_memory.add_turn(question, answer)
            summary_memory.add_turn(question, answer)

    buffer_context = buffer_memory.get_context_for_prompt()
    summary_context = summary_memory.get_context_for_prompt()

    assert buffer_memory.turn_count() == len(turns)
    assert summary_memory.turn_count() == len(turns)
    # La propiedad central: tras 4 turnos, el resumen es notablemente más
    # compacto que el historial completo.
    assert len(summary_context) < len(buffer_context)
    assert "[HISTORIAL_CONVERSACION - resumen progresivo]" in summary_context
    assert resumen_fijo in summary_context


def test_summary_memory_empty_context_when_no_turns() -> None:
    memory = SummaryMemory()
    assert memory.get_context_for_prompt() == ""
    assert memory.turn_count() == 0


# --------------------------------------------------------------------------- #
# 3) Interfaz uniforme esperada por rag_pipeline.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", ["buffer", "summary"])
def test_create_memory_factory_returns_expected_type(strategy: str) -> None:
    memory = create_memory(strategy)
    assert hasattr(memory, "add_turn")
    assert hasattr(memory, "get_context_for_prompt")
    assert hasattr(memory, "turn_count")
    assert callable(memory.add_turn)
    assert callable(memory.get_context_for_prompt)
    assert callable(memory.turn_count)
    assert isinstance(memory, BufferMemory if strategy == "buffer" else SummaryMemory)


def test_create_memory_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Estrategia de memoria no soportada"):
        create_memory("no_existe")


def test_create_memory_defaults_to_settings_memory_strategy() -> None:
    with patch("src.memory.settings") as mock_settings:
        mock_settings.memory_strategy = "buffer"
        memory = create_memory()
    assert isinstance(memory, BufferMemory)


def test_both_strategies_start_empty_and_report_zero_turns() -> None:
    """Invariante de interfaz: antes de cualquier add_turn, ambas estrategias
    deben reportar exactamente lo mismo (contexto vacío, cero turnos), para
    que rag_pipeline.py pueda tratarlas indistintamente."""
    buffer_memory = BufferMemory()
    summary_memory = SummaryMemory()

    assert buffer_memory.get_context_for_prompt() == summary_memory.get_context_for_prompt() == ""
    assert buffer_memory.turn_count() == summary_memory.turn_count() == 0
