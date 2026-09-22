"""
Cadena secundaria de verificación fáctica (fact-checking).

Recibe la respuesta ya generada por la cadena principal junto con el
contexto que se usó para generarla, y clasifica cada afirmación en
[Soportado] / [Contradictorio] / [No Mencionado], citando evidencia textual.

Esta cadena es independiente de la cadena principal (se ejecuta DESPUÉS,
como un segundo LLM "auditando" al primero) para reducir el riesgo de que
el mismo sesgo/alucinación de la generación contamine su propia verificación.
Su salida (`FactCheckReport`) alimenta directamente la métrica
`faithfulness` en `src/metrics.py`.
"""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from src.config import get_chat_model
from src.metrics import FactCheckReport
from src.prompts import FACT_CHECK_PROMPT_TEMPLATE

_parser = PydanticOutputParser(pydantic_object=FactCheckReport)


def build_fact_checking_chain() -> Runnable:
    """Construye la cadena LCEL de fact-checking: prompt | llm | parser."""
    llm = get_chat_model()
    return (
        FACT_CHECK_PROMPT_TEMPLATE.partial(format_instructions=_parser.get_format_instructions())
        | llm
        | _parser
    )


def verify_answer(answer: str, context: str) -> FactCheckReport:
    """Ejecuta la verificación fáctica de `answer` contra `context`.

    Devuelve un `FactCheckReport` con el veredicto y evidencia por cada
    afirmación detectada. Requiere credenciales de LLM configuradas (ver
    `src/config.py`); para pruebas deterministas sin credenciales usar
    `tests/test_benchmark.py`, que no depende de esta función.
    """
    chain = build_fact_checking_chain()
    return chain.invoke({"context": context, "answer": answer})
