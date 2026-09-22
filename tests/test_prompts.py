"""
Pruebas estáticas sobre las plantillas de prompt (src/prompts.py): no
invocan ningún LLM, solo verifican que los guardrails y técnicas de prompt
engineering exigidas por el enunciado estén efectivamente presentes en el
texto de los prompts (evita regresiones silenciosas al editar prompts.py).
"""

from __future__ import annotations

from src.prompts import (
    AUDIT_SYSTEM_PROMPT,
    FACT_CHECK_SYSTEM_PROMPT,
    NO_DATA_RULE,
)


def test_system_prompt_has_xml_guardrail_delimiters() -> None:
    assert "<instrucciones>" in AUDIT_SYSTEM_PROMPT
    assert "</instrucciones>" in AUDIT_SYSTEM_PROMPT
    assert "<ejemplos_few_shot>" in AUDIT_SYSTEM_PROMPT


def test_system_prompt_has_anti_hallucination_rule() -> None:
    assert "No dispongo de esta información en los documentos auditados" in AUDIT_SYSTEM_PROMPT
    assert NO_DATA_RULE in AUDIT_SYSTEM_PROMPT


def test_system_prompt_has_chain_of_thought_steps() -> None:
    for step_keyword in ["identifica", "verifica", "evalúa", "concluye"]:
        assert step_keyword in AUDIT_SYSTEM_PROMPT.lower()


def test_system_prompt_has_few_shot_examples() -> None:
    assert AUDIT_SYSTEM_PROMPT.count("Output:") >= 3


def test_fact_check_prompt_defines_three_verdicts() -> None:
    for verdict in ["[Soportado]", "[Contradictorio]", "[No Mencionado]"]:
        assert verdict in FACT_CHECK_SYSTEM_PROMPT
