"""
Pruebas unitarias deterministas de las 4 métricas RAG (src/metrics.py).

No requieren credenciales de LLM: `context_precision`, `context_recall` y
`answer_relevancy` son puramente aritméticas/léxicas, y `faithfulness` se
prueba con una lista de `ClaimVerification` construida a mano (simulando la
salida de `fact_checking.py` sin invocarlo).
"""

from __future__ import annotations

import pytest

from src.metrics import (
    ClaimVerification,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def test_context_precision_perfect_match() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1", "c2", "c3"}
    assert context_precision(retrieved, relevant) == 1.0


def test_context_precision_partial_match() -> None:
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3"}
    assert context_precision(retrieved, relevant) == 0.5


def test_context_precision_empty_retrieved() -> None:
    assert context_precision([], {"c1"}) == 0.0


def test_context_recall_perfect_match() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1", "c2"}
    assert context_recall(retrieved, relevant) == 1.0


def test_context_recall_partial_match() -> None:
    retrieved = ["c1"]
    relevant = {"c1", "c2"}
    assert context_recall(retrieved, relevant) == 0.5


def test_context_recall_no_relevant_docs() -> None:
    assert context_recall(["c1"], set()) == 1.0


def test_faithfulness_all_supported() -> None:
    claims = [
        ClaimVerification(afirmacion="a", veredicto="Soportado", evidencia="x"),
        ClaimVerification(afirmacion="b", veredicto="Soportado", evidencia="y"),
    ]
    assert faithfulness(claims) == 1.0


def test_faithfulness_mixed() -> None:
    claims = [
        ClaimVerification(afirmacion="a", veredicto="Soportado", evidencia="x"),
        ClaimVerification(afirmacion="b", veredicto="Contradictorio", evidencia="y"),
        ClaimVerification(afirmacion="c", veredicto="No Mencionado", evidencia="sin evidencia"),
        ClaimVerification(afirmacion="d", veredicto="Soportado", evidencia="z"),
    ]
    assert faithfulness(claims) == 0.5


def test_faithfulness_empty() -> None:
    assert faithfulness([]) == 0.0


def test_answer_relevancy_identical_text_is_maximal() -> None:
    text = "monto total factura F-1023 riesgo bajo"
    # pytest.approx: la similitud coseno de un vector consigo mismo puede
    # rendir 0.9999999999999999 por redondeo de punto flotante en sqrt().
    assert answer_relevancy(text, text) == pytest.approx(1.0)


def test_answer_relevancy_unrelated_text_is_low() -> None:
    question = "cual es el monto total de la factura"
    answer = "el clima en santiago esta nublado hoy"
    assert answer_relevancy(answer, question) < 0.3


def test_answer_relevancy_bounds() -> None:
    score = answer_relevancy("el monto total es 1245000 clp", "cual es el monto total")
    assert 0.0 <= score <= 1.0
