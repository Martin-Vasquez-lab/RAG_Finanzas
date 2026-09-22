"""
Script de benchmarking determinista: Exact Match + RegEx sobre RUT/montos.

Este benchmark NO depende de credenciales de LLM: valida, contra un
ground truth pequeño (`tests/ground_truth.json`), que los extractores por
RegEx de RUT y montos en CLP funcionan correctamente sobre los documentos
fuente reales del proyecto (`data/raw/`). Es la misma lógica que se usaría
para calificar automáticamente la salida `InformeAuditoria` del LLM
(comparando `informe.rut_emisor` / `informe.monto_total` contra estos
valores esperados vía Exact Match) sin necesitar invocar el modelo en CI.

Ejecutar con: `pytest tests/test_benchmark.py -v`
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.json"

# RUT chileno: 1-2 dígitos, puntos opcionales por millar, guion, dígito verificador o K.
RUT_REGEX = re.compile(r"\b\d{1,2}(?:\.\d{3}){2}-[\dkK]\b")

# Montos en CLP con separador de miles por puntos, precedidos por "$".
MONTO_CLP_REGEX = re.compile(r"\$\s?([\d.]+)\s?CLP")


def extract_ruts(text: str) -> list[str]:
    """Extrae todos los RUT con formato chileno estándar presentes en `text`."""
    return RUT_REGEX.findall(text)


def extract_montos_clp(text: str) -> list[float]:
    """Extrae todos los montos en CLP (formato "$1.245.000 CLP") como float."""
    matches = MONTO_CLP_REGEX.findall(text)
    return [float(m.replace(".", "")) for m in matches]


def _load_ground_truth() -> list[dict]:
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as f:
        return json.load(f)


GROUND_TRUTH = _load_ground_truth()


@pytest.mark.parametrize("case", GROUND_TRUTH, ids=lambda c: c["id"])
def test_rut_exact_match(case: dict) -> None:
    """El RUT esperado (si existe) debe extraerse exactamente del documento fuente."""
    if case["expected_rut"] is None:
        pytest.skip(f"Caso {case['id']} no involucra un RUT específico.")

    text = (DATA_RAW_DIR / case["source_file"]).read_text(encoding="utf-8")
    found_ruts = extract_ruts(text)

    assert case["expected_rut"] in found_ruts, (
        f"RUT esperado {case['expected_rut']!r} no encontrado (Exact Match) "
        f"en {case['source_file']}. RUT detectados: {found_ruts}"
    )


@pytest.mark.parametrize("case", GROUND_TRUTH, ids=lambda c: c["id"])
def test_monto_exact_match(case: dict) -> None:
    """El monto esperado debe extraerse exactamente (Exact Match) del documento fuente."""
    text = (DATA_RAW_DIR / case["source_file"]).read_text(encoding="utf-8")
    found_montos = extract_montos_clp(text)

    assert case["expected_monto_clp"] in found_montos, (
        f"Monto esperado {case['expected_monto_clp']} no encontrado (Exact Match) "
        f"en {case['source_file']}. Montos detectados: {found_montos}"
    )


def test_ground_truth_files_exist() -> None:
    """Sanity check: todos los archivos fuente referenciados en el ground truth existen."""
    for case in GROUND_TRUTH:
        path = DATA_RAW_DIR / case["source_file"]
        assert path.exists(), f"Falta el documento fuente {path} referenciado en ground_truth.json"
