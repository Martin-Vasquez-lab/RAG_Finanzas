"""
Pruebas deterministas de la fuente EXTERNA (src/external_data.py, IL1.2/IE3).

Mockean por completo `requests.get` — NO hacen ninguna llamada de red real
a mindicador.cl ni gastan cuota de la API de Groq. Cubren:
  1) Respuesta exitosa: se parsean UF/dólar/UTM y se formatean con la
     etiqueta "[FUENTE EXTERNA: ...]".
  2) Fallas (timeout, HTTP error, JSON malformado): degradación con gracia
     (`disponible=False`), nunca una excepción.
  3) Que `rag_pipeline.merge_context` efectivamente inyecta el bloque de la
     fuente externa junto al contexto interno (chunks de FAISS).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.external_data import (
    IndicadoresExternos,
    fetch_indicadores_economicos,
    format_external_context,
)
from src.rag_pipeline import merge_context

MOCK_RESPONSE_OK: dict = {
    "version": "1.5.0",
    "autor": "mindicador.cl",
    "fecha": "2026-09-22T03:00:00.000Z",
    "uf": {
        "codigo": "uf",
        "nombre": "Unidad de fomento (UF)",
        "unidad_medida": "Pesos",
        "fecha": "2026-09-22T03:00:00.000Z",
        "valor": 39000.12,
    },
    "dolar": {
        "codigo": "dolar",
        "nombre": "Dólar observado",
        "unidad_medida": "Pesos",
        "fecha": "2026-09-22T03:00:00.000Z",
        "valor": 970.5,
    },
    "utm": {
        "codigo": "utm",
        "nombre": "Unidad Tributaria Mensual (UTM)",
        "unidad_medida": "Pesos",
        "fecha": "2026-09-01T03:00:00.000Z",
        "valor": 68000.0,
    },
}


def _mock_response(json_data: dict, status_ok: bool = True) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    if status_ok:
        mock_resp.raise_for_status.return_value = None
    else:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    return mock_resp


# --------------------------------------------------------------------------- #
# 1) Respuesta exitosa
# --------------------------------------------------------------------------- #

def test_fetch_indicadores_exitoso_parsea_uf_dolar_utm() -> None:
    with patch("src.external_data.requests.get", return_value=_mock_response(MOCK_RESPONSE_OK)):
        resultado = fetch_indicadores_economicos()

    assert resultado.disponible is True
    assert resultado.error is None
    assert resultado.indicadores["uf"].valor == 39000.12
    assert resultado.indicadores["dolar"].valor == 970.5
    assert resultado.indicadores["utm"].valor == 68000.0


def test_format_external_context_incluye_etiqueta_fuente_externa() -> None:
    with patch("src.external_data.requests.get", return_value=_mock_response(MOCK_RESPONSE_OK)):
        resultado = fetch_indicadores_economicos()

    texto = format_external_context(resultado)
    assert "FUENTE EXTERNA: mindicador.cl" in texto
    assert "39000.12" in texto
    assert "Unidad de fomento (UF)" in texto


# --------------------------------------------------------------------------- #
# 2) Degradación con gracia ante fallas
# --------------------------------------------------------------------------- #

def test_fetch_indicadores_timeout_degrada_con_gracia() -> None:
    with patch("src.external_data.requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        resultado = fetch_indicadores_economicos()

    assert resultado.disponible is False
    assert resultado.indicadores == {}
    assert "timed out" in (resultado.error or "")


def test_fetch_indicadores_error_http_degrada_con_gracia() -> None:
    with patch("src.external_data.requests.get", return_value=_mock_response({}, status_ok=False)):
        resultado = fetch_indicadores_economicos()

    assert resultado.disponible is False
    assert resultado.error is not None


def test_fetch_indicadores_json_sin_campos_esperados() -> None:
    with patch("src.external_data.requests.get", return_value=_mock_response({"otro_campo": 1})):
        resultado = fetch_indicadores_economicos()

    assert resultado.disponible is False
    assert "esperados" in (resultado.error or "")


def test_format_external_context_no_disponible_no_inventa_datos() -> None:
    resultado = IndicadoresExternos(
        fecha_consulta="2026-09-22T10:00:00", disponible=False, error="timeout"
    )
    texto = format_external_context(resultado)

    assert "FUENTE EXTERNA NO DISPONIBLE" in texto
    assert "no está disponible" in texto
    # No debe contener ningún valor numérico inventado de UF/dólar.
    assert "Unidad de fomento" not in texto


# --------------------------------------------------------------------------- #
# 3) Inyección en el contexto combinado (rag_pipeline.merge_context)
# --------------------------------------------------------------------------- #

def test_merge_context_inyecta_fuente_externa_junto_a_interna() -> None:
    with patch("src.external_data.requests.get", return_value=_mock_response(MOCK_RESPONSE_OK)):
        indicadores = fetch_indicadores_economicos()
    external_block = format_external_context(indicadores)

    internal_block = "[fuente=factura_F2100.txt | chunk_id=0]\nFactura F-2100 | Monto: 30 UF"

    merged = merge_context(internal_block, external_block)

    # Ambas fuentes deben estar presentes y ser distinguibles por su etiqueta.
    assert internal_block in merged
    assert external_block in merged
    assert "[fuente=factura_F2100.txt" in merged
    assert "[FUENTE EXTERNA: mindicador.cl" in merged
    assert "39000.12" in merged


def test_merge_context_funciona_aunque_externa_no_este_disponible() -> None:
    indicadores_caidos = IndicadoresExternos(
        fecha_consulta="2026-09-22T10:00:00", disponible=False, error="timeout"
    )
    external_block = format_external_context(indicadores_caidos)
    internal_block = "[fuente=doc.txt | chunk_id=0]\nContenido interno"

    merged = merge_context(internal_block, external_block)

    assert internal_block in merged
    assert "FUENTE EXTERNA NO DISPONIBLE" in merged
