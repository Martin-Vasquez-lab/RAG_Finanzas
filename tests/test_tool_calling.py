"""Prueba de humo (paso 0 del diseño del agente EP2): confirma que el
modelo de chat activo soporta tool-calling nativo vía `bind_tools` ANTES
de construir el grafo del agente sobre ese mecanismo (IE1/IE2).

Es una prueba `live`: llama a Groq real, no es determinista en su
contenido textual, y se salta automáticamente si no hay GROQ_API_KEY
configurada (ver pytest.ini para el registro del marcador).
"""
from __future__ import annotations

import pytest
from langchain_core.tools import tool

from src.config import get_chat_model, settings


@tool
def consultar_tipo_cambio(moneda: str) -> str:
    """Consulta el valor de una moneda (UF, USD, UTM) para el día de hoy."""
    return f"1 {moneda} = 39000 CLP (simulado para la prueba de humo)"


@tool
def registrar_hallazgo(descripcion: str, nivel_riesgo: str) -> str:
    """Registra un hallazgo de auditoría en el sistema de casos."""
    return f"Hallazgo registrado: {descripcion} (riesgo={nivel_riesgo})"


@pytest.mark.live
@pytest.mark.skipif(
    not settings.groq_api_key,
    reason="Requiere GROQ_API_KEY real (prueba live, no determinista).",
)
def test_groq_oss_120b_soporta_tool_calling_nativo():
    """Dadas dos herramientas, el modelo debe elegir la pertinente al
    contexto de la pregunta (no la irrelevante) y devolver un tool_call
    estructurado, no texto a parsear manualmente."""
    llm = get_chat_model()
    llm_with_tools = llm.bind_tools([consultar_tipo_cambio, registrar_hallazgo])

    response = llm_with_tools.invoke(
        "El contrato CA-078 está en UF. Necesito saber a cuánto equivale "
        "hoy en pesos chilenos antes de poder calcular el monto total."
    )

    assert response.tool_calls, "El modelo no generó ningún tool_call."
    assert response.tool_calls[0]["name"] == "consultar_tipo_cambio"
    assert response.response_metadata.get("finish_reason") == "tool_calls"
