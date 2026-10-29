"""Tests LIVE del grafo del agente (EP2, IE1/IE7): llaman a Groq real.

Qué prueba este archivo (distinto de `test_agent_graph.py`): que el
MODELO REAL enruta distinto según el CONTENIDO de la pregunta -algo que
ningún test con `ScriptedToolCallingLLM` puede demostrar, porque ahí la
secuencia la fija el guion del test, no el razonamiento del modelo-.

Cada escenario corre 3 veces (el enrutamiento de un LLM real no es
determinista de una corrida a otra) y registra la secuencia de
herramientas de cada corrida con `print`, para inspección manual además
de las aserciones automáticas (ver salida de `pytest -s -m live`).
"""

from __future__ import annotations

import time

import pytest
from groq import RateLimitError

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.graph import build_agent_graph, extract_tool_call_sequence, run_agent_turn
from src.agent.chunk_store import ChunkStore
from src.agent.informe_store import InformeStore
from src.agent.tools import AgentDependencies
from src.config import get_chat_model, get_embeddings, settings
from src.external_data import IndicadorEconomico, IndicadoresExternos
from tests.agent_test_helpers import make_chunk

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not settings.groq_api_key, reason="Requiere GROQ_API_KEY real (prueba live, no determinista)."
    ),
]

N_CORRIDAS = 3

_UF_DISPONIBLE = IndicadoresExternos(
    fecha_consulta="2026-10-28",
    disponible=True,
    indicadores={
        "uf": IndicadorEconomico(
            codigo="uf", nombre="Unidad de fomento (UF)", valor=39000.12, unidad_medida="Pesos", fecha="2026-10-28"
        )
    },
)


class _FakeRetrieverForLive:
    """Retriever fijo: siempre devuelve el/los mismo(s) chunk(s). Aísla el
    test de si el índice FAISS real está construido en disco -lo que se
    prueba aquí es el enrutamiento del LLM, no la ingesta (ya cubierta en
    tests/test_loaders.py)."""

    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query, config=None):
        return self._docs


def _run_with_rate_limit_backoff(graph, deps, pregunta, thread_id, max_attempts=4, backoff_seconds=65.0):
    """La cuenta de Groq usada en este proyecto tiene un límite de 8000
    tokens/minuto; un solo turno del grafo (decisión + juez por chunk +
    generar_informe, cada uno reenviando historial y esquemas de
    herramientas) puede agotarlo. Reintenta ante 429 esperando a que se
    reabra la ventana del minuto, en vez de fallar la corrida entera."""
    last_exc: RateLimitError | None = None
    for attempt in range(max_attempts):
        try:
            return run_agent_turn(graph, deps, pregunta, thread_id=thread_id)
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                print(
                    f"  [rate limit] intento {attempt + 1}/{max_attempts} para thread_id={thread_id}; "
                    f"esperando {backoff_seconds:.0f}s a que se reabra la ventana..."
                )
                time.sleep(backoff_seconds)
    raise last_exc


def _deps_live(tmp_path, docs, external_fetch=None):
    return AgentDependencies(
        retriever=_FakeRetrieverForLive(docs),
        llm=get_chat_model(),
        chunk_store=ChunkStore(),
        informe_store=InformeStore(),
        case_log=CaseLog(tmp_path / "hallazgos.jsonl"),
        findings_index=FindingsIndex(get_embeddings()),
        external_fetch=external_fetch
        or (lambda: IndicadoresExternos(fecha_consulta="2026-10-28", disponible=False, error="no usado en este test")),
    )


def test_live_consulta_simple_en_clp_no_llama_tipo_cambio_ni_registra_hallazgo(tmp_path):
    docs = [
        make_chunk(
            0,
            "factura_F1023.txt",
            "Factura F-1023 | Emisor: Comercial Rios Ltda. | RUT: 76.123.456-7 | Monto total: $1.245.000 CLP",
        )
    ]
    pregunta = "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?"

    secuencias = []
    for i in range(N_CORRIDAS):
        deps = _deps_live(tmp_path / f"simple-{i}", docs)
        resultado = _run_with_rate_limit_backoff(
            build_agent_graph(deps), deps, pregunta, thread_id=f"live-simple-{i}"
        )
        secuencia = extract_tool_call_sequence(resultado["messages"])
        secuencias.append(secuencia)
        print(f"\n[consulta simple | corrida {i + 1}/{N_CORRIDAS}] secuencia = {secuencia}")

    for secuencia in secuencias:
        assert "consultar_tipo_cambio" not in secuencia, secuencias
        assert "registrar_hallazgo" not in secuencia, secuencias
        assert secuencia, "El agente no llamó a ninguna herramienta."
        assert secuencia[0] == "recuperar_documentos", secuencias


def test_live_factura_en_uf_si_llama_tipo_cambio(tmp_path):
    docs = [
        make_chunk(0, "contrato_arriendo_oficina_UF.txt", "Contrato de arriendo CA-078 | Canon mensual: 30 UF")
    ]
    pregunta = (
        "¿Cuál es el canon de arriendo del contrato CA-078 en UF, y a cuánto "
        "equivale en CLP con el valor de la UF de hoy?"
    )

    secuencias = []
    for i in range(N_CORRIDAS):
        deps = _deps_live(tmp_path / f"uf-{i}", docs, external_fetch=lambda: _UF_DISPONIBLE)
        resultado = _run_with_rate_limit_backoff(
            build_agent_graph(deps), deps, pregunta, thread_id=f"live-uf-{i}"
        )
        secuencia = extract_tool_call_sequence(resultado["messages"])
        secuencias.append(secuencia)
        print(f"\n[factura en UF | corrida {i + 1}/{N_CORRIDAS}] secuencia = {secuencia}")

    for secuencia in secuencias:
        assert "consultar_tipo_cambio" in secuencia, secuencias
        assert secuencia[0] == "recuperar_documentos", secuencias
