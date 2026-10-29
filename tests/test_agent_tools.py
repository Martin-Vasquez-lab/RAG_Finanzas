"""Tests deterministas de las 7 herramientas atómicas del agente (EP2,
IE1/IE2). Sin API: LLM, retriever y fuente externa son fakes (ver
tests/agent_test_helpers.py). Cubren, por herramienta, el caso feliz, al
menos un caso de error devuelto como observación (nunca una excepción
propagada al grafo), y las reglas de escritura/aprobación acordadas.
"""

from __future__ import annotations

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.tools import AgentDependencies, build_agent_tools
from src.external_data import IndicadorEconomico, IndicadoresExternos
from tests.agent_test_helpers import (
    FACT_CHECK_JSON_OK,
    INFORME_JSON_OK,
    FakeChatModel,
    FakeEmbeddings,
    FakeRetriever,
    make_chunk,
)


def _tool(tools, name):
    return {t.name: t for t in tools}[name]


def _deps(tmp_path, llm_responses, retriever_docs=None, external_fetch=None):
    return AgentDependencies(
        retriever=FakeRetriever(retriever_docs or []),
        llm=FakeChatModel(llm_responses),
        chunk_store=ChunkStore(),
        case_log=CaseLog(tmp_path / "hallazgos.jsonl"),
        findings_index=FindingsIndex(FakeEmbeddings()),
        external_fetch=external_fetch
        or (lambda: IndicadoresExternos(fecha_consulta="2026-10-28", disponible=False, error="no usado en este test")),
        max_retries=1,
        retry_backoff_seconds=0.0,
    )


# --------------------------------------------------------------------------- #
# 0) La lista de herramientas expuestas al LLM es exactamente la acordada
# --------------------------------------------------------------------------- #


def test_build_agent_tools_expone_exactamente_las_7_herramientas_acordadas(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    nombres = {t.name for t in build_agent_tools(deps)}

    assert nombres == {
        "recuperar_documentos",
        "juzgar_relevancia",
        "consultar_tipo_cambio",
        "generar_informe",
        "verificar_afirmaciones",
        "buscar_hallazgos_previos",
        "registrar_hallazgo",
    }
    # La aprobación (src/agent/approval.py) NUNCA debe quedar expuesta al LLM.
    assert "aprobar_hallazgo" not in nombres


# --------------------------------------------------------------------------- #
# 1) recuperar_documentos
# --------------------------------------------------------------------------- #


def test_recuperar_documentos_guarda_en_chunk_store_y_devuelve_snippets(tmp_path):
    docs = [make_chunk(0, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP")]
    deps = _deps(tmp_path, llm_responses=[], retriever_docs=docs)
    tool = _tool(build_agent_tools(deps), "recuperar_documentos")

    resultado = tool.invoke({"pregunta": "monto de la factura F-1023"})

    assert resultado["chunk_ids"] == [0]
    assert resultado["fragmentos"][0]["fuente"] == "factura_F1023.txt"
    assert "Monto total" in resultado["fragmentos"][0]["snippet"]
    # El texto completo NUNCA viaja como argumento de otra herramienta:
    # solo vive en el ChunkStore, accesible por chunk_id.
    assert deps.chunk_store.get(0).page_content == docs[0].page_content


def test_recuperar_documentos_error_del_retriever_se_devuelve_como_observacion(tmp_path):
    class RetrieverQueFalla:
        def invoke(self, query, config=None):
            raise RuntimeError("FAISS no disponible")

    deps = _deps(tmp_path, llm_responses=[])
    deps.retriever = RetrieverQueFalla()
    tool = _tool(build_agent_tools(deps), "recuperar_documentos")

    resultado = tool.invoke({"pregunta": "cualquier cosa"})

    assert "error" in resultado


# --------------------------------------------------------------------------- #
# 2) juzgar_relevancia
# --------------------------------------------------------------------------- #


def test_juzgar_relevancia_filtra_por_puntaje(tmp_path):
    deps = _deps(tmp_path, llm_responses=["9", "2"])
    deps.chunk_store.put_many(
        [make_chunk(0, "f.txt", "chunk relevante"), make_chunk(1, "f.txt", "chunk irrelevante")]
    )
    tool = _tool(build_agent_tools(deps), "juzgar_relevancia")

    resultado = tool.invoke({"chunk_ids": [0, 1], "pregunta": "¿cuál es el monto?"})

    assert resultado["chunk_ids_relevantes"] == [0]
    assert resultado["puntajes"] == {0: 9.0, 1: 2.0}


def test_juzgar_relevancia_sin_chunk_ids_validos_devuelve_error(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "juzgar_relevancia")

    resultado = tool.invoke({"chunk_ids": [999], "pregunta": "algo"})

    assert "error" in resultado


def test_juzgar_relevancia_si_todos_bajo_el_umbral_conserva_el_original(tmp_path):
    deps = _deps(tmp_path, llm_responses=["1", "2"])
    deps.chunk_store.put_many([make_chunk(0, "f.txt", "a"), make_chunk(1, "f.txt", "b")])
    tool = _tool(build_agent_tools(deps), "juzgar_relevancia")

    resultado = tool.invoke({"chunk_ids": [0, 1], "pregunta": "algo"})

    assert set(resultado["chunk_ids_relevantes"]) == {0, 1}


def test_juzgar_relevancia_reintenta_ante_respuesta_vacia(tmp_path):
    deps = _deps(tmp_path, llm_responses=["", "7"])
    deps.chunk_store.put_many([make_chunk(0, "f.txt", "chunk")])
    tool = _tool(build_agent_tools(deps), "juzgar_relevancia")

    resultado = tool.invoke({"chunk_ids": [0], "pregunta": "algo"})

    assert resultado["puntajes"][0] == 7.0
    assert len(deps.llm.calls) == 2


# --------------------------------------------------------------------------- #
# 3) consultar_tipo_cambio
# --------------------------------------------------------------------------- #


def test_consultar_tipo_cambio_disponible_devuelve_valor_y_bloque(tmp_path):
    indicadores = IndicadoresExternos(
        fecha_consulta="2026-10-28",
        disponible=True,
        indicadores={
            "uf": IndicadorEconomico(
                codigo="uf", nombre="Unidad de fomento (UF)", valor=39000.12, unidad_medida="Pesos", fecha="2026-10-28"
            )
        },
    )
    deps = _deps(tmp_path, llm_responses=[], external_fetch=lambda: indicadores)
    tool = _tool(build_agent_tools(deps), "consultar_tipo_cambio")

    resultado = tool.invoke({"moneda": "UF"})

    assert resultado["disponible"] is True
    assert resultado["valor"] == 39000.12
    assert "mindicador.cl" in resultado["bloque_contexto"]


def test_consultar_tipo_cambio_api_caida_degrada_con_gracia_no_lanza(tmp_path):
    indisponible = IndicadoresExternos(fecha_consulta="2026-10-28", disponible=False, error="timeout")
    deps = _deps(tmp_path, llm_responses=[], external_fetch=lambda: indisponible)
    tool = _tool(build_agent_tools(deps), "consultar_tipo_cambio")

    resultado = tool.invoke({"moneda": "USD"})

    assert resultado["disponible"] is False
    assert "no disponible" in resultado["mensaje"]


# --------------------------------------------------------------------------- #
# 4) generar_informe
# --------------------------------------------------------------------------- #


def test_generar_informe_camino_feliz_devuelve_informe_parseado(tmp_path):
    deps = _deps(tmp_path, llm_responses=[INFORME_JSON_OK])
    deps.chunk_store.put_many(
        [make_chunk(0, "factura_F1023.txt", "Factura F-1023 | Monto total: $1.245.000 CLP | RUT: 76.123.456-7")]
    )
    tool = _tool(build_agent_tools(deps), "generar_informe")

    resultado = tool.invoke({"chunk_ids": [0], "pregunta": "¿monto y RUT de F-1023?"})

    assert resultado["informe"]["monto_total"] == 1245000
    assert resultado["informe"]["nivel_riesgo_fraude"] == "bajo"


def test_generar_informe_sin_chunk_ids_validos_devuelve_error(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "generar_informe")

    resultado = tool.invoke({"chunk_ids": [999], "pregunta": "algo"})

    assert "error" in resultado


def test_generar_informe_respuesta_vacia_tras_reintentos_es_error_no_excepcion(tmp_path):
    deps = _deps(tmp_path, llm_responses=["", ""])
    deps.chunk_store.put_many([make_chunk(0, "f.txt", "texto")])
    tool = _tool(build_agent_tools(deps), "generar_informe")

    resultado = tool.invoke({"chunk_ids": [0], "pregunta": "algo"})

    assert "error" in resultado


def test_generar_informe_salida_no_parseable_es_error_no_excepcion(tmp_path):
    deps = _deps(tmp_path, llm_responses=["esto no es JSON"])
    deps.chunk_store.put_many([make_chunk(0, "f.txt", "texto")])
    tool = _tool(build_agent_tools(deps), "generar_informe")

    resultado = tool.invoke({"chunk_ids": [0], "pregunta": "algo"})

    assert "error" in resultado


# --------------------------------------------------------------------------- #
# 5) verificar_afirmaciones
# --------------------------------------------------------------------------- #


def test_verificar_afirmaciones_camino_feliz(tmp_path):
    deps = _deps(tmp_path, llm_responses=[FACT_CHECK_JSON_OK])
    deps.chunk_store.put_many([make_chunk(0, "f.txt", "Monto total: $1.245.000 CLP")])
    tool = _tool(build_agent_tools(deps), "verificar_afirmaciones")

    resultado = tool.invoke({"informe_json": INFORME_JSON_OK, "chunk_ids": [0]})

    assert resultado["afirmaciones"][0]["veredicto"] == "Soportado"


def test_verificar_afirmaciones_sin_chunk_ids_validos_devuelve_error(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "verificar_afirmaciones")

    resultado = tool.invoke({"informe_json": "{}", "chunk_ids": [999]})

    assert "error" in resultado


# --------------------------------------------------------------------------- #
# 6) buscar_hallazgos_previos
# --------------------------------------------------------------------------- #


def test_buscar_hallazgos_previos_devuelve_resultados_del_indice(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    deps.findings_index.add(
        "id-1", "Emisión duplicada del mismo monto en NC-045", metadata={"nivel_riesgo": "alto"}
    )
    tool = _tool(build_agent_tools(deps), "buscar_hallazgos_previos")

    resultado = tool.invoke({"pregunta": "NC-045 emisión duplicada"})

    assert any(h["finding_id"] == "id-1" for h in resultado["hallazgos_previos"])


def test_buscar_hallazgos_previos_vacio_no_es_error(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "buscar_hallazgos_previos")

    resultado = tool.invoke({"pregunta": "algo que no existe"})

    assert resultado["hallazgos_previos"] == []


# --------------------------------------------------------------------------- #
# 7) registrar_hallazgo — reglas acordadas del punto 5
# --------------------------------------------------------------------------- #


def test_registrar_hallazgo_riesgo_bajo_queda_aprobado_y_se_indexa(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "registrar_hallazgo")

    resultado = tool.invoke(
        {"descripcion": "Monto dentro de rango esperado", "nivel_riesgo": "bajo", "chunk_ids_relacionados": [0]}
    )

    assert resultado["estado"] == "aprobado"
    assert deps.findings_index.search("monto dentro de rango esperado") != []


def test_registrar_hallazgo_riesgo_alto_nunca_escribe_aprobado_directo(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "registrar_hallazgo")

    resultado = tool.invoke(
        {
            "descripcion": "Emisión duplicada del mismo monto en 24 horas",
            "nivel_riesgo": "alto",
            "chunk_ids_relacionados": [3, 4],
        }
    )

    assert resultado["estado"] == "pendiente_de_aprobación"
    # No debe quedar indexado en memoria de largo plazo hasta la aprobación.
    assert deps.findings_index.search("emisión duplicada del mismo monto") == []


def test_registrar_hallazgo_duplicado_no_crea_segunda_entrada(tmp_path):
    deps = _deps(tmp_path, llm_responses=[])
    tool = _tool(build_agent_tools(deps), "registrar_hallazgo")
    primero = tool.invoke(
        {"descripcion": "Hallazgo repetido", "nivel_riesgo": "alto", "chunk_ids_relacionados": [1]}
    )

    segundo = tool.invoke(
        {"descripcion": "Hallazgo repetido", "nivel_riesgo": "alto", "chunk_ids_relacionados": [1]}
    )

    assert segundo["duplicado"] is True
    assert segundo["id"] == primero["id"]
