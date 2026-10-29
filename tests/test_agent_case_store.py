"""Tests deterministas de CaseLog, FindingsIndex y aprobar_hallazgo
(EP2, IE1 "escritura" / IE3 / IE4). Sin API: FindingsIndex usa
FakeEmbeddings, nunca el modelo local real de sentence-transformers.
"""

from __future__ import annotations

import pytest

from src.agent.approval import aprobar_hallazgo
from src.agent.case_store import CaseLog, FindingsIndex
from tests.agent_test_helpers import FakeEmbeddings

# --------------------------------------------------------------------------- #
# CaseLog: registro, deduplicación, estados pendiente/aprobado
# --------------------------------------------------------------------------- #


def test_register_bajo_riesgo_queda_aprobado_de_inmediato(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")

    resultado = log.register("Monto dentro de rango esperado", "bajo", [0])

    assert resultado.estado == "aprobado"
    assert resultado.duplicado is False


def test_register_medio_riesgo_queda_aprobado_de_inmediato(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")

    resultado = log.register("Inconsistencia menor de fecha", "medio", [])

    assert resultado.estado == "aprobado"


def test_register_riesgo_alto_nunca_queda_aprobado_directo(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")

    resultado = log.register("Emisión duplicada del mismo monto en 24 horas", "alto", [3, 4])

    assert resultado.estado == "pendiente_de_aprobación"


def test_register_misma_descripcion_y_chunks_se_detecta_como_duplicado(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")
    primero = log.register("Emisión duplicada del mismo monto en 24 horas", "alto", [3, 4])

    segundo = log.register("  EMISIÓN duplicada del mismo monto en 24 horas  ", "alto", [4, 3])

    assert segundo.duplicado is True
    assert segundo.id == primero.id


def test_register_descripcion_distinta_no_es_duplicado(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")
    log.register("Hallazgo A", "bajo", [0])

    segundo = log.register("Hallazgo B", "bajo", [0])

    assert segundo.duplicado is False


def test_mark_approved_actualiza_estado_y_persiste(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")
    resultado = log.register("Riesgo alto pendiente", "alto", [1])

    entry = log.mark_approved(resultado.id)

    assert entry["estado"] == "aprobado"
    assert log.get(resultado.id)["estado"] == "aprobado"


def test_mark_approved_id_inexistente_lanza_key_error(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")
    with pytest.raises(KeyError):
        log.mark_approved("no-existe")


def test_case_log_persiste_entre_instancias_sobre_el_mismo_archivo(tmp_path):
    path = tmp_path / "hallazgos.jsonl"
    CaseLog(path).register("Hallazgo persistente", "bajo", [])

    segunda_instancia = CaseLog(path)
    assert segunda_instancia.find_by_dedup_key("hallazgo persistente|[]") is not None


# --------------------------------------------------------------------------- #
# FindingsIndex: memoria de largo plazo, solo hallazgos aprobados
# --------------------------------------------------------------------------- #


def test_findings_index_empty_search_returns_empty_list():
    index = FindingsIndex(FakeEmbeddings())
    assert index.search("cualquier cosa") == []


def test_findings_index_search_finds_added_finding():
    index = FindingsIndex(FakeEmbeddings())
    index.add(
        "id-1",
        "Emisión duplicada del mismo monto en factura F-1023",
        metadata={"nivel_riesgo": "alto"},
    )

    resultados = index.search("factura F-1023 emisión duplicada", k=3)

    assert any(r["finding_id"] == "id-1" for r in resultados)


# --------------------------------------------------------------------------- #
# aprobar_hallazgo: el gate humano, deliberadamente FUERA de las
# herramientas del LLM (ver también tests/test_agent_tools.py)
# --------------------------------------------------------------------------- #


def test_aprobar_hallazgo_indexa_en_memoria_de_largo_plazo(tmp_path):
    log = CaseLog(tmp_path / "hallazgos.jsonl")
    index = FindingsIndex(FakeEmbeddings())
    pendiente = log.register("Emisión duplicada del mismo monto en 24 horas", "alto", [3, 4])

    assert index.search("emisión duplicada") == []  # aún no aprobado: no debe aparecer

    aprobar_hallazgo(pendiente.id, log, index)

    assert log.get(pendiente.id)["estado"] == "aprobado"
    assert any(
        r["finding_id"] == pendiente.id for r in index.search("emisión duplicada del mismo monto")
    )
