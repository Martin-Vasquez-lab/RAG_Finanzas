"""Aprobación de hallazgos pendientes (EP2, gate humano).

Deliberadamente SEPARADA de `src/agent/tools.py`: esta función NO está en
la lista de herramientas que el LLM puede invocar, y `auto_approve` nunca
se lee de un argumento que el modelo produzca. Se llama desde fuera del
agente: por un humano (CLI/notebook) o, si `settings.auto_approve_hallazgos`
está activo, por el orquestador del grafo (próximo commit) al cierre del
turno. Esto evita que el agente se autoapruebe un hallazgo de riesgo alto.
"""

from __future__ import annotations

from src.agent.case_store import CaseLog, FindingsIndex


def aprobar_hallazgo(finding_id: str, case_log: CaseLog, findings_index: FindingsIndex) -> dict:
    """Marca `finding_id` como aprobado en el log y lo indexa en la
    memoria de largo plazo. Lanza `KeyError` si el ID no existe (a
    diferencia de las herramientas del LLM, esta función SÍ puede lanzar
    excepciones: no forma parte del loop del agente)."""
    entry = case_log.mark_approved(finding_id)
    findings_index.add(entry["id"], entry["descripcion"], metadata={"nivel_riesgo": entry["nivel_riesgo"]})
    return entry
