"""Plan priorizado del agente (EP2, IE1/IE5).

El agente genera un plan ANTES de actuar (nodo `planner` del grafo) y lo
REPLANIFICA -incrementando `version`- cuando cambian las condiciones a
mitad de tarea. Cada replanificación queda registrada como un mensaje
visible en la conversación (ver `graph.py`), no solo en un campo interno,
para que sea auditable y para que realmente pese en la siguiente decisión
del LLM (no sea decorativo).

Disparador CONCRETO de replanificación (acordado antes de implementar):
un hallazgo de riesgo "alto" registrado por `registrar_hallazgo`
(`nivel_riesgo="alto"` -> queda `pendiente_de_aprobación`, ver
`case_store.py`). Es un evento observable EN CÓDIGO -el grafo lo detecta
inspeccionando el resultado de la herramienta, no depende de que el LLM
decida replanificar por su cuenta- y empíricamente es justo la situación
donde planificar de nuevo importa: un hallazgo grave cambia qué es
prioritario hacer a continuación (p.ej. revisar hallazgos previos
relacionados antes de cerrar el informe).

La versión del plan la gestiona el GRAFO, no el LLM: el LLM solo propone
`pasos`; `PlanAuditoria.version` se calcula en `graph.py` a partir del
plan anterior, para no depender de que el modelo cuente versiones bien.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PasoPlan(BaseModel):
    orden: int = Field(description="Posición del paso en el plan (1 = primero).")
    prioridad: Literal["alta", "media", "baja"]
    accion: str = Field(
        description="Nombre de la herramienta que corresponde a este paso (p.ej. 'recuperar_documentos')."
    )
    razon: str = Field(description="Por qué este paso es necesario para responder la pregunta.")


class PlanPasos(BaseModel):
    """Lo que el LLM genera: solo los pasos. El número de versión NO es
    parte de lo que se le pide al modelo (ver `graph.py`)."""

    pasos: list[PasoPlan] = Field(default_factory=list)


class PlanAuditoria(BaseModel):
    """Lo que vive en el estado del grafo: los pasos + la versión,
    gestionada por el grafo (se incrementa en cada replanificación)."""

    pasos: list[PasoPlan] = Field(default_factory=list)
    version: int = 1

    def render(self, es_replanificacion: bool = False) -> str:
        """Texto que se inyecta en la conversación para que el plan
        realmente pese en la siguiente decisión del LLM (no quede solo en
        el estado interno del grafo)."""
        etiqueta = "PLAN ACTUALIZADO" if es_replanificacion else "PLAN"
        lineas = [f"[{etiqueta} v{self.version}]"]
        for paso in sorted(self.pasos, key=lambda p: p.orden):
            lineas.append(f"{paso.orden}. (prioridad {paso.prioridad}) {paso.accion} -> {paso.razon}")
        if not self.pasos:
            lineas.append("(sin pasos propuestos; procede con tu propio criterio)")
        lineas.append(f"[/{etiqueta}]")
        return "\n".join(lineas)
