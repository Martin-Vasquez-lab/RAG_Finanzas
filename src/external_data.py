"""
Fuente de datos EXTERNA para el pipeline RAG (requisito IL1.2 / IE3: los
flujos RAG deben combinar fuentes de datos internas y externas).

La fuente interna es el índice FAISS sobre `data/raw/` (facturas, notas de
crédito, informes de riesgo). Este módulo agrega una fuente externa REAL:
la API pública **mindicador.cl** (gratuita, sin API key), que publica
indicadores económicos oficiales de Chile del día — UF, dólar observado,
UTM, entre otros. Son datos directamente relevantes para auditoría
financiera: conversión de montos expresados en UF a CLP, o de operaciones
en USD al tipo de cambio del día.

Diseño defensivo: una API externa no crítica (no es la fuente de verdad de
los documentos que se están auditando, solo un dato de apoyo) NUNCA debe
tumbar una consulta de auditoría. Si mindicador.cl está caída, responde
lento, o cambia su formato, `fetch_indicadores_economicos()` captura el
error y retorna un resultado con `disponible=False`, y el pipeline sigue
funcionando solo con la fuente interna (degradación con gracia).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

MINDICADOR_URL: str = "https://mindicador.cl/api"
DEFAULT_TIMEOUT_SECONDS: float = 5.0

# Indicadores relevantes para auditoría financiera: UF y UTM (para montos
# expresados en esas unidades, comunes en contratos/boletas honorarios
# chilenos) y dólar observado (para operaciones en moneda extranjera).
CAMPOS_RELEVANTES: tuple[str, ...] = ("uf", "dolar", "utm")


@dataclass(frozen=True)
class IndicadorEconomico:
    """Un indicador económico individual (ej. UF, dólar) del día."""

    codigo: str
    nombre: str
    valor: float
    unidad_medida: str
    fecha: str


@dataclass(frozen=True)
class IndicadoresExternos:
    """Resultado completo de la consulta a la fuente externa.

    `disponible=False` indica que la fuente externa no pudo consultarse
    (timeout, caída del servicio, respuesta malformada); en ese caso
    `indicadores` queda vacío y `error` describe la causa.
    """

    fecha_consulta: str
    indicadores: dict[str, IndicadorEconomico] = field(default_factory=dict)
    disponible: bool = False
    error: str | None = None


def fetch_indicadores_economicos(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> IndicadoresExternos:
    """Consulta mindicador.cl y retorna los indicadores relevantes para auditoría.

    Nunca lanza una excepción hacia quien la llama: cualquier falla de red,
    timeout, código de error HTTP o respuesta con formato inesperado se
    captura aquí y se traduce en `IndicadoresExternos(disponible=False, ...)`,
    para que `rag_pipeline.py` pueda seguir operando solo con la fuente
    interna (FAISS) sin que la consulta de auditoría se caiga por un
    problema de un servicio externo no crítico.
    """
    ahora = datetime.now().isoformat(timespec="seconds")

    try:
        response = requests.get(MINDICADOR_URL, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("Fuente externa mindicador.cl no disponible: %s", exc)
        return IndicadoresExternos(fecha_consulta=ahora, disponible=False, error=str(exc))

    indicadores: dict[str, IndicadorEconomico] = {}
    for campo in CAMPOS_RELEVANTES:
        item = data.get(campo)
        if not item:
            continue
        try:
            indicadores[campo] = IndicadorEconomico(
                codigo=item["codigo"],
                nombre=item["nombre"],
                valor=float(item["valor"]),
                unidad_medida=item.get("unidad_medida", "Pesos"),
                fecha=item.get("fecha", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Campo '%s' malformado en respuesta de mindicador.cl: %s", campo, exc)

    if not indicadores:
        return IndicadoresExternos(
            fecha_consulta=ahora,
            disponible=False,
            error="Respuesta de mindicador.cl sin los campos esperados (uf/dolar/utm).",
        )

    return IndicadoresExternos(
        fecha_consulta=data.get("fecha", ahora),
        indicadores=indicadores,
        disponible=True,
        error=None,
    )


def format_external_context(indicadores: IndicadoresExternos) -> str:
    """Formatea los indicadores como bloque de contexto citable para el prompt.

    Se etiqueta explícitamente como **FUENTE EXTERNA** (a diferencia de los
    chunks de FAISS, etiquetados `[fuente=archivo | chunk_id=N]` por
    `rag_pipeline.format_docs`), para que tanto el LLM como un auditor
    humano puedan distinguir la procedencia de cada dato citado en
    `citas_textuales`.
    """
    if not indicadores.disponible:
        return (
            f"[FUENTE EXTERNA NO DISPONIBLE: mindicador.cl | intento={indicadores.fecha_consulta} "
            f"| motivo={indicadores.error}]\n"
            "No se pudo obtener el tipo de cambio/UF del día. Si la pregunta "
            "requiere una conversión UF<->CLP o de tipo de cambio, indica que "
            "ese dato no está disponible en este momento (no lo inventes)."
        )

    lineas = [f"[FUENTE EXTERNA: mindicador.cl | fecha={indicadores.fecha_consulta}]"]
    for indicador in indicadores.indicadores.values():
        lineas.append(f"  {indicador.nombre} ({indicador.codigo}): {indicador.valor} {indicador.unidad_medida}")
    return "\n".join(lineas)
