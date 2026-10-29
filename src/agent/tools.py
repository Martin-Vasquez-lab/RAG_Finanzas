"""Herramientas atómicas del agente de auditoría (EP2, IE1/IE2).

Envuelven lo ya implementado en la EP1 (retriever FAISS, LLM-as-judge,
generador de InformeAuditoria, fact-checking, fuente externa mindicador.cl)
SIN modificar `src/rag_pipeline.py`: cada herramienta importa sus piezas
(prompts, parser, funciones de formateo) y las recombina de forma atómica,
para que el LLM -no una cadena fija- decida cuál usar y en qué orden (ver
el grafo del próximo commit).

Tres reglas de diseño, acordadas antes de escribir este archivo:

1. Los chunks NUNCA viajan como texto por los argumentos del LLM: viven en
   `ChunkStore` (ver `chunk_store.py`) y las herramientas reciben/devuelven
   IDs + snippets cortos, nunca el texto completo de un chunk.
2. Ninguna herramienta lanza una excepción hacia el grafo: los errores
   (fuente externa caída, chunk_ids inexistentes, respuesta vacía del LLM
   tras reintentos, salida no parseable) se devuelven como observación
   (`{"error": "..."}"`) para que el LLM pueda razonar sobre ellos y
   decidir un siguiente paso.
3. `registrar_hallazgo` es la única herramienta de ESCRITURA. Nunca
   escribe un hallazgo de riesgo "alto" como aprobado directamente: la
   aprobación (`src/agent/approval.py`) está deliberadamente fuera de la
   lista de herramientas del LLM.

Las descripciones de cada herramienta (abajo, constantes `DESC_*`) son las
que más pesan en qué tan bien el LLM elige la herramienta correcta: cada
una dice explícitamente CUÁNDO usarla y CUÁNDO NO, y en qué se diferencia
de las demás (ver `tests/test_agent_tools.py` para la cobertura de que el
ruteo real elige bien entre herramientas de contenido similar).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agent.case_store import CaseLog, FindingsIndex
from src.agent.chunk_store import ChunkStore
from src.agent.informe_store import InformeStore
from src.agent.llm_utils import invoke_with_retry
from src.config import settings
from src.external_data import IndicadoresExternos, fetch_indicadores_economicos, format_external_context
from src.fact_checking import _parser as fact_check_parser
from src.prompts import AUDIT_PROMPT_TEMPLATE, FACT_CHECK_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE
from src.rag_pipeline import _parse_score, _parser as informe_parser, format_docs, merge_context

# --------------------------------------------------------------------------- #
# Dependencias inyectables (nada de estado global): una instancia por
# ejecución del agente, construida por quien arme el grafo/la demo.
# --------------------------------------------------------------------------- #


@dataclass
class AgentDependencies:
    retriever: Any  # objeto con .invoke(str) -> list[Document]: FAISS real o fake en tests
    llm: Any  # chat model LangChain estándar: ChatGroq real o fake en tests
    chunk_store: ChunkStore
    informe_store: InformeStore
    case_log: CaseLog
    findings_index: FindingsIndex
    judge_min_score: float = field(default_factory=lambda: settings.judge_min_score)
    external_fetch: Callable[[], IndicadoresExternos] = fetch_indicadores_economicos
    max_retries: int = 2
    # Expuesto aparte de max_retries para que los tests puedan poner 0 y no
    # esperar segundos reales al ejercitar el camino de reintento.
    retry_backoff_seconds: float = 1.0


def _safe(label: str, fn: Callable[[], dict]) -> dict:
    """Ninguna herramienta debe lanzar una excepción hacia el grafo del
    agente: cualquier falla no anticipada se convierte en una observación
    de error legible."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - frontera deliberada hacia el grafo
        return {"error": f"{label} falló de forma inesperada: {exc}"}


def _is_empty_response(response: Any) -> bool:
    return not getattr(response, "content", "").strip()


# --------------------------------------------------------------------------- #
# 1) recuperar_documentos — CONSULTA (fuente interna, FAISS)
# --------------------------------------------------------------------------- #

DESC_RECUPERAR_DOCUMENTOS = (
    "Busca fragmentos en los documentos INTERNOS de este caso (facturas, "
    "notas de crédito, informes de riesgo, contratos) usando similitud "
    "semántica. Úsala PRIMERO cuando la pregunta pida un dato concreto de "
    "un documento (monto, RUT, fecha, cláusula, nivel de riesgo de un "
    "documento específico). Devuelve IDs y un resumen corto de cada "
    "fragmento, NUNCA el texto completo. NO la uses para hallazgos de "
    "auditorías anteriores (usa buscar_hallazgos_previos) ni para "
    "UF/USD/UTM (usa consultar_tipo_cambio)."
)


class RecuperarDocumentosInput(BaseModel):
    pregunta: str = Field(
        description="Pregunta o sub-tarea actual, en lenguaje natural, para buscar en los documentos internos del banco."
    )


def _recuperar_documentos_impl(pregunta: str, deps: AgentDependencies) -> dict:
    docs = deps.retriever.invoke(pregunta)
    ids = deps.chunk_store.put_many(docs)
    fragmentos = [
        {
            "chunk_id": cid,
            "fuente": deps.chunk_store.source(cid),
            "snippet": deps.chunk_store.snippet(cid),
        }
        for cid in ids
    ]
    return {"chunk_ids": ids, "fragmentos": fragmentos}


# --------------------------------------------------------------------------- #
# 2) juzgar_relevancia — RAZONAMIENTO (LLM-as-judge)
# --------------------------------------------------------------------------- #

DESC_JUZGAR_RELEVANCIA = (
    "Filtra, de un conjunto de chunk_ids YA recuperados con "
    "recuperar_documentos, cuáles son realmente relevantes para la "
    "pregunta actual (descarta fragmentos que podrían mezclar cifras de "
    "documentos distintos). Requiere haber llamado recuperar_documentos "
    "antes, en este mismo turno, y usar EXACTAMENTE esos chunk_ids. NO "
    "inventes chunk_ids ni la uses sin haber recuperado documentos primero."
)


class JuzgarRelevanciaInput(BaseModel):
    chunk_ids: list[int] = Field(
        description="IDs de chunk devueltos por recuperar_documentos en este mismo turno."
    )
    pregunta: str = Field(
        description="La misma pregunta usada en recuperar_documentos, para evaluar la relevancia de cada chunk."
    )


def _juzgar_relevancia_impl(chunk_ids: list[int], pregunta: str, deps: AgentDependencies) -> dict:
    docs = deps.chunk_store.get_many(chunk_ids)
    if not docs:
        return {
            "error": "Ninguno de los chunk_ids indicados existe en el almacén de esta sesión. "
            "Llama primero a recuperar_documentos."
        }

    judge_chain = JUDGE_PROMPT_TEMPLATE | deps.llm
    puntajes: dict[int, float] = {}
    kept: list[int] = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        response = invoke_with_retry(
            lambda doc=doc: judge_chain.invoke({"question": pregunta, "chunk": doc.page_content}),
            is_empty=_is_empty_response,
            max_retries=deps.max_retries,
            backoff_seconds=deps.retry_backoff_seconds,
        )
        score = _parse_score(getattr(response, "content", str(response)))
        puntajes[cid] = score
        if score >= deps.judge_min_score:
            kept.append(cid)

    if not kept:
        # Igual que en rag_pipeline.AuditRagPipeline._judge_filter: es
        # preferible dejar el top-k original disponible a que
        # generar_informe se quede sin contexto.
        kept = [d.metadata.get("chunk_id") for d in docs]

    return {"chunk_ids_relevantes": kept, "puntajes": puntajes}


# --------------------------------------------------------------------------- #
# 3) consultar_tipo_cambio — CONSULTA (fuente externa, mindicador.cl)
# --------------------------------------------------------------------------- #

DESC_CONSULTAR_TIPO_CAMBIO = (
    "Consulta el valor del día (fuente externa mindicador.cl) de UF, USD "
    "(dólar observado) o UTM en pesos chilenos. Úsala SOLO si el documento "
    "o la pregunta mencionan un monto en UF, USD o UTM que haya que "
    "convertir a CLP. NO la uses si todos los montos relevantes ya están "
    "en pesos chilenos (CLP), y NO la uses para buscar datos dentro de los "
    "documentos del banco."
)

_MONEDA_A_CAMPO: dict[str, str] = {"UF": "uf", "USD": "dolar", "UTM": "utm"}


class ConsultarTipoCambioInput(BaseModel):
    moneda: Literal["UF", "USD", "UTM"] = Field(
        description="Unidad a convertir a pesos chilenos: UF, USD (dólar observado) o UTM."
    )


def _consultar_tipo_cambio_impl(moneda: str, deps: AgentDependencies) -> dict:
    indicadores = deps.external_fetch()
    if not indicadores.disponible:
        return {
            "disponible": False,
            "mensaje": (
                f"mindicador.cl no disponible en este momento ({indicadores.error}). "
                "Continúa sin este dato: si la pregunta requería la conversión, indica "
                "explícitamente en el informe que no está disponible (no lo inventes)."
            ),
        }
    campo = _MONEDA_A_CAMPO[moneda]
    indicador = indicadores.indicadores.get(campo)
    if indicador is None:
        return {"disponible": False, "mensaje": f"La fuente externa no incluyó el campo '{campo}' hoy."}
    return {
        "disponible": True,
        "valor": indicador.valor,
        "unidad": indicador.unidad_medida,
        "fecha": indicador.fecha,
        "bloque_contexto": format_external_context(indicadores),
    }


# --------------------------------------------------------------------------- #
# 4) generar_informe — RAZONAMIENTO (genera el InformeAuditoria final)
# --------------------------------------------------------------------------- #

DESC_GENERAR_INFORME = (
    "Genera el InformeAuditoria final (monto, RUT, nivel de riesgo, "
    "alertas, citas textuales) a partir de chunk_ids ya recuperados "
    "(idealmente ya filtrados con juzgar_relevancia). Si antes llamaste a "
    "consultar_tipo_cambio, pasa su resultado (bloque_contexto) en "
    "contexto_externo. Es la herramienta que produce la RESPUESTA al "
    "usuario: úsala solo cuando ya tengas reunidos los chunk_ids "
    "necesarios. SOLO UNA VEZ por pregunta -un segundo intento en la "
    "misma ejecución se rechaza con un error-; devuelve un informe_id que "
    "debes usar tal cual en verificar_afirmaciones (nunca retipees el "
    "informe)."
)


class GenerarInformeInput(BaseModel):
    chunk_ids: list[int] = Field(
        description="IDs de chunk (idealmente ya filtrados con juzgar_relevancia) a usar como contexto del informe."
    )
    pregunta: str = Field(description="Pregunta original del área de auditoría.")
    contexto_externo: str | None = Field(
        default=None,
        description="Bloque de texto devuelto por consultar_tipo_cambio (campo bloque_contexto), si se usó. Omitir si no aplica.",
    )


def _generar_informe_impl(
    chunk_ids: list[int], pregunta: str, contexto_externo: str | None, deps: AgentDependencies
) -> dict:
    if deps.informe_store.count() > 0:
        return {
            "error": "Ya generaste un informe en esta ejecución (una por pregunta). Si necesitas "
            "corregirlo usa verificar_afirmaciones sobre el informe_id ya obtenido; no generes otro."
        }
    docs = deps.chunk_store.get_many(chunk_ids)
    if not docs:
        return {
            "error": "Ninguno de los chunk_ids indicados existe en el almacén de esta sesión. "
            "Llama primero a recuperar_documentos."
        }
    internal_context = format_docs(docs)
    context_text = merge_context(internal_context, contexto_externo) if contexto_externo else internal_context

    chain = AUDIT_PROMPT_TEMPLATE.partial(format_instructions=informe_parser.get_format_instructions()) | deps.llm
    response = invoke_with_retry(
        lambda: chain.invoke({"context": context_text, "question": pregunta}),
        is_empty=_is_empty_response,
        max_retries=deps.max_retries,
        backoff_seconds=deps.retry_backoff_seconds,
    )
    content = getattr(response, "content", "")
    if not content.strip():
        return {"error": "El modelo devolvió una respuesta vacía tras los reintentos. Puedes reintentar la pregunta."}
    try:
        informe = informe_parser.parse(content)
    except Exception as exc:  # noqa: BLE001 - error de parsing, no de infraestructura
        return {"error": f"No se pudo interpretar la salida del modelo como InformeAuditoria: {exc}"}

    informe_id = deps.informe_store.put(informe)
    return {"informe_id": informe_id, "informe": informe.model_dump(), "chunk_ids_usados": chunk_ids}


# --------------------------------------------------------------------------- #
# 5) verificar_afirmaciones — RAZONAMIENTO (fact-checking independiente)
# --------------------------------------------------------------------------- #

DESC_VERIFICAR_AFIRMACIONES = (
    "Verifica que cada afirmación de un InformeAuditoria YA GENERADO esté "
    "respaldada textualmente por los chunk_ids indicados (detecta "
    "alucinaciones). Recibe el informe_id devuelto por generar_informe -"
    "NUNCA le copies el contenido del informe-. Úsala DESPUÉS de "
    "generar_informe, nunca antes ni en su reemplazo. Especialmente "
    "importante si nivel_riesgo_fraude es 'medio' o 'alto'."
)


class VerificarAfirmacionesInput(BaseModel):
    informe_id: str = Field(description="El informe_id devuelto por generar_informe (no el texto del informe).")
    chunk_ids: list[int] = Field(description="Los mismos chunk_ids usados para generar ese informe.")


def _verificar_afirmaciones_impl(informe_id: str, chunk_ids: list[int], deps: AgentDependencies) -> dict:
    informe = deps.informe_store.get(informe_id)
    if informe is None:
        return {
            "error": "No existe un informe con ese informe_id en esta ejecución. "
            "Llama primero a generar_informe y usa el informe_id que te devuelva."
        }
    docs = deps.chunk_store.get_many(chunk_ids)
    if not docs:
        return {"error": "Ninguno de los chunk_ids indicados existe en el almacén de esta sesión."}
    context_text = format_docs(docs)
    informe_json = informe.model_dump_json()

    chain = (
        FACT_CHECK_PROMPT_TEMPLATE.partial(format_instructions=fact_check_parser.get_format_instructions())
        | deps.llm
    )
    response = invoke_with_retry(
        lambda: chain.invoke({"context": context_text, "answer": informe_json}),
        is_empty=_is_empty_response,
        max_retries=deps.max_retries,
        backoff_seconds=deps.retry_backoff_seconds,
    )
    content = getattr(response, "content", "")
    if not content.strip():
        return {"error": "El modelo devolvió una respuesta vacía tras los reintentos."}
    try:
        reporte = fact_check_parser.parse(content)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"No se pudo interpretar el reporte de verificación: {exc}"}

    return {"afirmaciones": [a.model_dump() for a in reporte.afirmaciones]}


# --------------------------------------------------------------------------- #
# 6) buscar_hallazgos_previos — CONSULTA (memoria de largo plazo)
# --------------------------------------------------------------------------- #

DESC_BUSCAR_HALLAZGOS_PREVIOS = (
    "Busca, por similitud semántica, hallazgos de auditoría YA REGISTRADOS "
    "Y APROBADOS en preguntas o sesiones anteriores de este mismo caso "
    "(memoria de largo plazo del agente). Útil para continuidad en "
    "conversaciones largas (p.ej. 'qué se detectó antes sobre X'). NO "
    "busca dentro de los documentos originales del banco (usa "
    "recuperar_documentos) ni devuelve hallazgos pendientes de aprobación."
)


class BuscarHallazgosPreviosInput(BaseModel):
    pregunta: str = Field(
        description="Pregunta o tema actual, para buscar hallazgos previos relacionados por similitud semántica."
    )


def _buscar_hallazgos_previos_impl(pregunta: str, deps: AgentDependencies) -> dict:
    resultados = deps.findings_index.search(pregunta, k=3)
    return {"hallazgos_previos": resultados}


# --------------------------------------------------------------------------- #
# 7) registrar_hallazgo — ESCRITURA (la única herramienta que escribe)
# --------------------------------------------------------------------------- #

DESC_REGISTRAR_HALLAZGO = (
    "Registra un hallazgo de auditoría (alerta, riesgo, inconsistencia) en "
    "el sistema de casos, para trazabilidad y memoria de largo plazo. "
    "Úsala cuando generar_informe o verificar_afirmaciones hayan detectado "
    "alertas_detectadas no vacías o nivel_riesgo_fraude medio/alto. Si "
    "nivel_riesgo es 'alto', el hallazgo queda pendiente_de_aprobación y "
    "NO se indexa en memoria de largo plazo hasta que un humano lo "
    "apruebe (comportamiento esperado, no un error). No la llames dos "
    "veces para el mismo hallazgo: una descripción+chunk_ids ya "
    "registrados se detecta automáticamente como duplicado."
)


class RegistrarHallazgoInput(BaseModel):
    descripcion: str = Field(
        description="Descripción breve y específica del hallazgo (ej. 'Emisión duplicada del mismo monto en 24 horas en NC-045')."
    )
    nivel_riesgo: Literal["bajo", "medio", "alto"] = Field(description="Nivel de riesgo del hallazgo.")
    chunk_ids_relacionados: list[int] = Field(
        default_factory=list, description="chunk_ids que respaldan este hallazgo, si corresponde."
    )


def _registrar_hallazgo_impl(
    descripcion: str, nivel_riesgo: str, chunk_ids_relacionados: list[int] | None, deps: AgentDependencies
) -> dict:
    resultado = deps.case_log.register(descripcion, nivel_riesgo, chunk_ids_relacionados or [])
    if resultado.estado == "aprobado" and not resultado.duplicado:
        deps.findings_index.add(resultado.id, resultado.descripcion, metadata={"nivel_riesgo": nivel_riesgo})
    return {
        "id": resultado.id,
        "estado": resultado.estado,
        "descripcion": resultado.descripcion,
        "duplicado": resultado.duplicado,
    }


# --------------------------------------------------------------------------- #
# Fábrica: une las implementaciones con `deps` y las expone como StructuredTool
# --------------------------------------------------------------------------- #


def build_agent_tools(deps: AgentDependencies) -> list[StructuredTool]:
    """Construye la lista de herramientas del agente, todas ligadas a la
    misma instancia de `deps` (mismo ChunkStore, mismo LLM, mismo caso).
    Cada función expuesta solo recibe los campos de su `args_schema`: `deps`
    queda cerrado por clausura, nunca visible para el LLM."""

    def recuperar_documentos(pregunta: str) -> dict:
        return _safe("recuperar_documentos", lambda: _recuperar_documentos_impl(pregunta, deps))

    def juzgar_relevancia(chunk_ids: list[int], pregunta: str) -> dict:
        return _safe("juzgar_relevancia", lambda: _juzgar_relevancia_impl(chunk_ids, pregunta, deps))

    def consultar_tipo_cambio(moneda: Literal["UF", "USD", "UTM"]) -> dict:
        return _safe("consultar_tipo_cambio", lambda: _consultar_tipo_cambio_impl(moneda, deps))

    def generar_informe(chunk_ids: list[int], pregunta: str, contexto_externo: str | None = None) -> dict:
        return _safe(
            "generar_informe", lambda: _generar_informe_impl(chunk_ids, pregunta, contexto_externo, deps)
        )

    def verificar_afirmaciones(informe_id: str, chunk_ids: list[int]) -> dict:
        return _safe(
            "verificar_afirmaciones", lambda: _verificar_afirmaciones_impl(informe_id, chunk_ids, deps)
        )

    def buscar_hallazgos_previos(pregunta: str) -> dict:
        return _safe("buscar_hallazgos_previos", lambda: _buscar_hallazgos_previos_impl(pregunta, deps))

    def registrar_hallazgo(
        descripcion: str,
        nivel_riesgo: Literal["bajo", "medio", "alto"],
        chunk_ids_relacionados: list[int] | None = None,
    ) -> dict:
        return _safe(
            "registrar_hallazgo",
            lambda: _registrar_hallazgo_impl(descripcion, nivel_riesgo, chunk_ids_relacionados, deps),
        )

    return [
        StructuredTool.from_function(
            func=recuperar_documentos,
            name="recuperar_documentos",
            description=DESC_RECUPERAR_DOCUMENTOS,
            args_schema=RecuperarDocumentosInput,
        ),
        StructuredTool.from_function(
            func=juzgar_relevancia,
            name="juzgar_relevancia",
            description=DESC_JUZGAR_RELEVANCIA,
            args_schema=JuzgarRelevanciaInput,
        ),
        StructuredTool.from_function(
            func=consultar_tipo_cambio,
            name="consultar_tipo_cambio",
            description=DESC_CONSULTAR_TIPO_CAMBIO,
            args_schema=ConsultarTipoCambioInput,
        ),
        StructuredTool.from_function(
            func=generar_informe,
            name="generar_informe",
            description=DESC_GENERAR_INFORME,
            args_schema=GenerarInformeInput,
        ),
        StructuredTool.from_function(
            func=verificar_afirmaciones,
            name="verificar_afirmaciones",
            description=DESC_VERIFICAR_AFIRMACIONES,
            args_schema=VerificarAfirmacionesInput,
        ),
        StructuredTool.from_function(
            func=buscar_hallazgos_previos,
            name="buscar_hallazgos_previos",
            description=DESC_BUSCAR_HALLAZGOS_PREVIOS,
            args_schema=BuscarHallazgosPreviosInput,
        ),
        StructuredTool.from_function(
            func=registrar_hallazgo,
            name="registrar_hallazgo",
            description=DESC_REGISTRAR_HALLAZGO,
            args_schema=RegistrarHallazgoInput,
        ),
    ]
