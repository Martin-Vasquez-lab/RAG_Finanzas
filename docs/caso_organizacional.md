# Caso organizacional — EP1 ISY0101

## 1. Organización

**Banco Andino S.A.**, institución financiera cuyo **Departamento de
Auditoría Interna y Cumplimiento Tributario** es el usuario del sistema.
El nombre aparece consistentemente en el corpus de documentos de prueba
del proyecto (`data/raw/informe_riesgo_Q2_2025.txt`: "Institución: Banco
Andino S.A."; también como receptor/arrendatario en las facturas, notas de
crédito y contratos de ejemplo), y se mantiene como identidad ficticia
consistente en todo el repositorio, tal como permite el enunciado del EP1
("real o ficticia, con preferencia real").

- **Rubro**: servicios financieros (banca).
- **Tamaño / contexto**: institución con volumen de documentación
  financiera recurrente (facturas electrónicas de proveedores, notas de
  crédito, informes de riesgo trimestrales, contratos de arriendo y otros
  compromisos en UF) que hoy se revisan manualmente.
- **Área usuaria**: Auditoría Interna y Cumplimiento Tributario, responsable
  de detectar irregularidades (duplicidad de montos, fraccionamiento,
  documentos faltantes) y de producir hallazgos trazables ante reguladores
  y auditores externos.

## 2. Problema / desafío

La revisión manual de documentación financiera (facturas, notas de
crédito, informes de riesgo, contratos) es lenta y propensa a que un
analista humano no cruce automáticamente señales de riesgo que aparecen
distribuidas entre varios documentos — por ejemplo, una nota de crédito
emitida tres veces por el mismo monto en una ventana de 24 horas (caso real
del corpus de prueba, `nota_credito_NC045.txt`), que solo es visible si se
comparan varios documentos entre sí.

Además, cualquier solución basada en LLM introduce un riesgo propio: que el
modelo **invente** cifras (montos, RUT) que no están en la documentación
auditada — inaceptable en un contexto donde un dato mal citado tiene
implicancias regulatorias.

## 3. Objetivos del proyecto (medibles)

Los objetivos se definieron para ser verificables contra evidencia real
generada por el propio sistema (`docs/evidencia_pruebas.md`,
`docs/comparacion_memoria.md`, `pytest`), no como aspiraciones genéricas:

1. **Cero alucinaciones de datos financieros en las pruebas realizadas.**
   Medible directamente: en la corrida de evidencia real
   (`docs/evidencia_pruebas.md`), la cadena de fact-checking clasificó
   **0 afirmaciones como `[Contradictorio]`** sobre las respuestas
   evaluadas, y el caso de un documento inexistente (factura "F-9999")
   fue respondido con `nivel_riesgo_fraude: "no_determinable"` y una
   alerta explícita, en vez de un RUT inventado.
2. **Detectar automáticamente duplicidad de montos en ventana de 24 horas**
   sin intervención humana. Medible: en las 3 corridas reales documentadas
   contra el caso NC-045 (emisión del mismo monto 3 veces en 24h), el
   sistema marcó consistentemente `nivel_riesgo_fraude: "alto"` con la
   alerta de emisión múltiple citando el fragmento textual exacto.
3. **Trazabilidad del 100% de las cifras reportadas hasta su fuente.**
   Medible por diseño del esquema `InformeAuditoria`: todo campo
   (`monto_total`, `rut_emisor`, `nivel_riesgo_fraude`) debe venir
   acompañado de una entrada en `citas_textuales` con el fragmento exacto
   del documento (`src/rag_pipeline.py: format_docs`), verificado por 49
   tests automáticos (`pytest`, `test_benchmark.py` con Exact Match/RegEx
   contra `tests/ground_truth.json`).
4. **Resolver preguntas de seguimiento sin degradar precisión.** Medible:
   en la comparación real buffer vs. resumen (`docs/comparacion_memoria.md`),
   una pregunta de seguimiento que dependía de un turno anterior (RUT de
   "esa misma factura", sin nombrarla) se respondió correctamente
   (`76.123.456-7`) en ambas estrategias de memoria, sobre 2 corridas
   completas de 5 turnos cada una.
5. **Costo operativo acotado y medido.** Medible: costo real observado de
   ≈$0,007–0,010 USD por sesión de 5 preguntas de auditoría completas
   (`docs/evidencia_pruebas.md`, `docs/comparacion_memoria.md`), con
   registro automático de tokens/latencia/costo por llamada
   (`AuditCallbackHandler`, `src/metrics.py`).
6. **Combinar fuentes internas y externas para conversión de unidades.**
   Medible: al menos una pregunta real sobre un monto en UF fue resuelta
   citando tanto el documento interno como el valor de la UF del día
   obtenido de la API pública `mindicador.cl` (`fuente_externa_utilizada:
   true`, con el valor de UF usado registrado en `uf_referencia_clp`).

## 4. Datos disponibles

El corpus de prueba (`data/raw/`, 4 documentos sintéticos, texto plano)
representa los tipos de documento que el Departamento de Auditoría procesa
habitualmente:

| Documento | Tipo | Qué demuestra |
|---|---|---|
| `factura_electronica_F1023.txt` | Factura electrónica | Extracción de monto/RUT limpia, caso "sin hallazgos" (riesgo bajo) |
| `nota_credito_NC045.txt` | Nota de crédito | Caso de riesgo alto: emisión triplicada del mismo monto en 24h |
| `informe_riesgo_Q2_2025.txt` | Informe de riesgo / estado de resultados | Síntesis de riesgo entre documentos, cifras de resultado operacional |
| `contrato_arriendo_oficina_UF.txt` | Contrato de arriendo | Monto expresado en UF, requiere la fuente externa para convertir a CLP |

Estos documentos son simulados (síntesis realista, no datos reales de
clientes), tal como habilita explícitamente el enunciado del EP1 ("pueden
ser simulados"). El diseño de `src/loaders.py` (chunking, metadatos
`source`/`page`/`fecha_ingesta`/`tipo_documento`) es agnóstico al origen
real de los documentos: en producción, la misma ingesta aplicaría sobre el
repositorio documental real del banco (facturación electrónica, sistema de
gestión de contratos, informes de riesgo internos), sin cambios de código.

Adicionalmente, el sistema consulta en vivo la fuente externa pública
**mindicador.cl** (UF, dólar, UTM del día; `src/external_data.py`) para
enriquecer respuestas que requieren conversión de unidades monetarias.

## 5. Restricciones operacionales

- **Privacidad de datos financieros**: los documentos de auditoría
  contienen RUT y montos que, en un despliegue real, corresponderían a
  información sensible de clientes/proveedores del banco. El diseño no
  envía datos a ningún proveedor de embeddings externo por defecto — los
  embeddings corren localmente (`sentence-transformers`,
  `EMBEDDING_PROVIDER=local`), y solo el texto estrictamente necesario para
  responder una pregunta puntual se envía al proveedor de chat (Groq) vía
  la API, nunca el corpus completo de una sola vez.
- **Trazabilidad exigida para auditores**: toda respuesta debe poder
  rastrearse hasta el documento y fragmento exacto (`citas_textuales`,
  metadatos `source`/`chunk_id`) — un auditor humano debe poder verificar
  cada cifra sin confiar ciegamente en el modelo.
- **No alucinación de cifras**: restricción no negociable dado el uso en
  un contexto de cumplimiento tributario; implementada como guardrail
  explícito en el prompt (`NO_DATA_RULE`, `src/prompts.py`) y verificada
  con un caso de prueba real (documento inexistente, ver objetivo 1).
- **Límites de presupuesto/tokens**: el proyecto documenta y mide el costo
  real de cada corrida (`AuditCallbackHandler`), y la elección de proveedor
  (Groq, tier gratuito) y de estrategia de memoria (`buffer` por defecto)
  está fundamentada explícitamente en el costo medido, no solo en
  capacidad técnica — ver `docs/comparacion_memoria.md` y la sección
  "Sobre los proveedores de LLM/embeddings" del `README.md`.
- **Disponibilidad de fuentes externas no garantizada**: `mindicador.cl` es
  un servicio de terceros fuera del control del banco; el sistema debe
  seguir operando (con la fuente interna) si esa API falla — verificado con
  un caso real de timeout durante las pruebas (`docs/evidencia_pruebas.md`).

## 6. Motivación para agentes de IA, LLMs y RAG

Un sistema de reglas fijas (p. ej. validaciones SQL sobre campos
estructurados) no puede leer documentos en lenguaje natural con formato
heterogéneo (facturas, notas de crédito, contratos) ni razonar sobre
señales de riesgo que requieren correlacionar información entre varios
documentos (p. ej. "el mismo monto aparece 3 veces en 24 horas en
documentos distintos"). Un LLM puede hacerlo, pero sin control de contexto
alucina cifras — de ahí la arquitectura RAG (recuperación restringida al
corpus real) más los guardrails, el juez de relevancia y la verificación
fáctica posterior implementados en este proyecto, en vez de usar el LLM
"a secas" con conocimiento general.

## 7. Referencias y anexos

- Diagrama de arquitectura: [`docs/architecture.md`](architecture.md) /
  [`docs/architecture.png`](architecture.png).
- Boceto conceptual de la interfaz de consulta: [`docs/boceto_interfaz.svg`](boceto_interfaz.svg) / [`docs/boceto_interfaz.png`](boceto_interfaz.png).
- Evidencia de pruebas de software: [`docs/evidencia_pruebas.md`](evidencia_pruebas.md).
- Comparación empírica de memoria conversacional: [`docs/comparacion_memoria.md`](comparacion_memoria.md).
- Corpus de prueba: [`data/raw/`](../data/raw/).
- Código fuente: [`src/`](../src/), suite de tests: [`tests/`](../tests/).
