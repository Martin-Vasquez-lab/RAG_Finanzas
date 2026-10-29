# Evidencia de pruebas de software realizadas

Este documento registra la evidencia de la última corrida real del pipeline
completo contra la API de Groq (`openai/gpt-oss-120b`) y la fuente externa
`mindicador.cl`, ejecutada el **2026-09-22** con `tests/manual_benchmark_groq.py`
(script manual, no forma parte de la suite automática de `pytest` porque
gasta cuota real de API — ver docstring del script).

## 1. Suite automática (determinista, sin costo de API)

```
pytest -v
```

**Resultado: 38 passed, 1 skipped, 0 failed.** Cubre chunking, prompts,
las 4 métricas RAG, extracción RUT/monto por RegEx, y la fuente externa
(`tests/test_external_data.py`, 8 tests con `requests.get` mockeado —
éxito, timeout, error HTTP, JSON malformado, y la inyección del bloque
externo en el contexto vía `merge_context()`).

## 2. Corrida real contra Groq — 5 preguntas

Corpus: `data/raw/` (4 documentos: factura F-1023, nota de crédito NC-045,
informe de riesgo Q2 2025, contrato de arriendo CA-078 en UF).

| # | Pregunta | Resultado del `InformeAuditoria` | Fuente externa |
|---|---|---|---|
| 1 | Monto y RUT emisor de F-1023 | `monto_total=1245000 CLP`, `rut_emisor=76.123.456-7`, `riesgo=bajo` | no aplicaba |
| 2 | Fraude en NC-045 | `monto_total=5000000 CLP`, `riesgo=alto`, 2 alertas (emisión múltiple/duplicidad) | no aplicaba |
| 3 | Resultado operacional Q2 2025 | `monto_total=368000000 CLP`, `riesgo=bajo` | no aplicaba |
| 4 | RUT receptor de **F-9999 (no existe)** | `monto_total=null`, `riesgo=no_determinable`, alerta "Documento F-9999 no encontrado" | no aplicaba |
| 5 | Canon de arriendo CA-078 (45 UF) → CLP | `alertas=["Valor de la UF no disponible para conversión a CLP"]`, **`fuente_externa_utilizada=false`** | **no disponible** (timeout real de red, ver §3) |

### Caso destacado 1 — Guardrail anti-alucinación (pregunta 4)

El documento "F-9999" **no existe** en el corpus. El modelo, en vez de
inventar un RUT, respondió:

```json
{
  "monto_total": null, "divisa": null, "rut_emisor": null,
  "nivel_riesgo_fraude": "no_determinable",
  "alertas_detectadas": ["Documento F-9999 no encontrado en el contexto auditado"],
  "citas_textuales": [],
  "fuente_externa_utilizada": false, "uf_referencia_clp": null
}
```

Esto valida en producción (no en un mock) la regla anti-alucinación de
`src/prompts.py` (`NO_DATA_RULE`) y es evidencia directa para IE6
("Determina la coherencia entre los datos y las respuestas del modelo").

### Caso destacado 2 — Degradación con gracia de la fuente externa (pregunta 5, intento 1)

Durante esta corrida, la llamada a `mindicador.cl` tuvo un **timeout real
de red** (`Read timed out. (read timeout=5.0)`), no simulado. El pipeline
**no se cayó**: `fetch_indicadores_economicos()` capturó el error y marcó
`disponible=False`; el LLM recibió el bloque
`[FUENTE EXTERNA NO DISPONIBLE: mindicador.cl | ... | motivo=...]` y, en
vez de inventar un valor de UF, respondió:

```json
{
  "citas_textuales": [
    "Canon de arriendo mensual: 45 UF",
    "[FUENTE EXTERNA NO DISPONIBLE: mindicador.cl | intento=2026-09-22T22:00:22 | motivo=HTTPSConnectionPool(host='mindicador.cl', port=443): Read timed out. (read timeout=5.0)]"
  ],
  "alertas_detectadas": ["Valor de la UF no disponible para conversión a CLP"],
  "fuente_externa_utilizada": false
}
```

Es la evidencia más fuerte del requisito de degradación con gracia: no fue
una falla simulada en un test, fue una caída real de la fuente externa en
plena corrida, y el sistema completo (interno + externo) siguió operando.

### Caso destacado 3 — Uso real y exitoso de la fuente externa (pregunta 5, reintento)

Al confirmar que `mindicador.cl` volvió a responder (1.27 s, UF del día =
$40.991,75 CLP), se repitió la pregunta 5:

```json
{
  "monto_total": 1844628.75,
  "divisa": "CLP",
  "nivel_riesgo_fraude": "bajo",
  "citas_textuales": [
    "Canon de arriendo mensual: 45 UF",
    "Unidad de fomento (UF) (uf): 40991.75 Pesos"
  ],
  "fuente_externa_utilizada": true,
  "uf_referencia_clp": 40991.75
}
```

`45 UF × $40.991,75 = $1.844.628,75 CLP` — cálculo correcto, citando
explícitamente tanto el chunk interno (contrato CA-078) como el bloque de
la fuente externa, y marcando `fuente_externa_utilizada: true`. Esta es la
evidencia concreta de que el pipeline **combina fuentes de datos internas y
externas** (requisito IL1.2 / IE3).

## 3. Fact-checking (pregunta 5, corrida original)

```
[Soportado]     El canon de arriendo mensual es de 45 UF.
[Soportado]     El valor de la UF no está disponible para conversión a CLP.
[No Mencionado] El campo monto_total es nulo.
[No Mencionado] El campo divisa es nulo.
[No Mencionado] El campo rut_emisor es nulo.
[No Mencionado] El nivel de riesgo de fraude es bajo.
[Soportado]     No se utilizó fuente externa para la información.
[No Mencionado] El campo uf_referencia_clp es nulo.
```

Sin ninguna afirmación `[Contradictorio]`: cero alucinaciones detectadas
por la cadena de fact-checking independiente.

## 4. Consumo total de API (Groq) — corrida de 5 preguntas

| Métrica | Valor |
|---|---|
| Preguntas procesadas | 5 |
| Llamadas totales a Groq | 25 (5 por pregunta: hasta 4 del juez + 1 de generación) |
| Tokens de entrada | 22.134 |
| Tokens de salida | 4.968 |
| Tokens totales | 27.102 |
| **Costo estimado total** | **$0,007046 USD** |
| Latencia acumulada | 86,19 s |

El reintento adicional de la pregunta 5 (con `mindicador.cl` disponible)
sumó 5 llamadas más: 4.254 tokens de entrada, 1.068 de salida, costo
estimado $0,001439 USD, latencia promedio 0,83 s.

**Costo total de esta sesión de pruebas: ≈ $0,0085 USD (~30 llamadas a
Groq).** Confirma que el desarrollo y testing iterativo de este proyecto es
económicamente viable incluso con un tier gratuito.

## Cómo reproducir esta evidencia

```bash
python -m src.loaders                    # reconstruye el índice FAISS (embeddings locales)
python tests/manual_benchmark_groq.py    # corre las 5 preguntas contra Groq real
```

Requiere `GROQ_API_KEY` configurada en `.env`. La fuente externa
(`mindicador.cl`) no requiere ninguna clave.

---

# EP2 — Agente funcional: evidencia de corridas `live`

A diferencia de la sección anterior (EP1, una corrida puntual), el agente
de EP2 se valida con **escenarios repetidos** (el enrutamiento de un LLM
real no es determinista de una corrida a otra) contra Groq real, marcados
`@pytest.mark.live` en `tests/test_agent_*_live.py`. Cada corrida deja un
registro de tokens/costo/latencia MEDIDO (`src/agent/token_tracker.py`,
misma tabla de precios que `AuditCallbackHandler`), no estimado.

**Metodología de ejecución** (acordada para no repetir el problema de
cuota): por tandas, empezando con 1 corrida de cada escenario crítico
antes de lanzar las demás; detención si el consumo acumulado del día
supera ~70% del cupo diario (140.000 de 200.000 tokens/día en la cuenta
usada); se registran TODAS las corridas, incluidas las fallidas o
interrumpidas, con fecha y hora.

## Corridas fallidas/interrumpidas durante el desarrollo (2026-10-28)

Antes de llegar a una corrida limpia del escenario de memoria de largo
plazo, varios intentos fallaron — cada uno encontró un problema real
(nunca una herramienta oculta ni un parche silencioso):

| # | Escenario | Qué falló | Causa raíz | Corrección |
|---|---|---|---|---|
| 1 | Grafo (consulta simple) | `groq.BadRequestError`: el modelo generó un `tool_call` a `judzar_relevancia` (nombre inexistente, typo de `juzgar_relevancia`) | Error de GENERACIÓN del proveedor, no de ejecución de herramienta — no lo cubría `tools.py` | `llm_node` reintenta acotado con una corrección explícita (commit `707bc4d`) |
| 2 | Memoria largo plazo, sesión A | El modelo registró el hallazgo de NC-045 (fraude de manual) con `nivel_riesgo="medio"`, no `"alto"` → no quedó pendiente | `AGENT_SYSTEM_PROMPT` no tenía criterio de calibración de riesgo para `registrar_hallazgo` (a diferencia del prompt interno de `generar_informe`, que sí tiene few-shot) | Se agregó el criterio explícito al prompt del agente (commit `e43526f`) |
| 3 | Memoria largo plazo, sesión B | `UnicodeEncodeError` al imprimir la respuesta del modelo (contenía `‑` U+2011) | Consola Windows (cp1252), no un bug del agente | `sys.stdout.reconfigure(encoding="utf-8")` en los archivos `_live` |
| 4 | Memoria largo plazo, sesión B | Aserción del test exigía el UUID crudo del hallazgo en la respuesta en prosa al auditor | Aserción poco realista (mala UX esperar un UUID en texto natural); además `FindingsIndex.search()` no devolvía `nivel_riesgo` | Aserción relajada a verificar el ID en la OBSERVACIÓN de la herramienta; se agregó `nivel_riesgo` a `search()` |
| 5 | Memoria largo plazo, sesión A | `secuencia_a = []`: el turno terminó vacío en ~50s sin usar ninguna herramienta | `llm_node` y `tools._safe()` atrapaban **cualquier** excepción, incluyendo `RateLimitError`, convirtiéndola en "no se pudo completar" en vez de dejarla propagar para el reintento con espera | Ambos re-lanzan `RateLimitError` explícitamente (commit `e43526f`) |
| 6 | Memoria largo plazo, sesión A (tras la corrección #5) | `groq.RateLimitError`: "tokens per day (TPD): Limit 200000, Used 199338" durante el nodo `replanner` | Cupo DIARIO de la cuenta casi agotado por la depuración acumulada del día, no un bug | Se detuvo la ejecución live por ese día (sin corrida limpia de extremo a extremo el 2026-10-28) |

## Tanda 1 — 2026-10-29, ~16:30 hora local (completada, 3/3 escenarios)

Primera corrida de cada escenario crítico (memoria de largo plazo, ruteo
consulta simple, ruteo factura UF), con `src/agent/token_tracker.py`
conectado. Las 3 pasaron.

| Escenario | Secuencia de herramientas | Tokens in/out | Costo real | Llamadas | Latencia |
|---|---|---|---|---|---|
| Memoria largo plazo — sesión A | `recuperar_documentos, juzgar_relevancia, registrar_hallazgo, recuperar_documentos, buscar_hallazgos_previos, generar_informe, verificar_afirmaciones` | 26.044 / 4.579 | $0,007341 | 13 | 155,7 s |
| Memoria largo plazo — sesión B (relacionada) | `recuperar_documentos, buscar_hallazgos_previos` | 6.599 / 1.307 | $0,001970 | 4 | 54,2 s |
| Memoria largo plazo — control negativo | `recuperar_documentos, juzgar_relevancia, generar_informe, verificar_afirmaciones, generar_informe, verificar_afirmaciones` | 21.926 / 4.445 | $0,006623 | 12 | 83,0 s |
| **Memoria largo plazo — total corrida 1** | | **54.569 / 10.331** | **$0,015934** | **29** | **292,9 s** |
| Consulta simple (ruteo) — corrida 1 | `recuperar_documentos, juzgar_relevancia, generar_informe, verificar_afirmaciones` | 15.601 / 2.836 | $0,004467 | 9 | 111,6 s |
| Factura en UF (ruteo) — corrida 1 | `recuperar_documentos, juzgar_relevancia, consultar_tipo_cambio, generar_informe, verificar_afirmaciones` | 18.594 / 3.720 | $0,005579 | 10 | 120,5 s |
| **Total tanda 1** | | **~88.764 / ~16.887 (105.651 tokens)** | **$0,025980** | **48** | **~525 s** |

**Lectura de la sesión B relacionada (memoria de largo plazo):** el agente
respondió primero `"No dispongo de esta información en los documentos
auditados"` (correcto: `recuperar_documentos` no encontró nada relevante
en esa sesión nueva) y a continuación citó, desde `buscar_hallazgos_previos`,
la descripción y el nivel de riesgo REAL del hallazgo registrado en la
sesión A — sin inventar uno, sin reutilizar el historial de corto plazo
(sesión/`thread_id` distinto). El `finding_id` recuperado coincidió con el
registrado y aprobado en la sesión A (verificado por aserción, ver
`tests/test_agent_long_term_memory_live.py`).

**Observación de costo real vs. estimado:** la estimación previa a esta
tanda (basada en tamaño de prompts + secuencias observadas el día
anterior) calculó ~36.900 tokens para memoria de largo plazo; la corrida
real costó 64.900 (+76%), porque el modelo tomó pasos adicionales no
anticipados (una segunda `recuperar_documentos` en sesión A, dos intentos
de `generar_informe` en el control negativo antes de conformarse con la
respuesta). Con este dato real, completar las 3 corridas de memoria de
largo plazo por sí solas costaría ~195.000 tokens — casi el cupo diario
completo — por lo que el plan se mantiene en 3 corridas por escenario,
pero repartido en varios días (no 2), respetando el tope de ~70%/día.

## Cambio de criterio tras la Tanda 1: amplitud antes que repetición

IE7 pide demostrar decisiones adaptativas con **múltiples ejemplos
distintos**, no la misma repetida 3 veces. Las "3 corridas por escenario
crítico" fueron un estándar propio, no un requisito de la pauta — y con
el costo real de memoria de largo plazo (~65k/corrida), repetirla 3 veces
gasta casi un día completo de cupo para añadir poco sobre lo ya
demostrado (1 corrida real + cobertura determinista completa). Nuevo
criterio: priorizar 1 corrida de cada escenario que todavía no tiene
ninguna corrida real; las repeticiones (2ª y 3ª) quedan opcionales al
final, solo para los escenarios que se vayan a mostrar en la demo.

### Documento inexistente — 2026-10-29, ~17:10 hora local (1 corrida, completada)

Mismo caso que el few-shot #4 de `src/prompts.py` (ya validado
determinísticamente en EP1): el contexto solo contiene la factura
F-1023; se pregunta por la F-9999, que no existe.

- **Secuencia:** `recuperar_documentos, recuperar_documentos` (el agente
  reintentó la búsqueda interna una vez; nunca llegó a llamar
  `generar_informe` — respondió directo tras concluir que el documento no
  estaba).
- **Respuesta final:** *"No dispongo de esta información en los
  documentos auditados."* — no inventó el RUT de F-1023 (76.123.456-7)
  como si fuera el de F-9999 (verificado por aserción).
- **Costo real:** 6.586 in / 631 out = 7.217 tokens, $0,001461 USD,
  4 llamadas, 3,6 s de latencia — bastante más barato que la estimación
  (~15.000-20.000), porque el agente nunca llegó a invocar
  `generar_informe` (el prompt interno más caro, ~2.230 tokens de
  sistema) ni `juzgar_relevancia`.

**Primer desglose real por tipo de llamada** (con `TokenTracker.summary_by_label()`,
confirma la hipótesis de que `llm_node` domina el consumo):

| Tipo de llamada | Llamadas | Tokens in/out | Costo |
|---|---|---|---|
| `llm_node` (decisión del agente) | 3 | 6.000 / 225 | $0,001069 |
| `planner` | 1 | 586 / 406 | $0,000392 |

`llm_node` concentra el 91% de los tokens de ENTRADA (reenvía el prompt
del agente + 7 esquemas de herramientas en cada vuelta), pero su salida
por llamada es pequeña (~75 tokens, solo decide la siguiente acción).
`planner` tiene menos tokens de entrada pero una salida relativamente
grande (406, el plan estructurado completo) para una sola llamada. Dato
a confirmar con escenarios más largos (memoria de largo plazo, donde
`llm_node` se llama 7-8 veces) antes de sacar conclusiones sobre dónde
optimizar — **no se tocaron prompts ni comportamiento**, esto es solo
observación.

## Estado y corridas pendientes

**Cupo usado hoy (2026-10-29): 112.868 de 200.000 tokens (~56%).** Se
detiene la ejecución live por hoy aquí, dentro del margen de seguridad
(~70%).

Faltan, para completar el paso 7, 1 corrida cada uno de: replanificación,
riesgo alto (aislado), fuente externa caída (aislado, no solo como parte
de otro escenario), evidencia contradictoria, e inyección de
instrucciones — planificadas para el día siguiente. Las repeticiones
(2ª y 3ª corrida) de memoria de largo plazo, consulta simple y factura UF
quedan opcionales, solo para los escenarios que se vayan a mostrar en la
demo, y solo si queda cupo al final.
