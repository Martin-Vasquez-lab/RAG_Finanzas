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
