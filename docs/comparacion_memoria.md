# Comparación empírica: BufferMemory vs SummaryMemory (IE4 / IE8)

Corrida real contra Groq (`openai/gpt-oss-120b`), fecha **2026-09-24**, con
`tests/manual_benchmark_memoria.py`. Misma conversación de **5 turnos**
ejecutada dos veces, una por estrategia, cada una en una sesión nueva
(`session_id` propio, sin compartir memoria entre estrategias).

## La conversación usada

| Turno | Pregunta | Depende de un turno anterior |
|---|---|---|
| 1 | ¿Cuál es el monto total y el RUT del emisor en la factura F-1023? | No |
| 2 | ¿Existen señales de riesgo de fraude en la nota de crédito NC-045? | No |
| 3 | ¿Cuál es el resultado operacional del Q2 2025? | No |
| 4 | ¿Y cuál era el RUT del emisor de esa misma factura que mencionamos al principio? | **Sí — turno 1** (no nombra el documento, solo "esa misma factura") |
| 5 | ¿Cuál es el canon de arriendo del contrato CA-078 en UF, y a cuánto equivale en CLP con el valor de la UF de hoy? | No (pero ejercita la fuente externa) |

## Resultados

| Métrica | `buffer` | `summary` | Diferencia |
|---|---:|---:|---:|
| Llamadas totales a la API | 25 | 30 | summary usa **+5** (1 extra por turno: la actualización del resumen) |
| Tokens de entrada | 30.511 | 30.955 | summary +444 |
| Tokens de salida | 5.609 | 7.417 | summary +1.808 |
| **Tokens totales** | **36.120** | **38.372** | summary **+2.252 (+6,2%)** |
| **Costo estimado total** | **$0,008783 USD** | **$0,010206 USD** | summary **+16,2%** |
| RUT turno 4 (esperado `76.123.456-7`) | `76.123.456-7` ✅ | `76.123.456-7` ✅ | **Ambas correctas, sin degradación** |
| Tamaño del contexto de memoria al final (caracteres) | 1.169 | 725 | summary sí comprime el contexto en un **38%** |

### Detalle por turno (tokens de entrada, para ver la tendencia de crecimiento)

| Turno | `buffer` tokens_in | `summary` tokens_in |
|---|---:|---:|
| 1 | 5.143 | 5.410 |
| 2 | 5.941 | 6.157 |
| 3 | 6.197 | 6.325 |
| 4 | 6.393 | 6.458 |
| 5 | 6.837 | 6.605 |

## Lectura de los resultados

1. **Ninguna estrategia perdió precisión en esta corrida.** El turno 4 (la
   pregunta de seguimiento que depende del turno 1, sin nombrar el
   documento) devolvió el RUT correcto (`76.123.456-7`) tanto con buffer
   como con summary. La hipótesis inicial —que el resumen podría
   distorsionar cifras exactas— **no se materializó a esta escala** (5
   turnos, resúmenes cortos), probablemente porque el prompt de
   actualización de `SummaryMemory` instruye explícitamente a preservar
   cifras exactas (ver `src/memory.py`), y el LLM lo respetó en esta
   corrida. Esto no es una garantía matemática para conversaciones más
   largas — solo evidencia de que, a 5 turnos, el resumen no falló.

2. **`summary` salió MÁS caro que `buffer`, no más barato**, contrario a la
   intuición de "comprimir ahorra tokens". La razón: `summary` necesita
   **una llamada LLM extra por turno** (la actualización del resumen), y
   esa llamada completa (system prompt + resumen previo + turno nuevo +
   resumen generado) cuesta más tokens de los que ahorra en las llamadas
   subsiguientes al juez/generador — porque, a esta escala, el bloque de
   contexto RAG (documentos + fuente externa, ~4-5 KB) que se reenvía en
   **cada turno sin importar la estrategia de memoria** domina el tamaño
   del prompt. El historial (buffer o resumen) es una fracción pequeña de
   ese total, así que comprimirlo apenas mueve la aguja del costo por
   turno, mientras que la llamada extra de resumen sí se paga completa.

3. **`summary` sí cumple su promesa de compresión del CONTEXTO** (725 vs
   1.169 caracteres al final, -38%), y esa ventaja crecería con
   conversaciones más largas: el contexto de `buffer` crece linealmente
   sin límite, mientras que el de `summary` se mantiene acotado. El punto
   de cruce (donde `summary` empieza a ser más barato en tokens totales)
   no se alcanzó en 5 turnos — probablemente requeriría sesiones bastante
   más largas (decenas de turnos) para que el ahorro de contexto supere el
   costo fijo de la llamada extra por turno.

4. **`summary` agrega una fuente de fallo adicional**: una llamada LLM más
   por turno es una llamada más que puede toparse con rate limits, timeouts
   o errores del proveedor (de hecho, la primera corrida de este mismo
   benchmark falló con `429 rate_limit_exceeded` en el turno 4 de
   `summary`, y hubo que reintentar con más pausa — ver commit history).
   `buffer` no tiene ese punto de falla adicional.

## Conclusión: `buffer` queda como estrategia por defecto

`MEMORY_STRATEGY=buffer` es el default (`src/config.py`) por tres razones
respaldadas por esta corrida, no solo por intuición:

- **Más barato** en esta escala de conversación (5 turnos): -16,2% de
  costo, -5 llamadas a la API.
- **Igual de preciso**: cero degradación observada en el dato de
  seguimiento (RUT), y cero riesgo *estructural* de que un resumen
  distorsione una cifra (al no resumir nada).
- **Más simple y con menos puntos de falla**: sin llamada LLM extra por
  turno, sin dependencia adicional del proveedor de chat para mantener el
  historial.

`summary` sigue disponible (`MEMORY_STRATEGY=summary`) y sería la elección
correcta para sesiones de auditoría **mucho más largas** (decenas de
turnos), donde el crecimiento sin límite del buffer eventualmente se
vuelve el problema dominante — pero para el uso típico de este proyecto
(sesiones de auditoría puntuales, pocas preguntas de seguimiento), el
buffer gana en el balance costo/precisión/simplicidad medido acá.

## Cómo reproducir esta comparación

```bash
python tests/manual_benchmark_memoria.py                    # ambas estrategias
python tests/manual_benchmark_memoria.py --strategy buffer  # solo una
python tests/manual_benchmark_memoria.py --strategy summary
```

Requiere `GROQ_API_KEY` configurada en `.env`. El script reintenta
automáticamente ante `429 rate_limit_exceeded` del tier gratuito de Groq
(hasta 3 intentos, con espera entre cada uno).
