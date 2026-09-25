# RAG_Finanzas — Agente RAG de Auditoría Financiera (EP1 - ISY0101)

Proyecto académico de la Evaluación Parcial N°1 de **Ingeniería de Soluciones
con IA (ISY0101)**: un agente basado en LLM + RAG para apoyar al Departamento
de Auditoría Interna y Cumplimiento Tributario de una institución financiera
en la revisión de facturas electrónicas, notas de crédito, estados de
resultados, balances trimestrales e informes de riesgo.

## Problema que resuelve

- **Alucinaciones en montos/RUT**: guardrails explícitos + esquema Pydantic
  estricto + verificación fáctica posterior a la generación.
- **Ruido en la recuperación RAG**: un paso de LLM-as-a-judge filtra chunks
  irrelevantes antes de que lleguen al generador.
- **Falta de trazabilidad**: cada chunk lleva metadatos (`source`, `page`,
  `fecha_ingesta`, `tipo_documento`, `chunk_id`) y cada respuesta debe citar
  el fragmento textual exacto que la respalda.
- **Latencia y costo de tokens**: un callback (`AuditCallbackHandler`)
  registra latencia, tokens y costo estimado de cada llamada al LLM.
- **Fuentes de datos internas y externas** (`src/external_data.py`): además
  del índice FAISS sobre `data/raw/` (fuente interna), el pipeline consulta
  en vivo la API pública **mindicador.cl** (UF, dólar, UTM del día) como
  fuente externa, para resolver preguntas que requieren convertir montos en
  UF/USD a CLP. Si la fuente externa falla (timeout, caída), el pipeline
  degrada con gracia y sigue respondiendo solo con la fuente interna — ver
  evidencia de un caso real de esto en
  [`docs/evidencia_pruebas.md`](docs/evidencia_pruebas.md).
- **Preguntas de seguimiento sin perder precisión** (`src/memory.py`): el
  pipeline mantiene memoria conversacional por sesión (`session_id`), para
  resolver referencias como "esa misma factura" sin que el guardrail
  anti-alucinación se relaje — ver
  [Memoria conversacional](#memoria-conversacional-ie4) más abajo.

Ver [`docs/architecture.md`](docs/architecture.md) (con imagen renderizada
en [`docs/architecture.png`](docs/architecture.png)) para el diagrama de
arquitectura completo y la justificación de decisiones, y
[`docs/caso_organizacional.md`](docs/caso_organizacional.md) para la
descripción formal de la organización, objetivos medibles, datos y
restricciones del caso.

## Estructura del repositorio

```
README.md
requirements.txt
.env.example
conftest.py
data/raw/                    # Documentos fuente (4 ejemplos sintéticos: factura,
                              # nota de crédito, informe de riesgo, contrato en UF)
data/processed/               # Artefactos intermedios (vacío, generado en ejecución)
vectorstore/                  # Índice FAISS persistido (generado, no versionado)
notebooks/                     # Notebook de demo end-to-end
src/
  config.py                    # Conexión a LLM (Groq / Google / GitHub) y embeddings (local / Google / GitHub)
  loaders.py                    # Ingesta, chunking, construcción del índice FAISS (fuente INTERNA)
  external_data.py               # Fuente EXTERNA: API mindicador.cl (UF/dólar/UTM), degrada con gracia
  memory.py                       # Memoria conversacional (IE4): BufferMemory / SummaryMemory
  prompts.py                       # Plantillas: rol experto, few-shot, CoT, guardrails XML
  rag_pipeline.py                   # Ensamblado LCEL: retriever | judge | fuente externa | memoria | prompt | LLM | parser
  metrics.py                         # Callback de costo/latencia + 4 métricas RAG + esquemas Pydantic
  fact_checking.py                    # Cadena secundaria de verificación fáctica
tests/
  test_*.py                            # Suite automática determinista (pytest, sin costo de API)
  manual_benchmark_groq.py              # Benchmark MANUAL contra Groq real (evidencia de pruebas)
  manual_benchmark_memoria.py            # Benchmark MANUAL comparativo buffer vs summary
docs/
  caso_organizacional.md                 # Organización, objetivos medibles, datos, restricciones (IE1)
  architecture.md / .mmd / .png           # Diagrama de arquitectura (fuente Mermaid + imagen renderizada, IE7)
  boceto_interfaz.svg / .png               # Boceto conceptual de la interfaz de consulta del auditor
  evidencia_pruebas.md                      # Evidencia de la última corrida real (consumo de API, casos destacados)
  comparacion_memoria.md                     # Comparación empírica BufferMemory vs SummaryMemory (IE4/IE8)
```

## Instalación

Requiere **Python 3.11+**.

```bash
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Variables de entorno

Copia `.env.example` a `.env` y completa tu clave de Groq:

```bash
cp .env.example .env
```

| Variable | Requerida | Descripción |
|---|---|---|
| `GROQ_API_KEY` | Sí (proveedor de chat activo por defecto: `groq`) | Clave de [Groq Console](https://console.groq.com/keys). |
| `LLM_PROVIDER` | No (default `groq`) | `groq`, `google` o `github` — proveedor de **chat**, ver abajo. |
| `GROQ_CHAT_MODEL` | No (default `openai/gpt-oss-120b`) | Modelo de generación servido por Groq. |
| `EMBEDDING_PROVIDER` | No (default `local`) | `local`, `google` o `github` — proveedor de **embeddings**, independiente del de chat. |
| `LOCAL_EMBEDDING_MODEL` | No (default `sentence-transformers/all-MiniLM-L6-v2`) | Modelo de embeddings local (CPU, sin API). |
| `GOOGLE_API_KEY` / `GOOGLE_CHAT_MODEL` / `GOOGLE_EMBEDDING_MODEL` | Solo si usas `google` en `LLM_PROVIDER` o `EMBEDDING_PROVIDER` | Ver justificación abajo. |
| `TEMPERATURE` | No (default `0.0`) | Determinismo exigido para un dominio de auditoría. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | No (default `1200`/`200`) | Ver justificación en `src/loaders.py`. |
| `RETRIEVER_K` | No (default `8`) | Top-k de chunks recuperados antes del filtro LLM-judge. |
| `JUDGE_MIN_SCORE` | No (default `6.0`) | Umbral (0-10) para que un chunk pase el filtro del juez. |
| `MEMORY_STRATEGY` | No (default `buffer`) | `buffer` o `summary` — estrategia de memoria conversacional, ver [Memoria conversacional](#memoria-conversacional-ie4). |

**Nunca subas tu `.env` real al repositorio** (`.gitignore` ya lo excluye).

### Sobre los proveedores de LLM/embeddings

El proveedor de **chat** y el de **embeddings** se configuran por separado
(`LLM_PROVIDER` / `EMBEDDING_PROVIDER`), porque no todos los proveedores
ofrecen ambas capacidades:

- **Chat activo por defecto: Groq** (`LLM_PROVIDER=groq`). Tres razones
  concretas para esta elección:
  1. **Costo**: la corrida real de pruebas documentada en
     [`docs/evidencia_pruebas.md`](docs/evidencia_pruebas.md) costó
     ≈$0,0085 USD por ~30 llamadas (25 preguntas de auditoría completas +
     reintentos) — economía suficiente para iterar sin preocuparse por
     cuota durante el desarrollo y las pruebas de este proyecto.
  2. **Sin bloqueo de facturación**: el proyecto de Google Cloud asociado a
     la cuenta usada originalmente quedó bloqueado con
     `403 PERMISSION_DENIED: Your project has been denied access`
     independientemente del modelo invocado (ver nota de vigencia abajo);
     Groq no exige tarjeta de crédito ni facturación habilitada para su
     tier gratuito.
  3. **Compatibilidad con el patrón OpenAI del propio curso**: el endpoint
     de Groq es compatible con la API de OpenAI (mismo formato de
     `chat.completions`), el mismo patrón que ya usa GitHub Models en los
     notebooks de la asignatura — cambiar entre ambos es trivial y no
     introduce un SDK nuevo que aprender.
  El modelo, `openai/gpt-oss-120b`, es además un modelo "razonador" de
  pesos abiertos servido por Groq — encaja con el diseño Chain-of-Thought
  del prompt de auditoría.
- **Embeddings activos por defecto: locales** (`EMBEDDING_PROVIDER=local`),
  vía `sentence-transformers` (`all-MiniLM-L6-v2`) corriendo en CPU, **no
  Gemini**. Dos razones: (a) **Groq no ofrece API de embeddings**, así que
  de todas formas se necesitaba un segundo proveedor solo para esa etapa; y
  (b) al ejecutar embeddings localmente, esa etapa del pipeline no depende
  de ningún proveedor externo — sin costo, sin llamadas de red, sin cuotas,
  y sin el riesgo de bloqueo de cuenta que sí afectó a Google (punto 2
  arriba).
- **Google AI Studio/Gemini** (el proveedor que exige explícitamente el
  enunciado del EP1, con `text-embedding-004` + Gemini) **se mantiene
  disponible e íntegro** en `src/config.py` para uso futuro: basta con
  `LLM_PROVIDER=google` y/o `EMBEDDING_PROVIDER=google` en `.env`.
- **GitHub Models** (patrón usado en los notebooks de clase) también sigue
  disponible como alternativa para chat y embeddings.

Toda la lógica de conexión vive aislada en `src/config.py`
(`get_chat_model()` / `get_embeddings()`), de modo que cambiar de proveedor
es solo una variable de entorno, nunca una reescritura de código.

> **Nota de vigencia (importante para la entrega):** durante el desarrollo,
> varios nombres de modelo que en algún momento parecían vigentes dejaron
> de estarlo: en Google, `text-embedding-004`, `gemini-1.5-flash` y luego
> `gemini-2.5-flash`; en Groq, `llama-3.3-70b-versatile`. En ambos casos se
> verificó el catálogo real invocando la API directamente antes de fijar un
> nombre en el código (`client.models.list()` para Google vía SDK
> `google-genai`; `GET /openai/v1/models` + una llamada real a
> `/chat/completions` para Groq, ya que `/models` puede listar modelos sin
> acceso real de inferencia para la cuenta). Vale la pena mencionar esto en
> el informe como parte de las "limitaciones del modelo utilizado" (IL1.4).
>
> Además, si alguna vez usas el proveedor `google` y obtienes
> `403 PERMISSION_DENIED: Your project has been denied access` con
> CUALQUIER modelo, no es un problema de nombre de modelo: es un bloqueo a
> nivel del proyecto de Google Cloud/AI Studio asociado a esa
> `GOOGLE_API_KEY` (posibles causas: falta de facturación habilitada, el
> proyecto quedó marcado/suspendido, o restricciones propias de la cuenta).
> La solución no es de código: genera una API key nueva desde un proyecto
> distinto, revisa el estado de facturación/cuota en Google Cloud Console,
> o contacta al soporte de Google como indica el mensaje de error. Por eso
> el proveedor de chat activo por defecto en este proyecto es Groq.

## Cómo correr el pipeline

1. **Construir el índice** a partir de `data/raw/` (incluye 4 documentos
   sintéticos de ejemplo: una factura, una nota de crédito con alerta de
   duplicidad, un informe de riesgo con estado de resultados, y un contrato
   de arriendo en UF para probar la fuente externa):

   ```bash
   python -m src.loaders
   ```

2. **Consultar el agente**:

   ```bash
   python -m src.rag_pipeline
   ```

   O de forma programática:

   ```python
   from src.rag_pipeline import AuditRagPipeline

   pipeline = AuditRagPipeline()
   resultado = pipeline.run("¿Existen señales de riesgo de fraude en la nota de crédito NC-045?")
   print(resultado.informe.model_dump_json(indent=2))
   print(resultado.callback.summary())
   ```

3. **Notebook de demo end-to-end**: `notebooks/demo_pipeline.ipynb` recorre
   ingesta -> consulta -> fact-checking -> métricas.

4. **Benchmark manual con evidencia de consumo de API** (gasta cuota real
   de Groq, no lo corre `pytest` automáticamente):

   ```bash
   python tests/manual_benchmark_groq.py
   ```

   Corre 5 preguntas reales (incluyendo el caso guardrail de un documento
   inexistente y una conversión UF→CLP vía la fuente externa) y resume
   llamadas/tokens/costo/latencia. El resultado de la última corrida está
   documentado en [`docs/evidencia_pruebas.md`](docs/evidencia_pruebas.md).

## Cómo correr los tests

```bash
pytest -v
```

Los tests están divididos según su dependencia de credenciales:

- `tests/test_loaders.py`, `tests/test_metrics.py`, `tests/test_prompts.py`,
  `tests/test_benchmark.py`, `tests/test_external_data.py`: **deterministas,
  no requieren ninguna API key ni llamadas de red reales** (ni siquiera de
  embeddings: `load_raw_documents`/`split_documents` no llaman a ningún
  proveedor; `test_external_data.py` mockea `requests.get` por completo).
  `test_benchmark.py` valida, con Exact Match y RegEx, la extracción de RUT
  y montos CLP contra `tests/ground_truth.json` sobre los documentos reales
  de `data/raw/`.
- Construir el índice FAISS (`python -m src.loaders`) usa embeddings
  **locales** por defecto: no requiere ninguna API key.
- La ejecución completa del pipeline contra el LLM real (`src/rag_pipeline.py`
  ejecutado como script, `tests/manual_benchmark_groq.py`, o las celdas del
  notebook que llaman al modelo) **sí requiere** `GROQ_API_KEY` configurada
  (o `GOOGLE_API_KEY`/`GITHUB_TOKEN` si cambias `LLM_PROVIDER`). La fuente
  externa (`mindicador.cl`) no requiere ninguna API key, pero sí acceso a
  internet; si no está disponible, el pipeline degrada con gracia (ver
  `docs/evidencia_pruebas.md` para un caso real de esto).

## Métricas de evaluación RAG

`src/metrics.py` implementa las 4 métricas oficiales del curso de forma
determinista (sin dependencia de RAGAS/LangSmith), pero con nombres y firmas
compatibles para facilitar una futura migración:

- **Context Precision**: `|recuperados ∩ relevantes| / |recuperados|`.
- **Context Recall**: `|recuperados ∩ relevantes| / |relevantes|`.
- **Faithfulness**: proporción de afirmaciones `[Soportado]` según
  `fact_checking.py`.
- **Answer Relevancy**: proxy léxico (similitud coseno TF) entre pregunta y
  respuesta — ver nota de diseño en `src/metrics.py` sobre cómo reemplazarlo
  por una implementación basada en embeddings reales.

## Memoria conversacional (IE4)

El pipeline soporta preguntas de seguimiento dentro de una misma sesión
(`session_id`), p.ej. "¿Cuál es el RUT de F-1023?" seguido de "¿Y cuál era
el RUT de esa misma factura otra vez?". `src/memory.py` implementa dos
estrategias intercambiables vía `MEMORY_STRATEGY`:

- **`buffer`** (default): guarda cada turno (pregunta + respuesta) tal
  cual. Sin llamadas LLM extra, sin pérdida de precisión, pero el contexto
  crece con la conversación.
- **`summary`**: mantiene un resumen progresivo, actualizado con una
  llamada extra al LLM (Groq) por turno. Contexto casi constante en
  tamaño, con el riesgo de que el resumen pierda precisión de cifras
  exactas al comprimir.

El historial se inyecta como texto ANTES de la pregunta actual — al JUEZ y
al GENERADOR (ambos basados en LLM), pero **no** al retriever (para no
diluir el embedding de búsqueda) — sin una llamada extra de reformulación
de pregunta ("history-aware retriever"), por ser más simple y confiable
dado que ya existen guardrails anti-alucinación estrictos. El prompt deja
explícito que el historial solo sirve para resolver referencias
ambiguas, nunca como fuente del dato reportado (ver `src/prompts.py`).

```python
from src.rag_pipeline import AuditRagPipeline

pipeline = AuditRagPipeline()  # usa MEMORY_STRATEGY de .env
pipeline.run("¿Cuál es el RUT del emisor en la factura F-1023?", session_id="sesion-1")
pipeline.run("¿Y cuál era el RUT de esa misma factura otra vez?", session_id="sesion-1")
```

**Comparación empírica (buffer vs summary)**, corrida real contra Groq de
la misma conversación de 5 turnos con una pregunta de seguimiento: ambas
estrategias respondieron el dato de seguimiento correctamente (sin
degradación observada), pero `summary` costó **16,2% más** ($0,010206 USD
vs $0,008783 USD) y usó **5 llamadas extra** a la API — la llamada
adicional de actualización del resumen superó, a esta escala, el ahorro de
tokens de comprimir el historial. Ver
[`docs/comparacion_memoria.md`](docs/comparacion_memoria.md) para la tabla
completa, el detalle por turno y la conclusión (llena el requisito de
"pruebas comparativas de hiperparámetros" de IE8).

## Uso de Inteligencia Artificial en este proyecto

Este repositorio fue generado con apoyo de **Claude Code (Anthropic)** para
redactar el andamiaje de código (estructura, módulos, prompts, tests,
diagrama). Conforme a las indicaciones del enunciado del EP1:

- Todo el contenido generado fue revisado y debe ser validado por el equipo
  antes de la entrega.
- **Las conclusiones, justificaciones técnicas y reflexiones individuales del
  informe NO fueron generadas por IA** y deben ser redactadas a mano por
  cada integrante del equipo, tal como exige la normativa de integridad
  académica del curso.
- Declarar el uso de IA en el informe final según
  <https://bibliotecas.duoc.cl/ia>.

## Limitaciones conocidas

- `answer_relevancy` es un proxy léxico, no semántico-embeddings; documentado
  explícitamente en `src/metrics.py`.
- El LLM-as-a-judge evalúa chunks uno por uno (no en batch), lo que es
  simple y trazable pero no está optimizado para lotes grandes de documentos.
- Los precios en `PRICING_USD_PER_1K_TOKENS` (`src/metrics.py`) son valores
  de referencia para ilustrar la metodología de costeo, no tarifas vigentes
  garantizadas.
