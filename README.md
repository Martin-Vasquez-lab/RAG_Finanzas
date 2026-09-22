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

Ver [`docs/architecture.md`](docs/architecture.md) para el diagrama de
arquitectura completo y la justificación de decisiones.

## Estructura del repositorio

```
README.md
requirements.txt
.env.example
conftest.py
data/raw/            # Documentos fuente (3 ejemplos sintéticos incluidos)
data/processed/       # Artefactos intermedios (vacío, generado en ejecución)
vectorstore/          # Índice FAISS persistido (generado, no versionado)
notebooks/             # Notebook de demo end-to-end
src/
  config.py            # Conexión a LLM/embeddings (Google Gemini / GitHub Models)
  loaders.py            # Ingesta, chunking, construcción del índice FAISS
  prompts.py            # Plantillas: rol experto, few-shot, CoT, guardrails XML
  rag_pipeline.py        # Ensamblado LCEL: retriever | judge | prompt | LLM | parser
  metrics.py              # Callback de costo/latencia + 4 métricas RAG + esquemas Pydantic
  fact_checking.py         # Cadena secundaria de verificación fáctica
tests/                     # Benchmarking determinista (Exact Match + RegEx) y unit tests
docs/                       # Diagrama de arquitectura (Mermaid) y notas de diseño
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

**Nunca subas tu `.env` real al repositorio** (`.gitignore` ya lo excluye).

### Sobre los proveedores de LLM/embeddings

El proveedor de **chat** y el de **embeddings** se configuran por separado
(`LLM_PROVIDER` / `EMBEDDING_PROVIDER`), porque no todos los proveedores
ofrecen ambas capacidades:

- **Chat activo por defecto: Groq** (`LLM_PROVIDER=groq`), rápido y con tier
  gratuito. El modelo, `openai/gpt-oss-120b`, es un modelo "razonador" de
  pesos abiertos servido por Groq — encaja con el diseño Chain-of-Thought
  del prompt de auditoría.
- **Embeddings activos por defecto: locales** (`EMBEDDING_PROVIDER=local`),
  vía `sentence-transformers` corriendo en CPU. **Groq no ofrece API de
  embeddings**, así que el pipeline no depende de ningún proveedor externo
  para esa etapa — sin costo, sin llamadas de red, sin cuotas.
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

1. **Construir el índice** a partir de `data/raw/` (incluye 3 documentos
   sintéticos de ejemplo: una factura, una nota de crédito con alerta de
   duplicidad, y un informe de riesgo con estado de resultados):

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

## Cómo correr los tests

```bash
pytest -v
```

Los tests están divididos según su dependencia de credenciales:

- `tests/test_loaders.py`, `tests/test_metrics.py`, `tests/test_prompts.py`,
  `tests/test_benchmark.py`: **deterministas, no requieren ninguna API key**
  (ni siquiera de embeddings: `load_raw_documents`/`split_documents` no
  llaman a ningún proveedor). `test_benchmark.py` valida, con Exact Match y
  RegEx, la extracción de RUT y montos CLP contra `tests/ground_truth.json`
  sobre los documentos reales de `data/raw/`.
- Construir el índice FAISS (`python -m src.loaders`) usa embeddings
  **locales** por defecto: no requiere ninguna API key.
- La ejecución completa del pipeline contra el LLM real (`src/rag_pipeline.py`
  ejecutado como script, o las celdas del notebook que llaman al modelo)
  **sí requiere** `GROQ_API_KEY` configurada (o `GOOGLE_API_KEY`/`GITHUB_TOKEN`
  si cambias `LLM_PROVIDER`).

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
