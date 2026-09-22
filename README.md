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

Copia `.env.example` a `.env` y completa tu clave de Google AI Studio:

```bash
cp .env.example .env
```

| Variable | Requerida | Descripción |
|---|---|---|
| `GOOGLE_API_KEY` | Sí (proveedor `google`, el usado por defecto) | Clave de [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `LLM_PROVIDER` | No (default `google`) | `google` o `github`, ver justificación abajo. |
| `GOOGLE_CHAT_MODEL` | No (default `gemini-1.5-flash`) | Modelo de generación. |
| `GOOGLE_EMBEDDING_MODEL` | No (default `models/text-embedding-004`) | Modelo de embeddings. |
| `TEMPERATURE` | No (default `0.0`) | Determinismo exigido para un dominio de auditoría. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | No (default `1200`/`200`) | Ver justificación en `src/loaders.py`. |
| `RETRIEVER_K` | No (default `8`) | Top-k de chunks recuperados antes del filtro LLM-judge. |
| `JUDGE_MIN_SCORE` | No (default `6.0`) | Umbral (0-10) para que un chunk pase el filtro del juez. |

**Nunca subas tu `.env` real al repositorio** (`.gitignore` ya lo excluye).

### Sobre el proveedor de LLM

El curso enseña dos patrones de conexión: **GitHub Models** (usado en los
notebooks de clase) y **Google AI Studio/Gemini**. El enunciado del EP1 pide
explícitamente `text-embedding-004` + Gemini, por lo que ese es el proveedor
activo por defecto. Toda la lógica de conexión vive aislada en
`src/config.py` (`get_chat_model()` / `get_embeddings()`), de modo que
cambiar a `LLM_PROVIDER=github` en `.env` no requiere tocar ningún otro
archivo del pipeline.

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
  `tests/test_benchmark.py`: **deterministas, no requieren `GOOGLE_API_KEY`**.
  `test_benchmark.py` valida, con Exact Match y RegEx, la extracción de RUT y
  montos CLP contra `tests/ground_truth.json` sobre los documentos reales de
  `data/raw/`.
- La ejecución completa del pipeline contra el LLM real (`src/rag_pipeline.py`
  ejecutado como script, o las celdas del notebook que llaman a Gemini)
  **sí requiere** `GOOGLE_API_KEY` configurada.

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
