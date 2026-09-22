# Arquitectura de la solución — Agente RAG de Auditoría Financiera (EP1 - ISY0101)

## Diagrama de flujo (Mermaid)

```mermaid
flowchart TD
    U[Usuario / Auditor] -->|Pregunta en lenguaje natural| R

    subgraph Ingesta["Ingesta offline (src/loaders.py)"]
        RAW[("data/raw/\nfacturas, notas de crédito,\nestados de resultados, balances")] --> SPLIT["RecursiveCharacterTextSplitter\n(chunk_size=1200, overlap=200,\nseparadores anti-corte de tablas)"]
        SPLIT --> META["Metadatos por chunk:\nsource, page, fecha_ingesta,\ntipo_documento, chunk_id"]
        META --> EMB["Embeddings\nGoogle gemini-embedding-001"]
        EMB --> IDX[("Índice FAISS\nvectorstore/")]
    end

    subgraph Consulta["Consulta online (src/rag_pipeline.py)"]
        R["Retriever FAISS\n(top-k=8)"] --> J["LLM-as-a-Judge\n(filtra chunks < score 6/10)"]
        J --> CTX["Contexto citable\n<contexto_auditoria>"]
        CTX --> P["ChatPromptTemplate\n(rol Auditor Senior +\nfew-shot + CoT +\nguardrails XML)"]
        P --> LLM["Gemini Flash\n(temperature=0.0)"]
        LLM --> OUT["Parser Pydantic estricto\nInformeAuditoria"]
    end

    subgraph Verificacion["Verificación (src/fact_checking.py + src/metrics.py)"]
        OUT --> FC["Cadena de Fact-Checking\n[Soportado]/[Contradictorio]/[No Mencionado]"]
        FC --> MET["Métricas RAG\ncontext_precision, context_recall,\nfaithfulness, answer_relevancy"]
        LLM --> CB["AuditCallbackHandler\nlatencia, tokens, costo estimado"]
    end

    IDX --> R
    OUT --> RESP[Respuesta estructurada + trazabilidad]
    MET --> RESP
    CB --> RESP
    RESP --> U
```

## Componentes clave

| Componente | Módulo | Responsabilidad |
|---|---|---|
| Ingesta y chunking | `src/loaders.py` | Carga PDF/TXT, fragmenta sin cortar tablas financieras, adjunta metadatos de trazabilidad, construye/persiste FAISS. |
| Configuración y proveedor LLM | `src/config.py` | Único punto de conexión a Google AI Studio/Gemini (o GitHub Models como alternativa), parámetros centrales. |
| Prompts | `src/prompts.py` | Rol experto, few-shot con delimitadores, Chain-of-Thought explícito, guardrails XML anti-alucinación. |
| Orquestación RAG | `src/rag_pipeline.py` | Ensambla `retriever \| LLM-judge \| prompt \| model \| parser` vía LCEL; expone `AuditRagPipeline`. |
| Verificación fáctica | `src/fact_checking.py` | Segunda cadena LLM que clasifica cada afirmación de la respuesta contra el contexto. |
| Observabilidad y métricas | `src/metrics.py` | `AuditCallbackHandler` (latencia/tokens/costo) + las 4 métricas RAG deterministas + esquemas Pydantic. |

## Justificación de decisiones (resumen; ver informe técnico para el detalle completo)

- **LLM-as-a-judge antes del generador**: en auditoría, un chunk irrelevante que "contamina" el contexto puede inducir al modelo a mezclar cifras de documentos distintos. Se prioriza precisión sobre recall.
- **Salida Pydantic estricta**: permite validar programáticamente cada respuesta (tipos, campos obligatorios) antes de que llegue a un sistema downstream de cumplimiento.
- **Fact-checking como cadena separada**: reduce el riesgo de que el mismo sesgo de la generación contamine su propia verificación.
- **`temperature=0.0`**: en un dominio donde un monto o RUT mal citado tiene implicancias regulatorias, se prioriza determinismo sobre creatividad.
