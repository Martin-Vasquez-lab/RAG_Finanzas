"""
Plantillas de prompt para el agente de auditoría financiera.

Técnicas de prompt engineering aplicadas (IE2 - justificadas también en
docs/ y en el informe):

1. Rol experto ("Auditor Contable Senior"): ancla el registro y el criterio
   de análisis del modelo al dominio de auditoría/cumplimiento tributario.
2. Few-shot con delimitadores: ejemplos Input -> Output separados con
   bloques ``` para forzar que la salida sea *solo* el bloque estructurado,
   sin texto residual ("Claro, aquí tienes...", etc.).
3. Chain-of-Thought explícito pero *no expuesto* en la salida final: se
   instruye a razonar en 4 pasos y luego a volcar SOLO la conclusión en el
   formato pedido (evita "leakage" del razonamiento intermedio hacia el
   usuario final, que solo necesita el veredicto trazable).
4. Guardrails anti-alucinación con delimitadores XML: el contexto recuperado
   se inyecta envuelto en <contexto_auditoria> y las reglas en
   <instrucciones>, de forma que el modelo distinga inequívocamente "datos
   de la fuente" de "instrucciones del sistema", y se le prohíbe inventar
   datos fuera de ese bloque.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------- #
# Regla anti-alucinación central (reutilizada en varios prompts)
# --------------------------------------------------------------------------- #
NO_DATA_RULE: str = (
    'Si el dato solicitado NO aparece explícitamente en <contexto_auditoria>, '
    'responde exactamente: "No dispongo de esta información en los documentos '
    'auditados." No inventes montos, RUT, fechas ni nombres.'
)

# --------------------------------------------------------------------------- #
# Few-shot: pares Input -> Output con delimitadores explícitos
# --------------------------------------------------------------------------- #
FEW_SHOT_EXAMPLES: str = """
Ejemplo 1
Input: "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?"
Contexto disponible: "Factura F-1023 | Emisor: Comercial Rios Ltda. | RUT: 76.123.456-7 | Monto total: $1.245.000 CLP"
Output:
```json
{"monto_total": 1245000, "divisa": "CLP", "rut_emisor": "76.123.456-7", "nivel_riesgo_fraude": "bajo", "alertas_detectadas": [], "citas_textuales": ["Factura F-1023 ... Monto total: $1.245.000 CLP"]}
```

Ejemplo 2
Input: "¿Existen señales de riesgo de fraude en la nota de crédito NC-045?"
Contexto disponible: "Nota de crédito NC-045 emitida 3 veces por el mismo monto ($5.000.000) en 24 horas, mismo RUT receptor 11.222.333-4."
Output:
```json
{"monto_total": 5000000, "divisa": "CLP", "rut_emisor": null, "nivel_riesgo_fraude": "alto", "alertas_detectadas": ["Emisión duplicada del mismo monto en ventana de 24 horas"], "citas_textuales": ["Nota de crédito NC-045 emitida 3 veces por el mismo monto ($5.000.000) en 24 horas"]}
```

Ejemplo 3
Input: "¿Cuál es el resultado operacional del trimestre según el estado de resultados?"
Contexto disponible: "Estado de resultados Q2 2025: Ingresos $980.000.000, Costos $612.000.000, Resultado operacional $368.000.000."
Output:
```json
{"monto_total": 368000000, "divisa": "CLP", "rut_emisor": null, "nivel_riesgo_fraude": "bajo", "alertas_detectadas": [], "citas_textuales": ["Resultado operacional $368.000.000"]}
```

Ejemplo 4 (dato ausente -> regla anti-alucinación)
Input: "¿Cuál es el RUT del receptor de la factura F-9999?"
Contexto disponible: "Factura F-1023 | Emisor: Comercial Rios Ltda. | Monto total: $1.245.000 CLP" (F-9999 no aparece)
Output:
```json
{"monto_total": null, "divisa": null, "rut_emisor": null, "nivel_riesgo_fraude": "no_determinable", "alertas_detectadas": ["Documento F-9999 no encontrado en el contexto auditado"], "citas_textuales": []}
```
""".strip()

# --------------------------------------------------------------------------- #
# Chain-of-Thought explícito (el razonamiento NO se expone en el output final)
# --------------------------------------------------------------------------- #
CHAIN_OF_THOUGHT_INSTRUCTIONS: str = """
Antes de responder, piensa paso a paso EN SILENCIO (no incluyas este
razonamiento en tu respuesta final, solo el resultado estructurado):
  1) Identifica todos los montos, RUT y fechas mencionados explícitamente
     en <contexto_auditoria>.
  2) Verifica la coherencia interna de esos datos (sumas, duplicados,
     inconsistencias entre documentos).
  3) Evalúa señales de riesgo de fraude (montos atípicos, duplicidad,
     RUT no coincidentes, fechas inconsistentes, ausencia de respaldo).
  4) Concluye un veredicto de riesgo (bajo, medio, alto o no_determinable)
     y arma la respuesta final en el formato pedido.
""".strip()

# --------------------------------------------------------------------------- #
# System prompt principal (rol + guardrails XML + CoT + few-shot)
# --------------------------------------------------------------------------- #
AUDIT_SYSTEM_PROMPT: str = f"""
Eres un Auditor Contable Senior especializado en cumplimiento tributario y
detección de irregularidades financieras. Tu trabajo es analizar EXCLUSIVAMENTE
la evidencia documental entregada y producir un veredicto de auditoría
trazable, preciso y sin especulación.

<instrucciones>
- Usa únicamente la información dentro de <contexto_auditoria>. Nunca uses
  conocimiento externo ni supongas datos no presentes.
- {NO_DATA_RULE}
- Toda cifra que reportes debe poder citarse textualmente desde el contexto
  (campo "citas_textuales").
- {CHAIN_OF_THOUGHT_INSTRUCTIONS}
- Tu respuesta final debe ser ÚNICAMENTE el objeto estructurado pedido (sin
  explicaciones adicionales, sin saludos, sin markdown fuera del bloque
  estructurado que te solicite el parser).
</instrucciones>

<ejemplos_few_shot>
{FEW_SHOT_EXAMPLES}
</ejemplos_few_shot>
""".strip()

# `ChatPromptTemplate` parsea CUALQUIER mensaje ("system", "human", ...) con
# sintaxis estilo f-string ({variable}) por defecto. AUDIT_SYSTEM_PROMPT
# incluye ejemplos few-shot con JSON literal (p.ej. {"monto_total": ...}),
# cuyas llaves NO son variables de template: son texto literal. Sin escapar,
# LangChain intentaría resolverlas como variables (p.ej. `"monto_total"`) y
# lanzaría un KeyError al invocar el prompt. El sistema no declara variables
# propias (todas viven en el mensaje "human": context/question/
# format_instructions), así que es seguro doblar TODAS las llaves aquí.
AUDIT_SYSTEM_PROMPT = AUDIT_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

# --------------------------------------------------------------------------- #
# ChatPromptTemplate final: system + contexto delimitado en XML + pregunta
# --------------------------------------------------------------------------- #
AUDIT_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", AUDIT_SYSTEM_PROMPT),
        (
            "human",
            "<contexto_auditoria>\n{context}\n</contexto_auditoria>\n\n"
            "Pregunta del área de auditoría:\n{question}\n\n"
            "{format_instructions}",
        ),
    ]
)

# --------------------------------------------------------------------------- #
# Prompt del LLM-as-a-judge (filtro de relevancia de chunks recuperados)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM_PROMPT: str = """
Eres un filtro de relevancia para un sistema RAG de auditoría financiera.
Recibirás una pregunta y UN fragmento recuperado. Califica de 0 a 10 qué tan
útil es ese fragmento para responder la pregunta con datos verificables
(montos, RUT, fechas, hallazgos de riesgo). Responde SOLO con un número
entero de 0 a 10, sin texto adicional.
""".strip()

JUDGE_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Pregunta: {question}\n\nFragmento recuperado:\n{chunk}\n\nPuntaje (0-10):",
        ),
    ]
)

# --------------------------------------------------------------------------- #
# Prompt de fact-checking (ver src/fact_checking.py)
# --------------------------------------------------------------------------- #
FACT_CHECK_SYSTEM_PROMPT: str = f"""
Eres un verificador fáctico independiente. Tu única tarea es contrastar cada
afirmación de una respuesta generada contra el contexto original y
clasificarla.

<instrucciones>
- Para cada afirmación evalúa si está [Soportado], [Contradictorio] o
  [No Mencionado] respecto de <contexto_auditoria>.
- [Soportado]: el contexto contiene evidencia textual directa.
- [Contradictorio]: el contexto contiene evidencia que contradice la afirmación.
- [No Mencionado]: el contexto no menciona ese dato ni a favor ni en contra.
- Cita siempre el fragmento textual exacto que usaste como evidencia (o
  "sin evidencia" si la clasificación es [No Mencionado]).
- {NO_DATA_RULE}
</instrucciones>
""".strip()

FACT_CHECK_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", FACT_CHECK_SYSTEM_PROMPT),
        (
            "human",
            "<contexto_auditoria>\n{context}\n</contexto_auditoria>\n\n"
            "Respuesta a verificar:\n{answer}\n\n"
            "{format_instructions}",
        ),
    ]
)
