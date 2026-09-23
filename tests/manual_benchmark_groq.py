"""
Benchmark MANUAL contra Groq real — evidencia de pruebas de software.

IMPORTANTE: este script **no es parte de la suite automática de pytest**.
No sigue la convención `test_*.py` a propósito, para que `pytest` no lo
recolecte ni lo ejecute junto con `tests/test_*.py` (esos sí son
deterministas y gratuitos). Este script SÍ gasta cuota real de la API de
Groq (`GROQ_API_KEY` debe estar configurada en `.env`) y tarda varios
segundos por pregunta, así que se ejecuta a mano cuando se necesita generar
evidencia fresca, no en cada corrida de tests.

Uso:
    python tests/manual_benchmark_groq.py

Corre varias preguntas reales contra el pipeline completo (retriever FAISS
+ fuente externa mindicador.cl + LLM-as-a-judge + Groq + fact-checking) y
acumula el consumo total de API (llamadas, tokens, costo estimado,
latencia). El resultado de la última corrida real queda documentado en
`docs/evidencia_pruebas.md`.
"""

from __future__ import annotations

import time

from src.fact_checking import verify_answer
from src.rag_pipeline import AuditRagPipeline

# Pausa entre preguntas para no exceder el límite de tokens/minuto (TPM) del
# tier gratuito de Groq (8000 TPM en el momento de escribir esto). Cada
# pregunta consume ~4000-5000 tokens entre el juez y la respuesta final.
PAUSA_ENTRE_PREGUNTAS_S: float = 15.0

PREGUNTAS: list[str] = [
    "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?",
    "¿Existen señales de riesgo de fraude en la nota de crédito NC-045?",
    "¿Cuál es el resultado operacional del Q2 2025?",
    # Documento inexistente a propósito: valida el guardrail anti-alucinación
    # (el modelo debe responder "no_determinable" y NO inventar un RUT).
    "¿Cuál es el RUT del receptor de la factura F-9999?",
    # Requiere la fuente EXTERNA (mindicador.cl) para convertir UF -> CLP:
    # valida que el pipeline combine fuente interna + externa (IL1.2/IE3).
    "¿Cuál es el canon de arriendo mensual del contrato CA-078 y a cuánto "
    "equivale en CLP con el valor de la UF de hoy?",
]


def run_benchmark() -> None:
    pipeline = AuditRagPipeline()

    total_llamadas = 0
    total_in = 0
    total_out = 0
    total_costo = 0.0
    total_latencia = 0.0
    result = None

    for i, pregunta in enumerate(PREGUNTAS, start=1):
        if i > 1:
            time.sleep(PAUSA_ENTRE_PREGUNTAS_S)
        print(f"\n{'='*70}\nPregunta {i}: {pregunta}\n{'='*70}")
        result = pipeline.run(pregunta)
        print(result.informe.model_dump_json(indent=2))

        resumen = result.callback.summary()
        print(
            f"  -> llamadas={resumen['num_llamadas']} "
            f"tokens_in={resumen['tokens_entrada_total']} "
            f"tokens_out={resumen['tokens_salida_total']} "
            f"costo=${resumen['costo_estimado_total_usd']:.6f} "
            f"lat_prom={resumen['latencia_promedio_s']:.2f}s "
            f"fuente_externa_disponible={result.external_data_available}"
        )

        total_llamadas += resumen["num_llamadas"]
        total_in += resumen["tokens_entrada_total"]
        total_out += resumen["tokens_salida_total"]
        total_costo += resumen["costo_estimado_total_usd"]
        total_latencia += resumen["latencia_promedio_s"] * resumen["num_llamadas"]

    # Fact-checking de la última respuesta, para incluir también su consumo.
    if result is not None:
        report = verify_answer(answer=result.informe.model_dump_json(), context=result.context_text)
        print(f"\nFact-check de la última respuesta: {len(report.afirmaciones)} afirmaciones verificadas")
        for c in report.afirmaciones:
            print(f"  [{c.veredicto}] {c.afirmacion}")

    print(f"\n{'#'*70}")
    print("# CONSUMO TOTAL DE API (Groq) EN ESTA CORRIDA")
    print(f"{'#'*70}")
    print(f"  Preguntas procesadas:      {len(PREGUNTAS)}")
    print(f"  Llamadas totales a Groq:   {total_llamadas}")
    print(f"  Tokens de entrada totales: {total_in}")
    print(f"  Tokens de salida totales:  {total_out}")
    print(f"  Tokens totales:            {total_in + total_out}")
    print(f"  Costo estimado total:      ${total_costo:.6f} USD")
    print(f"  Latencia acumulada:        {total_latencia:.2f} s")


if __name__ == "__main__":
    run_benchmark()
