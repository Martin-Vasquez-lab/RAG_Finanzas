"""
Benchmark MANUAL comparativo: BufferMemory vs SummaryMemory (IE4 / IE8).

IMPORTANTE: este script **no es parte de la suite automática de pytest**
(no sigue la convención `test_*.py` a propósito, igual que
`manual_benchmark_groq.py`). Gasta cuota real de la API de Groq y tarda
varios minutos (corre la MISMA conversación de 5 turnos, una o dos veces
según la estrategia elegida, con pausas entre turnos para no exceder el
límite de tokens/minuto del tier gratuito).

Uso:
    python tests/manual_benchmark_memoria.py                  # ambas estrategias
    python tests/manual_benchmark_memoria.py --strategy buffer
    python tests/manual_benchmark_memoria.py --strategy summary

Mide y compara, para la sesión completa (5 turnos):
  - Tokens totales consumidos (incluye, para "summary", la llamada extra de
    actualización del resumen en cada turno).
  - Número de llamadas a la API.
  - Costo total estimado.
  - Si la pregunta de seguimiento (turno 4, que depende del turno 1) se
    respondió con el RUT correcto en AMBAS estrategias.

Reintenta automáticamente (con espera) ante `429 rate_limit_exceeded` del
tier gratuito de Groq, ya que una conversación de 5 turnos con historial
acumulado consume más tokens por turno que una pregunta aislada.

El resultado de la última corrida real queda documentado en
`docs/comparacion_memoria.md`.
"""

from __future__ import annotations

import argparse
import time

from src.rag_pipeline import AuditRagPipeline, RagRunResult

# Pausa entre turnos para no exceder el límite de tokens/minuto (TPM) del
# tier gratuito de Groq. El historial acumulado hace que cada turno pese
# más que en el benchmark sin memoria, así que la pausa es más generosa.
PAUSA_ENTRE_TURNOS_S: float = 30.0
PAUSA_ENTRE_ESTRATEGIAS_S: float = 30.0
MAX_REINTENTOS_RATE_LIMIT: int = 3
ESPERA_REINTENTO_S: float = 20.0

# RUT esperado del emisor de la factura F-1023 (ground truth, ver
# tests/ground_truth.json) — se usa para verificar si el turno 4
# (seguimiento) recuerda correctamente el dato del turno 1.
RUT_ESPERADO_F1023 = "76.123.456-7"

CONVERSACION: list[str] = [
    "¿Cuál es el monto total y el RUT del emisor en la factura F-1023?",
    "¿Existen señales de riesgo de fraude en la nota de crédito NC-045?",
    "¿Cuál es el resultado operacional del Q2 2025?",
    # Turno de seguimiento: depende del turno 1, no nombra el documento.
    "¿Y cuál era el RUT del emisor de esa misma factura que mencionamos al principio?",
    "¿Cuál es el canon de arriendo del contrato CA-078 en UF, y a cuánto "
    "equivale en CLP con el valor de la UF de hoy?",
]


def _run_turn_with_retry(pipeline: AuditRagPipeline, pregunta: str, session_id: str) -> RagRunResult:
    """Ejecuta un turno, reintentando ante `429 rate_limit_exceeded` del tier
    gratuito de Groq (espera fija + reintento, hasta MAX_REINTENTOS_RATE_LIMIT)."""
    for intento in range(1, MAX_REINTENTOS_RATE_LIMIT + 1):
        try:
            return pipeline.run(pregunta, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - se re-lanza si no es rate limit
            es_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if not es_rate_limit or intento == MAX_REINTENTOS_RATE_LIMIT:
                raise
            print(f"  [rate limit, intento {intento}/{MAX_REINTENTOS_RATE_LIMIT}] "
                  f"esperando {ESPERA_REINTENTO_S}s y reintentando...")
            time.sleep(ESPERA_REINTENTO_S)
    raise RuntimeError("No debería llegar aquí")  # pragma: no cover


def run_conversation(strategy: str) -> dict:
    """Corre la CONVERSACION completa con la estrategia de memoria dada y
    retorna un resumen de consumo + los informes de cada turno."""
    print(f"\n{'#'*70}\n# Estrategia: {strategy}\n{'#'*70}")
    pipeline = AuditRagPipeline(memory_strategy=strategy)
    session_id = f"benchmark-{strategy}"

    total_llamadas = 0
    total_in = 0
    total_out = 0
    total_costo = 0.0
    informes = []

    for i, pregunta in enumerate(CONVERSACION, start=1):
        if i > 1:
            time.sleep(PAUSA_ENTRE_TURNOS_S)
        print(f"\n--- Turno {i}: {pregunta}")
        result = _run_turn_with_retry(pipeline, pregunta, session_id)
        print(result.informe.model_dump_json(indent=2))
        print(f"  history_turns_used={result.history_turns_used}")

        resumen = result.callback.summary()
        print(
            f"  -> llamadas={resumen['num_llamadas']} "
            f"tokens_in={resumen['tokens_entrada_total']} "
            f"tokens_out={resumen['tokens_salida_total']} "
            f"costo=${resumen['costo_estimado_total_usd']:.6f}"
        )

        total_llamadas += resumen["num_llamadas"]
        total_in += resumen["tokens_entrada_total"]
        total_out += resumen["tokens_salida_total"]
        total_costo += resumen["costo_estimado_total_usd"]
        informes.append(result.informe)

    rut_turno_4 = informes[3].rut_emisor
    rut_correcto = rut_turno_4 == RUT_ESPERADO_F1023

    memoria = pipeline._sessions[session_id]
    contexto_final = memoria.get_context_for_prompt()

    print(f"\n=== Resumen estrategia '{strategy}' ===")
    print(f"  Llamadas totales:        {total_llamadas}")
    print(f"  Tokens entrada:          {total_in}")
    print(f"  Tokens salida:           {total_out}")
    print(f"  Tokens totales:          {total_in + total_out}")
    print(f"  Costo estimado total:    ${total_costo:.6f} USD")
    print(f"  RUT turno 4 (esperado {RUT_ESPERADO_F1023}): {rut_turno_4} -> {'OK' if rut_correcto else 'DEGRADADO'}")
    print(f"  Tamaño del contexto de memoria al final (chars): {len(contexto_final)}")

    return {
        "strategy": strategy,
        "total_llamadas": total_llamadas,
        "total_in": total_in,
        "total_out": total_out,
        "total_costo": total_costo,
        "rut_turno_4": rut_turno_4,
        "rut_correcto": rut_correcto,
        "context_size_chars": len(contexto_final),
        "informes": informes,
    }


def print_comparison(resultado_buffer: dict, resultado_summary: dict) -> None:
    print(f"\n{'='*70}\nCOMPARACIÓN FINAL: buffer vs summary\n{'='*70}")
    header = f"{'Métrica':<38} {'buffer':>14} {'summary':>14}"
    print(header)
    print("-" * len(header))
    filas = [
        ("Llamadas totales a la API", "total_llamadas", "{:>14}"),
        ("Tokens de entrada", "total_in", "{:>14}"),
        ("Tokens de salida", "total_out", "{:>14}"),
        ("Costo estimado total (USD)", "total_costo", "${:>13.6f}"),
        ("Tamaño contexto final (chars)", "context_size_chars", "{:>14}"),
        ("RUT turno 4 correcto", "rut_correcto", "{:>14}"),
    ]
    for etiqueta, clave, fmt in filas:
        v_buffer = fmt.format(resultado_buffer[clave])
        v_summary = fmt.format(resultado_summary[clave])
        print(f"{etiqueta:<38} {v_buffer:>14} {v_summary:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=["both", "buffer", "summary"], default="both",
        help="Qué estrategia(s) correr (default: both).",
    )
    args = parser.parse_args()

    if args.strategy in ("both", "buffer"):
        resultado_buffer = run_conversation("buffer")
    if args.strategy in ("both", "summary"):
        if args.strategy == "both":
            time.sleep(PAUSA_ENTRE_ESTRATEGIAS_S)
        resultado_summary = run_conversation("summary")

    if args.strategy == "both":
        print_comparison(resultado_buffer, resultado_summary)


if __name__ == "__main__":
    main()
