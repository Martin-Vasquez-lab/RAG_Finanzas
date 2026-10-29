"""Reintento acotado ante respuesta vacía del LLM, compartido por las
herramientas del agente que llaman al modelo (EP2, IE1).

`openai/gpt-oss-120b` en Groq devuelve ocasionalmente una respuesta vacía
(observado en el chequeo de salud real del proyecto, ver
scripts de diagnóstico). Las herramientas que dependen del LLM reintentan
de forma acotada en vez de propagar una excepción que cortaría el grafo
del agente a mitad de turno.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 1.0


def invoke_with_retry(
    call: Callable[[], T],
    *,
    is_empty: Callable[[T], bool],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Ejecuta `call()` hasta `max_retries` veces mientras el resultado sea
    "vacío" según `is_empty`. Devuelve el último resultado obtenido (vacío
    o no) si se agotan los reintentos: nunca lanza una excepción propia,
    para que la herramienta que la use decida cómo convertir ese resultado
    en una observación clara."""
    attempt = 0
    result = call()
    while is_empty(result) and attempt < max_retries:
        attempt += 1
        if backoff_seconds:
            sleep(backoff_seconds)
        result = call()
    return result
