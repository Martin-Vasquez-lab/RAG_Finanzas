"""Fakes deterministas compartidos por los tests del agente (EP2).

No es un archivo de tests (pytest no lo colecciona: no empieza con
`test_`): solo utilidades para no depender de red, API ni del modelo real
de embeddings en la suite determinista.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

_VECTOR_DIM = 64


class FakeEmbeddings(Embeddings):
    """Embeddings deterministas (hashing estable de palabras, no el hash()
    nativo de Python que varía con PYTHONHASHSEED) para `FindingsIndex` en
    tests, sin descargar ni correr sentence-transformers. Hereda de
    `Embeddings` porque `FAISS` exige `isinstance(..., Embeddings)` y, si
    no, trata el objeto como una función callable legacy."""

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * _VECTOR_DIM
        for word in text.lower().split():
            idx = zlib.crc32(word.encode("utf-8")) % _VECTOR_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class _FakeResponse:
    content: str
    response_metadata: dict = field(default_factory=dict)


class FakeChatModel(Runnable):
    """Chat model falso, compatible con el operador LCEL `prompt | llm`
    (hereda de `Runnable`, así `coerce_to_runnable` lo acepta tal cual).
    Devuelve, en orden, las respuestas de `responses` como `.content`;
    registra cada `invoke()` en `.calls` para afirmar sobre el orden real
    de las llamadas en los tests."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[object] = []

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 - firma estándar de Runnable
        self.calls.append(input)
        if not self._responses:
            raise AssertionError("FakeChatModel: se pidieron más respuestas de las configuradas en el test.")
        return _FakeResponse(self._responses.pop(0))


class FakeRetriever:
    """Retriever falso: siempre devuelve la misma lista de `Document`,
    sin tocar FAISS ni embeddings reales."""

    def __init__(self, docs: list[Document]):
        self._docs = docs
        self.calls: list[str] = []

    def invoke(self, query: str, config=None):
        self.calls.append(query)
        return list(self._docs)


def make_chunk(chunk_id: int, source: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": source})


class ScriptedToolCallingLLM(Runnable):
    """LLM simulado para tests del GRAFO (no de herramientas individuales):
    cada `invoke()` devuelve el siguiente `AIMessage` de un guion fijo
    (con o sin `tool_calls`). `bind_tools()` se ignora -no valida el
    esquema real- y devuelve el mismo objeto: el punto de este fake es
    probar que el GRAFO obedece lo que el "modelo" decide en cada paso,
    sin cablear ningún orden, no probar razonamiento real."""

    def __init__(self, script: list[AIMessage]):
        self._script = list(script)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        self.calls += 1
        if not self._script:
            raise AssertionError("ScriptedToolCallingLLM: se agotó el guion configurado en el test.")
        return self._script.pop(0)


def ai_tool_call(name: str, args: dict, call_id: str, usage_metadata: dict | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=usage_metadata,
    )


def ai_final(content: str) -> AIMessage:
    return AIMessage(content=content)


INFORME_JSON_OK = (
    '{"monto_total": 1245000, "divisa": "CLP", "rut_emisor": "76.123.456-7", '
    '"nivel_riesgo_fraude": "bajo", "alertas_detectadas": [], '
    '"citas_textuales": ["Factura F-1023 ... Monto total: $1.245.000 CLP"], '
    '"fuente_externa_utilizada": false, "uf_referencia_clp": null}'
)

PLAN_JSON_OK = (
    '{"pasos": [{"orden": 1, "prioridad": "alta", "accion": "recuperar_documentos", '
    '"razon": "Obtener el contexto necesario antes de cualquier otro paso."}]}'
)

FACT_CHECK_JSON_OK = (
    '{"afirmaciones": [{"afirmacion": "monto_total=1245000", "veredicto": "Soportado", '
    '"evidencia": "Monto total: $1.245.000 CLP"}]}'
)
