"""Tests deterministas de ChunkStore (EP2, IE1/IE2): sin API, sin red."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.agent.chunk_store import ChunkStore
from tests.agent_test_helpers import make_chunk


def test_put_many_returns_ids_and_get_many_round_trips():
    store = ChunkStore()
    docs = [
        make_chunk(0, "factura.txt", "Monto total: $1.000.000"),
        make_chunk(1, "factura.txt", "RUT: 76.123.456-7"),
    ]

    ids = store.put_many(docs)

    assert ids == [0, 1]
    assert store.get_many([0, 1]) == docs
    assert store.get(0).page_content == "Monto total: $1.000.000"


def test_get_many_ignores_unknown_ids_silently():
    store = ChunkStore()
    store.put_many([make_chunk(0, "factura.txt", "texto")])

    result = store.get_many([0, 999])

    assert len(result) == 1


def test_put_many_without_chunk_id_raises():
    store = ChunkStore()
    with pytest.raises(ValueError, match="chunk_id"):
        store.put_many([Document(page_content="sin id", metadata={})])


def test_snippet_truncates_long_text_and_collapses_whitespace():
    store = ChunkStore()
    long_text = "A" * 500
    store.put_many([make_chunk(0, "f.txt", f"line1\n\n  line2   {long_text}")])

    snippet = store.snippet(0, max_chars=20)

    assert len(snippet) == 21  # 20 chars + el caracter de elipsis
    assert "\n" not in snippet
    assert snippet.endswith("…")


def test_snippet_of_unknown_chunk_is_empty_string():
    store = ChunkStore()
    assert store.snippet(999) == ""


def test_source_returns_metadata_source_or_desconocido():
    store = ChunkStore()
    store.put_many([make_chunk(0, "factura_F1023.txt", "texto")])

    assert store.source(0) == "factura_F1023.txt"
    assert store.source(999) == "desconocido"
