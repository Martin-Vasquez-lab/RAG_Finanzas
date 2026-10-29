"""Almacén de chunks por ejecución del agente (EP2, IE1/IE2).

Los chunks NUNCA viajan como texto por los argumentos de las herramientas
del LLM: `recuperar_documentos` los guarda aquí y expone solo IDs + un
snippet corto; las herramientas que necesitan el texto completo
(`juzgar_relevancia`, `generar_informe`, `verificar_afirmaciones`) lo leen
de aquí por ID. Esto evita que el modelo retipee -y potencialmente altere-
las citas textuales (lo que hace confiable el sistema), y ahorra tokens de
entrada en cada llamada.

Una instancia por ejecución (no es un singleton global): se inyecta en
`AgentDependencies` (`src/agent/tools.py`) y, en el grafo del próximo
commit, vivirá en el estado compartido entre nodos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class ChunkStore:
    _chunks: dict[int, Document] = field(default_factory=dict)

    def put_many(self, docs: list[Document]) -> list[int]:
        """Guarda `docs` (deben traer `chunk_id` en su metadata, como los
        produce `src/loaders.split_documents`) y devuelve sus IDs."""
        ids: list[int] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                raise ValueError(
                    "Documento sin 'chunk_id' en metadata; no se puede indexar en ChunkStore."
                )
            self._chunks[chunk_id] = doc
            ids.append(chunk_id)
        return ids

    def get(self, chunk_id: int) -> Document | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[int]) -> list[Document]:
        """Ignora silenciosamente IDs que no existan en el almacén (p.ej. si
        el LLM inventa un ID): el llamador decide si eso es un error."""
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def snippet(self, chunk_id: int, max_chars: int = 160) -> str:
        """Fragmento corto y legible para que el LLM decida qué chunk_ids
        pasar a juzgar_relevancia/generar_informe, sin exponerle el texto
        completo del documento."""
        doc = self._chunks.get(chunk_id)
        if doc is None:
            return ""
        text = " ".join(doc.page_content.split())
        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    def source(self, chunk_id: int) -> str:
        doc = self._chunks.get(chunk_id)
        return doc.metadata.get("source", "desconocido") if doc else "desconocido"

    def clear(self) -> None:
        self._chunks.clear()
