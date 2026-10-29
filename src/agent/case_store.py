"""Sistema de casos y memoria de largo plazo de hallazgos de auditoría
(EP2, IE1 "escritura" / IE3 / IE4).

Dos piezas separadas, con responsabilidad distinta:

- `CaseLog`: persiste CADA hallazgo (aprobado o pendiente) en un archivo
  JSONL local, con deduplicación por clave de contenido. Es la fuente de
  verdad de "qué se registró y cuándo" (trazabilidad).
- `FindingsIndex`: índice FAISS SEPARADO del índice de documentos de la
  EP1 (`src/loaders.py`), que solo indexa hallazgos ya APROBADOS,
  consultable por similitud semántica entre preguntas/sesiones. Es la
  memoria de LARGO plazo del agente -distinta de `src/memory.py`, que es
  memoria de CORTO plazo de una sola sesión, y distinta del FAISS de
  documentos, que es la fuente RAG, no memoria del agente.

La aprobación de un hallazgo "pendiente" vive en `src/agent/approval.py`,
deliberadamente FUERA de este módulo y fuera de la lista de herramientas
del LLM (`src/agent/tools.py`): así el agente nunca puede auto-aprobarse
un hallazgo de riesgo alto.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

NivelRiesgo = Literal["bajo", "medio", "alto"]
EstadoHallazgo = Literal["aprobado", "pendiente_de_aprobación"]


def _dedup_key(descripcion: str, chunk_ids: list[int]) -> str:
    """Clave de deduplicación: descripción normalizada (espacios/mayúsculas
    colapsados) + chunk_ids ordenados. Dos llamadas con la misma intención
    (aunque el LLM varíe el redactado levemente) no deben crear dos
    entradas distintas en el log."""
    normalizada = re.sub(r"\s+", " ", descripcion.strip().lower())
    return f"{normalizada}|{sorted(chunk_ids)}"


@dataclass(frozen=True)
class RegistroResult:
    id: str
    estado: EstadoHallazgo
    descripcion: str
    duplicado: bool = False


class CaseLog:
    """Log de casos en JSONL. Las escrituras reescriben el archivo completo
    (no solo append): es lo que permite `mark_approved` actualizar el
    estado de una entrada existente sin un motor de base de datos, a costa
    aceptable dado el volumen bajo esperado de hallazgos por sesión."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def _write_all(self, entries: list[dict]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def find_by_dedup_key(self, key: str) -> dict | None:
        for entry in self._read_all():
            if entry.get("dedup_key") == key:
                return entry
        return None

    def register(
        self,
        descripcion: str,
        nivel_riesgo: NivelRiesgo,
        chunk_ids_relacionados: list[int] | None = None,
    ) -> RegistroResult:
        """Registra un hallazgo. Si `nivel_riesgo == 'alto'`, NUNCA escribe
        como aprobado directamente: queda `pendiente_de_aprobación` y no se
        indexa en `FindingsIndex` hasta que `approval.aprobar_hallazgo` lo
        confirme. Para 'bajo'/'medio' queda aprobado de inmediato (el
        llamador en `tools.py` es responsable de indexarlo)."""
        chunk_ids_relacionados = chunk_ids_relacionados or []
        key = _dedup_key(descripcion, chunk_ids_relacionados)

        existing = self.find_by_dedup_key(key)
        if existing is not None:
            return RegistroResult(
                id=existing["id"],
                estado=existing["estado"],
                descripcion=existing["descripcion"],
                duplicado=True,
            )

        estado: EstadoHallazgo = "pendiente_de_aprobación" if nivel_riesgo == "alto" else "aprobado"
        entry = {
            "id": str(uuid.uuid4()),
            "dedup_key": key,
            "descripcion": descripcion,
            "nivel_riesgo": nivel_riesgo,
            "chunk_ids_relacionados": chunk_ids_relacionados,
            "estado": estado,
            "registrado_en": datetime.now(timezone.utc).isoformat(),
        }
        entries = self._read_all()
        entries.append(entry)
        self._write_all(entries)

        return RegistroResult(id=entry["id"], estado=estado, descripcion=descripcion, duplicado=False)

    def get(self, finding_id: str) -> dict | None:
        for entry in self._read_all():
            if entry["id"] == finding_id:
                return entry
        return None

    def mark_approved(self, finding_id: str) -> dict:
        """Usado SOLO por `approval.aprobar_hallazgo` (nunca directo desde
        una herramienta del LLM)."""
        entries = self._read_all()
        updated = None
        for entry in entries:
            if entry["id"] == finding_id:
                entry["estado"] = "aprobado"
                entry["aprobado_en"] = datetime.now(timezone.utc).isoformat()
                updated = entry
                break
        if updated is None:
            raise KeyError(f"No existe un hallazgo con id={finding_id!r} en {self.path}")
        self._write_all(entries)
        return updated


class FindingsIndex:
    """Índice FAISS separado del de documentos de la EP1: solo hallazgos
    APROBADOS. `embeddings` es inyectable (en producción, los embeddings
    locales de `src.config.get_embeddings()`; en tests, un fake
    determinista, ver `tests/agent_test_helpers.py`) para no depender de
    red ni de un modelo real en la suite determinista."""

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings
        self._vectorstore = None  # se crea perezosamente en el primer add()

    def add(self, finding_id: str, descripcion: str, metadata: dict | None = None) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        doc = Document(page_content=descripcion, metadata={"finding_id": finding_id, **(metadata or {})})
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents([doc], self._embeddings)
        else:
            self._vectorstore.add_documents([doc])

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self._vectorstore is None:
            return []
        docs = self._vectorstore.similarity_search(query, k=k)
        return [{"finding_id": d.metadata.get("finding_id"), "descripcion": d.page_content} for d in docs]
