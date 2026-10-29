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
    red ni de un modelo real en la suite determinista.

    Persistencia en disco OPCIONAL (`persist_dir`): sin ella, la memoria
    de largo plazo solo vive en memoria del proceso -útil para tests
    aislados-. Con `persist_dir`, cada `add()` se guarda inmediatamente
    (mismo patrón que `src/loaders.build_faiss_index`/`load_faiss_index`
    para el índice de documentos), y una instancia NUEVA construida sobre
    el mismo `persist_dir` recarga lo ya indexado -es lo que demuestra
    que la memoria de largo plazo sobrevive a un reinicio del proceso,
    no solo a una sesión de chat nueva."""

    def __init__(self, embeddings, persist_dir: Path | None = None, index_name: str = "hallazgos_faiss_index") -> None:
        self._embeddings = embeddings
        self._persist_dir = persist_dir
        self._index_name = index_name
        self._vectorstore = self._load_from_disk_if_exists()

    def _index_file(self) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / f"{self._index_name}.faiss"

    def _load_from_disk_if_exists(self):
        index_file = self._index_file()
        if index_file is None or not index_file.exists():
            return None
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            str(self._persist_dir),
            self._embeddings,
            index_name=self._index_name,
            # Seguro en este proyecto: este índice solo lo escribe nuestro
            # propio código (registrar_hallazgo/aprobar_hallazgo), nunca se
            # carga un .faiss de origen externo/no confiable.
            allow_dangerous_deserialization=True,
        )

    def add(self, finding_id: str, descripcion: str, metadata: dict | None = None) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        doc = Document(page_content=descripcion, metadata={"finding_id": finding_id, **(metadata or {})})
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents([doc], self._embeddings)
        else:
            self._vectorstore.add_documents([doc])
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._vectorstore.save_local(str(self._persist_dir), index_name=self._index_name)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Devuelve `nivel_riesgo` del metadata original (no solo
        `finding_id`/`descripcion`): sin esto, un LLM que cita un hallazgo
        recuperado tendría que RE-INFERIR el riesgo desde el texto de la
        descripción en vez de leer el valor ya calibrado que se guardó al
        registrar el hallazgo -observado en vivo: el modelo reportó un
        nivel de riesgo distinto al realmente registrado."""
        if self._vectorstore is None:
            return []
        docs = self._vectorstore.similarity_search(query, k=k)
        return [
            {
                "finding_id": d.metadata.get("finding_id"),
                "descripcion": d.page_content,
                "nivel_riesgo": d.metadata.get("nivel_riesgo"),
            }
            for d in docs
        ]
