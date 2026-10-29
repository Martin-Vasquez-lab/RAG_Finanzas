"""Almacén de InformeAuditoria generados durante una ejecución del agente
(EP2, IE1/IE2).

Misma razón que `chunk_store.ChunkStore`: `verificar_afirmaciones` debe
referirse a un informe por ID, no recibir su JSON completo como argumento
del LLM (evita gasto de tokens y la posibilidad de que el modelo altere
el contenido al retipearlo).

Además, una `InformeStore` vacía al construirse es lo que permite validar
EN CÓDIGO -sin depender de que el prompt se obedezca- que `generar_informe`
se invoque como máximo una vez por pregunta: una instancia de
`InformeStore` vive una sola ejecución del agente (igual convención que
`ChunkStore`), así que basta con rechazar una segunda escritura.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.metrics import InformeAuditoria


@dataclass
class InformeStore:
    _informes: dict[str, InformeAuditoria] = field(default_factory=dict)

    def put(self, informe: InformeAuditoria) -> str:
        informe_id = str(uuid.uuid4())
        self._informes[informe_id] = informe
        return informe_id

    def get(self, informe_id: str) -> InformeAuditoria | None:
        return self._informes.get(informe_id)

    def count(self) -> int:
        return len(self._informes)
