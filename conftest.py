"""
Conftest raíz: garantiza que la raíz del repo esté en sys.path para que
`tests/` pueda hacer `from src...` sin importar el proyecto como paquete
instalado (no se usa `pip install -e .` en este proyecto académico).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
