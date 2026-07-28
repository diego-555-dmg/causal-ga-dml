"""Configuración común de las pruebas: hace importable `src/` sin instalar el paquete."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
