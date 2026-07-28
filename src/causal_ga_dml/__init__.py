"""causal_ga_dml — Descubrimiento causal automatizado con Algoritmos Genéticos
y estimación de efectos con Double Machine Learning.

Paquete de referencia del proyecto de tesis doctoral:
"Especificación automatizada en descubrimiento causal mediante el enfoque genético".

Autor: Diego Alonso Córdova Ayala
Universidad Nacional de Ingeniería (UNI), Perú — Taller de Inferencia Causal (DES-304).
"""

__version__ = "1.0.0"
__author__ = "Diego Alonso Córdova Ayala"
__license__ = "MIT"

from .config import ExperimentConfig, load_config  # noqa: F401
from .seeds import set_global_seed, seed_sequence  # noqa: F401

__all__ = [
    "ExperimentConfig",
    "load_config",
    "set_global_seed",
    "seed_sequence",
    "__version__",
]
