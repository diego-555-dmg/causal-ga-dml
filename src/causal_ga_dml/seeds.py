"""Control centralizado de la aleatoriedad.

Toda fuente de aleatoriedad del pipeline (Python `random`, NumPy, scikit-learn y
el muestreador de pgmpy) se deriva de una única semilla maestra. Esto garantiza
que dos ejecuciones con la misma configuración produzcan resultados idénticos
bit a bit en la misma versión de las librerías.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Fija la semilla de todos los generadores globales y devuelve un Generator.

    Parameters
    ----------
    seed : int
        Semilla maestra del experimento.

    Returns
    -------
    numpy.random.Generator
        Generador dedicado, preferible a las funciones globales de ``np.random``
        porque su estado no puede ser alterado por código de terceros.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def seed_sequence(master_seed: int, n: int) -> List[int]:
    """Deriva `n` semillas hijas reproducibles a partir de una semilla maestra.

    Se usa `numpy.random.SeedSequence`, que garantiza flujos estadísticamente
    independientes entre réplicas — condición necesaria para que la media y la
    desviación estándar entre réplicas sean interpretables.
    """
    ss = np.random.SeedSequence(master_seed)
    return [int(s.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))
            for s in ss.spawn(n)]
