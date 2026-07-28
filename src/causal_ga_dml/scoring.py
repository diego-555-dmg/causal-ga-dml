"""Score gaussiano BIC descomponible basado en la matriz de covarianza.

La varianza residual de un nodo dado un conjunto de padres se obtiene por
complemento de Schur (varianza parcial). Esto evita recorrer los datos en cada
evaluación de aptitud: el costo pasa de O(n·p) por evaluación a O(|Pa|^3), lo
que hace viable ejecutar decenas de miles de evaluaciones dentro del AG.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


class GaussianBICScore:
    """Score BIC gaussiano local y descomponible.

    Para un nodo :math:`X_i` con padres :math:`Pa_i`:

    .. math::
        \\mathrm{score}(X_i \\mid Pa_i) =
        -\\frac{n}{2}\\log \\hat{\\sigma}^2_{i \\mid Pa_i}
        - \\frac{\\lambda}{2}\\,k\\,\\log n

    donde :math:`\\hat{\\sigma}^2` es la varianza parcial, :math:`k = |Pa_i| + 1`
    y :math:`\\lambda` es el factor de penalización (`penalty`). Con
    :math:`\\lambda = 1` se recupera el BIC clásico; valores mayores penalizan
    con más fuerza la complejidad y reducen los falsos positivos estructurales.
    """

    def __init__(self, data: pd.DataFrame, penalty: float = 3.5, ridge: float = 1e-6):
        self.columns: List[str] = list(data.columns)
        self._index: Dict[str, int] = {c: i for i, c in enumerate(self.columns)}

        X = np.asarray(data.values, dtype=float)
        # Estandarización: estabilidad numérica e invarianza de escala del score.
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        self.n, self.p = X.shape
        self.cov = np.cov(X, rowvar=False) + ridge * np.eye(self.p)
        self.penalty = float(penalty)
        self._cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.n_evaluations = 0

    # ------------------------------------------------------------------ #
    def residual_variance(self, node: str, parents: Sequence[str]) -> float:
        """Varianza parcial de `node` dado `parents` (complemento de Schur)."""
        i = self._index[node]
        if not parents:
            return float(self.cov[i, i])
        P = [self._index[p] for p in parents]
        Spp = self.cov[np.ix_(P, P)]
        Spi = self.cov[np.ix_(P, [i])]
        var = self.cov[i, i] - float(Spi.T @ np.linalg.solve(Spp, Spi))
        return float(max(var, 1e-8))

    def local_score(self, node: str, parents: Iterable[str]) -> float:
        """Score BIC local del nodo dado sus padres (memoizado)."""
        key = (node, tuple(sorted(parents)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        var = self.residual_variance(node, key[1])
        k = len(key[1]) + 1  # coeficientes de regresión + varianza residual
        score = -0.5 * self.n * np.log(var) - 0.5 * self.penalty * k * np.log(self.n)

        self._cache[key] = score
        self.n_evaluations += 1
        return float(score)

    def graph_score(self, parents_map: Dict[str, Sequence[str]]) -> float:
        """Score global del DAG: suma de los scores locales (descomponibilidad)."""
        return float(sum(self.local_score(node, pa) for node, pa in parents_map.items()))

    @property
    def cache_size(self) -> int:
        return len(self._cache)
