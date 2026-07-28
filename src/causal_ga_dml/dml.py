"""Estimación del efecto causal con Double Machine Learning (DML).

Se implementa el modelo parcialmente lineal de Chernozhukov et al. (2018):

.. math::
    Y = \\theta_0 T + g_0(Z) + U, \\qquad T = m_0(Z) + V

El estimador ortogonal de Neyman con *cross-fitting* es

.. math::
    \\hat{\\theta} = \\frac{\\widehat{\\mathrm{Cov}}(\\tilde{Y}, \\tilde{T})}
                          {\\widehat{\\mathrm{Var}}(\\tilde{T})}

donde :math:`\\tilde{Y}` y :math:`\\tilde{T}` son los residuos fuera de muestra
de :math:`Y` y :math:`T` sobre el conjunto de ajuste :math:`Z`. La
ortogonalización elimina el sesgo de regularización de primer orden y el
cross-fitting el sesgo de sobreajuste, entregando convergencia raíz-n aun con
aprendices no paramétricos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from .config import DMLConfig


@dataclass
class DMLEstimate:
    """Estimación puntual del ATE con su inferencia asintótica."""

    theta: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    adjustment_set: List[str]

    def as_dict(self) -> dict:
        return {
            "ATE": round(self.theta, 4),
            "SE": round(self.std_error, 4),
            "IC95_inf": round(self.ci_low, 4),
            "IC95_sup": round(self.ci_high, 4),
            "n": self.n_obs,
            "conjunto_ajuste": self.adjustment_set,
        }


def backdoor_adjustment_set(
    dag: nx.DiGraph,
    treatment: str,
    outcome: Optional[str] = None,
) -> List[str]:
    """Conjunto de ajuste por el criterio de la puerta trasera.

    Se usan los **padres del tratamiento**: Pearl (2009) demuestra que este
    conjunto siempre satisface el criterio backdoor en un DAG causal, propiedad
    que permite derivar los confusores automáticamente de la estructura
    descubierta sin intervención humana.

    Si se indica `outcome`, la variable de resultado se excluye explícitamente.
    Esta salvaguarda no es cosmética: un método de descubrimiento que oriente
    mal la arista tratamiento-resultado colocaría Y entre los "padres" de T, y
    condicionar sobre el propio resultado anula la varianza residual y degenera
    la estimación. Excluirlo replica lo que haría cualquier analista y permite
    que la comparación entre métodos siga siendo informativa; la contaminación
    estructural se reporta aparte en `metrics.adjustment_set_quality`.
    """
    if treatment not in dag:
        return []
    parents = [p for p in dag.predecessors(treatment) if p != outcome]
    return sorted(parents)


def dml_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    seed: int = 42,
    config: Optional[DMLConfig] = None,
) -> DMLEstimate:
    """Estima el ATE mediante DML parcialmente lineal con cross-fitting."""
    cfg = config or DMLConfig()
    adjustment = list(adjustment)

    Z = data[adjustment].to_numpy(dtype=float) if adjustment else np.zeros((len(data), 1))
    T = data[treatment].to_numpy(dtype=float)
    Y = data[outcome].to_numpy(dtype=float)

    res_y = np.zeros_like(Y)
    res_t = np.zeros_like(T)
    splitter = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in splitter.split(Z):
        model_y = RandomForestRegressor(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf, random_state=seed, n_jobs=cfg.n_jobs)
        model_t = RandomForestRegressor(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf, random_state=seed, n_jobs=cfg.n_jobs)
        model_y.fit(Z[train_idx], Y[train_idx])
        model_t.fit(Z[train_idx], T[train_idx])
        res_y[test_idx] = Y[test_idx] - model_y.predict(Z[test_idx])
        res_t[test_idx] = T[test_idx] - model_t.predict(Z[test_idx])

    var_t = float(np.var(res_t))
    if var_t < 1e-12:  # pragma: no cover - tratamiento degenerado
        return DMLEstimate(0.0, float("inf"), float("-inf"), float("inf"),
                           len(data), adjustment)

    theta = float(np.cov(res_y, res_t)[0, 1] / var_t)

    # Error estándar por la teoría de momentos de Neyman (varianza sandwich).
    psi = (res_y - theta * res_t) * res_t
    n = len(T)
    std_error = float(np.sqrt(np.mean(psi**2) / (np.mean(res_t**2) ** 2)) / np.sqrt(n))

    return DMLEstimate(
        theta=theta,
        std_error=std_error,
        ci_low=theta - 1.96 * std_error,
        ci_high=theta + 1.96 * std_error,
        n_obs=n,
        adjustment_set=adjustment,
    )
