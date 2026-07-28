"""Refutadores de robustez (en el espíritu de DoWhy).

Un refutador no valida la hipótesis causal: la somete a una prueba que debería
fallar si el efecto estimado fuese un artefacto del procedimiento. Se aplican
tres, cada uno con un criterio de aprobación explícito y verificable.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DMLConfig
from .dml import dml_ate


def random_common_cause(
    data: pd.DataFrame, treatment: str, outcome: str, adjustment: Sequence[str],
    seed: int = 42, config: Optional[DMLConfig] = None,
) -> float:
    """Añade un confusor irrelevante: el efecto no debería cambiar."""
    rng = np.random.default_rng(seed)
    augmented = data.copy()
    augmented["_causa_comun_aleatoria"] = rng.normal(size=len(data))
    return dml_ate(augmented, treatment, outcome,
                   list(adjustment) + ["_causa_comun_aleatoria"], seed, config).theta


def placebo_treatment(
    data: pd.DataFrame, treatment: str, outcome: str, adjustment: Sequence[str],
    seed: int = 42, config: Optional[DMLConfig] = None,
) -> float:
    """Permuta el tratamiento: el efecto debería colapsar a cero."""
    rng = np.random.default_rng(seed)
    placebo = data.copy()
    placebo[treatment] = rng.permutation(placebo[treatment].to_numpy())
    return dml_ate(placebo, treatment, outcome, adjustment, seed, config).theta


def data_subset(
    data: pd.DataFrame, treatment: str, outcome: str, adjustment: Sequence[str],
    fraction: float = 0.8, seed: int = 42, config: Optional[DMLConfig] = None,
) -> float:
    """Reestima con una submuestra: el efecto debería mantenerse estable."""
    subset = data.sample(frac=fraction, random_state=seed)
    return dml_ate(subset, treatment, outcome, adjustment, seed, config).theta


def run_all_refuters(
    data: pd.DataFrame, treatment: str, outcome: str, adjustment: Sequence[str],
    baseline_theta: float, seed: int = 42, config: Optional[DMLConfig] = None,
    tolerance: float = 0.15,
) -> Dict[str, object]:
    """Ejecuta los tres refutadores y evalúa los criterios de aprobación.

    Criterios
    ---------
    * Causa común aleatoria: |Δ relativo| <= `tolerance`.
    * Tratamiento placebo: |efecto| <= `tolerance` * |efecto base|.
    * Submuestra 80 %: |Δ relativo| <= `tolerance`.
    """
    rcc = random_common_cause(data, treatment, outcome, adjustment, seed, config)
    placebo = placebo_treatment(data, treatment, outcome, adjustment, seed, config)
    subset = data_subset(data, treatment, outcome, adjustment, 0.8, seed, config)

    denominator = abs(baseline_theta) if abs(baseline_theta) > 1e-9 else 1.0
    return {
        "efecto_base": round(baseline_theta, 4),
        "causa_comun_aleatoria": round(rcc, 4),
        "tratamiento_placebo": round(placebo, 4),
        "submuestra_80": round(subset, 4),
        "desvio_relativo_causa_comun": round(abs(rcc - baseline_theta) / denominator, 4),
        "desvio_relativo_submuestra": round(abs(subset - baseline_theta) / denominator, 4),
        "aprueba_causa_comun": bool(abs(rcc - baseline_theta) / denominator <= tolerance),
        "aprueba_placebo": bool(abs(placebo) <= tolerance * denominator),
        "aprueba_submuestra": bool(abs(subset - baseline_theta) / denominator <= tolerance),
    }
