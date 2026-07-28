"""Pruebas del score gaussiano BIC."""

import numpy as np
import pandas as pd
import pytest

from causal_ga_dml.scoring import GaussianBICScore


@pytest.fixture
def datos() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(size=800)
    b = 2.0 * a + 0.3 * rng.normal(size=800)
    c = rng.normal(size=800)  # independiente
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_prefiere_el_padre_verdadero(datos):
    score = GaussianBICScore(datos)
    assert score.local_score("B", ["A"]) > score.local_score("B", [])
    assert score.local_score("B", ["A"]) > score.local_score("B", ["C"])


def test_penaliza_padres_irrelevantes(datos):
    score = GaussianBICScore(datos)
    assert score.local_score("B", ["A"]) > score.local_score("B", ["A", "C"])


def test_varianza_parcial_no_negativa(datos):
    score = GaussianBICScore(datos)
    assert score.residual_variance("B", ["A"]) > 0
    assert score.residual_variance("B", ["A"]) < score.residual_variance("B", [])


def test_memoizacion_es_consistente(datos):
    score = GaussianBICScore(datos)
    primero = score.local_score("B", ["A", "C"])
    segundo = score.local_score("B", ["C", "A"])  # mismo conjunto, otro orden
    assert primero == segundo
    assert score.cache_size == 1
