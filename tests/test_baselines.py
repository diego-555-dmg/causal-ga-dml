"""Pruebas de los métodos de referencia."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_ga_dml.baselines import hill_climbing, pc_algorithm, random_order_baseline
from causal_ga_dml.scoring import GaussianBICScore


@pytest.fixture
def cadena() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    a = rng.normal(size=1200)
    b = 1.5 * a + 0.25 * rng.normal(size=1200)
    c = 1.5 * b + 0.25 * rng.normal(size=1200)
    d = rng.normal(size=1200)  # variable aislada
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


def test_hill_climbing_devuelve_un_dag(cadena):
    score = GaussianBICScore(cadena)
    g = hill_climbing(score, list(cadena.columns))
    assert nx.is_directed_acyclic_graph(g)


def test_hill_climbing_recupera_el_esqueleto_de_la_cadena(cadena):
    score = GaussianBICScore(cadena)
    g = hill_climbing(score, list(cadena.columns))
    esqueleto = {frozenset(e) for e in g.edges()}
    assert frozenset({"A", "B"}) in esqueleto
    assert frozenset({"B", "C"}) in esqueleto


def test_pc_devuelve_un_dag_y_aisla_la_variable_independiente(cadena):
    g = pc_algorithm(cadena, alpha=0.01, max_cond_set=2)
    assert nx.is_directed_acyclic_graph(g)
    assert g.degree("D") == 0


def test_orden_aleatorio_es_reproducible(cadena):
    score = GaussianBICScore(cadena)
    a = random_order_baseline(score, list(cadena.columns), seed=3)
    b = random_order_baseline(score, list(cadena.columns), seed=3)
    assert set(a.edges()) == set(b.edges())
