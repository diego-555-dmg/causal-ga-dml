"""Pruebas del algoritmo genético y de sus operadores."""

import random

import numpy as np
import pandas as pd
import pytest

from causal_ga_dml.config import GAConfig
from causal_ga_dml.ga import (
    greedy_parents_from_order, order_crossover, run_ga, swap_mutation,
)
from causal_ga_dml.scoring import GaussianBICScore


@pytest.fixture
def cadena() -> pd.DataFrame:
    """Cadena causal A -> B -> C con ruido bajo."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=1000)
    b = 1.5 * a + 0.2 * rng.normal(size=1000)
    c = 1.5 * b + 0.2 * rng.normal(size=1000)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_order_crossover_devuelve_permutacion_valida():
    rng = random.Random(0)
    p1, p2 = list("ABCDEF"), list("FEDCBA")
    hijo = order_crossover(p1, p2, rng)
    assert sorted(hijo) == sorted(p1)
    assert len(hijo) == len(p1)


def test_mutacion_preserva_los_elementos():
    rng = random.Random(0)
    orden = list("ABCDE")
    mutado = swap_mutation(orden, 1.0, rng)
    assert sorted(mutado) == sorted(orden)


def test_mutacion_nula_no_cambia_nada():
    orden = list("ABCDE")
    assert swap_mutation(orden, 0.0, random.Random(0)) == orden


def test_padres_avaros_respetan_el_orden_topologico(cadena):
    score = GaussianBICScore(cadena)
    parents, total = greedy_parents_from_order(["A", "B", "C"], score)
    assert parents["A"] == []                 # no tiene predecesores
    assert "A" in parents["B"]                # descubre A -> B
    assert np.isfinite(total)


def test_max_indegree_se_respeta(cadena):
    score = GaussianBICScore(cadena)
    parents, _ = greedy_parents_from_order(["A", "B", "C"], score, max_indegree=1)
    assert all(len(p) <= 1 for p in parents.values())


def test_el_dag_resultante_es_aciclico(cadena):
    import networkx as nx

    score = GaussianBICScore(cadena)
    cfg = GAConfig(population_size=6, n_generations=4)
    resultado = run_ga(score, list(cadena.columns), cfg, seed=1, verbose=False)
    assert nx.is_directed_acyclic_graph(resultado.dag)


def test_la_aptitud_nunca_decrece(cadena):
    score = GaussianBICScore(cadena)
    cfg = GAConfig(population_size=8, n_generations=6)
    historia = run_ga(score, list(cadena.columns), cfg, seed=1, verbose=False).history
    assert all(b >= a - 1e-9 for a, b in zip(historia, historia[1:]))


def test_misma_semilla_mismo_resultado(cadena):
    score_a = GaussianBICScore(cadena)
    score_b = GaussianBICScore(cadena)
    cfg = GAConfig(population_size=8, n_generations=5)
    r1 = run_ga(score_a, list(cadena.columns), cfg, seed=123, verbose=False)
    r2 = run_ga(score_b, list(cadena.columns), cfg, seed=123, verbose=False)
    assert r1.best_order == r2.best_order
    assert r1.best_fitness == pytest.approx(r2.best_fitness)
