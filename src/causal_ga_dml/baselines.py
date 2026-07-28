"""Métodos de referencia para contrastar el algoritmo genético.

Se incluyen tres competidores que cubren las tres familias clásicas del
descubrimiento causal, todos evaluados sobre exactamente los mismos datos y,
cuando aplica, con el mismo score, de modo que la comparación aísle el efecto
de la estrategia de búsqueda:

1. **Hill-Climbing con BIC gaussiano** (basado en score, búsqueda local voraz).
2. **PC con test Fisher-z** (basado en restricciones de independencia condicional).
3. **Orden aleatorio + padres avaros** (ablación: mide cuánto aporta la búsqueda
   evolutiva por encima de la mera decodificación orden -> DAG).
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .ga import greedy_parents_from_order, parents_to_dag
from .scoring import GaussianBICScore


# --------------------------------------------------------------------------- #
# 1. Hill-Climbing con el mismo score BIC gaussiano
# --------------------------------------------------------------------------- #
def hill_climbing(
    score: GaussianBICScore,
    nodes: Sequence[str],
    max_indegree: int = 4,
    max_iter: int = 200,
) -> nx.DiGraph:
    """Búsqueda local voraz con operadores de adición, borrado e inversión.

    Parte del grafo vacío y aplica en cada iteración el movimiento que más
    incrementa el score global, verificando aciclicidad. Es el referente
    estándar de los métodos basados en score y el punto de comparación natural
    para evaluar si la búsqueda global del AG evita óptimos locales.
    """
    nodes = list(nodes)
    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    for _ in range(max_iter):
        best_delta, best_move = 1e-8, None

        for u, v in itertools.permutations(nodes, 2):
            # --- adición ------------------------------------------------- #
            if not dag.has_edge(u, v) and not dag.has_edge(v, u):
                if len(parents[v]) < max_indegree and not nx.has_path(dag, v, u):
                    delta = (score.local_score(v, parents[v] + [u])
                             - score.local_score(v, parents[v]))
                    if delta > best_delta:
                        best_delta, best_move = delta, ("add", u, v)

            # --- borrado -------------------------------------------------- #
            elif dag.has_edge(u, v):
                remaining = [p for p in parents[v] if p != u]
                delta = score.local_score(v, remaining) - score.local_score(v, parents[v])
                if delta > best_delta:
                    best_delta, best_move = delta, ("del", u, v)

                # --- inversión -------------------------------------------- #
                if len(parents[u]) < max_indegree:
                    tmp = dag.copy()
                    tmp.remove_edge(u, v)
                    if not nx.has_path(tmp, u, v):
                        delta_rev = (
                            score.local_score(v, remaining)
                            - score.local_score(v, parents[v])
                            + score.local_score(u, parents[u] + [v])
                            - score.local_score(u, parents[u])
                        )
                        if delta_rev > best_delta:
                            best_delta, best_move = delta_rev, ("rev", u, v)

        if best_move is None:
            break

        kind, u, v = best_move
        if kind == "add":
            parents[v].append(u)
            dag.add_edge(u, v)
        elif kind == "del":
            parents[v].remove(u)
            dag.remove_edge(u, v)
        else:  # inversión
            parents[v].remove(u)
            dag.remove_edge(u, v)
            parents[u].append(v)
            dag.add_edge(v, u)

    return dag


# --------------------------------------------------------------------------- #
# 2. Algoritmo PC con test de independencia condicional Fisher-z
# --------------------------------------------------------------------------- #
def _fisher_z_pvalue(corr: np.ndarray, idx: Dict[str, int], n: int,
                     x: str, y: str, cond: Sequence[str]) -> float:
    """p-valor del test Fisher-z de correlación parcial ``rho(x, y | cond)``."""
    variables = [x, y] + list(cond)
    positions = [idx[v] for v in variables]
    sub = corr[np.ix_(positions, positions)]
    try:
        precision = np.linalg.inv(sub)
    except np.linalg.LinAlgError:  # pragma: no cover - matriz singular
        return 1.0
    denominator = np.sqrt(precision[0, 0] * precision[1, 1])
    if denominator <= 0:
        return 1.0
    rho = -precision[0, 1] / denominator
    rho = float(np.clip(rho, -0.999999, 0.999999))
    dof = n - len(cond) - 3
    if dof <= 0:
        return 1.0
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(dof)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def pc_algorithm(
    data: pd.DataFrame,
    alpha: float = 0.01,
    max_cond_set: int = 3,
) -> nx.DiGraph:
    """Algoritmo PC (Spirtes et al., 2000) con test Fisher-z.

    Fase 1: elimina adyacencias mediante independencias condicionales crecientes.
    Fase 2: orienta v-estructuras (colisionadores) y aplica las reglas de Meek.
    Fase 3: extiende el CPDAG resultante a un DAG consistente, necesario para
    derivar un conjunto de ajuste comparable al del AG.
    """
    nodes = list(data.columns)
    idx = {c: i for i, c in enumerate(nodes)}
    n = len(data)
    corr = np.corrcoef(np.asarray(data.values, dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0) + 1e-8 * np.eye(len(nodes))

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(itertools.combinations(nodes, 2))
    sepset: Dict[Tuple[str, str], Set[str]] = {}

    for level in range(max_cond_set + 1):
        for x, y in list(graph.edges()):
            neighbours = [z for z in graph.neighbors(x) if z != y]
            if len(neighbours) < level:
                continue
            for cond in itertools.combinations(neighbours, level):
                if _fisher_z_pvalue(corr, idx, n, x, y, cond) > alpha:
                    graph.remove_edge(x, y)
                    sepset[(x, y)] = set(cond)
                    sepset[(y, x)] = set(cond)
                    break

    # --- Fase 2: v-estructuras ------------------------------------------- #
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    oriented: Set[Tuple[str, str]] = set()
    for z in nodes:
        for x, y in itertools.combinations(list(graph.neighbors(z)), 2):
            if graph.has_edge(x, y):
                continue
            if z not in sepset.get((x, y), {z}):
                oriented.add((x, z))
                oriented.add((y, z))

    for u, v in oriented:
        if (v, u) not in oriented:
            dag.add_edge(u, v)

    # --- Fase 3: extensión consistente del resto de aristas --------------- #
    for x, y in graph.edges():
        if dag.has_edge(x, y) or dag.has_edge(y, x):
            continue
        if not nx.has_path(dag, y, x):
            dag.add_edge(x, y)
        elif not nx.has_path(dag, x, y):
            dag.add_edge(y, x)
    return dag


# --------------------------------------------------------------------------- #
# 3. Ablación: orden topológico aleatorio + selección avara de padres
# --------------------------------------------------------------------------- #
def random_order_baseline(
    score: GaussianBICScore,
    nodes: Sequence[str],
    max_indegree: int = 4,
    seed: int = 42,
    n_restarts: int = 1,
) -> nx.DiGraph:
    """Mejor DAG obtenido a partir de `n_restarts` ordenamientos aleatorios.

    Aísla la contribución de la búsqueda evolutiva: comparte con el AG la
    representación y la decodificación, pero carece de selección, cruce y
    mutación.
    """
    rng = random.Random(seed)
    nodes = list(nodes)
    best_parents, best_fitness = None, -np.inf

    for _ in range(max(1, n_restarts)):
        order = nodes[:]
        rng.shuffle(order)
        parents, fitness = greedy_parents_from_order(order, score, max_indegree)
        if fitness > best_fitness:
            best_parents, best_fitness = parents, fitness

    assert best_parents is not None
    return parents_to_dag(best_parents, nodes)
