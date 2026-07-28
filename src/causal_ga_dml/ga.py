"""Algoritmo genético basado en ordenamientos para el aprendizaje de DAG.

Representación
--------------
Cada individuo es una **permutación** de los nodos que define un orden
topológico. Dado ese orden, los padres de cada nodo se eligen por selección
avara hacia adelante entre sus predecesores. La aciclicidad queda garantizada
*por construcción*, lo que elimina el costo de reparar u ordenar grafos
inválidos —principal cuello de botella de las representaciones matriciales— y
reduce el espacio de búsqueda de :math:`O(3^{p^2})` grafos a :math:`p!`
ordenamientos (Larrañaga et al., 1996).

Operadores
----------
* Cruce: Order Crossover (OX), que preserva la validez de la permutación.
* Mutación: intercambio de dos posiciones (swap).
* Selección: torneo de tamaño `k` con elitismo.

Reproducibilidad
----------------
Toda la aleatoriedad del AG proviene de una única instancia local
``random.Random(seed)``; no se usa el estado global del módulo ``random``, de
modo que la ejecución es insensible a lo que hagan otras partes del programa.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .config import GAConfig
from .scoring import GaussianBICScore


# --------------------------------------------------------------------------- #
# Decodificación orden -> DAG
# --------------------------------------------------------------------------- #
def greedy_parents_from_order(
    order: Sequence[str],
    score: GaussianBICScore,
    max_indegree: int = 4,
) -> Tuple[Dict[str, List[str]], float]:
    """Selecciona por avidez el mejor conjunto de padres de cada nodo.

    Para cada nodo se añade iterativamente el predecesor que más incrementa el
    score local, hasta que ninguna incorporación mejore o se alcance
    `max_indegree`. Devuelve el mapa de padres y el score total del DAG.
    """
    parents: Dict[str, List[str]] = {}
    total = 0.0

    for position, node in enumerate(order):
        predecessors = order[:position]
        chosen: List[str] = []
        best = score.local_score(node, chosen)

        improving = True
        while improving and len(chosen) < max_indegree:
            improving = False
            best_candidate, best_candidate_score = None, best
            for candidate in predecessors:
                if candidate in chosen:
                    continue
                s = score.local_score(node, chosen + [candidate])
                if s > best_candidate_score + 1e-10:
                    best_candidate, best_candidate_score = candidate, s
            if best_candidate is not None:
                chosen.append(best_candidate)
                best = best_candidate_score
                improving = True

        parents[node] = chosen
        total += best

    return parents, float(total)


def parents_to_dag(parents: Dict[str, Sequence[str]], nodes: Sequence[str]) -> nx.DiGraph:
    """Convierte un mapa nodo -> padres en un ``networkx.DiGraph``."""
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for node, pa in parents.items():
        for p in pa:
            dag.add_edge(p, node)
    return dag


# --------------------------------------------------------------------------- #
# Operadores genéticos
# --------------------------------------------------------------------------- #
def order_crossover(parent_a: Sequence[str], parent_b: Sequence[str],
                    rng: random.Random) -> List[str]:
    """Order Crossover (OX): hereda un segmento de A y completa en el orden de B."""
    size = len(parent_a)
    a, b = sorted(rng.sample(range(size), 2))
    child: List[Optional[str]] = [None] * size
    child[a : b + 1] = list(parent_a[a : b + 1])
    segment = set(child[a : b + 1])
    rest = [g for g in parent_b if g not in segment]
    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = rest[j]
            j += 1
    return [g for g in child if g is not None]


def swap_mutation(order: Sequence[str], p_mutation: float,
                  rng: random.Random) -> List[str]:
    """Mutación por intercambio de dos posiciones con probabilidad `p_mutation`."""
    mutated = list(order)
    if rng.random() < p_mutation:
        i, j = rng.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def tournament_selection(population: Sequence[Sequence[str]],
                         fitness: Sequence[float],
                         k: int,
                         rng: random.Random) -> List[str]:
    """Selección por torneo: gana el individuo de mayor aptitud entre `k` sorteados."""
    aspirants = rng.sample(range(len(population)), k)
    winner = max(aspirants, key=lambda i: fitness[i])
    return list(population[winner])


# --------------------------------------------------------------------------- #
# Bucle evolutivo
# --------------------------------------------------------------------------- #
@dataclass
class GAResult:
    """Salida completa de una ejecución del algoritmo genético."""

    dag: nx.DiGraph
    best_order: List[str]
    best_fitness: float
    history: List[float] = field(default_factory=list)
    mean_history: List[float] = field(default_factory=list)
    generations_run: int = 0
    n_score_evaluations: int = 0


def run_ga(
    score: GaussianBICScore,
    nodes: Sequence[str],
    config: GAConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> GAResult:
    """Ejecuta el AG order-based y devuelve el mejor DAG encontrado."""
    cfg = config or GAConfig()
    rng = random.Random(seed)
    nodes = list(nodes)

    population: List[List[str]] = []
    for _ in range(cfg.population_size):
        individual = nodes[:]
        rng.shuffle(individual)
        population.append(individual)

    def fitness_of(order: Sequence[str]) -> float:
        return greedy_parents_from_order(order, score, cfg.max_indegree)[1]

    best_order: List[str] = list(population[0])
    best_fitness = -np.inf
    history: List[float] = []
    mean_history: List[float] = []
    stagnant = 0
    generations_run = 0

    for generation in range(cfg.n_generations):
        fitness = [fitness_of(ind) for ind in population]
        order_idx = np.argsort(fitness)[::-1]
        population = [population[i] for i in order_idx]
        fitness = [fitness[i] for i in order_idx]

        if fitness[0] > best_fitness + 1e-9:
            best_fitness = fitness[0]
            best_order = list(population[0])
            stagnant = 0
        else:
            stagnant += 1

        history.append(best_fitness)
        mean_history.append(float(np.mean(fitness)))
        generations_run = generation + 1

        if verbose and (generation + 1) % 10 == 0:
            print(f"    generación {generation + 1:3d} | mejor BIC = {best_fitness:.2f} "
                  f"| media = {mean_history[-1]:.2f}")

        if (cfg.early_stopping_patience is not None
                and stagnant >= cfg.early_stopping_patience):
            if verbose:
                print(f"    parada temprana en la generación {generation + 1}")
            break

        # --- nueva generación --------------------------------------------- #
        offspring = [list(population[i]) for i in range(cfg.elitism)]
        while len(offspring) < cfg.population_size:
            p1 = tournament_selection(population, fitness, cfg.tournament_size, rng)
            p2 = tournament_selection(population, fitness, cfg.tournament_size, rng)
            child = (order_crossover(p1, p2, rng)
                     if rng.random() < cfg.p_crossover else list(p1))
            offspring.append(swap_mutation(child, cfg.p_mutation, rng))
        population = offspring

    parents, _ = greedy_parents_from_order(best_order, score, cfg.max_indegree)
    return GAResult(
        dag=parents_to_dag(parents, nodes),
        best_order=best_order,
        best_fitness=float(best_fitness),
        history=history,
        mean_history=mean_history,
        generations_run=generations_run,
        n_score_evaluations=score.n_evaluations,
    )
