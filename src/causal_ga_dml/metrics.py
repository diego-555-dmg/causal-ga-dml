"""Métricas de evaluación de la estructura recuperada.

Se reportan por separado las métricas **dirigidas** (que exigen acertar la
orientación) y las del **esqueleto** (que solo exigen acertar la adyacencia).
La distinción es esencial: los puntajes basados en verosimilitud son
equivalentes dentro de una clase de equivalencia de Markov, de modo que la
orientación de algunas aristas no es identificable a partir de datos puramente
observacionales.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import networkx as nx


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def structural_hamming_distance(true_dag: nx.DiGraph, est_dag: nx.DiGraph) -> int:
    """SHD = aristas ausentes + aristas sobrantes + aristas invertidas."""
    true_edges: Set[Tuple[str, str]] = set(true_dag.edges())
    est_edges: Set[Tuple[str, str]] = set(est_dag.edges())

    reversed_edges = sum(
        1 for (u, v) in est_edges if (v, u) in true_edges and (u, v) not in true_edges
    )
    missing = len(true_edges - est_edges - {(v, u) for (u, v) in est_edges})
    extra = len(est_edges - true_edges - {(v, u) for (u, v) in true_edges})
    return int(missing + extra + reversed_edges)


def evaluate_structure(true_dag: nx.DiGraph, est_dag: nx.DiGraph) -> Dict[str, float]:
    """Calcula SHD, precisión, recall y F1 dirigidos y de esqueleto."""
    true_edges = set(true_dag.edges())
    est_edges = set(est_dag.edges())

    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)
    precision, recall, f1 = _prf(tp, fp, fn)

    reversed_edges = sum(
        1 for (u, v) in est_edges if (v, u) in true_edges and (u, v) not in true_edges
    )

    true_skeleton = {frozenset(e) for e in true_edges}
    est_skeleton = {frozenset(e) for e in est_edges}
    tp_s = len(true_skeleton & est_skeleton)
    precision_s, recall_s, f1_s = _prf(
        tp_s, len(est_skeleton) - tp_s, len(true_skeleton) - tp_s
    )

    return {
        "aristas_verdaderas": len(true_edges),
        "aristas_estimadas": len(est_edges),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "invertidas": reversed_edges,
        "SHD": structural_hamming_distance(true_dag, est_dag),
        "precision_dirigida": round(precision, 4),
        "recall_dirigido": round(recall, 4),
        "f1_dirigido": round(f1, 4),
        "precision_esqueleto": round(precision_s, 4),
        "recall_esqueleto": round(recall_s, 4),
        "f1_esqueleto": round(f1_s, 4),
    }


def adjustment_set_quality(
    true_dag: nx.DiGraph, est_dag: nx.DiGraph, treatment: str
) -> Dict[str, object]:
    """Compara el conjunto de ajuste estimado con el del DAG verdadero.

    Un conjunto de ajuste es útil para la identificación si (i) contiene los
    confusores relevantes y (ii) no incluye descendientes del tratamiento, que
    inducirían sesgo de sobreajuste o de colisionador.
    """
    true_set = set(true_dag.predecessors(treatment)) if treatment in true_dag else set()
    est_set = set(est_dag.predecessors(treatment)) if treatment in est_dag else set()
    descendants = nx.descendants(true_dag, treatment) if treatment in true_dag else set()

    return {
        "ajuste_verdadero": sorted(true_set),
        "ajuste_estimado": sorted(est_set),
        "confusores_recuperados": sorted(true_set & est_set),
        "confusores_omitidos": sorted(true_set - est_set),
        "variables_extra": sorted(est_set - true_set),
        "cobertura_confusores": round(len(true_set & est_set) / len(true_set), 4)
        if true_set
        else 1.0,
        "descendientes_incluidos": sorted(est_set & descendants),
        "contamina_descendientes": bool(est_set & descendants),
    }
