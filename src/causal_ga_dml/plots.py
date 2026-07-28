"""Figuras del artículo y de la tesis.

Todas las figuras se generan con el backend no interactivo ``Agg`` y se guardan
en PNG a 300 ppp (requisito habitual de las revistas SciELO) y en PDF vectorial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {
    "tratamiento": "#E76F51",
    "resultado": "#2A9D8F",
    "nodo": "#ADD8E6",
    "arista": "#8A8A8A",
    "acierto": "#2A9D8F",
    "error": "#E76F51",
}


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dag_comparison(true_dag: nx.DiGraph, est_dag: nx.DiGraph,
                        treatment: str, outcome: str,
                        path: str | Path,
                        titles: Sequence[str] = ("DAG verdadero (ALARM)",
                                                 "DAG recuperado por el AG")) -> Path:
    """DAG verdadero frente al recuperado; en verde las aristas acertadas."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    pos = nx.spring_layout(true_dag, k=1.2, iterations=200, seed=42)
    highlight = {treatment: PALETTE["tratamiento"], outcome: PALETTE["resultado"]}
    true_skeleton = {frozenset(e) for e in true_dag.edges()}

    for ax, graph, title in zip(axes, (true_dag, est_dag), titles):
        colors = [highlight.get(n, PALETTE["nodo"]) for n in graph.nodes()]
        edge_colors = [
            PALETTE["acierto"] if frozenset(e) in true_skeleton else PALETTE["error"]
            for e in graph.edges()
        ] if graph is est_dag else [PALETTE["arista"]] * graph.number_of_edges()

        nx.draw_networkx_nodes(graph, pos, node_size=900, node_color=colors,
                               edgecolors="gray", ax=ax)
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrowsize=12,
                               node_size=900, width=1.4,
                               connectionstyle="arc3,rad=0.05", ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=6, font_weight="bold", ax=ax)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, path)


def plot_convergence(history: Sequence[float], mean_history: Sequence[float],
                     path: str | Path) -> Path:
    """Curva de convergencia del AG: mejor individuo frente a media poblacional."""
    fig, ax = plt.subplots(figsize=(8, 5))
    generations = np.arange(1, len(history) + 1)
    ax.plot(generations, history, color="#264653", lw=2, label="Mejor individuo")
    ax.plot(generations, mean_history, color="#E9C46A", lw=2, ls="--",
            label="Media poblacional")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Score BIC gaussiano")
    ax.set_title("Convergencia del algoritmo genético")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_method_comparison(summary: Dict[str, dict], path: str | Path,
                           metric: str = "f1_esqueleto",
                           label: str = "F1 del esqueleto") -> Path:
    """Barras con media ± DE de una métrica estructural por método."""
    methods = list(summary.keys())
    means = [summary[m]["estructura"][metric]["media"] for m in methods]
    stds = [summary[m]["estructura"][metric]["de"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2A9D8F" if m == "AG" else "#A8DADC" for m in methods]
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors,
           edgecolor="#264653", linewidth=1.1)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.015, f"{mean:.3f}", ha="center", fontsize=10)
    ax.set_ylabel(label)
    ax.set_title(f"{label} por método (media ± DE sobre réplicas)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_ate_distribution(runs: List[dict], path: str | Path,
                          methods: Sequence[str] = ("AG", "Hill-Climbing", "PC")) -> Path:
    """Distribución del ATE por método frente al valor del DAG verdadero."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [[r["metodos"][m]["estimacion"]["ATE"] for r in runs]
            for m in methods if m in runs[0]["metodos"]]
    labels = [m for m in methods if m in runs[0]["metodos"]]
    oracle = float(np.mean([r["referencias"]["ATE_dag_verdadero"]["ATE"] for r in runs]))

    parts = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    for patch, name in zip(parts["boxes"], labels):
        patch.set_facecolor("#2A9D8F" if name == "AG" else "#A8DADC")
        patch.set_alpha(0.85)
    ax.axhline(oracle, color="#E76F51", ls="--", lw=2,
               label=f"ATE con DAG verdadero = {oracle:.3f}")
    ax.set_ylabel("ATE estimado (CO → BP)")
    ax.set_title("Distribución del efecto estimado sobre réplicas independientes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)
