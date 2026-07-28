"""Simulación de datos observacionales a partir de una red bayesiana de referencia.

La red ALARM (Beinlich et al., 1989) actúa como *verdad fundamental*: su DAG es
conocido, lo que permite evaluar objetivamente la calidad de la estructura
recuperada, algo imposible con datos reales.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import networkx as nx
import pandas as pd


def load_true_dag(network: str = "alarm") -> nx.DiGraph:
    """Devuelve el DAG verdadero de una red bayesiana de referencia de pgmpy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  # pgmpy >= 1.2
            from pgmpy.example_models import load_model  # type: ignore

            model = load_model(network)
        except Exception:  # pgmpy < 1.2
            from pgmpy.utils import get_example_model

            model = get_example_model(network)

    dag = nx.DiGraph()
    dag.add_nodes_from(model.nodes())
    dag.add_edges_from(model.edges())
    dag.graph["pgmpy_model"] = model
    return dag


def simulate(
    network: str = "alarm",
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Simula `n_samples` observaciones por muestreo directo de la conjunta.

    Las variables categóricas se codifican ordinalmente como enteros. Esta es la
    aproximación estándar para aplicar puntajes gaussianos escalables al
    aprendizaje de estructura sobre datos ordinales; su impacto se discute
    explícitamente entre las limitaciones del estudio.

    Returns
    -------
    (DataFrame, DiGraph)
        Muestra numérica y DAG verdadero.
    """
    dag = load_true_dag(network)
    model = dag.graph["pgmpy_model"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = model.simulate(n_samples=n_samples, seed=seed, show_progress=False)

    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = df_num[col].astype("category").cat.codes.astype(float)

    # Orden de columnas determinista: evita que un cambio de orden interno de
    # pgmpy altere los resultados entre versiones.
    df_num = df_num.reindex(sorted(df_num.columns), axis=1)
    return df_num, dag
