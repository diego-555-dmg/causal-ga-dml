"""Pruebas de las métricas estructurales sobre grafos con respuesta conocida."""

import networkx as nx
import pytest

from causal_ga_dml.metrics import (
    adjustment_set_quality, evaluate_structure, structural_hamming_distance,
)


@pytest.fixture
def true_dag() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    return g


def test_shd_es_cero_para_grafos_identicos(true_dag):
    assert structural_hamming_distance(true_dag, true_dag.copy()) == 0


def test_shd_cuenta_arista_invertida(true_dag):
    estimado = nx.DiGraph()
    estimado.add_edges_from([("Z", "T"), ("Z", "Y"), ("Y", "T")])
    assert structural_hamming_distance(true_dag, estimado) == 1


def test_shd_cuenta_faltante_y_sobrante(true_dag):
    estimado = nx.DiGraph()
    estimado.add_edges_from([("Z", "T"), ("Z", "Y")])  # falta T -> Y
    assert structural_hamming_distance(true_dag, estimado) == 1


def test_metricas_perfectas_para_grafo_identico(true_dag):
    m = evaluate_structure(true_dag, true_dag.copy())
    assert m["f1_dirigido"] == 1.0
    assert m["f1_esqueleto"] == 1.0
    assert m["SHD"] == 0


def test_esqueleto_ignora_orientacion(true_dag):
    estimado = nx.DiGraph()
    estimado.add_edges_from([("T", "Z"), ("Y", "Z"), ("T", "Y")])
    m = evaluate_structure(true_dag, estimado)
    assert m["f1_esqueleto"] == 1.0
    assert m["f1_dirigido"] < 1.0


def test_calidad_del_conjunto_de_ajuste_detecta_descendientes(true_dag):
    estimado = nx.DiGraph()
    estimado.add_edges_from([("Z", "T"), ("Y", "T")])  # Y es descendiente de T
    q = adjustment_set_quality(true_dag, estimado, "T")
    assert q["contamina_descendientes"] is True
    assert q["cobertura_confusores"] == 1.0
    assert q["variables_extra"] == ["Y"]
