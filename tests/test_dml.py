"""Pruebas del estimador DML sobre un modelo con efecto causal conocido."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_ga_dml.config import DMLConfig
from causal_ga_dml.dml import backdoor_adjustment_set, dml_ate
from causal_ga_dml.refuters import placebo_treatment


THETA_VERDADERO = 2.0


@pytest.fixture
def confundido() -> pd.DataFrame:
    """Z confunde T e Y; el efecto causal verdadero de T sobre Y es 2,0."""
    rng = np.random.default_rng(11)
    n = 3000
    z = rng.normal(size=n)
    t = 1.2 * z + rng.normal(size=n)
    y = THETA_VERDADERO * t + 1.8 * z + rng.normal(size=n)
    return pd.DataFrame({"Z": z, "T": t, "Y": y})


@pytest.fixture
def cfg() -> DMLConfig:
    return DMLConfig(n_splits=2, n_estimators=60, max_depth=6, n_jobs=1)


def test_recupera_el_efecto_al_ajustar_por_el_confusor(confundido, cfg):
    est = dml_ate(confundido, "T", "Y", ["Z"], seed=0, config=cfg)
    assert est.theta == pytest.approx(THETA_VERDADERO, abs=0.12)


def test_sin_ajuste_el_estimador_esta_sesgado(confundido, cfg):
    sesgado = dml_ate(confundido, "T", "Y", [], seed=0, config=cfg).theta
    insesgado = dml_ate(confundido, "T", "Y", ["Z"], seed=0, config=cfg).theta
    assert abs(sesgado - THETA_VERDADERO) > abs(insesgado - THETA_VERDADERO)


def test_intervalo_de_confianza_cubre_el_valor_verdadero(confundido, cfg):
    est = dml_ate(confundido, "T", "Y", ["Z"], seed=0, config=cfg)
    assert est.ci_low < THETA_VERDADERO < est.ci_high


def test_placebo_colapsa_a_cero(confundido, cfg):
    theta = placebo_treatment(confundido, "T", "Y", ["Z"], seed=0, config=cfg)
    assert abs(theta) < 0.1


def test_conjunto_de_ajuste_son_los_padres_del_tratamiento():
    g = nx.DiGraph()
    g.add_edges_from([("Z", "T"), ("W", "T"), ("T", "Y")])
    assert backdoor_adjustment_set(g, "T") == ["W", "Z"]


def test_conjunto_de_ajuste_excluye_el_resultado():
    g = nx.DiGraph()
    g.add_edges_from([("Z", "T"), ("Y", "T")])  # orientación errónea Y -> T
    assert backdoor_adjustment_set(g, "T", outcome="Y") == ["Z"]


def test_es_determinista_con_la_misma_semilla(confundido, cfg):
    a = dml_ate(confundido, "T", "Y", ["Z"], seed=5, config=cfg).theta
    b = dml_ate(confundido, "T", "Y", ["Z"], seed=5, config=cfg).theta
    assert a == b
