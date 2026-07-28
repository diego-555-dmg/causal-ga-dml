"""Pruebas de reproducibilidad extremo a extremo.

Verifican la propiedad que sostiene todo el proyecto: fijada la semilla, el
pipeline completo devuelve exactamente los mismos números.
"""

import pytest

from causal_ga_dml.config import load_config
from causal_ga_dml.data import simulate
from causal_ga_dml.experiment import run_single
from causal_ga_dml.seeds import seed_sequence, set_global_seed


def _config_rapida():
    cfg = load_config()
    cfg.data.n_samples = 600
    cfg.ga.population_size = 6
    cfg.ga.n_generations = 4
    cfg.dml.n_estimators = 25
    cfg.dml.n_jobs = 1
    cfg.baselines.run_hill_climbing = False
    cfg.baselines.run_pc = False
    return cfg


def test_semillas_hijas_son_deterministas_y_distintas():
    a = seed_sequence(42, 5)
    assert a == seed_sequence(42, 5)
    assert len(set(a)) == 5
    assert a != seed_sequence(7, 5)


def test_la_simulacion_es_reproducible():
    d1, g1 = simulate("alarm", 300, seed=42)
    d2, g2 = simulate("alarm", 300, seed=42)
    assert d1.equals(d2)
    assert set(g1.edges()) == set(g2.edges())


def test_semillas_distintas_generan_datos_distintos():
    d1, _ = simulate("alarm", 300, seed=1)
    d2, _ = simulate("alarm", 300, seed=2)
    assert not d1.equals(d2)


def test_alarm_tiene_37_nodos_y_46_aristas():
    _, dag = simulate("alarm", 100, seed=0)
    assert dag.number_of_nodes() == 37
    assert dag.number_of_edges() == 46


def test_set_global_seed_devuelve_un_generador_reproducible():
    r1 = set_global_seed(3).normal(size=5)
    r2 = set_global_seed(3).normal(size=5)
    assert (r1 == r2).all()


@pytest.mark.slow
def test_el_pipeline_completo_es_reproducible():
    cfg = _config_rapida()
    a = run_single(cfg, seed=99, verbose=False)
    b = run_single(cfg, seed=99, verbose=False)
    assert a["metodos"]["AG"]["estimacion"]["ATE"] == b["metodos"]["AG"]["estimacion"]["ATE"]
    assert a["metodos"]["AG"]["estructura"]["SHD"] == b["metodos"]["AG"]["estructura"]["SHD"]
    assert a["ag"]["mejor_bic"] == b["ag"]["mejor_bic"]
