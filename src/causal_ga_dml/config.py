"""Configuración tipada del experimento (dataclasses + YAML).

Todos los hiperparámetros del pipeline viven aquí y se serializan junto con los
resultados, de modo que cualquier cifra reportada en el artículo o en la tesis
puede rastrearse hasta la configuración exacta que la produjo.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Parámetros de la simulación de datos observacionales."""

    network: str = "alarm"          # red bayesiana de referencia (pgmpy)
    n_samples: int = 5000           # tamaño de la muestra observacional
    treatment: str = "CO"           # gasto cardíaco
    outcome: str = "BP"             # presión arterial


@dataclass
class ScoreConfig:
    """Parámetros del score gaussiano BIC descomponible."""

    penalty: float = 3.5            # factor de penalización de complejidad
    ridge: float = 1e-6             # regularización de la matriz de covarianza


@dataclass
class GAConfig:
    """Hiperparámetros del algoritmo genético basado en ordenamientos."""

    population_size: int = 40
    n_generations: int = 60
    max_indegree: int = 4
    p_crossover: float = 0.90
    p_mutation: float = 0.30
    tournament_size: int = 3
    elitism: int = 2
    early_stopping_patience: Optional[int] = None  # None = sin parada temprana


@dataclass
class DMLConfig:
    """Hiperparámetros del estimador Double Machine Learning."""

    n_splits: int = 2               # particiones del cross-fitting
    n_estimators: int = 100         # árboles del bosque aleatorio
    max_depth: int = 6              # profundidad máxima de cada árbol
    min_samples_leaf: int = 1
    n_jobs: int = 1                 # 1 = determinismo bit a bit entre plataformas


@dataclass
class BaselineConfig:
    """Métodos de referencia contra los que se compara el AG."""

    run_hill_climbing: bool = True
    run_pc: bool = True
    run_random_order: bool = True   # ablación: orden topológico aleatorio
    pc_alpha: float = 0.01          # nivel de significación del test Fisher-z
    pc_max_cond_set: int = 3        # tamaño máximo del conjunto condicionante
    hc_max_iter: int = 200          # iteraciones máximas del hill-climbing


@dataclass
class ExperimentConfig:
    """Configuración completa de un experimento reproducible."""

    seed: int = 42
    n_replications: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    dml: DMLConfig = field(default_factory=DMLConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    output_dir: str = "results"
    figures_dir: str = "figures"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


_SECTIONS = {
    "data": DataConfig,
    "score": ScoreConfig,
    "ga": GAConfig,
    "dml": DMLConfig,
    "baselines": BaselineConfig,
}


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Carga la configuración desde YAML y aplica sobrescrituras puntuales.

    Examples
    --------
    >>> cfg = load_config("configs/default.yaml", seed=7)
    >>> cfg.seed
    7
    """
    raw: Dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS and isinstance(value, dict):
            kwargs[key] = _SECTIONS[key](**value)
        else:
            kwargs[key] = value

    cfg = ExperimentConfig(**kwargs)
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise KeyError(f"Parámetro de configuración desconocido: {key!r}")
        setattr(cfg, key, value)
    return cfg


def environment_report() -> Dict[str, Any]:
    """Registro del entorno de ejecución: hardware y versiones de librerías.

    Se guarda junto a los resultados para documentar el contexto exacto en que
    se obtuvieron las cifras reportadas.
    """
    import platform
    import sys

    def _v(name: str) -> str:
        try:
            module = __import__(name)
            return getattr(module, "__version__", "desconocida")
        except Exception:  # pragma: no cover - entorno incompleto
            return "no instalada"

    try:
        import multiprocessing

        n_cpu: Any = multiprocessing.cpu_count()
    except Exception:  # pragma: no cover
        n_cpu = "desconocido"

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "so": platform.platform(),
        "arquitectura": platform.machine(),
        "procesador": platform.processor() or "no reportado",
        "nucleos_logicos": n_cpu,
        "librerias": {
            "numpy": _v("numpy"),
            "pandas": _v("pandas"),
            "networkx": _v("networkx"),
            "scikit-learn": _v("sklearn"),
            "scipy": _v("scipy"),
            "pgmpy": _v("pgmpy"),
            "matplotlib": _v("matplotlib"),
            "pyyaml": _v("yaml"),
        },
    }
