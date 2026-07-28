"""Orquestación del experimento: corrida única y estudio Monte Carlo multi-semilla."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .baselines import hill_climbing, pc_algorithm, random_order_baseline
from .config import ExperimentConfig, environment_report
from .data import simulate
from .dml import backdoor_adjustment_set, dml_ate
from .ga import run_ga
from .metrics import adjustment_set_quality, evaluate_structure
from .refuters import run_all_refuters
from .scoring import GaussianBICScore
from .seeds import seed_sequence, set_global_seed


# --------------------------------------------------------------------------- #
def run_single(config: ExperimentConfig, seed: Optional[int] = None,
               verbose: bool = True) -> Dict[str, Any]:
    """Ejecuta el pipeline completo para una semilla y devuelve sus resultados."""
    seed = config.seed if seed is None else seed
    set_global_seed(seed)
    started = time.perf_counter()

    treatment, outcome = config.data.treatment, config.data.outcome

    if verbose:
        print(f"[1/6] Simulando {config.data.n_samples} observaciones de "
              f"'{config.data.network}' (semilla {seed})...")
    data, true_dag = simulate(config.data.network, config.data.n_samples, seed)

    score = GaussianBICScore(data, config.score.penalty, config.score.ridge)
    nodes = list(data.columns)

    if verbose:
        print(f"[2/6] Algoritmo genético (población {config.ga.population_size}, "
              f"{config.ga.n_generations} generaciones)...")
    t0 = time.perf_counter()
    ga_result = run_ga(score, nodes, config.ga, seed, verbose)
    ga_seconds = time.perf_counter() - t0

    methods: Dict[str, nx.DiGraph] = {"AG": ga_result.dag}
    timings: Dict[str, float] = {"AG": ga_seconds}

    if verbose:
        print("[3/6] Ejecutando métodos de referencia...")
    if config.baselines.run_hill_climbing:
        t0 = time.perf_counter()
        methods["Hill-Climbing"] = hill_climbing(
            score, nodes, config.ga.max_indegree, config.baselines.hc_max_iter)
        timings["Hill-Climbing"] = time.perf_counter() - t0
    if config.baselines.run_pc:
        t0 = time.perf_counter()
        methods["PC"] = pc_algorithm(
            data, config.baselines.pc_alpha, config.baselines.pc_max_cond_set)
        timings["PC"] = time.perf_counter() - t0
    if config.baselines.run_random_order:
        t0 = time.perf_counter()
        methods["Orden aleatorio"] = random_order_baseline(
            score, nodes, config.ga.max_indegree, seed, n_restarts=1)
        timings["Orden aleatorio"] = time.perf_counter() - t0

    if verbose:
        print("[4/6] Evaluando estructura y estimando efectos causales...")
    per_method: Dict[str, Any] = {}
    for name, dag in methods.items():
        adjustment = backdoor_adjustment_set(dag, treatment, outcome)
        estimate = dml_ate(data, treatment, outcome, adjustment, seed, config.dml)
        per_method[name] = {
            "estructura": evaluate_structure(true_dag, dag),
            "conjunto_ajuste": adjustment_set_quality(true_dag, dag, treatment),
            "estimacion": estimate.as_dict(),
            "segundos": round(timings[name], 2),
        }

    # --- referencias: DAG verdadero y modelo sin ajuste ------------------- #
    true_adjustment = backdoor_adjustment_set(true_dag, treatment, outcome)
    oracle = dml_ate(data, treatment, outcome, true_adjustment, seed, config.dml)
    naive = dml_ate(data, treatment, outcome, [], seed, config.dml)

    ga_theta = per_method["AG"]["estimacion"]["ATE"]

    if verbose:
        print("[5/6] Ejecutando refutadores de robustez sobre el ajuste del AG...")
    robustness = run_all_refuters(
        data, treatment, outcome,
        backdoor_adjustment_set(ga_result.dag, treatment, outcome),
        ga_theta, seed, config.dml)

    if verbose:
        print("[6/6] Consolidando resultados...")

    return {
        "semilla": seed,
        "segundos_totales": round(time.perf_counter() - started, 2),
        "datos": {
            "red": config.data.network,
            "n_muestras": config.data.n_samples,
            "n_nodos": true_dag.number_of_nodes(),
            "n_aristas_verdaderas": true_dag.number_of_edges(),
            "tratamiento": treatment,
            "resultado": outcome,
        },
        "ag": {
            "mejor_bic": round(ga_result.best_fitness, 2),
            "generaciones": ga_result.generations_run,
            "evaluaciones_score": ga_result.n_score_evaluations,
            "historia_mejor": [round(h, 2) for h in ga_result.history],
            "historia_media": [round(h, 2) for h in ga_result.mean_history],
        },
        "metodos": per_method,
        "referencias": {
            "ATE_dag_verdadero": oracle.as_dict(),
            "ATE_sin_ajuste": naive.as_dict(),
            "sesgo_AG_vs_oraculo": round(ga_theta - oracle.theta, 4),
            "sesgo_sin_ajuste_vs_oraculo": round(naive.theta - oracle.theta, 4),
        },
        "robustez": robustness,
        "_dags": {name: sorted(dag.edges()) for name, dag in methods.items()},
        "_dag_verdadero": sorted(true_dag.edges()),
    }


# --------------------------------------------------------------------------- #
def _aggregate(values: List[float]) -> Dict[str, float]:
    """Media, desviación estándar, mediana e IC 95 % (normal) de una réplica."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    return {
        "media": round(mean, 4),
        "de": round(std, 4),
        "mediana": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "IC95_inf": round(mean - 1.96 * se, 4),
        "IC95_sup": round(mean + 1.96 * se, 4),
        "n_replicas": n,
    }


def run_multiseed(config: ExperimentConfig, n_replications: Optional[int] = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """Estudio Monte Carlo: repite el pipeline con semillas independientes.

    Reportar media ± desviación estándar sobre réplicas independientes evita
    que una conclusión dependa de una única realización afortunada del muestreo
    y del proceso evolutivo, requisito básico de rigor en estudios de simulación.
    """
    n_reps = n_replications or config.n_replications
    seeds = seed_sequence(config.seed, n_reps)
    runs: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"\n===== Réplica {i}/{n_reps} (semilla {seed}) =====")
        runs.append(run_single(config, seed, verbose=False))
        if verbose:
            ag = runs[-1]["metodos"]["AG"]
            print(f"  AG: F1 esqueleto = {ag['estructura']['f1_esqueleto']:.3f} | "
                  f"SHD = {ag['estructura']['SHD']} | "
                  f"ATE = {ag['estimacion']['ATE']:.4f} | "
                  f"{runs[-1]['segundos_totales']:.0f} s")

    method_names = list(runs[0]["metodos"].keys())
    structure_keys = ["f1_esqueleto", "recall_esqueleto", "precision_esqueleto",
                      "f1_dirigido", "SHD", "aristas_estimadas", "invertidas"]

    summary: Dict[str, Any] = {"por_metodo": {}}
    for name in method_names:
        summary["por_metodo"][name] = {
            "estructura": {
                key: _aggregate([r["metodos"][name]["estructura"][key] for r in runs])
                for key in structure_keys
            },
            "ATE": _aggregate([r["metodos"][name]["estimacion"]["ATE"] for r in runs]),
            "cobertura_confusores": _aggregate(
                [r["metodos"][name]["conjunto_ajuste"]["cobertura_confusores"]
                 for r in runs]),
            "segundos": _aggregate([r["metodos"][name]["segundos"] for r in runs]),
            "sesgo_absoluto_vs_oraculo": _aggregate(
                [abs(r["metodos"][name]["estimacion"]["ATE"]
                     - r["referencias"]["ATE_dag_verdadero"]["ATE"]) for r in runs]),
        }

    summary["referencias"] = {
        "ATE_dag_verdadero": _aggregate(
            [r["referencias"]["ATE_dag_verdadero"]["ATE"] for r in runs]),
        "ATE_sin_ajuste": _aggregate(
            [r["referencias"]["ATE_sin_ajuste"]["ATE"] for r in runs]),
    }
    summary["robustez"] = {
        key: _aggregate([r["robustez"][key] for r in runs])
        for key in ["efecto_base", "causa_comun_aleatoria",
                    "tratamiento_placebo", "submuestra_80"]
    }
    summary["tasa_aprobacion_refutadores"] = {
        key: round(float(np.mean([r["robustez"][key] for r in runs])), 4)
        for key in ["aprueba_causa_comun", "aprueba_placebo", "aprueba_submuestra"]
    }

    return {
        "configuracion": config.to_dict(),
        "entorno": environment_report(),
        "n_replicas": n_reps,
        "semillas": seeds,
        "resumen": summary,
        "corridas": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
    }


# --------------------------------------------------------------------------- #
def save_results(results: Dict[str, Any], path: str | Path) -> Path:
    """Serializa los resultados a JSON con codificación UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                               default=str), encoding="utf-8")
    return path


def summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Tabla comparativa lista para el artículo (una fila por método)."""
    rows = []
    for name, block in results["resumen"]["por_metodo"].items():
        rows.append({
            "Método": name,
            "F1 esqueleto": f"{block['estructura']['f1_esqueleto']['media']:.3f} "
                            f"± {block['estructura']['f1_esqueleto']['de']:.3f}",
            "Recall esqueleto": f"{block['estructura']['recall_esqueleto']['media']:.3f} "
                                f"± {block['estructura']['recall_esqueleto']['de']:.3f}",
            "F1 dirigido": f"{block['estructura']['f1_dirigido']['media']:.3f} "
                           f"± {block['estructura']['f1_dirigido']['de']:.3f}",
            "SHD": f"{block['estructura']['SHD']['media']:.1f} "
                   f"± {block['estructura']['SHD']['de']:.1f}",
            "Cobertura confusores": f"{block['cobertura_confusores']['media']:.3f}",
            "ATE": f"{block['ATE']['media']:.3f} ± {block['ATE']['de']:.3f}",
            "|Sesgo| vs oráculo": f"{block['sesgo_absoluto_vs_oraculo']['media']:.4f}",
            "Tiempo (s)": f"{block['segundos']['media']:.1f}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ejecución incremental y resumible
# --------------------------------------------------------------------------- #
def run_replications_incremental(
    config: ExperimentConfig,
    n_replications: int,
    store_dir: str | Path,
    time_budget: Optional[float] = None,
    verbose: bool = True,
    offset: int = 0,
    stride: int = 1,
) -> Dict[str, Any]:
    """Ejecuta las réplicas pendientes y las persiste una a una.

    Cada réplica se guarda en su propio archivo JSON en `store_dir`. Al reinvocar
    la función, las réplicas ya presentes se omiten. Esto hace el experimento
    **resumible**: puede interrumpirse y retomarse sin perder trabajo, y permite
    ejecutarlo por lotes en entornos con límite de tiempo por proceso (colas
    HPC, CI, cuadernos gestionados) sin alterar en absoluto los resultados,
    porque cada réplica depende solo de su semilla.

    Parameters
    ----------
    time_budget : float, opcional
        Segundos máximos de esta invocación. Se comprueba entre réplicas, nunca
        interrumpe una réplica a medias.
    offset, stride : int
        Permiten repartir las réplicas entre varios procesos que corren en
        paralelo (el proceso `k` de `s` toma las réplicas con
        ``(i - 1) % s == k``). Como cada réplica depende únicamente de su
        semilla, el reparto no altera ningún resultado.
    """
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    seeds = seed_sequence(config.seed, n_replications)
    started = time.perf_counter()
    executed = 0

    for i, seed in enumerate(seeds, start=1):
        if (i - 1) % stride != offset:
            continue
        target = store / f"replica_{i:03d}_seed{seed}.json"
        if target.exists():
            continue
        if time_budget is not None and (time.perf_counter() - started) > time_budget:
            break

        result = run_single(config, seed, verbose=False)
        save_results(result, target)
        executed += 1
        if verbose:
            ag = result["metodos"]["AG"]
            print(f"  réplica {i:>3}/{n_replications} (semilla {seed}) | "
                  f"F1 esq. {ag['estructura']['f1_esqueleto']:.3f} | "
                  f"SHD {ag['estructura']['SHD']:>3} | "
                  f"ATE {ag['estimacion']['ATE']:.4f} | "
                  f"{result['segundos_totales']:.0f} s", flush=True)

    done = sorted(store.glob("replica_*.json"))
    return {
        "ejecutadas_en_esta_invocacion": executed,
        "completadas": len(done),
        "pendientes": n_replications - len(done),
        "segundos": round(time.perf_counter() - started, 1),
    }


def aggregate_from_store(config: ExperimentConfig, store_dir: str | Path,
                         n_replications: Optional[int] = None) -> Dict[str, Any]:
    """Reconstruye el resumen Monte Carlo a partir de las réplicas persistidas."""
    store = Path(store_dir)
    files = sorted(store.glob("replica_*.json"))
    if not files:
        raise FileNotFoundError(f"No hay réplicas en {store}")

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    runs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]

    method_names = list(runs[0]["metodos"].keys())
    structure_keys = ["f1_esqueleto", "recall_esqueleto", "precision_esqueleto",
                      "f1_dirigido", "SHD", "aristas_estimadas", "invertidas"]

    summary: Dict[str, Any] = {"por_metodo": {}}
    for name in method_names:
        summary["por_metodo"][name] = {
            "estructura": {
                key: _aggregate([r["metodos"][name]["estructura"][key] for r in runs])
                for key in structure_keys
            },
            "ATE": _aggregate([r["metodos"][name]["estimacion"]["ATE"] for r in runs]),
            "cobertura_confusores": _aggregate(
                [r["metodos"][name]["conjunto_ajuste"]["cobertura_confusores"]
                 for r in runs]),
            "segundos": _aggregate([r["metodos"][name]["segundos"] for r in runs]),
            "sesgo_absoluto_vs_oraculo": _aggregate(
                [abs(r["metodos"][name]["estimacion"]["ATE"]
                     - r["referencias"]["ATE_dag_verdadero"]["ATE"]) for r in runs]),
            "tasa_contaminacion_descendientes": round(float(np.mean(
                [r["metodos"][name]["conjunto_ajuste"]["contamina_descendientes"]
                 for r in runs])), 4),
        }

    summary["referencias"] = {
        "ATE_dag_verdadero": _aggregate(
            [r["referencias"]["ATE_dag_verdadero"]["ATE"] for r in runs]),
        "ATE_sin_ajuste": _aggregate(
            [r["referencias"]["ATE_sin_ajuste"]["ATE"] for r in runs]),
    }
    summary["robustez"] = {
        key: _aggregate([r["robustez"][key] for r in runs])
        for key in ["efecto_base", "causa_comun_aleatoria",
                    "tratamiento_placebo", "submuestra_80"]
    }
    summary["tasa_aprobacion_refutadores"] = {
        key: round(float(np.mean([r["robustez"][key] for r in runs])), 4)
        for key in ["aprueba_causa_comun", "aprueba_placebo", "aprueba_submuestra"]
    }

    return {
        "configuracion": config.to_dict(),
        "entorno": environment_report(),
        "n_replicas": len(runs),
        "semillas": [r["semilla"] for r in runs],
        "resumen": summary,
        "corridas": runs,
    }
