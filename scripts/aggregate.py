#!/usr/bin/env python3
"""Agrega las réplicas persistidas, produce las tablas del artículo y las figuras.

    python scripts/aggregate.py --config configs/multiseed.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402

from causal_ga_dml.config import load_config                       # noqa: E402
from causal_ga_dml.data import simulate                             # noqa: E402
from causal_ga_dml.experiment import (                              # noqa: E402
    aggregate_from_store, run_single, save_results, summary_table,
)
from causal_ga_dml.plots import (                                   # noqa: E402
    plot_ate_distribution, plot_convergence, plot_dag_comparison,
    plot_method_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiseed.yaml"))
    parser.add_argument("--store", default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = ROOT / config.output_dir
    figures_dir = ROOT / config.figures_dir
    store = Path(args.store) if args.store else results_dir / "replicas"

    results = aggregate_from_store(config, store)
    save_results(results, results_dir / f"multiseed_{results['n_replicas']}.json")

    table = summary_table(results)
    table.to_csv(results_dir / "tabla_comparativa.csv", index=False, encoding="utf-8")
    print(table.to_string(index=False))

    print("\nRobustez (media sobre réplicas):")
    for key, block in results["resumen"]["robustez"].items():
        print(f"  {key:>24}: {block['media']:.4f} ± {block['de']:.4f}")
    print("Tasa de aprobación de refutadores:",
          results["resumen"]["tasa_aprobacion_refutadores"])

    if not args.skip_figures:
        print("\nGenerando figuras...")
        reference = run_single(config, config.seed, verbose=False)
        data, true_dag = simulate(config.data.network, config.data.n_samples, config.seed)
        ga_dag = nx.DiGraph()
        ga_dag.add_nodes_from(data.columns)
        ga_dag.add_edges_from(reference["_dags"]["AG"])

        plot_dag_comparison(true_dag, ga_dag, config.data.treatment,
                            config.data.outcome, figures_dir / "fig1_dags.png")
        plot_convergence(reference["ag"]["historia_mejor"],
                         reference["ag"]["historia_media"],
                         figures_dir / "fig2_convergencia.png")
        plot_method_comparison(results["resumen"]["por_metodo"],
                               figures_dir / "fig3_comparacion_metodos.png")
        plot_ate_distribution(results["corridas"],
                              figures_dir / "fig4_distribucion_ate.png")
        save_results(reference, results_dir / f"single_seed{config.seed}.json")
        print(f"Figuras en: {figures_dir}")

    print(f"\n[OK] Resultados en: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
