#!/usr/bin/env python3
"""Corrida única y rápida con la semilla de referencia (~30 s)."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from causal_ga_dml.config import load_config          # noqa: E402
from causal_ga_dml.experiment import run_single, save_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    results = run_single(config, verbose=True)

    print("\n--- Resumen por método -------------------------------------------")
    for name, block in results["metodos"].items():
        print(f"{name:>16} | F1 esq. {block['estructura']['f1_esqueleto']:.3f} "
              f"| SHD {block['estructura']['SHD']:>3} "
              f"| ATE {block['estimacion']['ATE']:.4f} "
              f"| {block['segundos']:>5.1f} s")
    print(f"{'DAG verdadero':>16} | ATE "
          f"{results['referencias']['ATE_dag_verdadero']['ATE']:.4f}")
    print(f"{'Sin ajuste':>16} | ATE "
          f"{results['referencias']['ATE_sin_ajuste']['ATE']:.4f}")

    path = save_results(results, ROOT / config.output_dir / f"single_seed{config.seed}.json")
    print(f"\n[OK] Resultados guardados en: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
