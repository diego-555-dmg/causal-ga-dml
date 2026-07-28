#!/usr/bin/env python3
"""Reproduce el experimento completo del artículo en un solo comando.

Equivale a encadenar `run_replicas.py` y `aggregate.py`:

    python scripts/run_multiseed.py --reps 30

Ejecuta las réplicas pendientes (de forma resumible: las ya calculadas se
reutilizan) y a continuación agrega los resultados, escribe las tablas y genera
las figuras.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiseed.yaml"))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--budget", type=float, default=None,
                        help="Segundos máximos para la fase de réplicas.")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("ESTUDIO MONTE CARLO — descubrimiento causal con AG + Double ML")
    print("=" * 78)

    replicas = [sys.executable, str(ROOT / "scripts" / "run_replicas.py"),
                "--config", args.config, "--reps", str(args.reps)]
    if args.budget is not None:
        replicas += ["--budget", str(args.budget)]
    code = subprocess.call(replicas)
    if code != 0:
        return code

    aggregate = [sys.executable, str(ROOT / "scripts" / "aggregate.py"),
                 "--config", args.config]
    if args.skip_figures:
        aggregate.append("--skip-figures")
    return subprocess.call(aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
