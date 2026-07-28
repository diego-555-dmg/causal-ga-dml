#!/usr/bin/env python3
"""Ejecuta réplicas de forma incremental y resumible.

Cada réplica se guarda por separado, de modo que el estudio completo puede
construirse en varias invocaciones sin repetir trabajo ni alterar resultados:

    python scripts/run_replicas.py --reps 30                # todas
    python scripts/run_replicas.py --reps 30 --budget 600   # por lotes de 10 min
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from causal_ga_dml.config import load_config                       # noqa: E402
from causal_ga_dml.experiment import run_replications_incremental   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiseed.yaml"))
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None,
                        help="Segundos máximos de esta invocación.")
    parser.add_argument("--store", default=None)
    parser.add_argument("--offset", type=int, default=0,
                        help="Índice de este proceso dentro del reparto paralelo.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Número total de procesos paralelos.")
    args = parser.parse_args()

    config = load_config(args.config)
    reps = args.reps or config.n_replications
    store = Path(args.store) if args.store else ROOT / config.output_dir / "replicas"

    status = run_replications_incremental(config, reps, store, args.budget,
                                          verbose=True, offset=args.offset,
                                          stride=args.stride)
    print(f"[estado] completadas {status['completadas']}/{reps} "
          f"| pendientes {status['pendientes']} "
          f"| {status['segundos']} s en esta invocación")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
