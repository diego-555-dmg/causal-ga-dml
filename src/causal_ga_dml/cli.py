"""Interfaz de línea de comandos del paquete.

Ejemplos
--------
    python -m causal_ga_dml.cli single    --config configs/default.yaml
    python -m causal_ga_dml.cli multiseed --config configs/multiseed.yaml --reps 30
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .config import load_config
from .experiment import run_multiseed, run_single, save_results, summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-ga-dml",
        description="Descubrimiento causal con algoritmos genéticos + Double ML.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [("single", "Ejecuta una corrida con una semilla."),
                            ("multiseed", "Ejecuta el estudio Monte Carlo.")]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", default="configs/default.yaml",
                        help="Ruta al archivo YAML de configuración.")
        sp.add_argument("--seed", type=int, default=None, help="Semilla maestra.")
        sp.add_argument("--output", default=None, help="Ruta del JSON de salida.")
        sp.add_argument("--quiet", action="store_true", help="Silencia el progreso.")
        if name == "multiseed":
            sp.add_argument("--reps", type=int, default=None,
                            help="Número de réplicas independientes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    config = load_config(args.config, seed=args.seed)
    verbose = not args.quiet

    if args.command == "single":
        results = run_single(config, verbose=verbose)
        default_out = Path(config.output_dir) / f"single_seed{config.seed}.json"
    else:
        results = run_multiseed(config, args.reps, verbose=verbose)
        default_out = Path(config.output_dir) / f"multiseed_{results['n_replicas']}.json"
        if verbose:
            print("\n" + summary_table(results).to_string(index=False))

    path = save_results(results, args.output or default_out)
    print(f"\n[OK] Resultados guardados en: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
