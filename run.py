#!/usr/bin/env python3
"""Run one of the five archived HEU experiments."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Sequence

EXPERIMENTS = {
    "exp01": "hh_envelope.experiments.exp_01_locking",
    "exp02": "hh_envelope.experiments.exp_02_two_voice_coupling",
    "exp02-v01": "hh_envelope.experiments.exp_02_coupling_v01",
    "exp03": "hh_envelope.experiments.exp_03_fit_envelope",
    "exp04": "hh_envelope.experiments.exp_04_impossibility",
    "exp05": "hh_envelope.experiments.exp_05_ablation_uncoupled",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument(
        "--interactive-plots",
        action="store_true",
        help="Allow an interactive Matplotlib backend instead of defaulting to Agg.",
    )
    args = parser.parse_args(argv)

    if not args.interactive_plots:
        os.environ.setdefault("MPLBACKEND", "Agg")

    experiment = importlib.import_module(EXPERIMENTS[args.experiment])
    result = experiment.main()
    if isinstance(result, bool):
        return 0 if result else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
