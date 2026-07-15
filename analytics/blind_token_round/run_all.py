#!/usr/bin/env python3
"""Regenerate BlindToken-stratified figures (stake 268 cohort)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main() -> None:
    scripts = [
        "plot_conditional_loss_by_round.py",
        "plot_mean_decision_time_by_round.py",
        "plot_conditional_loss_vs_mean_dt.py",
    ]
    for name in scripts:
        path = HERE / name
        r = subprocess.run([sys.executable, str(path)], cwd=REPO, check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("All blind_token_round plots OK.")


if __name__ == "__main__":
    main()
