#!/usr/bin/env python3
"""
Run the default data pipeline then training:

  parse_events.py -> granularize.py -> compute_persistent_state.py ->
  tensorize.py -> scripts/build_action_train_counts.py -> train.py --amp ...
                    (train counts artifact is for optional logit-adjust / eval flags)

  train.py --amp --loss-weight-scope family
       --loss-weight-beta 0.25 --loss-weight-max 5.0 --epochs 15

Invoke from the repo root (or anywhere): ``python run_pipeline.py``
Or on Windows: ``RUN_PIPELINE.bat``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(py: str, script: Path, argv: list[str] | None = None) -> None:
    cmd = [py, str(script)]
    if argv:
        cmd.extend(argv)
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=script.parent, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    py = sys.executable
    _run(py, root / "parse_events.py")
    _run(py, root / "granularize.py")
    _run(py, root / "compute_persistent_state.py")
    _run(py, root / "tensorize.py")
    _run(py, root / "scripts" / "build_action_train_counts.py")
    _run(
        py,
        root / "train.py",
        [
            "--amp",
            "--loss-weight-scope",
            "family",
            "--loss-weight-beta",
            "0.25",
            "--loss-weight-max",
            "5.0",
            "--epochs",
            "15",
        ],
    )
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
