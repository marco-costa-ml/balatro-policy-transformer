#!/usr/bin/env python3
"""
build_action_train_counts.py
============================
Histogram resolved flat ``target_action_id`` values on the **train** split only,
emit ``artifacts/action_train_counts.json`` for within-family logit adjustment.

Usage::
    python scripts/build_action_train_counts.py
        [--tensorized data/tensorized]
        [--splits artifacts/splits.json]
        [--action-config data/action_space_config.json]
        [--out artifacts/action_train_counts.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map
from dataset import BalatroStepDataset, load_split


SCHEMA_VERSION = "1.0.0"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensorized", type=Path, default=_REPO_ROOT / "data" / "tensorized")
    ap.add_argument("--splits", type=Path, default=_REPO_ROOT / "artifacts" / "splits.json")
    ap.add_argument(
        "--action-config",
        type=Path,
        default=_REPO_ROOT / "data" / "action_space_config.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "action_train_counts.json",
    )
    args = ap.parse_args(argv)

    action_map = compute_action_map(
        json.loads(args.action_config.read_text(encoding="utf-8"))
    )
    n_actions = int(action_map["n_actions"])
    index_to_label = list(action_map["index_to_label"])

    train_ds = BalatroStepDataset(
        tensorized_root=args.tensorized,
        split_videos=load_split(args.splits, "train"),
        include_unresolved=False,
        device=None,
    )
    if train_ds.n_actions() != n_actions:
        raise SystemExit(
            f"tensorized N_ACTIONS={train_ds.n_actions()} != action_map {n_actions}"
        )

    flat = train_ds.valid_indices()
    y = train_ds.all_tensors()["target_action_id"].index_select(0, flat)
    counts = torch.bincount(y.long().cpu(), minlength=n_actions)

    fam_totals: dict[str, int] = {}
    for i, c in enumerate(counts.tolist()):
        fam = index_to_label[i].split("_", 1)[0]
        fam_totals[fam] = fam_totals.get(fam, 0) + int(c)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorized": args.tensorized.as_posix(),
        "splits": args.splits.as_posix(),
        "action_config": args.action_config.as_posix(),
        "n_actions": n_actions,
        "train_steps_resolved": int(len(train_ds)),
        "count_per_action": [int(x) for x in counts.tolist()],
        "index_to_label": index_to_label,
        "family_totals": {k: int(v) for k, v in sorted(fam_totals.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path.as_posix()}  (train_steps={len(train_ds)})")


if __name__ == "__main__":
    main()
