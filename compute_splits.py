#!/usr/bin/env python3
"""
compute_splits.py
=================
Deterministically partition videos into train/val/test, keeping all runs
from a given video in the same split. Splitting by ``video_id`` (rather
than by step or by run) prevents behavior-cloning leakage: the same player
within the same recording shares mannerisms / opening lines / strategic
preferences that would otherwise appear in both train and val.

Algorithm
---------
1. Enumerate every ``data/tensorized/video_id=*/run_*.npz`` shard and
   compute, per video_id:
   - ``n_runs`` (count of run files)
   - ``n_steps`` (sum of step counts across run files)
2. Sort videos by ``n_steps`` descending and greedily assign each video
   to the split with the lowest current step total, biased toward the
   target ratios.
3. Emit ``artifacts/splits.json`` with the assignment plus per-split
   summary stats (videos / runs / steps / step share).

Targets default to ``train=0.70 val=0.15 test=0.15``.

Determinism
-----------
The greedy procedure breaks ties by ``(target_share - current_share)``
then by ``video_id`` (lexicographic), so the same corpus produces the
same splits across machines. Re-running with new shards is allowed; the
``schema_version`` field bumps when the algorithm changes.

Usage
-----
``python compute_splits.py
    [--src data/tensorized]
    [--out artifacts/splits.json]
    [--train 0.70 --val 0.15 --test 0.15]``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLITS_SCHEMA_VERSION = "1.0.0"

DEFAULT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _scan_videos(src: Path) -> dict[str, dict[str, int | list[str]]]:
    """Walk the tensorized corpus and collect per-video step / run counts."""
    by_video: dict[str, dict[str, int | list[str]]] = {}
    for partition in sorted(src.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        runs = sorted(partition.glob("run_*.npz"))
        if not runs:
            continue
        n_steps = 0
        for r in runs:
            with np.load(r) as z:
                n_steps += int(z["target_action_id"].shape[0])
        by_video[video_id] = {
            "n_runs": len(runs),
            "n_steps": n_steps,
            "run_files": [r.relative_to(src.parent).as_posix() for r in runs],
        }
    return by_video


def _greedy_assign(
    videos: dict[str, dict[str, int | list[str]]],
    targets: dict[str, float],
) -> dict[str, list[str]]:
    """Largest-video-first assignment, biased toward the most-underfilled split."""
    total_steps = sum(int(v["n_steps"]) for v in videos.values())
    target_steps = {name: total_steps * frac for name, frac in targets.items()}

    current: dict[str, int] = {name: 0 for name in targets}
    assignment: dict[str, list[str]] = {name: [] for name in targets}

    # Sort videos: largest first, then lexicographic tie-break.
    sorted_videos = sorted(
        videos.items(),
        key=lambda kv: (-int(kv[1]["n_steps"]), kv[0]),
    )

    for video_id, meta in sorted_videos:
        # Score each split: target - current. Highest deficit wins; ties broken
        # by split-name to keep the algorithm fully deterministic.
        deficits = sorted(
            (
                (target_steps[name] - current[name], name)
                for name in targets
            ),
            key=lambda x: (-x[0], x[1]),
        )
        chosen = deficits[0][1]
        assignment[chosen].append(video_id)
        current[chosen] += int(meta["n_steps"])

    # Sort assigned video_ids for deterministic output.
    for name in assignment:
        assignment[name].sort()
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--train", type=float, default=DEFAULT_TARGETS["train"])
    ap.add_argument("--val", type=float, default=DEFAULT_TARGETS["val"])
    ap.add_argument("--test", type=float, default=DEFAULT_TARGETS["test"])
    args = ap.parse_args()

    targets = {"train": args.train, "val": args.val, "test": args.test}
    if not abs(sum(targets.values()) - 1.0) < 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0; got {targets}")
    for name, frac in targets.items():
        if frac < 0:
            raise SystemExit(f"--{name} must be >= 0; got {frac}")

    if not args.src.exists():
        raise SystemExit(f"tensorized root not found: {args.src}")

    print(f"scanning {args.src.as_posix()} ...")
    videos = _scan_videos(args.src)
    if not videos:
        raise SystemExit("no videos found; run tensorize.py first")

    total_videos = len(videos)
    total_runs = sum(int(v["n_runs"]) for v in videos.values())
    total_steps = sum(int(v["n_steps"]) for v in videos.values())
    print(f"  videos={total_videos}  runs={total_runs}  steps={total_steps}")

    print("\nassigning ...")
    assignment = _greedy_assign(videos, targets)

    splits_summary: dict[str, dict[str, int | float | list[str]]] = {}
    for name, vids in assignment.items():
        s_runs = sum(int(videos[v]["n_runs"]) for v in vids)
        s_steps = sum(int(videos[v]["n_steps"]) for v in vids)
        splits_summary[name] = {
            "video_count": len(vids),
            "run_count": s_runs,
            "step_count": s_steps,
            "step_share": s_steps / total_steps if total_steps else 0.0,
            "target_share": targets[name],
            "video_ids": vids,
        }
        print(
            f"  {name:5s}  videos={len(vids):4d}  runs={s_runs:5d}  "
            f"steps={s_steps:6d}  share={s_steps/total_steps:.4f} "
            f"(target {targets[name]:.4f})"
        )

    payload = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src": args.src.as_posix(),
        "targets": targets,
        "totals": {
            "videos": total_videos,
            "runs": total_runs,
            "steps": total_steps,
        },
        "splits": splits_summary,
        "video_metadata": {
            vid: {
                "n_runs": int(meta["n_runs"]),
                "n_steps": int(meta["n_steps"]),
                "run_files": list(meta["run_files"]),
            }
            for vid, meta in sorted(videos.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out.as_posix()}")


if __name__ == "__main__":
    main()
