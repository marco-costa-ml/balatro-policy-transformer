#!/usr/bin/env python3
"""Granularized dataset stats: deck class_id counts and debounced ante transitions.

Ante is taken from event[\"state\"][\"ante\"] (OCR-derived). To reduce flicker noise,
a reading only becomes \"confirmed\" after ``--streak`` consecutive events show the
same ante; transitions are counted only between confirmed values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_run_files(granular_root: Path) -> list[Path]:
    files = sorted(granular_root.rglob("run_*.json"))
    return files


def parse_video_run_index(path: Path) -> tuple[str, int]:
    """video_id=123/run_004.json -> (123, 4)."""
    video_dir = path.parent.name
    if video_dir.startswith("video_id="):
        vid = video_dir.split("=", 1)[1]
    else:
        vid = video_dir
    stem = path.stem  # run_004
    idx = int(stem.split("_", 1)[1])
    return vid, idx


def count_deck_class_ids(granular_root: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in iter_run_files(granular_root):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("events", []):
            for obj in ev.get("objects") or []:
                if obj.get("object_type") != "deck":
                    continue
                cid = obj.get("class_id")
                if cid is not None:
                    counts[int(cid)] += 1
    return counts


def analyze_ante_transitions(
    granular_root: Path,
    streak_required: int = 3,
) -> dict:
    """Chain runs within each video (sorted by run index). Reset state between videos."""
    files = iter_run_files(granular_root)
    by_video: dict[str, list[tuple[int, Path]]] = {}
    for p in files:
        vid, ridx = parse_video_run_index(p)
        by_video.setdefault(vid, []).append((ridx, p))
    for vid in by_video:
        by_video[vid].sort(key=lambda t: t[0])

    trans = Counter()  # (a, b) -> count
    from_eight = Counter()  # b -> count for debounced 8 -> b
    eight_to_nine = 0
    to_ante_one = 0
    runs_processed = 0
    events_seen = 0

    for vid in sorted(by_video.keys(), key=lambda x: int(x) if x.isdigit() else x):
        confirmed: int | None = None
        streak_val: int | None = None
        streak_count = 0

        for _ridx, path in by_video[vid]:
            runs_processed += 1
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            for ev in data.get("events", []):
                events_seen += 1
                raw = ev.get("state", {}).get("ante")
                if raw is not None:
                    raw = int(raw)
                if raw is None:
                    continue

                if raw == streak_val:
                    streak_count += 1
                else:
                    streak_val = raw
                    streak_count = 1

                if streak_count != streak_required:
                    continue
                if streak_val is None or streak_val == confirmed:
                    continue

                prev = confirmed
                confirmed = streak_val
                if prev is None:
                    continue
                trans[(prev, confirmed)] += 1
                if prev == 8:
                    from_eight[confirmed] += 1
                if prev == 8 and confirmed == 9:
                    eight_to_nine += 1
                if confirmed == 1 and prev != 1:
                    to_ante_one += 1

    fe_list = sorted(from_eight.items(), key=lambda x: (-x[1], x[0]))

    return {
        "streak_required": streak_required,
        "videos": len(by_video),
        "runs_processed": runs_processed,
        "events_seen": events_seen,
        "transition_8_to_9": eight_to_nine,
        "transition_any_to_1_excluding_init": to_ante_one,
        "ratio_8_9_over_to_1": (
            round(eight_to_nine / to_ante_one, 6) if to_ante_one else None
        ),
        "debounced_from_ante_8": fe_list,
        "debounced_from_ante_8_total": sum(from_eight.values()),
        "top_transition_pairs": trans.most_common(25),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--granular-dir",
        type=Path,
        default=Path("data/granularized"),
        help="Root with video_id=*/run_*.json",
    )
    ap.add_argument(
        "--streak",
        type=int,
        default=3,
        help="Consecutive identical ante readings required to confirm (default 3).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full results JSON.",
    )
    args = ap.parse_args()
    root: Path = args.granular_dir

    deck_counts = count_deck_class_ids(root)
    deck_total = sum(deck_counts.values())
    ante_stats = analyze_ante_transitions(root, streak_required=args.streak)

    deck_dist = [
        {"class_id": cid, "count": n, "fraction": round(n / deck_total, 8) if deck_total else 0.0}
        for cid, n in sorted(deck_counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    out = {
        "deck_class_id_counts": {str(k): v for k, v in sorted(deck_counts.items())},
        "deck_observations_total": deck_total,
        "deck_distribution_sorted": deck_dist,
        "ante_transitions": ante_stats,
    }

    print(f"Granular root: {root.resolve()}")
    print()
    print("=== Deck objects (object_type=='deck') by class_id ===")
    print(f"Total deck observations: {deck_total}")
    for row in deck_dist[:40]:
        print(f"  class_id {row['class_id']:>4}  {row['count']:>8}  ({100 * row['fraction']:.4f}%)")
    if len(deck_dist) > 40:
        print(f"  ... {len(deck_dist) - 40} more class_ids")
    print()
    print("=== Debounced ante transitions (streak >= {}) ===".format(args.streak))
    for k, v in ante_stats.items():
        if k != "top_transition_pairs":
            print(f"  {k}: {v}")
    print("  top (prev, next) transition counts:")
    for (a, b), n in ante_stats["top_transition_pairs"]:
        print(f"    {a} -> {b}: {n}")
    print("  debounced transitions leaving confirmed ante 8 (by next ante):")
    for nxt, n in ante_stats["debounced_from_ante_8"]:
        print(f"    8 -> {nxt}: {n}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print()
        print(f"Wrote {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
