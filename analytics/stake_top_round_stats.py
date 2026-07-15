#!/usr/bin/env python3
"""Max OCR ``state.round`` per life, split by last ``CurrentStake`` class_id.

Lives are **not** split on ``StartNewRun``. A death + fresh run is inferred from
**debounced** ``state.round`` only:

1. **Candidate death:** confirmed round goes from **>= 1** to **0** (needs
   ``round_streak`` identical raw readings to confirm each value).
2. **Recovery / new run:** after that, confirmed transitions must include
   **0 → 1**, then **1 → 2** (configurable). If the player dies again before
   recovery finishes, recovery restarts at the new **>=1 → 0** anchor.

The previous life ends at the event **before** the confirmed ``0`` at step (1).
The next life begins at that ``0`` event and continues through the tail of the
video (or until the next closed life). Pending recovery at EOF keeps one open
life: ``(life_start, end)``.

``top_round`` is still ``max(state.round)`` over raw readings in the slice (not
only debounced values). Stake is the **last** ``CurrentStake`` ``class_id`` in
that slice (tie-break: ``position_in_zone``, ``slot_id``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ANALYTICS = Path(__file__).resolve().parent
if str(_ANALYTICS) not in sys.path:
    sys.path.insert(0, str(_ANALYTICS))

import analyze_granularized_decks_and_ante as granular  # noqa: E402


def pick_current_stake_class(objects: list | None) -> int | None:
    stakes = [
        o
        for o in (objects or [])
        if o.get("zone") == "CurrentStake" and o.get("class_id") is not None
    ]
    if not stakes:
        return None
    stakes.sort(key=lambda o: (o.get("position_in_zone", 0), o.get("slot_id", 0)))
    return int(stakes[-1]["class_id"])


def load_events_in_order(paths: list[tuple[int, Path]]) -> list[dict]:
    events: list[dict] = []
    for _ridx, path in sorted(paths, key=lambda t: t[0]):
        data = json.loads(path.read_text(encoding="utf-8"))
        events.extend(data.get("events", []))
    return events


def iter_confirmed_round_transitions(
    events: list[dict], streak: int
) -> list[tuple[int, int | None, int]]:
    """Indices are event positions; ``prev`` is None on first confirmed value."""
    out: list[tuple[int, int | None, int]] = []
    confirmed: int | None = None
    streak_val: int | None = None
    streak_count = 0
    for i, ev in enumerate(events):
        r = ev.get("state", {}).get("round")
        if r is None:
            continue
        r = int(r)
        if r == streak_val:
            streak_count += 1
        else:
            streak_val = r
            streak_count = 1
        if streak_count < streak:
            continue
        if streak_val == confirmed:
            continue
        prev = confirmed
        confirmed = streak_val
        out.append((i, prev, confirmed))
    return out


def life_index_ranges(
    events: list[dict],
    round_streak: int,
    require_round_2: bool,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Inclusive index ranges for each life. Also returns small diagnostic counters."""
    transitions = iter_confirmed_round_transitions(events, round_streak)
    life_start = 0
    pending: dict[str, int | bool] | None = None
    ranges: list[tuple[int, int]] = []
    stats = {
        "death_candidates": 0,
        "recoveries_completed": 0,
        "recoveries_reanchored": 0,
    }

    for i, prev, new in transitions:
        if pending is not None:
            if prev is not None and prev >= 1 and new == 0:
                stats["recoveries_reanchored"] += 1
                pending = {"zero_i": i, "s1": False, "s2": False}
                continue
            if prev == 0 and new == 1:
                pending["s1"] = True
            elif prev == 1 and new == 2:
                pending["s2"] = True

            done = bool(pending["s2"]) if require_round_2 else bool(pending["s1"])
            if done:
                z = int(pending["zero_i"])
                stats["recoveries_completed"] += 1
                if z > life_start:
                    ranges.append((life_start, z - 1))
                life_start = z
                pending = None
        elif prev is not None and prev >= 1 and new == 0:
            stats["death_candidates"] += 1
            pending = {"zero_i": i, "s1": False, "s2": False}

    if len(events) == 0:
        return [], stats

    ranges.append((life_start, len(events) - 1))
    return ranges, stats


def summarize(xs: list[int]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "stdev_sample": None, "min": None, "max": None}
    mean = statistics.mean(xs)
    stdev = statistics.stdev(xs) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 6),
        "stdev_sample": round(stdev, 6),
        "min": min(xs),
        "max": max(xs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--granular-dir",
        type=Path,
        default=Path("data/granularized"),
        help="Root with video_id=*/run_*.json",
    )
    ap.add_argument("--stake-a", type=int, default=268)
    ap.add_argument("--stake-b", type=int, default=273)
    ap.add_argument(
        "--round-streak",
        type=int,
        default=2,
        help="Consecutive matching raw round readings to confirm a value (default 2).",
    )
    ap.add_argument(
        "--require-round-2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require confirmed 1→2 after death (default: true). Use --no-require-round-2 for only 0→1.",
    )
    args = ap.parse_args()
    root = args.granular_dir

    files = granular.iter_run_files(root)
    by_video: dict[str, list[tuple[int, Path]]] = {}
    for p in files:
        vid, ridx = granular.parse_video_run_index(p)
        by_video.setdefault(vid, []).append((ridx, p))

    rounds_a: list[int] = []
    rounds_b: list[int] = []
    skipped_no_stake = 0
    skipped_other_stake: dict[int, int] = {}
    agg_stats: dict[str, int] = {
        "death_candidates": 0,
        "recoveries_completed": 0,
        "recoveries_reanchored": 0,
        "lives_total": 0,
    }

    for vid in sorted(by_video.keys(), key=lambda x: int(x) if x.isdigit() else x):
        events = load_events_in_order(by_video[vid])
        ranges, st = life_index_ranges(
            events,
            round_streak=args.round_streak,
            require_round_2=args.require_round_2,
        )
        for k in ("death_candidates", "recoveries_completed", "recoveries_reanchored"):
            agg_stats[k] += st[k]

        for a, b in ranges:
            if b < a:
                continue
            agg_stats["lives_total"] += 1
            slice_ev = events[a : b + 1]
            last_stake: int | None = None
            top_round = 0
            for ev in slice_ev:
                cid = pick_current_stake_class(ev.get("objects"))
                if cid is not None:
                    last_stake = cid
                r = ev.get("state", {}).get("round")
                if r is not None:
                    top_round = max(top_round, int(r))

            if last_stake is None:
                skipped_no_stake += 1
                continue
            if last_stake == args.stake_a:
                rounds_a.append(top_round)
            elif last_stake == args.stake_b:
                rounds_b.append(top_round)
            else:
                skipped_other_stake[last_stake] = skipped_other_stake.get(last_stake, 0) + 1

    sa = summarize(rounds_a)
    sb = summarize(rounds_b)

    print(f"Granular root: {root.resolve()}")
    print(
        f"Life detection: round_streak={args.round_streak}, "
        f"require_round_2={args.require_round_2}"
    )
    print(f"Diagnostics (summed over videos): {agg_stats}")
    print(f"Stake A (class_id={args.stake_a}): {sa}")
    print(f"Stake B (class_id={args.stake_b}): {sb}")
    print(f"Lives skipped (no CurrentStake class_id observed): {skipped_no_stake}")
    if skipped_other_stake:
        top_skip = sorted(skipped_other_stake.items(), key=lambda x: -x[1])[:12]
        print(f"Lives with other final stakes (top 12): {top_skip}")


if __name__ == "__main__":
    main()
