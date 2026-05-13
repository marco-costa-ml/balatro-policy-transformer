#!/usr/bin/env python3
"""
compute_persistent_state.py
===========================
Run the persistent-state reducer (``state_reducer.apply_step``) over every
granularized run and materialize per-step state snapshots to disk.

For every ``data/granularized/video_id=*/run_NNN.json`` we write a parallel
``data/persistent_state/video_id=*/run_NNN.json`` with the schema:

    {
      "video_id": str,
      "run_index": int,
      "schema_version": "1.0.0",
      "state_reducer_version": "1.0.0",
      "n_steps": int,
      "model_visible_keys": [...],
      "internal_keys": [...],
      "states": [
        {<full state BEFORE step 0>},
        {<full state BEFORE step 1>},
        ...
        {<full state BEFORE step n-1>}
      ]
    }

Each ``states[t]`` is the persistent state the model sees as input for
step ``t``. The full state (including INTERNAL fields) is stored so that
the mask builder can read it; the tensorizer must apply
``state_reducer.to_model_visible`` to filter to model-input fields.

Usage
-----
``python compute_persistent_state.py [--src data/granularized]
   [--dst data/persistent_state] [--report artifacts/persistent_state_report.json]``
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from state_reducer import (
    INTERNAL_KEYS,
    MODEL_VISIBLE_KEYS,
    apply_step,
    default_state,
    parse_base_action,
)

PERSISTENT_STATE_SCHEMA_VERSION = "2.0.0"  # tracked_deck_cards + planet/black-hole hand levels
STATE_REDUCER_VERSION = "2.0.0"


def _process_run(run: dict[str, Any], stats: collections.Counter) -> dict[str, Any]:
    events = run.get("events") or []
    states: list[dict[str, Any]] = []
    state = default_state()

    for step in events:
        states.append(copy.deepcopy(state))
        action = step.get("action") or ""
        base = parse_base_action(action)
        stats[("event_base", base)] += 1

        prev_random = state["unhandled_random_consumable_count"]
        prev_deck_size = len(state["tracked_deck_cards"])
        state = apply_step(state, step)
        if state["unhandled_random_consumable_count"] > prev_random:
            stats[("random_consumable", "skipped")] += 1
        delta = len(state["tracked_deck_cards"]) - prev_deck_size
        if delta > 0:
            stats[("tracked_deck", "cards_added")] += delta
        elif delta < 0:
            stats[("tracked_deck", "cards_removed")] += -delta

    # Per-run final-state diagnostics.
    stats[("final_state", "deck_detected" if state["deck_detected"] else "no_deck_detected")] += 1
    if state["deck_detected"]:
        deck_class_id = state["deck"]["class_id"]
        stats[("final_state_deck_class_id", str(deck_class_id))] += 1

    stats[("tracked_deck_size", str(len(state["tracked_deck_cards"])))] += 1

    return {
        "video_id": run.get("video_id"),
        "run_index": run.get("run_index"),
        "schema_version": PERSISTENT_STATE_SCHEMA_VERSION,
        "state_reducer_version": STATE_REDUCER_VERSION,
        "n_steps": len(states),
        "model_visible_keys": sorted(MODEL_VISIBLE_KEYS),
        "internal_keys": sorted(INTERNAL_KEYS),
        "states": states,
    }


def _iter_run_files(src_root: Path):
    for partition in sorted(src_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for run_file in sorted(partition.glob("run_*.json")):
            yield video_id, run_file


def _state_to_jsonable(state: dict[str, Any]) -> dict[str, Any]:
    """
    JSON-encode the state. With ``tracked_deck_cards`` (list[dict]) replacing
    the previous int-keyed ``cards_in_deck_counts`` dict, no special encoding
    is required; ``json.dumps`` handles list-of-dicts directly. This wrapper
    is retained as a hook in case future fields require custom encoding.
    """
    return state


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Materialize per-step persistent-state snapshots for granularized runs."
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--dst", type=Path, default=Path("data/persistent_state"))
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/persistent_state_report.json"),
        help="Where to write the aggregate run/event/state report.",
    )
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"granularized root not found: {args.src}")

    stats: collections.Counter = collections.Counter()
    runs_processed = 0
    steps_processed = 0
    current_video: str | None = None

    for video_id, run_file in _iter_run_files(args.src):
        if video_id != current_video:
            print(f"video_id={video_id}")
            current_video = video_id

        run = json.loads(run_file.read_text(encoding="utf-8"))
        out = _process_run(run, stats)

        # Encode states (handle int-keyed dicts) before writing.
        out["states"] = [_state_to_jsonable(s) for s in out["states"]]

        dst_dir = args.dst / f"video_id={video_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / run_file.name
        dst_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        runs_processed += 1
        steps_processed += out["n_steps"]
        print(f"  {run_file.name}  ({out['n_steps']} states)")

    # ------------------------------------------------------------------
    # Aggregate console report.
    # ------------------------------------------------------------------
    print()
    print(f"runs processed:  {runs_processed}")
    print(f"states written:  {steps_processed}")

    print()
    print("--- per-action event counts ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 != "event_base":
            continue
        print(f"  {k2:30s} {n}")

    print()
    print("--- per-run deck detection ---")
    deck_detected = stats.get(("final_state", "deck_detected"), 0)
    no_deck = stats.get(("final_state", "no_deck_detected"), 0)
    print(f"  deck detected:    {deck_detected}")
    print(f"  no deck detected: {no_deck}")

    if any(k1 == "final_state_deck_class_id" for (k1, _) in stats):
        print()
        print("--- final-state deck distribution ---")
        deck_counts = sorted(
            (
                (k2, n)
                for (k1, k2), n in stats.items()
                if k1 == "final_state_deck_class_id"
            ),
            key=lambda kv: -kv[1],
        )
        for cid, n in deck_counts:
            print(f"  class_id={cid}  runs={n}")

    print()
    print("--- tracked_deck deltas (cumulative) ---")
    print(f"  cards added:    {stats.get(('tracked_deck', 'cards_added'), 0)}")
    print(f"  cards removed:  {stats.get(('tracked_deck', 'cards_removed'), 0)}")
    print(f"  random consumables skipped: {stats.get(('random_consumable', 'skipped'), 0)}")

    print()
    print("--- final tracked_deck size distribution (top 10) ---")
    size_counts = sorted(
        (
            (int(k2), n)
            for (k1, k2), n in stats.items()
            if k1 == "tracked_deck_size"
        ),
        key=lambda kv: -kv[1],
    )[:10]
    for size, n in size_counts:
        print(f"  {size} cards: {n} runs")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src": args.src.as_posix(),
        "dst": args.dst.as_posix(),
        "state_reducer_version": STATE_REDUCER_VERSION,
        "runs_processed": runs_processed,
        "states_written": steps_processed,
        "event_base_counts": {
            k2: n for (k1, k2), n in stats.items() if k1 == "event_base"
        },
        "deck_detection": {
            "deck_detected": stats.get(("final_state", "deck_detected"), 0),
            "no_deck_detected": stats.get(("final_state", "no_deck_detected"), 0),
        },
        "final_state_deck_class_ids": {
            k2: n for (k1, k2), n in stats.items() if k1 == "final_state_deck_class_id"
        },
        "tracked_deck": {
            "cards_added_total": stats.get(("tracked_deck", "cards_added"), 0),
            "cards_removed_total": stats.get(("tracked_deck", "cards_removed"), 0),
            "random_consumables_skipped": stats.get(
                ("random_consumable", "skipped"), 0
            ),
            "final_size_distribution": {
                k2: n for (k1, k2), n in stats.items() if k1 == "tracked_deck_size"
            },
        },
    }
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"wrote report -> {args.report.as_posix()}")


if __name__ == "__main__":
    main()
