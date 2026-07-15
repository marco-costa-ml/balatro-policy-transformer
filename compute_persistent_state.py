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
   [--dst data/persistent_state] [--workers 6]
   [--report artifacts/persistent_state_report.json]
   [--export-parsed] [--parsed-src data/parsed]
   [--parsed-dst data_export/persistent_state_parsed]
   [--parsed-workers 20]
   [--parsed-report artifacts/persistent_state_parsed_report.json]
   [--parsed-only]``

Run processing defaults to parallel (``ProcessPoolExecutor``, **6 workers**).
Each worker loads one granularized ``run_*.json``, computes ``states``, and
writes the matching persistent-state JSON under ``--dst`` (no buffering of
full runs in the parent). Completed jobs return only small stat shards merged
into the aggregate report.

By default the script also exports a parsed-event-aligned copy under
``data_export/persistent_state_parsed`` (one state per parsed event, **20
workers**). Pass ``--no-export-parsed`` to skip that pass.

Use ``--workers 1`` for a fully sequential pass (easier profiling / debugging).

Worker jobs are independent — statistics counter shards merge to the same
aggregate report as the old single-thread loop. Console output is sorted by
``(video_id, run_name)`` for stable grouping.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from card_effects import find_closest_in_deck_index
from granularize import (
    _card_pool_for_event,
    _normalized_action_label,
    collapse_zones,
    emit_objects,
    parse_event_action,
    resolve_target,
)
from state_reducer import (
    INTERNAL_KEYS,
    MODEL_VISIBLE_KEYS,
    apply_step,
    default_state,
    parse_base_action,
)

PERSISTENT_STATE_SCHEMA_VERSION = "2.0.0"  # tracked_deck_cards + planet/black-hole hand levels
PARSED_PERSISTENT_STATE_SCHEMA_VERSION = "2.1.0"  # + current_deck_cards (parsed export only)
STATE_REDUCER_VERSION = "2.0.0"
PARSED_EXPORT_DEFAULT_DST = Path("data_export/persistent_state_parsed")
PARSED_EXPORT_DEFAULT_WORKERS = 20
PARSED_EXPORT_KEYS = ("current_deck_cards",)


def parsed_event_to_reducer_step(event: dict[str, Any]) -> dict[str, Any]:
    """Adapt one parsed event into a reducer-compatible step dict.

    Unlike granularized micro-steps, parsed events are parent-level: card
    selections appear in ``selected_zones`` and are folded into
    ``pending_cards`` on the single commit step. SWAP synthesis is not
    performed (parsed streams have no ``SWAP_i_j`` actions).
    """
    base_action, subtype = parse_event_action(event)
    all_zones, selected_zones, other_zones = collapse_zones(event.get("objects") or [])
    target_zone, target_position, target_obj = resolve_target(
        base_action, subtype, all_zones, selected_zones
    )
    action = _normalized_action_label(
        base_action, target_zone, target_position, None
    )

    target_class_id: int | None = None
    if isinstance(target_obj, dict):
        cid = target_obj.get("class_id")
        if isinstance(cid, int):
            target_class_id = cid

    pending_cards: list[dict[str, Any]] = []
    pool_spec = _card_pool_for_event(
        base_action,
        subtype,
        event.get("page_name"),
        target_class_id,
    )
    if pool_spec is not None:
        selected_base, _pool_base = pool_spec
        pending_cards = [
            copy.deepcopy(c) for c in (selected_zones.get(selected_base) or [])
        ]

    objects = emit_objects(all_zones, other_zones, pending_cards=pending_cards)

    return {
        "frame_idx": event.get("frame_idx"),
        "page_name": event.get("page_name"),
        "action": action,
        "action_subtype": subtype,
        "selected_object": (
            {"object": copy.deepcopy(target_obj)} if target_obj is not None else None
        ),
        "pending_cards": pending_cards,
        "state": copy.deepcopy(event.get("state") or {}),
        "objects": objects,
    }


def _hand_cards_from_parsed_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return playing cards in the hand zone for PlayHand/DiscardHand removal."""
    raw = [
        copy.deepcopy(o)
        for o in (event.get("objects") or [])
        if isinstance(o, dict) and o.get("zone") == "CurrentHandAll"
    ]
    if raw:
        return raw
    all_zones, _, _ = collapse_zones(event.get("objects") or [])
    return [copy.deepcopy(o) for o in (all_zones.get("CurrentHand") or [])]


def _update_current_deck_cards(
    current_deck_cards: list[dict[str, Any]],
    tracked_state: dict[str, Any],
    event: dict[str, Any],
    base_action: str,
) -> list[dict[str, Any]]:
    """Apply parsed-export rules to ``current_deck_cards`` after one event."""
    if base_action == "StartNewRun" or event.get("page_name") != "In_Blind":
        return copy.deepcopy(tracked_state["tracked_deck_cards"])
    if base_action in ("PlayHand", "DiscardHand"):
        deck = list(current_deck_cards)
        for card in _hand_cards_from_parsed_event(event):
            idx = find_closest_in_deck_index(deck, card)
            if idx is not None:
                deck.pop(idx)
        return deck
    return current_deck_cards


def _process_run(
    run: dict[str, Any],
    stats: collections.Counter,
    *,
    event_source: str = "granularized",
    adapt_step: Any | None = None,
) -> dict[str, Any]:
    events = run.get("events") or []
    states: list[dict[str, Any]] = []
    state = default_state()
    track_current_deck = event_source == "parsed"
    current_deck_cards: list[dict[str, Any]] = (
        copy.deepcopy(state["tracked_deck_cards"]) if track_current_deck else []
    )

    for step in events:
        snapshot = copy.deepcopy(state)
        if track_current_deck:
            snapshot["current_deck_cards"] = copy.deepcopy(current_deck_cards)
        states.append(snapshot)

        reducer_step = adapt_step(step) if adapt_step is not None else step
        action = reducer_step.get("action") or ""
        base = parse_base_action(action)
        stats[("event_base", base)] += 1

        prev_random = state["unhandled_random_consumable_count"]
        prev_deck_size = len(state["tracked_deck_cards"])
        state = apply_step(state, reducer_step)
        if track_current_deck:
            parsed_base, _ = parse_event_action(step)
            current_deck_cards = _update_current_deck_cards(
                current_deck_cards,
                state,
                step,
                parsed_base,
            )
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

    schema_version = (
        PARSED_PERSISTENT_STATE_SCHEMA_VERSION
        if event_source == "parsed"
        else PERSISTENT_STATE_SCHEMA_VERSION
    )
    out: dict[str, Any] = {
        "video_id": run.get("video_id"),
        "run_index": run.get("run_index"),
        "schema_version": schema_version,
        "state_reducer_version": STATE_REDUCER_VERSION,
        "event_source": event_source,
        "n_steps": len(states),
        "model_visible_keys": sorted(MODEL_VISIBLE_KEYS),
        "internal_keys": sorted(INTERNAL_KEYS),
        "states": states,
    }
    if event_source == "parsed":
        out["parsed_export_keys"] = list(PARSED_EXPORT_KEYS)
    return out


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


def _process_run_file_to_disk(
    run_file: Path,
    dst_root: Path,
    *,
    event_source: str = "granularized",
) -> tuple[str, str, int, dict[tuple[str, str], int]]:
    """Compute persistent state for one run, write JSON, return print + stats shards.

    Must stay at module scope for multiprocessing pickle (Windows spawn).
    """
    stats: collections.Counter = collections.Counter()
    run = json.loads(run_file.read_text(encoding="utf-8"))
    adapt_step = parsed_event_to_reducer_step if event_source == "parsed" else None
    out = _process_run(
        run,
        stats,
        event_source=event_source,
        adapt_step=adapt_step,
    )
    out["states"] = [_state_to_jsonable(s) for s in out["states"]]
    video_id = run_file.parent.name.split("=", 1)[1]

    dst_dir = dst_root / f"video_id={video_id}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / run_file.name
    dst_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_steps = int(out["n_steps"])
    return video_id, run_file.name, n_steps, dict(stats)


def _run_export_pass(
    *,
    label: str,
    src: Path,
    dst: Path,
    workers: int,
    report: Path,
    event_source: str,
    missing_src_ok: bool = False,
) -> None:
    if not src.exists():
        if missing_src_ok:
            print(f"\n[{label}] skip — source not found: {src}")
            return
        raise SystemExit(f"{label} source not found: {src}")
    if workers < 1:
        raise SystemExit(f"{label}: workers must be >= 1")

    stats: collections.Counter = collections.Counter()
    runs_processed = 0
    steps_processed = 0

    tasks = list(_iter_run_files(src))
    dst_root = dst.resolve()

    summaries: list[tuple[str, str, int]] = []

    if not tasks:
        print(f"\n[{label}] no run_*.json under {src}")
    elif workers == 1:
        for video_id, run_file in tasks:
            vid, name, n_steps, local = _process_run_file_to_disk(
                run_file, dst_root, event_source=event_source
            )
            assert vid == video_id
            stats.update(local)
            summaries.append((vid, name, n_steps))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _process_run_file_to_disk,
                    rf,
                    dst_root,
                    event_source=event_source,
                ): rf
                for _, rf in tasks
            }
            for fut in as_completed(futures):
                vid, name, n_steps, local = fut.result()
                stats.update(local)
                summaries.append((vid, name, n_steps))

    summaries.sort(key=lambda t: (t[0], t[1]))

    print()
    print(f"=== {label} ===")
    current_video: str | None = None
    for video_id, run_name, n_steps in summaries:
        if video_id != current_video:
            print(f"video_id={video_id}")
            current_video = video_id
        runs_processed += 1
        steps_processed += n_steps
        print(f"  {run_name}  ({n_steps} states)")

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

    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_source": event_source,
        "src": src.as_posix(),
        "dst": dst.as_posix(),
        "workers_requested": workers,
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
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"wrote report -> {report.as_posix()}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize per-step persistent-state snapshots for granularized runs "
            "and optionally for parsed runs."
        )
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--dst", type=Path, default=Path("data/persistent_state"))
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Parallel worker processes for granularized run files. Default: 6. "
            "Use 1 to disable parallelism."
        ),
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/persistent_state_report.json"),
        help="Where to write the aggregate granularized run/event/state report.",
    )
    ap.add_argument(
        "--export-parsed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also export parsed-event persistent state under "
            f"{PARSED_EXPORT_DEFAULT_DST.as_posix()} (default: on)."
        ),
    )
    ap.add_argument(
        "--parsed-src",
        type=Path,
        default=Path("data/parsed"),
        help="Parsed run root for the export pass.",
    )
    ap.add_argument(
        "--parsed-dst",
        type=Path,
        default=PARSED_EXPORT_DEFAULT_DST,
        help="Output root for parsed-event persistent state.",
    )
    ap.add_argument(
        "--parsed-workers",
        type=int,
        default=PARSED_EXPORT_DEFAULT_WORKERS,
        metavar="N",
        help=(
            "Parallel worker processes for parsed run files. "
            f"Default: {PARSED_EXPORT_DEFAULT_WORKERS}."
        ),
    )
    ap.add_argument(
        "--parsed-report",
        type=Path,
        default=Path("artifacts/persistent_state_parsed_report.json"),
        help="Where to write the aggregate parsed run/event/state report.",
    )
    ap.add_argument(
        "--parsed-only",
        action="store_true",
        help="Skip granularized pass; export parsed persistent state only.",
    )
    args = ap.parse_args(argv)

    if not args.parsed_only:
        _run_export_pass(
            label="granularized persistent state",
            src=args.src,
            dst=args.dst,
            workers=args.workers,
            report=args.report,
            event_source="granularized",
        )

    if args.export_parsed or args.parsed_only:
        _run_export_pass(
            label="parsed persistent state export",
            src=args.parsed_src,
            dst=args.parsed_dst,
            workers=args.parsed_workers,
            report=args.parsed_report,
            event_source="parsed",
            missing_src_ok=True,
        )


if __name__ == "__main__":
    main()
