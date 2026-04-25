#!/usr/bin/env python3
"""
granularize.py
==============
Reads  data/parsed/video_id=*/run_*.json
Writes data/granularized/video_id=*/run_*.json

PlayHand and DiscardHand events are decomposed into sequential SelectCard
micro-actions, one per card in CurrentHandSelected, followed by a final
commit action (PlayHand / DiscardHand).

SelectCard ordering:
  Cards in CurrentHandSelected are sorted by position_in_zone, which
  encodes the order they were selected by the player.

SelectCard representation per step i:
  target_card   — the card being selected at step i
  selected_cards — cards selected in steps 0 … i-1
  current_hand  — shuffled pool of cards not yet selected:
                    original CurrentHand cards
                  + CurrentHandSelected cards not yet targeted (steps i+1 … N)
  objects       — all non-hand context objects (jokers, consumables, deck, …)

Final commit step:
  action        — PlayHand | DiscardHand
  selected_cards — all selected cards in selection order
  current_hand  — original CurrentHand cards (unselected)
  objects       — all non-hand context objects

All other events are passed through with action as a single string.

Usage:
    python granularize.py [--src data/parsed] [--dst data/granularized] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

HAND_ZONES = {"CurrentHand", "CurrentHandSelected"}
DECOMPOSE_ACTIONS = {"PlayHand", "DiscardHand"}


# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------

def group_by_zone(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    zones: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        zones.setdefault(obj["zone"], []).append(obj)
    return zones


def strip_zone_fields(obj: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of obj without zone-tracking fields."""
    return {k: v for k, v in obj.items() if k not in ("zone", "position_in_zone")}


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_hand_event(
    event: dict[str, Any],
    source_action: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Expand one PlayHand / DiscardHand event into SelectCard steps + commit.

    Returns an empty list if CurrentHandSelected is empty (edge case).
    """
    zones = group_by_zone(event["objects"])

    # Cards sorted by selection order (position_in_zone = order player picked them)
    selected_ordered: list[dict[str, Any]] = sorted(
        zones.get("CurrentHandSelected", []),
        key=lambda o: o["position_in_zone"],
    )

    # Original unselected cards (never got clicked)
    unselected_base: list[dict[str, Any]] = list(zones.get("CurrentHand", []))

    # Context objects — everything that is not a hand card
    context_objects: list[dict[str, Any]] = [
        obj for obj in event["objects"] if obj["zone"] not in HAND_ZONES
    ]

    if not selected_ordered:
        # No selected cards — emit the original action unchanged
        return [_passthrough(event)]

    micro_events: list[dict[str, Any]] = []

    for step_idx, target in enumerate(selected_ordered):
        prev_selected = selected_ordered[:step_idx]
        future_selected = selected_ordered[step_idx + 1:]

        # Unselected pool: original unselected + cards not yet targeted, shuffled
        pool: list[dict[str, Any]] = unselected_base + future_selected
        pool = copy.deepcopy(pool)
        rng.shuffle(pool)

        # Reassign position_in_zone in the shuffled pool
        current_hand = [
            {**obj, "zone": "CurrentHand", "position_in_zone": i}
            for i, obj in enumerate(pool)
        ]

        micro_events.append({
            "frame_idx":     event["frame_idx"],
            "page_name":     event["page_name"],
            "action":        "SelectCard",
            "step_index":    step_idx,
            "source_action": source_action,
            "target_card":   strip_zone_fields(copy.deepcopy(target)),
            "selected_cards": [strip_zone_fields(copy.deepcopy(c)) for c in prev_selected],
            "current_hand":  current_hand,
            "state":         event["state"],
            "objects":       copy.deepcopy(context_objects),
        })

    # Final commit step: PlayHand / DiscardHand
    # current_hand = original unselected only (selection is done)
    final_hand = [
        {**obj, "position_in_zone": i}
        for i, obj in enumerate(unselected_base)
    ]
    micro_events.append({
        "frame_idx":     event["frame_idx"],
        "page_name":     event["page_name"],
        "action":        source_action,
        "step_index":    len(selected_ordered),
        "source_action": source_action,
        "target_card":   None,
        "selected_cards": [strip_zone_fields(c) for c in selected_ordered],
        "current_hand":  final_hand,
        "state":         event["state"],
        "objects":       copy.deepcopy(context_objects),
    })

    return micro_events


def _passthrough(event: dict[str, Any]) -> dict[str, Any]:
    """Emit a non-hand event unchanged, normalising action to a single string."""
    actions = event.get("actions") or []
    action_str = actions[0] if actions else (event.get("action") or "Unknown")
    return {
        "frame_idx": event["frame_idx"],
        "page_name": event["page_name"],
        "action":    action_str,
        "state":     event["state"],
        "objects":   event["objects"],
    }


def granularize_event(
    event: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Route one parsed event to decomposition or pass-through."""
    # Support both list-actions (parse_events.py output) and string-action
    raw_actions = event.get("actions") or []
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    action_str = event.get("action") or (raw_actions[0] if raw_actions else "Unknown")

    if action_str in DECOMPOSE_ACTIONS:
        return decompose_hand_event(event, action_str, rng)

    return [_passthrough(event)]


def granularize_run(
    run: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    granular_events: list[dict[str, Any]] = []
    for ev in run["events"]:
        granular_events.extend(granularize_event(ev, rng))
    return {
        "video_id":  run["video_id"],
        "run_index": run["run_index"],
        "events":    granular_events,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_runs(src_root: Path) -> list[tuple[str, int, Path]]:
    """Return [(video_id, run_index, json_path)] for every run file."""
    results: list[tuple[str, int, Path]] = []
    if not src_root.exists():
        return results
    for partition in sorted(src_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for run_file in sorted(partition.glob("run_*.json")):
            stem = run_file.stem  # "run_000"
            try:
                run_idx = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            results.append((video_id, run_idx, run_file))
    return results


def write_run(run: dict[str, Any], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / ("run_%03d.json" % run["run_index"])
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    run_%03d.json  (%d events)" % (run["run_index"], len(run["events"])))


def write_config(dst_root: Path, src_root: Path) -> None:
    config = {
        "schema_version": "1.0.0",
        "source_directory": src_root.as_posix(),
        "output_directory": dst_root.as_posix(),
        "partition_format": "video_id={video_id}",
        "file_name": "run_{index:03d}.json",
        "decompose_actions": sorted(DECOMPOSE_ACTIONS),
        "note": (
            "PlayHand and DiscardHand events are expanded into SelectCard steps "
            "ordered by the player's original card-selection order (position_in_zone "
            "in CurrentHandSelected), followed by a final commit action. "
            "The unselected card pool in current_hand is shuffled at each step."
        ),
        "event_fields": {
            "all_events": {
                "frame_idx":  "int — source frame number",
                "page_name":  "str — game UI page",
                "action":     "str — SelectCard | PlayHand | DiscardHand | StartNewRun | …",
                "state":      "object — parsed OCR game state",
                "objects":    "[object] — context objects excluding hand cards",
            },
            "SelectCard_only": {
                "step_index":    "int — 0-based index within the selection sequence",
                "source_action": "str — PlayHand | DiscardHand (the original action being decomposed)",
                "target_card":   "object — the card being selected at this step",
                "selected_cards": "[object] — cards selected in prior steps (in selection order)",
                "current_hand":  "[object] — shuffled pool of not-yet-selected cards",
            },
            "commit_step_only": {
                "step_index":    "int — equals number of selected cards (final step)",
                "source_action": "str — PlayHand | DiscardHand",
                "target_card":   "null",
                "selected_cards": "[object] — all selected cards in selection order",
                "current_hand":  "[object] — remaining unselected cards (original CurrentHand)",
            },
        },
        "card_object_fields": {
            "slot_id":          "int",
            "class_id":         "int — 0-399; see data/class_map.csv",
            "object_type":      "str — card | deck | joker | consumable | …",
            "card":             "null | {rank, rank_index, suit, suit_index, is_ace, is_face}",
            "edition":          "null | str — e_foil | e_holo | e_negative | e_polychrome",
            "modifier":         "null | str — m_bonus | m_glass | m_gold | m_lucky | m_mult | m_steel | m_stone | m_wild",
            "seal":             "null | str — blue_seal | gold_seal | purple_seal | red_seal",
            "stickers":         "[str] — rental | perishable | eternal (jokers only)",
            "is_debuffed":      "bool",
        },
    }
    dst_root.mkdir(parents=True, exist_ok=True)
    out = dst_root / "config.json"
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote config -> %s" % out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Granularize parsed Balatro runs into SelectCard micro-actions."
    )
    ap.add_argument("--src", type=Path, default=Path("data/parsed"),
                    help="Root of parsed run files (default: data/parsed)")
    ap.add_argument("--dst", type=Path, default=Path("data/granularized"),
                    help="Root for granularized output (default: data/granularized)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for shuffling the unselected card pool (default: 42)")
    args = ap.parse_args(argv)

    src_root: Path = args.src
    dst_root: Path = args.dst
    rng = random.Random(args.seed)

    runs = find_runs(src_root)
    if not runs:
        print("no run_*.json files found under %s" % src_root, file=sys.stderr)
        sys.exit(1)

    print("found %d run file(s)\n" % len(runs))

    current_video: str | None = None
    for video_id, run_idx, run_path in runs:
        if video_id != current_video:
            print("video_id=%s" % video_id)
            current_video = video_id

        with open(run_path, encoding="utf-8") as fh:
            run = json.load(fh)

        print("  run_%03d  (%d parsed events)" % (run_idx, len(run["events"])))
        granular = granularize_run(run, rng)

        dst_dir = dst_root / ("video_id=%s" % video_id)
        write_run(granular, dst_dir)

    write_config(dst_root, src_root)
    print("\ndone.")


if __name__ == "__main__":
    main()
