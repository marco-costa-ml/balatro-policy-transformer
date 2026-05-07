#!/usr/bin/env python3
"""
granularize.py
==============
Reads  data/parsed/video_id=*/run_*.json
Writes data/granularized/video_id=*/run_*.json

Granularization goals:
- One action per output step.
- Unified selected_object abstraction.
- Decompose selection-heavy actions into SelectCard micro-steps + commit step.
- Synthesize SWAP_i_j steps by comparing CurrentJokersAll/CurrentJokers ordering
  between boundary events.

This stage is intentionally pre-tensorization: it keeps rich object snapshots,
selection context, and source-event provenance.

Usage:
    python granularize.py [--src data/parsed] [--dst data/granularized] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLD_HAND_ZONE = "CurrentHand"
OLD_HAND_SELECTED_ZONE = "CurrentHandSelected"
NEW_HAND_ZONE = "CurrentHandOrPackOfferings"
NEW_HAND_SELECTED_ZONE = "CurrentHandOrPackOfferingsSelected"

HAND_POOL_ZONES = {OLD_HAND_ZONE, NEW_HAND_ZONE}
HAND_SELECTED_ZONES = {OLD_HAND_SELECTED_ZONE, NEW_HAND_SELECTED_ZONE}

DECOMPOSE_HAND_ACTIONS = {"PlayHand", "DiscardHand"}

# Conditional granularization list from assignment/spec.
REQUIRES_CARD_SELECTION = {
    249, 251, 252, 259, 263, 264,
    298, 299, 300, 302, 304, 305,
    309, 310, 311, 312, 313, 314,
    315, 317, 319,
}

# Exactly one selected object required by schema for these actions.
SINGLE_SELECTED_OBJECT_ACTIONS = {
    "UseConsumable",
    "SelectCard",
    "SelectPackItem",
    "BuyAndUseShopConsumable",
    "BuyShopItem",
    "SellItem",
}

# Actions that are excluded from swap tracking logic.
SWAP_EXCLUDED_ACTIONS = {
    "StartNewRun",
    "SkipBlind",
    "SelectCard",
    "RerollShop",
    "RerollBossBlind",
    "SkipPack",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_by_zone(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    zones: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        zones.setdefault(str(obj.get("zone", "Unknown")), []).append(obj)
    return zones


def ordered_zone(zones: dict[str, list[dict[str, Any]]], zone_name: str) -> list[dict[str, Any]]:
    return sorted(
        zones.get(zone_name, []),
        key=lambda o: (int(o.get("position_in_zone", 0)), int(o.get("slot_id", -1))),
    )


def strip_zone_fields(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in ("zone", "position_in_zone")}


def get_action_str(event: dict[str, Any]) -> str:
    if isinstance(event.get("action"), str):
        return event["action"]
    actions = event.get("actions") or []
    if isinstance(actions, list) and actions:
        return str(actions[0])
    return "Unknown"


def get_primary_action_meta(event: dict[str, Any]) -> tuple[str, str | None]:
    """
    Return primary action type + optional subtype.

    Prefers parsed action_details[0] when available.
    """
    details = event.get("action_details")
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict) and d.get("type") is not None:
                subtype = d.get("subtype")
                subtype_str = str(subtype) if subtype is not None else None
                return str(d["type"]), subtype_str
    return get_action_str(event), None


def parse_action_base_and_index(action: str) -> tuple[str, int | None]:
    """
    Supports: Foo, Foo_3, Foo(3), Foo[3].
    Returns (base, idx_or_none).
    """
    action = action.strip()
    m = re.fullmatch(r"([A-Za-z]+)[_\[(]?(\d+)?\]?\)?", action)
    if not m:
        return action, None
    base = m.group(1)
    idx = int(m.group(2)) if m.group(2) is not None else None
    return base, idx


def normalized_action(base: str, idx: int | None = None) -> str:
    return f"{base}_{idx}" if idx is not None else base


def classify_selected_role(base_action: str) -> str:
    if base_action == "UseConsumable":
        return "consumable"
    if base_action == "SelectCard":
        return "hand_card"
    if base_action == "SelectPackItem":
        return "pack_item"
    if base_action == "BuyAndUseShopConsumable":
        return "consumable"
    if base_action == "BuyShopItem":
        return "shop_item"
    if base_action == "SellItem":
        return "inventory_item"
    if base_action == "SWAP":
        return "joker_slot_pair"
    return "none"


def pick_by_index(candidates: list[dict[str, Any]], idx: int | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if idx is None:
        return candidates[0]
    if 0 <= idx < len(candidates):
        return candidates[idx]
    return None


def pick_selected_object_for_action(
    base_action: str,
    idx: int | None,
    zones: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    # Preferred selected-zones first, then visible pools.
    if base_action == "UseConsumable":
        return pick_by_index(
            ordered_zone(zones, "CurrentConsumablesSelected")
            or ordered_zone(zones, "CurrentConsumables"),
            idx,
        )

    if base_action == "SelectCard":
        return pick_by_index(
            ordered_zone(zones, NEW_HAND_SELECTED_ZONE)
            or ordered_zone(zones, OLD_HAND_SELECTED_ZONE)
            or ordered_zone(zones, "TarotSpectralHandSelected")
            or ordered_zone(zones, NEW_HAND_ZONE)
            or ordered_zone(zones, OLD_HAND_ZONE),
            idx,
        )

    if base_action == "SelectPackItem":
        return pick_by_index(
            ordered_zone(zones, "CurrentPackSelected")
            or ordered_zone(zones, NEW_HAND_SELECTED_ZONE)
            or ordered_zone(zones, "CurrentPack")
            or ordered_zone(zones, NEW_HAND_ZONE),
            idx,
        )

    if base_action == "BuyAndUseShopConsumable":
        return pick_by_index(
            ordered_zone(zones, "ShopOfferingsSelected")
            or ordered_zone(zones, "ShopOfferings"),
            idx,
        )

    if base_action == "BuyShopItem":
        return pick_by_index(
            ordered_zone(zones, "ShopOfferingsSelected")
            or ordered_zone(zones, "ShopOfferings"),
            idx,
        )

    if base_action == "SellItem":
        selected = ordered_zone(zones, "CurrentJokersSelected") + ordered_zone(
            zones, "CurrentConsumablesSelected"
        )
        if selected:
            return pick_by_index(selected, idx)
        inventory = ordered_zone(zones, "CurrentJokers") + ordered_zone(zones, "CurrentConsumables")
        return pick_by_index(inventory, idx)

    return None


def to_selected_object(obj: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    if obj is None:
        return None
    slot_id = obj.get("slot_id")
    return {
        "role": role,
        "slot_id": slot_id,
        "object": strip_zone_fields(copy.deepcopy(obj)),
    }


def canonical_zones_snapshot(zones: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Stable schema-facing zone names for downstream tensorization/masking."""
    current_pool = ordered_zone(zones, NEW_HAND_ZONE) or ordered_zone(zones, OLD_HAND_ZONE)
    selected_pool = ordered_zone(zones, NEW_HAND_SELECTED_ZONE) or ordered_zone(zones, OLD_HAND_SELECTED_ZONE)

    return {
        "current_hand_or_pack": copy.deepcopy(current_pool),
        "selected_cards": copy.deepcopy(selected_pool),
        "current_jokers": copy.deepcopy(ordered_zone(zones, "CurrentJokers")),
        "current_jokers_all": copy.deepcopy(
            ordered_zone(zones, "CurrentJokersAll") or ordered_zone(zones, "CurrentJokers")
        ),
        "current_consumables": copy.deepcopy(ordered_zone(zones, "CurrentConsumables")),
        "shop_offerings": copy.deepcopy(ordered_zone(zones, "ShopOfferings")),
        "current_pack": copy.deepcopy(ordered_zone(zones, "CurrentPack")),
        "blind_offerings": copy.deepcopy(ordered_zone(zones, "BlindOffering")),
        "blind_offerings_next": copy.deepcopy(ordered_zone(zones, "BlindOfferingsNext")),
        "offered_tag": copy.deepcopy(ordered_zone(zones, "OfferedTag")),
        "tarot_spectral_hand": copy.deepcopy(ordered_zone(zones, "TarotSpectralHand")),
        "tarot_spectral_hand_selected": copy.deepcopy(
            ordered_zone(zones, "TarotSpectralHandSelected")
        ),
    }


def make_step(
    *,
    event: dict[str, Any],
    source_event_index: int,
    micro_index: int,
    action: str,
    action_subtype: str | None,
    source_action: str,
    source_action_subtype: str | None,
    selected_object: dict[str, Any] | None,
    selected_cards: list[dict[str, Any]] | None = None,
    current_hand_or_pack: list[dict[str, Any]] | None = None,
    objects: list[dict[str, Any]] | None = None,
    source_kind: str = "pass_through",
) -> dict[str, Any]:
    zones = group_by_zone(event.get("objects", []))
    step = {
        "frame_idx": event["frame_idx"],
        "page_name": event["page_name"],
        "action": action,
        "action_subtype": action_subtype,
        "source_action": source_action,
        "source_action_subtype": source_action_subtype,
        "source_event_index": source_event_index,
        "micro_index": micro_index,
        "source_kind": source_kind,
        "selected_object": selected_object,
        "state": copy.deepcopy(event["state"]),
        # Keep context objects separate from the pool lists emitted below.
        "objects": copy.deepcopy(objects if objects is not None else event.get("objects", [])),
        "zones": canonical_zones_snapshot(zones),
    }

    if selected_cards is not None:
        step["selected_cards"] = copy.deepcopy(selected_cards)
    if current_hand_or_pack is not None:
        # Include old field name for backwards compatibility.
        step["current_hand_or_pack"] = copy.deepcopy(current_hand_or_pack)
        step["current_hand"] = copy.deepcopy(current_hand_or_pack)

    return step


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def get_hand_and_selected_zones(
    zones: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    selected = ordered_zone(zones, NEW_HAND_SELECTED_ZONE)
    if selected:
        base_pool = ordered_zone(zones, NEW_HAND_ZONE)
        return base_pool, selected, NEW_HAND_ZONE

    selected = ordered_zone(zones, OLD_HAND_SELECTED_ZONE)
    base_pool = ordered_zone(zones, OLD_HAND_ZONE)
    return base_pool, selected, OLD_HAND_ZONE


def extract_context_without_hand(event_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in event_objects
        if obj.get("zone") not in HAND_POOL_ZONES and obj.get("zone") not in HAND_SELECTED_ZONES
    ]


def build_select_card_sequence(
    *,
    event: dict[str, Any],
    source_event_index: int,
    source_action: str,
    source_action_subtype: str | None,
    selected_ordered: list[dict[str, Any]],
    unselected_base: list[dict[str, Any]],
    pool_zone_name: str,
    context_objects: list[dict[str, Any]],
    rng: random.Random,
    start_micro_index: int,
) -> list[dict[str, Any]]:
    micro_steps: list[dict[str, Any]] = []

    for step_idx, target in enumerate(selected_ordered):
        prev_selected = selected_ordered[:step_idx]
        future_selected = selected_ordered[step_idx + 1 :]

        # Keep same semantics as existing implementation: shuffle unresolved pool.
        pool = copy.deepcopy(unselected_base + future_selected)
        rng.shuffle(pool)

        current_pool = [
            {**obj, "zone": pool_zone_name, "position_in_zone": i}
            for i, obj in enumerate(pool)
        ]

        micro_steps.append(
            make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=start_micro_index + step_idx,
                action="SelectCard",
                action_subtype=None,
                source_action=source_action,
                source_action_subtype=source_action_subtype,
                selected_object=to_selected_object(target, "hand_card"),
                selected_cards=[strip_zone_fields(copy.deepcopy(c)) for c in prev_selected],
                current_hand_or_pack=current_pool,
                objects=context_objects,
                source_kind="select",
            )
        )

    return micro_steps


def decompose_hand_action(
    event: dict[str, Any],
    source_event_index: int,
    source_action: str,
    source_action_subtype: str | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    zones = group_by_zone(event["objects"])
    unselected_base, selected_ordered, pool_zone_name = get_hand_and_selected_zones(zones)
    context_objects = extract_context_without_hand(event["objects"])

    if not selected_ordered:
        # No selected cards: fall back to single step.
        return [
            make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=0,
                action=source_action,
                action_subtype=source_action_subtype,
                source_action=source_action,
                source_action_subtype=source_action_subtype,
                selected_object=None,
                objects=event["objects"],
            )
        ]

    micro_steps = build_select_card_sequence(
        event=event,
        source_event_index=source_event_index,
        source_action=source_action,
        source_action_subtype=source_action_subtype,
        selected_ordered=selected_ordered,
        unselected_base=unselected_base,
        pool_zone_name=pool_zone_name,
        context_objects=context_objects,
        rng=rng,
        start_micro_index=0,
    )

    final_pool = [
        {**obj, "zone": pool_zone_name, "position_in_zone": i}
        for i, obj in enumerate(copy.deepcopy(unselected_base))
    ]

    micro_steps.append(
        make_step(
            event=event,
            source_event_index=source_event_index,
            micro_index=len(selected_ordered),
            action=source_action,
            action_subtype=source_action_subtype,
            source_action=source_action,
            source_action_subtype=source_action_subtype,
            selected_object=None,
            selected_cards=[strip_zone_fields(copy.deepcopy(c)) for c in selected_ordered],
            current_hand_or_pack=final_pool,
            objects=context_objects,
            source_kind="commit",
        )
    )

    return micro_steps


def parent_class_id_for(obj: dict[str, Any] | None) -> int | None:
    if not obj:
        return None
    # Some datasets already expose parent_class_id; otherwise class_id is the parent id.
    if obj.get("parent_class_id") is not None:
        try:
            return int(obj["parent_class_id"])
        except (TypeError, ValueError):
            return None
    if obj.get("class_id") is not None:
        try:
            return int(obj["class_id"])
        except (TypeError, ValueError):
            return None
    return None


def selected_cards_for_consumable_or_pack(
    event: dict[str, Any],
    base_action: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str] | None:
    zones = group_by_zone(event["objects"])

    if base_action == "UseConsumable":
        if event.get("page_name") == "In_TarotSpectral_Pack":
            selected = ordered_zone(zones, "TarotSpectralHandSelected")
            pool = ordered_zone(zones, "TarotSpectralHand")
            return pool, selected, "TarotSpectralHand"

        if event.get("page_name") == "In_Blind":
            selected = ordered_zone(zones, NEW_HAND_SELECTED_ZONE) or ordered_zone(zones, OLD_HAND_SELECTED_ZONE)
            pool = ordered_zone(zones, NEW_HAND_ZONE) or ordered_zone(zones, OLD_HAND_ZONE)
            return pool, selected, NEW_HAND_ZONE

        return None

    if base_action == "SelectPackItem":
        selected = ordered_zone(zones, "TarotSpectralHandSelected")
        pool = ordered_zone(zones, "TarotSpectralHand")
        return pool, selected, "TarotSpectralHand"

    return None


def decompose_conditional_selection_action(
    event: dict[str, Any],
    source_event_index: int,
    base_action: str,
    base_action_subtype: str | None,
    idx: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    zones = group_by_zone(event["objects"])
    selected_target = pick_selected_object_for_action(base_action, idx, zones)
    target_parent_class = parent_class_id_for(selected_target)

    context_objects = extract_context_without_hand(event["objects"])
    action_label = normalized_action(base_action, idx)

    # If target doesn't require card selection, just emit the action step.
    if target_parent_class not in REQUIRES_CARD_SELECTION:
        role = classify_selected_role(base_action)
        return [
            make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=0,
                action=action_label,
                action_subtype=base_action_subtype,
                source_action=base_action,
                source_action_subtype=base_action_subtype,
                selected_object=to_selected_object(selected_target, role),
                objects=event["objects"],
            )
        ]

    pools = selected_cards_for_consumable_or_pack(event, base_action)
    if not pools:
        role = classify_selected_role(base_action)
        return [
            make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=0,
                action=action_label,
                action_subtype=base_action_subtype,
                source_action=base_action,
                source_action_subtype=base_action_subtype,
                selected_object=to_selected_object(selected_target, role),
                objects=event["objects"],
            )
        ]

    unselected_base, selected_cards, pool_zone = pools

    micro_steps = build_select_card_sequence(
        event=event,
        source_event_index=source_event_index,
        source_action=base_action,
        source_action_subtype=base_action_subtype,
        selected_ordered=selected_cards,
        unselected_base=unselected_base,
        pool_zone_name=pool_zone,
        context_objects=context_objects,
        rng=rng,
        start_micro_index=0,
    )

    role = classify_selected_role(base_action)
    micro_steps.append(
        make_step(
            event=event,
            source_event_index=source_event_index,
            micro_index=len(selected_cards),
            action=action_label,
            action_subtype=base_action_subtype,
            source_action=base_action,
            source_action_subtype=base_action_subtype,
            selected_object=to_selected_object(selected_target, role),
            selected_cards=[strip_zone_fields(copy.deepcopy(c)) for c in selected_cards],
            current_hand_or_pack=None,
            objects=context_objects,
            source_kind="commit",
        )
    )

    return micro_steps


def passthrough_event(
    event: dict[str, Any],
    source_event_index: int,
    action: str,
    action_subtype: str | None,
) -> list[dict[str, Any]]:
    base, idx = parse_action_base_and_index(action)
    zones = group_by_zone(event["objects"])
    selected_obj = None
    if base in SINGLE_SELECTED_OBJECT_ACTIONS:
        selected_obj = to_selected_object(
            pick_selected_object_for_action(base, idx, zones),
            classify_selected_role(base),
        )

    return [
        make_step(
            event=event,
            source_event_index=source_event_index,
            micro_index=0,
            action=normalized_action(base, idx),
            action_subtype=action_subtype,
            source_action=base,
            source_action_subtype=action_subtype,
            selected_object=selected_obj,
            objects=event["objects"],
        )
    ]


# ---------------------------------------------------------------------------
# SWAP synthesis
# ---------------------------------------------------------------------------

def joker_slot_order(event: dict[str, Any]) -> list[int]:
    zones = group_by_zone(event.get("objects", []))
    jokers = ordered_zone(zones, "CurrentJokersAll") or ordered_zone(zones, "CurrentJokers")
    out: list[int] = []
    for obj in jokers:
        slot_id = obj.get("slot_id")
        if isinstance(slot_id, int):
            out.append(slot_id)
    return out


def swap_pairs_to_transform(start: list[int], end: list[int]) -> list[tuple[int, int]]:
    """
    Reconcile set differences first, then only swap over shared jokers.
    Algorithm matches the provided pseudocode.
    """
    start_shared = [x for x in start if x in end]
    end_shared = [x for x in end if x in start]

    current = start_shared.copy()
    actions: list[tuple[int, int]] = []

    for i in range(len(current)):
        if i >= len(end_shared):
            break
        if current[i] != end_shared[i]:
            try:
                j = current.index(end_shared[i])
            except ValueError:
                continue
            actions.append((i, j))
            current[i], current[j] = current[j], current[i]

    return actions


def synthesize_swap_steps(
    *,
    event: dict[str, Any],
    source_event_index: int,
    source_action: str,
    source_action_subtype: str | None,
    prev_joker_order: list[int],
    curr_joker_order: list[int],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pairs = swap_pairs_to_transform(prev_joker_order, curr_joker_order)
    for micro_idx, (i, j) in enumerate(pairs):
        selected_object = {
            "role": "joker_slot_pair",
            "slot_id": None,
            "pair": [i, j],
            "object": {"left_index": i, "right_index": j},
        }
        steps.append(
            make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=-(len(pairs) - micro_idx),
                action=f"SWAP_{i}_{j}",
                action_subtype=None,
                source_action=source_action,
                source_action_subtype=source_action_subtype,
                selected_object=selected_object,
                objects=event["objects"],
                source_kind="swap_synth",
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Main granularization routing
# ---------------------------------------------------------------------------

def granularize_event(
    event: dict[str, Any],
    source_event_index: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    action_raw, action_subtype = get_primary_action_meta(event)
    base_action, idx = parse_action_base_and_index(action_raw)

    if base_action in DECOMPOSE_HAND_ACTIONS:
        return decompose_hand_action(
            event,
            source_event_index,
            base_action,
            action_subtype,
            rng,
        )

    if base_action in {"UseConsumable", "SelectPackItem", "BuyAndUseShopConsumable"}:
        return decompose_conditional_selection_action(
            event=event,
            source_event_index=source_event_index,
            base_action=base_action,
            base_action_subtype=action_subtype,
            idx=idx,
            rng=rng,
        )

    return passthrough_event(event, source_event_index, action_raw, action_subtype)


def granularize_run(
    run: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    granular_steps: list[dict[str, Any]] = []

    prev_boundary_joker_order: list[int] | None = None

    for source_event_index, ev in enumerate(run["events"]):
        action_raw, action_subtype = get_primary_action_meta(ev)
        base_action, _ = parse_action_base_and_index(action_raw)

        # Synthesized SWAP steps before current boundary action.
        if base_action not in SWAP_EXCLUDED_ACTIONS:
            curr_order = joker_slot_order(ev)
            if prev_boundary_joker_order is not None and curr_order:
                granular_steps.extend(
                    synthesize_swap_steps(
                        event=ev,
                        source_event_index=source_event_index,
                        source_action=base_action,
                        source_action_subtype=action_subtype,
                        prev_joker_order=prev_boundary_joker_order,
                        curr_joker_order=curr_order,
                    )
                )
            if curr_order:
                prev_boundary_joker_order = curr_order

        granular_steps.extend(granularize_event(ev, source_event_index, rng))

    # Attach monotonically increasing step_id at the end.
    for i, step in enumerate(granular_steps):
        step["step_id"] = i

    return {
        "video_id": run["video_id"],
        "run_index": run["run_index"],
        "events": granular_steps,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_runs(src_root: Path) -> list[tuple[str, int, Path]]:
    results: list[tuple[str, int, Path]] = []
    if not src_root.exists():
        return results

    for partition in sorted(src_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for run_file in sorted(partition.glob("run_*.json")):
            try:
                run_idx = int(run_file.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            results.append((video_id, run_idx, run_file))

    return results


def write_run(run: dict[str, Any], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / ("run_%03d.json" % run["run_index"])
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    run_%03d.json  (%d granular steps)" % (run["run_index"], len(run["events"])))


def write_config(dst_root: Path, src_root: Path) -> None:
    config = {
        "schema_version": "2.1.0",
        "source_directory": src_root.as_posix(),
        "output_directory": dst_root.as_posix(),
        "partition_format": "video_id={video_id}",
        "file_name": "run_{index:03d}.json",
        "decompose_actions_always": sorted(DECOMPOSE_HAND_ACTIONS | {"SWAP"}),
        "decompose_actions_conditional": ["UseConsumable", "SelectPackItem", "BuyAndUseShopConsumable"],
        "single_selected_object_actions": sorted(SINGLE_SELECTED_OBJECT_ACTIONS),
        "note": (
            "Every output event is one granular step with one action. PlayHand/DiscardHand are decomposed "
            "into SelectCard micro-steps plus commit; UseConsumable/SelectPackItem/BuyAndUseShopConsumable "
            "are conditionally decomposed based on target class_id; SWAP_i_j steps are synthesized from "
            "joker-order changes between boundary events."
        ),
        "step_fields": {
            "step_id": "int — monotonic index within run",
            "frame_idx": "int — source frame number",
            "page_name": "str — game UI page",
            "action": "str — flattened action label (e.g. PlayHand, BuyShopItem_2, SWAP_1_3)",
            "action_subtype": "null | string — parsed subtype from action_details (e.g. buyvoucher)",
            "source_action": "str — macro action before decomposition",
            "source_action_subtype": "null | string — subtype of source_action",
            "source_event_index": "int — index in parsed run.events",
            "micro_index": "int — index within decomposed source event (negative for pre-action synthesized swaps)",
            "source_kind": "str — pass_through | select | commit | swap_synth",
            "selected_object": "null | {role, slot_id, object} | {role='joker_slot_pair', pair=[i,j], object={left_index,right_index}}",
            "state": "object — parsed OCR state",
            "objects": "[object] — context objects snapshot",
            "zones": "object — canonical zone snapshots for downstream masking/tensorization",
            "selected_cards": "optional [object] — already selected cards in sequence",
            "current_hand_or_pack": "optional [object] — dynamic pool during selection micro-steps",
            "current_hand": "optional [object] — compatibility alias of current_hand_or_pack",
        },
        "zones": {
            "current_hand_or_pack": "[object]",
            "selected_cards": "[object]",
            "current_jokers": "[object]",
            "current_jokers_all": "[object]",
            "current_consumables": "[object]",
            "shop_offerings": "[object]",
            "current_pack": "[object]",
            "blind_offerings": "[object]",
            "blind_offerings_next": "[object]",
            "offered_tag": "[object]",
            "tarot_spectral_hand": "[object]",
            "tarot_spectral_hand_selected": "[object]",
        },
        "selected_object_roles": [
            "hand_card",
            "pack_item",
            "consumable",
            "shop_item",
            "inventory_item",
            "joker_slot_pair",
            "none",
        ],
        "requires_card_selection_class_ids": sorted(REQUIRES_CARD_SELECTION),
        "card_object_fields": {
            "slot_id": "int",
            "class_id": "int — 0-399; see data/class_map.csv",
            "object_type": "str — card | deck | joker | consumable | …",
            "card": "null | {rank, rank_index, suit, suit_index, is_ace, is_face}",
            "edition": "null | str — e_foil | e_holo | e_negative | e_polychrome",
            "modifier": "null | str — m_bonus | m_glass | m_gold | m_lucky | m_mult | m_steel | m_stone | m_wild",
            "seal": "null | str — blue_seal | gold_seal | purple_seal | red_seal",
            "stickers": "[str] — rental | perishable | eternal",
            "is_debuffed": "bool",
        },
    }

    dst_root.mkdir(parents=True, exist_ok=True)
    out = dst_root / "config.json"
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote config -> %s" % out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Granularize parsed Balatro runs into single-action micro-steps."
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/parsed"),
        help="Root of parsed run files (default: data/parsed)",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("data/granularized"),
        help="Root for granularized output (default: data/granularized)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling unresolved selection pools (default: 42)",
    )
    args = ap.parse_args(argv)

    src_root = args.src
    dst_root = args.dst
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
