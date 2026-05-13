#!/usr/bin/env python3
"""
granularize.py
==============
Reads  data/parsed/video_id=*/run_*.json
Writes data/granularized/video_id=*/run_*.json

Granularization is a single-pass per-run transform that follows
``data/masking_schema_disorganized.md`` lines 264-374:

- All zones are normalized: ``FooAll`` -> ``Foo`` on output; ``FooSelected``
  zones are read only to identify targets and discarded after.
- For each parent event we compute a ``target_zone`` + ``target_position``
  pair (the position of the target object in the corresponding ``All``
  zone). Action labels use the per-zone form ``Base_Zone_i`` (e.g.
  ``BuyShopItem_VoucherShopOfferings_0``); base actions without targets
  emit bare labels (e.g. ``PlayHand``).
- ``PlayHand``/``DiscardHand`` always decompose into ``SelectCard``
  micro-steps followed by a commit step. ``UseConsumable`` and
  ``SelectPackItem`` decompose only when the target requires card
  selection (REQUIRES_AT_LEAST_ONE_CARD set / ``selectpackitemtarot``).
- A script-local ``PendingCards`` zone holds cards selected so far in
  the current parent sequence. Every emitted step records its current
  ``pending_cards`` snapshot (also represented as objects with
  ``zone="PendingCards"`` for the model encoder).
- ``SWAP_i_j`` steps are synthesized between events whenever the
  ``CurrentJokers`` order differs from the previous parent's snapshot.
  Each SWAP step records the parent's OCR + populated zones, with
  ``CurrentJokers`` replaced by the local ``last_jokers`` snapshot at
  that exact moment.

Output step schema:

    step_id, frame_idx, page_name,
    source_event_index, micro_index, source_kind,
    source_action, source_action_subtype,
    action, action_subtype,
    target_zone, target_position,
    swap_pair, selected_object, pending_cards,
    state, objects

Usage:
    python granularize.py [--src data/parsed] [--dst data/granularized]
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

GRANULARIZE_SCHEMA_VERSION = "3.0.0"

# Consumable class ids that require at least one playing card to be
# selected. Triggers conditional UseConsumable / SelectPackItem decomposition.
REQUIRES_AT_LEAST_ONE_CARD: frozenset[int] = frozenset(
    {
        249, 251, 252, 259, 263, 264,
        298, 299, 300, 302, 304, 305,
        309, 310, 311, 312, 313, 314,
        315, 317, 319,
    }
)

# Events that DON'T trigger SWAP comparison (per
# granularization_schema.md section 5).
SWAP_SNAPSHOT_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "StartNewRun",
        "SkipBlind",
        "SkipPack",
        "RerollShop",
        "RerollBossBlind",
        "SelectCard",
    }
)

# Events that trigger SWAP synthesis BEFORE being recorded (per spec).
SWAP_TRIGGER_BASES: frozenset[str] = frozenset(
    {
        "DiscardHand",
        "PlayHand",
        "UseConsumable",
        "CashOut",
        "SelectPackItem",
        "BuyAndUseShopConsumable",
        "BuyShopItem",
        "LeaveShop",
        "SellItem",
    }
)

# Target-object resolution table:
#   base_action -> (subtype -> selected_zone_base)
# When subtype is None or absent in the inner dict, fall back to "*".
TARGET_ZONE_BY_SUBTYPE: dict[str, dict[str, str]] = {
    "BuyShopItem": {
        "buyvoucher": "VoucherShopOfferings",
        "buyandopenplanetstandardbuffoonpack": "PackShopOfferings",
        "buyandopentarotspectralpack": "PackShopOfferings",
        "buytopshelfjoker": "TopShelfShopOfferings",
        "buytopshelfconsumable": "TopShelfShopOfferings",
    },
    "SelectPackItem": {
        "selectpackitemtarot": "PackOfferings",
        "selectpackitemcard": "PackOfferings",
        "selectpackitemjoker": "PackOfferings",
        "selectpackitemplanet": "PackOfferings",
        "*": "PackOfferings",
    },
    "SellItem": {
        "selljoker": "CurrentJokers",
        "sellconsumable": "CurrentConsumables",
    },
    "BuyAndUseShopConsumable": {
        "*": "TopShelfShopOfferings",
    },
    "UseConsumable": {
        "*": "CurrentConsumables",
    },
}

# Pool of playing cards for card-selection decomposition.
#   (base_action, subtype) -> (selected_zone_base, pool_zone_base)
# pool_zone_base is the *base* zone name (post-strip-All).
CARD_SELECTION_POOL: dict[tuple[str, str | None], tuple[str, str]] = {
    ("PlayHand", None): ("CurrentHand", "CurrentHand"),
    ("DiscardHand", None): ("CurrentHand", "CurrentHand"),
    # UseConsumable depends on page_name; resolved at runtime.
    # SelectPackItem only for selectpackitemtarot.
    ("SelectPackItem", "selectpackitemtarot"): (
        "TarotSpectralHand",
        "TarotSpectralHand",
    ),
}


# ---------------------------------------------------------------------------
# Parsed event helpers
# ---------------------------------------------------------------------------

def parse_event_action(event: dict[str, Any]) -> tuple[str, str | None]:
    """Return ``(base_action, subtype_or_None)`` from a parsed event."""
    details = event.get("action_details")
    if isinstance(details, list) and details:
        first = details[0]
        if isinstance(first, dict) and first.get("type"):
            base = str(first["type"])
            sub = first.get("subtype")
            return base, (str(sub) if sub else None)
    actions = event.get("actions") or []
    if isinstance(actions, list) and actions:
        return str(actions[0]), None
    if isinstance(event.get("action"), str):
        return str(event["action"]), None
    return "Unknown", None


def collapse_zones(
    parsed_objects: list[dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    """
    Group parsed objects into ``(all_zones, selected_zones, other_zones)``
    keyed by *base* zone name (no ``All`` / ``Selected`` suffix).

    Rules:
    - ``FooAll`` -> base ``Foo`` in ``all_zones``.
    - ``FooSelected`` -> base ``Foo`` in ``selected_zones``.
    - A bare zone ``Foo`` that has a paired ``FooAll`` or ``FooSelected``
      elsewhere is also collapsed into ``all_zones[Foo]`` (deduped by
      ``slot_id``).
    - Anything else goes into ``other_zones`` under its raw name.

    Each list is sorted by ``position_in_zone`` then ``slot_id``.
    """
    raw_groups: dict[str, list[dict[str, Any]]] = {}
    for obj in parsed_objects:
        if not isinstance(obj, dict):
            continue
        zone = obj.get("zone")
        if isinstance(zone, str):
            raw_groups.setdefault(zone, []).append(obj)
    for lst in raw_groups.values():
        lst.sort(
            key=lambda o: (
                int(o.get("position_in_zone") or 0),
                int(o.get("slot_id") or 0),
            )
        )

    selected_bases = {
        z[: -len("Selected")] for z in raw_groups if z.endswith("Selected")
    }
    all_bases = {z[: -len("All")] for z in raw_groups if z.endswith("All")}
    paired_bases = selected_bases | all_bases

    all_zones: dict[str, list[dict[str, Any]]] = {}
    selected_zones: dict[str, list[dict[str, Any]]] = {}
    other_zones: dict[str, list[dict[str, Any]]] = {}

    def _merge_into_all(base: str, lst: list[dict[str, Any]]) -> None:
        existing = all_zones.get(base, [])
        seen = {
            o.get("slot_id")
            for o in existing
            if isinstance(o.get("slot_id"), int)
        }
        merged = list(existing) + [
            o for o in lst if o.get("slot_id") not in seen
        ]
        all_zones[base] = merged

    for zone, lst in raw_groups.items():
        if zone.endswith("Selected"):
            selected_zones[zone[: -len("Selected")]] = lst
        elif zone.endswith("All"):
            _merge_into_all(zone[: -len("All")], lst)
        elif zone in paired_bases:
            _merge_into_all(zone, lst)
        else:
            other_zones[zone] = lst

    return all_zones, selected_zones, other_zones


def strip_object_for_output(obj: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy an object, drop transient fields."""
    out = dict(obj)
    return out


# ---------------------------------------------------------------------------
# Object emission
# ---------------------------------------------------------------------------

# Stable order for the model: dynamic/inventory zones first, then context,
# then PendingCards. The model doesn't rely on this order, but a fixed
# order keeps shards deterministic and easier to inspect.
_ZONE_PRIORITY: tuple[str, ...] = (
    "CurrentHand",
    "CurrentJokers",
    "CurrentConsumables",
    "TarotSpectralHand",
    "PackOfferings",
    "VoucherShopOfferings",
    "PackShopOfferings",
    "TopShelfShopOfferings",
    "BlindOffering",
    "BlindOfferingsNext",
    "OfferedTag",
    "BigBlindTag",
    "BlindToken",
    "CurrentTags",
    "CurrentDeck",
    "CurrentStake",
    "PackConsumableUse",
    "VoucherConsumableRedeemUse",
)
_ZONE_RANK: dict[str, int] = {z: i for i, z in enumerate(_ZONE_PRIORITY)}


def _zone_sort_key(zone: str) -> tuple[int, str]:
    return (_ZONE_RANK.get(zone, len(_ZONE_PRIORITY)), zone)


def emit_objects(
    all_zones: dict[str, list[dict[str, Any]]],
    other_zones: dict[str, list[dict[str, Any]]],
    pending_cards: list[dict[str, Any]],
    *,
    dynamic_overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Emit a flat object list with normalized zone names and renumbered
    ``position_in_zone`` per zone.

    ``dynamic_overrides[zone] = [...]`` replaces a zone's contents
    entirely (used by SWAP synth to swap in the ``last_jokers`` snapshot,
    and by card-selection decomposition to swap in the dynamic pool with
    pending cards already removed).
    """
    out: list[dict[str, Any]] = []
    overrides = dynamic_overrides or {}

    merged_zones: dict[str, list[dict[str, Any]]] = {}
    for zone, lst in all_zones.items():
        merged_zones[zone] = overrides.get(zone, lst)
    for zone, lst in other_zones.items():
        merged_zones[zone] = overrides.get(zone, lst)
    for zone, lst in overrides.items():
        merged_zones.setdefault(zone, lst)
    if pending_cards:
        merged_zones["PendingCards"] = pending_cards

    for zone in sorted(merged_zones, key=_zone_sort_key):
        for i, obj in enumerate(merged_zones[zone]):
            if not isinstance(obj, dict):
                continue
            copied = strip_object_for_output(obj)
            copied["zone"] = zone
            copied["position_in_zone"] = i
            out.append(copied)
    return out


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def _resolve_target_zone_base(
    base_action: str, subtype: str | None
) -> str | None:
    table = TARGET_ZONE_BY_SUBTYPE.get(base_action)
    if not table:
        return None
    if subtype is not None and subtype in table:
        return table[subtype]
    return table.get("*")


def _find_position_in_zone(
    target_obj: dict[str, Any], zone_list: list[dict[str, Any]]
) -> int | None:
    """Find ``target_obj`` in ``zone_list`` by ``slot_id`` (preferred) or identity."""
    if not target_obj:
        return None
    target_slot_id = target_obj.get("slot_id")
    if target_slot_id is not None:
        for i, o in enumerate(zone_list):
            if o.get("slot_id") == target_slot_id:
                return i
    # Fallback: identity tuple
    target_tuple = (
        target_obj.get("class_id"),
        target_obj.get("position_in_zone"),
    )
    for i, o in enumerate(zone_list):
        if (o.get("class_id"), o.get("position_in_zone")) == target_tuple:
            return i
    return None


def resolve_target(
    base_action: str,
    subtype: str | None,
    all_zones: dict[str, list[dict[str, Any]]],
    selected_zones: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, int | None, dict[str, Any] | None]:
    """
    Return ``(target_zone_base, target_position, target_object_copy)``
    for events that have a target object; ``(None, None, None)`` otherwise.

    The target object is identified as ``selected_zones[base][0]`` per
    the schema, then located by ``slot_id`` in the matching All zone to
    produce ``target_position``.
    """
    zone_base = _resolve_target_zone_base(base_action, subtype)
    if zone_base is None:
        return None, None, None

    sel = selected_zones.get(zone_base) or []
    if not sel:
        return None, None, None
    target_obj = sel[0]

    all_list = all_zones.get(zone_base) or []
    pos = _find_position_in_zone(target_obj, all_list)
    if pos is None:
        # Target not in All zone (data inconsistency). Try the Selected list
        # itself for identification purposes only; mark position as unknown.
        return zone_base, None, dict(target_obj)
    return zone_base, pos, dict(all_list[pos])


# ---------------------------------------------------------------------------
# Card selection decomposition
# ---------------------------------------------------------------------------

def _card_pool_for_event(
    base_action: str,
    subtype: str | None,
    page_name: str | None,
    target_class_id: int | None,
) -> tuple[str, str] | None:
    """
    Decide whether this event decomposes into SelectCard sub-events, and
    if so return ``(selected_zone_base, pool_zone_base)``.

    Returns None if no decomposition is needed.
    """
    if base_action in {"PlayHand", "DiscardHand"}:
        return ("CurrentHand", "CurrentHand")

    if base_action == "UseConsumable":
        if target_class_id not in REQUIRES_AT_LEAST_ONE_CARD:
            return None
        if page_name == "In_TarotSpectral_Pack":
            return ("TarotSpectralHand", "TarotSpectralHand")
        if page_name == "In_Blind":
            return ("CurrentHand", "CurrentHand")
        return None

    if base_action == "SelectPackItem" and subtype == "selectpackitemtarot":
        return ("TarotSpectralHand", "TarotSpectralHand")

    return None


# ---------------------------------------------------------------------------
# SWAP synthesis
# ---------------------------------------------------------------------------

def _slot_key(obj: dict[str, Any]) -> Any:
    """Stable identity for joker objects (slot_id first, then class_id)."""
    slot_id = obj.get("slot_id")
    if slot_id is not None:
        return ("slot", slot_id)
    return (
        "cid",
        obj.get("class_id"),
        obj.get("position_in_zone"),
        obj.get("edition"),
    )


def compute_swap_pairs(
    last_jokers: list[dict[str, Any]],
    curr_jokers: list[dict[str, Any]],
) -> tuple[list[tuple[int, int]], list[list[dict[str, Any]]]]:
    """
    Greedy in-place transform of ``last_jokers`` -> ``curr_jokers``,
    constrained to the *shared* subset (set-reconciled first).

    Returns:
      - ``pairs``: list of ``(i, j)`` indices over the shared subset.
      - ``snapshots``: list of joker-list snapshots, one per swap, taken
        IMMEDIATELY BEFORE each swap is applied (so snapshots[k] is what
        ``CurrentJokers`` should look like on the k-th SWAP step).
    """
    last_keys = [_slot_key(o) for o in last_jokers]
    curr_keys = [_slot_key(o) for o in curr_jokers]
    last_set = set(last_keys)
    curr_set = set(curr_keys)
    shared = last_set & curr_set

    # Maintain a parallel list of joker *objects* keyed by stable identity.
    by_key = {k: o for k, o in zip(last_keys, last_jokers)}
    for k, o in zip(curr_keys, curr_jokers):
        by_key.setdefault(k, o)

    current_keys = [k for k in last_keys if k in shared]
    target_keys = [k for k in curr_keys if k in shared]

    pairs: list[tuple[int, int]] = []
    snapshots: list[list[dict[str, Any]]] = []

    for i in range(len(current_keys)):
        if i >= len(target_keys):
            break
        if current_keys[i] == target_keys[i]:
            continue
        try:
            j = current_keys.index(target_keys[i], i + 1)
        except ValueError:
            continue
        snapshot = [by_key[k] for k in current_keys]
        pairs.append((i, j))
        snapshots.append(snapshot)
        current_keys[i], current_keys[j] = current_keys[j], current_keys[i]

    return pairs, snapshots


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------

def _make_step(
    *,
    event: dict[str, Any],
    source_event_index: int,
    micro_index: int,
    source_kind: str,
    source_action: str,
    source_action_subtype: str | None,
    action: str,
    action_subtype: str | None,
    target_zone: str | None,
    target_position: int | None,
    swap_pair: tuple[int, int] | list[int] | None,
    target_object: dict[str, Any] | None,
    pending_cards: list[dict[str, Any]],
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frame_idx": event.get("frame_idx"),
        "page_name": event.get("page_name"),
        "source_event_index": source_event_index,
        "micro_index": micro_index,
        "source_kind": source_kind,
        "source_action": source_action,
        "source_action_subtype": source_action_subtype,
        "action": action,
        "action_subtype": action_subtype,
        "target_zone": target_zone,
        "target_position": target_position,
        "swap_pair": list(swap_pair) if swap_pair is not None else None,
        "selected_object": (
            {"object": copy.deepcopy(target_object)}
            if target_object is not None
            else None
        ),
        "pending_cards": [copy.deepcopy(c) for c in pending_cards],
        "state": copy.deepcopy(event.get("state") or {}),
        "objects": objects,
    }


def _normalized_action_label(
    base_action: str,
    target_zone: str | None,
    target_position: int | None,
    swap_pair: tuple[int, int] | None,
) -> str:
    if swap_pair is not None:
        return f"SWAP_{swap_pair[0]}_{swap_pair[1]}"
    if target_zone is not None and target_position is not None:
        return f"{base_action}_{target_zone}_{target_position}"
    return base_action


# ---------------------------------------------------------------------------
# Main per-event processing
# ---------------------------------------------------------------------------

def _emit_swap_steps(
    event: dict[str, Any],
    source_event_index: int,
    base_action: str,
    subtype: str | None,
    all_zones: dict[str, list[dict[str, Any]]],
    other_zones: dict[str, list[dict[str, Any]]],
    last_jokers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """
    Emit SWAP_synth steps for the (last_jokers, curr_jokers) delta.

    Returns ``(steps, post_swap_jokers)`` where ``post_swap_jokers`` is
    the joker order after applying all swaps (so the caller can update
    ``last_jokers`` to reflect the in-progress reordering). Returns
    ``([], None)`` when no swaps were needed.
    """
    curr_jokers = all_zones.get("CurrentJokers") or []
    pairs, snapshots = compute_swap_pairs(last_jokers, curr_jokers)
    if not pairs:
        return [], None

    steps: list[dict[str, Any]] = []
    for k, ((i, j), snapshot) in enumerate(zip(pairs, snapshots)):
        objects = emit_objects(
            all_zones,
            other_zones,
            pending_cards=[],
            dynamic_overrides={"CurrentJokers": snapshot},
        )
        steps.append(
            _make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=-(len(pairs) - k),
                source_kind="swap_synth",
                source_action=base_action,
                source_action_subtype=subtype,
                action=_normalized_action_label(
                    "SWAP", None, None, (i, j)
                ),
                action_subtype=None,
                target_zone=None,
                target_position=None,
                swap_pair=(i, j),
                target_object=None,
                pending_cards=[],
                objects=objects,
            )
        )
    # post-swap snapshot = the final shared-only ordering
    final_keys = []
    last_keys = [_slot_key(o) for o in last_jokers]
    curr_keys = [_slot_key(o) for o in curr_jokers]
    shared = set(last_keys) & set(curr_keys)
    current_keys = [k for k in last_keys if k in shared]
    target_keys = [k for k in curr_keys if k in shared]
    for i in range(len(current_keys)):
        if i >= len(target_keys):
            break
        if current_keys[i] != target_keys[i]:
            try:
                j = current_keys.index(target_keys[i], i + 1)
            except ValueError:
                continue
            current_keys[i], current_keys[j] = current_keys[j], current_keys[i]
    by_key = {k: o for k, o in zip(last_keys, last_jokers)}
    for k, o in zip(curr_keys, curr_jokers):
        by_key.setdefault(k, o)
    post_swap = [by_key[k] for k in current_keys]
    return steps, post_swap


def _emit_decomposed_event(
    event: dict[str, Any],
    source_event_index: int,
    base_action: str,
    subtype: str | None,
    all_zones: dict[str, list[dict[str, Any]]],
    selected_zones: dict[str, list[dict[str, Any]]],
    other_zones: dict[str, list[dict[str, Any]]],
    target_zone: str | None,
    target_position: int | None,
    target_object: dict[str, Any] | None,
    pool_spec: tuple[str, str] | None,
) -> list[dict[str, Any]]:
    """
    Emit ``SelectCard_<pool>_<i>`` micro-steps + the parent commit step
    for events that decompose. If ``pool_spec`` is None this is a single
    pass-through step.
    """
    page_name = event.get("page_name")

    if pool_spec is None:
        action = _normalized_action_label(
            base_action, target_zone, target_position, None
        )
        objects = emit_objects(all_zones, other_zones, pending_cards=[])
        return [
            _make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=0,
                source_kind="pass_through",
                source_action=base_action,
                source_action_subtype=subtype,
                action=action,
                action_subtype=subtype,
                target_zone=target_zone,
                target_position=target_position,
                swap_pair=None,
                target_object=target_object,
                pending_cards=[],
                objects=objects,
            )
        ]

    selected_base, pool_base = pool_spec
    selected_cards_full = list(selected_zones.get(selected_base) or [])
    pool_list = list(all_zones.get(pool_base) or [])

    # Track cards by stable identity (slot_id) so we can find each in the
    # dynamic pool even after re-numbering.
    pool_by_key: dict[Any, dict[str, Any]] = {
        _slot_key(o): o for o in pool_list
    }
    # Dynamic pool we'll mutate as cards move to PendingCards.
    dynamic_pool_keys: list[Any] = [_slot_key(o) for o in pool_list]
    pending_keys: list[Any] = []

    # Selected-card iteration order: position in the Selected zone.
    selected_cards_full.sort(
        key=lambda o: (
            int(o.get("position_in_zone") or 0),
            int(o.get("slot_id") or 0),
        )
    )

    micro_steps: list[dict[str, Any]] = []
    for micro_idx, sel_card in enumerate(selected_cards_full):
        sel_key = _slot_key(sel_card)
        try:
            sel_pos = dynamic_pool_keys.index(sel_key)
        except ValueError:
            # Card not in the pool (data inconsistency); skip the SelectCard
            # for this card but keep the others so we can still commit.
            continue

        # Build dynamic pool snapshot at this micro-step (pre-removal).
        dyn_pool_snapshot = [pool_by_key[k] for k in dynamic_pool_keys]
        pending_snapshot = [pool_by_key[k] for k in pending_keys]
        objects = emit_objects(
            all_zones,
            other_zones,
            pending_cards=pending_snapshot,
            dynamic_overrides={pool_base: dyn_pool_snapshot},
        )

        action = _normalized_action_label(
            "SelectCard", pool_base, sel_pos, None
        )
        target_object_dyn = dict(pool_by_key[sel_key])
        micro_steps.append(
            _make_step(
                event=event,
                source_event_index=source_event_index,
                micro_index=micro_idx,
                source_kind="select",
                source_action=base_action,
                source_action_subtype=subtype,
                action=action,
                action_subtype=None,
                target_zone=pool_base,
                target_position=sel_pos,
                swap_pair=None,
                target_object=target_object_dyn,
                pending_cards=pending_snapshot,
                objects=objects,
            )
        )

        # Move the card from the dynamic pool to pending.
        dynamic_pool_keys.pop(sel_pos)
        pending_keys.append(sel_key)

    # Parent / commit step: pool has cards removed, pending has all selected.
    final_pool = [pool_by_key[k] for k in dynamic_pool_keys]
    final_pending = [pool_by_key[k] for k in pending_keys]
    objects = emit_objects(
        all_zones,
        other_zones,
        pending_cards=final_pending,
        dynamic_overrides={pool_base: final_pool},
    )
    action = _normalized_action_label(
        base_action, target_zone, target_position, None
    )
    micro_steps.append(
        _make_step(
            event=event,
            source_event_index=source_event_index,
            micro_index=len(selected_cards_full),
            source_kind="commit",
            source_action=base_action,
            source_action_subtype=subtype,
            action=action,
            action_subtype=subtype,
            target_zone=target_zone,
            target_position=target_position,
            swap_pair=None,
            target_object=target_object,
            pending_cards=final_pending,
            objects=objects,
        )
    )

    return micro_steps


# ---------------------------------------------------------------------------
# Run-level loop
# ---------------------------------------------------------------------------

def granularize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Single-pass run granularizer per masking_schema_disorganized.md 264-374."""
    events = run.get("events") or []
    last_jokers: list[dict[str, Any]] = []

    emitted: list[dict[str, Any]] = []

    def _next_recordable_base_action(start_idx: int) -> str | None:
        """
        Return the next parsed event base action after ``start_idx``.

        This is used to suppress known parser glitches where a spurious
        ``SkipPack`` is inserted immediately before a real ``SelectPackItem``.
        """
        for j in range(start_idx + 1, len(events)):
            nxt = events[j]
            if not isinstance(nxt, dict):
                continue
            next_base, _ = parse_event_action(nxt)
            return next_base
        return None

    for source_event_index, event in enumerate(events):
        base_action, subtype = parse_event_action(event)
        if (
            base_action == "SkipPack"
            and _next_recordable_base_action(source_event_index) == "SelectPackItem"
        ):
            # Data bug: it's logically impossible to skip a pack and then pick
            # a pack item in the immediately following recorded event.
            # Drop only this spurious SkipPack event.
            continue
        all_zones, selected_zones, other_zones = collapse_zones(
            event.get("objects") or []
        )

        # --- Step 1: SWAP synthesis ---
        if base_action in SWAP_TRIGGER_BASES and last_jokers:
            swap_steps, post_swap = _emit_swap_steps(
                event,
                source_event_index,
                base_action,
                subtype,
                all_zones,
                other_zones,
                last_jokers,
            )
            if swap_steps:
                emitted.extend(swap_steps)
                if post_swap is not None:
                    last_jokers = post_swap

        # --- Step 2: target resolution ---
        target_zone, target_position, target_obj = resolve_target(
            base_action, subtype, all_zones, selected_zones
        )

        # --- Step 3: card-selection decomposition? ---
        target_class_id: int | None = None
        if isinstance(target_obj, dict):
            cid = target_obj.get("class_id")
            if isinstance(cid, int):
                target_class_id = cid
        pool_spec = _card_pool_for_event(
            base_action,
            subtype,
            event.get("page_name"),
            target_class_id,
        )

        # --- Step 4: emit micro-steps + parent ---
        emitted.extend(
            _emit_decomposed_event(
                event,
                source_event_index,
                base_action,
                subtype,
                all_zones,
                selected_zones,
                other_zones,
                target_zone,
                target_position,
                target_obj,
                pool_spec,
            )
        )

        # --- Step 5: update last_jokers ---
        if base_action not in SWAP_SNAPSHOT_EXCLUSIONS:
            curr_jokers = all_zones.get("CurrentJokers") or []
            if curr_jokers:
                last_jokers = list(curr_jokers)

    # Assign monotonic step ids.
    for i, step in enumerate(emitted):
        step["step_id"] = i

    return {
        "video_id": run.get("video_id"),
        "run_index": run.get("run_index"),
        "events": emitted,
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
    out.write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "    run_%03d.json  (%d granular steps)"
        % (run["run_index"], len(run["events"]))
    )


def write_config(dst_root: Path, src_root: Path, stats: collections.Counter) -> None:
    config = {
        "schema_version": GRANULARIZE_SCHEMA_VERSION,
        "source_directory": src_root.as_posix(),
        "output_directory": dst_root.as_posix(),
        "partition_format": "video_id={video_id}",
        "file_name": "run_{index:03d}.json",
        "zone_normalization": {
            "FooAll": "Foo",
            "FooSelected": "<discarded>",
            "bare_paired_with_All_or_Selected": "collapsed_into_Foo",
        },
        "action_label_format": (
            "Bare base for no-target families (e.g. PlayHand). "
            "Per-zone indexed `Base_Zone_i` for indexed families. "
            "Pair-indexed `SWAP_i_j` for joker swaps."
        ),
        "step_fields": {
            "step_id": "int",
            "frame_idx": "int",
            "page_name": "str",
            "source_event_index": "int",
            "micro_index": "int (negative for pre-event swap_synth)",
            "source_kind": "pass_through | select | commit | swap_synth",
            "source_action": "str",
            "source_action_subtype": "str | null",
            "action": "str",
            "action_subtype": "str | null",
            "target_zone": "str | null",
            "target_position": "int | null",
            "swap_pair": "[int, int] | null",
            "selected_object": "{object: <copy>} | null",
            "pending_cards": "[object, ...]",
            "state": "OCR dict",
            "objects": "[object, ...] - normalized zones; includes PendingCards",
        },
        "requires_at_least_one_card_class_ids": sorted(REQUIRES_AT_LEAST_ONE_CARD),
        "swap_trigger_bases": sorted(SWAP_TRIGGER_BASES),
        "swap_snapshot_exclusions": sorted(SWAP_SNAPSHOT_EXCLUSIONS),
        "target_zone_table": {
            base: dict(table) for base, table in TARGET_ZONE_BY_SUBTYPE.items()
        },
        "stats": {f"{k1}.{k2}": n for (k1, k2), n in stats.items()},
    }
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote config -> {(dst_root / 'config.json').as_posix()}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Granularize parsed Balatro runs into single-action micro-steps."
    )
    ap.add_argument("--src", type=Path, default=Path("data/parsed"))
    ap.add_argument("--dst", type=Path, default=Path("data/granularized"))
    args = ap.parse_args(argv)

    runs = find_runs(args.src)
    if not runs:
        print(f"no run_*.json files found under {args.src}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(runs)} run file(s)\n")

    stats: collections.Counter = collections.Counter()
    current_video: str | None = None
    for video_id, run_idx, run_path in runs:
        if video_id != current_video:
            print(f"video_id={video_id}")
            current_video = video_id

        with open(run_path, encoding="utf-8") as fh:
            run = json.load(fh)

        print(
            f"  run_{run_idx:03d}  ({len(run.get('events') or [])} parsed events)"
        )
        granular = granularize_run(run)

        for s in granular["events"]:
            stats[("source_kind", s.get("source_kind") or "")] += 1
            stats[("base", s.get("action", "").split("_", 1)[0])] += 1

        dst_dir = args.dst / f"video_id={video_id}"
        write_run(granular, dst_dir)

    write_config(args.dst, args.src, stats)
    print()
    print("--- source_kind counts ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 == "source_kind":
            print(f"  {k2:30s} {n}")
    print()
    print("--- base-action counts ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 == "base":
            print(f"  {k2:30s} {n}")
    print("\ndone.")


if __name__ == "__main__":
    main()
