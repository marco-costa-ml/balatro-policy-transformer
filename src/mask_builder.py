#!/usr/bin/env python3
"""
mask_builder.py
===============
Minimal v2 action-mask builder for granularized Balatro steps (per-zone
subfamily layout).

Implements the two core rules from ``mask_schema.md``:

1. **Page gating** — only families legal on the current ``page_name``
   (and a small set of globally-legal families) may be unmasked.
2. **Target exists** — for each per-zone indexed subfamily, only indices
   ``i < min(count_of(zone, step.objects), subfamily_size)`` may be
   unmasked. For ``SWAP_i_j``, only pairs with
   ``0 <= i < j < jokers_current``.

Things deliberately *not* in v2 (deferred to a richer pass once persistent
state is wired up):

- per-consumable cardinality / attribute constraints
- per-shop-item affordability / inventory caps
- ``RerollShop`` price gate, ``SkipBlind`` requires offered tag
- SWAP swap-count cap and last-swap suppression
- ``SellItem`` eternal-sticker exclusion

Branched-policy additions
-------------------------
The autoregressive policy operates over a 19-family ``family_map`` (see
``family_map.py``) plus per-zone pointer masks for each indexed / chained
family. The new helpers in this module:

- ``build_family_mask(step, action_map, family_map)`` — parent-start
  legality per family. Implementation note: v1's flat mask is page-gated
  only, so ``PlayHand`` / ``DiscardHand`` are flat-mask-legal at any
  In_Blind step regardless of ``pending_cards``. That matches our
  parent-start semantics. ``family_mask[f] = OR over the family's flat
  slice`` works uniformly for every family except the reserved
  ``StartNewRun`` (always False).
- ``build_item_pointer_mask(family, step, action_map, max_size)`` — per-
  position validity within the family's **item** zone.
- ``build_card_pointer_mask(family, step, action_map, max_size)`` — per-
  position validity within the family's **card-pool** zone (resolved by
  ``page_name`` for ``UseConsumable`` and by parent subtype for
  ``SelectPackItem``).
- ``build_swap_joker_mask(step, max_joker_slots)`` — per-position joker
  validity for the SWAP joker-pair head.

Pointer masks reuse the v1 subfamily slices wherever possible so legality
semantics stay in lockstep with ``build_action_mask``.

Importable helpers
------------------
- ``build_action_mask(step, action_map) -> np.ndarray[bool, N_ACTIONS]``
- ``build_family_mask(step, action_map, family_map) -> np.ndarray[bool, n_families]``
- ``build_item_pointer_mask(family, step, action_map, max_size) -> np.ndarray[bool, max_size]``
- ``build_card_pointer_mask(family, step, action_map, max_size) -> np.ndarray[bool, max_size]``
- ``build_swap_joker_mask(step, max_joker_slots) -> np.ndarray[bool, max_joker_slots]``
- ``candidate_count_for_subfamily(base, zone, step) -> int``
- ``allowed_families_for_page(page_name) -> set[str]``
- ``PAGE_GATE``, ``GLOBAL_FAMILIES`` constants

CLI
---
``python mask_builder.py --src data/granularized --action-map data/action_map.json``
prints aggregate per-page / per-family unmask counts and a sample of the
mask vector for a few steps.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np


# Families that are legal on every page (within their own per-step constraints).
# ``StartNewRun`` is a bootstrap event and is NEVER unmasked at training
# time (the tensorizer sets its target_action_id to -1); we keep it out
# of the global allow-set so the model is not encouraged to predict it.
GLOBAL_FAMILIES: frozenset[str] = frozenset(
    {"UseConsumable", "SellItem", "SWAP"}
)

# Per-page additions. Union with GLOBAL_FAMILIES gives the full allow-set.
PAGE_GATE: dict[str, frozenset[str]] = {
    "Blind_Select": frozenset({"SelectBlind", "SkipBlind", "RerollBossBlind"}),
    "Cash_Out": frozenset({"CashOut"}),
    "Dummy_Page": frozenset(),
    "In_Blind": frozenset({"PlayHand", "DiscardHand", "SelectCard"}),
    "In_JokerStandardPlanet_Pack": frozenset({"SelectPackItem", "SkipPack"}),
    "In_Shop": frozenset(
        {"BuyShopItem", "BuyAndUseShopConsumable", "LeaveShop", "RerollShop"}
    ),
    "In_TarotSpectral_Pack": frozenset({"SelectCard", "SelectPackItem", "SkipPack"}),
}


def allowed_families_for_page(page_name: str | None) -> set[str]:
    """Return the set of family base names that may be unmasked on this page."""
    extras = PAGE_GATE.get(page_name or "", frozenset())
    return set(GLOBAL_FAMILIES) | set(extras)


def _objects_by_zone(step: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group ``step.objects`` by canonical zone name."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in step.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        zone = obj.get("zone")
        if isinstance(zone, str):
            grouped.setdefault(zone, []).append(obj)
    return grouped


def _jokers_current(step: dict[str, Any]) -> int:
    """Best-effort joker count for SWAP target-exists gating."""
    state = step.get("state") or {}
    val = state.get("jokers_current")
    if isinstance(val, int) and val >= 0:
        return val
    grouped = _objects_by_zone(step)
    return len(grouped.get("CurrentJokers", []))


_CONSUMABLE_OBJECT_TYPES: frozenset[str] = frozenset(
    {"consumable", "tarot", "planet", "spectral"}
)


def _object_position(obj: dict[str, Any], fallback_idx: int) -> int:
    """Return the canonical position of ``obj`` within its zone.

    Granularize emits ``position_in_zone`` per object after normalization.
    Falls back to the object's index in ``step.objects`` for the zone
    when that field is missing (older snapshots).
    """
    pos = obj.get("position_in_zone")
    if isinstance(pos, int) and pos >= 0:
        return pos
    return fallback_idx


def candidate_positions_for_subfamily(
    base: str, zone: str, step: dict[str, Any]
) -> list[int]:
    """
    Return the list of ``position_in_zone`` values that are legal targets
    for the ``(base, zone)`` subfamily at this step.

    For most subfamilies this is just every object's position in the
    zone. Two subfamilies *share* ``TopShelfShopOfferings``:

    - ``BuyAndUseShopConsumable_TopShelfShopOfferings`` keeps only
      consumable-typed objects (``object_type in {consumable, tarot,
      planet, spectral}``).
    - ``BuyShopItem_TopShelfShopOfferings`` keeps only joker-typed
      objects (the consumable subset is handled by
      ``BuyAndUseShopConsumable``).

    Because the index suffix in the action label encodes the absolute
    zone position (matching ``target_position`` written by
    ``granularize.py``), we must surface the exact positions rather than
    a count: e.g. a shop with ``[joker, consumable, joker]`` yields
    BuyShopItem positions ``[0, 2]``, not ``[0, 1]``.
    """
    grouped = _objects_by_zone(step)
    objs = grouped.get(zone, [])
    if not objs:
        return []

    # ``BuyAndUseShopConsumable`` only operates on consumable-typed items;
    # the joker variant of "buy and use" is not a valid Balatro action.
    if base == "BuyAndUseShopConsumable" and zone == "TopShelfShopOfferings":
        return [
            _object_position(o, i)
            for i, o in enumerate(objs)
            if o.get("object_type") in _CONSUMABLE_OBJECT_TYPES
        ]
    # ``BuyShopItem`` on ``TopShelfShopOfferings`` accepts both jokers
    # (adds to joker slots) and consumables (adds to consumable slots).
    return [_object_position(o, i) for i, o in enumerate(objs)]


def candidate_count_for_subfamily(
    base: str, zone: str, step: dict[str, Any]
) -> int:
    """Convenience wrapper for diagnostics — number of legal positions."""
    return len(candidate_positions_for_subfamily(base, zone, step))


def build_action_mask(
    step: dict[str, Any], action_map: dict[str, Any]
) -> np.ndarray:
    """
    Build a length-``N_ACTIONS`` boolean mask for this step.

    True  = action is currently legal
    False = action is masked out
    """
    n_actions: int = int(action_map["n_actions"])
    family_offsets: dict[str, int] = action_map["family_offsets"]
    family_sizes: dict[str, int] = action_map["family_sizes"]
    indexed_families = action_map.get("indexed_families") or []

    mask = np.zeros(n_actions, dtype=bool)
    page_name = step.get("page_name")
    allowed = allowed_families_for_page(page_name)

    # 1) Fixed non-index actions: one slot each, gate by page.
    for fam in (
        "SelectBlind",
        "SkipBlind",
        "RerollBossBlind",
        "DiscardHand",
        "PlayHand",
        "CashOut",
        "LeaveShop",
        "SkipPack",
        "RerollShop",
    ):
        if fam in allowed and fam in family_offsets:
            mask[family_offsets[fam]] = True

    # 2) Per-zone indexed subfamilies.
    for entry in indexed_families:
        base = entry["base"]
        zone = entry["zone"]
        key = entry.get("subfamily_key") or f"{base}_{zone}"
        if base not in allowed:
            continue
        size = int(family_sizes.get(key, 0))
        if size <= 0:
            continue
        offset = int(family_offsets.get(key, -1))
        if offset < 0:
            continue
        for pos in candidate_positions_for_subfamily(base, zone, step):
            if 0 <= pos < size:
                mask[offset + pos] = True

    # 3) SWAP_i_j: page gate + (i, j < jokers_current).
    if "SWAP" in allowed and "SWAP" in family_offsets:
        jokers_current = _jokers_current(step)
        if jokers_current >= 2:
            swap_offset = family_offsets["SWAP"]
            for k, pair in enumerate(action_map["swap_family"]["pairs"]):
                i, j = int(pair[0]), int(pair[1])
                if i < jokers_current and j < jokers_current:
                    mask[swap_offset + k] = True

    return mask


# ---------------------------------------------------------------------------
# Branched-policy mask helpers (family + per-zone pointer masks)
# ---------------------------------------------------------------------------

# Per-family v1 subfamily key used to source the item-pointer mask. ``None``
# means the family does not have an item pointer (no_args / card_seq) OR is
# the SWAP joker pair (handled by ``build_swap_joker_mask``).
_FAMILY_TO_ITEM_SUBFAMILY: dict[str, str | None] = {
    # no_args
    "SelectBlind": None,
    "SkipBlind": None,
    "RerollBossBlind": None,
    "CashOut": None,
    "LeaveShop": None,
    "SkipPack": None,
    "RerollShop": None,
    # card_seq
    "PlayHand": None,
    "DiscardHand": None,
    # single_ptr
    "BuyShopItem_VoucherShopOfferings": "BuyShopItem_VoucherShopOfferings",
    "BuyShopItem_PackShopOfferings": "BuyShopItem_PackShopOfferings",
    "BuyShopItem_TopShelfShopOfferings": "BuyShopItem_TopShelfShopOfferings",
    "BuyAndUseShopConsumable_TopShelfShopOfferings": (
        "BuyAndUseShopConsumable_TopShelfShopOfferings"
    ),
    "SellItem_CurrentJokers": "SellItem_CurrentJokers",
    "SellItem_CurrentConsumables": "SellItem_CurrentConsumables",
    # chained_cards
    "UseConsumable_CurrentConsumables": "UseConsumable_CurrentConsumables",
    "SelectPackItem_PackOfferings": "SelectPackItem_PackOfferings",
    # joker_pair (handled by build_swap_joker_mask)
    "SWAP": None,
    # reserved
    "StartNewRun": None,
}


# Per-family card-pool zone resolution. The value can be:
#   - a literal canonical zone name (e.g. ``"CurrentHand"``),
#   - ``"BY_PAGE"`` to dispatch on ``step.page_name`` via
#     ``argument_spec.card_zone_for_use_consumable``,
#   - ``"BY_PACK_SUBTYPE"`` to dispatch on ``step.source_action_subtype``
#     via ``argument_spec.card_zone_for_select_pack_item``,
#   - ``None`` if the family has no card pool.
_FAMILY_CARD_ZONE_RULE: dict[str, str | None] = {
    "PlayHand": "CurrentHand",
    "DiscardHand": "CurrentHand",
    "UseConsumable_CurrentConsumables": "BY_PAGE",
    "SelectPackItem_PackOfferings": "BY_PACK_SUBTYPE",
}


# Map canonical card zone -> v1 subfamily key (whose flat slice gives the
# per-position validity bits we want).
_CARD_ZONE_TO_SUBFAMILY: dict[str, str] = {
    "CurrentHand": "SelectCard_CurrentHand",
    "TarotSpectralHand": "SelectCard_TarotSpectralHand",
}


def _slice_flat(
    flat_mask: np.ndarray, offset: int, size: int
) -> np.ndarray:
    """Return a copy of ``flat_mask[offset:offset+size]`` as a contiguous bool array."""
    out = np.zeros(size, dtype=bool)
    if size <= 0 or offset < 0:
        return out
    end = min(offset + size, flat_mask.shape[0])
    if end > offset:
        out[: end - offset] = flat_mask[offset:end]
    return out


def build_family_mask(
    step: dict[str, Any],
    action_map: dict[str, Any],
    family_map: dict[str, Any],
    *,
    flat_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a length-``n_families`` boolean mask of parent-start legality.

    For every family except the reserved ``StartNewRun``, legality reduces
    to ``OR`` over the family's slice in the v1 flat action mask. Because
    v1 gates by page only (and not by ``pending_cards.length`` for the
    ``PlayHand``/``DiscardHand`` slots), this also captures parent-start
    legality for ``card_seq`` families.

    Pass ``flat_mask`` to reuse a precomputed v1 mask; otherwise it's
    built from the step on demand.
    """
    if flat_mask is None:
        flat_mask = build_action_mask(step, action_map)

    n_families = int(family_map["n_families"])
    out = np.zeros(n_families, dtype=bool)
    family_to_id = family_map["family_to_id"]
    family_to_flat_offset = family_map["family_to_flat_offset"]
    family_to_flat_size = family_map["family_to_flat_size"]

    for fam, fid in family_to_id.items():
        if fam == "StartNewRun":
            continue
        off = int(family_to_flat_offset.get(fam, -1))
        sz = int(family_to_flat_size.get(fam, 0))
        if off < 0 or sz <= 0:
            continue
        slc = _slice_flat(flat_mask, off, sz)
        if bool(slc.any()):
            out[int(fid)] = True
    return out


def build_item_pointer_mask(
    family: str,
    step: dict[str, Any],
    action_map: dict[str, Any],
    max_size: int,
    *,
    flat_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a length-``max_size`` mask of valid item-pointer positions.

    The mask is the v1 subfamily slice for the family's item zone padded
    or truncated to ``max_size``. Returns an all-False array for families
    without an item pointer (no_args / card_seq / SWAP / reserved).
    """
    out = np.zeros(int(max_size), dtype=bool)
    subfamily_key = _FAMILY_TO_ITEM_SUBFAMILY.get(family)
    if subfamily_key is None:
        return out

    family_offsets: dict[str, int] = action_map["family_offsets"]
    family_sizes: dict[str, int] = action_map["family_sizes"]
    off = int(family_offsets.get(subfamily_key, -1))
    sz = int(family_sizes.get(subfamily_key, 0))
    if off < 0 or sz <= 0:
        return out

    if flat_mask is None:
        flat_mask = build_action_mask(step, action_map)
    slc = _slice_flat(flat_mask, off, sz)
    take = min(sz, int(max_size))
    out[:take] = slc[:take]
    return out


def resolve_card_zone(family: str, step: dict[str, Any]) -> str | None:
    """Resolve the active card-pool zone for ``family`` at ``step``.

    Returns ``None`` for families without a card pool, or when the
    page / subtype implies no card targeting (e.g. SelectPackItem with
    a non-tarot subtype).
    """
    rule = _FAMILY_CARD_ZONE_RULE.get(family)
    if rule is None:
        return None
    if rule == "BY_PAGE":
        from argument_spec import card_zone_for_use_consumable

        return card_zone_for_use_consumable(step.get("page_name"))
    if rule == "BY_PACK_SUBTYPE":
        from argument_spec import card_zone_for_select_pack_item

        # The granularized step records the subtype on the commit step;
        # use both source_action_subtype and action_subtype to be safe.
        subtype = step.get("source_action_subtype") or step.get("action_subtype")
        return card_zone_for_select_pack_item(subtype)
    return rule


def build_card_pointer_mask(
    family: str,
    step: dict[str, Any],
    action_map: dict[str, Any],
    max_size: int,
    *,
    flat_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a length-``max_size`` mask of valid card-pool pointer positions.

    Resolves the card zone via ``resolve_card_zone`` and reads the
    corresponding v1 ``SelectCard_<zone>`` subfamily slice. Returns
    all-False if the family has no card pool at this step.
    """
    out = np.zeros(int(max_size), dtype=bool)
    zone = resolve_card_zone(family, step)
    if zone is None:
        return out
    subfamily_key = _CARD_ZONE_TO_SUBFAMILY.get(zone)
    if subfamily_key is None:
        return out
    family_offsets: dict[str, int] = action_map["family_offsets"]
    family_sizes: dict[str, int] = action_map["family_sizes"]
    off = int(family_offsets.get(subfamily_key, -1))
    sz = int(family_sizes.get(subfamily_key, 0))
    if off < 0 or sz <= 0:
        return out
    if flat_mask is None:
        flat_mask = build_action_mask(step, action_map)
    slc = _slice_flat(flat_mask, off, sz)
    take = min(sz, int(max_size))
    out[:take] = slc[:take]
    return out


def build_swap_joker_mask(
    step: dict[str, Any],
    max_joker_slots: int,
) -> np.ndarray:
    """Return a length-``max_joker_slots`` mask of valid joker positions.

    A position ``i`` is valid iff ``i < jokers_current``. The joker-j
    "j != i" exclusion is applied dynamically inside the decoder.
    """
    out = np.zeros(int(max_joker_slots), dtype=bool)
    jokers_current = _jokers_current(step)
    take = min(int(jokers_current), int(max_joker_slots))
    if take > 0:
        out[:take] = True
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build action masks for granularized steps and print summary stats."
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--action-map", type=Path, default=Path("data/action_map.json"))
    ap.add_argument(
        "--sample-files",
        type=int,
        default=10,
        help="Limit to first N run files to keep CLI runs cheap (0 = all).",
    )
    args = ap.parse_args(argv)

    action_map = json.loads(args.action_map.read_text(encoding="utf-8"))
    n_actions = int(action_map["n_actions"])
    family_offsets: dict[str, int] = action_map["family_offsets"]
    family_sizes: dict[str, int] = action_map["family_sizes"]

    print(f"N_ACTIONS = {n_actions}")
    print("family layout:")
    for fam, off in sorted(family_offsets.items(), key=lambda kv: kv[1]):
        sz = family_sizes[fam]
        print(f"  {fam:55s} offset={off:4d} size={sz}")
    print()

    per_page_total: collections.Counter = collections.Counter()
    per_page_unmasked_sum: collections.Counter = collections.Counter()
    per_page_unmasked_min: dict[str, int] = {}
    per_page_unmasked_max: dict[str, int] = {}

    files_seen = 0
    steps_seen = 0
    for partition in sorted(args.src.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        for run_file in sorted(partition.glob("run_*.json")):
            run = json.loads(run_file.read_text(encoding="utf-8"))
            for step in run.get("events", []):
                mask = build_action_mask(step, action_map)
                page = step.get("page_name") or "<none>"
                cnt = int(mask.sum())
                steps_seen += 1
                per_page_total[page] += 1
                per_page_unmasked_sum[page] += cnt
                if page not in per_page_unmasked_min or cnt < per_page_unmasked_min[page]:
                    per_page_unmasked_min[page] = cnt
                if page not in per_page_unmasked_max or cnt > per_page_unmasked_max[page]:
                    per_page_unmasked_max[page] = cnt
            files_seen += 1
            if args.sample_files and files_seen >= args.sample_files:
                break
        if args.sample_files and files_seen >= args.sample_files:
            break

    print(f"scanned {steps_seen} steps from {files_seen} run files")
    print()
    print("--- unmasked-action counts per page (avg / min / max) ---")
    for page, total in sorted(per_page_total.items(), key=lambda kv: (-kv[1], kv[0])):
        avg = per_page_unmasked_sum[page] / max(total, 1)
        print(
            f"  {page:32s}  steps={total:6d}  "
            f"avg={avg:7.2f}  min={per_page_unmasked_min[page]:4d}  "
            f"max={per_page_unmasked_max[page]:4d}"
        )


if __name__ == "__main__":
    main()
