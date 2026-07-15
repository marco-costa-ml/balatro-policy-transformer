#!/usr/bin/env python3
"""
family_map.py
=============
Canonical family map for the branched autoregressive policy. Groups action
families by **decoder shape** (which heads run after the family is picked),
not by legacy flat-label form.

Categories (``decoder_shape``)
------------------------------
- ``no_args`` (7): family-head only.
- ``card_seq`` (2): ``num_cards`` then ordered card-pointer sequence over
  ``CurrentHand`` (``PlayHand`` / ``DiscardHand``).
- ``single_ptr`` (6): one zone pointer (Buy / Sell / BuyAndUse).
- ``chained_cards`` (2): item pointer then ``num_cards`` then optional
  card-pointer sequence (``UseConsumable`` / ``SelectPackItem``).
- ``joker_pair`` (1): ``joker_i`` then ``joker_j`` with ``j != i`` masking
  (``SWAP``). Emission canonicalized to ``i < j`` so labels match the
  existing ``SWAP_i_j`` format.
- ``reserved`` (1): ``StartNewRun`` — never predicted.

The flat ``data/action_map.json`` is **not** replaced; it stays the source
of truth for the legality mask and label-to-flat-index lookups. The family
map is an **orthogonal** view that the policy head + argument decoders
operate on.

Importable helpers
------------------
- ``compute_family_map(action_map)`` -> serializable dict
- ``family_id_for_step(step, family_map)`` -> int (target family id) or -1
- ``FAMILY_ORDER`` (locked tuple of family names)
- ``DECODER_SHAPES`` (mapping family name -> shape)
- ``ITEM_ZONE_FOR_FAMILY`` (mapping family name -> item zone or None)

CLI
---
``python family_map.py [--action-map data/action_map.json]
                       [--out data/family_map.json]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from action_map import load_action_map


FAMILY_MAP_SCHEMA_VERSION = "1.0.0"

# Canonical ordering. Keep stable across versions — bumping or reordering
# this list is a breaking change to model checkpoints and requires a new
# ``family_map_version`` (see ``compute_family_map``).
FAMILY_ORDER: tuple[str, ...] = (
    # reserved (id 0, never predicted)
    "StartNewRun",
    # no_args (7)
    "SelectBlind",
    "SkipBlind",
    "RerollBossBlind",
    "CashOut",
    "LeaveShop",
    "SkipPack",
    "RerollShop",
    # card_seq (2)
    "PlayHand",
    "DiscardHand",
    # single_ptr (6)
    "BuyShopItem_VoucherShopOfferings",
    "BuyShopItem_PackShopOfferings",
    "BuyShopItem_TopShelfShopOfferings",
    "BuyAndUseShopConsumable_TopShelfShopOfferings",
    "SellItem_CurrentJokers",
    "SellItem_CurrentConsumables",
    # chained_cards (2)
    "UseConsumable_CurrentConsumables",
    "SelectPackItem_PackOfferings",
    # joker_pair (1)
    "SWAP",
)


DECODER_SHAPES: dict[str, str] = {
    "StartNewRun": "reserved",
    "SelectBlind": "no_args",
    "SkipBlind": "no_args",
    "RerollBossBlind": "no_args",
    "CashOut": "no_args",
    "LeaveShop": "no_args",
    "SkipPack": "no_args",
    "RerollShop": "no_args",
    "PlayHand": "card_seq",
    "DiscardHand": "card_seq",
    "BuyShopItem_VoucherShopOfferings": "single_ptr",
    "BuyShopItem_PackShopOfferings": "single_ptr",
    "BuyShopItem_TopShelfShopOfferings": "single_ptr",
    "BuyAndUseShopConsumable_TopShelfShopOfferings": "single_ptr",
    "SellItem_CurrentJokers": "single_ptr",
    "SellItem_CurrentConsumables": "single_ptr",
    "UseConsumable_CurrentConsumables": "chained_cards",
    "SelectPackItem_PackOfferings": "chained_cards",
    "SWAP": "joker_pair",
}


# Item zone (where the item/joker pointer reads from) per family. None for
# no-args / card_seq / reserved families.
ITEM_ZONE_FOR_FAMILY: dict[str, str | None] = {
    "StartNewRun": None,
    "SelectBlind": None,
    "SkipBlind": None,
    "RerollBossBlind": None,
    "CashOut": None,
    "LeaveShop": None,
    "SkipPack": None,
    "RerollShop": None,
    "PlayHand": None,
    "DiscardHand": None,
    "BuyShopItem_VoucherShopOfferings": "VoucherShopOfferings",
    "BuyShopItem_PackShopOfferings": "PackShopOfferings",
    "BuyShopItem_TopShelfShopOfferings": "TopShelfShopOfferings",
    "BuyAndUseShopConsumable_TopShelfShopOfferings": "TopShelfShopOfferings",
    "SellItem_CurrentJokers": "CurrentJokers",
    "SellItem_CurrentConsumables": "CurrentConsumables",
    "UseConsumable_CurrentConsumables": "CurrentConsumables",
    "SelectPackItem_PackOfferings": "PackOfferings",
    "SWAP": "CurrentJokers",
}


# Default card pool zone for card_seq + chained_cards families. For
# UseConsumable the actual zone depends on page_name (see
# argument_spec.card_zone_for_use_consumable); the value here is the
# default-In_Blind variant. Live + tensorize code resolve it
# per-step via argument_spec helpers.
DEFAULT_CARD_ZONE_FOR_FAMILY: dict[str, str | None] = {
    "PlayHand": "CurrentHand",
    "DiscardHand": "CurrentHand",
    "UseConsumable_CurrentConsumables": "CurrentHand",
    "SelectPackItem_PackOfferings": "TarotSpectralHand",
}


def _family_name_for_action(label: str, base: str, zone: str | None) -> str:
    """Map an action_map label/family-key to a canonical family name.

    Examples:
    - "PlayHand"                              -> "PlayHand"
    - "BuyShopItem_VoucherShopOfferings_0"    -> "BuyShopItem_VoucherShopOfferings"
    - "SWAP_3_5"                              -> "SWAP"
    - "SelectCard_CurrentHand_2"              -> "" (no longer a family)
    """
    if base == "SWAP":
        return "SWAP"
    if zone is None:
        return base
    return f"{base}_{zone}"


def compute_family_map(action_map: dict[str, Any]) -> dict[str, Any]:
    """Build the family-map artifact from a loaded ``action_map`` dict."""
    family_to_id = {name: i for i, name in enumerate(FAMILY_ORDER)}
    id_to_family = list(FAMILY_ORDER)
    n_families = len(FAMILY_ORDER)

    # Build per-family flat-label slices for back-compat / debugging.
    family_to_flat_labels: dict[str, list[str]] = {f: [] for f in FAMILY_ORDER}
    family_to_flat_offset: dict[str, int] = {}
    family_to_flat_size: dict[str, int] = {}

    family_offsets: dict[str, int] = action_map["family_offsets"]
    family_sizes: dict[str, int] = action_map["family_sizes"]
    index_to_label: list[str] = action_map["index_to_label"]

    # Map each fixed action's offset directly.
    for fam in FAMILY_ORDER:
        if fam in family_offsets:
            family_to_flat_offset[fam] = int(family_offsets[fam])
            family_to_flat_size[fam] = int(family_sizes[fam])
            off = family_to_flat_offset[fam]
            sz = family_to_flat_size[fam]
            family_to_flat_labels[fam] = list(index_to_label[off : off + sz])

    # Compose decoder-shape counts for the artifact summary.
    shape_counts: dict[str, int] = {}
    for fam in FAMILY_ORDER:
        shape = DECODER_SHAPES.get(fam, "unknown")
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    payload = {
        "schema_version": FAMILY_MAP_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_map_version": action_map.get("action_map_version"),
        "n_families": n_families,
        "family_order": list(FAMILY_ORDER),
        "family_to_id": family_to_id,
        "id_to_family": id_to_family,
        "decoder_shapes": {fam: DECODER_SHAPES[fam] for fam in FAMILY_ORDER},
        "item_zone_for_family": {fam: ITEM_ZONE_FOR_FAMILY[fam] for fam in FAMILY_ORDER},
        "default_card_zone_for_family": {
            fam: DEFAULT_CARD_ZONE_FOR_FAMILY.get(fam) for fam in FAMILY_ORDER
        },
        "shape_counts": shape_counts,
        "family_to_flat_offset": family_to_flat_offset,
        "family_to_flat_size": family_to_flat_size,
        "family_to_flat_labels": family_to_flat_labels,
        "reserved_families": ["StartNewRun"],
    }
    payload["family_map_version"] = _family_map_version(payload)
    return payload


def _family_map_version(payload: dict[str, Any]) -> str:
    """Hash the immutable parts of the family-map artifact."""
    keys = (
        "family_order",
        "decoder_shapes",
        "item_zone_for_family",
        "default_card_zone_for_family",
        "schema_version",
    )
    serialized = json.dumps(
        {k: payload[k] for k in keys}, sort_keys=True
    ).encode("utf-8")
    return f"fv1.{hashlib.sha256(serialized).hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Family-id resolution from a granularized step
# ---------------------------------------------------------------------------

# Each granularized commit / pass_through step records a ``source_action``
# (PlayHand / UseConsumable / ...) and optionally ``source_action_subtype``
# / ``target_zone``. SWAP synth steps carry ``source_action == "SWAP"``.
# We use these to recover the family id for the supervised target.

def family_name_for_step(step: dict[str, Any]) -> str | None:
    """Return the canonical family name for a granularized step.

    Returns None when the step has no resolvable family (e.g. SelectCard
    micro-steps, which are folded into their parent commit's super-step,
    and labels that don't appear in ``FAMILY_ORDER``).

    NOTE: ``swap_synth`` micro-steps inherit ``source_action`` from the
    PARENT decision they were attached to (often ``PlayHand`` /
    ``DiscardHand`` / ``LeaveShop``), so we MUST disambiguate by
    ``source_kind`` first. ``swap_synth`` is always the SWAP family.
    """
    # swap_synth detection (source_action carries the parent's action;
    # source_kind is the reliable signal).
    if step.get("source_kind") == "swap_synth":
        return "SWAP"

    source_action = step.get("source_action")
    if source_action == "SWAP":
        return "SWAP"
    if source_action == "StartNewRun":
        return "StartNewRun"
    if source_action in (
        "SelectBlind",
        "SkipBlind",
        "RerollBossBlind",
        "CashOut",
        "LeaveShop",
        "SkipPack",
        "RerollShop",
        "PlayHand",
        "DiscardHand",
    ):
        return source_action
    target_zone = step.get("target_zone")
    if source_action and target_zone:
        candidate = f"{source_action}_{target_zone}"
        if candidate in DECODER_SHAPES:
            return candidate
    return None


def family_id_for_step(step: dict[str, Any], family_map: dict[str, Any]) -> int:
    """Return the integer family id or ``-1`` if unresolvable / reserved.

    ``StartNewRun`` is intentionally returned as ``-1`` so the loss ignores it.
    """
    name = family_name_for_step(step)
    if name is None or name == "StartNewRun":
        return -1
    return int(family_map["family_to_id"].get(name, -1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build canonical family_map.json from action_map.json.",
    )
    ap.add_argument(
        "--action-map",
        type=Path,
        default=Path("data/action_map.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/family_map.json"),
    )
    args = ap.parse_args()

    action_map = load_action_map(args.action_map)
    payload = compute_family_map(action_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {args.out.as_posix()}")
    print(f"  schema_version: {payload['schema_version']}")
    print(f"  family_map_version: {payload['family_map_version']}")
    print(f"  n_families: {payload['n_families']}")
    print(f"  shape_counts: {payload['shape_counts']}")
    print("  families:")
    for fam in payload["family_order"]:
        shape = payload["decoder_shapes"][fam]
        item = payload["item_zone_for_family"][fam] or "-"
        card = payload["default_card_zone_for_family"][fam] or "-"
        sz = payload["family_to_flat_size"].get(fam, 0)
        print(f"    {fam:55s} shape={shape:14s} item={item:22s} card={card:18s} flat_size={sz}")


if __name__ == "__main__":
    main()
