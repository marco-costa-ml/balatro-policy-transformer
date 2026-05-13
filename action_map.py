#!/usr/bin/env python3
"""
action_map.py
=============
Canonical flat-action-map builder + loader (schema 2.0.0).

Reads ``data/action_space_config.json`` (locked ``MAX_JOKER_SLOTS`` and
``max_values_per_zone``) and produces ``data/action_map.json`` with an
explicit ``index -> label`` mapping that respects
``action_space_schema.md`` ordering:

  1. Fixed (no target, no suffix), in order:
     StartNewRun, SelectBlind, SkipBlind, RerollBossBlind,
     DiscardHand, PlayHand, CashOut, LeaveShop, SkipPack, RerollShop.
  2. Per-zone indexed subfamilies in order:
     UseConsumable_CurrentConsumables_i, SelectCard_CurrentHand_i,
     SelectCard_TarotSpectralHand_i, SelectPackItem_PackOfferings_i,
     BuyAndUseShopConsumable_TopShelfShopOfferings_i,
     BuyShopItem_VoucherShopOfferings_i, BuyShopItem_PackShopOfferings_i,
     BuyShopItem_TopShelfShopOfferings_i, SellItem_CurrentJokers_i,
     SellItem_CurrentConsumables_i.
  3. Pair-index swap family:
     ``SWAP_i_j`` for all ``0 <= i < j < MAX_JOKER_SLOTS``
     (lexicographic by ``(i, j)``).

Importable helpers:
- ``compute_action_map(config)`` -> action_map dict
- ``load_action_map(path)``       -> action_map dict
- ``n_actions(config)``           -> int

CLI:
- ``python action_map.py [--src data/action_space_config.json]
   [--out data/action_map.json]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTION_MAP_SCHEMA_VERSION = "2.0.0"

FIXED_ACTIONS: list[str] = [
    "StartNewRun",
    "SelectBlind",
    "SkipBlind",
    "RerollBossBlind",
    "DiscardHand",
    "PlayHand",
    "CashOut",
    "LeaveShop",
    "SkipPack",
    "RerollShop",
]

# (base_action, zone_base) per-zone subfamilies. Each emits labels
# ``f"{base}_{zone}_{i}"`` for ``i in range(N_zone)`` where ``N_zone`` is
# read from ``config.max_values_per_zone[f"{base}_{zone}"]``.
INDEXED_FAMILIES: list[tuple[str, str]] = [
    ("UseConsumable", "CurrentConsumables"),
    ("SelectCard", "CurrentHand"),
    ("SelectCard", "TarotSpectralHand"),
    ("SelectPackItem", "PackOfferings"),
    ("BuyAndUseShopConsumable", "TopShelfShopOfferings"),
    ("BuyShopItem", "VoucherShopOfferings"),
    ("BuyShopItem", "PackShopOfferings"),
    ("BuyShopItem", "TopShelfShopOfferings"),
    ("SellItem", "CurrentJokers"),
    ("SellItem", "CurrentConsumables"),
]

SWAP_FAMILY = "SWAP"
SWAP_MAX_KEY = "MAX_JOKER_SLOTS"


def subfamily_key(base: str, zone: str) -> str:
    return f"{base}_{zone}"


def _swap_pairs(max_joker_slots: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(max_joker_slots)
        for j in range(i + 1, max_joker_slots)
    ]


def _action_map_version(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"v2.{hashlib.sha256(serialized).hexdigest()[:12]}"


def compute_action_map(config: dict[str, Any]) -> dict:
    """
    Build the canonical action map dict from an action-space config.

    ``config`` must include:
      - ``max_values["MAX_JOKER_SLOTS"]``
      - ``max_values_per_zone[f"{base}_{zone}"]`` for every entry in
        ``INDEXED_FAMILIES``.
    """
    max_values = config.get("max_values") or {}
    if SWAP_MAX_KEY not in max_values:
        raise ValueError(
            f"config.max_values is missing '{SWAP_MAX_KEY}'"
        )

    per_zone = config.get("max_values_per_zone") or {}
    required = {subfamily_key(b, z) for b, z in INDEXED_FAMILIES}
    missing = required - set(per_zone)
    if missing:
        raise ValueError(
            f"config.max_values_per_zone missing keys: {sorted(missing)}"
        )

    index_to_label: list[str] = []
    family_offsets: dict[str, int] = {}
    family_sizes: dict[str, int] = {}
    base_family_offsets: dict[str, tuple[int, int]] = {}

    # 1) Fixed actions.
    for name in FIXED_ACTIONS:
        family_offsets[name] = len(index_to_label)
        family_sizes[name] = 1
        index_to_label.append(name)

    # 2) Per-zone indexed subfamilies.
    base_spans: dict[str, list[int]] = {}
    for base, zone in INDEXED_FAMILIES:
        key = subfamily_key(base, zone)
        size = int(per_zone[key])
        if size < 0:
            raise ValueError(f"{key} must be non-negative; got {size}")
        family_offsets[key] = len(index_to_label)
        family_sizes[key] = size
        base_spans.setdefault(base, []).append(len(index_to_label))
        for i in range(size):
            index_to_label.append(f"{base}_{zone}_{i}")
        base_spans[base].append(len(index_to_label))

    for base, spans in base_spans.items():
        if spans:
            start = spans[0]
            end = spans[-1]
            base_family_offsets[base] = (start, end)

    # 3) Pair-index swap family.
    pairs = _swap_pairs(int(max_values[SWAP_MAX_KEY]))
    family_offsets[SWAP_FAMILY] = len(index_to_label)
    family_sizes[SWAP_FAMILY] = len(pairs)
    for i, j in pairs:
        index_to_label.append(f"{SWAP_FAMILY}_{i}_{j}")

    label_to_index = {label: idx for idx, label in enumerate(index_to_label)}
    if len(label_to_index) != len(index_to_label):
        raise RuntimeError("action map produced duplicate labels")

    return {
        "schema_version": ACTION_MAP_SCHEMA_VERSION,
        "action_map_version": _action_map_version(
            {
                "max_values": {k: int(v) for k, v in max_values.items()},
                "max_values_per_zone": {
                    k: int(v) for k, v in per_zone.items()
                },
            }
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_values": dict(max_values),
        "max_values_per_zone": dict(per_zone),
        "fixed_actions": list(FIXED_ACTIONS),
        "indexed_families": [
            {
                "base": base,
                "zone": zone,
                "subfamily_key": subfamily_key(base, zone),
                "offset": family_offsets[subfamily_key(base, zone)],
                "size": family_sizes[subfamily_key(base, zone)],
            }
            for base, zone in INDEXED_FAMILIES
        ],
        "swap_family": {
            "base": SWAP_FAMILY,
            "max_key": SWAP_MAX_KEY,
            "size": family_sizes[SWAP_FAMILY],
            "pairs": [list(p) for p in pairs],
        },
        "n_actions": len(index_to_label),
        "family_offsets": family_offsets,
        "family_sizes": family_sizes,
        "base_family_offsets": {
            base: [start, end] for base, (start, end) in base_family_offsets.items()
        },
        "index_to_label": index_to_label,
        "label_to_index": label_to_index,
    }


def load_action_map(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def n_actions(config: dict[str, Any]) -> int:
    return compute_action_map(config)["n_actions"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build canonical action_map.json from action_space_config.json."
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/action_space_config.json"),
        help="Path to locked action-space config.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/action_map.json"),
        help="Output action map JSON path.",
    )
    args = ap.parse_args()

    config = json.loads(args.src.read_text(encoding="utf-8"))
    action_map = compute_action_map(config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(action_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out.as_posix()}")
    print(f"  action_map_version: {action_map['action_map_version']}")
    print(f"  n_actions: {action_map['n_actions']}")
    for entry in action_map["indexed_families"]:
        key = entry["subfamily_key"]
        print(f"  {key:55s} size={entry['size']}  offset={entry['offset']}")
    print(f"  SWAP: size={action_map['swap_family']['size']}")


if __name__ == "__main__":
    main()
