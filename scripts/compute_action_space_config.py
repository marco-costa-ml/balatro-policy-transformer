#!/usr/bin/env python3
"""
scripts/compute_action_space_config.py
======================================
Scan ``data/parsed/`` and produce a per-zone action-space cap dict for the
new ``action_map.py`` per-zone subfamily layout.

For each ``(base_action, zone)`` subfamily we count the maximum number of
candidate objects observed in the relevant zone across the corpus. The
output is written to ``data/action_space_config.json`` under
``max_values_per_zone``, alongside the existing ``MAX_JOKER_SLOTS``
aggregate.

Counting rule per zone: union of objects whose raw zone is in
``{Foo, FooAll, FooSelected}`` deduplicated by ``slot_id``. This mirrors
the ``collapse_zones`` helper in ``granularize.py``.

Usage:
    python scripts/compute_action_space_config.py
        [--src data/parsed] [--out data/action_space_config.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Per-zone subfamilies the new action map cares about.
# (base_action, zone_base): the matching ``All`` zone in parsed data is
# ``{zone_base}All`` (or bare ``{zone_base}``). For some subfamilies we
# additionally restrict the count to a subset of objects (e.g. consumables
# only for BuyAndUseShopConsumable in the TopShelf bucket).
# ---------------------------------------------------------------------------
SUBFAMILIES: list[tuple[str, str]] = [
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

CONSUMABLE_OBJECT_TYPES: frozenset[str] = frozenset(
    {"consumable", "tarot", "planet", "spectral"}
)


def _subfamily_key(base: str, zone: str) -> str:
    return f"{base}_{zone}"


def _zone_count_for(zone_base: str, parsed_objects: list[dict[str, Any]]) -> int:
    """Count objects in zone ``zone_base`` (or ``{zone_base}All`` /
    ``{zone_base}Selected``), deduped by ``slot_id``.
    """
    target_zones = {zone_base, zone_base + "All", zone_base + "Selected"}
    seen: set[Any] = set()
    n = 0
    for obj in parsed_objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("zone") not in target_zones:
            continue
        slot_id = obj.get("slot_id")
        if isinstance(slot_id, int):
            if slot_id in seen:
                continue
            seen.add(slot_id)
        n += 1
    return n


def _consumable_count_in_zone(
    zone_base: str, parsed_objects: list[dict[str, Any]]
) -> int:
    """Count consumable-typed objects in ``zone_base`` (variants), deduped by slot_id."""
    target_zones = {zone_base, zone_base + "All", zone_base + "Selected"}
    seen: set[Any] = set()
    n = 0
    for obj in parsed_objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("zone") not in target_zones:
            continue
        if obj.get("object_type") not in CONSUMABLE_OBJECT_TYPES:
            continue
        slot_id = obj.get("slot_id")
        if isinstance(slot_id, int):
            if slot_id in seen:
                continue
            seen.add(slot_id)
        n += 1
    return n


def _joker_count(parsed_objects: list[dict[str, Any]]) -> int:
    return _zone_count_for("CurrentJokers", parsed_objects)


def _iter_run_files(src: Path):
    for partition in sorted(src.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        for run_file in sorted(partition.glob("run_*.json")):
            yield run_file


def scan(src: Path) -> tuple[dict[str, int], dict[str, Any], int, int]:
    """Returns (max_values_per_zone, provenance, events_scanned, files_scanned)."""
    max_values: dict[str, int] = {key: 0 for key in (
        _subfamily_key(b, z) for b, z in SUBFAMILIES
    )}
    provenance: dict[str, dict[str, Any]] = {}
    joker_max = 0
    joker_prov: dict[str, Any] | None = None

    files_scanned = 0
    events_scanned = 0

    for run_path in _iter_run_files(src):
        files_scanned += 1
        run = json.loads(run_path.read_text(encoding="utf-8"))
        events = run.get("events") or []
        for event_idx, event in enumerate(events):
            events_scanned += 1
            parsed_objects = event.get("objects") or []
            for base, zone in SUBFAMILIES:
                key = _subfamily_key(base, zone)
                if base == "BuyAndUseShopConsumable":
                    count = _consumable_count_in_zone(zone, parsed_objects)
                else:
                    count = _zone_count_for(zone, parsed_objects)
                if count > max_values[key]:
                    max_values[key] = count
                    provenance[key] = {
                        "file": run_path.as_posix(),
                        "event_index": event_idx,
                        "frame_idx": event.get("frame_idx"),
                        "page_name": event.get("page_name"),
                        "observed_value": count,
                    }
            jc = _joker_count(parsed_objects)
            if jc > joker_max:
                joker_max = jc
                joker_prov = {
                    "file": run_path.as_posix(),
                    "event_index": event_idx,
                    "frame_idx": event.get("frame_idx"),
                    "page_name": event.get("page_name"),
                    "observed_value": jc,
                }

    if joker_prov is not None:
        provenance["MAX_JOKER_SLOTS"] = joker_prov
    return max_values, provenance, events_scanned, files_scanned


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/parsed"))
    ap.add_argument(
        "--out", type=Path, default=Path("data/action_space_config.json")
    )
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"missing parsed directory: {args.src}")

    print(f"scanning {args.src.as_posix()}...")
    max_values_per_zone, provenance, n_events, n_files = scan(args.src)
    joker_max = max(
        max_values_per_zone.get("SellItem_CurrentJokers", 0),
        provenance.get("MAX_JOKER_SLOTS", {}).get("observed_value", 0),
    )

    payload = {
        "schema_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": args.src.as_posix(),
        "files_scanned": n_files,
        "events_scanned": n_events,
        "max_values": {
            "MAX_JOKER_SLOTS": int(joker_max),
        },
        "max_values_per_zone": {
            key: int(v) for key, v in sorted(max_values_per_zone.items())
        },
        "observed_maxima_per_zone": {
            key: int(v) for key, v in sorted(max_values_per_zone.items())
        },
        "provenance_per_zone": provenance,
        "policy": {
            "corpus": "parsed",
            "headroom": "none",
            "counting_rule": (
                "max(len(union of {Foo, FooAll, FooSelected})), deduped by slot_id"
            ),
            "BuyAndUseShopConsumable_rule": (
                "TopShelfShopOfferings restricted to object_type in "
                "{consumable, tarot, planet, spectral}."
            ),
        },
        "subfamilies": [
            {"base": base, "zone": zone, "key": _subfamily_key(base, zone)}
            for base, zone in SUBFAMILIES
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out.as_posix()}")
    print(f"  files_scanned={n_files} events_scanned={n_events}")
    print(f"  MAX_JOKER_SLOTS={joker_max}")
    for base, zone in SUBFAMILIES:
        key = _subfamily_key(base, zone)
        print(f"  {key:55s} = {max_values_per_zone[key]}")


if __name__ == "__main__":
    main()
