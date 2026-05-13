from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_KEYS = [
    "MAX_CONSUMABLE_TARGETS",
    "MAX_SELECT_CARD_TARGETS",
    "MAX_PACK_ITEM_TARGETS",
    "MAX_SHOP_CONSUMABLE_TARGETS",
    "MAX_BUYSHOPITEM_TARGETS",
    "MAX_SELLITEM_TARGETS",
    "MAX_JOKER_SLOTS",
]


def zone_list(zones: dict[str, Any], name: str) -> list[dict[str, Any]]:
    values = zones.get(name)
    if isinstance(values, list):
        return [x for x in values if isinstance(x, dict)]
    return []


def is_consumable(obj: dict[str, Any]) -> bool:
    cid = obj.get("class_id")
    return isinstance(cid, int) and ((236 <= cid <= 265) or (298 <= cid <= 319))


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_extracted_events(src_root: Path) -> tuple[int, int, list[tuple[Path, int, dict[str, Any]]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    file_count = 0
    event_count = 0

    for extracted_file in sorted(src_root.glob("video_id=*/*.json")):
        file_count += 1
        try:
            payload = json.loads(extracted_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse JSON: {extracted_file} ({exc})") from exc

        events: list[dict[str, Any]]
        if isinstance(payload, dict):
            raw_events = payload.get("events", [])
            events = raw_events if isinstance(raw_events, list) else []
        elif isinstance(payload, list):
            events = payload
        else:
            events = []

        for idx, event in enumerate(events):
            if isinstance(event, dict):
                event_count += 1
                records.append((extracted_file, idx, event))

    return file_count, event_count, records


def compute_counts(event: dict[str, Any]) -> dict[str, int]:
    state = event.get("state", {})
    zones = state.get("zones", {}) if isinstance(state, dict) else {}
    zones = zones if isinstance(zones, dict) else {}

    current_consumables_selected = zone_list(zones, "CurrentConsumablesSelected")
    current_consumables = zone_list(zones, "CurrentConsumables")
    current_hand_selected = zone_list(zones, "CurrentHandSelected")
    current_hand = zone_list(zones, "CurrentHand")
    tarot_hand_selected = zone_list(zones, "TarotSpectralHandSelected")
    tarot_hand = zone_list(zones, "TarotSpectralHand")
    pack_selected = zone_list(zones, "PackOfferingsSelected")
    pack_offerings = zone_list(zones, "PackOfferings")

    top_shelf_selected = zone_list(zones, "TopShelfShopOfferingsSelected")
    top_shelf = zone_list(zones, "TopShelfShopOfferings")
    shop_selected_compat = zone_list(zones, "ShopOfferingsSelected")
    shop_compat = zone_list(zones, "ShopOfferings")

    voucher_selected = zone_list(zones, "VoucherShopOfferingsSelected")
    voucher = zone_list(zones, "VoucherShopOfferings")
    pack_shop_selected = zone_list(zones, "PackShopOfferingsSelected")
    pack_shop = zone_list(zones, "PackShopOfferings")

    current_jokers_selected = zone_list(zones, "CurrentJokersSelected")
    current_jokers = zone_list(zones, "CurrentJokers")
    current_jokers_all = zone_list(zones, "CurrentJokersAll")
    joker_inventory = current_jokers if current_jokers else current_jokers_all

    max_consumable_targets = len(current_consumables_selected) + len(current_consumables)
    max_select_card_targets = max(
        len(current_hand_selected) + len(current_hand),
        len(tarot_hand_selected) + len(tarot_hand),
        len(pack_selected) + len(pack_offerings),
    )
    max_pack_item_targets = len(pack_selected) + len(pack_offerings)
    max_shop_consumable_targets = (
        sum(1 for obj in (top_shelf_selected + shop_selected_compat) if is_consumable(obj))
        + sum(1 for obj in (top_shelf + shop_compat) if is_consumable(obj))
    )
    max_buyshopitem_targets = (
        len(voucher_selected + pack_shop_selected + top_shelf_selected + shop_selected_compat)
        + len(voucher + pack_shop + top_shelf + shop_compat)
    )
    max_sellitem_targets = (
        len(current_jokers_selected)
        + len(current_consumables_selected)
        + len(joker_inventory)
        + len(current_consumables)
    )
    max_joker_slots = max(
        len(current_jokers_all),
        len(current_jokers),
        to_int(state.get("jokers_total") if isinstance(state, dict) else 0),
    )

    return {
        "MAX_CONSUMABLE_TARGETS": max_consumable_targets,
        "MAX_SELECT_CARD_TARGETS": max_select_card_targets,
        "MAX_PACK_ITEM_TARGETS": max_pack_item_targets,
        "MAX_SHOP_CONSUMABLE_TARGETS": max_shop_consumable_targets,
        "MAX_BUYSHOPITEM_TARGETS": max_buyshopitem_targets,
        "MAX_SELLITEM_TARGETS": max_sellitem_targets,
        "MAX_JOKER_SLOTS": max_joker_slots,
    }


def compute_maxima(src_root: Path) -> dict[str, Any]:
    file_count, event_count, records = iter_extracted_events(src_root)
    observed_maxima = {k: 0 for k in MAX_KEYS}
    max_provenance: dict[str, dict[str, Any] | None] = {k: None for k in MAX_KEYS}

    for extracted_file, event_index, event in records:
        counts = compute_counts(event)
        for key, value in counts.items():
            if value > observed_maxima[key]:
                observed_maxima[key] = value
                max_provenance[key] = {
                    "file": extracted_file.as_posix(),
                    "event_index": event_index,
                    "frame_idx": event.get("frame_idx"),
                    "page_name": event.get("page_name"),
                    "observed_value": value,
                }

    return {
        "files_scanned": file_count,
        "events_scanned": event_count,
        "observed_maxima": observed_maxima,
        "max_values": dict(observed_maxima),  # policy: no headroom
        "max_provenance": max_provenance,
    }


def load_locked_max_values(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("max_values", {}) if isinstance(payload, dict) else {}
    if not isinstance(values, dict):
        return {}
    out: dict[str, int] = {}
    for key in MAX_KEYS:
        out[key] = to_int(values.get(key))
    return out


def check_overflow(observed: dict[str, int], locked: dict[str, int]) -> list[str]:
    failures: list[str] = []
    for key in MAX_KEYS:
        if key not in locked:
            continue
        if observed[key] > locked[key]:
            failures.append(f"{key}: observed={observed[key]} exceeds locked={locked[key]}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute action-space MAX constants from extracted data using canonical "
            "zone counting rules."
        )
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/extracted"),
        help="Root containing extracted partitions (default: data/extracted)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/action_space_config.json"),
        help="Output config path (default: data/action_space_config.json)",
    )
    ap.add_argument(
        "--check-locked",
        type=Path,
        default=None,
        help=(
            "Optional locked config to validate against. Exits non-zero if observed maxima "
            "exceed locked max_values."
        ),
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write output JSON; compute and print only.",
    )
    args = ap.parse_args()

    stats = compute_maxima(args.src)
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": args.src.as_posix(),
        "policy": {
            "corpus": "extracted_only",
            "headroom": "none",
            "non_joker_counting_rule": "selected_plus_unselected_disjoint_sum",
            "joker_rule": "CurrentJokersAll is canonical slot basis; do not sum with CurrentJokers",
        },
        "counting_formulas": {
            "MAX_CONSUMABLE_TARGETS": "len(CurrentConsumablesSelected) + len(CurrentConsumables)",
            "MAX_SELECT_CARD_TARGETS": "max(len(CurrentHandSelected)+len(CurrentHand), len(TarotSpectralHandSelected)+len(TarotSpectralHand), len(PackOfferingsSelected)+len(PackOfferings))",
            "MAX_PACK_ITEM_TARGETS": "len(PackOfferingsSelected) + len(PackOfferings)",
            "MAX_SHOP_CONSUMABLE_TARGETS": "count_consumables(TopShelfShopOfferingsSelected + ShopOfferingsSelected) + count_consumables(TopShelfShopOfferings + ShopOfferings)",
            "MAX_BUYSHOPITEM_TARGETS": "len(VoucherShopOfferingsSelected + PackShopOfferingsSelected + TopShelfShopOfferingsSelected + ShopOfferingsSelected) + len(VoucherShopOfferings + PackShopOfferings + TopShelfShopOfferings + ShopOfferings)",
            "MAX_SELLITEM_TARGETS": "len(CurrentJokersSelected) + len(CurrentConsumablesSelected) + len(CurrentJokers or CurrentJokersAll) + len(CurrentConsumables)",
            "MAX_JOKER_SLOTS": "max(len(CurrentJokersAll), len(CurrentJokers), jokers_total)",
        },
        **stats,
    }

    if args.check_locked is not None:
        locked = load_locked_max_values(args.check_locked)
        failures = check_overflow(payload["observed_maxima"], locked)
        if failures:
            details = "\n".join(f"  - {line}" for line in failures)
            raise SystemExit("Locked MAX overflow detected:\n" + details)
        print(f"locked check passed against {args.check_locked.as_posix()}")

    print("files_scanned:", payload["files_scanned"])
    print("events_scanned:", payload["events_scanned"])
    for key in MAX_KEYS:
        print(f"{key}: {payload['max_values'][key]}")

    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.out.as_posix())


if __name__ == "__main__":
    main()
