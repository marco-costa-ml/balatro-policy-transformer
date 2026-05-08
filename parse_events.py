#!/usr/bin/env python3
"""
parse_events.py
===============
Reads  data/extracted/video_id=*/events.json
Writes data/parsed/video_id=*/run_000.json, run_001.json, …
       data/parsed/config.json

Events are split into runs on "StartNewRun" actions.
Events that appear before the first StartNewRun in a video are discarded.

Each run file contains:
  {
    "video_id": str,
    "run_index": int,
    "events": [
      {
        "frame_idx": int,
        "page_name": str,
        "state": { hands_left, discards_left, dollars, ante, round,
                   deck_remaining, deck_total, round_score },
        "objects": [ { slot_id, zone, position_in_zone, class_id,
                        object_type, card, edition, modifier,
                        seal, stickers, is_debuffed } ],
        "actions": [ str ]
      }
    ]
  }

Usage:
    python parse_events.py [--src data/extracted] [--dst data/parsed]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Class ID constants  (authoritative: data/class_map.csv)
# ---------------------------------------------------------------------------
CARD_MIN = 0
CARD_MAX = 51
FACEDOWN_CLASS_ID = 231
EDITION_MIN = 68
EDITION_MAX = 71
MODIFIER_MIN = 72
MODIFIER_MAX = 79
DEBUFFED_CLASS_ID = 230
SEAL_MIN = 232
SEAL_MAX = 235
# Stickers: rental=367, perishable=368, eternal=369
STICKER_IDS = {367, 368, 369}

SUITS = ["spades", "hearts", "diamonds", "clubs"]
RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
FACE_RANKS = {"jack", "queen", "king"}

SPLIT_ACTION = "StartNewRun"


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------

def load_class_map(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(row["class_id"]): row["class_name"] for row in csv.DictReader(fh)}


# ---------------------------------------------------------------------------
# OCR parsing
# ---------------------------------------------------------------------------

def _to_int(value: Any) -> int | None:
    """Strip all non-digit characters and cast to int; None on failure."""
    if value is None:
        return None
    s = re.sub(r"[^\d]", "", str(value))
    return int(s) if s else None


def _split_pair(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    parts = str(value).split("/")
    left = _to_int(parts[0]) if len(parts) >= 1 else None
    right = _to_int(parts[1]) if len(parts) >= 2 else None
    return left, right


def parse_ocr(ocr: dict[str, Any]) -> dict[str, Any]:
    """
    Parse OCR game-state dict into clean typed values.

    Raw format → parsed:
      hands_left     int        → int
      discards_left  int        → int
      dollars        "$4"       → strip non-digits, int
      ante           "1/8"      → numerator only, int
      round          int        → int
      deck_values    "44/52"    → deck_remaining (int), deck_total (int)
      round_score    "*0"       → strip non-digits, int
    """
    ante_raw = ocr.get("ante")
    ante = _to_int(str(ante_raw).split("/")[0]) if ante_raw is not None else None
    deck_remaining, deck_total = _split_pair(ocr.get("deck_values"))
    consumables_current, consumables_total = _split_pair(ocr.get("consumable_values"))
    jokers_current, jokers_total = _split_pair(ocr.get("joker_values"))
    hand_size_current, hand_size_total = _split_pair(ocr.get("hand_values"))

    known_ocr_keys = {
        "hands_left", "discards_left", "dollars", "ante", "round",
        "deck_values", "round_score", "cash_out", "consumable_values",
        "joker_values", "hand_values", "reroll_price", "hand_and_level",
    }
    ocr_extra = {k: v for k, v in ocr.items() if k not in known_ocr_keys}

    return {
        "hands_left":     _to_int(ocr.get("hands_left")),
        "discards_left":  _to_int(ocr.get("discards_left")),
        "dollars":        _to_int(ocr.get("dollars")),
        "ante":           ante,
        "round":          _to_int(ocr.get("round")),
        "deck_remaining": deck_remaining,
        "deck_total":     deck_total,
        "round_score":    _to_int(ocr.get("round_score")),
        "cash_out":       _to_int(ocr.get("cash_out")),
        "reroll_price":   _to_int(ocr.get("reroll_price")),
        "consumables_current": consumables_current,
        "consumables_total": consumables_total,
        "jokers_current": jokers_current,
        "jokers_total": jokers_total,
        "hand_size_current": hand_size_current,
        "hand_size_total": hand_size_total,
        "hand_and_level_raw": (
            str(ocr.get("hand_and_level")) if ocr.get("hand_and_level") is not None else None
        ),
        "ocr_extra": ocr_extra,
    }


# ---------------------------------------------------------------------------
# Card info
# ---------------------------------------------------------------------------

def _card_fields(class_id: int) -> dict[str, Any]:
    suit_index = class_id // 13
    rank_index = class_id % 13
    rank = RANKS[rank_index]
    suit = SUITS[suit_index]
    return {
        "rank":       rank,
        "rank_index": rank_index,
        "suit":       suit,
        "suit_index": suit_index,
        "is_ace":     rank == "ace",
        "is_face":    rank in FACE_RANKS,
    }


def _unknown_card_fields() -> dict[str, Any]:
    return {
        "rank": None,
        "rank_index": None,
        "suit": None,
        "suit_index": None,
        "is_ace": None,
        "is_face": None,
    }


# ---------------------------------------------------------------------------
# Object type inference
# ---------------------------------------------------------------------------

def _object_type(class_id: int, class_map: dict[int, str]) -> str:
    if CARD_MIN <= class_id <= CARD_MAX:       return "card"
    if class_id == FACEDOWN_CLASS_ID:          return "card"
    if 52 <= class_id <= 67:                   return "deck"
    if EDITION_MIN <= class_id <= EDITION_MAX: return "edition"
    if MODIFIER_MIN <= class_id <= MODIFIER_MAX: return "modifier"
    if class_id == DEBUFFED_CLASS_ID:          return "debuffed_marker"
    if SEAL_MIN <= class_id <= SEAL_MAX:       return "seal"
    if class_id in STICKER_IDS:               return "sticker"
    name = class_map.get(class_id, "")
    if name.startswith("j_"):      return "joker"
    if name.startswith("c_"):      return "consumable"
    if name.startswith("v_"):      return "voucher"
    if name.startswith("bl_"):     return "blind"
    if name.startswith("p_"):      return "pack"
    if name.startswith("tag_"):    return "tag"
    if name.startswith("stake_"):  return "stake"
    return "unknown"


# ---------------------------------------------------------------------------
# Object parsing
# ---------------------------------------------------------------------------

def _resolve(value: Any) -> int | None:
    """Cast a nullable class-ID field to int, or None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_object(
    zone: str,
    raw: dict[str, Any],
    class_map: dict[int, str],
) -> dict[str, Any] | None:
    """
    Normalise one raw zone-object into a clean parsed record.

    Kept   : class_id, position_in_zone (object-level)
             slot_id, edition/modifier/seal/sticker/debuffed class IDs (children)
    Dropped: score, provenance, bbox (object-level)
             video_id, age_frames, time_on_screen_seconds, obs_count,
             track_density, is_observed, x1/y1/x2/y2, *_slot_id (children)
    """
    if not isinstance(raw, dict):
        return None

    class_id = _resolve(raw.get("class_id"))
    if class_id is None:
        return None

    children_raw = raw.get("children")
    children: dict[str, Any] = children_raw if isinstance(children_raw, dict) else {}

    card: dict[str, Any] | None
    if CARD_MIN <= class_id <= CARD_MAX:
        card = _card_fields(class_id)
    elif class_id == FACEDOWN_CLASS_ID:
        card = _unknown_card_fields()
    else:
        card = None

    # Edition — cards, jokers, consumables, planets, spectrals, tarots
    edition: str | None = None
    ecid = _resolve(children.get("edition_class_id"))
    if ecid is not None and EDITION_MIN <= ecid <= EDITION_MAX:
        edition = class_map.get(ecid)

    # Modifier — cards only
    modifier: str | None = None
    mcid = _resolve(children.get("modifier_class_id"))
    if mcid is not None and MODIFIER_MIN <= mcid <= MODIFIER_MAX:
        modifier = class_map.get(mcid)

    # Seal — cards only
    seal: str | None = None
    scid = _resolve(children.get("seal_class_id"))
    if scid is not None and SEAL_MIN <= scid <= SEAL_MAX:
        seal = class_map.get(scid)

    # Stickers — jokers only; up to two
    stickers: list[str] = []
    for field in ("sticker_1_class_id", "sticker_2_class_id"):
        sid = _resolve(children.get(field))
        if sid is not None and sid in STICKER_IDS:
            name = class_map.get(sid)
            if name:
                stickers.append(name)

    # Debuffed — cards and jokers
    dcid = _resolve(children.get("debuffed_class_id"))
    is_debuffed = dcid == DEBUFFED_CLASS_ID

    slot_id = _resolve(children.get("slot_id"))
    pos = _resolve(raw.get("position_in_zone"))

    return {
        "slot_id":          slot_id if slot_id is not None else 0,
        "zone":             zone,
        "position_in_zone": pos if pos is not None else 0,
        "class_id":         class_id,
        "object_type":      _object_type(class_id, class_map),
        "card":             card,
        "edition":          edition,
        "modifier":         modifier,
        "seal":             seal,
        "stickers":         stickers,
        "is_debuffed":      is_debuffed,
    }


def parse_zones(
    zones: dict[str, list[dict[str, Any]]],
    class_map: dict[int, str],
) -> list[dict[str, Any]]:
    """Flatten all zone objects into a single list sorted by zone then position.

    Missing zones are allowed. Duplicate objects are preserved as-is.
    """
    objects: list[dict[str, Any]] = []
    if not isinstance(zones, dict):
        return objects

    for zone_name, zone_objs in zones.items():
        # Be tolerant to malformed single-object zone payloads.
        if isinstance(zone_objs, dict):
            iter_objs = [zone_objs]
        elif isinstance(zone_objs, list):
            iter_objs = zone_objs
        else:
            continue

        for raw_obj in iter_objs:
            parsed = parse_object(str(zone_name), raw_obj, class_map)
            if parsed is not None:
                objects.append(parsed)

    objects.sort(key=lambda o: (o["zone"], o["position_in_zone"], o["slot_id"]))
    return objects


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_actions(raw_actions: Any) -> list[str]:
    """Extract normalized action-name strings from mixed action encodings."""
    if raw_actions is None:
        return []

    # Allow a single action string payload.
    if isinstance(raw_actions, str):
        return [raw_actions]

    # Allow one object payload: {"type": "..."}.
    if isinstance(raw_actions, dict):
        if "type" in raw_actions:
            return [str(raw_actions["type"])]
        return []

    out: list[str] = []
    if isinstance(raw_actions, list):
        for action in raw_actions:
            if isinstance(action, str):
                out.append(action)
            elif isinstance(action, dict) and "type" in action:
                out.append(str(action["type"]))
    return out


def _extract_action_subtype(action_id: str) -> str | None:
    """
    Extract subtype token from predictor-style IDs:
      pred_<subtype>_<frame>_<idx>
    Example:
      pred_buyvoucher_12258_47 -> buyvoucher
    """
    m = re.fullmatch(r"pred_(.+)_\d+_\d+", action_id)
    return m.group(1) if m else None


def parse_action_details(raw_actions: Any) -> list[dict[str, Any]]:
    """
    Parse action metadata while preserving type/id/subtype.

    Output schema:
      [{ "type": str, "id": str | null, "subtype": str | null }, ...]
    """
    details: list[dict[str, Any]] = []

    if raw_actions is None:
        return details

    if isinstance(raw_actions, str):
        details.append({"type": raw_actions, "id": None, "subtype": None})
        return details

    if isinstance(raw_actions, dict):
        action_type = raw_actions.get("type")
        if action_type is None:
            return details
        action_id = raw_actions.get("id")
        action_id_str = str(action_id) if action_id is not None else None
        details.append({
            "type": str(action_type),
            "id": action_id_str,
            "subtype": _extract_action_subtype(action_id_str) if action_id_str else None,
        })
        return details

    if isinstance(raw_actions, list):
        for action in raw_actions:
            if isinstance(action, str):
                details.append({"type": action, "id": None, "subtype": None})
                continue
            if isinstance(action, dict) and action.get("type") is not None:
                action_id = action.get("id")
                action_id_str = str(action_id) if action_id is not None else None
                details.append({
                    "type": str(action["type"]),
                    "id": action_id_str,
                    "subtype": _extract_action_subtype(action_id_str) if action_id_str else None,
                })
    return details


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

def parse_event(raw: dict[str, Any], class_map: dict[int, str]) -> dict[str, Any]:
    """Parse one raw event into a clean nested record (no video_id; set by caller)."""
    if not isinstance(raw, dict):
        raw = {}

    state_raw = raw.get("state")
    if not isinstance(state_raw, dict):
        state_raw = {}

    ocr_raw = state_raw.get("ocr")
    if not isinstance(ocr_raw, dict):
        ocr_raw = {}

    zones_raw = state_raw.get("zones")
    if not isinstance(zones_raw, dict):
        zones_raw = {}

    frame_idx = _resolve(raw.get("frame_idx"))
    action_details = parse_action_details(raw.get("actions"))
    return {
        "frame_idx": frame_idx if frame_idx is not None else 0,
        "page_name": str(raw.get("page_name") or ""),
        "state":     parse_ocr(ocr_raw),
        "objects":   parse_zones(zones_raw, class_map),
        "actions":   parse_actions(raw.get("actions")),
        "action_details": action_details,
    }


# ---------------------------------------------------------------------------
# Run splitting
# ---------------------------------------------------------------------------

def split_into_runs(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """
    Split a flat list of parsed events into runs.

    A new run starts whenever SPLIT_ACTION ("StartNewRun") appears in
    event["actions"]. The StartNewRun event is included as the first event
    of its run. Events before the first StartNewRun are discarded.

    If no StartNewRun exists but events are present (trimmed datasets),
    emit one fallback run containing all events.
    """
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for event in events:
        if SPLIT_ACTION in event["actions"]:
            if current is not None:
                runs.append(current)
            current = [event]
        elif current is not None:
            current.append(event)
    if current:
        runs.append(current)
    elif events:
        runs.append(events)
    return runs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_partitions(src_root: Path) -> list[tuple[str, Path]]:
    """Return [(video_id, json_path)] for every valid partition folder."""
    results: list[tuple[str, Path]] = []
    if not src_root.exists():
        return results
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("video_id="):
            continue
        json_path = entry / "events.json"
        if json_path.exists():
            video_id = entry.name.split("=", 1)[1]
            results.append((video_id, json_path))
    return results


def load_events(json_path: Path) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        events = data.get("events")
    elif isinstance(data, list):
        events = data
    else:
        events = []
    return events if isinstance(events, list) else []


def write_run(
    video_id: str,
    run_index: int,
    events: list[dict[str, Any]],
    dst_dir: Path,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id":  video_id,
        "run_index": run_index,
        "events":    events,
    }
    out = dst_dir / ("run_%03d.json" % run_index)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    run_%03d.json  (%d events)" % (run_index, len(events)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def write_config(
    dst_root: Path,
    src_root: Path,
    observed_zones: set[str] | None = None,
    observed_ocr_keys: set[str] | None = None,
) -> None:
    zones_known = sorted(observed_zones or set())
    ocr_known = sorted(observed_ocr_keys or set())
    config = {
        "schema_version": "1.3.0",
        "source_directory": src_root.as_posix(),
        "output_directory": dst_root.as_posix(),
        "partition_format": "video_id={video_id}",
        "file_name": "run_{index:03d}.json",
        "split_action": SPLIT_ACTION,
        "note": (
            "Events before the first StartNewRun in each video are discarded. "
            "The StartNewRun event itself is included as the first event of its run. "
            "Missing OCR keys remain null, omitted zones produce no objects, and duplicate "
            "objects are preserved."
        ),
        "observed_ocr_keys": ocr_known,
        "ocr_fields": {
            "hands_left": {
                "type": "int | null",
                "raw_key": "state.ocr.hands_left",
                "transform": "direct int cast",
            },
            "discards_left": {
                "type": "int | null",
                "raw_key": "state.ocr.discards_left",
                "transform": "direct int cast",
            },
            "dollars": {
                "type": "int | null",
                "raw_key": "state.ocr.dollars",
                "transform": "strip non-digits (e.g. '$'), cast int",
            },
            "ante": {
                "type": "int | null",
                "raw_key": "state.ocr.ante",
                "transform": "split on '/', take numerator, cast int",
            },
            "round": {
                "type": "int | null",
                "raw_key": "state.ocr.round",
                "transform": "direct int cast",
            },
            "deck_remaining": {
                "type": "int | null",
                "raw_key": "state.ocr.deck_values",
                "transform": "split on '/', take numerator, cast int",
            },
            "deck_total": {
                "type": "int | null",
                "raw_key": "state.ocr.deck_values",
                "transform": "split on '/', take denominator, cast int",
            },
            "round_score": {
                "type": "int | null",
                "raw_key": "state.ocr.round_score",
                "transform": "strip non-digits (e.g. '*'), cast int",
            },
            "cash_out": {
                "type": "int | null",
                "raw_key": "state.ocr.cash_out",
                "transform": "strip non-digits, cast int",
            },
            "reroll_price": {
                "type": "int | null",
                "raw_key": "state.ocr.reroll_price",
                "transform": "strip non-digits, cast int",
            },
            "consumables_current": {
                "type": "int | null",
                "raw_key": "state.ocr.consumable_values",
                "transform": "split on '/', take numerator, cast int",
            },
            "consumables_total": {
                "type": "int | null",
                "raw_key": "state.ocr.consumable_values",
                "transform": "split on '/', take denominator, cast int",
            },
            "jokers_current": {
                "type": "int | null",
                "raw_key": "state.ocr.joker_values",
                "transform": "split on '/', take numerator, cast int",
            },
            "jokers_total": {
                "type": "int | null",
                "raw_key": "state.ocr.joker_values",
                "transform": "split on '/', take denominator, cast int",
            },
            "hand_size_current": {
                "type": "int | null",
                "raw_key": "state.ocr.hand_values",
                "transform": "split on '/', take numerator, cast int",
            },
            "hand_size_total": {
                "type": "int | null",
                "raw_key": "state.ocr.hand_values",
                "transform": "split on '/', take denominator, cast int",
            },
            "hand_and_level_raw": {
                "type": "string | null",
                "raw_key": "state.ocr.hand_and_level",
                "transform": "string passthrough",
            },
            "ocr_extra": {
                "type": "object",
                "raw_key": "state.ocr.*",
                "transform": "pass through unknown OCR keys",
            },
        },
        "object_fields": {
            "slot_id": {
                "type": "int",
                "source": "children.slot_id",
            },
            "zone": {
                "type": "string",
                "source": "state.zones key",
                "known_values": zones_known,
            },
            "position_in_zone": {
                "type": "int",
                "source": "object.position_in_zone",
            },
            "class_id": {
                "type": "int",
                "source": "object.class_id",
                "description": "Detector class ID 0-399; see data/class_map.csv",
            },
            "object_type": {
                "type": "string",
                "description": (
                    "Inferred from class_id: card | deck | joker | consumable | voucher | "
                    "blind | pack | tag | stake | edition | modifier | "
                    "seal | sticker | debuffed_marker | unknown"
                ),
            },
            "card": {
                "type": "null | object",
                "description": "Non-null for playing cards and facedown cards (class_id 0-51, 231)",
                "fields": {
                    "rank":       "string — ace, 2-10, jack, queen, king",
                    "rank_index": "int — 0 (ace) to 12 (king)",
                    "suit":       "string — spades | hearts | diamonds | clubs",
                    "suit_index": "int — 0 (spades) to 3 (clubs)",
                    "is_ace":     "bool",
                    "is_face":    "bool — true for jack, queen, king",
                },
            },
            "edition": {
                "type": "null | string",
                "source": "children.edition_class_id",
                "class_id_range": [68, 71],
                "values": ["e_foil", "e_holo", "e_negative", "e_polychrome"],
            },
            "modifier": {
                "type": "null | string",
                "source": "children.modifier_class_id",
                "class_id_range": [72, 79],
                "values": [
                    "m_bonus", "m_glass", "m_gold", "m_lucky",
                    "m_mult", "m_steel", "m_stone", "m_wild",
                ],
            },
            "seal": {
                "type": "null | string",
                "source": "children.seal_class_id",
                "class_id_range": [232, 235],
                "values": ["blue_seal", "gold_seal", "purple_seal", "red_seal"],
            },
            "stickers": {
                "type": "[string]",
                "source": "children.sticker_1_class_id, children.sticker_2_class_id",
                "class_ids": [367, 368, 369],
                "values": ["rental", "perishable", "eternal"],
                "max_count": 2,
            },
            "is_debuffed": {
                "type": "bool",
                "source": "children.debuffed_class_id",
                "trigger_class_id": 230,
            },
        },
        "action_fields": {
            "actions": {
                "type": "[string]",
                "source": "event.actions (supports [{'type': ...}], [string], string)",
                "description": "Action names at this decision point, e.g. PlayHand, DiscardHand, StartNewRun",
            },
            "action_details": {
                "type": "[object]",
                "source": "event.actions[].{type,id}",
                "description": "Action metadata preserving type/id and extracted subtype when id matches pred_<subtype>_<frame>_<idx>",
                "fields": {
                    "type": "string — canonical action name (e.g. BuyShopItem)",
                    "id": "null | string — raw detector/predictor action id",
                    "subtype": "null | string — extracted subtype token (e.g. buyvoucher)",
                },
            },
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
    ap = argparse.ArgumentParser(description="Parse Balatro extracted events.json files into per-run JSONs.")
    ap.add_argument("--src", type=Path, default=Path("data/extracted"),
                    help="Root of extracted partitions (default: data/extracted)")
    ap.add_argument("--dst", type=Path, default=Path("data/parsed"),
                    help="Root for parsed output (default: data/parsed)")
    ap.add_argument("--class-map", type=Path, default=Path("data/class_map.csv"),
                    help="Path to class_map.csv (default: data/class_map.csv)")
    args = ap.parse_args(argv)

    src_root: Path = args.src
    dst_root: Path = args.dst
    class_map_path: Path = args.class_map

    if not class_map_path.exists():
        print("ERROR: class map not found at %s" % class_map_path, file=sys.stderr)
        sys.exit(1)

    class_map = load_class_map(class_map_path)
    print("loaded class map: %d classes" % len(class_map))

    partitions = find_partitions(src_root)
    observed_zones: set[str] = set()
    observed_ocr_keys: set[str] = set()
    if not partitions:
        print("no events.json files found under %s" % src_root)
        write_config(dst_root, src_root, observed_zones, observed_ocr_keys)
        return

    print("found %d partition(s)\n" % len(partitions))

    for video_id, json_path in partitions:
        print("video_id=%s" % video_id)
        raw_events = load_events(json_path)
        print("  loaded %d raw events" % len(raw_events))

        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            state_raw = raw.get("state")
            if not isinstance(state_raw, dict):
                continue
            ocr_raw = state_raw.get("ocr")
            if isinstance(ocr_raw, dict):
                observed_ocr_keys.update(str(k) for k in ocr_raw.keys())
            zones_raw = state_raw.get("zones")
            if isinstance(zones_raw, dict):
                observed_zones.update(str(k) for k in zones_raw.keys())

        parsed = [parse_event(ev, class_map) for ev in raw_events]
        runs = split_into_runs(parsed)
        print("  %d run(s) found" % len(runs))

        dst_dir = dst_root / ("video_id=%s" % video_id)
        for run_idx, run_events in enumerate(runs):
            write_run(video_id, run_idx, run_events, dst_dir)

    write_config(dst_root, src_root, observed_zones, observed_ocr_keys)
    print("\ndone.")


if __name__ == "__main__":
    main()
