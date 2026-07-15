#!/usr/bin/env python3
"""
history_features.py
===================
Shared rich-history tensorization for offline training and live inference.

For a target step ``t``, history contains only prior decision records
``< t``. Each prior record contributes one summary token plus a capped set
of local object tokens selected from the zones that explain that action.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from state_reducer import parse_base_action


HISTORY_STEPS_DEFAULT = 32
HISTORY_OBJECTS_PER_STEP_DEFAULT = 16
HISTORY_ACTION_PAD_ID = 0
HISTORY_ACTION_UNK_ID = 1
OCR_NUMERIC_KEYS = (
    "hands_left", "discards_left", "dollars", "ante", "round",
    "deck_remaining", "deck_total", "round_score", "cash_out",
    "reroll_price", "consumables_current", "consumables_total",
    "jokers_current", "jokers_total", "hand_size_current", "hand_size_total",
)
N_OCR = len(OCR_NUMERIC_KEYS)


def history_caps(feature_config: dict[str, Any]) -> tuple[int, int]:
    return (
        int(feature_config.get("HISTORY_STEPS", HISTORY_STEPS_DEFAULT)),
        int(
            feature_config.get(
                "HISTORY_OBJECTS_PER_STEP", HISTORY_OBJECTS_PER_STEP_DEFAULT
            )
        ),
    )


def _action_id(label: str | None, action_map: dict[str, Any]) -> int:
    if not label:
        return HISTORY_ACTION_PAD_ID
    idx = (action_map.get("label_to_index") or {}).get(label)
    if idx is None:
        return HISTORY_ACTION_UNK_ID
    return int(idx) + 2


def _int_plus_one(value: Any) -> int:
    return int(value) + 1 if isinstance(value, int) and value >= 0 else 0


def _state_numeric(
    step: dict[str, Any],
    norm: Any,
) -> tuple[np.ndarray, np.ndarray]:
    state = dict(step.get("state") or {})
    state.pop("hand_and_level_raw", None)
    state.pop("ocr_extra", None)
    values = np.zeros(N_OCR, dtype=np.float32)
    valid = np.zeros(N_OCR, dtype=bool)
    for i, key in enumerate(OCR_NUMERIC_KEYS):
        v = state.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            values[i] = norm.transform(f"ocr.{key}", v)
            valid[i] = True
    return values, valid


def _objects_by_zone(step: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in step.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        zone = obj.get("zone")
        if isinstance(zone, str):
            grouped.setdefault(zone, []).append(obj)
    for values in grouped.values():
        values.sort(
            key=lambda o: (
                int(o.get("position_in_zone") or 0),
                int(o.get("slot_id") or 0),
            )
        )
    return grouped


def _target_object(step: dict[str, Any]) -> dict[str, Any] | None:
    selected = step.get("selected_object")
    if isinstance(selected, dict) and isinstance(selected.get("object"), dict):
        return selected["object"]
    return None


def _slot_key(obj: dict[str, Any]) -> tuple[Any, ...]:
    if obj.get("slot_id") is not None:
        return ("slot", obj.get("slot_id"))
    return (
        "obj",
        obj.get("zone"),
        obj.get("position_in_zone"),
        obj.get("class_id"),
        obj.get("object_type"),
        obj.get("modifier"),
        obj.get("edition"),
        obj.get("seal"),
    )


def _label_parts(action: str | None) -> tuple[str, str | None]:
    action = action or ""
    if action.startswith("SWAP_"):
        return "SWAP", None
    base = parse_base_action(action)
    if "_" not in action:
        return base, None
    parts = action.rsplit("_", 1)
    try:
        int(parts[1])
    except (IndexError, ValueError):
        return base, None
    return base, parts[0]


def _history_zones(step: dict[str, Any]) -> list[str]:
    action = step.get("action") or ""
    page = step.get("page_name")
    base, subfamily = _label_parts(action)
    zones: list[str] = []

    if subfamily == "SelectCard_CurrentHand" or base in {"PlayHand", "DiscardHand"}:
        zones = ["PendingCards", "CurrentHand"]
    elif subfamily == "SelectCard_TarotSpectralHand":
        zones = ["PendingCards", "TarotSpectralHand", "PackOfferings"]
    elif subfamily == "SelectPackItem_PackOfferings":
        zones = ["PackOfferings", "PendingCards"]
        if page == "In_TarotSpectral_Pack":
            zones.append("TarotSpectralHand")
    elif subfamily == "UseConsumable_CurrentConsumables":
        zones = [
            "CurrentConsumables",
            "PendingCards",
            "CurrentHand",
            "TarotSpectralHand",
        ]
    elif subfamily == "BuyAndUseShopConsumable_TopShelfShopOfferings":
        zones = ["TopShelfShopOfferings"]
    elif subfamily and subfamily.startswith("BuyShopItem_"):
        zone = subfamily[len("BuyShopItem_") :]
        zones = [zone]
    elif subfamily == "SellItem_CurrentJokers" or base == "SWAP":
        zones = ["CurrentJokers"]
    elif subfamily == "SellItem_CurrentConsumables":
        zones = ["CurrentConsumables"]
    elif base in {"SelectBlind", "SkipBlind"}:
        zones = ["BlindOffering", "BlindOfferingsNext", "OfferedTag", "BigBlindTag"]
    else:
        zones = []

    # Preserve order while removing duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for zone in zones:
        if zone and zone not in seen:
            out.append(zone)
            seen.add(zone)
    return out


def relevant_history_objects(step: dict[str, Any], cap: int) -> list[dict[str, Any]]:
    grouped = _objects_by_zone(step)
    target = _target_object(step)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(obj: dict[str, Any] | None) -> None:
        if not isinstance(obj, dict):
            return
        key = _slot_key(obj)
        if key in seen:
            return
        selected.append(obj)
        seen.add(key)

    add(target)
    for zone in _history_zones(step):
        # Pending cards often explain the next commit, so keep them early.
        for obj in grouped.get(zone, []):
            add(obj)
            if len(selected) >= cap:
                return selected
    return selected[:cap]


def _encode_object_fields(
    obj: dict[str, Any],
    vocab: Any,
) -> dict[str, int | bool]:
    card = obj.get("card") or {}
    stickers = set(obj.get("stickers") or [])
    pos = obj.get("position_in_zone")
    return {
        "class_id": vocab.encode("class_id", obj.get("class_id")),
        "object_type_id": vocab.encode("object_type", obj.get("object_type")),
        "zone_id": vocab.encode("zone", obj.get("zone")),
        "position": int(pos) if isinstance(pos, int) else 0,
        "modifier_id": vocab.encode("modifier", obj.get("modifier")),
        "edition_id": vocab.encode("edition", obj.get("edition")),
        "seal_id": vocab.encode("seal", obj.get("seal")),
        "rank_id": vocab.encode("rank_index", card.get("rank_index")),
        "suit_id": vocab.encode("suit_index", card.get("suit_index")),
        "is_debuffed": bool(obj.get("is_debuffed", False)),
        "sticker_rental": "rental" in stickers,
        "sticker_perishable": "perishable" in stickers,
        "sticker_eternal": "eternal" in stickers,
    }


def empty_history_tensors(
    feature_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    history_steps, history_objects = history_caps(feature_config)
    return {
        "history_action_id": np.zeros(history_steps, dtype=np.int32),
        "history_page_id": np.zeros(history_steps, dtype=np.int32),
        "history_recency": np.arange(history_steps, dtype=np.int32),
        "history_target_zone_id": np.zeros(history_steps, dtype=np.int32),
        "history_target_position": np.zeros(history_steps, dtype=np.int32),
        "history_swap_i": np.zeros(history_steps, dtype=np.int32),
        "history_swap_j": np.zeros(history_steps, dtype=np.int32),
        "history_ocr_numeric": np.zeros((history_steps, N_OCR), dtype=np.float32),
        "history_ocr_valid": np.zeros((history_steps, N_OCR), dtype=bool),
        "history_step_mask": np.zeros(history_steps, dtype=bool),
        "history_object_class_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_object_type_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_zone_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_position": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_modifier_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_edition_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_seal_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_rank_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_suit_id": np.zeros(
            (history_steps, history_objects), dtype=np.int32
        ),
        "history_object_is_debuffed": np.zeros(
            (history_steps, history_objects), dtype=bool
        ),
        "history_object_sticker_rental": np.zeros(
            (history_steps, history_objects), dtype=bool
        ),
        "history_object_sticker_perishable": np.zeros(
            (history_steps, history_objects), dtype=bool
        ),
        "history_object_sticker_eternal": np.zeros(
            (history_steps, history_objects), dtype=bool
        ),
        "history_object_mask": np.zeros((history_steps, history_objects), dtype=bool),
    }


def build_history_tensors(
    history_steps_list: list[dict[str, Any]],
    *,
    action_map: dict[str, Any],
    vocab: VocabLookup,
    norm: Normalizer,
    feature_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    out = empty_history_tensors(feature_config)
    history_steps, history_objects = history_caps(feature_config)
    recent = list(history_steps_list)[-history_steps:][::-1]

    for h, step in enumerate(recent):
        action = step.get("action")
        out["history_action_id"][h] = _action_id(action, action_map)
        out["history_page_id"][h] = vocab.encode("page", step.get("page_name"))
        out["history_recency"][h] = h
        out["history_target_zone_id"][h] = vocab.encode(
            "zone", step.get("target_zone")
        )
        out["history_target_position"][h] = _int_plus_one(
            step.get("target_position")
        )
        swap = step.get("swap_pair")
        if isinstance(swap, (list, tuple)) and len(swap) == 2:
            out["history_swap_i"][h] = _int_plus_one(swap[0])
            out["history_swap_j"][h] = _int_plus_one(swap[1])
        ocr, valid = _state_numeric(step, norm)
        out["history_ocr_numeric"][h] = ocr
        out["history_ocr_valid"][h] = valid
        out["history_step_mask"][h] = True

        for j, obj in enumerate(relevant_history_objects(step, history_objects)):
            fields = _encode_object_fields(obj, vocab)
            for key, value in fields.items():
                out[f"history_object_{key}"][h, j] = value
            out["history_object_mask"][h, j] = True

    return out


class HistoryBuffer:
    """Small rolling history used by live inference."""

    def __init__(self, maxlen: int) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._records.clear()

    def append(self, snapshot_step: dict[str, Any], action_label: str) -> None:
        step = dict(snapshot_step)
        step["action"] = action_label
        self._records.append(step)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)
