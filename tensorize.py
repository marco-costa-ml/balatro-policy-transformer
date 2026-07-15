#!/usr/bin/env python3
"""
tensorize.py
============
Materialize the on-disk training dataset from granularized + persistent_state.

Schema version 2.0.0 (matches ``granularize.py`` schema 3.0.0): the input
``step.objects`` is already canonical (no ``*Selected`` zones, ``*All``
collapsed to base names, ``PendingCards`` already inlined). No
canonicalization or selected-zone merging is performed here.

Produces, for each granularized run, a single ``.npz`` shard at
``data/tensorized/video_id={vid}/run_{NNN}.npz`` containing
``n_steps``-aligned arrays for every channel listed in
``artifacts/feature_config.json``. The schema is locked by:

- ``artifacts/vocab.json`` -> categorical IDs
- ``artifacts/normalization.json`` -> numeric transforms
- ``data/action_map.json`` -> target_action_id + action_mask layout

Public API:

- ``tensorize_step(step, persistent_state, action_map, vocabs, normalizer,
                  feature_config) -> dict[str, np.ndarray]``

For per-step debugging or single-record consumption.

CLI:
- ``python tensorize.py [--src data/granularized] [--persistent data/persistent_state]
       [--out data/tensorized] [--report artifacts/tensorizer_report.json]
       [--max-ante N]``

  ``--max-ante N`` keeps only super-steps whose encoder snapshot has OCR
  ``state.ante <= N``. Steps with a missing or non-numeric ``ante`` are
  skipped when this filter is active.

Per-step record schema (one element per step in every per-run array):

    GLOBAL CATEGORICAL (int32)
      page_id                                ()  via vocab["page"]
      source_kind_id                         ()  via vocab["source_kind"]
      action_subtype_id                      ()  via vocab["action_subtype"]
      deck_class_id                          ()  via vocab["deck_class_id"]
      stake_class_id                         ()  via vocab["stake_class_id"]
      last_tarot_planet_class_id             ()  via vocab["last_tarot_planet_class_id"]
      ante_boss_blind_class_id               ()  via vocab["ante_boss_blind_class_id"]
      small_status_id                        ()  via vocab["small_status"]
      big_status_id                          ()  via vocab["big_status"]

    GLOBAL NUMERIC (float32)  -- normalized per artifacts/normalization.json
      ocr_numeric                           (16,)  16 OCR fields, fixed order
      state_numeric                         ( 4,)  skips, hands_played, unused_discards, ecto_minus

    GLOBAL FLAGS (bool)
      flags                                 (12,)  3 bool persistent + 6 deck flags + 3 deck_modifiers
      ocr_valid                             (16,)  per-OCR-field validity bit (1=present, 0=missing)

    PER-HAND ARRAYS (float32)
      hand_levels                           (12,)  log1p_zscore-normalized
      hand_played                           (12,)  log1p_zscore-normalized
      hand_played_this_round                (12,)  minmax-normalized

    SUPPLEMENT FEATURES (float32)
      supplement_features                   (62,)  derived poker / joker /
                                                   held-card / game-state
                                                   flags + counts, see
                                                   supplement_features.py for
                                                   the locked feature order.

    INVENTORY MULTI-HOT (bool)
      vouchers_redeemed                     (32,)  bit per voucher class id (320..351)
      bosses_used                           (28,)  bit per boss class id

    TRACKED DECK [TRACKED_DECK_CAP=75 cap] (int32 + bool)
      deck_card_class_id                    (75,)
      deck_card_modifier_id                 (75,)
      deck_card_edition_id                  (75,)
      deck_card_seal_id                     (75,)
      deck_card_rank_id                     (75,)
      deck_card_suit_id                     (75,)
      deck_card_mask                        (75,)  bool

    PER-OBJECT [MAX_OBJECTS_PER_STEP=32 cap] (int32 + bool)
      object_class_id                       (32,)
      object_object_type_id                 (32,)
      object_zone_id                        (32,)
      object_position                       (32,)
      object_modifier_id                    (32,)
      object_edition_id                     (32,)
      object_seal_id                        (32,)
      object_rank_id                        (32,)
      object_suit_id                        (32,)
      object_is_debuffed                    (32,)  bool
      object_sticker_rental                 (32,)  bool
      object_sticker_perishable             (32,)  bool
      object_sticker_eternal                (32,)  bool
      object_mask                           (32,)  bool

    ACTION TARGET (bool + int32) -- legacy flat head, kept for back-compat
      action_mask                           (N_ACTIONS,)
      target_action_id                      ()    -1 if label could not be resolved

    BRANCHED POLICY TARGETS + MASKS (schema 3.0.0)
      family_id                             ()       -1 if reserved / unresolved
      num_cards                             ()       -1 if N/A; 0..MAX_CARDS_PER_DECISION
      item_ptr_local                        ()       -1 if family has no item pointer
      item_ptr_slot                         ()       slot_id (debug / live unroll)
      card_ptr_local_seq                    (5,)     -1 padding (MAX_CARDS_PER_DECISION)
      card_ptr_slot_seq                     (5,)     slot_ids (debug / live unroll)
      swap_i_local, swap_j_local            ()       -1 if not a SWAP super-step
      swap_i_slot,  swap_j_slot             ()       slot_ids (debug / live unroll)
      family_mask                           (N_FAMILIES,)  parent-start legality
      item_pointer_mask                     (15,)    MAX_ITEM_ZONE_SIZE
      card_pointer_mask                     (15,)    MAX_CARD_ZONE_SIZE
      swap_joker_mask                       (10,)    MAX_JOKER_SLOTS

    For schema 3.0.0 each row is one PARENT-LEVEL super-step (not a granular
    micro-step). Select-then-commit blocks collapse to one row; each SWAP
    synth is its own row. See ``super_step.iter_super_steps``.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from action_map import compute_action_map
from argument_spec import MAX_CARDS_PER_DECISION
from family_map import compute_family_map, family_id_for_step, family_name_for_step
from history_features import build_history_tensors, history_caps
from label_resolver import LabelResolutionError, resolve_label
from mask_builder import (
    build_action_mask,
    build_card_pointer_mask,
    build_family_mask,
    build_item_pointer_mask,
    build_swap_joker_mask,
)
from state_reducer import apply_step, default_state, parse_base_action
from super_step import SuperStep, iter_super_steps, make_history_record
from supplement_features import compute_supplement_features


TENSORIZE_SCHEMA_VERSION = "3.0.0"

_WORKER_ACTION_MAP: dict[str, Any] | None = None
_WORKER_VOCAB: "VocabLookup | None" = None
_WORKER_NORM: "Normalizer | None" = None
_WORKER_FEATURE_CONFIG: dict[str, Any] | None = None
_WORKER_FAMILY_MAP: dict[str, Any] | None = None
_WORKER_BRANCHED_CAPS: dict[str, int] | None = None

# OCR channels in canonical order (matches normalization.json keys).
OCR_NUMERIC_KEYS = (
    "hands_left", "discards_left", "dollars", "ante", "round",
    "deck_remaining", "deck_total", "round_score", "cash_out",
    "reroll_price", "consumables_current", "consumables_total",
    "jokers_current", "jokers_total", "hand_size_current", "hand_size_total",
)
N_OCR = len(OCR_NUMERIC_KEYS)  # 16

# Persistent counters (matches normalization.json prefixes).
STATE_NUMERIC_KEYS = ("skips", "hands_played", "unused_discards", "ecto_minus")
N_STATE_NUM = len(STATE_NUMERIC_KEYS)  # 4

# Bool-flag layout. Order is fixed for the model's input contract.
PERSISTENT_BOOL_FIELDS = ("first_hand", "first_discard", "is_boss_blind_rerolled")
DECK_FLAG_FIELDS = (
    "is_magic", "is_nebula", "is_abandoned", "is_checkered",
    "is_zodiac", "is_erratic",
)
DECK_MODIFIER_FIELDS = (
    "no_face_cards_start", "spades_hearts_only_start", "randomized_starting_deck",
)
N_FLAGS = len(PERSISTENT_BOOL_FIELDS) + len(DECK_FLAG_FIELDS) + len(DECK_MODIFIER_FIELDS)  # 12

# With granularize 3.0.0+, ``step.objects`` is already the canonical,
# leak-free, normalized list (no ``*Selected`` zones, ``*All`` collapsed
# to base names, PendingCards inlined). Tensorize no longer rewrites
# zone names or shuffles dynamic pools.


# ---------------------------------------------------------------------------
# Vocab + normalizer wrappers
# ---------------------------------------------------------------------------

class VocabLookup:
    """Lightweight categorical-encoder. PAD = 0 for every channel."""

    def __init__(self, vocab_payload: dict[str, Any]) -> None:
        raw = vocab_payload["vocabularies"]
        self._maps: dict[str, dict[str, int]] = {}
        self._sizes: dict[str, int] = {}
        for name, entry in raw.items():
            self._sizes[name] = int(entry.get("size", 1))
            if name == "class_id":
                # Identity encoding; nothing to store.
                self._maps[name] = {}
            else:
                self._maps[name] = dict(entry["value_to_index"])
        # CLASS_ID OOV slot.
        self._class_id_size = int(raw["class_id"]["size"])
        self._class_id_oov = int(raw["class_id"]["oov_index"])

    def size(self, name: str) -> int:
        return self._sizes[name]

    def encode(self, name: str, value: Any) -> int:
        """Map a value to its vocab index. Returns 0 (PAD) for None / unseen."""
        if value is None:
            return 0
        if name == "class_id":
            try:
                cid = int(value)
            except (TypeError, ValueError):
                return 0
            if 0 <= cid < self._class_id_size - 1:
                return cid
            return self._class_id_oov
        return self._maps[name].get(str(value), 0)


class Normalizer:
    """Apply the locked numeric transforms from normalization.json."""

    def __init__(self, norm_payload: dict[str, Any]) -> None:
        self._transforms: dict[str, str] = dict(norm_payload["transforms"])
        self._stats: dict[str, dict[str, float]] = dict(norm_payload["stats"])

    def transform(self, key: str, value: float | int | None) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            # Missing values become 0 (mean for z-score after subtracting mean=0;
            # neutral for minmax); validity bit communicates missingness separately.
            return 0.0
        x = float(value)
        kind = self._transforms.get(key, "zscore")
        st = self._stats.get(key) or {}
        mean = float(st.get("mean", 0.0))
        std = max(float(st.get("std", 1.0)), 1e-6)
        mn = float(st.get("min", 0.0))
        mx = float(st.get("max", 1.0))
        if kind == "identity":
            return x
        if kind == "minmax":
            denom = max(mx - mn, 1.0)
            return (x - mn) / denom
        if kind == "log1p_zscore":
            x = math.log1p(max(x, 0.0))
            mean = math.log1p(max(mean, 0.0)) if mean > 0 else 0.0
            std = math.log1p(max(std, 1.0))  # use log1p of std as stable scale
            return (x - mean) / max(std, 1e-6)
        # zscore
        return (x - mean) / std


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _persistent_state_for_step(
    persistent_states: list[dict[str, Any]],
    t: int,
) -> dict[str, Any]:
    """Return the persistent state BEFORE step ``t``; fall back to defaults."""
    if 0 <= t < len(persistent_states):
        return persistent_states[t]
    return default_state()


def _build_global_categoricals(
    step: dict[str, Any],
    pstate: dict[str, Any],
    vocab: VocabLookup,
) -> dict[str, np.int32]:
    deck = pstate.get("deck") or {}
    out = {
        "page_id": np.int32(vocab.encode("page", step.get("page_name"))),
        "source_kind_id": np.int32(vocab.encode("source_kind", step.get("source_kind"))),
        "action_subtype_id": np.int32(vocab.encode("action_subtype", step.get("action_subtype"))),
        "deck_class_id": np.int32(vocab.encode("deck_class_id", deck.get("class_id"))),
        "stake_class_id": np.int32(vocab.encode("stake_class_id", pstate.get("stake"))),
        "last_tarot_planet_class_id": np.int32(
            vocab.encode("last_tarot_planet_class_id", pstate.get("last_tarot_planet"))
        ),
        "ante_boss_blind_class_id": np.int32(
            vocab.encode("ante_boss_blind_class_id", pstate.get("ante_boss_blind"))
        ),
        "small_status_id": np.int32(vocab.encode("small_status", pstate.get("small_status"))),
        "big_status_id": np.int32(vocab.encode("big_status", pstate.get("big_status"))),
    }
    return out


def _build_global_numeric(
    step: dict[str, Any],
    pstate: dict[str, Any],
    norm: Normalizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ocr = dict(step.get("state") or {})
    # Granularized JSON may still carry legacy keys; they are not model inputs.
    ocr.pop("hand_and_level_raw", None)
    ocr.pop("ocr_extra", None)
    ocr_numeric = np.zeros(N_OCR, dtype=np.float32)
    ocr_valid = np.zeros(N_OCR, dtype=bool)
    for i, key in enumerate(OCR_NUMERIC_KEYS):
        v = ocr.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ocr_numeric[i] = norm.transform(f"ocr.{key}", v)
            ocr_valid[i] = True

    state_numeric = np.zeros(N_STATE_NUM, dtype=np.float32)
    for i, key in enumerate(STATE_NUMERIC_KEYS):
        v = pstate.get(key, 0)
        state_numeric[i] = norm.transform(f"state.{key}", v)

    return ocr_numeric, state_numeric, ocr_valid


def _step_ocr_ante(step: dict[str, Any]) -> int | float | None:
    """Return the raw OCR ``state.ante`` value, or None if absent/invalid."""
    v = (step.get("state") or {}).get("ante")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    return None


def _step_passes_max_ante(step: dict[str, Any], max_ante: int | None) -> bool:
    """True when ``max_ante`` is unset or encoder-step OCR ante is <= limit."""
    if max_ante is None:
        return True
    ante = _step_ocr_ante(step)
    if ante is None:
        return False
    return ante <= max_ante


def _build_flags(pstate: dict[str, Any]) -> np.ndarray:
    flags = np.zeros(N_FLAGS, dtype=bool)
    deck = pstate.get("deck") or {}
    deck_mods = pstate.get("deck_modifiers") or {}
    idx = 0
    for k in PERSISTENT_BOOL_FIELDS:
        flags[idx] = bool(pstate.get(k, False))
        idx += 1
    for k in DECK_FLAG_FIELDS:
        flags[idx] = bool(deck.get(k, False))
        idx += 1
    for k in DECK_MODIFIER_FIELDS:
        flags[idx] = bool(deck_mods.get(k, False))
        idx += 1
    return flags


def _build_per_hand(pstate: dict[str, Any], norm: Normalizer, hand_names: list[str]) -> dict[str, np.ndarray]:
    hands = pstate.get("hands") or {}
    n = len(hand_names)
    hand_levels = np.zeros(n, dtype=np.float32)
    hand_played = np.zeros(n, dtype=np.float32)
    hand_played_this_round = np.zeros(n, dtype=np.float32)
    for i, name in enumerate(hand_names):
        entry = hands.get(name) or {}
        hand_levels[i] = norm.transform("hand_level", entry.get("level", 1))
        hand_played[i] = norm.transform("hand_played", entry.get("played", 0))
        hand_played_this_round[i] = norm.transform(
            "hand_played_this_round", entry.get("played_this_round", 0)
        )
    return {
        "hand_levels": hand_levels,
        "hand_played": hand_played,
        "hand_played_this_round": hand_played_this_round,
    }


def _build_voucher_boss_multihot(
    pstate: dict[str, Any],
    voucher_list: list[int],
    boss_list: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    voucher_index = {cid: i for i, cid in enumerate(voucher_list)}
    boss_index = {cid: i for i, cid in enumerate(boss_list)}
    vouchers_redeemed = np.zeros(len(voucher_list), dtype=bool)
    bosses_used = np.zeros(len(boss_list), dtype=bool)
    for cid in pstate.get("vouchers_redeemed") or []:
        i = voucher_index.get(int(cid))
        if i is not None:
            vouchers_redeemed[i] = True
    for cid in pstate.get("bosses_used") or []:
        i = boss_index.get(int(cid))
        if i is not None:
            bosses_used[i] = True
    return vouchers_redeemed, bosses_used


def _build_tracked_deck(
    pstate: dict[str, Any],
    vocab: VocabLookup,
    cap: int,
) -> dict[str, np.ndarray]:
    deck = pstate.get("tracked_deck_cards") or []
    n = min(len(deck), cap)
    arrs = {
        "deck_card_class_id": np.zeros(cap, dtype=np.int32),
        "deck_card_modifier_id": np.zeros(cap, dtype=np.int32),
        "deck_card_edition_id": np.zeros(cap, dtype=np.int32),
        "deck_card_seal_id": np.zeros(cap, dtype=np.int32),
        "deck_card_rank_id": np.zeros(cap, dtype=np.int32),
        "deck_card_suit_id": np.zeros(cap, dtype=np.int32),
        "deck_card_mask": np.zeros(cap, dtype=bool),
    }
    for i, c in enumerate(deck[:cap]):
        arrs["deck_card_class_id"][i] = vocab.encode("class_id", c.get("class_id"))
        arrs["deck_card_modifier_id"][i] = vocab.encode("modifier", c.get("modifier"))
        arrs["deck_card_edition_id"][i] = vocab.encode("edition", c.get("edition"))
        arrs["deck_card_seal_id"][i] = vocab.encode("seal", c.get("seal"))
        cm = c.get("card") or {}
        arrs["deck_card_rank_id"][i] = vocab.encode("rank_index", cm.get("rank_index"))
        arrs["deck_card_suit_id"][i] = vocab.encode("suit_index", cm.get("suit_index"))
        arrs["deck_card_mask"][i] = True
    return arrs


def _build_objects(
    step: dict[str, Any],
    vocab: VocabLookup,
    cap: int,
) -> dict[str, np.ndarray]:
    """Encode ``step.objects`` (already canonical post-granularize 3.0.0)."""
    objs = list(step.get("objects") or [])
    n = min(len(objs), cap)
    arrs = {
        "object_class_id": np.zeros(cap, dtype=np.int32),
        "object_object_type_id": np.zeros(cap, dtype=np.int32),
        "object_zone_id": np.zeros(cap, dtype=np.int32),
        "object_position": np.zeros(cap, dtype=np.int32),
        "object_modifier_id": np.zeros(cap, dtype=np.int32),
        "object_edition_id": np.zeros(cap, dtype=np.int32),
        "object_seal_id": np.zeros(cap, dtype=np.int32),
        "object_rank_id": np.zeros(cap, dtype=np.int32),
        "object_suit_id": np.zeros(cap, dtype=np.int32),
        "object_is_debuffed": np.zeros(cap, dtype=bool),
        "object_sticker_rental": np.zeros(cap, dtype=bool),
        "object_sticker_perishable": np.zeros(cap, dtype=bool),
        "object_sticker_eternal": np.zeros(cap, dtype=bool),
        "object_mask": np.zeros(cap, dtype=bool),
    }
    for i, o in enumerate(objs[:cap]):
        arrs["object_class_id"][i] = vocab.encode("class_id", o.get("class_id"))
        arrs["object_object_type_id"][i] = vocab.encode("object_type", o.get("object_type"))
        arrs["object_zone_id"][i] = vocab.encode("zone", o.get("zone"))
        pos = o.get("position_in_zone")
        arrs["object_position"][i] = int(pos) if isinstance(pos, int) else 0
        arrs["object_modifier_id"][i] = vocab.encode("modifier", o.get("modifier"))
        arrs["object_edition_id"][i] = vocab.encode("edition", o.get("edition"))
        arrs["object_seal_id"][i] = vocab.encode("seal", o.get("seal"))
        cm = o.get("card") or {}
        arrs["object_rank_id"][i] = vocab.encode("rank_index", cm.get("rank_index"))
        arrs["object_suit_id"][i] = vocab.encode("suit_index", cm.get("suit_index"))
        arrs["object_is_debuffed"][i] = bool(o.get("is_debuffed", False))
        for st in o.get("stickers") or []:
            if st == "rental":
                arrs["object_sticker_rental"][i] = True
            elif st == "perishable":
                arrs["object_sticker_perishable"][i] = True
            elif st == "eternal":
                arrs["object_sticker_eternal"][i] = True
        arrs["object_mask"][i] = True
    return arrs


# ---------------------------------------------------------------------------
# Branched-policy capacities + supervised targets
# ---------------------------------------------------------------------------


def derive_branched_caps(
    action_map: dict[str, Any],
    family_map: dict[str, Any],
) -> dict[str, int]:
    """Compute fixed buffer sizes for the branched policy heads.

    All values are derived from ``action_map`` so the tensorizer, model,
    and live server stay in sync via a single source of truth.
    """
    family_sizes = action_map.get("family_sizes", {})
    family_to_flat_size = family_map.get("family_to_flat_size", {})

    item_zone_sizes: list[int] = []
    for fam in family_map["family_order"]:
        shape = family_map["decoder_shapes"].get(fam)
        if shape in ("single_ptr", "chained_cards"):
            item_zone_sizes.append(int(family_to_flat_size.get(fam, 0)))
    max_item_zone = max(item_zone_sizes) if item_zone_sizes else 0
    max_card_zone = max(
        int(family_sizes.get("SelectCard_CurrentHand", 0)),
        int(family_sizes.get("SelectCard_TarotSpectralHand", 0)),
    )
    max_joker_slots = int(action_map.get("max_values", {}).get("MAX_JOKER_SLOTS", 10))
    return {
        "MAX_ITEM_ZONE_SIZE": int(max_item_zone),
        "MAX_CARD_ZONE_SIZE": int(max_card_zone),
        "MAX_JOKER_SLOTS": int(max_joker_slots),
        "MAX_CARDS_PER_DECISION": int(MAX_CARDS_PER_DECISION),
    }


def _empty_branched_targets(caps: dict[str, int]) -> dict[str, np.ndarray]:
    """Return default (all -1 / 0) supervised target channels.

    Used both for the live encoder (no supervision) and as the starting
    point for ``_fill_branched_targets``.
    """
    max_cards = int(caps["MAX_CARDS_PER_DECISION"])
    out: dict[str, np.ndarray] = {
        "family_id": np.int32(-1),
        "num_cards": np.int32(-1),
        "item_ptr_local": np.int32(-1),
        "item_ptr_slot": np.int32(-1),
        "card_ptr_local_seq": np.full(max_cards, -1, dtype=np.int32),
        "card_ptr_slot_seq": np.full(max_cards, -1, dtype=np.int32),
        "swap_i_local": np.int32(-1),
        "swap_j_local": np.int32(-1),
        "swap_i_slot": np.int32(-1),
        "swap_j_slot": np.int32(-1),
    }
    return out


def _empty_branched_masks(
    n_families: int,
    caps: dict[str, int],
) -> dict[str, np.ndarray]:
    """Return all-False mask channels (no legal action)."""
    return {
        "family_mask": np.zeros(int(n_families), dtype=bool),
        "item_pointer_mask": np.zeros(int(caps["MAX_ITEM_ZONE_SIZE"]), dtype=bool),
        "card_pointer_mask": np.zeros(int(caps["MAX_CARD_ZONE_SIZE"]), dtype=bool),
        "swap_joker_mask": np.zeros(int(caps["MAX_JOKER_SLOTS"]), dtype=bool),
    }


def _selected_position(step: dict[str, Any]) -> int | None:
    """Return the ORIGINAL position_in_zone of the selected object (or None)."""
    sel = step.get("selected_object") or {}
    obj = sel.get("object") or {}
    pos = obj.get("position_in_zone")
    return int(pos) if isinstance(pos, int) else None


def _selected_slot_id(step: dict[str, Any]) -> int | None:
    """Return the slot_id of the selected object (or None)."""
    sel = step.get("selected_object") or {}
    obj = sel.get("object") or {}
    sid = obj.get("slot_id")
    return int(sid) if isinstance(sid, int) else None


def _resolve_joker_slot(
    step: dict[str, Any], position_in_zone: int
) -> int | None:
    """Find the slot_id of the joker at ``position_in_zone`` in CurrentJokers."""
    for obj in step.get("objects") or []:
        if (
            obj.get("zone") == "CurrentJokers"
            and obj.get("position_in_zone") == position_in_zone
        ):
            sid = obj.get("slot_id")
            return int(sid) if isinstance(sid, int) else None
    return None


def _fill_branched_targets(
    ss: SuperStep,
    family_map: dict[str, Any],
    caps: dict[str, int],
) -> dict[str, np.ndarray]:
    """Compute supervised target channels for one super-step."""
    targets = _empty_branched_targets(caps)
    commit_step = ss.commit_step
    family_name = family_name_for_step(commit_step)
    fid = family_id_for_step(commit_step, family_map)
    if fid < 0:
        return targets
    targets["family_id"] = np.int32(fid)

    shape = family_map["decoder_shapes"].get(family_name)
    max_cards = int(caps["MAX_CARDS_PER_DECISION"])

    if shape == "no_args":
        # Family head fully determines the action; no further targets.
        targets["num_cards"] = np.int32(0)
        return targets

    if shape == "card_seq":
        n = min(len(ss.select_steps), max_cards)
        targets["num_cards"] = np.int32(n)
        for i, s in enumerate(ss.select_steps[:n]):
            pos = _selected_position(s)
            sid = _selected_slot_id(s)
            targets["card_ptr_local_seq"][i] = (
                np.int32(pos) if pos is not None else np.int32(-1)
            )
            targets["card_ptr_slot_seq"][i] = (
                np.int32(sid) if sid is not None else np.int32(-1)
            )
        return targets

    if shape == "single_ptr":
        # Commit step carries target_position + selected_object.
        pos = commit_step.get("target_position")
        if isinstance(pos, int):
            targets["item_ptr_local"] = np.int32(pos)
        sid = _selected_slot_id(commit_step)
        if sid is not None:
            targets["item_ptr_slot"] = np.int32(sid)
        targets["num_cards"] = np.int32(0)
        return targets

    if shape == "chained_cards":
        pos = commit_step.get("target_position")
        if isinstance(pos, int):
            targets["item_ptr_local"] = np.int32(pos)
        sid = _selected_slot_id(commit_step)
        if sid is not None:
            targets["item_ptr_slot"] = np.int32(sid)
        n = min(len(ss.select_steps), max_cards)
        targets["num_cards"] = np.int32(n)
        for i, s in enumerate(ss.select_steps[:n]):
            cpos = _selected_position(s)
            csid = _selected_slot_id(s)
            targets["card_ptr_local_seq"][i] = (
                np.int32(cpos) if cpos is not None else np.int32(-1)
            )
            targets["card_ptr_slot_seq"][i] = (
                np.int32(csid) if csid is not None else np.int32(-1)
            )
        return targets

    if shape == "joker_pair":
        pair = commit_step.get("swap_pair")
        if isinstance(pair, list) and len(pair) == 2:
            i_pos, j_pos = int(pair[0]), int(pair[1])
            targets["swap_i_local"] = np.int32(i_pos)
            targets["swap_j_local"] = np.int32(j_pos)
            i_slot = _resolve_joker_slot(commit_step, i_pos)
            j_slot = _resolve_joker_slot(commit_step, j_pos)
            if i_slot is not None:
                targets["swap_i_slot"] = np.int32(i_slot)
            if j_slot is not None:
                targets["swap_j_slot"] = np.int32(j_slot)
        targets["num_cards"] = np.int32(0)
        return targets

    # 'reserved' or unknown shape: leave all defaults.
    return targets


def _build_branched_masks(
    encoder_step: dict[str, Any],
    action_map: dict[str, Any],
    family_map: dict[str, Any],
    caps: dict[str, int],
    family_name_for_pointer: str | None,
    flat_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute family + pointer masks from the encoder-input step.

    ``family_name_for_pointer`` is the family the pointer masks should
    correspond to (commit family for training, None for live encoders
    where the family will be decoded). When None the pointer masks are
    zero-filled; callers can recompute after family selection.
    """
    out = _empty_branched_masks(int(family_map["n_families"]), caps)
    out["family_mask"] = build_family_mask(
        encoder_step, action_map, family_map, flat_mask=flat_mask
    )
    if family_name_for_pointer is None:
        return out
    out["item_pointer_mask"] = build_item_pointer_mask(
        family_name_for_pointer,
        encoder_step,
        action_map,
        max_size=int(caps["MAX_ITEM_ZONE_SIZE"]),
        flat_mask=flat_mask,
    )
    out["card_pointer_mask"] = build_card_pointer_mask(
        family_name_for_pointer,
        encoder_step,
        action_map,
        max_size=int(caps["MAX_CARD_ZONE_SIZE"]),
        flat_mask=flat_mask,
    )
    if family_name_for_pointer == "SWAP":
        out["swap_joker_mask"] = build_swap_joker_mask(
            encoder_step,
            max_joker_slots=int(caps["MAX_JOKER_SLOTS"]),
        )
    return out


# ---------------------------------------------------------------------------
# Top-level per-step entry point
# ---------------------------------------------------------------------------

def tensorize_step(
    step: dict[str, Any],
    persistent_state: dict[str, Any],
    action_map: dict[str, Any],
    vocab: VocabLookup,
    norm: Normalizer,
    feature_config: dict[str, Any],
    history_steps: list[dict[str, Any]] | None = None,
    *,
    family_map: dict[str, Any] | None = None,
    branched_caps: dict[str, int] | None = None,
    family_name_for_pointer: str | None = None,
) -> dict[str, np.ndarray]:
    """Encode a single granularized step + its persistent_state-BEFORE snapshot.

    When ``family_map`` is provided (recommended), the record also includes
    branched-policy mask channels and placeholder (-1) target channels.
    The legacy v1 ``target_action_id`` / ``action_mask`` channels are
    always emitted for back-compat.

    ``family_name_for_pointer`` is used to populate ``item_pointer_mask``
    / ``card_pointer_mask`` for the indicated family. Pass ``None`` for
    live encoders that haven't predicted a family yet (pointer masks
    stay zero-filled; callers can rebuild after family selection).
    """
    # Drop any legacy fields that aren't model inputs but might tag along
    # in older snapshots.
    state = dict(step.get("state") or {})
    state.pop("hand_and_level_raw", None)
    state.pop("ocr_extra", None)
    step = dict(step)
    step["state"] = state

    voucher_list = list(feature_config["VOUCHER_CLASS_LIST"])
    boss_list = list(feature_config["BOSS_CLASS_LIST"])
    cap_objects = int(feature_config["MAX_OBJECTS_PER_STEP"])
    cap_deck = int(feature_config["TRACKED_DECK_CAP"])
    hand_names = list(feature_config["POKER_HANDS"])

    out: dict[str, np.ndarray] = {}
    out.update(_build_global_categoricals(step, persistent_state, vocab))
    ocr_num, state_num, ocr_valid = _build_global_numeric(step, persistent_state, norm)
    out["ocr_numeric"] = ocr_num
    out["state_numeric"] = state_num
    out["ocr_valid"] = ocr_valid
    out["flags"] = _build_flags(persistent_state)
    out.update(_build_per_hand(persistent_state, norm, hand_names))
    out["supplement_features"] = compute_supplement_features(step, persistent_state)

    vouchers_redeemed, bosses_used = _build_voucher_boss_multihot(
        persistent_state, voucher_list, boss_list
    )
    out["vouchers_redeemed"] = vouchers_redeemed
    out["bosses_used"] = bosses_used

    out.update(_build_tracked_deck(persistent_state, vocab, cap_deck))
    out.update(_build_objects(step, vocab, cap_objects))
    out.update(
        build_history_tensors(
            history_steps or [],
            action_map=action_map,
            vocab=vocab,
            norm=norm,
            feature_config=feature_config,
        )
    )

    mask = build_action_mask(step, action_map)
    out["action_mask"] = mask.astype(bool)
    base_action = parse_base_action(step.get("action") or "")
    if base_action == "StartNewRun":
        # StartNewRun is a bootstrap artifact, not a policy decision we want
        # the model to learn. Mark unresolved so the dataset excludes it.
        out["target_action_id"] = np.int32(-1)
    else:
        try:
            aid = resolve_label(step, action_map.get("label_to_index", {}))
        except LabelResolutionError:
            # Upstream data anomalies (e.g. bare "BuyAndUseShopConsumable"
            # without an index suffix) are recorded as -1 so the training
            # loop can mask them out instead of crashing the entire run.
            aid = -1
        if aid is not None and aid >= 0 and not bool(mask[aid]):
            aid = -1
        out["target_action_id"] = np.int32(aid if aid is not None else -1)

    if family_map is not None:
        caps = branched_caps or derive_branched_caps(action_map, family_map)
        out.update(_empty_branched_targets(caps))
        out.update(
            _build_branched_masks(
                step,
                action_map,
                family_map,
                caps,
                family_name_for_pointer=family_name_for_pointer,
                flat_mask=mask,
            )
        )

    return out


def tensorize_super_step(
    ss: SuperStep,
    persistent_state: dict[str, Any],
    action_map: dict[str, Any],
    family_map: dict[str, Any],
    vocab: VocabLookup,
    norm: Normalizer,
    feature_config: dict[str, Any],
    branched_caps: dict[str, int],
    history_steps: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Tensorize one parent-level super-step.

    Encoder features + masks come from ``ss.encoder_step`` (the state at
    the start of the parent decision). Supervised targets — both the v1
    flat ``target_action_id`` and the v2 branched channels — come from
    ``ss.commit_step`` and the ordered list of ``ss.select_steps``.
    """
    family_name = family_name_for_step(ss.commit_step)
    rec = tensorize_step(
        ss.encoder_step,
        persistent_state,
        action_map,
        vocab,
        norm,
        feature_config,
        history_steps=history_steps,
        family_map=family_map,
        branched_caps=branched_caps,
        family_name_for_pointer=family_name,
    )

    # Override v1 flat target with the COMMIT step's action (e.g. PlayHand)
    # rather than the encoder step's action (which may be SelectCard_*).
    commit_action = ss.commit_step.get("action") or ""
    base = parse_base_action(commit_action)
    if base == "StartNewRun":
        rec["target_action_id"] = np.int32(-1)
    else:
        try:
            aid = resolve_label(ss.commit_step, action_map.get("label_to_index", {}))
        except LabelResolutionError:
            aid = -1
        if aid is not None and aid >= 0 and bool(rec["action_mask"][aid]):
            rec["target_action_id"] = np.int32(aid)
        else:
            rec["target_action_id"] = np.int32(-1)

    # Override empty branched targets with the supervised values.
    rec.update(_fill_branched_targets(ss, family_map, branched_caps))
    return rec


# ---------------------------------------------------------------------------
# Per-run materialization
# ---------------------------------------------------------------------------

def _stack_step_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-step dicts into a per-run dict of (n_steps, ...) arrays."""
    if not records:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in records[0]:
        out[key] = np.stack([r[key] for r in records], axis=0)
    return out


def _process_run(
    run: dict[str, Any],
    persistent: dict[str, Any] | None,
    action_map: dict[str, Any],
    vocab: VocabLookup,
    norm: Normalizer,
    feature_config: dict[str, Any],
    stats: collections.Counter,
    *,
    family_map: dict[str, Any],
    branched_caps: dict[str, int],
    max_ante: int | None = None,
) -> dict[str, np.ndarray]:
    events = run.get("events") or []
    pstates = (persistent or {}).get("states") or []

    # Re-derive persistent state if the file is missing; this keeps the
    # tensorizer self-contained even if compute_persistent_state has not run.
    if not pstates:
        pstates = []
        s = default_state()
        for ev in events:
            pstates.append(s)
            s = apply_step(s, ev)

    records: list[dict[str, np.ndarray]] = []
    history_records: list[dict[str, Any]] = []

    for ss in iter_super_steps(events):
        if not _step_passes_max_ante(ss.encoder_step, max_ante):
            if max_ante is not None:
                ante = _step_ocr_ante(ss.encoder_step)
                if ante is None:
                    stats[("filtered", "max_ante_missing")] += 1
                else:
                    stats[("filtered", "max_ante_excluded")] += 1
            continue

        t = ss.encoder_step_idx
        ps = _persistent_state_for_step(pstates, t)
        rec = tensorize_super_step(
            ss,
            ps,
            action_map,
            family_map,
            vocab,
            norm,
            feature_config,
            branched_caps,
            history_steps=history_records,
        )
        records.append(rec)

        if int(rec["target_action_id"]) < 0:
            stats[("target", "unresolved")] += 1
        else:
            stats[("target", "resolved")] += 1
        if int(rec["family_id"]) < 0:
            stats[("family", "unresolved")] += 1
        else:
            stats[("family", "resolved")] += 1
            fam_name = family_map["id_to_family"][int(rec["family_id"])]
            stats[("family_name", fam_name)] += 1

        stats[("event_base", parse_base_action(ss.commit_step.get("action") or ""))] += 1
        stats[("super_step_kind", ss.kind)] += 1
        if ss.kind == "regular" and len(ss.select_steps) > 0:
            stats[("num_selects", str(len(ss.select_steps)))] += 1

        # Record the executed parent action in history. Skip StartNewRun.
        # For decomposed parents we synthesize a record that carries the
        # parent action label but lifts PendingCards / pending_cards from
        # the LAST select step so history captures which cards were
        # actually played / used.
        if (ss.commit_step.get("action") or "") != "StartNewRun":
            history_records.append(make_history_record(ss))

    return _stack_step_records(records)


def _iter_run_pairs(granularized_root: Path, persistent_root: Path):
    for partition in sorted(granularized_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        ppartition = persistent_root / partition.name
        for grun in sorted(partition.glob("run_*.json")):
            prun = ppartition / grun.name
            yield video_id, grun, prun if prun.exists() else None


def _init_worker(
    vocab_payload: dict[str, Any],
    norm_payload: dict[str, Any],
    feature_config: dict[str, Any],
    action_config: dict[str, Any],
) -> None:
    """Initialize per-process immutable artifacts once."""
    global _WORKER_ACTION_MAP, _WORKER_VOCAB, _WORKER_NORM, _WORKER_FEATURE_CONFIG
    global _WORKER_FAMILY_MAP, _WORKER_BRANCHED_CAPS
    _WORKER_VOCAB = VocabLookup(vocab_payload)
    _WORKER_NORM = Normalizer(norm_payload)
    _WORKER_FEATURE_CONFIG = feature_config
    _WORKER_ACTION_MAP = compute_action_map(action_config)
    _WORKER_FAMILY_MAP = compute_family_map(_WORKER_ACTION_MAP)
    _WORKER_BRANCHED_CAPS = derive_branched_caps(_WORKER_ACTION_MAP, _WORKER_FAMILY_MAP)


def _process_run_file_job(
    job: tuple[str, str, str | None, str, int | None],
) -> dict[str, Any]:
    """Worker entry point: process one run file and write its shard."""
    if (
        _WORKER_ACTION_MAP is None
        or _WORKER_VOCAB is None
        or _WORKER_NORM is None
        or _WORKER_FEATURE_CONFIG is None
        or _WORKER_FAMILY_MAP is None
        or _WORKER_BRANCHED_CAPS is None
    ):
        raise RuntimeError("tensorize worker was not initialized")

    video_id, gpath_s, ppath_s, out_s, max_ante = job
    gpath = Path(gpath_s)
    ppath = Path(ppath_s) if ppath_s else None
    out_root = Path(out_s)

    if not gpath.exists():
        return {
            "video_id": video_id,
            "run_name": gpath.name,
            "dst_file": None,
            "n_steps": 0,
            "bytes_written": 0,
            "stats": {("run", "missing_granularized"): 1},
            "warning": f"missing granularized run: {gpath_s}",
        }

    run = json.loads(gpath.read_text(encoding="utf-8"))
    psnap = json.loads(ppath.read_text(encoding="utf-8")) if ppath else None

    stats: collections.Counter = collections.Counter()
    record = _process_run(
        run,
        psnap,
        _WORKER_ACTION_MAP,
        _WORKER_VOCAB,
        _WORKER_NORM,
        _WORKER_FEATURE_CONFIG,
        stats,
        family_map=_WORKER_FAMILY_MAP,
        branched_caps=_WORKER_BRANCHED_CAPS,
        max_ante=max_ante,
    )
    if not record:
        stats[("run", "empty")] += 1
        return {
            "video_id": video_id,
            "run_name": gpath.name,
            "dst_file": None,
            "n_steps": 0,
            "bytes_written": 0,
            "stats": dict(stats),
        }

    dst_dir = out_root / f"video_id={video_id}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / gpath.with_suffix(".npz").name
    np.savez_compressed(dst_file, **record)
    n_steps = next(iter(record.values())).shape[0]
    return {
        "video_id": video_id,
        "run_name": gpath.name,
        "dst_file": dst_file.as_posix(),
        "n_steps": int(n_steps),
        "bytes_written": int(dst_file.stat().st_size),
        "stats": dict(stats),
    }


def _merge_worker_stats(
    stats: collections.Counter,
    worker_stats: dict[Any, int],
) -> None:
    for key, value in worker_stats.items():
        # Tuple keys survive ProcessPoolExecutor on Windows, but JSON-style
        # callers/tests may hand back list keys. Normalize defensively.
        if isinstance(key, list):
            key = tuple(key)
        stats[key] += int(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--persistent", type=Path, default=Path("data/persistent_state"))
    ap.add_argument("--out", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--vocab", type=Path, default=Path("artifacts/vocab.json"))
    ap.add_argument("--normalization", type=Path, default=Path("artifacts/normalization.json"))
    ap.add_argument("--feature-config", type=Path, default=Path("artifacts/feature_config.json"))
    ap.add_argument("--action-config", type=Path, default=Path("data/action_space_config.json"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/tensorizer_report.json"))
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel run workers. 0 = auto (min(cpu_count, 8)); 1 = serial.",
    )
    ap.add_argument(
        "--max-ante",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep only super-steps whose encoder snapshot has OCR state.ante "
            "<= N. Steps with missing ante are skipped. Omit to tensorize all "
            "super-steps."
        ),
    )
    args = ap.parse_args()
    if args.max_ante is not None and args.max_ante < 0:
        ap.error("--max-ante must be >= 0")

    vocab_payload = json.loads(args.vocab.read_text(encoding="utf-8"))
    norm_payload = json.loads(args.normalization.read_text(encoding="utf-8"))
    feature_config = json.loads(args.feature_config.read_text(encoding="utf-8"))
    action_config = json.loads(args.action_config.read_text(encoding="utf-8"))
    vocab = VocabLookup(vocab_payload)
    norm = Normalizer(norm_payload)
    action_map = compute_action_map(action_config)
    family_map = compute_family_map(action_map)
    branched_caps = derive_branched_caps(action_map, family_map)

    args.out.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    runs_processed = 0
    steps_processed = 0
    bytes_written = 0
    start = time.time()
    current_video = None

    print(f"N_ACTIONS = {action_map['n_actions']}")
    print(f"action_map_version = {action_map['action_map_version']}")
    print(f"family_map_version = {family_map['family_map_version']}")
    print(f"n_families = {family_map['n_families']}")
    print(f"branched_caps = {branched_caps}")
    print(f"MAX_OBJECTS_PER_STEP = {feature_config['MAX_OBJECTS_PER_STEP']}")
    print(f"TRACKED_DECK_CAP = {feature_config['TRACKED_DECK_CAP']}")
    h_steps, h_objects = history_caps(feature_config)
    print(f"HISTORY_STEPS = {h_steps}")
    print(f"HISTORY_OBJECTS_PER_STEP = {h_objects}")
    workers = args.workers
    if workers <= 0:
        workers = min(os.cpu_count() or 1, 8)
    workers = max(1, workers)
    print(f"workers = {workers}")
    if args.max_ante is not None:
        print(f"max_ante = {args.max_ante} (OCR state.ante <= {args.max_ante})")
    print()

    jobs = [
        (
            video_id,
            gpath.as_posix(),
            ppath.as_posix() if ppath else None,
            args.out.as_posix(),
            args.max_ante,
        )
        for video_id, gpath, ppath in _iter_run_pairs(args.src, args.persistent)
    ]

    if workers == 1:
        _init_worker(vocab_payload, norm_payload, feature_config, action_config)
        result_iter = map(_process_run_file_job, jobs)
        for result in result_iter:
            _merge_worker_stats(stats, result["stats"])
            if result.get("warning"):
                print(f"WARNING {result['warning']}", flush=True)
            if result["n_steps"] <= 0:
                continue
            if result["video_id"] != current_video:
                print(f"video_id={result['video_id']}", flush=True)
                current_video = result["video_id"]
            runs_processed += 1
            steps_processed += int(result["n_steps"])
            bytes_written += int(result["bytes_written"])
            print(
                f"  {result['run_name']} -> {Path(result['dst_file']).name}  "
                f"({result['n_steps']} steps, {result['bytes_written']/1024:.1f} KiB)",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(vocab_payload, norm_payload, feature_config, action_config),
        ) as ex:
            future_to_job = {
                ex.submit(_process_run_file_job, job): job for job in jobs
            }
            completed = 0
            for fut in concurrent.futures.as_completed(future_to_job):
                completed += 1
                try:
                    result = fut.result()
                except Exception as e:
                    _, gpath_s, _, _ = future_to_job[fut]
                    stats[("run", "failed")] += 1
                    print(
                        f"[{completed}/{len(jobs)}] FAILED {gpath_s}: {e!r}",
                        flush=True,
                    )
                    continue
                _merge_worker_stats(stats, result["stats"])
                if result.get("warning"):
                    print(f"[{completed}/{len(jobs)}] WARNING {result['warning']}", flush=True)
                if result["n_steps"] <= 0:
                    continue
                runs_processed += 1
                steps_processed += int(result["n_steps"])
                bytes_written += int(result["bytes_written"])
                print(
                    f"[{completed}/{len(jobs)}] video_id={result['video_id']} "
                    f"{result['run_name']} -> {Path(result['dst_file']).name}  "
                    f"({result['n_steps']} steps, {result['bytes_written']/1024:.1f} KiB)",
                    flush=True,
                )

    elapsed = time.time() - start
    print()
    print(f"runs processed:  {runs_processed}")
    print(f"steps processed: {steps_processed}")
    print(f"bytes written:   {bytes_written/1024/1024:.2f} MiB")
    print(f"elapsed:         {elapsed:.2f}s")
    print()
    print("--- target resolution ---")
    print(f"  resolved:   {stats.get(('target', 'resolved'), 0)}")
    print(f"  unresolved: {stats.get(('target', 'unresolved'), 0)}")
    print()
    print("--- family resolution ---")
    print(f"  resolved:   {stats.get(('family', 'resolved'), 0)}")
    print(f"  unresolved: {stats.get(('family', 'unresolved'), 0)}")
    print()
    print("--- family counts ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 != "family_name":
            continue
        print(f"  {k2:55s} {n}")
    print()
    print("--- super-step kinds ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 != "super_step_kind":
            continue
        print(f"  {k2:30s} {n}")
    print()
    print("--- num_selects per regular super-step ---")
    for (k1, k2), n in sorted(stats.items(), key=lambda kv: int(kv[0][1]) if kv[0][0] == "num_selects" else 0):
        if k1 != "num_selects":
            continue
        print(f"  {k2:>3s} cards: {n}")
    print()
    print("--- event base counts ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 != "event_base":
            continue
        print(f"  {k2:30s} {n}")
    print()
    print("--- filtered rows ---")
    for (k1, k2), n in sorted(stats.items()):
        if k1 != "filtered":
            continue
        print(f"  {k2:40s} {n}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TENSORIZE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src": args.src.as_posix(),
        "persistent": args.persistent.as_posix(),
        "out": args.out.as_posix(),
        "workers": workers,
        "max_ante": args.max_ante,
        "n_actions": action_map["n_actions"],
        "action_map_version": action_map["action_map_version"],
        "family_map_version": family_map["family_map_version"],
        "n_families": family_map["n_families"],
        "branched_caps": branched_caps,
        "feature_config": {
            "MAX_OBJECTS_PER_STEP": feature_config["MAX_OBJECTS_PER_STEP"],
            "TRACKED_DECK_CAP": feature_config["TRACKED_DECK_CAP"],
            "N_HANDS": feature_config["N_HANDS"],
            "N_VOUCHERS": feature_config["N_VOUCHERS"],
            "N_BOSSES": feature_config["N_BOSSES"],
            "N_FLAGS": N_FLAGS,
            "N_OCR": N_OCR,
            "N_STATE_NUM": N_STATE_NUM,
            "HISTORY_STEPS": h_steps,
            "HISTORY_OBJECTS_PER_STEP": h_objects,
        },
        "runs_processed": runs_processed,
        "steps_processed": steps_processed,
        "bytes_written": bytes_written,
        "elapsed_seconds": elapsed,
        "target_resolved": stats.get(("target", "resolved"), 0),
        "target_unresolved": stats.get(("target", "unresolved"), 0),
        "family_resolved": stats.get(("family", "resolved"), 0),
        "family_unresolved": stats.get(("family", "unresolved"), 0),
        "family_counts": {
            k2: n for (k1, k2), n in stats.items() if k1 == "family_name"
        },
        "super_step_kinds": {
            k2: n for (k1, k2), n in stats.items() if k1 == "super_step_kind"
        },
        "num_selects_histogram": {
            k2: n for (k1, k2), n in stats.items() if k1 == "num_selects"
        },
        "event_base_counts": {
            k2: n for (k1, k2), n in stats.items() if k1 == "event_base"
        },
        "filtered_counts": {
            k2: n for (k1, k2), n in stats.items() if k1 == "filtered"
        },
    }
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote report -> {args.report.as_posix()}")


if __name__ == "__main__":
    main()
