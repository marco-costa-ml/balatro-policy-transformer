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
       [--out data/tensorized] [--report artifacts/tensorizer_report.json]``

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

    ACTION TARGET (bool + int32)
      action_mask                           (N_ACTIONS,)
      target_action_id                      ()    -1 if label could not be resolved
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from action_map import compute_action_map
from label_resolver import LabelResolutionError, resolve_label
from mask_builder import build_action_mask
from state_reducer import apply_step, default_state, parse_base_action


TENSORIZE_SCHEMA_VERSION = "2.0.0"

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
# Top-level per-step entry point
# ---------------------------------------------------------------------------

def tensorize_step(
    step: dict[str, Any],
    persistent_state: dict[str, Any],
    action_map: dict[str, Any],
    vocab: VocabLookup,
    norm: Normalizer,
    feature_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Encode a single granularized step + its persistent_state-BEFORE snapshot."""
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

    vouchers_redeemed, bosses_used = _build_voucher_boss_multihot(
        persistent_state, voucher_list, boss_list
    )
    out["vouchers_redeemed"] = vouchers_redeemed
    out["bosses_used"] = bosses_used

    out.update(_build_tracked_deck(persistent_state, vocab, cap_deck))
    out.update(_build_objects(step, vocab, cap_objects))

    mask = build_action_mask(step, action_map)
    out["action_mask"] = mask.astype(bool)
    base_action = parse_base_action(step.get("action") or "")
    if base_action == "StartNewRun":
        # StartNewRun is a bootstrap artifact, not a policy decision we want
        # the model to learn. Mark unresolved so the dataset excludes it.
        out["target_action_id"] = np.int32(-1)
        return out
    try:
        aid = resolve_label(step, action_map.get("label_to_index", {}))
    except LabelResolutionError:
        # Upstream data anomalies (e.g. bare "BuyAndUseShopConsumable" without
        # an index suffix) are recorded as -1 so the training loop can mask
        # them out instead of crashing the entire run. The validator script
        # already enumerates these.
        aid = -1
    # The mask-builder invariant ``action_mask[target_action_id] == 1`` is
    # required for finite cross-entropy loss; on rare data anomalies the
    # label resolves but the mask masks it out (e.g. a parsed
    # ``BuyAndUseShopConsumable`` event whose target object_type is
    # ``joker``). Drop these from training rather than letting
    # ``cross_entropy`` produce ``+inf`` losses with nan gradients.
    if aid is not None and aid >= 0 and not bool(mask[aid]):
        aid = -1
    out["target_action_id"] = np.int32(aid if aid is not None else -1)

    return out


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
    for t, step in enumerate(events):
        ps = _persistent_state_for_step(pstates, t)
        rec = tensorize_step(step, ps, action_map, vocab, norm, feature_config)
        records.append(rec)
        if int(rec["target_action_id"]) < 0:
            stats[("target", "unresolved")] += 1
        else:
            stats[("target", "resolved")] += 1
        stats[("event_base", parse_base_action(step.get("action") or ""))] += 1

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
    args = ap.parse_args()

    vocab = VocabLookup(json.loads(args.vocab.read_text(encoding="utf-8")))
    norm = Normalizer(json.loads(args.normalization.read_text(encoding="utf-8")))
    feature_config = json.loads(args.feature_config.read_text(encoding="utf-8"))
    action_config = json.loads(args.action_config.read_text(encoding="utf-8"))
    action_map = compute_action_map(action_config)

    args.out.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    runs_processed = 0
    steps_processed = 0
    bytes_written = 0
    start = time.time()
    current_video = None

    print(f"N_ACTIONS = {action_map['n_actions']}")
    print(f"action_map_version = {action_map['action_map_version']}")
    print(f"MAX_OBJECTS_PER_STEP = {feature_config['MAX_OBJECTS_PER_STEP']}")
    print(f"TRACKED_DECK_CAP = {feature_config['TRACKED_DECK_CAP']}")
    print()

    for video_id, gpath, ppath in _iter_run_pairs(args.src, args.persistent):
        if video_id != current_video:
            print(f"video_id={video_id}")
            current_video = video_id

        run = json.loads(gpath.read_text(encoding="utf-8"))
        psnap = json.loads(ppath.read_text(encoding="utf-8")) if ppath else None

        record = _process_run(run, psnap, action_map, vocab, norm, feature_config, stats)
        if not record:
            stats[("run", "empty")] += 1
            continue

        dst_dir = args.out / f"video_id={video_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / gpath.with_suffix(".npz").name
        np.savez_compressed(dst_file, **record)

        runs_processed += 1
        n_steps = next(iter(record.values())).shape[0]
        steps_processed += n_steps
        sz = dst_file.stat().st_size
        bytes_written += sz
        print(f"  {gpath.name} -> {dst_file.name}  ({n_steps} steps, {sz/1024:.1f} KiB)")

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
        "n_actions": action_map["n_actions"],
        "action_map_version": action_map["action_map_version"],
        "feature_config": {
            "MAX_OBJECTS_PER_STEP": feature_config["MAX_OBJECTS_PER_STEP"],
            "TRACKED_DECK_CAP": feature_config["TRACKED_DECK_CAP"],
            "N_HANDS": feature_config["N_HANDS"],
            "N_VOUCHERS": feature_config["N_VOUCHERS"],
            "N_BOSSES": feature_config["N_BOSSES"],
            "N_FLAGS": N_FLAGS,
            "N_OCR": N_OCR,
            "N_STATE_NUM": N_STATE_NUM,
        },
        "runs_processed": runs_processed,
        "steps_processed": steps_processed,
        "bytes_written": bytes_written,
        "elapsed_seconds": elapsed,
        "target_resolved": stats.get(("target", "resolved"), 0),
        "target_unresolved": stats.get(("target", "unresolved"), 0),
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
