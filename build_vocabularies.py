#!/usr/bin/env python3
"""
build_vocabularies.py
=====================
Scan the granularized + persistent_state corpora and emit:

- ``artifacts/vocab.json`` — every categorical vocabulary in one place,
  one ``{name: {"pad_index": 0, "size": K, "values": ["<PAD>", v1, ..., vK-1]}}``
  block per channel. Index 0 is reserved for PAD/None.
- ``artifacts/normalization.json`` — per-numeric-feature stats
  (count, mean, std, min, max, p1, p99) and the recommended transform.
- ``artifacts/feature_config.json`` — frozen tensor shapes / caps:
  ``MAX_OBJECTS_PER_STEP``, ``TRACKED_DECK_CAP``, ``N_HANDS``,
  ``N_VOUCHERS``, ``N_BOSSES``, plus pointers to the other artifacts.

These three files together fully specify what ``tensorize.py`` will emit;
they are the contract the model + training loop depend on.

Vocab strategy
--------------
- All vocabs reserve index ``0`` for ``"<PAD>"`` (None / missing / out-of-vocab).
- ``class_id`` uses the class id itself as the index (size = 400 + 1 OOV).
  This gives the model a stable embedding-table layout regardless of which
  class ids appeared in the granularized corpus.
- Other categoricals (page, source_kind, action_subtype, zone, object_type,
  modifier, edition, seal, sticker, deck.class_id, stake.class_id,
  small_status / big_status, last_tarot_planet, ante_boss_blind, voucher_class_id,
  boss_class_id) are seeded with the canonical set from the schemas and
  augmented with anything else observed in the corpus.

Normalization strategy
----------------------
For each numeric channel we compute (mean, std, min, max, p1, p99) and
choose a transform from {``identity``, ``minmax``, ``log1p_zscore``,
``zscore``}:

- counters that grow without bound (``round_score``, ``dollars`` after stacking,
  ``hands[*].level``): ``log1p_zscore`` so the tail doesn't dominate.
- bounded small ints (``hands_left``, ``discards_left``, ``jokers_current``,
  ``ante``, etc.): ``minmax`` with the locked ``[min, max]`` from the corpus.
- everything else: ``zscore`` with mean/std.

Tensorize and the model both load these stats and apply the documented
transform exactly once.

Usage
-----
``python build_vocabularies.py
    [--granularized data/granularized]
    [--persistent  data/persistent_state]
    [--metadata    data/metadata_map.csv]
    [--out         artifacts]``
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Import the canonical state-reducer constants so vocabs stay in sync with
# what the reducer can actually produce.
from card_effects import (
    PLANET_TO_HAND,
    BLACK_HOLE_CLASS,
    STONE_CARD_CLASS,
)
from state_reducer import (
    BIG_BLIND,
    BOSS_BLIND_RANGE,
    DECK_CLASS_RANGE,
    PLANET_RANGE,
    POKER_HANDS,
    SMALL_BLIND,
    SPECTRAL_RANGE,
    TAROT_RANGE,
    VOUCHER_CLASS_RANGE,
)


VOCAB_SCHEMA_VERSION = "1.0.0"
NORMALIZATION_SCHEMA_VERSION = "1.0.0"
FEATURE_CONFIG_VERSION = "1.0.0"

PAD_TOKEN = "<PAD>"

# Class-id space (data/metadata_map.csv covers 0..399).
CLASS_ID_VOCAB_SIZE = 401  # 0..399 + 1 OOV slot at index 400
CLASS_ID_OOV = 400

# Hard caps frozen from the corpus survey (see chat history).
MAX_OBJECTS_PER_STEP = 48       # observed max 41 after merging hand back into objects (see _check_object_cap.py); +headroom
TRACKED_DECK_CAP = 75           # matches card_effects.TRACKED_DECK_CAP
N_HANDS = len(POKER_HANDS)      # 12

# Multi-hot vocab spans (used for vouchers_redeemed / bosses_used).
VOUCHER_CLASS_LIST = list(VOUCHER_CLASS_RANGE)             # 320..351
BOSS_CLASS_LIST = [
    cid for cid in BOSS_BLIND_RANGE
    if cid not in (BIG_BLIND, SMALL_BLIND)
]

# Stake range from the schema (266..273).
STAKE_CLASS_LIST = list(range(266, 274))

# Small-status / big-status discrete domain (0=current ante's blind, 1=fresh,
# 2=skipped). The reducer enforces these explicitly.
SMALL_BIG_STATUS_VALUES = [0, 1, 2]

# last_tarot_planet domain (planets + tarots + None).
LAST_TAROT_PLANET_VALUES = list(PLANET_RANGE) + list(TAROT_RANGE)

# Boolean-flag channels collected directly from the persistent state.
PERSISTENT_BOOL_FIELDS = [
    "first_hand",
    "first_discard",
    "is_boss_blind_rerolled",
]
DECK_FLAG_FIELDS = [
    "is_magic", "is_nebula", "is_abandoned", "is_checkered",
    "is_zodiac", "is_erratic",
]
DECK_MODIFIER_FIELDS = [
    "no_face_cards_start", "spades_hearts_only_start", "randomized_starting_deck",
]


# Canonical seed values keep the vocab stable even when a value happens not
# to appear in the current corpus. Indexes for these are still assigned in
# sorted order, so adding a new seed bumps the schema version implicitly.
CANONICAL_SEEDS: dict[str, list[Any]] = {
    "page": [
        "Blind_Select",
        "Cash_Out",
        "Dummy_Page",
        "In_Blind",
        "In_JokerStandardPlanet_Pack",
        "In_Shop",
        "In_TarotSpectral_Pack",
    ],
    "source_kind": ["pass_through", "select", "commit", "swap_synth"],
    "object_type": [
        "card", "joker", "consumable", "deck", "stake", "tag", "blind",
        "pack", "voucher", "modifier", "edition", "enhancement",
        "spectral", "tarot", "planet", "sticker",
    ],
    "modifier": [
        "m_bonus", "m_glass", "m_gold", "m_lucky",
        "m_mult", "m_steel", "m_stone", "m_wild",
    ],
    "edition": ["e_foil", "e_holo", "e_negative", "e_polychrome"],
    "seal": ["red_seal", "blue_seal", "gold_seal", "purple_seal"],
    "sticker": ["rental", "perishable", "eternal"],
    "rank_index": list(range(0, 13)),  # 0=A..12=K
    "suit_index": list(range(0, 4)),   # 0=Spades..3=Clubs
    "small_status": SMALL_BIG_STATUS_VALUES,
    "big_status": SMALL_BIG_STATUS_VALUES,
    "stake_class_id": STAKE_CLASS_LIST,
    "deck_class_id": list(DECK_CLASS_RANGE),
    "last_tarot_planet_class_id": LAST_TAROT_PLANET_VALUES,
    "ante_boss_blind_class_id": BOSS_CLASS_LIST,
    "voucher_class_id": VOUCHER_CLASS_LIST,
    "boss_class_id": BOSS_CLASS_LIST,
}


# Numeric channels we extract per step. Each is mapped to one of:
# - "minmax": (x - min) / max(max - min, 1)         (bounded ints)
# - "log1p_zscore": (log1p(max(x,0)) - mu) / sigma  (heavy-tailed counters)
# - "zscore": (x - mu) / sigma                      (default)
# - "identity": x                                   (booleans)
NUMERIC_TRANSFORMS: dict[str, str] = {
    # OCR (per-step) -- bounded by game rules
    "ocr.hands_left": "minmax",
    "ocr.discards_left": "minmax",
    "ocr.dollars": "log1p_zscore",
    "ocr.ante": "minmax",
    "ocr.round": "minmax",
    "ocr.deck_remaining": "minmax",
    "ocr.deck_total": "minmax",
    "ocr.round_score": "log1p_zscore",
    "ocr.cash_out": "log1p_zscore",
    "ocr.reroll_price": "log1p_zscore",
    "ocr.consumables_current": "minmax",
    "ocr.consumables_total": "minmax",
    "ocr.jokers_current": "minmax",
    "ocr.jokers_total": "minmax",
    "ocr.hand_size_current": "minmax",
    "ocr.hand_size_total": "minmax",
    # Persistent counters
    "state.skips": "log1p_zscore",
    "state.hands_played": "log1p_zscore",
    "state.unused_discards": "log1p_zscore",
    "state.ecto_minus": "minmax",
    # Per-hand progression (per-hand index slots; same transform for all 12)
    "hand_level": "log1p_zscore",
    "hand_played": "log1p_zscore",
    "hand_played_this_round": "minmax",
}


# ---------------------------------------------------------------------------
# Numeric stats accumulator
# ---------------------------------------------------------------------------

class StatsAccumulator:
    """Online mean/std + min/max + reservoir-light percentile estimate."""

    __slots__ = ("count", "_mean", "_m2", "min", "max", "_values", "_reservoir_cap")

    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        # Bounded-size reservoir for percentile estimates.
        self._values: list[float] = []
        self._reservoir_cap = 200_000

    def add(self, x: float) -> None:
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        if len(self._values) < self._reservoir_cap:
            self._values.append(x)

    @property
    def mean(self) -> float:
        return self._mean if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        var = self._m2 / (self.count - 1)
        return math.sqrt(max(var, 0.0))

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
        return sorted_vals[idx]

    def to_json(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0, "std": 1.0,
                "min": 0.0, "max": 0.0,
                "p1": 0.0, "p99": 0.0,
            }
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std if self.std > 0 else 1.0,
            "min": float(self.min),
            "max": float(self.max),
            "p1": self.percentile(1.0),
            "p99": self.percentile(99.0),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_run_pairs(granularized: Path, persistent: Path) -> Iterable[tuple[Path, Path | None]]:
    """Yield (granularized_run_file, matching_persistent_file_or_None)."""
    for partition in sorted(granularized.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        persistent_dir = persistent / partition.name
        for gpath in sorted(partition.glob("run_*.json")):
            ppath = persistent_dir / gpath.name
            yield gpath, ppath if ppath.exists() else None


def _load_metadata_map(metadata_csv: Path) -> dict[int, dict[str, Any]]:
    """Read class_id -> {name, object_type, ...} from metadata_map.csv."""
    meta: dict[int, dict[str, Any]] = {}
    with metadata_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["class_id"])
            except (KeyError, ValueError):
                continue
            meta[cid] = {
                "name": row.get("class_name"),
                "object_type": row.get("object_type"),
            }
    return meta


def _categorical_to_vocab(name: str, observed: set[Any], seeds: list[Any] | None = None) -> dict[str, Any]:
    """Build a {value: index} vocab with PAD at 0; deterministic ordering."""
    universe = set(observed)
    if seeds:
        universe.update(seeds)
    universe.discard(None)
    universe.discard(PAD_TOKEN)
    sorted_values = sorted(universe, key=lambda v: (str(type(v).__name__), str(v)))
    values = [PAD_TOKEN] + sorted_values
    return {
        "name": name,
        "pad_index": 0,
        "size": len(values),
        "values": values,
        "value_to_index": {str(v): i for i, v in enumerate(values)},
    }


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--granularized", type=Path, default=Path("data/granularized"))
    ap.add_argument("--persistent", type=Path, default=Path("data/persistent_state"))
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata_map.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_map = _load_metadata_map(args.metadata)

    # Categorical observation buckets.
    observed: dict[str, set[Any]] = collections.defaultdict(set)
    # Numeric stats buckets.
    stats: dict[str, StatsAccumulator] = collections.defaultdict(StatsAccumulator)

    n_steps = 0
    n_runs = 0
    max_objects_seen = 0
    n_persistent_steps = 0

    print("scanning corpus...")
    for gpath, ppath in _iter_run_pairs(args.granularized, args.persistent):
        n_runs += 1
        run = json.loads(gpath.read_text(encoding="utf-8"))
        psnap = (
            json.loads(ppath.read_text(encoding="utf-8")) if ppath else None
        )
        pstates = (psnap or {}).get("states") or []

        for t, step in enumerate(run.get("events") or []):
            n_steps += 1
            # ---- categorical observations ----
            observed["page"].add(step.get("page_name"))
            observed["source_kind"].add(step.get("source_kind"))
            observed["action_subtype"].add(step.get("action_subtype"))

            for o in step.get("objects") or []:
                observed["zone"].add(o.get("zone"))
                observed["object_type"].add(o.get("object_type"))
                observed["modifier"].add(o.get("modifier"))
                observed["edition"].add(o.get("edition"))
                observed["seal"].add(o.get("seal"))
                cm = o.get("card") or {}
                if cm:
                    observed["rank_index"].add(cm.get("rank_index"))
                    observed["suit_index"].add(cm.get("suit_index"))
                for st in (o.get("stickers") or []):
                    observed["sticker"].add(st)

            objs = step.get("objects") or []
            if len(objs) > max_objects_seen:
                max_objects_seen = len(objs)

            # ---- per-step OCR numerics ----
            ocr = step.get("state") or {}
            for key in (
                "hands_left", "discards_left", "dollars", "ante", "round",
                "deck_remaining", "deck_total", "round_score", "cash_out",
                "reroll_price", "consumables_current", "consumables_total",
                "jokers_current", "jokers_total", "hand_size_current",
                "hand_size_total",
            ):
                v = ocr.get(key)
                if isinstance(v, (int, float)):
                    stats[f"ocr.{key}"].add(float(v))

            # ---- persistent state numerics ----
            if t < len(pstates):
                ps = pstates[t]
                n_persistent_steps += 1

                stats["state.skips"].add(float(ps.get("skips", 0)))
                stats["state.hands_played"].add(float(ps.get("hands_played", 0)))
                stats["state.unused_discards"].add(float(ps.get("unused_discards", 0)))
                stats["state.ecto_minus"].add(float(ps.get("ecto_minus", 0)))

                # Hand progression.
                hands = ps.get("hands") or {}
                for hand_name, entry in hands.items():
                    if not isinstance(entry, dict):
                        continue
                    stats["hand_level"].add(float(entry.get("level", 1)))
                    stats["hand_played"].add(float(entry.get("played", 0)))
                    stats["hand_played_this_round"].add(
                        float(entry.get("played_this_round", 0))
                    )

                # Persistent categoricals (deck class id, stake, etc.) -- vocab seeds
                # already cover these; we just need to confirm any unseen IDs.
                deck = ps.get("deck") or {}
                if isinstance(deck.get("class_id"), int):
                    observed["deck_class_id"].add(deck["class_id"])
                if isinstance(ps.get("stake"), int):
                    observed["stake_class_id"].add(ps["stake"])
                if isinstance(ps.get("last_tarot_planet"), int):
                    observed["last_tarot_planet_class_id"].add(ps["last_tarot_planet"])
                if isinstance(ps.get("ante_boss_blind"), int):
                    observed["ante_boss_blind_class_id"].add(ps["ante_boss_blind"])
                if isinstance(ps.get("small_status"), int):
                    observed["small_status"].add(ps["small_status"])
                if isinstance(ps.get("big_status"), int):
                    observed["big_status"].add(ps["big_status"])
                for cid in ps.get("vouchers_redeemed") or []:
                    observed["voucher_class_id"].add(cid)
                for cid in ps.get("bosses_used") or []:
                    observed["boss_class_id"].add(cid)

                # Tracked deck observations
                for c in ps.get("tracked_deck_cards") or []:
                    observed["modifier"].add(c.get("modifier"))
                    observed["edition"].add(c.get("edition"))
                    observed["seal"].add(c.get("seal"))
                    cm = c.get("card") or {}
                    if cm:
                        observed["rank_index"].add(cm.get("rank_index"))
                        observed["suit_index"].add(cm.get("suit_index"))
                    for st in c.get("stickers") or []:
                        observed["sticker"].add(st)

        if n_runs % 50 == 0:
            print(f"  runs scanned: {n_runs}/{n_steps} steps")

    print(f"done. runs={n_runs} steps={n_steps} persistent_state_steps={n_persistent_steps}")
    print(f"observed max objects per step: {max_objects_seen}")

    # ------------------------------------------------------------------
    # Emit categorical vocabularies
    # ------------------------------------------------------------------
    vocabs: dict[str, dict[str, Any]] = {}
    for name, obs in observed.items():
        seeds = CANONICAL_SEEDS.get(name)
        vocabs[name] = _categorical_to_vocab(name, obs, seeds)

    # Add canonical-only vocabs that aren't strictly populated above.
    for name, seeds in CANONICAL_SEEDS.items():
        if name not in vocabs:
            vocabs[name] = _categorical_to_vocab(name, set(), seeds)

    # class_id is special — use the class id itself as the index.
    vocabs["class_id"] = {
        "name": "class_id",
        "pad_index": 0,
        "size": CLASS_ID_VOCAB_SIZE,
        "encoding": "identity",
        "oov_index": CLASS_ID_OOV,
        "metadata_map_path": str(args.metadata),
        "metadata_map_count": len(meta_map),
        "comment": (
            "class_id encodes objects directly: vocab_index == class_id for "
            "0..399; index 400 is OOV; index 0 is unused for non-class objects "
            "(e.g. modifier-only entries) — callers must explicitly map None to 0."
        ),
    }

    vocab_payload = {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": n_runs,
        "n_steps": n_steps,
        "n_persistent_steps": n_persistent_steps,
        "vocabularies": vocabs,
    }
    (args.out / "vocab.json").write_text(
        json.dumps(vocab_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote artifacts/vocab.json ({len(vocabs)} categorical channels)")
    for name, vocab in sorted(vocabs.items()):
        size = vocab.get("size") or vocab.get("vocab_size") or "?"
        print(f"  {name:32s} size={size}")

    # ------------------------------------------------------------------
    # Emit numeric normalization stats
    # ------------------------------------------------------------------
    norm_payload = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_steps": n_steps,
        "transforms": NUMERIC_TRANSFORMS,
        "stats": {name: s.to_json() for name, s in sorted(stats.items())},
    }
    (args.out / "normalization.json").write_text(
        json.dumps(norm_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote artifacts/normalization.json ({len(stats)} numeric channels)")
    for name, st in sorted(stats.items()):
        s = st.to_json()
        print(
            f"  {name:32s} count={s['count']:7d} "
            f"mean={s['mean']:9.3f} std={s['std']:9.3f} "
            f"min={s['min']:8.2f} max={s['max']:9.2f} "
            f"p1={s['p1']:8.2f} p99={s['p99']:9.2f}"
        )

    # ------------------------------------------------------------------
    # Emit feature-shape config
    # ------------------------------------------------------------------
    feature_config = {
        "schema_version": FEATURE_CONFIG_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "MAX_OBJECTS_PER_STEP": MAX_OBJECTS_PER_STEP,
        "MAX_OBJECTS_PER_STEP_observed": max_objects_seen,
        "TRACKED_DECK_CAP": TRACKED_DECK_CAP,
        "N_HANDS": N_HANDS,
        "POKER_HANDS": list(POKER_HANDS),
        "N_VOUCHERS": len(VOUCHER_CLASS_LIST),
        "VOUCHER_CLASS_LIST": VOUCHER_CLASS_LIST,
        "N_BOSSES": len(BOSS_CLASS_LIST),
        "BOSS_CLASS_LIST": BOSS_CLASS_LIST,
        "PERSISTENT_BOOL_FIELDS": PERSISTENT_BOOL_FIELDS,
        "DECK_FLAG_FIELDS": DECK_FLAG_FIELDS,
        "DECK_MODIFIER_FIELDS": DECK_MODIFIER_FIELDS,
        "BLACK_HOLE_CLASS": BLACK_HOLE_CLASS,
        "STONE_CARD_CLASS": STONE_CARD_CLASS,
        "PLANET_TO_HAND": {str(k): v for k, v in PLANET_TO_HAND.items()},
        "vocab_path": "artifacts/vocab.json",
        "normalization_path": "artifacts/normalization.json",
        "action_map_path": "data/action_map.json",
        "action_space_config_path": "data/action_space_config.json",
    }
    (args.out / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote artifacts/feature_config.json")


if __name__ == "__main__":
    main()
