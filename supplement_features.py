#!/usr/bin/env python3
"""
supplement_features.py
======================
Derive 62 fixed-length game-state features from one granularized step
(or live snapshot) plus its persistent_state. The output is the
``supplement_features`` channel that ``tensorize_step`` packs into the
model's global input alongside ``ocr_numeric`` / ``state_numeric`` /
``flags``.

Everything here is a pure function of the incoming snapshot. There is
no separate reducer / tracker — the values are computed from
``step.objects`` (which already carries the canonical zones, including
``PendingCards`` and ``CurrentHand`` / ``CurrentJokers``) plus
``persistent_state["hands"]`` for the two hand-history flags. This
means the live agent inherits these features automatically because
``LiveEncoder.encode`` routes through ``tensorize_step``.

Joker class IDs (see ``data/class_map.csv``):
  133 j_four_fingers, 174 j_pareidolia, 195 j_shortcut,
  198 j_smeared, 202 j_splash.

Card-field semantics (see ``parse_events.py``):
  rank_index: 0 ace, 1..8 -> "2".."9", 9 ten, 10 jack, 11 queen, 12 king
  suit_index: 0 spades, 1 hearts, 2 diamonds, 3 clubs
  modifier == "m_wild"  -> wild card, counts as every suit when not debuffed
  modifier == "m_stone" -> stone card, no rank / no suit, ALWAYS scores
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOKER_CLASS_FOUR_FINGERS = 133
JOKER_CLASS_PAREIDOLIA = 174
JOKER_CLASS_SHORTCUT = 195
JOKER_CLASS_SMEARED = 198
JOKER_CLASS_SPLASH = 202

# `make_*` block — exactly ONE of these is 1.0 whenever PendingCards is
# non-empty. Royal Flush is NOT a make-type; a royal flush sets
# `make_straight_flush = 1`.
_MAKE_NAMES: tuple[str, ...] = (
    "high_card", "pair", "two_pair", "three_kind", "straight",
    "flush", "full_house", "four_kind", "straight_flush",
    "five_kind", "flush_house", "flush_five",
)

# `have_*` block — zero or more true. `royal_flush` only appears here.
_HAVE_NAMES: tuple[str, ...] = (
    "high_card", "pair", "two_pair", "three_kind", "straight",
    "flush", "full_house", "four_kind", "straight_flush", "royal_flush",
    "five_kind", "flush_house", "flush_five",
)

# Map make-type identifier to the `persistent_state["hands"]` row key
# (see ``state_schema.md`` lines 532-549). Royal flush is intentionally
# absent: the game tracks it under "Straight Flush".
_HAND_KEY_BY_MAKE: dict[str, str] = {
    "high_card": "High Card",
    "pair": "Pair",
    "two_pair": "Two Pair",
    "three_kind": "Three of a Kind",
    "straight": "Straight",
    "flush": "Flush",
    "full_house": "Full House",
    "four_kind": "Four of a Kind",
    "straight_flush": "Straight Flush",
    "five_kind": "Five of a Kind",
    "flush_house": "Flush House",
    "flush_five": "Flush Five",
}


def _build_feature_names() -> list[str]:
    """Locked, ordered list of every feature this module emits."""
    out: list[str] = []
    out.extend(f"selected_cards_make_{n}" for n in _MAKE_NAMES)
    out.extend(f"selected_cards_have_{n}" for n in _HAVE_NAMES)
    out.extend(
        [
            "selected_contains_exactly_1_card",
            "selected_contains_exactly_4_cards",
            "selected_contains_exactly_5_cards",
            "selected_hand_type_already_played_this_round",
            "selected_hand_type_is_most_played_hand",
            "selected_position_0_scored_is_a_face_card",
            "selected_position_0_scored_is_debuffed",
            "selected_face_card_scored_count",
            "selected_ace_scored_count",
            "selected_2_scored_count",
            "selected_6_scored_count",
            "selected_8_scored_count",
            "selected_ace_2_3_5_scored_and_8_count",
            "selected_2_3_4_or_5_scored_count",
            "selected_even_rank_scored_count",
            "selected_odd_rank_scored_count",
            "selected_spade_scored_count",
            "selected_heart_scored_count",
            "selected_diamond_scored_count",
            "selected_club_scored_count",
            "selected_scored_debuff_count",
            "is_final_hand_of_round",
            "money_less_than_or_equal_to_4",
            "is_ante_1",
            "has_four_fingers",
            "has_shortcut",
            "has_smeared_joker",
            "has_pareidolia",
            "has_splash",
            "held_king_count",
            "held_queen_count",
            "held_face_card_count",
            "held_spade_count",
            "held_club_count",
            "held_spade_or_club_count",
            "held_lowest_rank_value",
            "held_lowest_rank_is_debuffed",
        ]
    )
    return out


SUPPLEMENT_FEATURE_NAMES: tuple[str, ...] = tuple(_build_feature_names())
N_SUPPLEMENT: int = len(SUPPLEMENT_FEATURE_NAMES)
_INDEX_BY_NAME: dict[str, int] = {n: i for i, n in enumerate(SUPPLEMENT_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Card-record extraction
# ---------------------------------------------------------------------------

def _is_card_object(obj: dict[str, Any]) -> bool:
    """Playing-card-shaped object (incl. stones)."""
    if obj.get("object_type") == "card":
        return True
    # Defensive: live shapes may omit object_type but still carry a card dict.
    return obj.get("object_type") is None and isinstance(obj.get("card"), dict)


def _extract_card(obj: dict[str, Any]) -> dict[str, Any]:
    """Simplified card record used by all hand-detection helpers."""
    cm = obj.get("card") or {}
    modifier = obj.get("modifier")
    is_debuffed = bool(obj.get("is_debuffed", False))
    rank_raw = cm.get("rank_index")
    suit_raw = cm.get("suit_index")
    rank: int | None = int(rank_raw) if isinstance(rank_raw, int) else None
    suit: int | None = int(suit_raw) if isinstance(suit_raw, int) else None
    is_stone = modifier == "m_stone"
    is_wild = (modifier == "m_wild") and not is_debuffed
    if is_stone:
        rank = None
        suit = None
    pos_raw = obj.get("position_in_zone")
    position = int(pos_raw) if isinstance(pos_raw, int) else 0
    return {
        "rank": rank,
        "suit": suit,
        "is_stone": is_stone,
        "is_wild": is_wild,
        "is_debuffed": is_debuffed,
        "position": position,
        # `is_face_base` is the BASE face-ness (J / Q / K). Pareidolia is
        # applied at usage sites because it depends on the joker flag.
        "is_face_base": rank in (10, 11, 12),
    }


def _collect_zone(
    objects: list[dict[str, Any]], zone_name: str
) -> list[dict[str, Any]]:
    out = [
        _extract_card(o)
        for o in objects
        if isinstance(o, dict)
        and o.get("zone") == zone_name
        and _is_card_object(o)
    ]
    out.sort(key=lambda c: c["position"])
    return out


def _collect_pending(step: dict[str, Any]) -> list[dict[str, Any]]:
    """PendingCards via the canonical zone, falling back to top-level list."""
    objects = step.get("objects") or []
    cards = _collect_zone(objects, "PendingCards")
    if cards:
        return cards
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(step.get("pending_cards") or []):
        if not isinstance(raw, dict):
            continue
        synthetic = dict(raw)
        synthetic.setdefault("position_in_zone", i)
        out.append(_extract_card(synthetic))
    return out


def _collect_held(step: dict[str, Any]) -> list[dict[str, Any]]:
    return _collect_zone(step.get("objects") or [], "CurrentHand")


def _joker_flags(step: dict[str, Any]) -> dict[str, bool]:
    """Boolean flags for the 5 scoring-relevant jokers (undebuffed only)."""
    flags = {
        "four_fingers": False,
        "shortcut": False,
        "smeared": False,
        "pareidolia": False,
        "splash": False,
    }
    for o in step.get("objects") or []:
        if not isinstance(o, dict):
            continue
        if o.get("zone") != "CurrentJokers":
            continue
        if bool(o.get("is_debuffed", False)):
            continue
        try:
            cid = int(o.get("class_id"))
        except (TypeError, ValueError):
            continue
        if cid == JOKER_CLASS_FOUR_FINGERS:
            flags["four_fingers"] = True
        elif cid == JOKER_CLASS_PAREIDOLIA:
            flags["pareidolia"] = True
        elif cid == JOKER_CLASS_SHORTCUT:
            flags["shortcut"] = True
        elif cid == JOKER_CLASS_SMEARED:
            flags["smeared"] = True
        elif cid == JOKER_CLASS_SPLASH:
            flags["splash"] = True
    return flags


# ---------------------------------------------------------------------------
# Suit / rank helpers
# ---------------------------------------------------------------------------

# Smeared groupings: spades(0)/clubs(3) -> group 0; hearts(1)/diamonds(2) -> group 1.
_SMEARED_GROUP = {0: 0, 1: 1, 2: 1, 3: 0}


def _suit_group(suit: int | None, smeared: bool) -> int | None:
    if suit is None:
        return None
    if smeared:
        return _SMEARED_GROUP[suit]
    return suit


def _card_can_be_in_suit_group(
    card: dict[str, Any], group: int, smeared: bool
) -> bool:
    if card["is_stone"]:
        return False
    if card["is_wild"]:
        return True
    return _suit_group(card["suit"], smeared) == group


def _rank_value(rank: int | None) -> int:
    """Ace-high numeric rank value: A=14, K=13, ..., 3=3, 2=2; None -> 0."""
    if rank is None:
        return 0
    return 14 if rank == 0 else rank + 1


# ---------------------------------------------------------------------------
# Hand-property detectors
# ---------------------------------------------------------------------------

def _rank_counts(cards: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for c in cards:
        r = c["rank"]
        if r is None:
            continue
        counts[r] = counts.get(r, 0) + 1
    return counts


def _is_straight(
    ranks: list[int], required_len: int, shortcut: bool
) -> bool:
    """True iff `ranks` (multiset) contains a `required_len`-card straight.

    The ace is dual-purpose: low (rank_index 0) and high (treated as 13).
    With `shortcut`, the step between consecutive used ranks may be 1 OR 2.
    """
    if required_len <= 0:
        return False
    distinct = sorted(set(ranks))
    if len(distinct) < required_len:
        return False

    def _scan(seq: list[int]) -> bool:
        for start in range(len(seq) - required_len + 1):
            window = seq[start:start + required_len]
            diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
            if shortcut:
                if all(1 <= d <= 2 for d in diffs):
                    return True
            else:
                if all(d == 1 for d in diffs):
                    return True
        return False

    if _scan(distinct):
        return True
    if 0 in distinct:
        # Ace-high: drop the ace-low at index 0 and append 13.
        high = sorted([r for r in distinct if r != 0] + [13])
        if _scan(high):
            return True
    return False


def _has_flush(
    cards: list[dict[str, Any]], required_len: int, smeared: bool
) -> bool:
    """True iff at least `required_len` cards can share one suit group."""
    if required_len <= 0:
        return False
    groups = (0, 1) if smeared else (0, 1, 2, 3)
    for g in groups:
        n = sum(1 for c in cards if _card_can_be_in_suit_group(c, g, smeared))
        if n >= required_len:
            return True
    return False


def _subset_is_straight_flush(
    subset: list[dict[str, Any]], shortcut: bool, smeared: bool
) -> bool:
    if any(c["is_stone"] for c in subset):
        # Stones don't have a rank/suit, so they can't be part of a SF.
        return False
    ranks = [c["rank"] for c in subset if c["rank"] is not None]
    if len(ranks) != len(subset):
        return False
    if not _is_straight(ranks, len(subset), shortcut):
        return False
    return _has_flush(subset, len(subset), smeared)


def _has_straight_flush(
    cards: list[dict[str, Any]],
    required_len: int,
    shortcut: bool,
    smeared: bool,
) -> bool:
    if required_len <= 0 or len(cards) < required_len:
        return False
    for subset in combinations(cards, required_len):
        if _subset_is_straight_flush(list(subset), shortcut, smeared):
            return True
    return False


def _has_full_house(cards: list[dict[str, Any]]) -> bool:
    counts = _rank_counts(cards)
    for r3, n3 in counts.items():
        if n3 < 3:
            continue
        for r2, n2 in counts.items():
            if r2 != r3 and n2 >= 2:
                return True
    return False


def _has_flush_five(
    cards: list[dict[str, Any]], four_fingers: bool, smeared: bool
) -> bool:
    counts = _rank_counts(cards)
    rank5 = next((r for r, n in counts.items() if n >= 5), None)
    if rank5 is None:
        return False
    five_cards = [c for c in cards if c["rank"] == rank5][:5]
    threshold = 4 if four_fingers else 5
    return _has_flush(five_cards, threshold, smeared)


def _has_flush_house(
    cards: list[dict[str, Any]], four_fingers: bool, smeared: bool
) -> bool:
    threshold = 4 if four_fingers else 5
    counts = _rank_counts(cards)
    for r3, n3 in counts.items():
        if n3 < 3:
            continue
        for r2, n2 in counts.items():
            if r2 == r3 or n2 < 2:
                continue
            three_cards = [c for c in cards if c["rank"] == r3][:3]
            two_cards = [c for c in cards if c["rank"] == r2][:2]
            if _has_flush(three_cards + two_cards, threshold, smeared):
                return True
    return False


def _has_royal_flush(
    cards: list[dict[str, Any]],
    shortcut: bool,
    smeared: bool,
    four_fingers: bool,
) -> bool:
    """True iff any 5- (or 4-with-ff) card subset is a straight flush AND
    every card's rank is >= 10 (rank_index in {0=ace-high, 9, 10, 11, 12}).
    """
    sizes = [5]
    if four_fingers:
        sizes.append(4)
    for size in sizes:
        if len(cards) < size:
            continue
        for subset in combinations(cards, size):
            sub = list(subset)
            if not _subset_is_straight_flush(sub, shortcut, smeared):
                continue
            if all(c["rank"] in (0, 9, 10, 11, 12) for c in sub):
                return True
    return False


# ---------------------------------------------------------------------------
# Make-type + have-set
# ---------------------------------------------------------------------------

def _detect_make_type(
    cards: list[dict[str, Any]], jokers: dict[str, bool]
) -> str | None:
    """Highest-tier hand the cards represent, or None for empty PendingCards.

    Precedence (highest first), per schema lines 87, 91-118:
    flush_five > flush_house > five_kind > straight_flush > four_kind >
    full_house > flush > straight > three_kind > two_pair > pair > high_card.
    """
    if not cards:
        return None
    ff = jokers["four_fingers"]
    sc = jokers["shortcut"]
    sm = jokers["smeared"]
    counts = _rank_counts(cards)
    max_count = max(counts.values()) if counts else 0
    straight_len = 4 if ff else 5

    if _has_flush_five(cards, ff, sm):
        return "flush_five"
    if _has_flush_house(cards, ff, sm):
        return "flush_house"
    if max_count >= 5:
        return "five_kind"
    if _has_straight_flush(cards, straight_len, sc, sm):
        return "straight_flush"
    if max_count >= 4:
        return "four_kind"
    if _has_full_house(cards):
        return "full_house"
    if _has_flush(cards, straight_len, sm):
        return "flush"
    ranks = [c["rank"] for c in cards if c["rank"] is not None]
    if _is_straight(ranks, straight_len, sc):
        return "straight"
    if max_count >= 3:
        return "three_kind"
    pair_ranks = sum(1 for n in counts.values() if n >= 2)
    if pair_ranks >= 2:
        return "two_pair"
    if max_count >= 2:
        return "pair"
    return "high_card"


def _detect_have_set(
    cards: list[dict[str, Any]], jokers: dict[str, bool]
) -> set[str]:
    """Every hand category some subset of `cards` can form."""
    out: set[str] = set()
    if not cards:
        return out
    ff = jokers["four_fingers"]
    sc = jokers["shortcut"]
    sm = jokers["smeared"]
    counts = _rank_counts(cards)
    max_count = max(counts.values()) if counts else 0
    pair_ranks = sum(1 for n in counts.values() if n >= 2)
    straight_len = 4 if ff else 5

    out.add("high_card")  # Any non-empty PendingCards has a high card.
    if max_count >= 2:
        out.add("pair")
    if pair_ranks >= 2:
        out.add("two_pair")
    if max_count >= 3:
        out.add("three_kind")
    if max_count >= 4:
        out.add("four_kind")
    if max_count >= 5:
        out.add("five_kind")
    if _has_full_house(cards):
        out.add("full_house")
    ranks = [c["rank"] for c in cards if c["rank"] is not None]
    if _is_straight(ranks, straight_len, sc):
        out.add("straight")
    if _has_flush(cards, straight_len, sm):
        out.add("flush")
    if _has_straight_flush(cards, straight_len, sc, sm):
        out.add("straight_flush")
    if _has_royal_flush(cards, sc, sm, ff):
        out.add("royal_flush")
    if _has_flush_house(cards, ff, sm):
        out.add("flush_house")
    if _has_flush_five(cards, ff, sm):
        out.add("flush_five")
    return out


# ---------------------------------------------------------------------------
# Scoring-subset selection
# ---------------------------------------------------------------------------

def _select_straight_cards(
    cards: list[dict[str, Any]], required_len: int, shortcut: bool
) -> list[dict[str, Any]]:
    """One card per rank, picked to form a `required_len` straight."""
    distinct_by_rank: dict[int, dict[str, Any]] = {}
    for c in cards:
        if c["rank"] is None:
            continue
        existing = distinct_by_rank.get(c["rank"])
        if existing is None or c["position"] < existing["position"]:
            distinct_by_rank[c["rank"]] = c

    def _scan(seq: list[int]) -> list[int] | None:
        for start in range(len(seq) - required_len + 1):
            window = seq[start:start + required_len]
            diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
            if shortcut:
                if all(1 <= d <= 2 for d in diffs):
                    return window
            else:
                if all(d == 1 for d in diffs):
                    return window
        return None

    low = sorted(distinct_by_rank)
    win = _scan(low)
    if win is not None:
        return [distinct_by_rank[r] for r in win]
    if 0 in distinct_by_rank:
        ace = distinct_by_rank[0]
        high_map: dict[int, dict[str, Any]] = {
            r: distinct_by_rank[r] for r in distinct_by_rank if r != 0
        }
        high_map[13] = ace
        high = sorted(high_map)
        win = _scan(high)
        if win is not None:
            return [high_map[r] for r in win]
    return []


def _select_flush_cards(
    cards: list[dict[str, Any]], required_len: int, smeared: bool
) -> list[dict[str, Any]]:
    """Largest subset of cards (>= required_len) sharing one suit group."""
    groups = (0, 1) if smeared else (0, 1, 2, 3)
    best: list[dict[str, Any]] = []
    for g in groups:
        members = [c for c in cards if _card_can_be_in_suit_group(c, g, smeared)]
        if len(members) >= required_len and len(members) > len(best):
            best = sorted(members, key=lambda c: c["position"])
    return best


def _select_straight_flush_cards(
    cards: list[dict[str, Any]],
    required_len: int,
    shortcut: bool,
    smeared: bool,
) -> list[dict[str, Any]]:
    if len(cards) < required_len:
        return []
    for subset in combinations(cards, required_len):
        sub = list(subset)
        if _subset_is_straight_flush(sub, shortcut, smeared):
            return sorted(sub, key=lambda c: c["position"])
    return []


def _cards_that_score(
    cards: list[dict[str, Any]],
    make_type: str | None,
    jokers: dict[str, bool],
) -> list[dict[str, Any]]:
    """Return the cards in PendingCards that would actually score for `make_type`.

    Game rules (schema line 89): only the cards relevant to the hand score,
    EXCEPT stone cards (always score) and `j_splash` (every card scores).
    """
    if not cards or make_type is None:
        return []
    ff = jokers["four_fingers"]
    sc = jokers["shortcut"]
    sm = jokers["smeared"]

    if jokers["splash"]:
        return list(cards)

    stones = [c for c in cards if c["is_stone"]]
    non_stones = [c for c in cards if not c["is_stone"]]
    counts = _rank_counts(non_stones)
    base: list[dict[str, Any]] = []

    if make_type == "high_card":
        if non_stones:
            base = [max(non_stones, key=lambda c: _rank_value(c["rank"]))]
    elif make_type == "pair":
        pair_rank = max(
            (r for r, n in counts.items() if n >= 2),
            key=_rank_value,
            default=None,
        )
        if pair_rank is not None:
            base = [c for c in non_stones if c["rank"] == pair_rank][:2]
    elif make_type == "two_pair":
        paired = [r for r, n in counts.items() if n >= 2]
        base = [c for c in non_stones if c["rank"] in paired][:4]
    elif make_type == "three_kind":
        three_rank = max(
            (r for r, n in counts.items() if n >= 3),
            key=_rank_value,
            default=None,
        )
        if three_rank is not None:
            base = [c for c in non_stones if c["rank"] == three_rank][:3]
    elif make_type == "four_kind":
        four_rank = max(
            (r for r, n in counts.items() if n >= 4),
            key=_rank_value,
            default=None,
        )
        if four_rank is not None:
            base = [c for c in non_stones if c["rank"] == four_rank][:4]
    elif make_type == "five_kind":
        five_rank = next((r for r, n in counts.items() if n >= 5), None)
        if five_rank is not None:
            base = [c for c in non_stones if c["rank"] == five_rank][:5]
    elif make_type in ("full_house", "flush_house", "flush_five"):
        # All 5 played cards score; precedence guarantees the layout.
        base = list(non_stones)[:5]
    elif make_type == "straight":
        base = _select_straight_cards(non_stones, 5, sc)
        if not base and ff:
            base = _select_straight_cards(non_stones, 4, sc)
    elif make_type == "flush":
        base = _select_flush_cards(non_stones, 5, sm)
        if not base and ff:
            base = _select_flush_cards(non_stones, 4, sm)
    elif make_type == "straight_flush":
        base = _select_straight_flush_cards(non_stones, 5, sc, sm)
        if not base and ff:
            base = _select_straight_flush_cards(non_stones, 4, sc, sm)
    return base + stones


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_supplement_features(
    step: dict[str, Any],
    persistent_state: dict[str, Any] | None,
) -> np.ndarray:
    """Build the (N_SUPPLEMENT,) float32 supplement-feature vector for one step."""
    out = np.zeros(N_SUPPLEMENT, dtype=np.float32)

    pending = _collect_pending(step)
    held = _collect_held(step)
    jokers = _joker_flags(step)
    pareidolia = jokers["pareidolia"]

    make_type = _detect_make_type(pending, jokers)
    have_set = _detect_have_set(pending, jokers)

    if make_type is not None:
        out[_INDEX_BY_NAME[f"selected_cards_make_{make_type}"]] = 1.0
    for ht in have_set:
        idx = _INDEX_BY_NAME.get(f"selected_cards_have_{ht}")
        if idx is not None:
            out[idx] = 1.0

    n_pending = len(pending)
    if n_pending == 1:
        out[_INDEX_BY_NAME["selected_contains_exactly_1_card"]] = 1.0
    elif n_pending == 4:
        out[_INDEX_BY_NAME["selected_contains_exactly_4_cards"]] = 1.0
    elif n_pending == 5:
        out[_INDEX_BY_NAME["selected_contains_exactly_5_cards"]] = 1.0

    hands = (persistent_state or {}).get("hands") or {}
    if make_type is not None:
        hand_key = _HAND_KEY_BY_MAKE.get(make_type)
        if hand_key is not None:
            entry = hands.get(hand_key) or {}
            try:
                ptr = int(entry.get("played_this_round") or 0)
            except (TypeError, ValueError):
                ptr = 0
            if ptr > 0:
                out[_INDEX_BY_NAME["selected_hand_type_already_played_this_round"]] = 1.0
            try:
                played_self = int(entry.get("played") or 0)
            except (TypeError, ValueError):
                played_self = 0
            max_played = 0
            for k in _HAND_KEY_BY_MAKE.values():
                try:
                    p = int((hands.get(k) or {}).get("played") or 0)
                except (TypeError, ValueError):
                    p = 0
                if p > max_played:
                    max_played = p
            if max_played > 0 and played_self == max_played:
                out[_INDEX_BY_NAME["selected_hand_type_is_most_played_hand"]] = 1.0

    scored = _cards_that_score(pending, make_type, jokers)

    if scored:
        first = min(scored, key=lambda c: c["position"])
        is_face = first["is_face_base"] or (pareidolia and not first["is_stone"])
        out[_INDEX_BY_NAME["selected_position_0_scored_is_a_face_card"]] = float(bool(is_face))
        out[_INDEX_BY_NAME["selected_position_0_scored_is_debuffed"]] = float(bool(first["is_debuffed"]))

    face_count = 0
    ace_count = 0
    two_count = 0
    six_count = 0
    eight_count = 0
    ace_2_3_5_8_count = 0
    two_three_four_five_count = 0
    even_count = 0
    odd_count = 0
    spade_count = 0
    heart_count = 0
    diamond_count = 0
    club_count = 0
    debuff_count = 0
    for c in scored:
        if c["is_debuffed"]:
            debuff_count += 1
        if c["is_stone"]:
            continue
        r = c["rank"]
        if r is None:
            continue
        if c["is_face_base"] or pareidolia:
            face_count += 1
        if r == 0:
            ace_count += 1
        if r == 1:
            two_count += 1
        if r == 5:
            six_count += 1
        if r == 7:
            eight_count += 1
        if r in (0, 1, 2, 4, 7):
            ace_2_3_5_8_count += 1
        if r in (1, 2, 3, 4):
            two_three_four_five_count += 1
        rv = _rank_value(r)
        if rv % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
        s = c["suit"]
        if s == 0:
            spade_count += 1
        elif s == 1:
            heart_count += 1
        elif s == 2:
            diamond_count += 1
        elif s == 3:
            club_count += 1
    out[_INDEX_BY_NAME["selected_face_card_scored_count"]] = float(face_count)
    out[_INDEX_BY_NAME["selected_ace_scored_count"]] = float(ace_count)
    out[_INDEX_BY_NAME["selected_2_scored_count"]] = float(two_count)
    out[_INDEX_BY_NAME["selected_6_scored_count"]] = float(six_count)
    out[_INDEX_BY_NAME["selected_8_scored_count"]] = float(eight_count)
    out[_INDEX_BY_NAME["selected_ace_2_3_5_scored_and_8_count"]] = float(ace_2_3_5_8_count)
    out[_INDEX_BY_NAME["selected_2_3_4_or_5_scored_count"]] = float(two_three_four_five_count)
    out[_INDEX_BY_NAME["selected_even_rank_scored_count"]] = float(even_count)
    out[_INDEX_BY_NAME["selected_odd_rank_scored_count"]] = float(odd_count)
    out[_INDEX_BY_NAME["selected_spade_scored_count"]] = float(spade_count)
    out[_INDEX_BY_NAME["selected_heart_scored_count"]] = float(heart_count)
    out[_INDEX_BY_NAME["selected_diamond_scored_count"]] = float(diamond_count)
    out[_INDEX_BY_NAME["selected_club_scored_count"]] = float(club_count)
    out[_INDEX_BY_NAME["selected_scored_debuff_count"]] = float(debuff_count)

    state = step.get("state") or {}
    hands_left = state.get("hands_left")
    dollars = state.get("dollars")
    ante = state.get("ante")
    if isinstance(hands_left, (int, float)) and not isinstance(hands_left, bool):
        if int(hands_left) == 1:
            out[_INDEX_BY_NAME["is_final_hand_of_round"]] = 1.0
    if isinstance(dollars, (int, float)) and not isinstance(dollars, bool):
        if int(dollars) <= 4:
            out[_INDEX_BY_NAME["money_less_than_or_equal_to_4"]] = 1.0
    if isinstance(ante, (int, float)) and not isinstance(ante, bool):
        if int(ante) == 1:
            out[_INDEX_BY_NAME["is_ante_1"]] = 1.0

    out[_INDEX_BY_NAME["has_four_fingers"]] = float(jokers["four_fingers"])
    out[_INDEX_BY_NAME["has_shortcut"]] = float(jokers["shortcut"])
    out[_INDEX_BY_NAME["has_smeared_joker"]] = float(jokers["smeared"])
    out[_INDEX_BY_NAME["has_pareidolia"]] = float(jokers["pareidolia"])
    out[_INDEX_BY_NAME["has_splash"]] = float(jokers["splash"])

    held_king = 0
    held_queen = 0
    held_face = 0
    held_spade = 0
    held_club = 0
    held_spade_or_club = 0
    for c in held:
        if c["is_stone"]:
            continue
        r = c["rank"]
        if r == 12:
            held_king += 1
        if r == 11:
            held_queen += 1
        if c["is_face_base"] or pareidolia:
            held_face += 1
        s = c["suit"]
        if s == 0:
            held_spade += 1
            held_spade_or_club += 1
        elif s == 3:
            held_club += 1
            held_spade_or_club += 1
    out[_INDEX_BY_NAME["held_king_count"]] = float(held_king)
    out[_INDEX_BY_NAME["held_queen_count"]] = float(held_queen)
    out[_INDEX_BY_NAME["held_face_card_count"]] = float(held_face)
    out[_INDEX_BY_NAME["held_spade_count"]] = float(held_spade)
    out[_INDEX_BY_NAME["held_club_count"]] = float(held_club)
    out[_INDEX_BY_NAME["held_spade_or_club_count"]] = float(held_spade_or_club)

    held_with_rank = [c for c in held if c["rank"] is not None]
    if held_with_rank:
        # Lowest rank value with high-ace semantics, leftmost tie-break.
        held_with_rank.sort(key=lambda c: (_rank_value(c["rank"]), c["position"]))
        lowest = held_with_rank[0]
        out[_INDEX_BY_NAME["held_lowest_rank_value"]] = float(_rank_value(lowest["rank"]))
        out[_INDEX_BY_NAME["held_lowest_rank_is_debuffed"]] = float(bool(lowest["is_debuffed"]))

    return out
