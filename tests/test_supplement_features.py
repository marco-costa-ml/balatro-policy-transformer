#!/usr/bin/env python3
"""
test_supplement_features.py
===========================
Unit tests for the supplement-feature computation. Runnable two ways:

    pytest tests/test_supplement_features.py
    python tests/test_supplement_features.py

The second mode short-circuits straight to ``main()`` and just walks every
``test_*`` function in this module, so we don't take a dependency on
pytest. Failures are surfaced as plain AssertionError tracebacks.

The tests are split into small scenarios. Each scenario:

  1. Builds a synthetic granularized-style step (objects + pending_cards +
     state) plus an optional persistent_state.
  2. Calls ``compute_supplement_features``.
  3. Asserts the expected feature values via ``get_feature(name, vec)``.

All expected values are hand-derived from Balatro hand-ranking rules and
the joker class IDs (133/174/195/198/202).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from supplement_features import (
    N_SUPPLEMENT,
    SUPPLEMENT_FEATURE_NAMES,
    compute_supplement_features,
)


# ---------------------------------------------------------------------------
# Tiny test helpers (no pytest dependency needed)
# ---------------------------------------------------------------------------

def get(name: str, vec: np.ndarray) -> float:
    if name not in SUPPLEMENT_FEATURE_NAMES:
        raise KeyError(f"unknown supplement feature: {name}")
    idx = SUPPLEMENT_FEATURE_NAMES.index(name)
    return float(vec[idx])


def card(zone: str, pos: int, rank_index: int, suit_index: int,
         modifier: str | None = None, is_debuffed: bool = False) -> dict:
    """Build a canonical card object the way granularize.py emits them."""
    return {
        "class_id": suit_index * 13 + rank_index,
        "object_type": "card",
        "zone": zone,
        "position_in_zone": pos,
        "modifier": modifier,
        "edition": None,
        "seal": None,
        "is_debuffed": is_debuffed,
        "card": {
            "rank_index": rank_index,
            "suit_index": suit_index,
            "is_face": rank_index in (10, 11, 12),
            "is_ace": rank_index == 0,
        },
    }


def joker(pos: int, class_id: int, is_debuffed: bool = False) -> dict:
    return {
        "class_id": class_id,
        "object_type": "joker",
        "zone": "CurrentJokers",
        "position_in_zone": pos,
        "modifier": None,
        "edition": None,
        "seal": None,
        "is_debuffed": is_debuffed,
        "card": None,
    }


def make_step(pending: list[dict], held: list[dict] | None = None,
              jokers: list[dict] | None = None,
              state: dict | None = None) -> dict:
    objects: list[dict] = []
    for c in pending:
        objects.append({**c, "zone": "PendingCards"})
    if held:
        for c in held:
            objects.append({**c, "zone": "CurrentHand"})
    if jokers:
        objects.extend(jokers)
    return {
        "page_name": "In_Blind",
        "state": state or {"hands_left": 4, "dollars": 4, "ante": 1},
        "objects": objects,
        "pending_cards": pending,
    }


# Convenience rank constants (0=A, 1=2, ..., 9=T, 10=J, 11=Q, 12=K).
A, R2, R3, R4, R5, R6, R7, R8, R9, T_, J, Q, K = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
S, H, D, C = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Make-type tests (12 base hand types)
# ---------------------------------------------------------------------------

def test_high_card_single() -> None:
    """Single card → make_high_card; exactly_1_card; first-scored = that card."""
    step = make_step([card("PendingCards", 0, K, S)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_high_card", v) == 1.0
    assert get("selected_contains_exactly_1_card", v) == 1.0
    assert get("selected_face_card_scored_count", v) == 1.0
    assert get("selected_spade_scored_count", v) == 1.0
    assert get("selected_position_0_scored_is_a_face_card", v) == 1.0


def test_pair_two_of_a_kind() -> None:
    """7,7,K,Q,2 → pair. Only the two 7s score: 2 odd-rank, no face, no ace."""
    step = make_step([
        card("PendingCards", 0, R7, S),
        card("PendingCards", 1, R7, H),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, Q, C),
        card("PendingCards", 4, R2, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_pair", v) == 1.0
    assert get("selected_cards_make_high_card", v) == 0.0
    assert get("selected_contains_exactly_5_cards", v) == 1.0
    # Scored = 2x 7 only.
    assert get("selected_face_card_scored_count", v) == 0.0
    assert get("selected_ace_scored_count", v) == 0.0
    assert get("selected_odd_rank_scored_count", v) == 2.0
    assert get("selected_spade_scored_count", v) == 1.0
    assert get("selected_heart_scored_count", v) == 1.0
    # have-set covers higher categories too.
    assert get("selected_cards_have_pair", v) == 1.0
    assert get("selected_cards_have_high_card", v) == 1.0
    assert get("selected_cards_have_two_pair", v) == 0.0


def test_two_pair() -> None:
    """K,K,7,7,2 → two_pair; both pairs score (4 cards)."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, R7, D),
        card("PendingCards", 3, R7, C),
        card("PendingCards", 4, R2, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_two_pair", v) == 1.0
    # Scored = 2 kings + 2 sevens. The lone 2 does not score.
    assert get("selected_face_card_scored_count", v) == 2.0
    # K=13 (odd), 7=7 (odd) so all 4 scored cards are odd by rank_value.
    assert get("selected_odd_rank_scored_count", v) == 4.0
    assert get("selected_even_rank_scored_count", v) == 0.0


def test_three_of_a_kind() -> None:
    """K,K,K,7,2 → three_kind; 3 kings score."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, R7, C),
        card("PendingCards", 4, R2, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_three_kind", v) == 1.0
    assert get("selected_face_card_scored_count", v) == 3.0
    assert get("selected_cards_have_three_kind", v) == 1.0
    assert get("selected_cards_have_pair", v) == 1.0


def test_straight_normal() -> None:
    """8,9,T,J,Q (mixed suits) → straight; all 5 score."""
    step = make_step([
        card("PendingCards", 0, R8, S),
        card("PendingCards", 1, R9, H),
        card("PendingCards", 2, T_, D),
        card("PendingCards", 3, J, C),
        card("PendingCards", 4, Q, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight", v) == 1.0
    # 5 cards score; faces = J, Q = 2 face cards.
    assert get("selected_face_card_scored_count", v) == 2.0
    assert get("selected_8_scored_count", v) == 1.0


def test_straight_ace_low() -> None:
    """A,2,3,4,5 → straight using ace-low."""
    step = make_step([
        card("PendingCards", 0, A, S),
        card("PendingCards", 1, R2, H),
        card("PendingCards", 2, R3, D),
        card("PendingCards", 3, R4, C),
        card("PendingCards", 4, R5, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight", v) == 1.0
    assert get("selected_ace_scored_count", v) == 1.0
    assert get("selected_2_3_4_or_5_scored_count", v) == 4.0


def test_flush_normal() -> None:
    """5 spades, non-consecutive → flush; all 5 score."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R9, S),
        card("PendingCards", 2, R7, S),
        card("PendingCards", 3, R5, S),
        card("PendingCards", 4, R2, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 1.0
    assert get("selected_spade_scored_count", v) == 5.0


def test_full_house() -> None:
    """K,K,K,7,7 → full_house; all 5 score."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, R7, C),
        card("PendingCards", 4, R7, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_full_house", v) == 1.0
    assert get("selected_face_card_scored_count", v) == 3.0
    # have-set should also flag three_kind/pair but NOT two_pair (distinct-rank rule).
    assert get("selected_cards_have_pair", v) == 1.0
    assert get("selected_cards_have_three_kind", v) == 1.0
    assert get("selected_cards_have_full_house", v) == 1.0


def test_four_of_a_kind_precedes_flush() -> None:
    """K,K,K,K,2 all diamonds → four_kind (NOT flush, by precedence)."""
    step = make_step([
        card("PendingCards", 0, K, D),
        card("PendingCards", 1, K, D),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, K, D),
        card("PendingCards", 4, R2, D),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_four_kind", v) == 1.0
    assert get("selected_cards_make_flush", v) == 0.0
    # have-set should still include flush + four_kind.
    assert get("selected_cards_have_flush", v) == 1.0
    assert get("selected_cards_have_four_kind", v) == 1.0
    # 4 kings score, the 2 of diamonds does NOT.
    assert get("selected_face_card_scored_count", v) == 4.0
    assert get("selected_diamond_scored_count", v) == 4.0


def test_straight_flush_low() -> None:
    """A,2,3,4,5 all spades → straight_flush (ace-low). NOT royal."""
    step = make_step([
        card("PendingCards", 0, A, S),
        card("PendingCards", 1, R2, S),
        card("PendingCards", 2, R3, S),
        card("PendingCards", 3, R4, S),
        card("PendingCards", 4, R5, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight_flush", v) == 1.0
    assert get("selected_cards_have_royal_flush", v) == 0.0
    # have-set covers everything below.
    assert get("selected_cards_have_straight", v) == 1.0
    assert get("selected_cards_have_flush", v) == 1.0
    assert get("selected_cards_have_straight_flush", v) == 1.0


def test_royal_flush_have_flag() -> None:
    """T,J,Q,K,A all spades → straight_flush (NOT make_royal_flush) AND have_royal_flush."""
    step = make_step([
        card("PendingCards", 0, T_, S),
        card("PendingCards", 1, J, S),
        card("PendingCards", 2, Q, S),
        card("PendingCards", 3, K, S),
        card("PendingCards", 4, A, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight_flush", v) == 1.0
    # CRITICAL: royal flush is NOT in the make-* group; only the have flag fires.
    assert get("selected_cards_have_royal_flush", v) == 1.0
    assert get("selected_cards_have_straight_flush", v) == 1.0
    # Face count: J, Q, K = 3 faces (T and A are not faces by default).
    assert get("selected_face_card_scored_count", v) == 3.0
    assert get("selected_ace_scored_count", v) == 1.0


def test_five_of_a_kind() -> None:
    """K,K,K,K,K (mixed suits) → five_kind. Not flush_five (suits not all same)."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, K, C),
        card("PendingCards", 4, K, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_five_kind", v) == 1.0
    assert get("selected_cards_make_flush_five", v) == 0.0
    assert get("selected_face_card_scored_count", v) == 5.0


def test_flush_house() -> None:
    """K,K,K,2,2 all hearts → flush_house. All 5 score."""
    step = make_step([
        card("PendingCards", 0, K, H),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, K, H),
        card("PendingCards", 3, R2, H),
        card("PendingCards", 4, R2, H),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush_house", v) == 1.0
    assert get("selected_cards_make_full_house", v) == 0.0
    assert get("selected_cards_make_flush", v) == 0.0
    assert get("selected_heart_scored_count", v) == 5.0


def test_flush_five() -> None:
    """K,K,K,K,K all spades → flush_five (highest tier)."""
    step = make_step([
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, S),
        card("PendingCards", 2, K, S),
        card("PendingCards", 3, K, S),
        card("PendingCards", 4, K, S),
    ])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush_five", v) == 1.0
    assert get("selected_cards_make_five_kind", v) == 0.0
    assert get("selected_face_card_scored_count", v) == 5.0


# ---------------------------------------------------------------------------
# Joker-effect tests
# ---------------------------------------------------------------------------

JOKER_FOUR_FINGERS = 133
JOKER_PAREIDOLIA = 174
JOKER_SHORTCUT = 195
JOKER_SMEARED = 198
JOKER_SPLASH = 202


def test_four_fingers_enables_4card_flush() -> None:
    """4 spades + 1 heart → flush only with four_fingers."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R9, S),
        card("PendingCards", 2, R7, S),
        card("PendingCards", 3, R3, S),
        card("PendingCards", 4, R2, H),
    ]
    # Without four_fingers: should NOT be flush.
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 0.0
    assert get("selected_cards_have_flush", v) == 0.0
    # With four_fingers undebuffed:
    step = make_step(pending, jokers=[joker(0, JOKER_FOUR_FINGERS)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 1.0
    assert get("selected_cards_have_flush", v) == 1.0
    # Only the 4 scored cards count.
    assert get("selected_spade_scored_count", v) == 4.0
    assert get("has_four_fingers", v) == 1.0


def test_four_fingers_debuffed_does_not_help() -> None:
    """A debuffed four_fingers joker should NOT enable the 4-card flush."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R9, S),
        card("PendingCards", 2, R7, S),
        card("PendingCards", 3, R3, S),
        card("PendingCards", 4, R2, H),
    ]
    step = make_step(pending,
                     jokers=[joker(0, JOKER_FOUR_FINGERS, is_debuffed=True)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 0.0
    assert get("has_four_fingers", v) == 0.0  # mask only counts undebuffed.


def test_shortcut_enables_gap_straight() -> None:
    """10,8,6,5,3 → straight only with shortcut."""
    pending = [
        card("PendingCards", 0, T_, S),
        card("PendingCards", 1, R8, H),
        card("PendingCards", 2, R6, D),
        card("PendingCards", 3, R5, C),
        card("PendingCards", 4, R3, S),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight", v) == 0.0
    step = make_step(pending, jokers=[joker(0, JOKER_SHORTCUT)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight", v) == 1.0
    assert get("has_shortcut", v) == 1.0


def test_smeared_treats_hearts_and_diamonds_as_one_suit() -> None:
    """3 hearts + 2 diamonds → flush only with smeared joker."""
    pending = [
        card("PendingCards", 0, K, H),
        card("PendingCards", 1, R9, H),
        card("PendingCards", 2, R7, H),
        card("PendingCards", 3, R5, D),
        card("PendingCards", 4, R2, D),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 0.0
    step = make_step(pending, jokers=[joker(0, JOKER_SMEARED)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 1.0
    # Counters still use BASE suit (not the smeared grouping).
    assert get("selected_heart_scored_count", v) == 3.0
    assert get("selected_diamond_scored_count", v) == 2.0
    assert get("has_smeared_joker", v) == 1.0


def test_pareidolia_turns_all_cards_into_face_cards() -> None:
    """A,2,3,4,5 with pareidolia → 5 face cards scored (in the straight)."""
    pending = [
        card("PendingCards", 0, A, S),
        card("PendingCards", 1, R2, S),
        card("PendingCards", 2, R3, S),
        card("PendingCards", 3, R4, S),
        card("PendingCards", 4, R5, S),
    ]
    step = make_step(pending, jokers=[joker(0, JOKER_PAREIDOLIA)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_straight_flush", v) == 1.0
    assert get("selected_face_card_scored_count", v) == 5.0
    assert get("has_pareidolia", v) == 1.0


def test_splash_makes_all_cards_score() -> None:
    """High Card scenario with splash joker → all 5 cards score, not just one."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R7, H),
        card("PendingCards", 2, R5, D),
        card("PendingCards", 3, R3, C),
        card("PendingCards", 4, R2, S),
    ]
    step = make_step(pending, jokers=[joker(0, JOKER_SPLASH)])
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_high_card", v) == 1.0
    # All 5 score under splash.
    n_scored = (
        get("selected_spade_scored_count", v)
        + get("selected_heart_scored_count", v)
        + get("selected_diamond_scored_count", v)
        + get("selected_club_scored_count", v)
    )
    assert n_scored == 5.0
    assert get("has_splash", v) == 1.0


# ---------------------------------------------------------------------------
# Wild + Stone card tests
# ---------------------------------------------------------------------------

def test_wild_card_completes_flush() -> None:
    """4 spades + 1 wild heart → flush (wild counts as any suit)."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R9, S),
        card("PendingCards", 2, R7, S),
        card("PendingCards", 3, R3, S),
        card("PendingCards", 4, R2, H, modifier="m_wild"),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 1.0
    # Suit counts use BASE suit; the wild card is still a heart printed.
    assert get("selected_spade_scored_count", v) == 4.0
    assert get("selected_heart_scored_count", v) == 1.0


def test_wild_card_debuffed_reverts_to_base_suit() -> None:
    """4 spades + 1 debuffed wild heart → no flush."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, R9, S),
        card("PendingCards", 2, R7, S),
        card("PendingCards", 3, R3, S),
        card("PendingCards", 4, R2, H, modifier="m_wild", is_debuffed=True),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_flush", v) == 0.0
    # And the debuff count should pick up the debuffed wild even though it
    # didn't score (when make-type doesn't include it). For high_card the
    # only scored card is K♠, so debuff_count is 0 here.
    assert get("selected_scored_debuff_count", v) == 0.0


def test_stone_card_always_scores_under_pair() -> None:
    """Stone + K,K,7,2 → pair (2 Ks score) + 1 stone always scores."""
    pending = [
        card("PendingCards", 0, R2, S, modifier="m_stone"),
        card("PendingCards", 1, K, H),
        card("PendingCards", 2, K, D),
        card("PendingCards", 3, R7, C),
        card("PendingCards", 4, R2, S),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_pair", v) == 1.0
    # 2 kings score (faces) + 1 stone (no rank, contributes only to debuff).
    assert get("selected_face_card_scored_count", v) == 2.0
    # Stone has no rank, so 2-scored-count is unaffected by it.
    assert get("selected_2_scored_count", v) == 0.0
    # Stone has no suit, so suit counters are unaffected.
    spades_h_d_c = (
        get("selected_spade_scored_count", v)
        + get("selected_heart_scored_count", v)
        + get("selected_diamond_scored_count", v)
        + get("selected_club_scored_count", v)
    )
    assert spades_h_d_c == 2.0
    # Stone is leftmost (position 0); the stone is NOT a face card.
    assert get("selected_position_0_scored_is_a_face_card", v) == 0.0


def test_debuffed_stone_counts_in_scored_debuff_count() -> None:
    """Debuffed stone contributes to selected_scored_debuff_count."""
    pending = [
        card("PendingCards", 0, R2, S, modifier="m_stone", is_debuffed=True),
        card("PendingCards", 1, K, H),
    ]
    step = make_step(pending)
    v = compute_supplement_features(step, {})
    assert get("selected_cards_make_high_card", v) == 1.0
    assert get("selected_scored_debuff_count", v) == 1.0


# ---------------------------------------------------------------------------
# Empty PendingCards
# ---------------------------------------------------------------------------

def test_empty_pending_cards_is_all_zero_selected() -> None:
    """No PendingCards → every selected_* feature is 0."""
    step = make_step([])
    v = compute_supplement_features(step, {})
    for name in SUPPLEMENT_FEATURE_NAMES:
        if name.startswith("selected_"):
            assert v[SUPPLEMENT_FEATURE_NAMES.index(name)] == 0.0, (
                f"{name} expected 0 with no PendingCards, got {v[SUPPLEMENT_FEATURE_NAMES.index(name)]}"
            )


# ---------------------------------------------------------------------------
# Game-state flags
# ---------------------------------------------------------------------------

def test_game_state_flags() -> None:
    """hands_left==1 → final_hand; dollars<=4 → money flag; ante==1 → is_ante_1."""
    step = make_step(
        [card("PendingCards", 0, K, S)],
        state={"hands_left": 1, "dollars": 3, "ante": 1},
    )
    v = compute_supplement_features(step, {})
    assert get("is_final_hand_of_round", v) == 1.0
    assert get("money_less_than_or_equal_to_4", v) == 1.0
    assert get("is_ante_1", v) == 1.0
    # And conversely.
    step = make_step(
        [card("PendingCards", 0, K, S)],
        state={"hands_left": 3, "dollars": 100, "ante": 5},
    )
    v = compute_supplement_features(step, {})
    assert get("is_final_hand_of_round", v) == 0.0
    assert get("money_less_than_or_equal_to_4", v) == 0.0
    assert get("is_ante_1", v) == 0.0


# ---------------------------------------------------------------------------
# Hand-history flags via persistent_state
# ---------------------------------------------------------------------------

def test_already_played_this_round_and_most_played() -> None:
    """already_played_this_round and is_most_played_hand both light up correctly."""
    pending = [
        card("PendingCards", 0, K, S),
        card("PendingCards", 1, K, H),
    ]
    step = make_step(pending)
    pstate = {
        "hands": {
            "Pair":           {"level": 3, "played": 5, "played_this_round": 1},
            "High Card":      {"level": 1, "played": 2, "played_this_round": 0},
            "Two Pair":       {"level": 1, "played": 0, "played_this_round": 0},
            "Flush":          {"level": 1, "played": 1, "played_this_round": 0},
        },
    }
    v = compute_supplement_features(step, pstate)
    assert get("selected_cards_make_pair", v) == 1.0
    assert get("selected_hand_type_already_played_this_round", v) == 1.0
    assert get("selected_hand_type_is_most_played_hand", v) == 1.0
    # If we change to high_card make-type, neither should fire.
    step = make_step([card("PendingCards", 0, K, S)])
    v = compute_supplement_features(step, pstate)
    assert get("selected_cards_make_high_card", v) == 1.0
    assert get("selected_hand_type_already_played_this_round", v) == 0.0
    assert get("selected_hand_type_is_most_played_hand", v) == 0.0


# ---------------------------------------------------------------------------
# Held-card counters (CurrentHand)
# ---------------------------------------------------------------------------

def test_held_counts_basic() -> None:
    """held_* counters reflect CurrentHand contents (independent of PendingCards)."""
    pending = [card("PendingCards", 0, A, S)]
    held = [
        card("CurrentHand", 0, K, S),
        card("CurrentHand", 1, K, C),
        card("CurrentHand", 2, Q, H),
        card("CurrentHand", 3, R2, D),
    ]
    step = make_step(pending, held=held)
    v = compute_supplement_features(step, {})
    assert get("held_king_count", v) == 2.0
    assert get("held_queen_count", v) == 1.0
    assert get("held_face_card_count", v) == 3.0  # 2K + 1Q
    assert get("held_spade_count", v) == 1.0       # K♠
    assert get("held_club_count", v) == 1.0        # K♣
    assert get("held_spade_or_club_count", v) == 2.0
    # Lowest rank is 2 (rank_value 2); leftmost-tied is just the 2.
    assert get("held_lowest_rank_value", v) == 2.0
    assert get("held_lowest_rank_is_debuffed", v) == 0.0


def test_held_face_count_with_pareidolia() -> None:
    """Pareidolia → all non-stone held cards count as face."""
    held = [
        card("CurrentHand", 0, R2, S),
        card("CurrentHand", 1, R5, H),
        card("CurrentHand", 2, R9, D),
    ]
    step = make_step([], held=held, jokers=[joker(0, JOKER_PAREIDOLIA)])
    v = compute_supplement_features(step, {})
    assert get("held_face_card_count", v) == 3.0


def test_held_lowest_rank_with_debuffed_card() -> None:
    """Tie-broken on position; lowest 2 is debuffed."""
    held = [
        card("CurrentHand", 0, R2, S, is_debuffed=True),
        card("CurrentHand", 1, R2, H),  # tie on rank, later position
        card("CurrentHand", 2, K, D),
    ]
    step = make_step([], held=held)
    v = compute_supplement_features(step, {})
    assert get("held_lowest_rank_value", v) == 2.0
    assert get("held_lowest_rank_is_debuffed", v) == 1.0


# ---------------------------------------------------------------------------
# Sanity / shape
# ---------------------------------------------------------------------------

def test_supplement_vector_shape_and_dtype() -> None:
    v = compute_supplement_features({"objects": [], "state": {}}, {})
    assert v.shape == (N_SUPPLEMENT,), f"expected ({N_SUPPLEMENT},), got {v.shape}"
    assert v.dtype == np.float32


def test_feature_name_count_matches() -> None:
    assert len(SUPPLEMENT_FEATURE_NAMES) == N_SUPPLEMENT
    # All names must be unique.
    assert len(set(SUPPLEMENT_FEATURE_NAMES)) == N_SUPPLEMENT


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    import traceback
    funcs = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed: list[tuple[str, str]] = []
    for name, fn in funcs:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
    print()
    print(f"{passed}/{len(funcs)} tests passed")
    if failed:
        print()
        for name, tb in failed:
            print(f"--- FAIL: {name} ---")
            print(tb)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
