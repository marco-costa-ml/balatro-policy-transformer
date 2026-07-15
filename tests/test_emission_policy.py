#!/usr/bin/env python3
"""
test_emission_policy.py
=======================
Unit tests for ``live/emission_policy.py``. Verifies that:

- ``expand_decision`` produces the canonical IPC label sequence for
  every decoder shape (no_args / card_seq / single_ptr / chained_cards
  / joker_pair) and is robust to malformed inputs.
- ``EmissionPolicy`` correctly drains a planned label queue across
  ticks and self-invalidates on page-change / illegal-label conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from live.emission_policy import (
    EmissionPolicy,
    card_zone_for_emission,
    expand_decision,
    original_to_dynamic_pool_indices,
)


# ---------------------------------------------------------------------------
# original_to_dynamic_pool_indices
# ---------------------------------------------------------------------------


def test_dyn_translation_step_zero_is_identity():
    """The first pick is always against the full pool, so dyn idx == p_0."""
    assert original_to_dynamic_pool_indices([3]) == [3]


def test_dyn_translation_ascending_picks_collapse_to_leading_zeros():
    """Picking the first 5 original positions of an 8-card hand pulls the
    leftmost remaining card every time -> every dyn idx is 0."""
    assert original_to_dynamic_pool_indices([0, 1, 2, 3, 4]) == [0, 0, 0, 0, 0]


def test_dyn_translation_descending_picks_unchanged():
    """No earlier pick is smaller than the current one -> no shift."""
    assert original_to_dynamic_pool_indices([7, 5, 1]) == [7, 5, 1]


def test_dyn_translation_mixed_ordering():
    """[3, 0, 5, 2]:
      - 3 -> 3 (nothing earlier)
      - 0 -> 0 (nothing < 0)
      - 5 -> 5 - 2 = 3  (3 and 0 are both < 5)
      - 2 -> 2 - 1 = 1  (only 0 is < 2; 3 and 5 are not)
    """
    assert original_to_dynamic_pool_indices([3, 0, 5, 2]) == [3, 0, 3, 1]


def test_dyn_translation_preserves_negative_padding():
    """Padding entries (-1) pass through unchanged so the caller can drop them."""
    assert original_to_dynamic_pool_indices([2, -1, 3, -1]) == [2, -1, 2, -1]


def test_dyn_translation_empty_sequence():
    assert original_to_dynamic_pool_indices([]) == []


# ---------------------------------------------------------------------------
# card_zone_for_emission
# ---------------------------------------------------------------------------


def test_card_zone_for_play_discard_is_current_hand():
    assert card_zone_for_emission("PlayHand", "In_Blind") == "CurrentHand"
    assert card_zone_for_emission("DiscardHand", "In_Blind") == "CurrentHand"


def test_card_zone_for_use_consumable_follows_page():
    assert (
        card_zone_for_emission("UseConsumable_CurrentConsumables", "In_Blind")
        == "CurrentHand"
    )
    assert (
        card_zone_for_emission(
            "UseConsumable_CurrentConsumables", "In_TarotSpectral_Pack"
        )
        == "TarotSpectralHand"
    )


def test_card_zone_for_select_pack_item_is_tarot_hand():
    # SelectPackItem only emits SelectCard labels in tarot/spectral packs;
    # for joker/planet/standard packs num_cards is 0 so the zone is unused.
    assert (
        card_zone_for_emission("SelectPackItem_PackOfferings", "In_TarotSpectral_Pack")
        == "TarotSpectralHand"
    )


# ---------------------------------------------------------------------------
# expand_decision
# ---------------------------------------------------------------------------


def test_expand_no_args_emits_single_label():
    result = expand_decision(
        family_name="SelectBlind",
        decoder_shape="no_args",
        page_name="Blind_Select",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == ["SelectBlind"]
    assert result.family_name == "SelectBlind"
    assert result.decoder_shape == "no_args"


def test_expand_card_seq_orders_selects_before_commit():
    """Card-seq picks are translated from original positions to the
    dynamic-pool indices Lua expects. For picks [2, 5, 0]:
      - p=2: nothing earlier was smaller -> dyn=2.
      - p=5: one earlier pick (2) was smaller -> dyn=4.
      - p=0: nothing earlier was smaller (0 isn't > 0) -> dyn=0.
    """
    result = expand_decision(
        family_name="PlayHand",
        decoder_shape="card_seq",
        page_name="In_Blind",
        num_cards=3,
        card_ptr_local_seq=[2, 5, 0, -1, -1],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_CurrentHand_2",
        "SelectCard_CurrentHand_4",
        "SelectCard_CurrentHand_0",
        "PlayHand",
    ]


def test_expand_card_seq_truncates_negative_pointers():
    """Padded -1 pointer slots get skipped rather than emitted.

    Translation still runs on the non-negative picks: with [1, -1, 3, -1]
    the legit picks are [1, 3] and 1 < 3, so dyn picks are [1, 2].
    """
    result = expand_decision(
        family_name="DiscardHand",
        decoder_shape="card_seq",
        page_name="In_Blind",
        num_cards=4,
        card_ptr_local_seq=[1, -1, 3, -1],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_CurrentHand_1",
        "SelectCard_CurrentHand_2",
        "DiscardHand",
    ]


def test_expand_card_seq_five_of_eight_stays_in_range():
    """Regression for the live-test bug: picking the first 5 original
    positions out of an 8-card hand used to emit ``SelectCard_4`` as
    the 5th label, which is out of range once Lua's pool has shrunk
    to 4 cards. With the translation, the same model output unrolls
    to five ``SelectCard_CurrentHand_0`` labels — each pulls the
    leftmost remaining card, which is exactly what the original-position
    picks imply.
    """
    result = expand_decision(
        family_name="PlayHand",
        decoder_shape="card_seq",
        page_name="In_Blind",
        num_cards=5,
        card_ptr_local_seq=[0, 1, 2, 3, 4],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_CurrentHand_0",
        "SelectCard_CurrentHand_0",
        "SelectCard_CurrentHand_0",
        "SelectCard_CurrentHand_0",
        "SelectCard_CurrentHand_0",
        "PlayHand",
    ]


def test_expand_card_seq_descending_picks_no_shift():
    """When picks go in strictly decreasing order none of the earlier
    picks is smaller than the current one, so the dyn-pool index
    equals the original position for every step.
    """
    result = expand_decision(
        family_name="PlayHand",
        decoder_shape="card_seq",
        page_name="In_Blind",
        num_cards=3,
        card_ptr_local_seq=[7, 5, 1],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_CurrentHand_7",
        "SelectCard_CurrentHand_5",
        "SelectCard_CurrentHand_1",
        "PlayHand",
    ]


def test_expand_single_ptr_emits_one_label():
    result = expand_decision(
        family_name="BuyShopItem_VoucherShopOfferings",
        decoder_shape="single_ptr",
        page_name="In_Shop",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=1,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == ["BuyShopItem_VoucherShopOfferings_1"]


def test_expand_single_ptr_with_negative_item_returns_empty():
    result = expand_decision(
        family_name="BuyShopItem_VoucherShopOfferings",
        decoder_shape="single_ptr",
        page_name="In_Shop",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=-1,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == []


def test_expand_chained_cards_emits_selects_then_use_consumable():
    result = expand_decision(
        family_name="UseConsumable_CurrentConsumables",
        decoder_shape="chained_cards",
        page_name="In_Blind",
        num_cards=2,
        card_ptr_local_seq=[3, 1],
        item_ptr_local=0,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_CurrentHand_3",
        "SelectCard_CurrentHand_1",
        "UseConsumable_CurrentConsumables_0",
    ]


def test_expand_chained_cards_no_cards_is_just_commit():
    result = expand_decision(
        family_name="UseConsumable_CurrentConsumables",
        decoder_shape="chained_cards",
        page_name="In_Blind",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=2,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == ["UseConsumable_CurrentConsumables_2"]


def test_expand_chained_cards_in_pack_uses_tarot_hand():
    result = expand_decision(
        family_name="UseConsumable_CurrentConsumables",
        decoder_shape="chained_cards",
        page_name="In_TarotSpectral_Pack",
        num_cards=1,
        card_ptr_local_seq=[0],
        item_ptr_local=1,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == [
        "SelectCard_TarotSpectralHand_0",
        "UseConsumable_CurrentConsumables_1",
    ]


def test_expand_joker_pair_canonicalizes_ordering():
    """SWAP labels must be ``SWAP_<min(i,j)>_<max(i,j)>`` regardless of
    which order the joker_pair decoder picked first."""
    a = expand_decision(
        family_name="SWAP",
        decoder_shape="joker_pair",
        page_name="In_Shop",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=None,
        swap_i_local=3,
        swap_j_local=1,
    )
    b = expand_decision(
        family_name="SWAP",
        decoder_shape="joker_pair",
        page_name="In_Shop",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=None,
        swap_i_local=1,
        swap_j_local=3,
    )
    assert a.labels == ["SWAP_1_3"] == b.labels


def test_expand_joker_pair_rejects_same_index():
    result = expand_decision(
        family_name="SWAP",
        decoder_shape="joker_pair",
        page_name="In_Shop",
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=None,
        swap_i_local=2,
        swap_j_local=2,
    )
    assert result.labels == []


def test_expand_reserved_returns_empty():
    result = expand_decision(
        family_name="StartNewRun",
        decoder_shape="reserved",
        page_name=None,
        num_cards=0,
        card_ptr_local_seq=[],
        item_ptr_local=None,
        swap_i_local=None,
        swap_j_local=None,
    )
    assert result.labels == []


# ---------------------------------------------------------------------------
# EmissionPolicy state machine
# ---------------------------------------------------------------------------


def _make_decision_snapshot(page: str = "In_Blind") -> dict:
    return {"page_name": page, "legal_actions": []}


def _set_plan(policy: EmissionPolicy, labels: list[str], *, page: str = "In_Blind") -> None:
    policy.set_plan(
        labels,
        family_name="PlayHand",
        decoder_shape="card_seq",
        page_name=page,
        decision_snapshot=_make_decision_snapshot(page),
        decision={"family_name": "PlayHand"},
    )


def test_emission_policy_starts_empty():
    policy = EmissionPolicy()
    assert not policy.has_pending()
    assert not policy.finished()
    assert policy.last_committed() is None


def test_emission_policy_drains_in_order():
    policy = EmissionPolicy()
    labels = [
        "SelectCard_CurrentHand_0",
        "SelectCard_CurrentHand_2",
        "PlayHand",
    ]
    _set_plan(policy, labels)
    legal = set(labels)
    emitted: list[str] = []
    for _ in range(len(labels)):
        nxt = policy.pop_next(legal, "In_Blind")
        assert nxt is not None
        emitted.append(nxt)
    assert emitted == labels
    assert policy.finished()
    last = policy.last_committed()
    assert last is not None
    assert last.emitted == labels


def test_emission_policy_invalidates_on_page_change():
    policy = EmissionPolicy()
    _set_plan(policy, ["A", "B"], page="In_Blind")
    legal = {"A", "B"}
    # First poll happens on a different page → plan should be dropped.
    assert policy.pop_next(legal, "In_Shop") is None
    assert policy.invalidated()
    assert not policy.has_pending()


def test_emission_policy_invalidates_on_illegal_next_label():
    policy = EmissionPolicy()
    _set_plan(policy, ["A", "B"])
    # First emit OK.
    assert policy.pop_next({"A", "B"}, "In_Blind") == "A"
    # Now Lua's legal set doesn't include "B" any more.
    assert policy.pop_next({"X"}, "In_Blind") is None
    assert policy.invalidated()
    assert not policy.has_pending()


def test_emission_policy_clear_resets_state():
    policy = EmissionPolicy()
    _set_plan(policy, ["A"])
    policy.clear()
    assert not policy.has_pending()
    assert not policy.invalidated()
    assert policy.last_committed() is None


def test_emission_policy_finished_only_after_full_drain():
    policy = EmissionPolicy()
    _set_plan(policy, ["A", "B"])
    assert not policy.finished()
    assert policy.pop_next({"A", "B"}, "In_Blind") == "A"
    assert not policy.finished()  # still has "B"
    assert policy.pop_next({"A", "B"}, "In_Blind") == "B"
    assert policy.finished()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
