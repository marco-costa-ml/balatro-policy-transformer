"""Tests for the parent-start family / pointer masks in mask_builder.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from action_map import load_action_map
from family_map import compute_family_map
from mask_builder import (
    build_action_mask,
    build_card_pointer_mask,
    build_family_mask,
    build_item_pointer_mask,
    build_swap_joker_mask,
    resolve_card_zone,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def action_map() -> dict[str, Any]:
    return load_action_map(_REPO_ROOT / "data" / "action_map.json")


@pytest.fixture(scope="module")
def family_map(action_map: dict[str, Any]) -> dict[str, Any]:
    return compute_family_map(action_map)


def _hand_card(slot_id: int, pos: int, *, seal: str | None = None, edition: str | None = None) -> dict[str, Any]:
    return {
        "zone": "CurrentHand",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": 1,
        "object_type": "playing_card",
        "seal": seal,
        "edition": edition,
        "rank_index": 5,
        "suit_index": 1,
        "modifier_id": None,
        "is_debuffed": False,
    }


def _consumable(slot_id: int, pos: int, class_id: int) -> dict[str, Any]:
    return {
        "zone": "CurrentConsumables",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": class_id,
        "object_type": "tarot",
    }


def _joker(slot_id: int, pos: int) -> dict[str, Any]:
    return {
        "zone": "CurrentJokers",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": 100 + slot_id,
        "object_type": "joker",
    }


def _pack_item(slot_id: int, pos: int, class_id: int = 298) -> dict[str, Any]:
    return {
        "zone": "PackOfferings",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": class_id,
        "object_type": "tarot",
    }


def _tarot_hand_card(slot_id: int, pos: int) -> dict[str, Any]:
    return {
        "zone": "TarotSpectralHand",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": 1,
        "object_type": "playing_card",
        "rank_index": 5,
        "suit_index": 1,
    }


def _shop_voucher(slot_id: int, pos: int) -> dict[str, Any]:
    return {
        "zone": "VoucherShopOfferings",
        "slot_id": slot_id,
        "position_in_zone": pos,
        "class_id": 320 + slot_id,
        "object_type": "voucher",
    }


# --------------------------------------------------------------------------- #
# build_family_mask: parent-start legality per family
# --------------------------------------------------------------------------- #


def test_family_mask_play_hand_legal_at_empty_pending(action_map, family_map) -> None:
    """At the first SelectCard step of a PlayHand decision, family_mask[PlayHand] must be True
    even though pending_cards == [] and v1's flat PlayHand bit is decided by page-gate only."""
    step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_hand_card(slot_id=10, pos=0), _hand_card(slot_id=11, pos=1)],
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    play_id = family_map["family_to_id"]["PlayHand"]
    discard_id = family_map["family_to_id"]["DiscardHand"]
    assert mask[play_id], "PlayHand must be parent-start legal on In_Blind with cards in hand"
    assert mask[discard_id], "DiscardHand must be parent-start legal on In_Blind with cards in hand"
    # SWAP not legal: zero jokers
    assert not mask[family_map["family_to_id"]["SWAP"]]


def test_family_mask_play_hand_illegal_on_non_blind_page(action_map, family_map) -> None:
    step = {
        "page_name": "In_Shop",
        "pending_cards": [],
        "objects": [],
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    assert not mask[family_map["family_to_id"]["PlayHand"]]
    assert not mask[family_map["family_to_id"]["DiscardHand"]]
    # Shop families gated by page should be legal once their zones are populated.
    assert not mask[family_map["family_to_id"]["BuyShopItem_VoucherShopOfferings"]], (
        "no shop items in objects -> no flat mask bits -> family illegal"
    )
    assert mask[family_map["family_to_id"]["LeaveShop"]]


def test_family_mask_buy_shop_legal_when_item_available(action_map, family_map) -> None:
    step = {
        "page_name": "In_Shop",
        "pending_cards": [],
        "objects": [_shop_voucher(slot_id=1, pos=0)],
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    assert mask[family_map["family_to_id"]["BuyShopItem_VoucherShopOfferings"]]
    assert mask[family_map["family_to_id"]["LeaveShop"]]


def test_family_mask_use_consumable_legal_when_consumable_present(action_map, family_map) -> None:
    step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_consumable(slot_id=5, pos=0, class_id=298)],  # c_chariot
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    assert mask[family_map["family_to_id"]["UseConsumable_CurrentConsumables"]]


def test_family_mask_select_pack_item_legal_on_pack_page(action_map, family_map) -> None:
    step = {
        "page_name": "In_TarotSpectral_Pack",
        "pending_cards": [],
        "objects": [_pack_item(slot_id=1, pos=0), _pack_item(slot_id=2, pos=1)],
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    assert mask[family_map["family_to_id"]["SelectPackItem_PackOfferings"]]


def test_family_mask_swap_requires_two_jokers(action_map, family_map) -> None:
    base = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_joker(slot_id=1, pos=0)],
        "state": {"jokers_current": 1},
    }
    mask_one = build_family_mask(base, action_map, family_map)
    assert not mask_one[family_map["family_to_id"]["SWAP"]]

    two_jokers = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_joker(slot_id=1, pos=0), _joker(slot_id=2, pos=1)],
        "state": {"jokers_current": 2},
    }
    mask_two = build_family_mask(two_jokers, action_map, family_map)
    assert mask_two[family_map["family_to_id"]["SWAP"]]


def test_family_mask_start_new_run_never_set(action_map, family_map) -> None:
    step = {"page_name": "In_Blind", "objects": [], "state": {"jokers_current": 0}}
    mask = build_family_mask(step, action_map, family_map)
    assert not mask[family_map["family_to_id"]["StartNewRun"]]


def test_family_mask_at_least_one_legal_on_any_realistic_step(action_map, family_map) -> None:
    # A snapshot mid-blind with a card in hand should have a non-zero family mask.
    step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_hand_card(slot_id=1, pos=0)],
        "state": {"jokers_current": 0},
    }
    mask = build_family_mask(step, action_map, family_map)
    assert int(mask.sum()) >= 1


# --------------------------------------------------------------------------- #
# build_item_pointer_mask
# --------------------------------------------------------------------------- #


def test_item_pointer_mask_use_consumable(action_map, family_map) -> None:
    step = {
        "page_name": "In_Blind",
        "objects": [
            _consumable(slot_id=10, pos=0, class_id=298),
            _consumable(slot_id=11, pos=1, class_id=299),
        ],
        "state": {"jokers_current": 0},
    }
    mask = build_item_pointer_mask(
        "UseConsumable_CurrentConsumables", step, action_map, max_size=4
    )
    assert mask.dtype == bool
    assert mask.shape == (4,)
    assert mask[0] and mask[1]
    assert not mask[2] and not mask[3]


def test_item_pointer_mask_bare_family_is_all_false(action_map, family_map) -> None:
    step = {"page_name": "In_Blind", "objects": [], "state": {"jokers_current": 0}}
    for fam in ("PlayHand", "DiscardHand", "CashOut", "SWAP", "StartNewRun"):
        mask = build_item_pointer_mask(fam, step, action_map, max_size=4)
        assert mask.shape == (4,)
        assert not mask.any(), f"family {fam} should have no item pointer mask"


def test_item_pointer_mask_select_pack_item_filters_to_available(action_map, family_map) -> None:
    step = {
        "page_name": "In_TarotSpectral_Pack",
        "objects": [_pack_item(slot_id=1, pos=0), _pack_item(slot_id=2, pos=2)],
        "state": {"jokers_current": 0},
    }
    mask = build_item_pointer_mask(
        "SelectPackItem_PackOfferings", step, action_map, max_size=6
    )
    # positions 0 and 2 valid; 1 invalid (no object there).
    assert mask[0] and mask[2]
    assert not mask[1]


# --------------------------------------------------------------------------- #
# build_card_pointer_mask
# --------------------------------------------------------------------------- #


def test_card_pointer_mask_play_hand(action_map) -> None:
    step = {
        "page_name": "In_Blind",
        "objects": [_hand_card(slot_id=1, pos=0), _hand_card(slot_id=2, pos=2)],
        "state": {"jokers_current": 0},
    }
    mask = build_card_pointer_mask("PlayHand", step, action_map, max_size=15)
    assert mask[0] and mask[2]
    assert not mask[1]


def test_card_pointer_mask_use_consumable_uses_currenthand_in_blind(action_map) -> None:
    step = {
        "page_name": "In_Blind",
        "objects": [_hand_card(slot_id=1, pos=0)],
        "state": {"jokers_current": 0},
    }
    mask = build_card_pointer_mask(
        "UseConsumable_CurrentConsumables", step, action_map, max_size=15
    )
    assert mask[0]
    assert not mask[1]


def test_card_pointer_mask_use_consumable_uses_tarotspectral_in_pack(action_map) -> None:
    step = {
        "page_name": "In_TarotSpectral_Pack",
        "objects": [_tarot_hand_card(slot_id=10, pos=0), _tarot_hand_card(slot_id=11, pos=1)],
        "state": {"jokers_current": 0},
    }
    mask = build_card_pointer_mask(
        "UseConsumable_CurrentConsumables", step, action_map, max_size=15
    )
    assert mask[0] and mask[1]


def test_card_pointer_mask_select_pack_item_only_with_tarot_subtype(action_map) -> None:
    base_step = {
        "page_name": "In_TarotSpectral_Pack",
        "objects": [_tarot_hand_card(slot_id=10, pos=0)],
        "state": {"jokers_current": 0},
    }
    tarot_step = {**base_step, "source_action_subtype": "selectpackitemtarot"}
    mask_tarot = build_card_pointer_mask(
        "SelectPackItem_PackOfferings", tarot_step, action_map, max_size=15
    )
    assert mask_tarot[0]

    planet_step = {**base_step, "source_action_subtype": "selectpackitemplanet"}
    mask_planet = build_card_pointer_mask(
        "SelectPackItem_PackOfferings", planet_step, action_map, max_size=15
    )
    assert not mask_planet.any(), "non-tarot pack items should not have a card pool"


def test_card_pointer_mask_bare_or_no_card_families_are_empty(action_map) -> None:
    step = {"page_name": "In_Blind", "objects": [], "state": {"jokers_current": 0}}
    for fam in ("CashOut", "LeaveShop", "SkipPack", "BuyShopItem_VoucherShopOfferings"):
        mask = build_card_pointer_mask(fam, step, action_map, max_size=15)
        assert not mask.any(), f"family {fam} should have no card pointer mask"


# --------------------------------------------------------------------------- #
# build_swap_joker_mask
# --------------------------------------------------------------------------- #


def test_swap_joker_mask_marks_first_n_positions(action_map) -> None:
    step = {
        "page_name": "In_Blind",
        "objects": [_joker(slot_id=1, pos=0), _joker(slot_id=2, pos=1), _joker(slot_id=3, pos=2)],
        "state": {"jokers_current": 3},
    }
    mask = build_swap_joker_mask(step, max_joker_slots=10)
    assert mask.shape == (10,)
    assert mask[:3].all()
    assert not mask[3:].any()


def test_swap_joker_mask_zero_when_no_jokers(action_map) -> None:
    step = {"page_name": "In_Blind", "objects": [], "state": {"jokers_current": 0}}
    mask = build_swap_joker_mask(step, max_joker_slots=10)
    assert not mask.any()


# --------------------------------------------------------------------------- #
# resolve_card_zone
# --------------------------------------------------------------------------- #


def test_resolve_card_zone_dispatch() -> None:
    assert resolve_card_zone("PlayHand", {"page_name": "In_Blind"}) == "CurrentHand"
    assert resolve_card_zone("DiscardHand", {"page_name": "In_Blind"}) == "CurrentHand"
    assert (
        resolve_card_zone(
            "UseConsumable_CurrentConsumables", {"page_name": "In_Blind"}
        )
        == "CurrentHand"
    )
    assert (
        resolve_card_zone(
            "UseConsumable_CurrentConsumables",
            {"page_name": "In_TarotSpectral_Pack"},
        )
        == "TarotSpectralHand"
    )
    assert (
        resolve_card_zone(
            "SelectPackItem_PackOfferings",
            {"source_action_subtype": "selectpackitemtarot"},
        )
        == "TarotSpectralHand"
    )
    assert resolve_card_zone(
        "SelectPackItem_PackOfferings",
        {"source_action_subtype": "selectpackitemplanet"},
    ) is None
    assert resolve_card_zone("CashOut", {}) is None
    assert resolve_card_zone("BuyShopItem_VoucherShopOfferings", {}) is None


# --------------------------------------------------------------------------- #
# v1 flat mask stays untouched
# --------------------------------------------------------------------------- #


def test_build_action_mask_still_works(action_map) -> None:
    step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": [_hand_card(slot_id=1, pos=0)],
        "state": {"jokers_current": 0},
    }
    mask = build_action_mask(step, action_map)
    assert mask.shape == (int(action_map["n_actions"]),)
    # PlayHand bit should be set (page gate only).
    play_offset = int(action_map["family_offsets"]["PlayHand"])
    assert mask[play_offset]
