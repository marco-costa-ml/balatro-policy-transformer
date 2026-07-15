"""Tests for the locked family_map and argument_spec artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from action_map import load_action_map
from argument_spec import (
    ARGUMENT_SPEC,
    INVENTORY_GATES,
    MAX_CARDS_PER_DECISION,
    cardinality_for_play_discard,
    cardinality_for_select_pack_item,
    cardinality_for_use_consumable,
    spec_for_class_id,
)
from family_map import (
    DECODER_SHAPES,
    FAMILY_ORDER,
    ITEM_ZONE_FOR_FAMILY,
    compute_family_map,
    family_id_for_step,
    family_name_for_step,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def action_map() -> dict:
    return load_action_map(_REPO_ROOT / "data" / "action_map.json")


@pytest.fixture(scope="module")
def family_map(action_map: dict) -> dict:
    return compute_family_map(action_map)


# --------------------------------------------------------------------------- #
# family_map shape / counts
# --------------------------------------------------------------------------- #


def test_family_order_is_locked_19_families() -> None:
    assert len(FAMILY_ORDER) == 19
    assert FAMILY_ORDER[0] == "StartNewRun"


def test_shape_counts_match_plan() -> None:
    counts: dict[str, int] = {}
    for fam in FAMILY_ORDER:
        counts[DECODER_SHAPES[fam]] = counts.get(DECODER_SHAPES[fam], 0) + 1
    assert counts == {
        "reserved": 1,
        "no_args": 7,
        "card_seq": 2,
        "single_ptr": 6,
        "chained_cards": 2,
        "joker_pair": 1,
    }


def test_every_family_has_a_decoder_shape_and_item_zone_entry() -> None:
    for fam in FAMILY_ORDER:
        assert fam in DECODER_SHAPES, f"missing DECODER_SHAPES entry for {fam}"
        assert fam in ITEM_ZONE_FOR_FAMILY, f"missing ITEM_ZONE entry for {fam}"


def test_single_ptr_and_chained_cards_have_item_zones(family_map: dict) -> None:
    for fam in FAMILY_ORDER:
        shape = family_map["decoder_shapes"][fam]
        zone = family_map["item_zone_for_family"][fam]
        if shape in {"single_ptr", "chained_cards", "joker_pair"}:
            assert zone is not None, f"{fam} has shape={shape} but item_zone is None"
        elif shape in {"no_args", "card_seq", "reserved"}:
            assert zone is None, f"{fam} has shape={shape} but item_zone={zone!r}"


def test_family_map_version_is_stable(family_map: dict) -> None:
    assert family_map["family_map_version"].startswith("fv1.")
    assert family_map["n_families"] == 19


def test_play_discard_card_zone_is_currenthand(family_map: dict) -> None:
    for fam in ("PlayHand", "DiscardHand"):
        assert family_map["default_card_zone_for_family"][fam] == "CurrentHand"


# --------------------------------------------------------------------------- #
# family_id_for_step
# --------------------------------------------------------------------------- #


def test_family_id_for_bare_step(family_map: dict) -> None:
    step = {"source_action": "PlayHand"}
    assert family_id_for_step(step, family_map) == family_map["family_to_id"]["PlayHand"]


def test_family_id_for_indexed_step(family_map: dict) -> None:
    step = {
        "source_action": "BuyShopItem",
        "target_zone": "VoucherShopOfferings",
    }
    assert family_name_for_step(step) == "BuyShopItem_VoucherShopOfferings"
    expected = family_map["family_to_id"]["BuyShopItem_VoucherShopOfferings"]
    assert family_id_for_step(step, family_map) == expected


def test_family_id_for_swap_step(family_map: dict) -> None:
    step = {"source_action": "SWAP", "swap_pair": [0, 1]}
    assert family_id_for_step(step, family_map) == family_map["family_to_id"]["SWAP"]


def test_swap_synth_inherits_parent_source_action_but_is_swap_family(family_map: dict) -> None:
    # Granularize emits swap_synth steps with source_action == parent
    # (e.g. PlayHand). The family map MUST classify them as SWAP, not
    # PlayHand, to avoid mis-folding them into the parent's super-step.
    step = {
        "source_kind": "swap_synth",
        "source_action": "PlayHand",
        "swap_pair": [0, 2],
        "action": "SWAP_0_2",
    }
    assert family_name_for_step(step) == "SWAP"
    assert family_id_for_step(step, family_map) == family_map["family_to_id"]["SWAP"]

    # Same for LeaveShop-parented swap_synths.
    step2 = {
        "source_kind": "swap_synth",
        "source_action": "LeaveShop",
        "swap_pair": [0, 1],
        "action": "SWAP_0_1",
    }
    assert family_name_for_step(step2) == "SWAP"


def test_family_id_for_start_new_run_is_minus_one(family_map: dict) -> None:
    assert family_id_for_step({"source_action": "StartNewRun"}, family_map) == -1


def test_family_id_for_unknown_action_is_minus_one(family_map: dict) -> None:
    assert family_id_for_step({"source_action": "DoesNotExist"}, family_map) == -1


def test_select_card_steps_have_no_family(family_map: dict) -> None:
    # SelectCard micro-steps are folded into the parent commit super-step;
    # they must not resolve to any family on their own.
    step = {
        "source_action": "PlayHand",
        "source_kind": "select",
        "action": "SelectCard_CurrentHand_3",
        "target_zone": "CurrentHand",
        "target_position": 3,
    }
    # source_action is PlayHand (the parent), so family_name_for_step
    # returns PlayHand. The tensorizer must use source_kind to filter out
    # select sub-steps from family-target emission.
    assert family_name_for_step(step) == "PlayHand"
    assert step["source_kind"] == "select"


# --------------------------------------------------------------------------- #
# argument_spec sanity
# --------------------------------------------------------------------------- #


def test_argument_spec_has_21_entries() -> None:
    # 21 = REQUIRES_AT_LEAST_ONE_CARD set in granularize.py
    assert len(ARGUMENT_SPEC) == 21


def test_argument_spec_matches_granularize_requires_at_least_one_card() -> None:
    from granularize import REQUIRES_AT_LEAST_ONE_CARD

    assert set(ARGUMENT_SPEC.keys()) == set(REQUIRES_AT_LEAST_ONE_CARD)


def test_attribute_masks_are_set_for_seal_and_aura_consumables() -> None:
    assert ARGUMENT_SPEC[263]["attribute_mask"] == "seal_gold"     # c_talisman
    assert ARGUMENT_SPEC[252]["attribute_mask"] == "seal_red"      # c_deja_vu
    assert ARGUMENT_SPEC[264]["attribute_mask"] == "seal_blue"     # c_trance
    assert ARGUMENT_SPEC[259]["attribute_mask"] == "seal_purple"   # c_medium
    assert ARGUMENT_SPEC[249]["attribute_mask"] == "edition_any"   # c_aura


def test_death_is_exactly_two_cards() -> None:
    spec = ARGUMENT_SPEC[299]
    assert spec["min_cards"] == 2
    assert spec["max_cards"] == 2


def test_world_is_up_to_three_cards() -> None:
    spec = ARGUMENT_SPEC[319]
    assert spec["min_cards"] == 1
    assert spec["max_cards"] == 3


def test_unknown_class_id_has_zero_card_default() -> None:
    spec = spec_for_class_id(999999)
    assert spec == {"min_cards": 0, "max_cards": 0, "attribute_mask": None}


def test_inventory_gates_cover_known_consumables() -> None:
    # Mask schema 4.2-4.3 enumerates these; double-check key examples.
    assert INVENTORY_GATES[248] == "joker_count_>=_1"           # c_ankh
    assert INVENTORY_GATES[256] == "hex_no_editioned_jokers"    # c_hex
    assert INVENTORY_GATES[262] == "needs_joker_slot"           # c_soul
    assert INVENTORY_GATES[303] == "fool_last_planet_not_303"   # c_fool


def test_max_cards_per_decision_is_at_least_play_hand_cap() -> None:
    assert MAX_CARDS_PER_DECISION >= 5


# --------------------------------------------------------------------------- #
# CardCardinality dispatch
# --------------------------------------------------------------------------- #


def test_cardinality_play_discard_max_5() -> None:
    card = cardinality_for_play_discard()
    assert card.family_kind == "card_seq"
    assert card.min_cards == 1
    assert card.max_cards == 5
    assert card.card_zone == "CurrentHand"


def test_cardinality_use_consumable_resolves_card_zone_by_page() -> None:
    # c_chariot (298): exactly 1 card, In_Blind -> CurrentHand
    card = cardinality_for_use_consumable(298, "In_Blind")
    assert card.card_zone == "CurrentHand"
    assert card.min_cards == 1
    assert card.max_cards == 1

    # Same consumable in tarot pack page -> TarotSpectralHand
    card2 = cardinality_for_use_consumable(298, "In_TarotSpectral_Pack")
    assert card2.card_zone == "TarotSpectralHand"


def test_cardinality_use_consumable_no_card_target() -> None:
    # c_temperance (316) does not consume cards
    card = cardinality_for_use_consumable(316, "In_Blind")
    assert card.min_cards == 0
    assert card.max_cards == 0
    assert card.card_zone is None


def test_cardinality_select_pack_item_only_tarot_subtype_has_cards() -> None:
    # c_chariot opened from a tarot pack with tarot subtype consumes cards
    card = cardinality_for_select_pack_item(298, "selectpackitemtarot")
    assert card.min_cards == 1
    assert card.max_cards == 1
    assert card.card_zone == "TarotSpectralHand"

    # Same class as a planet pack item (different subtype) -> no cards
    card_planet = cardinality_for_select_pack_item(298, "selectpackitemplanet")
    assert card_planet.min_cards == 0
    assert card_planet.max_cards == 0
    assert card_planet.card_zone is None


# --------------------------------------------------------------------------- #
# Cross-consistency: flat action_map slices match family_to_flat_size
# --------------------------------------------------------------------------- #


def test_family_to_flat_offsets_consistent_with_action_map(
    family_map: dict, action_map: dict
) -> None:
    # Every family present in FAMILY_ORDER except 'StartNewRun' is also
    # present in the v1 action_map family_offsets; SWAP and indexed
    # families are stored under their subfamily key.
    family_offsets = action_map["family_offsets"]
    for fam in FAMILY_ORDER:
        if fam == "StartNewRun":
            continue
        assert fam in family_offsets, f"{fam} missing from action_map.family_offsets"
        assert family_map["family_to_flat_offset"][fam] == int(family_offsets[fam])
