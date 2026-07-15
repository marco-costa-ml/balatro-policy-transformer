"""End-to-end tests for the super-step tensorizer (schema 3.0.0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from action_map import compute_action_map
from family_map import compute_family_map
from tensorize import (
    Normalizer,
    VocabLookup,
    _process_run,
    derive_branched_caps,
    tensorize_super_step,
)
from super_step import SuperStep


_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fixtures() -> dict:
    vocab_payload = json.loads((_REPO_ROOT / "artifacts/vocab.json").read_text(encoding="utf-8"))
    norm_payload = json.loads((_REPO_ROOT / "artifacts/normalization.json").read_text(encoding="utf-8"))
    feature_config = json.loads((_REPO_ROOT / "artifacts/feature_config.json").read_text(encoding="utf-8"))
    action_config = json.loads((_REPO_ROOT / "data/action_space_config.json").read_text(encoding="utf-8"))
    vocab = VocabLookup(vocab_payload)
    norm = Normalizer(norm_payload)
    action_map = compute_action_map(action_config)
    family_map = compute_family_map(action_map)
    caps = derive_branched_caps(action_map, family_map)
    return {
        "vocab": vocab,
        "norm": norm,
        "feature_config": feature_config,
        "action_map": action_map,
        "family_map": family_map,
        "caps": caps,
    }


# --------------------------------------------------------------------------- #
# Synthetic super-steps
# --------------------------------------------------------------------------- #


def _hand_card(slot: int, pos: int) -> dict:
    return {
        "zone": "CurrentHand",
        "slot_id": slot,
        "position_in_zone": pos,
        "class_id": 1,
        "object_type": "playing_card",
        "rank_index": 5,
        "suit_index": 1,
        "card": {"rank_index": 5, "suit_index": 1},
    }


def _joker(slot: int, pos: int) -> dict:
    return {
        "zone": "CurrentJokers",
        "slot_id": slot,
        "position_in_zone": pos,
        "class_id": 100 + slot,
        "object_type": "joker",
    }


def _empty_persistent_state() -> dict:
    from state_reducer import default_state
    return default_state()


def test_play_hand_super_step_emits_card_seq_targets(fixtures: dict) -> None:
    hand = [_hand_card(11, 0), _hand_card(12, 1), _hand_card(13, 2), _hand_card(14, 3), _hand_card(15, 4)]

    encoder_step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": hand,
        "state": {"jokers_current": 0},
        "source_action": "PlayHand",
        "source_kind": "select",
        "source_event_index": 1,
        "action": "SelectCard_CurrentHand_0",
        "target_zone": "CurrentHand",
    }
    select_steps = []
    for i, (slot, pos) in enumerate([(11, 0), (13, 2), (14, 3)]):
        sel = dict(encoder_step)
        sel["selected_object"] = {"object": {"slot_id": slot, "zone": "CurrentHandAll", "position_in_zone": pos}}
        select_steps.append(sel)
    commit_step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": hand,
        "state": {"jokers_current": 0},
        "source_action": "PlayHand",
        "source_kind": "commit",
        "action": "PlayHand",
        "target_zone": None,
        "target_position": None,
    }
    ss = SuperStep(
        kind="regular",
        encoder_step=select_steps[0],
        commit_step=commit_step,
        select_steps=select_steps,
        encoder_step_idx=0,
    )
    rec = tensorize_super_step(
        ss, _empty_persistent_state(),
        fixtures["action_map"], fixtures["family_map"],
        fixtures["vocab"], fixtures["norm"], fixtures["feature_config"],
        fixtures["caps"],
    )

    play_id = fixtures["family_map"]["family_to_id"]["PlayHand"]
    assert int(rec["family_id"]) == play_id
    assert int(rec["num_cards"]) == 3
    # zone-local positions match selected_object.object.position_in_zone
    assert rec["card_ptr_local_seq"][:3].tolist() == [0, 2, 3]
    assert rec["card_ptr_local_seq"][3:].tolist() == [-1, -1]
    # slot_ids preserved for live unroll
    assert rec["card_ptr_slot_seq"][:3].tolist() == [11, 13, 14]
    # No item pointer for card_seq
    assert int(rec["item_ptr_local"]) == -1
    # Family mask must include PlayHand
    assert bool(rec["family_mask"][play_id])
    # Pointer card mask should be set at the three selected positions
    cpm = rec["card_pointer_mask"]
    assert bool(cpm[0]) and bool(cpm[2]) and bool(cpm[3])


def test_use_consumable_with_cards_super_step(fixtures: dict) -> None:
    consumable = {
        "zone": "CurrentConsumables",
        "slot_id": 500,
        "position_in_zone": 0,
        "class_id": 302,   # c_empress -> up to 2 cards
        "object_type": "tarot",
    }
    hand = [_hand_card(11, 0), _hand_card(12, 1)]

    encoder_step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": hand + [consumable],
        "state": {"jokers_current": 0},
        "source_action": "UseConsumable",
        "source_kind": "select",
        "source_event_index": 5,
        "action": "SelectCard_CurrentHand_0",
        "target_zone": "CurrentHand",
    }
    sel1 = dict(encoder_step)
    sel1["selected_object"] = {"object": {"slot_id": 11, "zone": "CurrentHandAll", "position_in_zone": 0}}
    sel2 = dict(encoder_step)
    sel2["selected_object"] = {"object": {"slot_id": 12, "zone": "CurrentHandAll", "position_in_zone": 1}}

    commit_step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": hand + [consumable],
        "state": {"jokers_current": 0},
        "source_action": "UseConsumable",
        "source_kind": "commit",
        "action": "UseConsumable_CurrentConsumables_0",
        "target_zone": "CurrentConsumables",
        "target_position": 0,
        "selected_object": {"object": {"slot_id": 500, "zone": "CurrentConsumablesAll", "position_in_zone": 0}},
    }
    ss = SuperStep(
        kind="regular",
        encoder_step=sel1,
        commit_step=commit_step,
        select_steps=[sel1, sel2],
        encoder_step_idx=0,
    )
    rec = tensorize_super_step(
        ss, _empty_persistent_state(),
        fixtures["action_map"], fixtures["family_map"],
        fixtures["vocab"], fixtures["norm"], fixtures["feature_config"],
        fixtures["caps"],
    )

    fam_id = fixtures["family_map"]["family_to_id"]["UseConsumable_CurrentConsumables"]
    assert int(rec["family_id"]) == fam_id
    assert int(rec["item_ptr_local"]) == 0
    assert int(rec["item_ptr_slot"]) == 500
    assert int(rec["num_cards"]) == 2
    assert rec["card_ptr_local_seq"][:2].tolist() == [0, 1]
    assert rec["card_ptr_slot_seq"][:2].tolist() == [11, 12]


def test_swap_super_step(fixtures: dict) -> None:
    jokers = [_joker(1, 0), _joker(2, 1), _joker(3, 2)]
    swap_step = {
        "page_name": "In_Blind",
        "pending_cards": [],
        "objects": jokers,
        "state": {"jokers_current": 3},
        "source_action": "SWAP",
        "source_kind": "swap_synth",
        "action": "SWAP_0_2",
        "swap_pair": [0, 2],
    }
    ss = SuperStep(
        kind="swap",
        encoder_step=swap_step,
        commit_step=swap_step,
        select_steps=[],
        encoder_step_idx=0,
    )
    rec = tensorize_super_step(
        ss, _empty_persistent_state(),
        fixtures["action_map"], fixtures["family_map"],
        fixtures["vocab"], fixtures["norm"], fixtures["feature_config"],
        fixtures["caps"],
    )

    swap_fam = fixtures["family_map"]["family_to_id"]["SWAP"]
    assert int(rec["family_id"]) == swap_fam
    assert int(rec["swap_i_local"]) == 0
    assert int(rec["swap_j_local"]) == 2
    assert int(rec["swap_i_slot"]) == 1
    assert int(rec["swap_j_slot"]) == 3
    assert int(rec["num_cards"]) == 0
    # SWAP joker mask first 3 set, rest unset
    assert rec["swap_joker_mask"][:3].all()
    assert not rec["swap_joker_mask"][3:].any()


def test_start_new_run_super_step_is_unsupervised(fixtures: dict) -> None:
    step = {
        "page_name": "Dummy_Page",
        "objects": [],
        "state": {},
        "source_action": "StartNewRun",
        "source_kind": "pass_through",
        "action": "StartNewRun",
        "target_zone": None,
        "target_position": None,
    }
    ss = SuperStep(
        kind="regular", encoder_step=step, commit_step=step,
        select_steps=[], encoder_step_idx=0,
    )
    rec = tensorize_super_step(
        ss, _empty_persistent_state(),
        fixtures["action_map"], fixtures["family_map"],
        fixtures["vocab"], fixtures["norm"], fixtures["feature_config"],
        fixtures["caps"],
    )
    assert int(rec["family_id"]) == -1
    assert int(rec["target_action_id"]) == -1


# --------------------------------------------------------------------------- #
# End-to-end: process one real run and check invariants
# --------------------------------------------------------------------------- #


def test_process_run_on_real_data_passes_invariants(fixtures: dict) -> None:
    runs = sorted((_REPO_ROOT / "data/granularized").rglob("run_*.json"))[:1]
    if not runs:
        pytest.skip("no granularized runs found in workspace")
    run = json.loads(runs[0].read_text(encoding="utf-8"))
    psnap_path = _REPO_ROOT / "data/persistent_state" / runs[0].parent.name / runs[0].name
    psnap = json.loads(psnap_path.read_text(encoding="utf-8")) if psnap_path.exists() else None
    import collections
    stats: collections.Counter = collections.Counter()

    rec = _process_run(
        run, psnap,
        fixtures["action_map"],
        fixtures["vocab"], fixtures["norm"], fixtures["feature_config"],
        stats,
        family_map=fixtures["family_map"],
        branched_caps=fixtures["caps"],
    )
    n = int(rec["family_id"].shape[0])
    assert n > 0
    assert rec["family_mask"].shape == (n, fixtures["family_map"]["n_families"])
    assert rec["card_ptr_local_seq"].shape == (n, fixtures["caps"]["MAX_CARDS_PER_DECISION"])

    # For each resolved row, family_mask[family_id] must be True.
    resolved = rec["family_id"] >= 0
    for k in np.where(resolved)[0]:
        fid = int(rec["family_id"][k])
        assert bool(rec["family_mask"][k, fid]), f"row {k}: family_mask[{fid}] == False"

    # For each resolved row, action_mask[target_action_id] must be True.
    resolved_v1 = rec["target_action_id"] >= 0
    for k in np.where(resolved_v1)[0]:
        aid = int(rec["target_action_id"][k])
        assert bool(rec["action_mask"][k, aid])

    # Sanity: num_cards never exceeds MAX_CARDS_PER_DECISION
    max_cards = fixtures["caps"]["MAX_CARDS_PER_DECISION"]
    assert int(rec["num_cards"].max()) <= max_cards
