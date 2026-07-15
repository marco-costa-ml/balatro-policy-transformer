"""Tests for ``super_step.iter_super_steps``."""

from __future__ import annotations

from super_step import SuperStep, iter_super_steps


def _step(idx: int, source_idx: int, micro: int, kind: str, **extras) -> dict:
    return {
        "step_id": idx,
        "source_event_index": source_idx,
        "micro_index": micro,
        "source_kind": kind,
        **extras,
    }


def test_single_pass_through_event() -> None:
    events = [_step(0, 0, 0, "pass_through", source_action="CashOut")]
    steps = list(iter_super_steps(events))
    assert len(steps) == 1
    assert steps[0].kind == "regular"
    assert steps[0].encoder_step is events[0]
    assert steps[0].commit_step is events[0]
    assert steps[0].select_steps == []


def test_select_commit_block_collapses_to_one_super_step() -> None:
    events = [
        _step(0, 0, 0, "select", source_action="PlayHand"),
        _step(1, 0, 1, "select", source_action="PlayHand"),
        _step(2, 0, 2, "select", source_action="PlayHand"),
        _step(3, 0, 3, "commit", source_action="PlayHand"),
    ]
    steps = list(iter_super_steps(events))
    assert len(steps) == 1
    ss = steps[0]
    assert ss.kind == "regular"
    assert ss.encoder_step is events[0]
    assert ss.commit_step is events[3]
    assert ss.select_steps == events[:3]
    assert ss.encoder_step_idx == 0


def test_swap_synth_each_becomes_its_own_super_step() -> None:
    events = [
        _step(0, 1, -2, "swap_synth", source_action="SWAP", swap_pair=[0, 1]),
        _step(1, 1, -1, "swap_synth", source_action="SWAP", swap_pair=[1, 2]),
        _step(2, 1, 0, "select", source_action="PlayHand"),
        _step(3, 1, 1, "commit", source_action="PlayHand"),
    ]
    steps = list(iter_super_steps(events))
    assert len(steps) == 3
    assert steps[0].kind == "swap" and steps[0].commit_step is events[0]
    assert steps[1].kind == "swap" and steps[1].commit_step is events[1]
    assert steps[2].kind == "regular"
    assert steps[2].encoder_step is events[2]
    assert steps[2].commit_step is events[3]


def test_orphan_selects_are_skipped() -> None:
    events = [
        _step(0, 0, 0, "select", source_action="PlayHand"),
        _step(1, 0, 1, "select", source_action="PlayHand"),
        # NO commit / pass_through with source_idx=0 — malformed.
        _step(2, 1, 0, "pass_through", source_action="CashOut"),
    ]
    steps = list(iter_super_steps(events))
    # Only the CashOut survives; the orphan selects are dropped.
    assert len(steps) == 1
    assert steps[0].commit_step is events[2]


def test_multiple_events_walk_in_order() -> None:
    events = [
        _step(0, 0, 0, "pass_through", source_action="SelectBlind"),
        _step(1, 1, 0, "select", source_action="PlayHand"),
        _step(2, 1, 1, "commit", source_action="PlayHand"),
        _step(3, 2, 0, "pass_through", source_action="CashOut"),
        _step(4, 3, 0, "pass_through", source_action="LeaveShop"),
    ]
    steps = list(iter_super_steps(events))
    assert [s.commit_step["source_action"] for s in steps] == [
        "SelectBlind",
        "PlayHand",
        "CashOut",
        "LeaveShop",
    ]


def test_use_consumable_with_no_decomposition_is_one_super_step() -> None:
    # c_temperance / c_hermit / planet cards don't take card targets, so
    # granularize emits a single pass_through step.
    events = [
        _step(
            0, 0, 0, "pass_through",
            source_action="UseConsumable",
            action="UseConsumable_CurrentConsumables_0",
            target_zone="CurrentConsumables",
            target_position=0,
        ),
    ]
    steps = list(iter_super_steps(events))
    assert len(steps) == 1
    assert steps[0].kind == "regular"
    assert steps[0].select_steps == []


def test_select_pack_item_tarot_two_cards_full_sequence() -> None:
    # SelectPackItem(tarot) opens the card-selection screen and the agent
    # then picks 2 cards. Granularize records:
    #   - SelectCard_TarotSpectralHand select 1
    #   - SelectCard_TarotSpectralHand select 2
    #   - SelectPackItem commit
    events = [
        _step(0, 5, 0, "select", source_action="SelectPackItem", action="SelectCard_TarotSpectralHand_0"),
        _step(1, 5, 1, "select", source_action="SelectPackItem", action="SelectCard_TarotSpectralHand_1"),
        _step(
            2, 5, 2, "commit",
            source_action="SelectPackItem",
            action="SelectPackItem_PackOfferings_0",
            target_zone="PackOfferings",
            target_position=0,
            source_action_subtype="selectpackitemtarot",
        ),
    ]
    steps = list(iter_super_steps(events))
    assert len(steps) == 1
    ss = steps[0]
    assert ss.kind == "regular"
    assert ss.encoder_step is events[0]
    assert len(ss.select_steps) == 2
    assert ss.commit_step is events[2]
