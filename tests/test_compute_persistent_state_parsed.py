"""Tests for parsed-export persistent state (current_deck_cards)."""

from __future__ import annotations

import collections

from card_effects import make_standard_card
from compute_persistent_state import (
    _process_run,
    _update_current_deck_cards,
    parsed_event_to_reducer_step,
)


def _card(class_id: int, slot_id: int, zone: str = "CurrentHandAll") -> dict:
    return {
        "zone": zone,
        "slot_id": slot_id,
        "class_id": class_id,
        "position_in_zone": slot_id,
        "object_type": "card",
        "card": make_standard_card(class_id)["card"],
        "modifier": None,
        "edition": None,
        "seal": None,
    }


def test_current_deck_resets_on_non_in_blind_page():
    tracked = {"tracked_deck_cards": [make_standard_card(i) for i in range(3)]}
    event = {"page_name": "In_Shop", "objects": []}
    out = _update_current_deck_cards([], tracked, event, "BuyShopItem")
    assert len(out) == 3
    assert out[0]["class_id"] == 0


def test_playhand_removes_current_hand_all_cards():
    deck = [make_standard_card(i) for i in range(5)]
    hand = [_card(2, 100), _card(4, 101)]
    tracked = {"tracked_deck_cards": deck}
    current = list(deck)
    event = {
        "page_name": "In_Blind",
        "objects": hand,
        "action_details": [{"type": "PlayHand", "id": "x", "subtype": "playhand"}],
    }
    out = _update_current_deck_cards(current, tracked, event, "PlayHand")
    assert len(out) == 3
    class_ids = {c["class_id"] for c in out}
    assert 2 not in class_ids
    assert 4 not in class_ids


def test_process_run_injects_current_deck_cards_on_parsed_export():
    run = {
        "video_id": "t",
        "run_index": 0,
        "events": [
            {
                "frame_idx": 1,
                "page_name": "In_Blind",
                "state": {"ante": 1, "round": 0},
                "objects": [
                    {
                        "zone": "CurrentDeckAll",
                        "slot_id": 1,
                        "class_id": 0,
                        "position_in_zone": 0,
                        "object_type": "card",
                    }
                ],
                "action_details": [
                    {"type": "StartNewRun", "id": "a", "subtype": "startnewrun"}
                ],
            },
            {
                "frame_idx": 2,
                "page_name": "In_Shop",
                "state": {"ante": 1, "round": 0},
                "objects": [],
                "action_details": [
                    {"type": "LeaveShop", "id": "b", "subtype": "leaveshop"}
                ],
            },
        ],
    }
    stats: collections.Counter = collections.Counter()
    out = _process_run(
        run,
        stats,
        event_source="parsed",
        adapt_step=parsed_event_to_reducer_step,
    )
    assert out["schema_version"] == "2.1.0"
    assert out["parsed_export_keys"] == ["current_deck_cards"]
    assert "current_deck_cards" in out["states"][0]
    assert "current_deck_cards" in out["states"][1]
    assert len(out["states"][1]["current_deck_cards"]) == len(
        out["states"][1]["tracked_deck_cards"]
    )
