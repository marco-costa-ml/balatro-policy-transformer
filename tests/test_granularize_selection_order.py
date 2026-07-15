from __future__ import annotations

from granularize import _selected_card_scramble_key, granularize_run


def _card(zone: str, pos: int, *, slot_id: int) -> dict:
    return {
        "class_id": pos,
        "object_type": "playing_card",
        "zone": zone,
        "position_in_zone": pos,
        "slot_id": slot_id,
        "edition": None,
        "seal": None,
        "modifier": None,
        "card": None,
    }


def _play_hand_run() -> dict:
    all_cards = [
        _card("CurrentHandAll", pos, slot_id=100 + pos)
        for pos in range(5)
    ]
    # Selected-zone positions are intentionally sorted left-to-right; the
    # granularizer should not preserve this artificial order.
    selected_cards = [
        _card("CurrentHandSelected", pos, slot_id=100 + pos)
        for pos in range(5)
    ]
    return {
        "video_id": 1,
        "run_index": 0,
        "events": [
            {
                "frame_idx": 123,
                "page_name": "In_Blind",
                "action": "PlayHand",
                "objects": [*all_cards, *selected_cards],
                "state": {},
            }
        ],
    }


def test_card_selection_order_is_deterministically_scrambled() -> None:
    run = _play_hand_run()

    first = granularize_run(run)
    second = granularize_run(run)

    first_selects = [
        step for step in first["events"] if step["source_kind"] == "select"
    ]
    second_selects = [
        step for step in second["events"] if step["source_kind"] == "select"
    ]

    first_positions = [
        step["selected_object"]["object"]["position_in_zone"]
        for step in first_selects
    ]
    second_positions = [
        step["selected_object"]["object"]["position_in_zone"]
        for step in second_selects
    ]

    source_cards = [
        _card("CurrentHandSelected", pos, slot_id=100 + pos)
        for pos in range(5)
    ]
    expected_positions = [
        c["position_in_zone"]
        for c in sorted(
            source_cards,
            key=lambda o: _selected_card_scramble_key(
                o,
                source_event_index=0,
                base_action="PlayHand",
                subtype=None,
            ),
        )
    ]

    assert first_positions == expected_positions
    assert second_positions == expected_positions
    assert first_positions != [0, 1, 2, 3, 4]


def test_scrambled_selects_still_emit_valid_dynamic_pool_indices() -> None:
    granular = granularize_run(_play_hand_run())
    selects = [
        step for step in granular["events"] if step["source_kind"] == "select"
    ]

    pool = [100 + pos for pos in range(5)]
    emitted_dynamic_indices: list[int] = []
    for step in selects:
        selected_slot = step["selected_object"]["object"]["slot_id"]
        dyn_idx = pool.index(selected_slot)
        emitted_dynamic_indices.append(dyn_idx)
        assert step["target_position"] == dyn_idx
        assert step["action"] == f"SelectCard_CurrentHand_{dyn_idx}"
        pool.pop(dyn_idx)

    assert emitted_dynamic_indices == [2, 0, 1, 1, 0]
    assert granular["events"][-1]["action"] == "PlayHand"
