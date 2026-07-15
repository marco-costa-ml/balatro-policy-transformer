"""Tests for incomplete-blind OCR/action timing rules."""

from __future__ import annotations

from analytics.count_incomplete_blinds import BlindSegment, analyze_run


def _ev(
    *,
    frame: int,
    round_n: int = 1,
    hands_left: int | None = 4,
    discards_left: int | None = 3,
    action: str = "SelectCard",
) -> dict:
    return {
        "frame_idx": frame,
        "page_name": "In_Blind",
        "state": {
            "round": round_n,
            "hands_left": hands_left,
            "discards_left": discards_left,
        },
        "action_details": [{"type": action, "id": f"{action}_{frame}", "subtype": action.lower()}],
    }


def test_playhand_with_zero_delta_is_valid_post_action_ocr():
    run = {
        "run_index": 0,
        "events": [
            _ev(frame=100, hands_left=4, discards_left=3, action="SelectCard"),
            _ev(frame=101, hands_left=3, discards_left=3, action="SelectCard"),
            _ev(frame=102, hands_left=3, discards_left=3, action="PlayHand"),
        ],
    }
    segs = analyze_run(run, "vid")
    assert len(segs) == 1
    assert not segs[0].incomplete


def test_playhand_with_one_delta_is_valid():
    run = {
        "run_index": 0,
        "events": [
            _ev(frame=100, hands_left=4, action="SelectCard"),
            _ev(frame=101, hands_left=3, action="PlayHand"),
        ],
    }
    segs = analyze_run(run, "vid")
    assert not segs[0].incomplete


def test_unlabeled_drop_without_nearby_playhand_is_incomplete():
    run = {
        "run_index": 0,
        "events": [
            _ev(frame=100, hands_left=4, action="SelectCard"),
            _ev(frame=101, hands_left=3, action="SelectCard"),
            _ev(frame=102, hands_left=3, action="SelectCard"),
        ],
    }
    segs = analyze_run(run, "vid")
    assert segs[0].incomplete
    assert any("without nearby PlayHand" in r for r in segs[0].reasons)


def test_discards_not_checked_on_playhand_frame():
    run = {
        "run_index": 0,
        "events": [
            _ev(frame=100, hands_left=4, discards_left=3, action="SelectCard"),
            _ev(frame=101, hands_left=3, discards_left=2, action="PlayHand"),
        ],
    }
    segs = analyze_run(run, "vid")
    assert not any("discards_left" in r for r in segs[0].reasons)
