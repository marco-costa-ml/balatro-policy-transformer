"""Tests for --max-ante filtering in tensorize.py."""

from __future__ import annotations

import collections

import pytest

from tensorize import _process_run, _step_ocr_ante, _step_passes_max_ante


def _blind_select_step(*, ante: int | float | None) -> dict:
    state: dict = {
        "hands_left": 4,
        "discards_left": 3,
        "dollars": 4,
        "round": 1,
    }
    if ante is not None:
        state["ante"] = ante
    return {
        "page_name": "Blind_Select",
        "action": "SelectBlind",
        "source_kind": "pass_through",
        "source_event_index": 0,
        "micro_index": 0,
        "objects": [],
        "state": state,
    }


def test_step_ocr_ante_reads_state() -> None:
    assert _step_ocr_ante(_blind_select_step(ante=7)) == 7
    assert _step_ocr_ante(_blind_select_step(ante=None)) is None


def test_step_passes_max_ante() -> None:
    step = _blind_select_step(ante=12)
    assert _step_passes_max_ante(step, None) is True
    assert _step_passes_max_ante(step, 12) is True
    assert _step_passes_max_ante(step, 11) is False
    assert _step_passes_max_ante(_blind_select_step(ante=None), 12) is False


def test_process_run_max_ante_filters_super_steps(fixtures: dict) -> None:
    run = {
        "video_id": 1,
        "run_index": 0,
        "events": [
            _blind_select_step(ante=8),
            _blind_select_step(ante=12),
            _blind_select_step(ante=15),
        ],
    }
    stats: collections.Counter = collections.Counter()

    rec = _process_run(
        run,
        None,
        fixtures["action_map"],
        fixtures["vocab"],
        fixtures["norm"],
        fixtures["feature_config"],
        stats,
        family_map=fixtures["family_map"],
        branched_caps=fixtures["caps"],
        max_ante=10,
    )

    assert int(rec["family_id"].shape[0]) == 1
    assert stats[("filtered", "max_ante_excluded")] == 2
    assert stats.get(("filtered", "max_ante_missing"), 0) == 0


@pytest.fixture
def fixtures() -> dict:
    import json
    from pathlib import Path

    from action_map import compute_action_map
    from family_map import compute_family_map
    from tensorize import Normalizer, VocabLookup, derive_branched_caps

    root = Path(__file__).resolve().parent.parent
    vocab_payload = json.loads((root / "artifacts/vocab.json").read_text(encoding="utf-8"))
    norm_payload = json.loads((root / "artifacts/normalization.json").read_text(encoding="utf-8"))
    feature_config = json.loads((root / "artifacts/feature_config.json").read_text(encoding="utf-8"))
    action_config = json.loads((root / "data/action_space_config.json").read_text(encoding="utf-8"))
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
