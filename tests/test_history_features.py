#!/usr/bin/env python3
"""Small no-pytest smoke tests for rich history tensorization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map
from history_features import build_history_tensors, history_caps
from tensorize import Normalizer, VocabLookup, tensorize_step
from state_reducer import default_state


def _artifacts():
    vocab = VocabLookup(json.loads((_REPO_ROOT / "artifacts" / "vocab.json").read_text(encoding="utf-8")))
    norm = Normalizer(json.loads((_REPO_ROOT / "artifacts" / "normalization.json").read_text(encoding="utf-8")))
    feat = json.loads((_REPO_ROOT / "artifacts" / "feature_config.json").read_text(encoding="utf-8"))
    action_cfg = json.loads((_REPO_ROOT / "data" / "action_space_config.json").read_text(encoding="utf-8"))
    amap = compute_action_map(action_cfg)
    return vocab, norm, feat, amap


def _card(zone: str, pos: int, class_id: int) -> dict:
    return {
        "class_id": class_id,
        "object_type": "card",
        "zone": zone,
        "position_in_zone": pos,
        "modifier": None,
        "edition": None,
        "seal": None,
        "card": {"rank_index": class_id % 13, "suit_index": class_id // 13},
    }


def _pack_item(pos: int, class_id: int) -> dict:
    return {
        "class_id": class_id,
        "object_type": "spectral",
        "zone": "PackOfferings",
        "position_in_zone": pos,
        "modifier": None,
        "edition": None,
        "seal": None,
        "card": None,
    }


def test_empty_history_shape() -> None:
    vocab, norm, feat, amap = _artifacts()
    tensors = build_history_tensors([], action_map=amap, vocab=vocab, norm=norm, feature_config=feat)
    h, o = history_caps(feat)
    assert tensors["history_step_mask"].shape == (h,)
    assert tensors["history_object_mask"].shape == (h, o)
    assert not tensors["history_step_mask"].any()
    assert not tensors["history_object_mask"].any()


def test_history_uses_only_prior_steps_and_pack_objects() -> None:
    vocab, norm, feat, amap = _artifacts()
    prior = {
        "page_name": "In_TarotSpectral_Pack",
        "action": "SelectPackItem_PackOfferings_1",
        "target_zone": "PackOfferings",
        "target_position": 1,
        "state": {"hands_left": 4, "dollars": 4, "ante": 1},
        "objects": [
            _card("TarotSpectralHand", 0, 0),
            _card("PendingCards", 0, 13),
            _pack_item(0, 308),
            _pack_item(1, 309),
        ],
        "selected_object": {"object": _pack_item(1, 309)},
    }
    current = {
        "page_name": "In_Blind",
        "action": "PlayHand",
        "state": {"hands_left": 4, "dollars": 4, "ante": 1},
        "objects": [_card("CurrentHand", 0, 0)],
    }
    rec = tensorize_step(
        current,
        default_state(),
        amap,
        vocab,
        norm,
        feat,
        history_steps=[prior],
    )
    assert bool(rec["history_step_mask"][0])
    assert not bool(rec["history_step_mask"][1])
    assert int(rec["history_target_position"][0]) == 2
    assert int(rec["history_object_class_id"][0, 0]) == 309
    assert int(rec["history_object_mask"][0].sum()) >= 3


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} history feature tests passed")


if __name__ == "__main__":
    main()
