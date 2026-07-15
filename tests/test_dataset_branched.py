"""Tests for the branched-policy validity filter in dataset.BalatroStepDataset.

Uses a tiny synthetic ``.npz`` shard so the tests are independent of the
real corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from action_map import compute_action_map
from dataset import BalatroStepDataset
from family_map import compute_family_map
from tensorize import derive_branched_caps


def _load_caps_and_family_map():
    import json

    cfg = json.loads(Path("data/action_space_config.json").read_text(encoding="utf-8"))
    action_map = compute_action_map(cfg)
    family_map = compute_family_map(action_map)
    caps = derive_branched_caps(action_map, family_map)
    return family_map, caps, action_map


def _make_synthetic_shard(
    out_dir: Path,
    family_map: dict,
    caps: dict,
    action_map: dict,
) -> None:
    """Build a 4-row shard exercising each decoder shape."""
    n_actions = int(action_map["n_actions"])
    n_families = int(family_map["n_families"])
    max_item = int(caps["MAX_ITEM_ZONE_SIZE"])
    max_card = int(caps["MAX_CARD_ZONE_SIZE"])
    max_joker = int(caps["MAX_JOKER_SLOTS"])
    max_cards_per = int(caps["MAX_CARDS_PER_DECISION"])

    n = 5

    def _zero(shape):
        return np.zeros(shape, dtype=np.int64)

    arrays: dict[str, np.ndarray] = {
        "target_action_id": np.array([0, 0, 0, 0, -1], dtype=np.int64),
        "action_mask": np.ones((n, n_actions), dtype=bool),
        "family_id": np.array(
            [
                family_map["family_to_id"]["PlayHand"],
                family_map["family_to_id"]["UseConsumable_CurrentConsumables"],
                family_map["family_to_id"]["SelectBlind"],
                family_map["family_to_id"]["SWAP"],
                -1,
            ],
            dtype=np.int64,
        ),
        "num_cards": np.array([3, 0, 0, 0, -1], dtype=np.int64),
        "item_ptr_local": np.array([-1, 2, -1, -1, -1], dtype=np.int64),
        "card_ptr_local_seq": np.full((n, max_cards_per), -1, dtype=np.int64),
        "swap_i_local": np.array([-1, -1, -1, 0, -1], dtype=np.int64),
        "swap_j_local": np.array([-1, -1, -1, 1, -1], dtype=np.int64),
        "item_ptr_slot": _zero((n,)) - 1,
        "card_ptr_slot_seq": _zero((n, max_cards_per)) - 1,
        "swap_i_slot": _zero((n,)) - 1,
        "swap_j_slot": _zero((n,)) - 1,
        "family_mask": np.zeros((n, n_families), dtype=bool),
        "item_pointer_mask": np.zeros((n, max_item), dtype=bool),
        "card_pointer_mask": np.zeros((n, max_card), dtype=bool),
        "swap_joker_mask": np.zeros((n, max_joker), dtype=bool),
    }
    # Card sequence for row 0 (PlayHand): [4, 1, 2]
    arrays["card_ptr_local_seq"][0, :3] = np.array([4, 1, 2])
    # Set masks so the rows pass.
    for k in range(4):
        arrays["family_mask"][k, arrays["family_id"][k]] = True
    arrays["card_pointer_mask"][0, [1, 2, 4]] = True
    arrays["item_pointer_mask"][1, 2] = True
    arrays["swap_joker_mask"][3, [0, 1]] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run_000.npz", **arrays)


def test_branched_dataset_filter_drops_invalid_rows(tmp_path: Path) -> None:
    family_map, caps, action_map = _load_caps_and_family_map()
    shard_root = tmp_path / "tensorized"
    _make_synthetic_shard(
        shard_root / "video_id=synth0",
        family_map,
        caps,
        action_map,
    )

    ds = BalatroStepDataset(
        tensorized_root=shard_root,
        split_videos=["synth0"],
        family_map=family_map,
        branched_caps=caps,
        require_branched=True,
    )
    # 4 supervised rows; row 4 has target_action_id=-1 → dropped.
    assert len(ds) == 4
    fids = ds._tensors["family_id"].index_select(0, ds._valid).cpu().numpy()
    expected = {
        family_map["family_to_id"]["PlayHand"],
        family_map["family_to_id"]["UseConsumable_CurrentConsumables"],
        family_map["family_to_id"]["SelectBlind"],
        family_map["family_to_id"]["SWAP"],
    }
    assert set(int(x) for x in fids) == expected


def test_branched_dataset_drops_card_mask_violations(tmp_path: Path) -> None:
    family_map, caps, action_map = _load_caps_and_family_map()
    shard_root = tmp_path / "tensorized"
    _make_synthetic_shard(
        shard_root / "video_id=synth1",
        family_map,
        caps,
        action_map,
    )
    # Corrupt the shard: zero out card_pointer_mask[0, 1] so row 0's
    # card_seq has an illegal pointer.
    shard = shard_root / "video_id=synth1" / "run_000.npz"
    with np.load(shard) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["card_pointer_mask"][0, 1] = False
    np.savez(shard, **arrays)

    ds = BalatroStepDataset(
        tensorized_root=shard_root,
        split_videos=["synth1"],
        family_map=family_map,
        branched_caps=caps,
        require_branched=True,
    )
    assert len(ds) == 3
    fids = ds._tensors["family_id"].index_select(0, ds._valid).cpu().numpy()
    assert family_map["family_to_id"]["PlayHand"] not in set(int(x) for x in fids)


def test_branched_dataset_drops_swap_i_eq_j(tmp_path: Path) -> None:
    family_map, caps, action_map = _load_caps_and_family_map()
    shard_root = tmp_path / "tensorized"
    _make_synthetic_shard(
        shard_root / "video_id=synth2",
        family_map,
        caps,
        action_map,
    )
    shard = shard_root / "video_id=synth2" / "run_000.npz"
    with np.load(shard) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["swap_i_local"][3] = 0
    arrays["swap_j_local"][3] = 0
    np.savez(shard, **arrays)

    ds = BalatroStepDataset(
        tensorized_root=shard_root,
        split_videos=["synth2"],
        family_map=family_map,
        branched_caps=caps,
        require_branched=True,
    )
    # Row 3 dropped (i == j).
    assert len(ds) == 3


def test_include_unresolved_keeps_all_rows(tmp_path: Path) -> None:
    family_map, caps, action_map = _load_caps_and_family_map()
    shard_root = tmp_path / "tensorized"
    _make_synthetic_shard(
        shard_root / "video_id=synth3",
        family_map,
        caps,
        action_map,
    )
    ds = BalatroStepDataset(
        tensorized_root=shard_root,
        split_videos=["synth3"],
        family_map=family_map,
        branched_caps=caps,
        include_unresolved=True,
    )
    assert len(ds) == 5


def test_dataset_metadata_accessors(tmp_path: Path) -> None:
    family_map, caps, action_map = _load_caps_and_family_map()
    shard_root = tmp_path / "tensorized"
    _make_synthetic_shard(
        shard_root / "video_id=synth4",
        family_map,
        caps,
        action_map,
    )
    ds = BalatroStepDataset(
        tensorized_root=shard_root,
        split_videos=["synth4"],
        family_map=family_map,
        branched_caps=caps,
    )
    assert ds.n_actions() == int(action_map["n_actions"])
    assert ds.n_families() == int(family_map["n_families"])
    assert ds.branched_caps() == caps
    hist = ds.family_id_histogram()
    assert sum(hist.values()) == len(ds)
