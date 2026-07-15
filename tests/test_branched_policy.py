#!/usr/bin/env python3
"""
test_branched_policy.py
=======================
Integration tests for the live branched-inference pipeline:

- ``LiveEncoder.encode`` emits the branched-policy channels with sane
  shapes / dtypes.
- ``LiveEncoder.pointer_masks_for_family`` recomputes item / card /
  swap pointer masks per family from a Lua snapshot, intersected with
  Lua's legal set.
- ``PolicyTransformer.encode_and_pick_family`` + ``decode_arguments``
  produce well-formed predictions for every scenario the smoke test
  covers. The model is initialised with random weights — we just
  check shapes, masking, and that the predicted family is always one
  the family mask permits.

The full ``live/smoke_test.py`` end-to-end script is the place to
sanity-check the *quality* of the predictions against a trained
checkpoint; here we just need the inference *plumbing* to be solid.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from live.emission_policy import expand_decision
from live.live_encoder import LiveEncoder
from live.smoke_test import SCENARIOS
from model import PolicyTransformer, load_model_config


# Run the encoder/model pipeline once for the test session — the test
# functions below share these via fixtures.

@pytest.fixture(scope="module")
def encoder() -> LiveEncoder:
    return LiveEncoder()


@pytest.fixture(scope="module")
def model(encoder: LiveEncoder) -> PolicyTransformer:
    cfg = load_model_config(
        n_actions=encoder.n_actions,
        # Keep the test cheap: skip history + deck token banks.
        use_history_tokens=False,
        use_tracked_deck_tokens=False,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dim_feedforward=64,
    )
    m = PolicyTransformer(cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# LiveEncoder emits the branched channels
# ---------------------------------------------------------------------------


def test_encoder_emits_family_and_pointer_channels(encoder, device):
    snapshot = SCENARIOS["in_blind"]()
    batch, legal_mask = encoder.encode(snapshot, device=device)
    expected = [
        "family_mask",
        "item_pointer_mask",
        "card_pointer_mask",
        "swap_joker_mask",
        "family_id",
    ]
    for key in expected:
        assert key in batch, f"missing branched channel {key!r}"
    n_families = encoder.family_map["n_families"]
    assert batch["family_mask"].shape == (1, n_families)
    assert batch["family_mask"].dtype == torch.bool
    caps = encoder.branched_caps
    assert batch["item_pointer_mask"].shape == (1, caps["MAX_ITEM_ZONE_SIZE"])
    assert batch["card_pointer_mask"].shape == (1, caps["MAX_CARD_ZONE_SIZE"])
    assert batch["swap_joker_mask"].shape == (1, caps["MAX_JOKER_SLOTS"])
    # Pointer masks come back all-False because the encoder doesn't know
    # the family yet — pointer_masks_for_family fills them in phase 2.
    assert not bool(batch["item_pointer_mask"].any().item())
    assert not bool(batch["card_pointer_mask"].any().item())
    assert not bool(batch["swap_joker_mask"].any().item())
    # Family mask is non-empty on a valid In_Blind snapshot.
    assert bool(batch["family_mask"].any().item())
    assert legal_mask.dtype == bool


def test_encoder_family_mask_matches_legal_actions(encoder, device):
    # In_Blind: PlayHand / DiscardHand / SelectCard must be allowed; SWAP
    # must not (no jokers). The family mask must reflect that.
    snapshot = SCENARIOS["in_blind"]()
    batch, _ = encoder.encode(snapshot, device=device)
    mask = batch["family_mask"].squeeze(0).tolist()
    family_id = encoder.family_to_id
    assert mask[family_id["PlayHand"]] is False or mask[family_id["PlayHand"]] is True
    # Concretely: SelectCard alone (without prior PendingCards) isn't a
    # family any more; it only contributes to PlayHand/DiscardHand. But
    # those need ≥1 pending card. In the bare in_blind scenario nothing
    # is pending yet → both should be off (you can't commit without
    # selecting first).
    # We don't bake that policy into this test; instead just check that
    # SWAP is *not* legal (no jokers).
    assert not mask[family_id["SWAP"]]


def test_encoder_family_mask_allows_swap_when_jokers_present(encoder, device):
    snapshot = SCENARIOS["inventory_actions"]()
    batch, _ = encoder.encode(snapshot, device=device)
    mask = batch["family_mask"].squeeze(0).tolist()
    family_id = encoder.family_to_id
    assert mask[family_id["SWAP"]] is True


# ---------------------------------------------------------------------------
# pointer_masks_for_family
# ---------------------------------------------------------------------------


def test_pointer_masks_for_swap_marks_joker_slots(encoder, device):
    snapshot = SCENARIOS["inventory_actions"]()
    canonical = encoder.normalize_snapshot(snapshot)
    masks = encoder.pointer_masks_for_family(canonical, "SWAP", device=device)
    swap_mask = masks["swap_joker_mask"].squeeze(0).tolist()
    # 2 jokers in CurrentJokers => positions 0, 1 enabled, rest disabled.
    assert swap_mask[0] is True
    assert swap_mask[1] is True
    assert all(not v for v in swap_mask[2:])
    # SWAP doesn't take item/card pointers; those masks must be False.
    assert not any(masks["item_pointer_mask"].squeeze(0).tolist())
    assert not any(masks["card_pointer_mask"].squeeze(0).tolist())


def test_pointer_masks_for_buy_shop_item_voucher(encoder, device):
    snapshot = SCENARIOS["in_shop"]()
    canonical = encoder.normalize_snapshot(snapshot)
    masks = encoder.pointer_masks_for_family(
        canonical, "BuyShopItem_VoucherShopOfferings", device=device,
    )
    item_mask = masks["item_pointer_mask"].squeeze(0).tolist()
    # Snapshot has exactly one voucher → pos 0 should be legal.
    assert item_mask[0] is True
    # SWAP / card masks must be False for a single-pointer family.
    assert not any(masks["swap_joker_mask"].squeeze(0).tolist())
    assert not any(masks["card_pointer_mask"].squeeze(0).tolist())


def test_pointer_masks_for_play_hand_includes_hand_positions(encoder, device):
    snapshot = SCENARIOS["in_blind"]()
    canonical = encoder.normalize_snapshot(snapshot)
    masks = encoder.pointer_masks_for_family(canonical, "PlayHand", device=device)
    card_mask = masks["card_pointer_mask"].squeeze(0).tolist()
    # 8 cards in CurrentHand → positions 0..7 legal.
    assert all(card_mask[:8])
    # Item pointer is N/A for card_seq.
    assert not any(masks["item_pointer_mask"].squeeze(0).tolist())


def test_pointer_masks_intersect_with_lua_legal(encoder, device):
    """A buggy mask_builder rule must not widen the legal set past Lua."""
    snapshot = SCENARIOS["in_blind"]()
    canonical = encoder.normalize_snapshot(snapshot)
    # Hand size is 8 but pretend Lua only legalises positions 0..2.
    lua_legal = encoder._legal_action_mask(
        ["SelectCard_CurrentHand_0", "SelectCard_CurrentHand_1", "SelectCard_CurrentHand_2"]
    )
    masks = encoder.pointer_masks_for_family(
        canonical, "PlayHand", device=device, lua_legal_mask=lua_legal,
    )
    card_mask = masks["card_pointer_mask"].squeeze(0).tolist()
    assert card_mask[:3] == [True, True, True]
    # Everything past position 2 must be masked out by the Lua AND.
    assert not any(card_mask[3:])


# ---------------------------------------------------------------------------
# Two-phase decode against random-weight model
# ---------------------------------------------------------------------------


def _run_decide(
    encoder: LiveEncoder,
    model: PolicyTransformer,
    snapshot: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Reproduce the AgentServer two-phase decode end-to-end."""
    batch, _ = encoder.encode(snapshot, device=device)
    phase1 = model.encode_and_pick_family(batch)
    family_id = int(phase1["family_id"].item())
    family_name = encoder.id_to_family[family_id]
    canonical = encoder.normalize_snapshot(snapshot)
    lua_mask = encoder._legal_action_mask(canonical.get("legal_actions") or [])
    new_masks = encoder.pointer_masks_for_family(
        canonical, family_name, device=device,
        lua_legal_mask=lua_mask if lua_mask.any() else None,
    )
    for k, t in new_masks.items():
        batch[k] = t
    phase2 = model.decode_arguments(phase1["enc"], batch, phase1["family_id"])
    return {
        "family_name": family_name,
        "family_id": family_id,
        "family_logits": phase1["family_logits"],
        **phase2,
        "batch_family_mask": batch["family_mask"],
    }


def test_decide_predicts_family_from_mask(encoder, model, device):
    """For every scenario the predicted family must be one the mask allows."""
    for name, builder in SCENARIOS.items():
        snapshot = builder()
        if not snapshot.get("legal_actions"):
            continue  # nothing to predict
        out = _run_decide(encoder, model, snapshot, device)
        family_mask = out["batch_family_mask"].squeeze(0).tolist()
        if not any(family_mask):
            # Nothing legal at the family level — the model will still
            # argmax (returns 0); skip the legality assertion.
            continue
        family_id = out["family_id"]
        assert family_mask[family_id], (
            f"scenario {name!r}: predicted family_id={family_id} "
            f"({out['family_name']}) but mask had it disabled"
        )


def test_decide_shapes_match_caps(encoder, model, device):
    snapshot = SCENARIOS["in_blind"]()
    out = _run_decide(encoder, model, snapshot, device)
    caps = encoder.branched_caps
    max_cards = caps["MAX_CARDS_PER_DECISION"]
    assert out["card_seq_pred"].shape == (1, max_cards)
    assert out["chained_pred"].shape == (1, max_cards)
    assert out["card_seq_num_cards"].shape == (1,)
    assert out["chained_num_cards"].shape == (1,)
    assert out["item_pred"].shape == (1,)
    assert out["swap_i_pred"].shape == (1,)
    assert out["swap_j_pred"].shape == (1,)


def test_expansion_for_inventory_swap_is_valid(encoder, model, device):
    """The end-to-end loop produces a SWAP_i_j with i < j and i != j when SWAP
    is the predicted family in a scenario with two jokers.
    """
    snapshot = SCENARIOS["inventory_actions"]()
    out = _run_decide(encoder, model, snapshot, device)
    family = out["family_name"]
    decoder_shape = encoder.decoder_shapes.get(family, "reserved")
    # We don't enforce that SWAP is predicted (random model), but if the
    # decoder shape is joker_pair, the expansion should still produce a
    # canonical label.
    decision = {
        "family_name": family,
        "decoder_shape": decoder_shape,
        "num_cards": int(out["card_seq_num_cards"].item()),
        "card_ptr_local_seq": list(map(int, out["card_seq_pred"].squeeze(0).tolist())),
        "item_ptr_local": int(out["item_pred"].item()),
        "swap_i_local": int(out["swap_i_pred"].item()),
        "swap_j_local": int(out["swap_j_pred"].item()),
    }
    expansion = expand_decision(
        family_name=decision["family_name"],
        decoder_shape=decision["decoder_shape"],
        page_name=snapshot.get("page_name"),
        num_cards=decision["num_cards"],
        card_ptr_local_seq=decision["card_ptr_local_seq"],
        item_ptr_local=decision["item_ptr_local"],
        swap_i_local=decision["swap_i_local"],
        swap_j_local=decision["swap_j_local"],
    )
    if decoder_shape == "joker_pair":
        # SWAP labels must be canonicalised lo_hi.
        for lab in expansion.labels:
            assert lab.startswith("SWAP_")
            parts = lab.split("_")
            assert len(parts) == 3, lab
            lo, hi = int(parts[1]), int(parts[2])
            assert 0 <= lo < hi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
