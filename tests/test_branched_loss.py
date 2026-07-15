"""Unit tests for branched_loss.py."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from action_map import compute_action_map
from branched_loss import BranchedLoss, build_decoder_shape_lut
from family_map import compute_family_map


def _family_map():
    cfg = json.loads(Path("data/action_space_config.json").read_text(encoding="utf-8"))
    action_map = compute_action_map(cfg)
    return compute_family_map(action_map)


def _synthetic_out_and_batch(family_map, B=6, K_item=15, K_card=15, K_joker=10, T=5):
    fid = family_map["family_to_id"]
    family_id = torch.tensor(
        [
            fid["SelectBlind"],          # no_args
            fid["PlayHand"],             # card_seq
            fid["DiscardHand"],          # card_seq
            fid["BuyShopItem_PackShopOfferings"],  # single_ptr
            fid["UseConsumable_CurrentConsumables"],  # chained_cards
            fid["SWAP"],                  # joker_pair
        ],
        dtype=torch.long,
    )
    num_cards = torch.tensor([0, 3, 1, 0, 2, 0], dtype=torch.long)
    item_ptr = torch.tensor([-1, -1, -1, 2, 1, -1], dtype=torch.long)
    card_ptr_seq = torch.full((B, T), -1, dtype=torch.long)
    card_ptr_seq[1, :3] = torch.tensor([0, 1, 2])  # PlayHand
    card_ptr_seq[2, :1] = torch.tensor([3])         # DiscardHand
    card_ptr_seq[4, :2] = torch.tensor([4, 5])      # UseConsumable
    swap_i = torch.tensor([-1, -1, -1, -1, -1, 3], dtype=torch.long)
    swap_j = torch.tensor([-1, -1, -1, -1, -1, 7], dtype=torch.long)
    batch = {
        "family_id": family_id,
        "num_cards": num_cards,
        "item_ptr_local": item_ptr,
        "card_ptr_local_seq": card_ptr_seq,
        "swap_i_local": swap_i,
        "swap_j_local": swap_j,
    }

    n_families = family_map["n_families"]
    out = {
        "family_logits": torch.randn(B, n_families, requires_grad=True),
        "item_logits": torch.randn(B, K_item, requires_grad=True),
        "card_seq_num_cards_logits": torch.randn(B, 6, requires_grad=True),
        "card_seq_ptr_logits": torch.randn(B, T, K_card, requires_grad=True),
        "chained_num_cards_logits": torch.randn(B, 6, requires_grad=True),
        "chained_ptr_logits": torch.randn(B, T, K_card, requires_grad=True),
        "swap_i_logits": torch.randn(B, K_joker, requires_grad=True),
        "swap_j_logits": torch.randn(B, K_joker, requires_grad=True),
    }
    return out, batch


def test_decoder_shape_lut_matches_family_map():
    fm = _family_map()
    lut = build_decoder_shape_lut(fm)
    assert lut.shape == (fm["n_families"],)
    fid = fm["family_to_id"]
    # Spot checks
    assert int(lut[fid["StartNewRun"]]) == 5  # reserved
    assert int(lut[fid["SelectBlind"]]) == 0  # no_args
    assert int(lut[fid["PlayHand"]]) == 1     # card_seq
    assert int(lut[fid["BuyShopItem_PackShopOfferings"]]) == 2  # single_ptr
    assert int(lut[fid["UseConsumable_CurrentConsumables"]]) == 3  # chained_cards
    assert int(lut[fid["SWAP"]]) == 4  # joker_pair


def test_branched_loss_dispatches_per_shape():
    fm = _family_map()
    loss_fn = BranchedLoss(fm)
    out, batch = _synthetic_out_and_batch(fm)
    result = loss_fn(out, batch)

    # All shape buckets should be populated with the expected counts.
    assert result.n_valid["family"] == 6
    assert result.n_valid["card_seq_num_cards"] == 2  # PlayHand + DiscardHand
    assert result.n_valid["card_seq_ptr"] == 4        # 3 + 1 card-pointer positions
    assert result.n_valid["single_ptr_item"] == 1     # one BuyShop row
    assert result.n_valid["chained_item"] == 1
    assert result.n_valid["chained_num_cards"] == 1
    assert result.n_valid["chained_ptr"] == 2
    assert result.n_valid["swap_i"] == 1
    assert result.n_valid["swap_j"] == 1


def test_branched_loss_backward():
    fm = _family_map()
    loss_fn = BranchedLoss(fm)
    out, batch = _synthetic_out_and_batch(fm)
    result = loss_fn(out, batch)
    result.total.backward()
    # All input logits should have grad.
    for k, v in out.items():
        assert v.grad is not None, f"missing grad for {k}"


def test_branched_loss_zero_when_no_rows_of_shape():
    fm = _family_map()
    loss_fn = BranchedLoss(fm)
    # Build a batch where no row is SWAP.
    out, batch = _synthetic_out_and_batch(fm)
    batch["family_id"][5] = fm["family_to_id"]["SelectBlind"]
    batch["swap_i_local"][5] = -1
    batch["swap_j_local"][5] = -1
    result = loss_fn(out, batch)
    # SWAP buckets are zero with n_valid == 0.
    assert result.n_valid["swap_i"] == 0
    assert result.n_valid["swap_j"] == 0
    assert float(result.components["swap_i"].item()) == 0.0


def test_branched_loss_per_family_top1():
    fm = _family_map()
    loss_fn = BranchedLoss(fm)
    out, batch = _synthetic_out_and_batch(fm)
    diag = loss_fn.per_family_top1(out, batch, fm["id_to_family"])
    # Six unique families in this batch; each should have n==1.
    assert len(diag) == 6
    for fam, info in diag.items():
        assert info["n"] == 1
        assert 0.0 <= info["top1"] <= 1.0
