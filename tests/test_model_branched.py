"""Unit tests for the branched PolicyTransformer.

Builds a tiny synthetic batch to validate output shapes, mask
application, and gradient flow through each decoder shape.
"""
from __future__ import annotations

import torch

from model import PolicyTransformer, load_model_config


def _make_dummy_batch(cfg, batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Construct a minimal batch covering every required input channel."""
    s = cfg.vocab_sizes
    B = batch_size
    n_obj = cfg.max_objects
    n_deck = cfg.max_deck_cards
    n_hist = cfg.history_steps
    n_hobj = cfg.history_objects_per_step

    def _z(shape, dtype=torch.long):
        return torch.zeros(shape, dtype=dtype)

    def _ones(shape, dtype=torch.bool):
        return torch.ones(shape, dtype=dtype)

    x: dict[str, torch.Tensor] = {
        "page_id": torch.tensor([4, 6, 7, 1], dtype=torch.long),  # In_Blind, In_Shop, In_TarotSpectral_Pack, Blind_Select
        "deck_class_id": _z((B,)),
        "stake_class_id": _z((B,)),
        "last_tarot_planet_class_id": _z((B,)),
        "ante_boss_blind_class_id": _z((B,)),
        "small_status_id": _z((B,)),
        "big_status_id": _z((B,)),
        # Numerics
        "ocr_numeric": torch.zeros((B, cfg.n_ocr), dtype=torch.float32),
        "state_numeric": torch.zeros((B, cfg.n_state_num), dtype=torch.float32),
        "ocr_valid": _ones((B, cfg.n_ocr)),
        "flags": _ones((B, cfg.n_flags)),
        "hand_levels": torch.zeros((B, cfg.n_hands), dtype=torch.float32),
        "hand_played": torch.zeros((B, cfg.n_hands), dtype=torch.float32),
        "hand_played_this_round": torch.zeros((B, cfg.n_hands), dtype=torch.float32),
        "vouchers_redeemed": _ones((B, cfg.n_vouchers)),
        "bosses_used": _ones((B, cfg.n_bosses)),
        "supplement_features": torch.zeros((B, cfg.n_supplement), dtype=torch.float32),
        # Objects (5 objects per row, each at a distinct zone/position)
        "object_class_id": _z((B, n_obj)),
        "object_modifier_id": _z((B, n_obj)),
        "object_edition_id": _z((B, n_obj)),
        "object_seal_id": _z((B, n_obj)),
        "object_rank_id": _z((B, n_obj)),
        "object_suit_id": _z((B, n_obj)),
        "object_object_type_id": _z((B, n_obj)),
        "object_zone_id": _z((B, n_obj)),
        "object_position": _z((B, n_obj)),
        "object_is_debuffed": _z((B, n_obj), dtype=torch.bool),
        "object_sticker_rental": _z((B, n_obj), dtype=torch.bool),
        "object_sticker_perishable": _z((B, n_obj), dtype=torch.bool),
        "object_sticker_eternal": _z((B, n_obj), dtype=torch.bool),
        "object_mask": _z((B, n_obj), dtype=torch.bool),
        # Tracked deck (disabled in cfg, but channels still need to exist
        # for compatibility — left zeroed/false).
        "deck_card_class_id": _z((B, n_deck)),
        "deck_card_modifier_id": _z((B, n_deck)),
        "deck_card_edition_id": _z((B, n_deck)),
        "deck_card_seal_id": _z((B, n_deck)),
        "deck_card_rank_id": _z((B, n_deck)),
        "deck_card_suit_id": _z((B, n_deck)),
        "deck_card_mask": _z((B, n_deck), dtype=torch.bool),
        # History
        "history_action_id": _z((B, n_hist)),
        "history_page_id": _z((B, n_hist)),
        "history_recency": _z((B, n_hist)),
        "history_target_zone_id": _z((B, n_hist)),
        "history_target_position": _z((B, n_hist)),
        "history_swap_i": _z((B, n_hist)),
        "history_swap_j": _z((B, n_hist)),
        "history_ocr_numeric": torch.zeros((B, n_hist, cfg.n_ocr), dtype=torch.float32),
        "history_ocr_valid": _ones((B, n_hist, cfg.n_ocr)),
        "history_step_mask": _z((B, n_hist), dtype=torch.bool),
        "history_object_mask": _z((B, n_hist, n_hobj), dtype=torch.bool),
        "history_object_class_id": _z((B, n_hist, n_hobj)),
        "history_object_modifier_id": _z((B, n_hist, n_hobj)),
        "history_object_edition_id": _z((B, n_hist, n_hobj)),
        "history_object_seal_id": _z((B, n_hist, n_hobj)),
        "history_object_rank_id": _z((B, n_hist, n_hobj)),
        "history_object_suit_id": _z((B, n_hist, n_hobj)),
        "history_object_object_type_id": _z((B, n_hist, n_hobj)),
        "history_object_zone_id": _z((B, n_hist, n_hobj)),
        "history_object_position": _z((B, n_hist, n_hobj)),
        "history_object_is_debuffed": _z((B, n_hist, n_hobj), dtype=torch.bool),
        "history_object_sticker_rental": _z((B, n_hist, n_hobj), dtype=torch.bool),
        "history_object_sticker_perishable": _z((B, n_hist, n_hobj), dtype=torch.bool),
        "history_object_sticker_eternal": _z((B, n_hist, n_hobj), dtype=torch.bool),
        # Branched targets / masks
        "family_id": torch.tensor([8, 11, 17, 1], dtype=torch.long),  # PlayHand, BuyShopItem_Pack, SelectPackItem, SelectBlind
        "family_mask": torch.ones((B, cfg.n_families), dtype=torch.bool),
        "item_pointer_mask": torch.ones((B, cfg.max_item_zone_size), dtype=torch.bool),
        "card_pointer_mask": torch.ones((B, cfg.max_card_zone_size), dtype=torch.bool),
        "swap_joker_mask": torch.ones((B, cfg.max_joker_slots), dtype=torch.bool),
        "item_ptr_local": torch.tensor([-1, 2, 1, -1], dtype=torch.long),
        "card_ptr_local_seq": torch.tensor(
            [
                [0, 1, 2, -1, -1],  # PlayHand: 3 cards
                [-1, -1, -1, -1, -1],
                [3, -1, -1, -1, -1],  # SelectPackItem with 1 card
                [-1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
        ),
        "swap_i_local": torch.tensor([-1, -1, -1, -1], dtype=torch.long),
        # v1 back-compat (legacy flat head will reject when flat_head=False).
        "action_mask": torch.ones((B, cfg.n_actions), dtype=torch.bool),
        "target_action_id": torch.zeros((B,), dtype=torch.long),
    }
    # Mark a few objects as valid with distinct (zone, position) coords.
    for b in range(B):
        x["object_mask"][b, :5] = True
        # Slot 0..4 each have a distinct zone vocab id (CurrentHand=7, CurrentJokers=8, ...)
        x["object_zone_id"][b, :5] = torch.tensor([7, 7, 8, 13, 12], dtype=torch.long)
        x["object_position"][b, :5] = torch.tensor([0, 1, 0, 2, 1], dtype=torch.long)
    return x


def test_branched_forward_shapes():
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = False
    model = PolicyTransformer(cfg)
    model.eval()
    x = _make_dummy_batch(cfg, batch_size=4)
    with torch.no_grad():
        out = model(x)
    B = 4
    assert out["family_logits"].shape == (B, cfg.n_families)
    assert out["item_logits"].shape == (B, cfg.max_item_zone_size)
    assert out["card_seq_num_cards_logits"].shape == (B, cfg.max_cards_per_decision + 1)
    assert out["card_seq_ptr_logits"].shape == (
        B,
        cfg.max_cards_per_decision,
        cfg.max_card_zone_size,
    )
    assert out["chained_num_cards_logits"].shape == (B, cfg.max_cards_per_decision + 1)
    assert out["chained_ptr_logits"].shape == (
        B,
        cfg.max_cards_per_decision,
        cfg.max_card_zone_size,
    )
    assert out["swap_i_logits"].shape == (B, cfg.max_joker_slots)
    assert out["swap_j_logits"].shape == (B, cfg.max_joker_slots)


def test_family_logits_masked_to_minus_inf():
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = False
    model = PolicyTransformer(cfg)
    model.eval()
    x = _make_dummy_batch(cfg)
    # Forbid family 0 (StartNewRun) and family 5 (SkipPack) in row 0.
    x["family_mask"][0, 0] = False
    x["family_mask"][0, 5] = False
    with torch.no_grad():
        out = model(x)
    assert torch.isinf(out["family_logits"][0, 0]) and out["family_logits"][0, 0] < 0
    assert torch.isinf(out["family_logits"][0, 5]) and out["family_logits"][0, 5] < 0


def test_already_picked_card_excluded_in_card_seq():
    """Once a card is teacher-forced at step 0, step 1 must mask it."""
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = False
    model = PolicyTransformer(cfg)
    model.eval()
    x = _make_dummy_batch(cfg)
    # Force PlayHand teacher labels to be [2, 0, -1, -1, -1] for row 0
    # so that after picking position 2, position 2 must be masked at step 1.
    x["card_ptr_local_seq"][0] = torch.tensor([2, 0, -1, -1, -1])
    with torch.no_grad():
        out = model(x)
    step1 = out["card_seq_ptr_logits"][0, 1]
    assert torch.isinf(step1[2]) and step1[2] < 0


def test_swap_j_excludes_swap_i():
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = False
    model = PolicyTransformer(cfg)
    model.eval()
    x = _make_dummy_batch(cfg)
    x["swap_i_local"] = torch.tensor([3, 0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    # For each row, j_logits at position i must be -inf.
    for b in range(4):
        i = int(x["swap_i_local"][b])
        assert torch.isinf(out["swap_j_logits"][b, i]) and out["swap_j_logits"][b, i] < 0


def test_branched_backward_runs_without_error():
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = False
    model = PolicyTransformer(cfg)
    model.train()
    x = _make_dummy_batch(cfg)
    out = model(x)
    # Sum a few logits to make a tiny pseudo-loss; just verifying the
    # autograd graph is connected end-to-end.
    loss = (
        out["family_logits"].nan_to_num(0.0).sum()
        + out["item_logits"].nan_to_num(0.0).sum()
        + out["card_seq_ptr_logits"].nan_to_num(0.0).sum()
        + out["chained_ptr_logits"].nan_to_num(0.0).sum()
        + out["swap_i_logits"].nan_to_num(0.0).sum()
        + out["swap_j_logits"].nan_to_num(0.0).sum()
    )
    loss.backward()
    has_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    assert has_grad > 0


def test_flat_head_optional():
    cfg = load_model_config(n_actions=134)
    cfg.flat_head = True
    model = PolicyTransformer(cfg)
    model.eval()
    x = _make_dummy_batch(cfg)
    with torch.no_grad():
        out = model(x)
    assert "action_logits" in out
    assert out["action_logits"].shape == (4, cfg.n_actions)
