#!/usr/bin/env python3
"""
smoke_test.py
=============
Standalone end-to-end sanity check for the live integration. Synthesizes a
minimal snapshot for each page_name (Blind_Select, In_Blind, In_Shop, ...),
runs it through ``LiveEncoder`` + the trained model, and prints the chosen
action + top-5. Use this before launching the modified Balatro to confirm
the inference pipeline can load the checkpoint and produce sensible output.

Usage:
    python live/smoke_test.py [--checkpoint artifacts/checkpoints/best.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import ModelConfig, PolicyTransformer
from live.live_encoder import LiveEncoder


def _empty_state(**overrides) -> dict[str, Any]:
    """Default OCR-equivalent numerics."""
    base = dict(
        hands_left=4, discards_left=3, dollars=4, ante=1, round=1,
        deck_remaining=52, deck_total=52, round_score=0, cash_out=0,
        reroll_price=5, consumables_current=0, consumables_total=2,
        jokers_current=0, jokers_total=5, hand_size_current=0, hand_size_total=8,
    )
    base.update(overrides)
    return base


def _empty_pstate(**overrides) -> dict[str, Any]:
    base = dict(
        stake=268,                   # Gold Stake class_id
        first_hand=True,
        first_discard=True,
        is_boss_blind_rerolled=False,
        skips=0,
        hands_played=0,
        unused_discards=0,
        ecto_minus=1,
        last_tarot_planet=None,
        ante_boss_blind=None,
        small_status=0,
        big_status=0,
        vouchers_redeemed=[],
        bosses_used=[],
        tracked_deck_cards=[],
        hands={
            name: {"level": 1, "played": 0, "played_this_round": 0}
            for name in (
                "Flush Five","Flush House","Five of a Kind","Straight Flush",
                "Four of a Kind","Full House","Flush","Straight",
                "Three of a Kind","Two Pair","Pair","High Card",
            )
        },
        deck={
            "class_id": 65,           # b_red
            "is_magic": False, "is_nebula": False, "is_abandoned": False,
            "is_checkered": False, "is_zodiac": False, "is_erratic": False,
        },
        deck_modifiers={
            "no_face_cards_start": False,
            "spades_hearts_only_start": False,
            "randomized_starting_deck": False,
        },
    )
    base.update(overrides)
    return base


def _card(zone: str, pos: int, class_id: int) -> dict[str, Any]:
    suit_index = class_id // 13
    rank_index = class_id % 13
    suit = ["Spades","Hearts","Diamonds","Clubs"][suit_index]
    rank_name = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"][rank_index]
    return {
        "class_id": class_id,
        "object_type": "card",
        "zone": zone,
        "position_in_zone": pos,
        "modifier": None, "edition": None, "seal": None,
        "card": {
            "rank": rank_name, "rank_index": rank_index,
            "suit": suit, "suit_index": suit_index,
            "is_ace": rank_index == 0,
            "is_face": rank_index >= 10 and rank_index <= 12,
        },
    }


def _pending_view(card: dict[str, Any]) -> dict[str, Any]:
    """Strip zone/position to match the snapshot.pending_cards top-level shape."""
    return {
        "class_id": card["class_id"],
        "object_type": card["object_type"],
        "modifier": card.get("modifier"),
        "edition": card.get("edition"),
        "seal": card.get("seal"),
        "card": card.get("card"),
    }


def _envelope(**kwargs) -> dict[str, Any]:
    """Build a baseline snapshot envelope; callers fill in objects/state/etc."""
    env: dict[str, Any] = {
        "request_id": kwargs.get("request_id", 0),
        "schema_version": "live/2.0.0",
        "page_name": kwargs["page_name"],
        "source_kind": None,
        "action_subtype": None,
        "state": kwargs.get("state") or _empty_state(),
        "objects": kwargs.get("objects") or [],
        "pending_cards": kwargs.get("pending_cards") or [],
        "target_zone": None,
        "target_position": None,
        "persistent_state": kwargs.get("persistent_state") or _empty_pstate(),
        "legal_actions": kwargs.get("legal_actions") or [],
        "meta": {"game_state_id": 0, "game_stage_id": 0, "sent_at_real_time": 0.0, "run_id": 0},
    }
    return env


def _scenario_blind_select() -> dict[str, Any]:
    return _envelope(
        request_id=1,
        page_name="Blind_Select",
        objects=[
            {"class_id": 65, "object_type": "deck", "zone": "CurrentDeck",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 268, "object_type": "stake", "zone": "CurrentStake",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 370, "object_type": "blind", "zone": "BlindOfferingsNext",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
        ],
        legal_actions=["SelectBlind", "SkipBlind"],
    )


def _scenario_in_blind() -> dict[str, Any]:
    # 8 unhighlighted cards in hand, none pending yet.
    pool = [_card("CurrentHand", i, [0, 13, 26, 39, 1, 14, 27, 40][i]) for i in range(8)]
    return _envelope(
        request_id=2,
        page_name="In_Blind",
        state=_empty_state(hand_size_current=8, deck_remaining=44),
        objects=[
            {"class_id": 370, "object_type": "blind", "zone": "BlindToken",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            *pool,
        ],
        legal_actions=[f"SelectCard_CurrentHand_{i}" for i in range(8)],
    )


def _scenario_in_blind_mid_select() -> dict[str, Any]:
    # Two cards already pending; the remaining 6 unhighlighted ones renumber
    # to positions 0..5 in CurrentHand (granularize.py 3.0 dynamic-pool rule).
    pending_cards = [_card("PendingCards", 0, 0), _card("PendingCards", 1, 13)]
    pool = [_card("CurrentHand", i, c) for i, c in enumerate([26, 39, 1, 14, 27, 40])]
    return _envelope(
        request_id=3,
        page_name="In_Blind",
        state=_empty_state(hand_size_current=8, deck_remaining=44),
        objects=[
            {"class_id": 370, "object_type": "blind", "zone": "BlindToken",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            *pool,
            *pending_cards,
        ],
        pending_cards=[_pending_view(c) for c in pending_cards],
        legal_actions=[
            "PlayHand", "DiscardHand",
            *[f"SelectCard_CurrentHand_{i}" for i in range(6)],
        ],
    )


def _scenario_cash_out() -> dict[str, Any]:
    return _envelope(
        request_id=4,
        page_name="Cash_Out",
        state=_empty_state(cash_out=12, round_score=300),
        objects=[
            {"class_id": 65, "object_type": "deck", "zone": "CurrentDeck",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
        ],
        legal_actions=["CashOut"],
    )


def _scenario_in_shop() -> dict[str, Any]:
    # Top-shelf: joker at 0, consumable at 1 (BuyAndUseShopConsumable_1 only).
    # Plus one voucher and one pack.
    return _envelope(
        request_id=5,
        page_name="In_Shop",
        state=_empty_state(dollars=12, jokers_current=0),
        objects=[
            {"class_id": 151, "object_type": "joker", "zone": "TopShelfShopOfferings",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 244, "object_type": "planet", "zone": "TopShelfShopOfferings",
             "position_in_zone": 1, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 320, "object_type": "voucher", "zone": "VoucherShopOfferings",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": None, "object_type": "pack", "zone": "PackShopOfferings",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None, "center_key": "p_arcana_normal_1"},
        ],
        legal_actions=[
            "LeaveShop", "RerollShop",
            "BuyShopItem_TopShelfShopOfferings_0",
            "BuyShopItem_TopShelfShopOfferings_1",
            "BuyAndUseShopConsumable_TopShelfShopOfferings_1",
            "BuyShopItem_VoucherShopOfferings_0",
            "BuyShopItem_PackShopOfferings_0",
        ],
    )


def _scenario_in_pack_joker() -> dict[str, Any]:
    # Joker/Standard/Planet pack offers 2 jokers; pick one or skip.
    return _envelope(
        request_id=6,
        page_name="In_JokerStandardPlanet_Pack",
        objects=[
            {"class_id": 151, "object_type": "joker", "zone": "PackOfferings",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 152, "object_type": "joker", "zone": "PackOfferings",
             "position_in_zone": 1, "modifier": None, "edition": None,
             "seal": None, "card": None},
        ],
        legal_actions=[
            "SkipPack",
            "SelectPackItem_PackOfferings_0",
            "SelectPackItem_PackOfferings_1",
        ],
    )


def _scenario_in_pack_tarot() -> dict[str, Any]:
    # Tarot/spectral pack — the player's hand sits under TarotSpectralHand
    # while the pack consumables are picked via SelectPackItem.
    hand = [_card("TarotSpectralHand", i, c) for i, c in enumerate([0, 13, 26, 39, 1, 14, 27, 40])]
    return _envelope(
        request_id=7,
        page_name="In_TarotSpectral_Pack",
        state=_empty_state(hand_size_current=8, deck_remaining=44),
        objects=hand,
        legal_actions=[
            "SkipPack",
            *[f"SelectCard_TarotSpectralHand_{i}" for i in range(8)],
        ],
    )


def _scenario_inventory_actions() -> dict[str, Any]:
    # In_Blind scenario but with jokers + consumables surfaced so we exercise
    # SellItem_<Zone>_i, UseConsumable_CurrentConsumables_i, SWAP_i_j.
    pool = [_card("CurrentHand", i, [0, 13, 26, 39, 1, 14, 27, 40][i]) for i in range(8)]
    return _envelope(
        request_id=8,
        page_name="In_Blind",
        state=_empty_state(hand_size_current=8, deck_remaining=44, jokers_current=2, consumables_current=1),
        objects=[
            {"class_id": 370, "object_type": "blind", "zone": "BlindToken",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            {"class_id": 151, "object_type": "joker", "zone": "CurrentJokers",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None, "stickers": [], "is_debuffed": False},
            {"class_id": 152, "object_type": "joker", "zone": "CurrentJokers",
             "position_in_zone": 1, "modifier": None, "edition": None,
             "seal": None, "card": None, "stickers": [], "is_debuffed": False},
            {"class_id": 244, "object_type": "consumable", "zone": "CurrentConsumables",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            *pool,
        ],
        legal_actions=[
            *[f"SelectCard_CurrentHand_{i}" for i in range(8)],
            "UseConsumable_CurrentConsumables_0",
            "SellItem_CurrentJokers_0", "SellItem_CurrentJokers_1",
            "SellItem_CurrentConsumables_0",
            "SWAP_0_1",
        ],
    )


SCENARIOS = {
    "blind_select":          _scenario_blind_select,
    "in_blind":              _scenario_in_blind,
    "in_blind_mid_select":   _scenario_in_blind_mid_select,
    "cash_out":              _scenario_cash_out,
    "in_shop":               _scenario_in_shop,
    "in_pack_joker":         _scenario_in_pack_joker,
    "in_pack_tarot":         _scenario_in_pack_tarot,
    "inventory_actions":     _scenario_inventory_actions,
}


def _load_model(ckpt_path: Path, device: torch.device) -> PolicyTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = PolicyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {ckpt_path.name} epoch={ckpt.get('epoch')} val_top1={ckpt.get('val_top1')}")
    return model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=_REPO_ROOT / "artifacts" / "checkpoints" / "best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--scenario", default="all",
                    choices=("all",) + tuple(SCENARIOS.keys()))
    ap.add_argument("--print-snapshot", action="store_true",
                    help="Print the synthetic snapshot dict before encoding.")
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"device: {device}")

    encoder = LiveEncoder()
    model = _load_model(args.checkpoint, device)

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    n_unresolved = 0
    for name in names:
        print()
        print(f"=== scenario: {name} ===")
        snapshot = SCENARIOS[name]()
        if args.print_snapshot:
            print(json.dumps(snapshot, indent=2))

        # Pre-flight: every legal label must exist in label_to_index. Catches
        # zone-typo regressions in agent_bridge.lua / smoke scenarios early.
        missing = [
            lab for lab in snapshot.get("legal_actions") or []
            if lab not in encoder.label_to_index
        ]
        if missing:
            n_unresolved += len(missing)
            print(f"  WARNING: {len(missing)} legal labels not in label_to_index: {missing[:5]}")

        batch, legal_mask = encoder.encode(snapshot, device=device)
        with torch.no_grad():
            logits = model(batch).squeeze(0)  # (N_ACTIONS,)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        n_legal = int(legal_mask.sum())
        top5 = sorted(
            ((float(probs[i]), encoder.label_for_index(i)) for i in range(probs.shape[0])
             if legal_mask[i]),
            reverse=True,
        )[:5]
        print(f"  page={snapshot['page_name']}  n_legal={n_legal}")
        if not top5:
            print("  no legal actions found — model would have nothing to pick")
            continue
        print(f"  argmax: {top5[0][1]}  p={top5[0][0]:.3f}")
        for p, lab in top5:
            print(f"    {lab:30s}  p={p:.3f}")

    print()
    if n_unresolved:
        print(f"FAIL: {n_unresolved} legal labels could not be resolved against the action map.")
        raise SystemExit(1)
    print("OK: all scenarios produced legal-mask-covered top-K predictions.")


if __name__ == "__main__":
    main()
