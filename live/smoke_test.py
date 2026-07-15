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

from live.emission_policy import expand_decision
from live.live_encoder import LiveEncoder
from model import ModelConfig, PolicyTransformer
from supplement_features import N_SUPPLEMENT, SUPPLEMENT_FEATURE_NAMES


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
    offerings = [
        {"class_id": 249, "object_type": "tarot", "zone": "PackOfferings",
         "position_in_zone": 0, "modifier": None, "edition": None,
         "seal": None, "card": None},
        {"class_id": 308, "object_type": "spectral", "zone": "PackOfferings",
         "position_in_zone": 1, "modifier": None, "edition": None,
         "seal": None, "card": None},
    ]
    return _envelope(
        request_id=7,
        page_name="In_TarotSpectral_Pack",
        state=_empty_state(hand_size_current=8, deck_remaining=44),
        objects=[*hand, *offerings],
        legal_actions=[
            "SkipPack",
            *[f"SelectCard_TarotSpectralHand_{i}" for i in range(8)],
            "SelectPackItem_PackOfferings_0",
            "SelectPackItem_PackOfferings_1",
        ],
    )


def _scenario_in_blind_pending_pair() -> dict[str, Any]:
    """In_Blind with two 7s already in PendingCards, exercising supplement_features.

    Used by the encoder pre-flight to check that ``make_pair = 1`` and that
    the scored-count derivations match Balatro hand rules (2 sevens score,
    everything else does not).
    """
    pending = [_card("PendingCards", 0, 6), _card("PendingCards", 1, 19)]   # 7♠, 7♥
    pool = [_card("CurrentHand", i, c) for i, c in enumerate([12, 25, 38, 51, 0, 13])]
    return _envelope(
        request_id=9,
        page_name="In_Blind",
        state=_empty_state(hand_size_current=8, deck_remaining=44),
        objects=[
            {"class_id": 370, "object_type": "blind", "zone": "BlindToken",
             "position_in_zone": 0, "modifier": None, "edition": None,
             "seal": None, "card": None},
            *pool,
            *pending,
        ],
        pending_cards=[_pending_view(c) for c in pending],
        legal_actions=[
            "PlayHand", "DiscardHand",
            *[f"SelectCard_CurrentHand_{i}" for i in range(6)],
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
    "blind_select":            _scenario_blind_select,
    "in_blind":                _scenario_in_blind,
    "in_blind_mid_select":     _scenario_in_blind_mid_select,
    "in_blind_pending_pair":   _scenario_in_blind_pending_pair,
    "cash_out":                _scenario_cash_out,
    "in_shop":                 _scenario_in_shop,
    "in_pack_joker":           _scenario_in_pack_joker,
    "in_pack_tarot":           _scenario_in_pack_tarot,
    "inventory_actions":       _scenario_inventory_actions,
}


def _encoder_preflight(encoder: LiveEncoder, device: torch.device) -> None:
    """Walk every scenario through the encoder and validate supplement_features.

    Shape check is unconditional. For the canonical "pending pair" scenario we
    also spot-check a few derived values so a regression in supplement_features
    is caught before we even touch the model.
    """
    print()
    print("=== supplement_features pre-flight ===")
    for name, build in SCENARIOS.items():
        snapshot = build()
        batch, _ = encoder.encode(snapshot, device=device)
        sf = batch.get("supplement_features")
        if sf is None:
            raise AssertionError(
                f"scenario {name!r}: tensorize_step did not emit supplement_features"
            )
        if tuple(sf.shape) != (1, N_SUPPLEMENT):
            raise AssertionError(
                f"scenario {name!r}: supplement_features shape {tuple(sf.shape)} != (1, {N_SUPPLEMENT})"
            )
        if sf.dtype != torch.float32:
            raise AssertionError(
                f"scenario {name!r}: supplement_features dtype {sf.dtype} != float32"
            )
        expected_history_steps = int(encoder.feature_config.get("HISTORY_STEPS", 32))
        if tuple(batch["history_step_mask"].shape) != (1, expected_history_steps):
            raise AssertionError(
                f"scenario {name!r}: history_step_mask shape {tuple(batch['history_step_mask'].shape)} != (1, {expected_history_steps})"
            )
        if bool(batch["history_step_mask"].any().item()):
            raise AssertionError(f"scenario {name!r}: fresh live encode should have empty history")
    # Spot-check the canonical pair scenario: 2 sevens in PendingCards.
    snapshot = _scenario_in_blind_pending_pair()
    batch, _ = encoder.encode(snapshot, device=device)
    sf = batch["supplement_features"].squeeze(0).cpu().numpy()

    def _bit(label: str) -> float:
        return float(sf[SUPPLEMENT_FEATURE_NAMES.index(label)])

    if _bit("selected_cards_make_pair") != 1.0:
        raise AssertionError("in_blind_pending_pair: expected make_pair=1.0")
    if _bit("selected_cards_make_high_card") != 0.0:
        raise AssertionError("in_blind_pending_pair: make_high_card should be 0")
    # Both sevens score → spade+heart counts = 1 each, debuff = 0.
    if _bit("selected_spade_scored_count") != 1.0:
        raise AssertionError("in_blind_pending_pair: spade_scored_count != 1")
    if _bit("selected_heart_scored_count") != 1.0:
        raise AssertionError("in_blind_pending_pair: heart_scored_count != 1")
    # Nothing else (K♠ at class_id 12 ace? class_id 6 is 7♠) should score.
    if _bit("selected_face_card_scored_count") != 0.0:
        raise AssertionError("in_blind_pending_pair: face_card_scored_count != 0")
    prior = encoder.build_step(_scenario_in_pack_tarot())
    prior["action"] = "SelectPackItem_PackOfferings_1"
    hist_batch, _ = encoder.encode(
        _scenario_in_blind(), device=device, history_steps=[prior]
    )
    if not bool(hist_batch["history_step_mask"][0, 0].item()):
        raise AssertionError("history pre-flight: expected most-recent history slot to be populated")
    if int(hist_batch["history_object_mask"][0, 0].sum().item()) == 0:
        raise AssertionError("history pre-flight: expected Tarot/Spectral PackOfferings history objects")
    print(f"  OK  every scenario emits supplement_features ({N_SUPPLEMENT},) float32")
    print(f"  OK  pair-scenario derived bits match Balatro rules")
    print("  OK  live history tensors populate prior PackOfferings context")


def _load_model(ckpt_path: Path, device: torch.device) -> PolicyTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["model_config"])
    if "history_steps" not in cfg_dict:
        cfg_dict.update(use_history_tokens=False, use_tracked_deck_tokens=True)
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                         if k in ModelConfig.__dataclass_fields__})
    model = PolicyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"loaded {ckpt_path.name} epoch={ckpt.get('epoch')} "
        f"val_top1={ckpt.get('val_top1')} "
        f"family_map={ckpt.get('family_map_version')}"
    )
    return model


@torch.no_grad()
def _decide_branched(
    encoder: LiveEncoder,
    model: PolicyTransformer,
    snapshot: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Branched two-phase decode mirroring ``AgentServer._decide_super_step``."""
    batch, _ = encoder.encode(snapshot, device=device)
    phase1 = model.encode_and_pick_family(batch)
    family_id = int(phase1["family_id"].item())
    family_name = encoder.id_to_family[family_id]
    decoder_shape = encoder.decoder_shapes.get(family_name, "reserved")

    canonical_snap = encoder.normalize_snapshot(snapshot)
    lua_mask = encoder._legal_action_mask(canonical_snap.get("legal_actions") or [])
    new_masks = encoder.pointer_masks_for_family(
        canonical_snap, family_name, device=device,
        lua_legal_mask=lua_mask if lua_mask.any() else None,
    )
    for k, t in new_masks.items():
        batch[k] = t
    phase2 = model.decode_arguments(phase1["enc"], batch, phase1["family_id"])

    family_logits = phase1["family_logits"].squeeze(0)
    family_probs = torch.softmax(family_logits, dim=-1)
    top_n = min(5, int((family_logits > -float("inf")).sum().item()) or 1)
    top_idx = torch.topk(family_probs, top_n).indices.tolist()
    top5 = [
        (encoder.id_to_family[int(i)], float(family_probs[int(i)].item()))
        for i in top_idx
    ]

    decision: dict[str, Any] = {
        "family_id": family_id,
        "family_name": family_name,
        "decoder_shape": decoder_shape,
        "family_top5": top5,
        "item_ptr_local": int(phase2["item_pred"].item()),
        "swap_i_local": int(phase2["swap_i_pred"].item()),
        "swap_j_local": int(phase2["swap_j_pred"].item()),
    }
    if decoder_shape == "card_seq":
        n = max(1, min(int(phase2["card_seq_num_cards"].item()),
                       model.cfg.max_cards_per_decision))
        seq = phase2["card_seq_pred"].squeeze(0).tolist()
        decision["num_cards"] = n
        decision["card_ptr_local_seq"] = list(map(int, seq[:n]))
    elif decoder_shape == "chained_cards":
        n = max(0, min(int(phase2["chained_num_cards"].item()),
                       model.cfg.max_cards_per_decision))
        seq = phase2["chained_pred"].squeeze(0).tolist()
        decision["num_cards"] = n
        decision["card_ptr_local_seq"] = list(map(int, seq[:n]))
    else:
        decision["num_cards"] = 0
        decision["card_ptr_local_seq"] = []
    return decision


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

    # Pre-flight: every scenario's encoded batch must carry supplement_features
    # with the contracted shape/dtype. We deliberately do this BEFORE loading
    # the checkpoint so the assertion still runs in a fresh repo (or after a
    # `n_supplement` bump) when the saved weights aren't yet compatible.
    _encoder_preflight(encoder, device)

    model = _load_model(args.checkpoint, device)

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    n_unresolved = 0
    n_illegal_first = 0
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

        legal_set = set(snapshot.get("legal_actions") or [])
        n_legal = len(legal_set)
        if not legal_set:
            print("  no legal actions — skipping decode (would have nothing to pick)")
            continue

        decision = _decide_branched(encoder, model, snapshot, device)
        expansion = expand_decision(
            family_name=decision["family_name"],
            decoder_shape=decision["decoder_shape"],
            page_name=snapshot.get("page_name"),
            num_cards=int(decision["num_cards"]),
            card_ptr_local_seq=decision["card_ptr_local_seq"],
            item_ptr_local=decision["item_ptr_local"],
            swap_i_local=decision["swap_i_local"],
            swap_j_local=decision["swap_j_local"],
        )
        print(f"  page={snapshot['page_name']}  n_legal={n_legal}")
        print(
            f"  family: {decision['family_name']:55s}  "
            f"shape={decision['decoder_shape']}"
        )
        print(f"  family_top5:")
        for fam, p in decision["family_top5"]:
            print(f"    {fam:55s}  p={p:.3f}")
        # The plan unrolls into 1..N labels. Only the FIRST label has to
        # be in Lua's current legal set — later labels become legal as
        # their predecessor (e.g. SelectCard) executes and Lua sends a
        # new snapshot. AgentServer's pop_next() validates each one in
        # turn before emitting.
        print(f"  unrolled plan ({len(expansion.labels)} labels):")
        for i, lab in enumerate(expansion.labels):
            if i == 0:
                tag = "" if lab in legal_set else "  [first label NOT in Lua's current legal set]"
            else:
                tag = "  (becomes legal after predecessor)"
            print(f"    {lab}{tag}")
        if expansion.labels and expansion.labels[0] not in legal_set:
            n_illegal_first += 1

    print()
    if n_unresolved:
        print(f"FAIL: {n_unresolved} legal labels could not be resolved against the action map.")
        raise SystemExit(1)
    if n_illegal_first:
        print(
            f"WARNING: {n_illegal_first} scenario(s) had a FIRST label outside Lua's "
            "current legal set. AgentServer will fall back gracefully, but the "
            "decoder picked something the live game wouldn't accept."
        )
    print("OK: branched decoder produced legal label plans for all scenarios.")


if __name__ == "__main__":
    main()
