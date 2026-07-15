#!/usr/bin/env python3
"""
model.py
========
Policy transformer for Balatro behavior cloning (branched autoregressive
edition).

Architecture
------------
We project three heterogeneous input groups into ``d_model`` token
embeddings and run a transformer encoder over the resulting token set:

1. **Global token** — one token aggregating all per-step scalars:
   global categoricals (page_id, source_kind_id, ...), per-hand arrays
   (level / played / played_this_round), per-OCR numerics, persistent
   state numerics, boolean flags, and the multi-hot voucher / boss
   inventories.
2. **Object tokens** — one token per non-padded object in the snapshot
   (capped at ``MAX_OBJECTS_PER_STEP``). Encodes class_id, zone, type,
   modifier/edition/seal, rank/suit, debuff, sticker bits.
3. **Tracked-deck tokens** — one token per non-padded deck card
   (capped at ``TRACKED_DECK_CAP``). Disabled by default; replaced by
   history tokens (see ``HistoryStepEncoder``).
4. **History tokens** — one summary token per prior decision + per-object
   history tokens, providing temporal context.

A learned ``[CLS]`` token is prepended; the transformer is run, and the
output sequence is split into:

- ``cls_out``: pooled global representation, used to drive the family
  head and decoder initial states.
- ``object_out``: per-object refined embeddings, gathered into per-zone
  fixed-size buffers via ``object_zone_id`` + ``object_position``.

The flat ``N_ACTIONS`` classifier has been REPLACED with a branched
autoregressive policy (per family_map.py decoder shapes):

- **Family head** — Linear(d_model -> n_families), masked by
  ``family_mask``.
- **No-args** families need no further heads.
- **card_seq** (PlayHand, DiscardHand): autoregressive pointer over
  ``CurrentHand`` objects + a ``num_cards`` head, both initialized
  from ``cls_out``.
- **single_ptr** (Buy / Sell / BuyAndUse): pointer over the family's
  ``ItemZone`` tokens.
- **chained_cards** (UseConsumable, SelectPackItem): item pointer ->
  num_cards (conditioned on picked item) -> autoregressive card-seq
  pointer (conditioned on item embedding).
- **joker_pair** (SWAP): two-step autoregressive pointer over
  ``CurrentJokers``. ``j != i`` enforced via running mask; emission
  canonicalised to ``i < j``.

All decoders are TEACHER-FORCED during training (use the ground-truth
``card_ptr_local_seq`` / ``item_ptr_local`` / ``swap_*_local`` to update
the decoder state). At inference time the caller (live agent /
``decode_greedy``) feeds the model's own argmax back in.

Embedding tables
----------------
Vocab sizes are read from ``artifacts/vocab.json`` to keep model and
data fully aligned. ``class_id`` uses identity encoding (vocab_index ==
class_id) and shares one embedding table across object_class_id and
deck_card_class_id. Other categoricals each get their own table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from supplement_features import N_SUPPLEMENT


@dataclass
class ModelConfig:
    n_actions: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    cat_embed_dim: int = 32
    max_objects: int = 32
    max_deck_cards: int = 75
    history_steps: int = 32
    history_objects_per_step: int = 16
    use_history_tokens: bool = True
    use_tracked_deck_tokens: bool = False
    history_step_dropout: float = 0.15
    history_object_dropout: float = 0.05

    # Branched-policy capacities. n_families and the zone caps must agree
    # with the artifacts written by tensorize.py / family_map.py.
    n_families: int = 19
    max_item_zone_size: int = 15
    max_card_zone_size: int = 15
    max_joker_slots: int = 10
    max_cards_per_decision: int = 5

    # Optional flat-head retention for legacy checkpoints. When True the
    # model exposes a flat ``head`` on top of cls_out and ``forward(x)``
    # ALSO returns flat ``action_logits`` for back-compat. The branched
    # heads are always present regardless.
    flat_head: bool = False

    # Vocab sizes -- filled from artifacts/vocab.json.
    vocab_sizes: dict[str, int] = field(default_factory=dict)

    # Per-step shape information (for global feature MLP).
    n_ocr: int = 16
    n_state_num: int = 4
    n_flags: int = 12
    n_hands: int = 12
    n_vouchers: int = 32
    n_bosses: int = 28
    n_supplement: int = N_SUPPLEMENT

    # Pre-built zone-id lookups (built at load_model_config time).
    # Each is a length-n_families list of int zone vocab ids.
    # -1 means "no zone" / "n/a" for that family.
    family_to_item_zone_id: list[int] = field(default_factory=list)
    family_to_card_zone_default_id: list[int] = field(default_factory=list)
    family_to_card_zone_pack_id: list[int] = field(default_factory=list)
    pack_page_id: int = -1


def load_model_config(
    n_actions: int,
    vocab_path: Path = Path("artifacts/vocab.json"),
    feature_config_path: Path = Path("artifacts/feature_config.json"),
    family_map_path: Path = Path("data/family_map.json"),
    **overrides: Any,
) -> ModelConfig:
    """Build a ModelConfig from on-disk artifacts.

    Reads ``vocab.json`` for embedding sizes + the zone id table,
    ``feature_config.json`` for input shapes + branched capacities,
    and ``family_map.json`` for the per-family zone lookups used by
    the pointer heads.
    """
    vocab_root = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = vocab_root["vocabularies"]
    fc = json.loads(feature_config_path.read_text(encoding="utf-8"))
    fmap = json.loads(family_map_path.read_text(encoding="utf-8"))
    sizes = {name: int(entry["size"]) for name, entry in vocab.items()}

    # Zone vocab lookup. -1 sentinel means "no zone" for this family
    # (no_args / reserved / shapes that don't use this slot).
    zone_to_id = {k: int(v) for k, v in vocab["zone"]["value_to_index"].items()}
    page_to_id = {k: int(v) for k, v in vocab["page"]["value_to_index"].items()}
    pack_page_id = int(page_to_id.get("In_TarotSpectral_Pack", -1))

    family_order = list(fmap["family_order"])
    item_zone_for_family = fmap["item_zone_for_family"]
    default_card_zone_for_family = fmap["default_card_zone_for_family"]

    def _zone_id(z: str | None) -> int:
        if z is None:
            return -1
        return int(zone_to_id.get(z, -1))

    family_to_item_zone_id = [_zone_id(item_zone_for_family.get(fam)) for fam in family_order]
    family_to_card_zone_default_id = [
        _zone_id(default_card_zone_for_family.get(fam)) for fam in family_order
    ]
    # Pack-context card zone: TarotSpectralHand if the family takes cards,
    # otherwise mirror the default.
    pack_card_zone_id = _zone_id("TarotSpectralHand")
    family_to_card_zone_pack_id = []
    for fam in family_order:
        if default_card_zone_for_family.get(fam) is None:
            family_to_card_zone_pack_id.append(-1)
            continue
        # Card_seq families (PlayHand/DiscardHand) never enter pack pages
        # while picking cards from the hand → keep default. Only chained
        # families (UseConsumable / SelectPackItem) flip to pack zone.
        shape = fmap["decoder_shapes"][fam]
        if shape == "chained_cards":
            family_to_card_zone_pack_id.append(pack_card_zone_id)
        else:
            family_to_card_zone_pack_id.append(family_to_card_zone_default_id[-1])

    cfg = ModelConfig(
        n_actions=n_actions,
        max_objects=int(fc["MAX_OBJECTS_PER_STEP"]),
        max_deck_cards=int(fc["TRACKED_DECK_CAP"]),
        history_steps=int(fc.get("HISTORY_STEPS", 32)),
        history_objects_per_step=int(fc.get("HISTORY_OBJECTS_PER_STEP", 16)),
        n_hands=int(fc["N_HANDS"]),
        n_vouchers=int(fc["N_VOUCHERS"]),
        n_bosses=int(fc["N_BOSSES"]),
        n_families=int(fmap["n_families"]),
        max_item_zone_size=int(fc.get("MAX_ITEM_ZONE_SIZE", 15)),
        max_card_zone_size=int(fc.get("MAX_CARD_ZONE_SIZE", 15)),
        max_joker_slots=int(fc.get("MAX_JOKER_SLOTS", 10)),
        max_cards_per_decision=int(fc.get("MAX_CARDS_PER_DECISION", 5)),
        vocab_sizes=sizes,
        family_to_item_zone_id=family_to_item_zone_id,
        family_to_card_zone_default_id=family_to_card_zone_default_id,
        family_to_card_zone_pack_id=family_to_card_zone_pack_id,
        pack_page_id=pack_page_id,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _embed(cat_size: int, dim: int) -> nn.Embedding:
    """Embedding with PAD-zero init at index 0."""
    layer = nn.Embedding(cat_size, dim)
    with torch.no_grad():
        layer.weight[0].zero_()
    return layer


class GlobalEncoder(nn.Module):
    """Encode all per-step scalars + flags + multi-hots into one token.

    LEAKAGE NOTE
    ------------
    Two channels available in the tensorized record are deliberately
    omitted here because they are granularizer artifacts derived from the
    chosen action and not from the observable game state:

    - ``action_subtype_id`` — alone predicts the action family with ~99%
      accuracy across train/val/test (see ``diagnose_leakage.py``).
    - ``source_kind_id`` — alone predicts the action family with ~65%
      accuracy. ``select`` -> SelectCard, ``swap_synth`` -> SWAP, etc.

    A live agent does not have these features at decision time, so the
    model must not be allowed to use them during training either.
    """

    def __init__(self, cfg: ModelConfig, class_id_embedding: nn.Embedding | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        s = cfg.vocab_sizes

        # Categorical embeddings for the 7 OBSERVABLE global categoricals.
        # (action_subtype_id and source_kind_id intentionally excluded.)
        self.emb_page = _embed(s["page"], cfg.cat_embed_dim)
        self.emb_deck_class = _embed(s["deck_class_id"], cfg.cat_embed_dim)
        self.emb_stake_class = _embed(s["stake_class_id"], cfg.cat_embed_dim)
        self.emb_last_tarot_planet = _embed(s["last_tarot_planet_class_id"], cfg.cat_embed_dim)
        self.emb_ante_boss_blind = _embed(s["ante_boss_blind_class_id"], cfg.cat_embed_dim)
        self.emb_small_status = _embed(s["small_status"], cfg.cat_embed_dim)
        self.emb_big_status = _embed(s["big_status"], cfg.cat_embed_dim)

        # Numerics: ocr_numeric (16) + state_numeric (4)
        # + ocr_valid (16) + flags (12)
        # + 3 per-hand arrays (3*12 = 36)
        # + vouchers_redeemed (32) + bosses_used (28)
        # + supplement_features (N_SUPPLEMENT = 62 derived poker/joker/held flags)
        n_cat = 7 * cfg.cat_embed_dim
        n_num = (
            cfg.n_ocr
            + cfg.n_state_num
            + cfg.n_ocr  # ocr_valid (boolean)
            + cfg.n_flags
            + 3 * cfg.n_hands
            + cfg.n_vouchers
            + cfg.n_bosses
            + cfg.n_supplement
        )
        self.proj = nn.Sequential(
            nn.Linear(n_cat + n_num, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(
            [
                self.emb_page(x["page_id"]),
                self.emb_deck_class(x["deck_class_id"]),
                self.emb_stake_class(x["stake_class_id"]),
                self.emb_last_tarot_planet(x["last_tarot_planet_class_id"]),
                self.emb_ante_boss_blind(x["ante_boss_blind_class_id"]),
                self.emb_small_status(x["small_status_id"]),
                self.emb_big_status(x["big_status_id"]),
            ],
            dim=-1,
        )
        num = torch.cat(
            [
                x["ocr_numeric"],
                x["state_numeric"],
                x["ocr_valid"].float(),
                x["flags"].float(),
                x["hand_levels"],
                x["hand_played"],
                x["hand_played_this_round"],
                x["vouchers_redeemed"].float(),
                x["bosses_used"].float(),
                x["supplement_features"],
            ],
            dim=-1,
        )
        return self.proj(torch.cat([cat, num], dim=-1))


class CardLikeTokenEncoder(nn.Module):
    """
    Encode per-card / per-object tokens.

    Shared between the object stream and the tracked-deck stream because
    both have the same primitive feature set (class_id, modifier, edition,
    seal, rank, suit, plus optional zone/type/position/debuff/stickers).
    Pass ``with_zone=True`` for objects, ``False`` for tracked-deck cards.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        class_id_embedding: nn.Embedding,
        with_zone: bool = True,
        with_position: bool = True,
        with_debuff: bool = True,
        with_stickers: bool = True,
        with_object_type: bool = True,
        max_position: int = 32,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.with_zone = with_zone
        self.with_position = with_position
        self.with_debuff = with_debuff
        self.with_stickers = with_stickers
        self.with_object_type = with_object_type

        s = cfg.vocab_sizes
        self.emb_class_id = class_id_embedding  # shared
        self.emb_modifier = _embed(s["modifier"], cfg.cat_embed_dim)
        self.emb_edition = _embed(s["edition"], cfg.cat_embed_dim)
        self.emb_seal = _embed(s["seal"], cfg.cat_embed_dim)
        self.emb_rank = _embed(s["rank_index"], cfg.cat_embed_dim)
        self.emb_suit = _embed(s["suit_index"], cfg.cat_embed_dim)
        self.emb_object_type = (
            _embed(s["object_type"], cfg.cat_embed_dim) if with_object_type else None
        )
        self.emb_zone = _embed(s["zone"], cfg.cat_embed_dim) if with_zone else None
        # Position is bounded by per-zone size (max ~14 in corpus); embed up to max_position.
        self.emb_position = (
            _embed(max_position + 1, cfg.cat_embed_dim) if with_position else None
        )

        # Numeric channel count: debuff (1) + stickers (3)
        n_num = 0
        if with_debuff:
            n_num += 1
        if with_stickers:
            n_num += 3

        n_cat = (
            cfg.cat_embed_dim
            * (
                6
                + (1 if with_zone else 0)
                + (1 if with_position else 0)
                + (1 if with_object_type else 0)
            )
        )
        self.proj = nn.Sequential(
            nn.Linear(n_cat + n_num, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, x: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        """``prefix`` selects which input keys to read (``object_`` or ``deck_card_``)."""
        cat_parts = [
            self.emb_class_id(x[f"{prefix}class_id"]),
            self.emb_modifier(x[f"{prefix}modifier_id"]),
            self.emb_edition(x[f"{prefix}edition_id"]),
            self.emb_seal(x[f"{prefix}seal_id"]),
            self.emb_rank(x[f"{prefix}rank_id"]),
            self.emb_suit(x[f"{prefix}suit_id"]),
        ]
        if self.with_object_type:
            cat_parts.append(self.emb_object_type(x[f"{prefix}object_type_id"]))
        if self.with_zone:
            cat_parts.append(self.emb_zone(x[f"{prefix}zone_id"]))
        if self.with_position:
            pos = x[f"{prefix}position"].clamp(0, self.emb_position.num_embeddings - 1)
            cat_parts.append(self.emb_position(pos))
        cat = torch.cat(cat_parts, dim=-1)

        num_parts: list[torch.Tensor] = []
        if self.with_debuff:
            num_parts.append(x[f"{prefix}is_debuffed"].float().unsqueeze(-1))
        if self.with_stickers:
            num_parts.append(
                torch.stack(
                    [
                        x[f"{prefix}sticker_rental"].float(),
                        x[f"{prefix}sticker_perishable"].float(),
                        x[f"{prefix}sticker_eternal"].float(),
                    ],
                    dim=-1,
                )
            )
        if num_parts:
            cat = torch.cat([cat] + num_parts, dim=-1)
        return self.proj(cat)


class HistoryStepEncoder(nn.Module):
    """Encode one summary token per prior decision."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        s = cfg.vocab_sizes
        self.emb_action = _embed(cfg.n_actions + 2, cfg.cat_embed_dim)
        self.emb_page = _embed(s["page"], cfg.cat_embed_dim)
        self.emb_recency = _embed(cfg.history_steps + 1, cfg.cat_embed_dim)
        self.emb_target_zone = _embed(s["zone"], cfg.cat_embed_dim)
        self.emb_position = _embed(65, cfg.cat_embed_dim)
        self.emb_swap = _embed(17, cfg.cat_embed_dim)
        n_cat = 6 * cfg.cat_embed_dim
        n_num = cfg.n_ocr * 2
        self.proj = nn.Sequential(
            nn.Linear(n_cat + n_num, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        recency = x["history_recency"].clamp(0, self.emb_recency.num_embeddings - 1)
        position = x["history_target_position"].clamp(
            0, self.emb_position.num_embeddings - 1
        )
        swap_i = x["history_swap_i"].clamp(0, self.emb_swap.num_embeddings - 1)
        swap_j = x["history_swap_j"].clamp(0, self.emb_swap.num_embeddings - 1)
        cat = torch.cat(
            [
                self.emb_action(x["history_action_id"]),
                self.emb_page(x["history_page_id"]),
                self.emb_recency(recency),
                self.emb_target_zone(x["history_target_zone_id"]),
                self.emb_position(position),
                self.emb_swap(swap_i) + self.emb_swap(swap_j),
            ],
            dim=-1,
        )
        num = torch.cat(
            [x["history_ocr_numeric"], x["history_ocr_valid"].float()],
            dim=-1,
        )
        return self.proj(torch.cat([cat, num], dim=-1))


def _gather_zone_tokens(
    obj_tokens: torch.Tensor,
    obj_mask: torch.Tensor,
    obj_zone_ids: torch.Tensor,
    obj_positions: torch.Tensor,
    target_zone_id: torch.Tensor,
    max_zone_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter object tokens into a per-row, per-zone fixed buffer.

    Args:
        obj_tokens:    (B, N, D) refined object embeddings from the encoder.
        obj_mask:      (B, N) bool — True for valid object slots.
        obj_zone_ids:  (B, N) int — zone vocab id per object.
        obj_positions: (B, N) int — zone-local position per object.
        target_zone_id: (B,) int — desired zone vocab id per row. ``-1``
            means "no zone for this row" — output is all zeros / mask=0.
        max_zone_size: K — output capacity per zone.

    Returns:
        zone_tokens: (B, K, D) — token at zone-local position p; zero
            where no object exists at that position.
        zone_mask:   (B, K) bool — True where an object was scattered in.
    """
    B, N, D = obj_tokens.shape
    valid_row = (target_zone_id >= 0).unsqueeze(-1)  # (B, 1)
    matches = (
        obj_mask
        & (obj_zone_ids == target_zone_id.unsqueeze(-1))
        & valid_row
    )  # (B, N)
    clamped_pos = obj_positions.clamp(min=0, max=max_zone_size - 1).long()
    dummy = torch.full_like(clamped_pos, max_zone_size)
    pos_or_dummy = torch.where(matches, clamped_pos, dummy)

    out_buf = obj_tokens.new_zeros(B, max_zone_size + 1, D)
    out_buf.scatter_(
        dim=1,
        index=pos_or_dummy.unsqueeze(-1).expand(-1, -1, D),
        src=obj_tokens,
    )
    zone_tokens = out_buf[:, :max_zone_size]

    mask_buf = obj_tokens.new_zeros(B, max_zone_size + 1, dtype=torch.bool)
    mask_buf.scatter_(dim=1, index=pos_or_dummy, src=matches)
    zone_mask = mask_buf[:, :max_zone_size]
    return zone_tokens, zone_mask


class PointerScorer(nn.Module):
    """Bilinear pointer scorer.

    Scores a (B, D) query against a (B, K, D) set of keys, returning
    (B, K) logits. Logits are divided by sqrt(D) for stable softmax.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)  # (B, D)
        k = self.k_proj(keys)   # (B, K, D)
        return (k * q.unsqueeze(1)).sum(-1) * self.scale


class CardSeqDecoder(nn.Module):
    """Autoregressive pointer decoder over a fixed-size card pool.

    Teacher-forced training: at each step ``t`` we score the current
    pointer query against the per-zone token bank, mask already-picked
    positions, then update the recurrent state using the ground-truth
    picked card embedding. At inference we feed back the argmax.

    Outputs ``num_cards`` and ``card_ptr_logits_seq`` jointly.
    """

    def __init__(self, d_model: int, max_cards: int, max_zone_size: int) -> None:
        super().__init__()
        self.max_cards = max_cards
        self.max_zone_size = max_zone_size
        self.num_cards_head = nn.Linear(d_model, max_cards + 1)
        self.pointer = PointerScorer(d_model)
        self.cell = nn.GRUCell(d_model, d_model)

    def num_cards_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.num_cards_head(state)

    def forward(
        self,
        initial_state: torch.Tensor,
        zone_tokens: torch.Tensor,
        zone_pointer_mask: torch.Tensor,
        card_ptr_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``max_cards`` autoregressive pointer steps.

        Args:
            initial_state: (B, D) starting decoder state.
            zone_tokens: (B, K, D) per-zone object embeddings.
            zone_pointer_mask: (B, K) bool — True where a pointer pick
                is legal at the start of the sequence.
            card_ptr_target: (B, max_cards) long — teacher labels (-1
                for padded steps). ``None`` triggers greedy decoding.

        Returns:
            logits: (B, max_cards, K) per-step pointer logits with
            previously-picked positions masked to -inf.
        """
        B, K, D = zone_tokens.shape
        state = initial_state
        already_picked = torch.zeros(B, K, dtype=torch.bool, device=state.device)
        legal = zone_pointer_mask & ~already_picked
        all_logits = []

        for t in range(self.max_cards):
            scores = self.pointer(state, zone_tokens)
            scores = scores.masked_fill(~legal, float("-inf"))
            all_logits.append(scores)

            if card_ptr_target is not None:
                tgt = card_ptr_target[:, t]
            else:
                # Greedy: argmax over legal slots.
                tgt = scores.argmax(dim=-1)

            valid = tgt >= 0
            tgt_safe = tgt.clamp(min=0).long()
            # Update GRU state using the picked card's embedding (no-op
            # when target is padding).
            picked = zone_tokens.gather(
                1, tgt_safe.view(-1, 1, 1).expand(-1, 1, D)
            ).squeeze(1)
            new_state = self.cell(picked, state)
            state = torch.where(valid.unsqueeze(-1), new_state, state)

            # Mark this position as already picked so the next step
            # can't repeat it.
            new_picked = torch.zeros_like(already_picked)
            new_picked.scatter_(1, tgt_safe.unsqueeze(-1), valid.unsqueeze(-1))
            already_picked = already_picked | new_picked
            legal = zone_pointer_mask & ~already_picked

        return torch.stack(all_logits, dim=1)  # (B, max_cards, K)


class JokerPairDecoder(nn.Module):
    """Two-step autoregressive pointer over CurrentJokers for SWAP.

    Step 0 emits ``swap_i_logits``, step 1 emits ``swap_j_logits``
    (with ``i`` excluded). At training time we teacher-force using
    ``swap_i_local``; at inference we feed back the argmax. Emission
    canonicalises to ``i < j`` outside the decoder.
    """

    def __init__(self, d_model: int, max_joker_slots: int) -> None:
        super().__init__()
        self.max_joker_slots = max_joker_slots
        self.pointer_i = PointerScorer(d_model)
        self.pointer_j = PointerScorer(d_model)
        self.update = nn.GRUCell(d_model, d_model)

    def forward(
        self,
        initial_state: torch.Tensor,
        joker_tokens: torch.Tensor,
        joker_mask: torch.Tensor,
        swap_i_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, D = joker_tokens.shape
        i_logits = self.pointer_i(initial_state, joker_tokens)
        i_logits = i_logits.masked_fill(~joker_mask, float("-inf"))

        if swap_i_target is not None:
            tgt_i = swap_i_target
        else:
            tgt_i = i_logits.argmax(dim=-1)
        valid_i = tgt_i >= 0
        tgt_i_safe = tgt_i.clamp(min=0).long()
        picked = joker_tokens.gather(
            1, tgt_i_safe.view(-1, 1, 1).expand(-1, 1, D)
        ).squeeze(1)
        state = self.update(picked, initial_state)
        state = torch.where(valid_i.unsqueeze(-1), state, initial_state)

        j_logits = self.pointer_j(state, joker_tokens)
        # Mask j == i and any invalid joker slot.
        exclude_i = torch.zeros(B, K, dtype=torch.bool, device=state.device)
        exclude_i.scatter_(1, tgt_i_safe.unsqueeze(-1), valid_i.unsqueeze(-1))
        j_mask = joker_mask & ~exclude_i
        j_logits = j_logits.masked_fill(~j_mask, float("-inf"))
        return i_logits, j_logits


class PolicyTransformer(nn.Module):
    """End-to-end branched-policy network mapping a step record to logits.

    forward() returns a dict of all branched-head logits with masks
    already applied. The training loop dispatches per family_id.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Shared class_id embedding (object_class_id and deck_card_class_id
        # both index into the same 401-entry table).
        self.class_id_embedding = _embed(cfg.vocab_sizes["class_id"], cfg.cat_embed_dim)

        self.global_encoder = GlobalEncoder(cfg)
        self.object_encoder = CardLikeTokenEncoder(
            cfg,
            self.class_id_embedding,
            with_zone=True,
            with_position=True,
            with_debuff=True,
            with_stickers=True,
            with_object_type=True,
        )
        self.deck_encoder = CardLikeTokenEncoder(
            cfg,
            self.class_id_embedding,
            with_zone=False,
            with_position=False,
            with_debuff=False,
            with_stickers=False,
            with_object_type=False,
        )
        if cfg.use_history_tokens:
            self.history_step_encoder = HistoryStepEncoder(cfg)
            self.history_object_encoder = CardLikeTokenEncoder(
                cfg,
                self.class_id_embedding,
                with_zone=True,
                with_position=True,
                with_debuff=True,
                with_stickers=True,
                with_object_type=True,
            )

        # Type tokens added to each token's embedding so the transformer can
        # tell them apart even though they share dimensionality.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.global_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.object_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.deck_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        if cfg.use_history_tokens:
            self.history_step_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.history_object_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.history_recency = _embed(cfg.history_steps + 1, cfg.d_model)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.global_type, std=0.02)
        nn.init.normal_(self.object_type, std=0.02)
        nn.init.normal_(self.deck_type, std=0.02)
        if cfg.use_history_tokens:
            nn.init.normal_(self.history_step_type, std=0.02)
            nn.init.normal_(self.history_object_type, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.cls_proj = nn.LayerNorm(cfg.d_model)

        # --- Branched-policy heads -------------------------------------
        self.family_head = nn.Linear(cfg.d_model, cfg.n_families)
        # Two card-seq decoders: one for PlayHand/DiscardHand (driven
        # purely by cls_out) and one for chained_cards (conditioned on
        # the picked item embedding). Sharing them would mix gradients
        # across very different decision modes; keep them separate.
        self.card_seq_decoder = CardSeqDecoder(
            cfg.d_model, cfg.max_cards_per_decision, cfg.max_card_zone_size
        )
        self.chained_card_decoder = CardSeqDecoder(
            cfg.d_model, cfg.max_cards_per_decision, cfg.max_card_zone_size
        )
        # Item pointer (used by single_ptr AND chained_cards' first step).
        self.item_pointer = PointerScorer(cfg.d_model)
        # State combiner for chained_cards: maps (cls_out, item_emb)
        # into the initial state for the card-seq decoder.
        self.chained_state_proj = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # Joker-pair decoder for SWAP.
        self.joker_pair_decoder = JokerPairDecoder(cfg.d_model, cfg.max_joker_slots)

        # Optional flat head for legacy checkpoints.
        if cfg.flat_head:
            self.flat_head_layer: nn.Module = nn.Linear(cfg.d_model, cfg.n_actions)
        else:
            self.flat_head_layer = nn.Identity()

        # Register lookup tables as buffers so they move with the
        # model's device and serialize with state_dict.
        item_zone_ids = torch.tensor(cfg.family_to_item_zone_id, dtype=torch.long)
        card_zone_default_ids = torch.tensor(cfg.family_to_card_zone_default_id, dtype=torch.long)
        card_zone_pack_ids = torch.tensor(cfg.family_to_card_zone_pack_id, dtype=torch.long)
        self.register_buffer("family_to_item_zone_id", item_zone_ids, persistent=False)
        self.register_buffer(
            "family_to_card_zone_default_id", card_zone_default_ids, persistent=False
        )
        self.register_buffer(
            "family_to_card_zone_pack_id", card_zone_pack_ids, persistent=False
        )

    # ------------------------------------------------------------------
    # Shared encoder pass
    # ------------------------------------------------------------------

    def encode(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the shared transformer and return cls + object embeddings.

        Returns a dict with:
        - ``cls_out``: (B, D) pooled state.
        - ``object_out``: (B, n_obj, D) refined per-object embeddings.
        - ``object_mask``: (B, n_obj) bool valid-slot mask.
        """
        b = x["page_id"].shape[0]

        global_token = self.global_encoder(x).unsqueeze(1) + self.global_type
        object_tokens = self.object_encoder(x, "object_") + self.object_type
        if self.cfg.use_tracked_deck_tokens:
            deck_tokens = self.deck_encoder(x, "deck_card_") + self.deck_type
        else:
            deck_tokens = None
        if self.cfg.use_history_tokens:
            history_step_tokens = (
                self.history_step_encoder(x) + self.history_step_type
            )
            history_object_tokens = self.history_object_encoder(x, "history_object_")
            h = history_object_tokens.shape[1]
            o = history_object_tokens.shape[2]
            recency = x["history_recency"].clamp(0, self.history_recency.num_embeddings - 1)
            history_object_tokens = (
                history_object_tokens
                + self.history_object_type
                + self.history_recency(recency).unsqueeze(2)
            ).reshape(b, h * o, -1)
        else:
            history_step_tokens = None
            history_object_tokens = None

        cls = self.cls_token.expand(b, -1, -1)
        token_parts = [cls, global_token, object_tokens]
        if deck_tokens is not None:
            token_parts.append(deck_tokens)
        if history_step_tokens is not None and history_object_tokens is not None:
            token_parts.extend([history_step_tokens, history_object_tokens])
        tokens = torch.cat(token_parts, dim=1)

        n_obj = object_tokens.shape[1]
        n_deck = deck_tokens.shape[1] if deck_tokens is not None else 0
        n_hist_steps = history_step_tokens.shape[1] if history_step_tokens is not None else 0
        n_hist_objects = history_object_tokens.shape[1] if history_object_tokens is not None else 0
        device = tokens.device
        pad = torch.zeros(
            b,
            2 + n_obj + n_deck + n_hist_steps + n_hist_objects,
            dtype=torch.bool,
            device=device,
        )
        pad[:, 2 : 2 + n_obj] = ~x["object_mask"]
        cursor = 2 + n_obj
        if deck_tokens is not None:
            pad[:, cursor : cursor + n_deck] = ~x["deck_card_mask"]
            cursor += n_deck
        if self.cfg.use_history_tokens:
            history_step_mask = x["history_step_mask"].bool()
            history_object_mask = x["history_object_mask"].bool()
            if self.training and self.cfg.history_step_dropout > 0:
                keep = torch.rand_like(history_step_mask.float()) >= self.cfg.history_step_dropout
                history_step_mask = history_step_mask & keep
                history_object_mask = history_object_mask & history_step_mask.unsqueeze(-1)
            if self.training and self.cfg.history_object_dropout > 0:
                keep_obj = torch.rand_like(history_object_mask.float()) >= self.cfg.history_object_dropout
                history_object_mask = history_object_mask & keep_obj
            pad[:, cursor : cursor + n_hist_steps] = ~history_step_mask
            cursor += n_hist_steps
            pad[:, cursor:] = ~history_object_mask.reshape(b, n_hist_objects)

        out = self.encoder(tokens, src_key_padding_mask=pad)
        cls_out = self.norm(out[:, 0])
        object_out = self.cls_proj(out[:, 2 : 2 + n_obj])
        return {
            "cls_out": cls_out,
            "object_out": object_out,
            "object_mask": x["object_mask"].bool(),
        }

    # ------------------------------------------------------------------
    # Pointer helpers
    # ------------------------------------------------------------------

    def _resolve_zone_ids(
        self,
        family_id: torch.Tensor,
        page_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-row item/card zone vocab ids from family + page.

        ``family_id`` may contain ``-1`` for rows with no family
        (StartNewRun); those rows get ``-1`` zones, which the gather
        helper interprets as "no zone".
        """
        n_families = self.family_to_item_zone_id.shape[0]
        safe_fid = family_id.clamp(min=0, max=n_families - 1).long()
        item_zone = self.family_to_item_zone_id[safe_fid]
        card_zone_default = self.family_to_card_zone_default_id[safe_fid]
        card_zone_pack = self.family_to_card_zone_pack_id[safe_fid]
        in_pack = page_id == self.cfg.pack_page_id
        card_zone = torch.where(in_pack, card_zone_pack, card_zone_default)
        # Rows with family_id < 0 get -1 zones (no gather happens).
        invalid = (family_id < 0).unsqueeze(-1)
        item_zone = item_zone.masked_fill(invalid.squeeze(-1), -1)
        card_zone = card_zone.masked_fill(invalid.squeeze(-1), -1)
        return item_zone, card_zone

    def _gather_item_card_zones(
        self,
        enc: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
        family_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Gather encoder objects into per-row item and card zone buffers."""
        item_zone, card_zone = self._resolve_zone_ids(family_id, x["page_id"])
        obj_tokens = enc["object_out"]
        obj_mask = enc["object_mask"]
        obj_zone_ids = x["object_zone_id"].long()
        obj_positions = x["object_position"].long()
        item_tokens, item_pres = _gather_zone_tokens(
            obj_tokens, obj_mask, obj_zone_ids, obj_positions, item_zone,
            self.cfg.max_item_zone_size,
        )
        card_tokens, card_pres = _gather_zone_tokens(
            obj_tokens, obj_mask, obj_zone_ids, obj_positions, card_zone,
            self.cfg.max_card_zone_size,
        )
        # CurrentJokers is the SWAP zone — always the same vocab id.
        joker_zone_id = self.family_to_item_zone_id.new_full(
            (family_id.shape[0],),
            int(self.cfg.family_to_item_zone_id[
                # ``SWAP`` family id = last entry in FAMILY_ORDER (id 18)
                # in the canonical 19-family layout. We rely on the
                # passed-in family_to_item_zone_id buffer to encode this.
                self._swap_family_id()
            ]),
        )
        joker_tokens, joker_pres = _gather_zone_tokens(
            obj_tokens, obj_mask, obj_zone_ids, obj_positions, joker_zone_id,
            self.cfg.max_joker_slots,
        )
        return {
            "item_tokens": item_tokens,
            "item_token_mask": item_pres,
            "card_tokens": card_tokens,
            "card_token_mask": card_pres,
            "joker_tokens": joker_tokens,
            "joker_token_mask": joker_pres,
        }

    def _swap_family_id(self) -> int:
        # SWAP is fixed in the family map: last entry (id 18) in v1.
        # Cached lazily on the module to avoid recomputing.
        if not hasattr(self, "_cached_swap_fid"):
            zone_for_fid = self.family_to_item_zone_id.tolist()
            # Find the entry that maps to CurrentJokers (zone id stored
            # in family_to_item_zone_id for SWAP). We can't look up by
            # name without family_map at hand, so we rely on the cfg
            # ordering: SWAP is the last family.
            self._cached_swap_fid = self.cfg.n_families - 1
        return int(self._cached_swap_fid)

    # ------------------------------------------------------------------
    # Branched forward (teacher-forced training)
    # ------------------------------------------------------------------

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the encoder + all branched heads with teacher forcing.

        Required input channels in ``x``:
        - All encoder inputs (object/global/history) per the existing
          tensorize layout.
        - ``family_id`` (B,): supervised target family. Used to gather
          the right zones and to teacher-force pointer targets.
        - ``family_mask`` (B, n_families) bool.
        - ``item_pointer_mask`` (B, MAX_ITEM_ZONE_SIZE) bool.
        - ``card_pointer_mask`` (B, MAX_CARD_ZONE_SIZE) bool.
        - ``swap_joker_mask`` (B, MAX_JOKER_SLOTS) bool.
        - ``card_ptr_local_seq`` (B, MAX_CARDS_PER_DECISION) long
          (teacher labels; -1 padded).
        - ``swap_i_local`` (B,) long.

        Returns dict with logits for every head (masked to -inf where
        legality dictates). The trainer picks the relevant logits per
        row using ``family_id`` to compute argument losses.
        """
        enc = self.encode(x)
        cls_out = enc["cls_out"]

        # ------------- family head ------------------------------------
        family_logits = self.family_head(cls_out)
        family_mask = x["family_mask"].bool()
        family_logits = family_logits.masked_fill(~family_mask, float("-inf"))

        # ------------- zone gather (item + card + joker) --------------
        # Use teacher-forced family_id to resolve target zones. At
        # inference the caller passes the model's argmax family.
        family_id_safe = x["family_id"].clamp(min=0).long()
        zones = self._gather_item_card_zones(enc, x, x["family_id"])

        item_mask = x["item_pointer_mask"].bool()
        card_mask = x["card_pointer_mask"].bool()
        swap_mask = x["swap_joker_mask"].bool()
        card_ptr_target = x["card_ptr_local_seq"].long()
        swap_i_target = x["swap_i_local"].long()

        # ------------- single-pointer head ----------------------------
        item_logits = self.item_pointer(cls_out, zones["item_tokens"])
        item_logits = item_logits.masked_fill(~item_mask, float("-inf"))

        # ------------- card_seq decoder (PlayHand / DiscardHand) ------
        card_seq_num_cards_logits = self.card_seq_decoder.num_cards_logits(cls_out)
        card_seq_ptr_logits = self.card_seq_decoder(
            initial_state=cls_out,
            zone_tokens=zones["card_tokens"],
            zone_pointer_mask=card_mask,
            card_ptr_target=card_ptr_target,
        )

        # ------------- chained_cards decoder --------------------------
        # First pick the item; then condition num_cards + card_seq on it.
        # Teacher-force the item pick: gather the labeled item token.
        item_ptr_target = x["item_ptr_local"].long()
        valid_item = item_ptr_target >= 0
        safe_item_idx = item_ptr_target.clamp(min=0)
        picked_item_emb = zones["item_tokens"].gather(
            1, safe_item_idx.view(-1, 1, 1).expand(-1, 1, cls_out.shape[-1])
        ).squeeze(1)
        picked_item_emb = picked_item_emb * valid_item.unsqueeze(-1).float()
        chained_initial = self.chained_state_proj(
            torch.cat([cls_out, picked_item_emb], dim=-1)
        )
        chained_num_cards_logits = self.chained_card_decoder.num_cards_logits(
            chained_initial
        )
        chained_ptr_logits = self.chained_card_decoder(
            initial_state=chained_initial,
            zone_tokens=zones["card_tokens"],
            zone_pointer_mask=card_mask,
            card_ptr_target=card_ptr_target,
        )

        # ------------- joker_pair (SWAP) ------------------------------
        swap_i_logits, swap_j_logits = self.joker_pair_decoder(
            initial_state=cls_out,
            joker_tokens=zones["joker_tokens"],
            joker_mask=swap_mask,
            swap_i_target=swap_i_target,
        )

        out: dict[str, torch.Tensor] = {
            "family_logits": family_logits,
            "item_logits": item_logits,
            "card_seq_num_cards_logits": card_seq_num_cards_logits,
            "card_seq_ptr_logits": card_seq_ptr_logits,
            "chained_num_cards_logits": chained_num_cards_logits,
            "chained_ptr_logits": chained_ptr_logits,
            "swap_i_logits": swap_i_logits,
            "swap_j_logits": swap_j_logits,
        }
        if self.cfg.flat_head:
            flat_logits = self.flat_head_layer(cls_out)
            flat_logits = flat_logits.masked_fill(
                ~x["action_mask"].bool(), float("-inf")
            )
            out["action_logits"] = flat_logits
        return out

    # ------------------------------------------------------------------
    # Greedy inference (no teacher forcing)
    # ------------------------------------------------------------------
    #
    # Inference at deploy time happens in two phases because the pointer
    # masks depend on the *predicted* family:
    #   1. encode_and_pick_family(x) runs the transformer + family head
    #      and returns the family argmax. The caller (live agent) uses
    #      the predicted family to recompute item / card / swap pointer
    #      masks from the snapshot, then re-injects them into ``x``.
    #   2. decode_arguments(enc, x, family_id) runs every argument head
    #      greedily against the now-correct pointer masks.
    #
    # ``decide()`` is the single-call form used in tests / smoke runs
    # where the masks are already correct for the predicted family
    # (e.g. supervised data evaluated greedily) — it composes both
    # phases without recomputing the masks in between.

    @torch.no_grad()
    def encode_and_pick_family(
        self,
        x: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Phase 1: encode + family argmax.

        Returns a dict with ``enc`` (the encoder output) plus the
        masked ``family_logits`` and predicted ``family_id``. Caller
        is expected to consult ``family_id`` to recompute pointer
        masks against the predicted family before phase 2.
        """
        enc = self.encode(x)
        cls_out = enc["cls_out"]
        family_logits = self.family_head(cls_out)
        family_logits = family_logits.masked_fill(
            ~x["family_mask"].bool(), float("-inf")
        )
        family_id = family_logits.argmax(dim=-1)
        return {
            "enc": enc,
            "family_id": family_id,
            "family_logits": family_logits,
        }

    @torch.no_grad()
    def decode_arguments(
        self,
        enc: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
        family_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Phase 2: greedy argument decode conditioned on ``family_id``.

        ``x`` must carry the **correct** pointer masks for ``family_id``
        in ``item_pointer_mask`` / ``card_pointer_mask`` / ``swap_joker_mask``.
        The caller is responsible for recomputing them after phase 1.

        All four decoder heads run regardless of family — callers pick
        the relevant outputs based on the family's decoder shape.
        """
        cls_out = enc["cls_out"]
        d_model = cls_out.shape[-1]
        zones = self._gather_item_card_zones(enc, x, family_id)

        item_mask = x["item_pointer_mask"].bool()
        card_mask = x["card_pointer_mask"].bool()
        swap_mask = x["swap_joker_mask"].bool()

        # -- single-pointer head ---------------------------------------
        item_logits = self.item_pointer(cls_out, zones["item_tokens"])
        item_logits = item_logits.masked_fill(~item_mask, float("-inf"))
        item_pred = item_logits.argmax(dim=-1)

        # -- card_seq head (PlayHand / DiscardHand) --------------------
        card_seq_n_logits = self.card_seq_decoder.num_cards_logits(cls_out)
        card_seq_num_cards = card_seq_n_logits.argmax(dim=-1)
        card_seq_ptr_logits = self.card_seq_decoder(
            initial_state=cls_out,
            zone_tokens=zones["card_tokens"],
            zone_pointer_mask=card_mask,
            card_ptr_target=None,
        )
        card_seq_pred = card_seq_ptr_logits.argmax(dim=-1)

        # -- chained_cards head (UseConsumable / SelectPackItem) -------
        valid_item = (item_pred >= 0) & item_mask.any(dim=-1)
        safe_item_idx = item_pred.clamp(min=0)
        picked_item_emb = zones["item_tokens"].gather(
            1, safe_item_idx.view(-1, 1, 1).expand(-1, 1, d_model)
        ).squeeze(1)
        picked_item_emb = picked_item_emb * valid_item.unsqueeze(-1).float()
        chained_initial = self.chained_state_proj(
            torch.cat([cls_out, picked_item_emb], dim=-1)
        )
        chained_n_logits = self.chained_card_decoder.num_cards_logits(chained_initial)
        chained_num_cards = chained_n_logits.argmax(dim=-1)
        chained_ptr_logits = self.chained_card_decoder(
            initial_state=chained_initial,
            zone_tokens=zones["card_tokens"],
            zone_pointer_mask=card_mask,
            card_ptr_target=None,
        )
        chained_pred = chained_ptr_logits.argmax(dim=-1)

        # -- joker_pair head (SWAP) ------------------------------------
        swap_i_logits, swap_j_logits = self.joker_pair_decoder(
            initial_state=cls_out,
            joker_tokens=zones["joker_tokens"],
            joker_mask=swap_mask,
            swap_i_target=None,
        )
        swap_i_pred = swap_i_logits.argmax(dim=-1)
        swap_j_pred = swap_j_logits.argmax(dim=-1)

        return {
            "item_pred": item_pred,
            "item_logits": item_logits,
            "card_seq_num_cards": card_seq_num_cards,
            "card_seq_num_cards_logits": card_seq_n_logits,
            "card_seq_pred": card_seq_pred,
            "card_seq_ptr_logits": card_seq_ptr_logits,
            "chained_num_cards": chained_num_cards,
            "chained_num_cards_logits": chained_n_logits,
            "chained_pred": chained_pred,
            "chained_ptr_logits": chained_ptr_logits,
            "swap_i_pred": swap_i_pred,
            "swap_i_logits": swap_i_logits,
            "swap_j_pred": swap_j_pred,
            "swap_j_logits": swap_j_logits,
        }

    @torch.no_grad()
    def decide(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Single-call greedy decode. Pointer masks in ``x`` must already
        be correct for the family argmax (typical for supervised data
        with teacher-forced masks; the live agent uses the two-phase API
        instead so it can recompute masks after the family is picked).
        """
        phase1 = self.encode_and_pick_family(x)
        phase2 = self.decode_arguments(phase1["enc"], x, phase1["family_id"])
        return {
            "family_id": phase1["family_id"],
            "family_logits": phase1["family_logits"],
            **phase2,
        }


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
