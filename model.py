#!/usr/bin/env python3
"""
model.py
========
Policy transformer for Balatro behavior cloning.

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
   (capped at ``TRACKED_DECK_CAP``). Mirrors the object-token feature
   set but draws from the persistent state's tracked deck.

A learned [CLS] token is prepended; its output state is projected into
``N_ACTIONS`` logits. At inference + training we mask invalid actions to
``-inf`` so the softmax / cross-entropy distribution is over the legal
action set only.

Embedding tables
----------------
Vocab sizes are read from ``artifacts/vocab.json`` to keep model and
data fully aligned. ``class_id`` uses identity encoding (vocab_index ==
class_id) and shares one embedding table across object_class_id and
deck_card_class_id. Other categoricals each get their own table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


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

    # Vocab sizes -- filled from artifacts/vocab.json.
    vocab_sizes: dict[str, int] = field(default_factory=dict)

    # Per-step shape information (for global feature MLP).
    n_ocr: int = 16
    n_state_num: int = 4
    n_flags: int = 12
    n_hands: int = 12
    n_vouchers: int = 32
    n_bosses: int = 28


def load_model_config(
    n_actions: int,
    vocab_path: Path = Path("artifacts/vocab.json"),
    feature_config_path: Path = Path("artifacts/feature_config.json"),
    **overrides: Any,
) -> ModelConfig:
    """Build a ModelConfig from on-disk artifacts."""
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))["vocabularies"]
    fc = json.loads(feature_config_path.read_text(encoding="utf-8"))
    sizes = {name: int(entry["size"]) for name, entry in vocab.items()}
    cfg = ModelConfig(
        n_actions=n_actions,
        max_objects=int(fc["MAX_OBJECTS_PER_STEP"]),
        max_deck_cards=int(fc["TRACKED_DECK_CAP"]),
        n_hands=int(fc["N_HANDS"]),
        n_vouchers=int(fc["N_VOUCHERS"]),
        n_bosses=int(fc["N_BOSSES"]),
        vocab_sizes=sizes,
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
        n_cat = 7 * cfg.cat_embed_dim
        n_num = (
            cfg.n_ocr
            + cfg.n_state_num
            + cfg.n_ocr  # ocr_valid (boolean)
            + cfg.n_flags
            + 3 * cfg.n_hands
            + cfg.n_vouchers
            + cfg.n_bosses
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


class PolicyTransformer(nn.Module):
    """End-to-end policy network mapping a step record to action logits."""

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

        # Type tokens added to each token's embedding so the transformer can
        # tell them apart even though they share dimensionality.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.global_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.object_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.deck_type = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.global_type, std=0.02)
        nn.init.normal_(self.object_type, std=0.02)
        nn.init.normal_(self.deck_type, std=0.02)

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
        self.head = nn.Linear(cfg.d_model, cfg.n_actions)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        return_logits_only: bool = True,
    ) -> torch.Tensor:
        b = x["page_id"].shape[0]

        global_token = self.global_encoder(x).unsqueeze(1) + self.global_type
        object_tokens = self.object_encoder(x, "object_") + self.object_type
        deck_tokens = self.deck_encoder(x, "deck_card_") + self.deck_type

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, global_token, object_tokens, deck_tokens], dim=1)

        # Build attention key-padding mask. True means position is masked
        # (PyTorch convention for src_key_padding_mask).
        n_obj = object_tokens.shape[1]
        n_deck = deck_tokens.shape[1]
        device = tokens.device
        pad = torch.zeros(b, 2 + n_obj + n_deck, dtype=torch.bool, device=device)
        # CLS (idx 0) and global (idx 1) are always present.
        # Objects: indices 2..2+n_obj
        pad[:, 2 : 2 + n_obj] = ~x["object_mask"]
        pad[:, 2 + n_obj :] = ~x["deck_card_mask"]

        out = self.encoder(tokens, src_key_padding_mask=pad)
        cls_out = self.norm(out[:, 0])
        logits = self.head(cls_out)

        # Apply legality mask -> -inf for illegal actions.
        action_mask = x["action_mask"].bool()
        logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
