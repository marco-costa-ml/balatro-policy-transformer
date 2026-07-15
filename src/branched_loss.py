#!/usr/bin/env python3
"""
branched_loss.py
================
Cross-entropy losses + top-1 metrics for the branched autoregressive
policy. Composed of:

- ``family CE``                    : always applied (one term per row).
- ``card_seq``  num_cards + ptr_t  : applied to PlayHand / DiscardHand rows.
- ``single_ptr`` item ptr          : applied to Buy / Sell / BuyAndUse rows.
- ``chained_cards`` item + nc + ptr: applied to UseConsumable / SelectPackItem.
- ``joker_pair`` swap_i + swap_j   : applied to SWAP rows.

The total loss is the **sum** of these components (each individually
mean-reduced over its own valid-row mask). This keeps individual head
losses comparable across runs and lets us inspect / weight them
independently. Pointer steps inside a card sequence are mean-reduced
over the union of valid (row, t) pairs.

Per-shape masks are derived from ``family_id`` via a lookup tensor
built at module construction time from a ``family_map`` dict
(``family_id -> decoder_shape``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SHAPE_NO_ARGS = 0
SHAPE_CARD_SEQ = 1
SHAPE_SINGLE_PTR = 2
SHAPE_CHAINED_CARDS = 3
SHAPE_JOKER_PAIR = 4
SHAPE_RESERVED = 5

_SHAPE_NAME_TO_ID = {
    "no_args": SHAPE_NO_ARGS,
    "card_seq": SHAPE_CARD_SEQ,
    "single_ptr": SHAPE_SINGLE_PTR,
    "chained_cards": SHAPE_CHAINED_CARDS,
    "joker_pair": SHAPE_JOKER_PAIR,
    "reserved": SHAPE_RESERVED,
}


def build_decoder_shape_lut(family_map: dict[str, Any]) -> torch.Tensor:
    """Return a ``(n_families,)`` long tensor mapping family id -> shape id."""
    order = family_map["family_order"]
    shapes = family_map["decoder_shapes"]
    return torch.tensor(
        [_SHAPE_NAME_TO_ID[shapes[fam]] for fam in order], dtype=torch.long
    )


@dataclass
class BranchedLossOutput:
    """Aggregated loss components and per-head metrics."""
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    n_valid: dict[str, int]
    top1: dict[str, float]
    top3: dict[str, float]


def _ce_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean CE over rows where ``row_mask`` is True.

    Selects only the masked rows before computing CE so padded targets
    (which would point at -inf logits and yield NaN) never enter the
    softmax denominator.
    """
    n = int(row_mask.sum().item())
    if n == 0:
        return logits.new_tensor(0.0), 0
    rows = row_mask.nonzero(as_tuple=False).squeeze(-1)
    sel_logits = logits.index_select(0, rows)
    sel_target = target.index_select(0, rows).long()
    return F.cross_entropy(sel_logits, sel_target, reduction="mean"), n


def _pointer_seq_ce(
    logits_seq: torch.Tensor,
    target_seq: torch.Tensor,
    row_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean CE over ``(row, t)`` pairs where both ``target_seq[b,t] >= 0``
    AND ``row_mask[b]`` are True. Only valid pairs are passed to CE."""
    B, T, K = logits_seq.shape
    valid = (target_seq >= 0) & row_mask.unsqueeze(-1)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return logits_seq.new_tensor(0.0), 0
    flat_valid = valid.reshape(B * T)
    flat_logits = logits_seq.reshape(B * T, K).index_select(
        0, flat_valid.nonzero(as_tuple=False).squeeze(-1)
    )
    flat_target = target_seq.reshape(B * T).index_select(
        0, flat_valid.nonzero(as_tuple=False).squeeze(-1)
    ).long()
    return F.cross_entropy(flat_logits, flat_target, reduction="mean"), n_valid


def _topk_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    k: int,
) -> tuple[int, int]:
    if int(row_mask.sum().item()) == 0:
        return 0, 0
    safe = target.clamp(min=0).long()
    k_eff = min(k, int(logits.shape[-1]))
    topk = torch.topk(logits, k=k_eff, dim=-1).indices
    hit = (topk == safe.unsqueeze(-1)).any(dim=-1) & row_mask
    return int(hit.sum().item()), int(row_mask.sum().item())


def _top1_masked(
    logits: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor
) -> tuple[int, int]:
    return _topk_masked(logits, target, row_mask, k=1)


def _pointer_seq_topk(
    logits_seq: torch.Tensor,
    target_seq: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    k: int,
) -> tuple[int, int]:
    valid = (target_seq >= 0) & row_mask.unsqueeze(-1)
    if int(valid.sum().item()) == 0:
        return 0, 0
    safe = target_seq.clamp(min=0).long()
    k_eff = min(k, int(logits_seq.shape[-1]))
    topk = torch.topk(logits_seq, k=k_eff, dim=-1).indices
    hit = (topk == safe.unsqueeze(-1)).any(dim=-1) & valid
    return int(hit.sum().item()), int(valid.sum().item())


def _pointer_seq_top1(
    logits_seq: torch.Tensor,
    target_seq: torch.Tensor,
    row_mask: torch.Tensor,
) -> tuple[int, int]:
    return _pointer_seq_topk(logits_seq, target_seq, row_mask, k=1)


class BranchedLoss(nn.Module):
    """Compute the branched-policy loss + per-head top-1 metrics.

    Construction reads the family_map dict (the artifact produced by
    ``family_map.compute_family_map``) once. Forward then dispatches
    per-shape losses purely from ``family_id``.
    """

    def __init__(self, family_map: dict[str, Any]) -> None:
        super().__init__()
        lut = build_decoder_shape_lut(family_map)
        self.register_buffer("shape_lut", lut, persistent=False)
        self.n_families = int(family_map["n_families"])

    def _shapes_per_row(self, family_id: torch.Tensor) -> torch.Tensor:
        safe = family_id.clamp(min=0, max=self.n_families - 1).long()
        return self.shape_lut[safe]

    def forward(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> BranchedLossOutput:
        family_id = batch["family_id"].long()
        valid_family = family_id >= 0
        shape_per_row = self._shapes_per_row(family_id)
        is_card_seq = (shape_per_row == SHAPE_CARD_SEQ) & valid_family
        is_single_ptr = (shape_per_row == SHAPE_SINGLE_PTR) & valid_family
        is_chained = (shape_per_row == SHAPE_CHAINED_CARDS) & valid_family
        is_joker_pair = (shape_per_row == SHAPE_JOKER_PAIR) & valid_family

        components: dict[str, torch.Tensor] = {}
        n_valid: dict[str, int] = {}
        top1: dict[str, float] = {}
        top3: dict[str, float] = {}

        def _record_topk(
            key: str,
            logits: torch.Tensor,
            target: torch.Tensor,
            row_mask: torch.Tensor,
        ) -> None:
            c1, t1 = _top1_masked(logits, target, row_mask)
            top1[key] = c1 / t1 if t1 else 0.0
            c3, t3 = _topk_masked(logits, target, row_mask, k=3)
            top3[key] = c3 / t3 if t3 else 0.0

        def _record_ptr_topk(
            key: str,
            logits_seq: torch.Tensor,
            target_seq: torch.Tensor,
            row_mask: torch.Tensor,
        ) -> None:
            c1, t1 = _pointer_seq_top1(logits_seq, target_seq, row_mask)
            top1[key] = c1 / t1 if t1 else 0.0
            c3, t3 = _pointer_seq_topk(logits_seq, target_seq, row_mask, k=3)
            top3[key] = c3 / t3 if t3 else 0.0

        # ---- family CE -----------------------------------------------
        fam_ce, n_fam = _ce_masked(out["family_logits"], family_id, valid_family)
        components["family"] = fam_ce
        n_valid["family"] = n_fam
        _record_topk("family", out["family_logits"], family_id, valid_family)

        # ---- card_seq -----------------------------------------------
        num_cards = batch["num_cards"].long()
        card_ptr = batch["card_ptr_local_seq"].long()

        nc_ce_seq, n_nc_seq = _ce_masked(
            out["card_seq_num_cards_logits"], num_cards, is_card_seq
        )
        components["card_seq_num_cards"] = nc_ce_seq
        n_valid["card_seq_num_cards"] = n_nc_seq
        _record_topk(
            "card_seq_num_cards", out["card_seq_num_cards_logits"], num_cards, is_card_seq
        )

        ptr_ce_seq, n_ptr_seq = _pointer_seq_ce(
            out["card_seq_ptr_logits"], card_ptr, is_card_seq
        )
        components["card_seq_ptr"] = ptr_ce_seq
        n_valid["card_seq_ptr"] = n_ptr_seq
        _record_ptr_topk("card_seq_ptr", out["card_seq_ptr_logits"], card_ptr, is_card_seq)

        # ---- single_ptr ---------------------------------------------
        item_ptr = batch["item_ptr_local"].long()
        item_ce_single, n_item_single = _ce_masked(
            out["item_logits"], item_ptr, is_single_ptr
        )
        components["single_ptr_item"] = item_ce_single
        n_valid["single_ptr_item"] = n_item_single
        _record_topk("single_ptr_item", out["item_logits"], item_ptr, is_single_ptr)

        # ---- chained_cards ------------------------------------------
        item_ce_chained, n_item_chained = _ce_masked(
            out["item_logits"], item_ptr, is_chained
        )
        components["chained_item"] = item_ce_chained
        n_valid["chained_item"] = n_item_chained
        _record_topk("chained_item", out["item_logits"], item_ptr, is_chained)

        nc_ce_chained, n_nc_chained = _ce_masked(
            out["chained_num_cards_logits"], num_cards, is_chained
        )
        components["chained_num_cards"] = nc_ce_chained
        n_valid["chained_num_cards"] = n_nc_chained
        _record_topk(
            "chained_num_cards", out["chained_num_cards_logits"], num_cards, is_chained
        )

        chained_ptr_ce, n_chained_ptr = _pointer_seq_ce(
            out["chained_ptr_logits"], card_ptr, is_chained
        )
        components["chained_ptr"] = chained_ptr_ce
        n_valid["chained_ptr"] = n_chained_ptr
        _record_ptr_topk(
            "chained_ptr", out["chained_ptr_logits"], card_ptr, is_chained
        )

        # ---- joker_pair (SWAP) --------------------------------------
        swap_i = batch["swap_i_local"].long()
        swap_j = batch["swap_j_local"].long()
        i_ce, n_i = _ce_masked(out["swap_i_logits"], swap_i, is_joker_pair)
        components["swap_i"] = i_ce
        n_valid["swap_i"] = n_i
        _record_topk("swap_i", out["swap_i_logits"], swap_i, is_joker_pair)

        j_ce, n_j = _ce_masked(out["swap_j_logits"], swap_j, is_joker_pair)
        components["swap_j"] = j_ce
        n_valid["swap_j"] = n_j
        _record_topk("swap_j", out["swap_j_logits"], swap_j, is_joker_pair)

        # ---- total ---------------------------------------------------
        total = sum(components.values())
        return BranchedLossOutput(
            total=total,
            components={k: v.detach() for k, v in components.items()},
            n_valid=n_valid,
            top1=top1,
            top3=top3,
        )

    def per_family_top1(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        id_to_family: list[str],
    ) -> dict[str, dict[str, float | int]]:
        """Compute per-family family-head accuracy for diagnostics."""
        family_id = batch["family_id"].long()
        valid = family_id >= 0
        preds = out["family_logits"].argmax(dim=-1)
        result: dict[str, dict[str, float | int]] = {}
        for fam_id, fam_name in enumerate(id_to_family):
            mask = valid & (family_id == fam_id)
            n = int(mask.sum().item())
            if n == 0:
                continue
            correct = int(((preds == fam_id) & mask).sum().item())
            result[fam_name] = {"n": n, "top1": correct / n}
        return result
