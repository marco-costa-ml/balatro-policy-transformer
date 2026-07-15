#!/usr/bin/env python3
"""
Within-family logit bias from train-split action counts.

    adjusted_logit(a) = logit(a) + tau * log((C_family + eps) / (n_a + eps))

where C_family is the sum of train counts over all flat labels sharing the same
family prefix (text before first '_').
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def family_totals_from_counts(
    counts: np.ndarray,
    index_to_family: list[str],
) -> np.ndarray:
    """Per flat index a: total train count for family(a)."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if len(index_to_family) != len(counts):
        raise ValueError(
            f"len(index_to_family)={len(index_to_family)} != len(counts)={len(counts)}"
        )
    fam_sum: dict[str, float] = {}
    for fam, c in zip(index_to_family, counts):
        fam_sum[fam] = fam_sum.get(fam, 0.0) + float(c)
    out = np.zeros(len(counts), dtype=np.float64)
    for i, fam in enumerate(index_to_family):
        out[i] = fam_sum[fam]
    return out


def log_ratio_bias(
    counts: np.ndarray,
    index_to_family: list[str],
    *,
    eps: float = 1.0,
    max_log_ratio: float | None = None,
) -> torch.Tensor:
    """Bias vector log((C+eps)/(n+eps)) per action index (CPU float32)."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    c_fam = family_totals_from_counts(counts, index_to_family)
    ratio = (c_fam + eps) / (counts + eps)
    ratio = np.maximum(ratio, 1e-30)
    log_r = np.log(ratio)
    if max_log_ratio is not None:
        log_r = np.minimum(log_r, float(max_log_ratio))
    return torch.from_numpy(log_r.astype(np.float32))


def apply_logit_adjust(
    logits: torch.Tensor,
    bias: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """logits: (B, n_actions); bias: (n_actions,)."""
    if tau == 0.0:
        return logits
    return logits + tau * bias.unsqueeze(0).to(device=logits.device, dtype=logits.dtype)


def load_counts_json(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load artifact written by ``scripts/build_action_train_counts.py``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = np.asarray(payload["count_per_action"], dtype=np.int64)
    labels = list(payload["index_to_label"])
    if len(labels) != len(counts):
        raise ValueError("count_per_action and index_to_label length mismatch")
    return counts, labels
