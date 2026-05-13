#!/usr/bin/env python3
"""
dataset.py
==========
PyTorch ``Dataset`` over the tensorized Balatro corpus.

Each item is a single (step, action_mask, target_action_id) triple drawn
from a per-run ``.npz`` shard at ``data/tensorized/video_id=*/run_*.npz``.

Why this shape?
---------------
We're training a per-step policy classifier (behavior cloning), so steps
are the natural units. Steps with ``target_action_id == -1`` (the small
upstream label-resolution failures) are excluded at index time.

Splits
------
``BalatroStepDataset`` accepts a ``split_videos`` argument (a list of
video IDs) and only yields steps from those videos. The companion
``load_split`` helper handles reading ``artifacts/splits.json``.

Loading model
-------------
Each shard is mmapped on first access and cached in memory; for the
~5 MiB total corpus this is a trivial overhead but keeps the on-disk
layout truly content-addressable per-video.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# Channels listed for the model. Keep order stable to mirror tensorize.py.
CATEGORICAL_KEYS = (
    "page_id",
    "source_kind_id",
    "action_subtype_id",
    "deck_class_id",
    "stake_class_id",
    "last_tarot_planet_class_id",
    "ante_boss_blind_class_id",
    "small_status_id",
    "big_status_id",
)

OBJECT_CATEGORICAL_KEYS = (
    "object_class_id",
    "object_object_type_id",
    "object_zone_id",
    "object_modifier_id",
    "object_edition_id",
    "object_seal_id",
    "object_rank_id",
    "object_suit_id",
)

OBJECT_BOOL_KEYS = (
    "object_is_debuffed",
    "object_sticker_rental",
    "object_sticker_perishable",
    "object_sticker_eternal",
)

DECK_CATEGORICAL_KEYS = (
    "deck_card_class_id",
    "deck_card_modifier_id",
    "deck_card_edition_id",
    "deck_card_seal_id",
    "deck_card_rank_id",
    "deck_card_suit_id",
)


def load_split(splits_path: Path, name: str) -> list[str]:
    """Return the video_ids for a named split (``train`` / ``val`` / ``test``)."""
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    return list(payload["splits"][name]["video_ids"])


class BalatroStepDataset(Dataset):
    """One sample = one granularized step's tensorized record.

    All matching shards are loaded eagerly and **concatenated into one
    tensor per channel** at construction time. This avoids per-item Python
    dispatch in the training loop (the entire corpus is ~5 MiB, so RAM is
    not a concern), and it lets us optionally move every channel onto the
    GPU once via ``to(device)``, after which DataLoader iteration is
    essentially free.
    """

    def __init__(
        self,
        tensorized_root: Path,
        split_videos: Iterable[str],
        include_unresolved: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        self.root = Path(tensorized_root)
        self.include_unresolved = include_unresolved

        # Per-channel concatenated tensor stash (CPU by default; pushed to
        # GPU at the end if ``device`` is set).
        per_channel: dict[str, list[torch.Tensor]] = {}
        valid_indices: list[np.ndarray] = []
        offset = 0
        meta: dict[str, Any] = {}

        for vid in sorted(split_videos):
            partition = self.root / f"video_id={vid}"
            if not partition.exists():
                continue
            for run_file in sorted(partition.glob("run_*.npz")):
                with np.load(run_file) as z:
                    arrays = {key: np.array(z[key]) for key in z.files}
                if not arrays:
                    continue
                if not meta:
                    meta = {k: tuple(v.shape[1:]) for k, v in arrays.items()}
                n = int(arrays["target_action_id"].shape[0])
                for key, arr in arrays.items():
                    per_channel.setdefault(key, []).append(_to_tensor_2d(arr))
                if include_unresolved:
                    valid_indices.append(np.arange(offset, offset + n, dtype=np.int64))
                else:
                    target = arrays["target_action_id"]
                    where = np.where(target >= 0)[0]
                    if len(where) > 0:
                        valid_indices.append(where.astype(np.int64) + offset)
                offset += n

        if not per_channel:
            self._tensors: dict[str, torch.Tensor] = {}
            self._valid: torch.Tensor = torch.empty(0, dtype=torch.long)
            self._meta = {}
            return

        self._tensors = {
            key: torch.cat(parts, dim=0) for key, parts in per_channel.items()
        }
        self._valid = torch.from_numpy(np.concatenate(valid_indices)) if valid_indices else torch.empty(0, dtype=torch.long)
        self._meta = meta

        if device is not None:
            target_device = torch.device(device)
            self._tensors = {
                k: v.to(target_device, non_blocking=True) for k, v in self._tensors.items()
            }
            self._valid = self._valid.to(target_device, non_blocking=True)

    # --- pythonic dataset API ---

    def __len__(self) -> int:
        return int(self._valid.shape[0])

    def meta(self) -> dict[str, Any]:
        """Per-channel non-batch shape, e.g. ``ocr_numeric -> (16,)``."""
        return dict(self._meta)

    def n_actions(self) -> int:
        return int(self._meta.get("action_mask", (0,))[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Translate from valid-only index to flat-corpus index.
        flat = int(self._valid[idx].item())
        return {key: tensor[flat] for key, tensor in self._tensors.items()}

    # --- bulk APIs for fast iteration ---

    def all_tensors(self) -> dict[str, torch.Tensor]:
        """Return the concatenated per-channel tensors (full corpus)."""
        return self._tensors

    def valid_indices(self) -> torch.Tensor:
        """Long indices into the flat corpus, restricted to resolvable steps."""
        return self._valid

    def gather_batch(self, sample_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather a batch from already-resolved valid indices."""
        if sample_indices.device != self._valid.device:
            sample_indices = sample_indices.to(self._valid.device)
        flat = self._valid.index_select(0, sample_indices)
        return {
            key: tensor.index_select(0, flat) for key, tensor in self._tensors.items()
        }


def _to_tensor_2d(arr: np.ndarray) -> torch.Tensor:
    """Convert a stacked-per-step ndarray (shape ``(n_steps, *)``) to a torch tensor."""
    if arr.dtype == bool:
        return torch.from_numpy(arr.astype(np.bool_, copy=True))
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr.astype(np.int64, copy=True))
    return torch.from_numpy(arr.astype(np.float32, copy=True))


def _to_tensor(arr) -> torch.Tensor:
    """Convert a numpy array (possibly a 0-d scalar) into a torch tensor.

    NOTE: do NOT route through ``np.ascontiguousarray``; on 0-d inputs it
    silently promotes to shape ``(1,)`` which then makes scalar fields look
    like ``(B, 1)`` after collation. ``arr.astype(..., copy=True)`` always
    yields a contiguous copy (the default) which is what ``from_numpy``
    needs, while preserving 0-d shape for scalars.
    """
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return torch.from_numpy(arr.astype(np.bool_, copy=True))
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr.astype(np.int64, copy=True))
    return torch.from_numpy(arr.astype(np.float32, copy=True))


def collate_steps(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-step dicts into a batch dict."""
    if not batch:
        return {}
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
