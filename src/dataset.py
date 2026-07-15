#!/usr/bin/env python3
"""
dataset.py
==========
PyTorch ``Dataset`` over the tensorized Balatro corpus.

Each item is a single **parent-level super-step** drawn from a per-run
``.npz`` shard at ``data/tensorized/video_id=*/run_*.npz`` (tensorize
schema 3.0.0). One super-step covers an entire branched-policy
decision: family head target + per-shape argument targets + their masks
+ the legacy v1 ``(target_action_id, action_mask)`` for back-compat.

Why this shape?
---------------
The branched autoregressive policy is trained per-decision (one parent
super-step at a time). All shape-specific tensors live alongside each
row, and the model picks the relevant head based on ``family_id``.

Validity filter
---------------
``include_unresolved=False`` (the default) excludes rows that cannot
supervise the model:

- ``target_action_id < 0``  (legacy v1 filter — covers StartNewRun and
  the small number of upstream label/mask anomalies).
- ``family_id < 0`` (StartNewRun / unresolved family).
- ``family_mask[family_id] == 0`` (pre-existing data anomaly).
- For each shape, the required pointer indices must be in range AND
  selected by their per-zone pointer mask. ``num_cards`` clamped to
  ``[1, MAX_CARDS_PER_DECISION]`` for card_seq, ``[0, ...]`` for
  chained_cards. SWAP additionally requires ``i != j``.

Set ``include_unresolved=True`` to retain every row (useful when
running diagnostics or training a pure family-only head).

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


def _compute_branched_validity(
    arrays: dict[str, np.ndarray],
    family_map: dict[str, Any] | None,
    caps: dict[str, int] | None,
    *,
    require_branched: bool = True,
) -> np.ndarray:
    """Return a boolean (n_rows,) mask of trainable rows.

    A row is trainable iff:
    - ``target_action_id >= 0`` (legacy v1 invariant), AND
    - if ``require_branched``: ``family_id >= 0`` and shape-specific
      pointer/cardinality invariants hold (mirrors validate_training_contract
      tensorized-mode checks).
    """
    target_action_id = arrays["target_action_id"]
    valid = target_action_id >= 0
    if not require_branched:
        return valid
    if family_map is None or caps is None:
        return valid

    family_ids = arrays["family_id"].astype(np.int64)
    n = family_ids.shape[0]
    n_families = int(family_map["n_families"])
    valid &= family_ids >= 0
    valid &= family_ids < n_families

    if not valid.any():
        return valid

    family_mask = arrays["family_mask"].astype(bool)
    # Gather family_mask[k, family_ids[k]] safely.
    safe_fids = np.where(valid, family_ids, 0)
    fam_legal = family_mask[np.arange(n), safe_fids]
    valid &= fam_legal

    id_to_family = family_map["id_to_family"]
    decoder_shapes = family_map["decoder_shapes"]
    max_cards = int(caps["MAX_CARDS_PER_DECISION"])
    max_item = int(caps["MAX_ITEM_ZONE_SIZE"])
    max_card_zone = int(caps["MAX_CARD_ZONE_SIZE"])
    max_joker = int(caps["MAX_JOKER_SLOTS"])

    item_pm = arrays["item_pointer_mask"].astype(bool)
    card_pm = arrays["card_pointer_mask"].astype(bool)
    swap_pm = arrays["swap_joker_mask"].astype(bool)
    item_ptr = arrays["item_ptr_local"].astype(np.int64)
    card_ptr_seq = arrays["card_ptr_local_seq"].astype(np.int64)
    num_cards = arrays["num_cards"].astype(np.int64)
    swap_i = arrays["swap_i_local"].astype(np.int64)
    swap_j = arrays["swap_j_local"].astype(np.int64)

    valid_idx = np.where(valid)[0]
    for k in valid_idx:
        fid = int(family_ids[k])
        fname = id_to_family[fid]
        shape = decoder_shapes.get(fname, "unknown")

        if shape in ("no_args", "reserved"):
            continue

        if shape == "card_seq":
            nc = int(num_cards[k])
            if not (1 <= nc <= max_cards):
                valid[k] = False
                continue
            ok = True
            for i in range(nc):
                p = int(card_ptr_seq[k, i])
                if not (0 <= p < max_card_zone) or not bool(card_pm[k, p]):
                    ok = False
                    break
            if not ok:
                valid[k] = False
            continue

        if shape == "single_ptr":
            p = int(item_ptr[k])
            if not (0 <= p < max_item) or not bool(item_pm[k, p]):
                valid[k] = False
            continue

        if shape == "chained_cards":
            p = int(item_ptr[k])
            if not (0 <= p < max_item) or not bool(item_pm[k, p]):
                valid[k] = False
                continue
            nc = int(num_cards[k])
            if not (0 <= nc <= max_cards):
                valid[k] = False
                continue
            ok = True
            for i in range(nc):
                cp = int(card_ptr_seq[k, i])
                if not (0 <= cp < max_card_zone) or not bool(card_pm[k, cp]):
                    ok = False
                    break
            if not ok:
                valid[k] = False
            continue

        if shape == "joker_pair":
            i_pos = int(swap_i[k])
            j_pos = int(swap_j[k])
            if i_pos == j_pos:
                valid[k] = False
                continue
            if not (0 <= i_pos < max_joker and 0 <= j_pos < max_joker):
                valid[k] = False
                continue
            if not (bool(swap_pm[k, i_pos]) and bool(swap_pm[k, j_pos])):
                valid[k] = False
            continue

        # Unknown shape — drop it.
        valid[k] = False

    return valid


class BalatroStepDataset(Dataset):
    """One sample = one parent-level super-step's tensorized record.

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
        family_map: dict[str, Any] | None = None,
        branched_caps: dict[str, int] | None = None,
        require_branched: bool = True,
    ) -> None:
        self.root = Path(tensorized_root)
        self.include_unresolved = include_unresolved
        self.require_branched = require_branched
        self._family_map = family_map
        self._branched_caps = branched_caps

        # Per-channel concatenated tensor stash (CPU by default; pushed to
        # GPU at the end if ``device`` is set).
        per_channel: dict[str, list[torch.Tensor]] = {}
        valid_indices: list[np.ndarray] = []
        offset = 0
        meta: dict[str, Any] = {}
        n_rows_seen = 0
        n_rows_kept = 0

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
                n_rows_seen += n
                for key, arr in arrays.items():
                    per_channel.setdefault(key, []).append(_to_tensor_2d(arr))
                if include_unresolved:
                    valid_indices.append(np.arange(offset, offset + n, dtype=np.int64))
                    n_rows_kept += n
                else:
                    valid_mask = _compute_branched_validity(
                        arrays,
                        family_map=family_map,
                        caps=branched_caps,
                        require_branched=require_branched,
                    )
                    where = np.where(valid_mask)[0]
                    if len(where) > 0:
                        valid_indices.append(where.astype(np.int64) + offset)
                    n_rows_kept += int(len(where))
                offset += n

        self.n_rows_seen = n_rows_seen
        self.n_rows_kept = n_rows_kept

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

    def n_families(self) -> int:
        """Number of branched-policy families (matches ``family_map``)."""
        return int(self._meta.get("family_mask", (0,))[0])

    def branched_caps(self) -> dict[str, int]:
        """Return the per-shape capacity tuple (item/card/joker zone + max cards)."""
        return {
            "MAX_ITEM_ZONE_SIZE": int(self._meta.get("item_pointer_mask", (0,))[0]),
            "MAX_CARD_ZONE_SIZE": int(self._meta.get("card_pointer_mask", (0,))[0]),
            "MAX_JOKER_SLOTS": int(self._meta.get("swap_joker_mask", (0,))[0]),
            "MAX_CARDS_PER_DECISION": int(self._meta.get("card_ptr_local_seq", (0, 0))[0]),
        }

    def family_id_histogram(self) -> dict[int, int]:
        """Per-family row counts over the valid sample set (after filtering)."""
        if "family_id" not in self._tensors or self._valid.numel() == 0:
            return {}
        fids = self._tensors["family_id"].index_select(0, self._valid.cpu()).cpu().numpy().astype(np.int64)
        unique, counts = np.unique(fids, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts, strict=False)}

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
