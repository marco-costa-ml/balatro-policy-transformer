#!/usr/bin/env python3
"""
live_encoder.py
===============
Convert a single Lua snapshot dict (as written by ``Balatro/agent_bridge.lua``)
into the model's input tensor batch. Reuses the locked tensorize_step() so the
exact same vocab indices, normalization stats, and feature layout the model
was trained on are produced -- no parallel encoder to drift out of sync.

The snapshot format for ``schema_version == "live/2.0.0"`` matches the
granularize.py 3.0 step shape (Lua emits canonical zone names and inlines
``PendingCards`` into ``objects``):

    {
      "request_id": int,
      "schema_version": "live/2.0.0",
      "page_name": str,
      "source_kind": null,
      "action_subtype": null,
      "state": { ocr-equivalent ints },
      "objects": [
          { "zone": "CurrentHand"|"CurrentJokers"|"PendingCards"|..., ... },
          ...
      ],
      "pending_cards": [ ... ],     # parallel copy of PendingCards entries
      "target_zone": null,
      "target_position": null,
      "persistent_state": { ... },
      "legal_actions": [
          "PlayHand", "SelectCard_CurrentHand_3",
          "BuyShopItem_VoucherShopOfferings_0", "SWAP_0_1", ...
      ]
    }

Legacy ``"live/1.0.0"`` snapshots (Lua's pre-rewrite shape with
``*Selected`` / ``*All`` zones, ``current_hand_or_pack`` /
``selected_cards`` top-level fields, and flat action labels like
``BuyShopItem_0`` / ``SellItem_3``) are accepted as a defensive shim:
``_normalize_legacy_snapshot`` rewrites them to the 2.0 shape and the
caller logs once per session. The shim exists purely to avoid bricking a
running game during a Lua/Python build skew; the hot path is canonical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

# Allow running as a module from anywhere relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map
from family_map import compute_family_map
from mask_builder import (
    build_action_mask,
    build_card_pointer_mask,
    build_item_pointer_mask,
    build_swap_joker_mask,
)
from tensorize import Normalizer, VocabLookup, derive_branched_caps, tensorize_step


# Page → canonical pool zone for the playing-card hand at this state. The
# legacy snapshot shim uses these to lift ``current_hand_or_pack`` cards
# into the right ``objects[].zone`` bucket.
_LEGACY_POOL_ZONE_BY_PAGE: dict[str, str] = {
    "In_Blind": "CurrentHand",
    "In_TarotSpectral_Pack": "TarotSpectralHand",
    "In_JokerStandardPlanet_Pack": "PackOfferings",
}

# Map old flat-index action label families to the new zoned scheme. Each
# entry says how to expand a legacy ``"<Base>_<i>"`` label against the
# current snapshot. Only the live-relevant subset is covered; anything
# unmapped simply gets dropped from the legal mask (Python's tensorizer
# mask is the fallback).
def _expand_legacy_label(label: str, snapshot: dict[str, Any]) -> list[str]:
    """Return a list of canonical labels for one legacy label."""
    # SWAP_i_j and fixed bare labels translate 1:1.
    if label.startswith("SWAP_") or "_" not in label:
        return [label]

    base, _, tail = label.rpartition("_")
    try:
        idx = int(tail)
    except ValueError:
        return [label]

    if base == "SelectCard":
        page = snapshot.get("page_name")
        if page == "In_Blind":
            return [f"SelectCard_CurrentHand_{idx}"]
        if page == "In_TarotSpectral_Pack":
            return [f"SelectCard_TarotSpectralHand_{idx}"]
        return []
    if base == "UseConsumable":
        return [f"UseConsumable_CurrentConsumables_{idx}"]
    if base == "BuyAndUseShopConsumable":
        return [f"BuyAndUseShopConsumable_TopShelfShopOfferings_{idx}"]
    if base == "SelectPackItem":
        return [f"SelectPackItem_PackOfferings_{idx}"]
    if base == "SellItem":
        # Old flat scheme: jokers 0..n-1 then consumables n..n+m-1.
        objects = snapshot.get("objects") or []
        n_jokers = sum(
            1 for o in objects if isinstance(o, dict) and o.get("zone") == "CurrentJokers"
        )
        if idx < n_jokers:
            return [f"SellItem_CurrentJokers_{idx}"]
        return [f"SellItem_CurrentConsumables_{idx - n_jokers}"]
    if base == "BuyShopItem":
        # Old flat scheme: top-shelf, then voucher, then pack.
        objects = snapshot.get("objects") or []
        n_top = sum(
            1 for o in objects if isinstance(o, dict) and o.get("zone") == "TopShelfShopOfferings"
        )
        n_vou = sum(
            1 for o in objects if isinstance(o, dict) and o.get("zone") == "VoucherShopOfferings"
        )
        if idx < n_top:
            return [f"BuyShopItem_TopShelfShopOfferings_{idx}"]
        if idx < n_top + n_vou:
            return [f"BuyShopItem_VoucherShopOfferings_{idx - n_top}"]
        return [f"BuyShopItem_PackShopOfferings_{idx - n_top - n_vou}"]
    return [label]


def _normalize_legacy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a ``"live/1.0.0"`` snapshot into the canonical 2.0 shape.

    Pure transform; the input dict is left untouched. Implementation notes:
    - Drops every object whose zone ends in ``Selected``.
    - Renames ``*All`` zones to their base (e.g. ``CurrentJokersAll`` →
      ``CurrentJokers``).
    - Folds ``current_hand_or_pack`` into ``objects`` under the canonical
      pool zone for the current page.
    - Folds ``selected_cards`` into ``objects`` under ``PendingCards`` and
      sets a top-level ``pending_cards``.
    - Expands flat ``BuyShopItem_<i>`` / ``SellItem_<i>`` / ``SelectCard_<i>``
      style labels in ``legal_actions``.
    - Drops ``target_index``; sets ``target_zone`` / ``target_position`` to
      ``None``.
    """
    out = dict(snapshot)
    out.pop("target_index", None)
    out["target_zone"] = None
    out["target_position"] = None

    objects: list[dict[str, Any]] = []
    for obj in snapshot.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        zone = obj.get("zone")
        if not isinstance(zone, str):
            continue
        if zone.endswith("Selected"):
            continue
        if zone.endswith("All"):
            obj = {**obj, "zone": zone[: -len("All")]}
        objects.append(obj)

    page = snapshot.get("page_name")
    pool_zone = _LEGACY_POOL_ZONE_BY_PAGE.get(page or "")
    if pool_zone:
        for i, card in enumerate(snapshot.get("current_hand_or_pack") or []):
            if not isinstance(card, dict):
                continue
            objects.append({**card, "zone": pool_zone, "position_in_zone": i})

    pending: list[dict[str, Any]] = []
    for i, card in enumerate(snapshot.get("selected_cards") or []):
        if not isinstance(card, dict):
            continue
        pending.append(card)
        objects.append({**card, "zone": "PendingCards", "position_in_zone": i})

    out["objects"] = objects
    out["pending_cards"] = pending
    out.pop("current_hand_or_pack", None)
    out.pop("selected_cards", None)

    legal = []
    for label in snapshot.get("legal_actions") or []:
        if not isinstance(label, str):
            continue
        legal.extend(_expand_legacy_label(label, out))
    out["legal_actions"] = legal
    out["schema_version"] = "live/2.0.0"
    return out


class LiveEncoder:
    """Stateless encoder bridging Lua snapshots and the model batch contract."""

    def __init__(
        self,
        vocab_path: Path = _REPO_ROOT / "artifacts" / "vocab.json",
        normalization_path: Path = _REPO_ROOT / "artifacts" / "normalization.json",
        feature_config_path: Path = _REPO_ROOT / "artifacts" / "feature_config.json",
        action_config_path: Path = _REPO_ROOT / "data" / "action_space_config.json",
    ) -> None:
        self.vocab = VocabLookup(json.loads(vocab_path.read_text(encoding="utf-8")))
        self.normalizer = Normalizer(json.loads(normalization_path.read_text(encoding="utf-8")))
        self.feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
        action_config = json.loads(action_config_path.read_text(encoding="utf-8"))
        self.action_map = compute_action_map(action_config)
        self.label_to_index: dict[str, int] = self.action_map["label_to_index"]
        self.index_to_label: list[str] = self.action_map["index_to_label"]
        self.n_actions: int = int(self.action_map["n_actions"])
        # Branched-policy artifacts: family_map (id ↔ name + decoder shapes)
        # and derived per-family pointer-zone capacities. We build the
        # family map in-process from action_map so the encoder is always
        # in sync with the live action space — no need to load a separate
        # artifact JSON.
        self.family_map = compute_family_map(self.action_map)
        self.branched_caps = derive_branched_caps(self.action_map, self.family_map)
        self.id_to_family: list[str] = list(self.family_map["id_to_family"])
        self.family_to_id: dict[str, int] = dict(self.family_map["family_to_id"])
        self.decoder_shapes: dict[str, str] = dict(self.family_map["decoder_shapes"])
        # Single-shot diagnostic toggle so the agent_server doesn't log
        # "legacy snapshot" on every frame after a Lua build skew.
        self.legacy_snapshot_seen: bool = False

    def normalize_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return a canonical-shape snapshot (no-op for live/2.0.0)."""
        if snapshot.get("schema_version") == "live/2.0.0":
            return snapshot
        # Anything else (no version field, "live/1.0.0", future shims that
        # forgot to update the tag) goes through the legacy translator.
        self.legacy_snapshot_seen = True
        return _normalize_legacy_snapshot(snapshot)

    def build_step(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Project a snapshot's top level into the dict tensorize_step expects."""
        return {
            "page_name": snapshot.get("page_name"),
            "source_kind": None,
            "action_subtype": None,
            "state": snapshot.get("state") or {},
            "objects": snapshot.get("objects") or [],
            "pending_cards": snapshot.get("pending_cards") or [],
            "target_zone": None,
            "target_position": None,
            # Live inference has no supervised target. Keep label-derived
            # indices out of the model batch.
            "action": "",   # unknown live; resolver will fail and we'll override
        }

    def _legal_action_mask(self, legal_labels: list[str]) -> np.ndarray:
        """Convert Lua-emitted legal labels into a length-N_ACTIONS bool mask."""
        mask = np.zeros(self.n_actions, dtype=bool)
        for label in legal_labels or []:
            idx = self.label_to_index.get(label)
            if idx is not None:
                mask[idx] = True
        return mask

    def encode(
        self,
        snapshot: dict[str, Any],
        device: torch.device | None = None,
        history_steps: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        """Encode one snapshot into a (batch_dict, action_mask_np) pair.

        ``batch_dict`` has every key the model expects, with batch dim 1.
        Pointer masks (``item_pointer_mask`` / ``card_pointer_mask`` /
        ``swap_joker_mask``) are zero-filled because the family is not
        known yet; the agent recomputes them via
        :meth:`pointer_masks_for_family` once the family head has fired.

        ``action_mask_np`` is the Lua-reported legality mask kept as
        numpy for diagnostics / fallback.
        """
        snapshot = self.normalize_snapshot(snapshot)
        step = self.build_step(snapshot)
        pstate = snapshot.get("persistent_state") or {}

        # Reuse the canonical tensorizer; it ignores `action_subtype_id` /
        # `source_kind_id` at model-input time (LeakageNote in model.py) so
        # null entries are fine. Passing the family_map causes
        # tensorize_step to emit the branched-policy channels
        # (family_mask, item_pointer_mask, card_pointer_mask,
        # swap_joker_mask, family_id, etc.); the pointer masks come back
        # all-False because ``family_name_for_pointer`` is None.
        record = tensorize_step(
            step=step,
            persistent_state=pstate,
            action_map=self.action_map,
            vocab=self.vocab,
            norm=self.normalizer,
            feature_config=self.feature_config,
            history_steps=history_steps or [],
            family_map=self.family_map,
            branched_caps=self.branched_caps,
            family_name_for_pointer=None,
        )

        # Override the auto-derived action mask with the Lua ground-truth
        # legality. The Lua side knows exactly which actions are dispatchable
        # (G.STATE + cardareas + dollars + voucher inventory).
        legal_mask = self._legal_action_mask(snapshot.get("legal_actions") or [])
        # If Lua reports nothing (defensive), fall back to the tensorizer's
        # mask so we don't softmax over an all-False vector.
        if not legal_mask.any():
            legal_mask = record["action_mask"].astype(bool)
        record["action_mask"] = legal_mask

        # Stack batch dim and ship to torch.
        batch: dict[str, torch.Tensor] = {}
        for key, arr in record.items():
            t = torch.from_numpy(np.asarray(arr))
            t = t.unsqueeze(0)  # batch dim
            if device is not None:
                t = t.to(device)
            batch[key] = t
        return batch, legal_mask

    def pointer_masks_for_family(
        self,
        snapshot: dict[str, Any],
        family_name: str,
        device: torch.device | None = None,
        lua_legal_mask: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute item / card / swap pointer masks for ``family_name``.

        Used by the live agent after the family head has selected a
        family — the encoder pass above filled these with zeros because
        the family wasn't known yet. ``snapshot`` should be in canonical
        ``live/2.0.0`` shape (call :meth:`normalize_snapshot` first).

        ``lua_legal_mask`` should be the Lua-emitted flat legality mask;
        we intersect with it so a buggy mask_builder rule cannot widen
        the set of legal arguments past what the game accepts. Pass
        ``None`` to skip this defensive AND.
        """
        step = self.build_step(snapshot)
        caps = self.branched_caps
        if lua_legal_mask is not None:
            flat_mask = lua_legal_mask.astype(bool)
        else:
            flat_mask = build_action_mask(step, self.action_map).astype(bool)
        item_mask = build_item_pointer_mask(
            family_name,
            step,
            self.action_map,
            max_size=int(caps["MAX_ITEM_ZONE_SIZE"]),
            flat_mask=flat_mask,
        )
        card_mask = build_card_pointer_mask(
            family_name,
            step,
            self.action_map,
            max_size=int(caps["MAX_CARD_ZONE_SIZE"]),
            flat_mask=flat_mask,
        )
        if family_name == "SWAP":
            swap_mask = build_swap_joker_mask(
                step, max_joker_slots=int(caps["MAX_JOKER_SLOTS"])
            )
        else:
            swap_mask = np.zeros(int(caps["MAX_JOKER_SLOTS"]), dtype=bool)

        def to_torch(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(arr)).unsqueeze(0)
            if device is not None:
                t = t.to(device)
            return t

        return {
            "item_pointer_mask": to_torch(item_mask),
            "card_pointer_mask": to_torch(card_mask),
            "swap_joker_mask": to_torch(swap_mask),
        }

    def label_for_index(self, idx: int) -> str:
        if 0 <= idx < len(self.index_to_label):
            return self.index_to_label[idx]
        return f"<oob:{idx}>"
