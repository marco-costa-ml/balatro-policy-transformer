#!/usr/bin/env python3
"""Runtime diagnostics for SWAP label/mask/model failures."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import BalatroStepDataset, load_split
from granularize import (
    SWAP_EXCLUDED_ACTIONS,
    joker_slot_order,
    objects_with_joker_order,
    parse_action_base_and_index,
    swap_pairs_to_transform,
)
from label_resolver import check_step_consistency, resolve_label
from mask_builder import build_action_mask


def _is_spurious_skippack_before_selection(step, next_step) -> bool:
    """Legacy in-stream filter for SkipPack rows immediately followed by a
    same-page select. Kept here as a no-op shim: the new granularize
    schema already prevents these from being emitted, but historical
    debug paths still reference this name.
    """
    return False

SESSION_ID = "fd8f21"
LOG_PATH = _REPO_ROOT / "debug-fd8f21.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _iter_run_files(root: Path):
    for partition in sorted(root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for run_file in sorted(partition.glob("run_*.json")):
            yield video_id, run_file


def _load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_swap_label(label: Any) -> bool:
    return isinstance(label, str) and label.startswith("SWAP_")


def _swap_offset_and_pairs(action_map: dict[str, Any]) -> tuple[int, list[list[int]]]:
    return int(action_map["family_offsets"]["SWAP"]), action_map["swap_family"]["pairs"]


def diagnose_action_map(action_map: dict[str, Any], run_id: str) -> None:
    swap_offset, pairs = _swap_offset_and_pairs(action_map)
    mismatches = []
    for k, pair in enumerate(pairs):
        label = f"SWAP_{int(pair[0])}_{int(pair[1])}"
        expected_idx = swap_offset + k
        actual_idx = int(action_map["label_to_index"].get(label, -1))
        index_label = action_map["index_to_label"][expected_idx]
        if actual_idx != expected_idx or index_label != label:
            mismatches.append(
                {
                    "pair": pair,
                    "label": label,
                    "expected_idx": expected_idx,
                    "actual_idx": actual_idx,
                    "index_label": index_label,
                }
            )

    # region agent log
    _debug_log(
        run_id,
        "H2_H3",
        "tools/debug_swap_diagnostics.py:diagnose_action_map",
        "SWAP action_map index and pair ordering check",
        {
            "swap_offset": swap_offset,
            "swap_size": len(pairs),
            "first_pairs": pairs[:10],
            "last_pairs": pairs[-5:],
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:10],
        },
    )
    # endregion


def diagnose_granularized(root: Path, action_map: dict[str, Any], run_id: str) -> None:
    swap_offset, pairs = _swap_offset_and_pairs(action_map)
    pair_to_k = {(int(i), int(j)): k for k, (i, j) in enumerate(pairs)}
    total = 0
    masked_out = 0
    consistency_failures = collections.Counter()
    pair_counts = collections.Counter()
    pages = collections.Counter()
    examples = []

    order_total = 0
    obs_matches_final = 0
    obs_matches_expected_pre = 0
    obs_matches_after_apply = 0
    order_examples = []
    wrong_timestep_examples = []
    wrong_set_same = 0
    wrong_set_changed = 0
    repairable_with_reorder = 0
    first_swap_by_source = 0
    later_swap_by_source = 0

    for _, run_file in _iter_run_files(root):
        run = _load_run(run_file)
        steps = run.get("events", [])
        last_boundary_order: list[int] | None = None
        last_boundary_step: dict[str, Any] | None = None
        source_to_prev_boundary: dict[int, list[int] | None] = {}
        source_to_prev_boundary_step: dict[int, dict[str, Any] | None] = {}
        source_to_final_order: dict[int, list[int]] = {}

        for step in steps:
            source_idx = int(step.get("source_event_index", -1))
            label = step.get("action")
            order = joker_slot_order(step)
            if _is_swap_label(label):
                source_to_prev_boundary.setdefault(source_idx, list(last_boundary_order) if last_boundary_order else None)
                source_to_prev_boundary_step.setdefault(source_idx, last_boundary_step)
                if order:
                    source_to_final_order.setdefault(source_idx, order)
            elif order:
                base_action, _ = parse_action_base_and_index(str(label or ""))
                if base_action not in SWAP_EXCLUDED_ACTIONS:
                    last_boundary_order = order
                    last_boundary_step = step

        block_progress: dict[int, list[int]] = {}
        block_counts: collections.Counter[int] = collections.Counter()
        for step in steps:
            label = step.get("action")
            if not _is_swap_label(label):
                continue

            total += 1
            pair = tuple(int(x) for x in (step.get("swap_pair") or []))
            pair_counts[pair] += 1
            pages[step.get("page_name") or "<none>"] += 1

            diagnostic = check_step_consistency(step, action_map)
            if diagnostic is not None:
                consistency_failures[diagnostic.split(":", 1)[0]] += 1

            mask = build_action_mask(step, action_map)
            target = resolve_label(step, action_map["label_to_index"])
            if not bool(mask[target]):
                masked_out += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "run_file": str(run_file.relative_to(root.parent)),
                            "step_id": step.get("step_id"),
                            "action": label,
                            "target": target,
                            "page": step.get("page_name"),
                            "jokers_current_state": (step.get("state") or {}).get("jokers_current"),
                            "mask_swap_enabled_count": int(mask[swap_offset : swap_offset + len(pairs)].sum()),
                        }
                    )

            source_idx = int(step.get("source_event_index", -1))
            prev_boundary = source_to_prev_boundary.get(source_idx)
            prev_boundary_step = source_to_prev_boundary_step.get(source_idx)
            final_order = source_to_final_order.get(source_idx) or joker_slot_order(step)
            obs_order = joker_slot_order(step)
            expected_pre = block_progress.get(source_idx)
            if expected_pre is None and prev_boundary:
                expected_pre = list(prev_boundary)
                block_progress[source_idx] = list(expected_pre)
            block_counts[source_idx] += 1
            if block_counts[source_idx] == 1:
                first_swap_by_source += 1
            else:
                later_swap_by_source += 1

            if expected_pre and len(pair) == 2:
                order_total += 1
                if obs_order == final_order:
                    obs_matches_final += 1
                if obs_order == expected_pre:
                    obs_matches_expected_pre += 1
                after = list(expected_pre)
                i, j = pair
                if i < len(after) and j < len(after):
                    after[i], after[j] = after[j], after[i]
                    if obs_order == after:
                        obs_matches_after_apply += 1
                    block_progress[source_idx] = after
                expected_k = pair_to_k.get(pair, -1)
                if target != swap_offset + expected_k and len(order_examples) < 8:
                    order_examples.append(
                        {
                            "run_file": str(run_file.relative_to(root.parent)),
                            "step_id": step.get("step_id"),
                            "action": label,
                            "pair": pair,
                            "target": target,
                            "expected_target": swap_offset + expected_k,
                        }
                    )
                if obs_order != expected_pre and len(wrong_timestep_examples) < 8:
                    same_set = set(obs_order) == set(expected_pre)
                    if same_set:
                        wrong_set_same += 1
                    else:
                        wrong_set_changed += 1
                    simulated_order = None
                    if prev_boundary_step is not None:
                        simulated = objects_with_joker_order(
                            prev_boundary_step.get("objects") or [],
                            expected_pre,
                        )
                        simulated_order = joker_slot_order({"objects": simulated})
                        if simulated_order == expected_pre:
                            repairable_with_reorder += 1
                    wrong_timestep_examples.append(
                        {
                            "run_file": str(run_file.relative_to(root.parent)),
                            "step_id": step.get("step_id"),
                            "source_event_index": source_idx,
                            "action": label,
                            "prev_boundary_order": prev_boundary,
                            "expected_pre_order_for_this_swap": expected_pre,
                            "observed_order_in_step": obs_order,
                            "final_order": final_order,
                            "same_joker_set": same_set,
                            "simulated_reorder_from_prev_boundary": simulated_order,
                            "prev_boundary_step_id": (
                                prev_boundary_step.get("step_id")
                                if prev_boundary_step is not None
                                else None
                            ),
                            "prev_boundary_action": (
                                prev_boundary_step.get("action")
                                if prev_boundary_step is not None
                                else None
                            ),
                        }
                    )
                elif obs_order != expected_pre:
                    if set(obs_order) == set(expected_pre):
                        wrong_set_same += 1
                    else:
                        wrong_set_changed += 1
                    if prev_boundary_step is not None:
                        simulated = objects_with_joker_order(
                            prev_boundary_step.get("objects") or [],
                            expected_pre,
                        )
                        simulated_order = joker_slot_order({"objects": simulated})
                        if simulated_order == expected_pre:
                            repairable_with_reorder += 1

    # region agent log
    _debug_log(
        run_id,
        "H1_H2_H3",
        "tools/debug_swap_diagnostics.py:diagnose_granularized",
        "Granularized SWAP label, mask, and target-index compatibility",
        {
            "swap_total": total,
            "masked_out_target_count": masked_out,
            "consistency_failures": dict(consistency_failures),
            "page_counts": dict(pages),
            "pair_counts_top10": pair_counts.most_common(10),
            "masked_out_examples": examples,
            "target_pair_order_mismatch_examples": order_examples,
        },
    )
    # endregion

    # region agent log
    _debug_log(
        run_id,
        "H5_H6",
        "tools/debug_swap_diagnostics.py:diagnose_granularized",
        "SWAP observation timing: pre-swap versus final joker order",
        {
            "order_checked_count": order_total,
            "obs_matches_expected_pre_count": obs_matches_expected_pre,
            "obs_matches_final_count": obs_matches_final,
            "obs_matches_after_apply_count": obs_matches_after_apply,
            "wrong_set_same_count": wrong_set_same,
            "wrong_set_changed_count": wrong_set_changed,
            "repairable_with_reorder_count": repairable_with_reorder,
            "first_swap_by_source_count": first_swap_by_source,
            "later_swap_by_source_count": later_swap_by_source,
            "wrong_timestep_examples": wrong_timestep_examples,
        },
    )
    # endregion


def diagnose_tensorized(root: Path, splits_path: Path, action_map: dict[str, Any], run_id: str) -> None:
    swap_offset, pairs = _swap_offset_and_pairs(action_map)
    swap_end = swap_offset + len(pairs)
    split_counts = {}
    split_masked = {}
    split_enabled_stats = {}
    split_target_counts = {}
    split_swap_legal_counts = {}
    split_swap_legal_negative_counts = {}
    split_swap_positive_rate_when_legal = {}

    for split in ("train", "val", "test"):
        ds = BalatroStepDataset(root, load_split(splits_path, split), include_unresolved=False)
        tensors = ds.all_tensors()
        if not tensors:
            continue
        valid = ds.valid_indices()
        target = tensors["target_action_id"].long().squeeze(-1)
        masks = tensors["action_mask"].bool()
        flat_target = target.index_select(0, valid)
        flat_masks = masks.index_select(0, valid)
        is_swap = (flat_target >= swap_offset) & (flat_target < swap_end)
        swap_legal_any = flat_masks[:, swap_offset:swap_end].any(dim=-1)
        swap_targets = flat_target[is_swap]
        swap_masks = flat_masks[is_swap]
        enabled = swap_masks[:, swap_offset:swap_end].sum(dim=-1) if int(is_swap.sum()) else None
        masked_target = 0
        if int(is_swap.sum()):
            rows = np.arange(int(is_swap.sum()))
            masked_target = int((~swap_masks[rows, swap_targets.numpy()]).sum())
        split_counts[split] = int(is_swap.sum())
        split_masked[split] = masked_target
        split_swap_legal_counts[split] = int(swap_legal_any.sum())
        split_swap_legal_negative_counts[split] = int((swap_legal_any & ~is_swap).sum())
        split_swap_positive_rate_when_legal[split] = float(is_swap.sum().item() / max(int(swap_legal_any.sum()), 1))
        split_enabled_stats[split] = {
            "min": int(enabled.min()) if enabled is not None else 0,
            "max": int(enabled.max()) if enabled is not None else 0,
            "mean": float(enabled.float().mean()) if enabled is not None else 0.0,
        }
        split_target_counts[split] = collections.Counter(
            action_map["index_to_label"][int(x)] for x in swap_targets.numpy().tolist()
        ).most_common(10)

    # region agent log
    _debug_log(
        run_id,
        "H1_H4",
        "tools/debug_swap_diagnostics.py:diagnose_tensorized",
        "Tensorized SWAP counts and mask-target compatibility by split",
        {
            "split_swap_counts": split_counts,
            "split_masked_target_counts": split_masked,
            "split_swap_legal_counts": split_swap_legal_counts,
            "split_swap_legal_negative_counts": split_swap_legal_negative_counts,
            "split_swap_positive_rate_when_legal": split_swap_positive_rate_when_legal,
            "split_swap_enabled_action_stats": split_enabled_stats,
            "split_target_counts_top10": split_target_counts,
        },
    )
    # endregion


def _metadata_for_split(root: Path, split: str) -> list[dict[str, Any]]:
    split_videos = set(load_split(root / "artifacts/splits.json", split))
    metas: list[dict[str, Any]] = []
    tensor_root = root / "data/tensorized"
    gran_root = root / "data/granularized"
    for partition in sorted(tensor_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        if video_id not in split_videos:
            continue
        for run_file in sorted(partition.glob("run_*.npz")):
            run_idx = int(run_file.stem.split("_", 1)[1])
            granular_path = gran_root / f"video_id={video_id}" / f"run_{run_idx:03d}.json"
            run = json.loads(granular_path.read_text(encoding="utf-8"))
            events = run.get("events") or []
            with np.load(run_file) as z:
                n = int(z["target_action_id"].shape[0])
            included_events = []
            for t, step in enumerate(events):
                next_step = events[t + 1] if t + 1 < len(events) else None
                if _is_spurious_skippack_before_selection(step, next_step):
                    continue
                included_events.append(step)
            for t in range(n):
                step = included_events[t] if t < len(included_events) else {}
                metas.append(
                    {
                        "video_id": video_id,
                        "run_index": run_idx,
                        "step_id": step.get("step_id"),
                        "page_name": step.get("page_name"),
                        "action": step.get("action"),
                        "source_action": step.get("source_action"),
                        "source_kind": step.get("source_kind"),
                        "source_event_index": step.get("source_event_index"),
                        "joker_order": joker_slot_order(step),
                    }
                )
    return metas


def diagnose_model(root: Path, split: str, action_map: dict[str, Any], ckpt_path: Path, run_id: str) -> None:
    import torch

    from model import ModelConfig, PolicyTransformer, load_model_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_in = ckpt.get("model_config")
    if cfg_in is not None:
        cfg = ModelConfig(**{k: v for k, v in cfg_in.items() if k in ModelConfig.__dataclass_fields__})
    else:
        cfg = load_model_config(int(action_map["n_actions"]), root / "artifacts/vocab.json")
    model = PolicyTransformer(cfg).to(device).eval()
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    model.load_state_dict(state)

    ds = BalatroStepDataset(root / "data/tensorized", load_split(root / "artifacts/splits.json", split), device=device)
    meta = _metadata_for_split(root, split)
    tensors = ds.all_tensors()
    valid = ds.valid_indices()
    target_all = tensors["target_action_id"].long().squeeze(-1)
    swap_offset, pairs = _swap_offset_and_pairs(action_map)
    swap_end = swap_offset + len(pairs)
    flat_target = target_all.index_select(0, valid)
    rel = torch.nonzero((flat_target >= swap_offset) & (flat_target < swap_end), as_tuple=False).squeeze(-1)

    pred_counts = collections.Counter()
    target_counts = collections.Counter()
    rank_counts = collections.Counter()
    examples = []
    top1 = 0
    top3 = 0
    total = int(rel.numel())

    with torch.no_grad():
        for start in range(0, total, 512):
            idx = rel[start : start + 512]
            batch = ds.gather_batch(idx)
            logits = model(batch)
            topk = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
            target = batch["target_action_id"].long().view(-1)
            preds = topk[:, 0]
            top1 += int((preds == target).sum().item())
            top3 += int((topk[:, :3] == target.unsqueeze(-1)).any(dim=-1).sum().item())
            for row in range(target.shape[0]):
                tgt = int(target[row].item())
                pred = int(preds[row].item())
                target_counts[action_map["index_to_label"][tgt]] += 1
                pred_counts[action_map["index_to_label"][pred]] += 1
                match = torch.nonzero(topk[row] == target[row], as_tuple=False)
                rank = int(match[0].item()) + 1 if int(match.numel()) else None
                rank_counts[str(rank) if rank is not None else ">10"] += 1
                if len(examples) < 10:
                    legal = batch["action_mask"][row].bool().detach().cpu().numpy()
                    legal_labels = [action_map["index_to_label"][i] for i, ok in enumerate(legal) if ok]
                    flat_idx = int(valid.index_select(0, idx)[row].detach().cpu().item())
                    examples.append(
                        {
                            "meta": meta[flat_idx] if flat_idx < len(meta) else {"flat_idx": flat_idx},
                            "target": action_map["index_to_label"][tgt],
                            "pred": action_map["index_to_label"][pred],
                            "target_rank_top10": rank,
                            "top5": [action_map["index_to_label"][int(x)] for x in topk[row, :5].detach().cpu().tolist()],
                            "legal_count": len(legal_labels),
                            "legal_non_swap_top20": [x for x in legal_labels if not x.startswith("SWAP_")][:20],
                            "legal_swap_count": sum(1 for x in legal_labels if x.startswith("SWAP_")),
                        }
                    )

    # region agent log
    _debug_log(
        run_id,
        "H4_H5_H6",
        "tools/debug_swap_diagnostics.py:diagnose_model",
        "Model predictions on tensorized SWAP validation rows",
        {
            "split": split,
            "total": total,
            "top1": top1 / max(total, 1),
            "top3": top3 / max(total, 1),
            "target_counts_top10": target_counts.most_common(10),
            "pred_counts_top10": pred_counts.most_common(10),
            "target_rank_counts": dict(rank_counts),
            "examples": examples,
        },
    )
    # endregion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=Path("artifacts/checkpoints/best.pt"))
    args = parser.parse_args()

    root = args.root.resolve()
    run_id = f"swap-diagnostics-{args.split}"
    action_map = json.loads((root / "data/action_map.json").read_text(encoding="utf-8"))

    diagnose_action_map(action_map, run_id)
    diagnose_granularized(root / "data/granularized", action_map, run_id)
    diagnose_tensorized(root / "data/tensorized", root / "artifacts/splits.json", action_map, run_id)
    if args.with_model:
        ckpt = args.ckpt if args.ckpt.is_absolute() else root / args.ckpt
        diagnose_model(root, args.split, action_map, ckpt, run_id)

    print(f"wrote diagnostics to {LOG_PATH}")


if __name__ == "__main__":
    main()
