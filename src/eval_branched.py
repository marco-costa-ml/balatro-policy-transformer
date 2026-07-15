#!/usr/bin/env python3
"""
eval_branched.py
================
Standalone offline evaluation for the branched-policy checkpoint.

Reuses the ``_evaluate`` function from ``train.py`` so the numbers match
what training reports per epoch. Loads a checkpoint, builds the val
(default) or test dataset, runs one forward pass over the whole split,
and prints per-head + per-family top-1 metrics.

Usage:
    python eval_branched.py
        --checkpoint artifacts/checkpoints/best.pt
        [--split val|test]
        [--tensorized data/tensorized]
        [--splits-path data/splits.json]
        [--batch-size 128]
        [--device auto]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from action_map import compute_action_map
from branched_loss import BranchedLoss
from family_map import compute_family_map
from model import ModelConfig, PolicyTransformer
from tensorize import derive_branched_caps
from train import _build_dataset, _evaluate


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _print_metrics(metrics: dict, split_name: str) -> None:
    head_loss = metrics.get("head_loss", {})
    head_top1 = metrics.get("head_top1", {})
    head_top3 = metrics.get("head_top3", {})
    per_fam = metrics.get("per_family_top1", {})
    per_fam_top3 = metrics.get("per_family_top3", {})
    per_fam_total = metrics.get("per_family_total", {})

    print()
    print(f"=== {split_name} metrics ===")
    print(f"loss:                {metrics['loss']:.4f}")
    print(f"n_rows:              {metrics['n_rows']:,}")
    print(f"family_top1:         {metrics['family_top1']:.4f}")
    if "family_top3" in metrics:
        print(f"family_top3:         {metrics['family_top3']:.4f}")
    if "macro_family_top1" in metrics:
        print(f"macro_family_top1:   {metrics['macro_family_top1']:.4f}")
    if "macro_family_top3" in metrics:
        print(f"macro_family_top3:   {metrics['macro_family_top3']:.4f}")
    for thr in (100, 500):
        key = f"macro_family_top1_min{thr}"
        if key in metrics:
            print(f"  top1 (>={thr} rows): {metrics[key]:.4f}")
        key3 = f"macro_family_top3_min{thr}"
        if key3 in metrics:
            print(f"  top3 (>={thr} rows): {metrics[key3]:.4f}")

    print()
    print("Per-head top-1 / top-3:")
    for head, acc in sorted(head_top1.items()):
        loss = head_loss.get(head)
        acc3 = head_top3.get(head)
        if loss is None:
            print(f"  {head:30s}  top1={acc:.4f}  top3={acc3:.4f}")
        else:
            print(f"  {head:30s}  top1={acc:.4f}  top3={acc3:.4f}  ce={loss:.4f}")

    print()
    print("Per-family top-1 / top-3 (sorted by row count, descending):")
    for fam, acc in sorted(per_fam.items(), key=lambda kv: -per_fam_total.get(kv[0], 0)):
        n_fam = per_fam_total.get(fam, 0)
        acc3 = per_fam_top3.get(fam, 0.0)
        print(f"  {fam:55s}  n={n_fam:7,}  top1={acc:.4f}  top3={acc3:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/best.pt"),
    )
    ap.add_argument(
        "--split",
        choices=("val", "test", "train"),
        default="val",
        help="Which split to evaluate (default: val).",
    )
    ap.add_argument(
        "--tensorized",
        type=Path,
        default=Path("data/tensorized"),
        help="Root of tensorized shards.",
    )
    ap.add_argument(
        "--splits-path",
        type=Path,
        default=Path("artifacts/splits.json"),
    )
    ap.add_argument(
        "--action-config",
        type=Path,
        default=Path("data/action_space_config.json"),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda | auto",
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"device:              {device}")
    print(f"checkpoint:          {args.checkpoint.as_posix()}")

    # Load the action / family map artifacts up-front so the dataset
    # filter can vet rows against the same caps as training.
    action_config = json.loads(args.action_config.read_text(encoding="utf-8"))
    action_map = compute_action_map(action_config)
    family_map = compute_family_map(action_map)
    branched_caps = derive_branched_caps(action_map, family_map)
    print(
        f"family_map_version:  {family_map['family_map_version']} "
        f"({family_map['n_families']} families)"
    )

    print(f"building {args.split} dataset...")
    dataset = _build_dataset(
        args.tensorized,
        args.splits_path,
        args.split,
        family_map=family_map,
        branched_caps=branched_caps,
        device=torch.device("cpu"),
    )
    print(f"  rows: {len(dataset):,}")
    n_actions = int(dataset.n_actions())

    # Reconstruct the model the same way the live agent does so the
    # head shapes match the checkpoint's saved state_dict exactly.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_payload = dict(ckpt["model_config"])
    cfg_payload["n_actions"] = n_actions
    cfg = ModelConfig(**{
        k: v for k, v in cfg_payload.items()
        if k in ModelConfig.__dataclass_fields__
    })
    model = PolicyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"  loaded epoch={ckpt.get('epoch')} "
        f"val_top1={ckpt.get('val_top1')} "
        f"schema={ckpt.get('schema_version')}"
    )

    loss_fn = BranchedLoss(family_map=family_map).to(device)

    metrics = _evaluate(
        model,
        dataset,
        loss_fn,
        args.batch_size,
        device,
        label=args.split,
        family_map=family_map,
    )
    _print_metrics(metrics, args.split)


if __name__ == "__main__":
    main()
