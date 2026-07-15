#!/usr/bin/env python3
"""
train.py
========
Behavior-cloning training loop for the branched-autoregressive Balatro
policy transformer.

Loss (branched_loss.BranchedLoss)
---------------------------------
We jointly train:

- **family head** — cross-entropy over the 19-way ``family_id`` target,
  with ``family_mask`` forcing ``logits[~mask] = -inf``.
- **shape-specific decoders** — for each row the relevant heads
  contribute additive CE terms (PlayHand/DiscardHand → num_cards +
  ordered card-pointer sequence; Buy/Sell/BuyAndUse → item pointer;
  UseConsumable/SelectPackItem → item + num_cards + chained card
  pointer sequence; SWAP → swap_i + swap_j with i excluded from j's
  mask). Each component is mean-reduced over its own valid-row mask so
  per-head losses stay comparable across batches.

The dataset's per-row branched-validity filter (see ``dataset.py``)
already drops the small fraction of rows where the supervised pointer
target would violate the legality mask, so masked-CE never sees an
unreachable target.

Metrics
-------
- ``family_top1`` and per-shape pointer / num_cards top-1.
- Per-family family-head accuracy (e.g. PlayHand vs SWAP vs
  BuyShopItem_*).
- ``macro_family_top1`` (unweighted mean over families that appear in
  the split).

Performance
-----------
The entire tensorized corpus is preloaded onto the training device
(GPU when available) at startup. Training and eval iterate via
``index_select`` on these resident tensors -- no DataLoader workers,
no host->device copies per batch -- which keeps a 4070-class GPU
saturated.

Throughput is logged every ``--log-every`` iterations along with
rolling total loss / family top-1 / steps-per-second / ETA.

Usage
-----
``python train.py
    [--tensorized data/tensorized]
    [--splits artifacts/splits.json]
    [--out artifacts/checkpoints]
    [--epochs 10] [--batch-size 32] [--lr 5e-4]
    [--limit-train 0] [--device auto]``

``--limit-train N`` (default 0 = no limit) is useful for quick smoke runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from action_map import compute_action_map
from branched_loss import BranchedLoss
from dataset import BalatroStepDataset, load_split
from family_map import compute_family_map
from model import ModelConfig, PolicyTransformer, load_model_config, n_params
from tensorize import derive_branched_caps


CHECKPOINT_SCHEMA_VERSION = "2.0.0"  # bumped for branched-policy heads


def _macro_family_metrics(
    per_family_top1: dict[str, float],
    per_family_total: dict[str, int],
) -> dict[str, float]:
    """Macro-average per-family top-1 across the families that appear."""
    accs = list(per_family_top1.values())
    out: dict[str, float] = {
        "macro_family_top1": float(sum(accs) / len(accs)) if accs else 0.0,
    }
    for thr in (100, 500):
        sub = [per_family_top1[f] for f, n in per_family_total.items() if n >= thr]
        key = f"macro_family_top1_min{thr}"
        out[key] = float(sum(sub) / len(sub)) if sub else None
    return out


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(
    tensorized: Path,
    splits_path: Path,
    name: str,
    *,
    family_map: dict | None = None,
    branched_caps: dict[str, int] | None = None,
    device: torch.device | None = None,
) -> BalatroStepDataset:
    return BalatroStepDataset(
        tensorized_root=tensorized,
        split_videos=load_split(splits_path, name),
        include_unresolved=False,
        device=device,
        family_map=family_map,
        branched_caps=branched_caps,
        require_branched=True,
    )


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move one gathered CPU-resident minibatch to the training device."""
    if device.type == "cpu":
        return batch
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:5.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m:2d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


def iter_batches(
    n: int,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
    device: torch.device,
):
    """Yield long-tensor index batches into ``[0, n)``, on ``device``."""
    if shuffle:
        order = torch.randperm(n, generator=generator, device=device)
    else:
        order = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def _evaluate(
    model: PolicyTransformer,
    dataset: BalatroStepDataset,
    loss_fn: BranchedLoss,
    batch_size: int,
    device: torch.device,
    *,
    label: str = "eval",
    log_every: int = 0,
    family_map: dict | None = None,
) -> dict[str, float | dict[str, float | int | None]]:
    """Evaluate the branched policy. Returns aggregate + per-head metrics."""
    model.eval()
    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    t0 = time.time()

    total_loss = 0.0
    total_rows = 0

    # Accumulators for per-head CE and top-1.
    head_loss_sum: dict[str, float] = collections.defaultdict(float)
    head_n_sum: dict[str, int] = collections.defaultdict(int)
    head_correct: dict[str, int] = collections.defaultdict(int)
    head_total: dict[str, int] = collections.defaultdict(int)
    head_top3_correct: dict[str, int] = collections.defaultdict(int)
    per_family_correct: collections.Counter = collections.Counter()
    per_family_top3_correct: collections.Counter = collections.Counter()
    per_family_total: collections.Counter = collections.Counter()

    id_to_family = family_map["id_to_family"] if family_map else []

    index_device = dataset.valid_indices().device
    for it, idx in enumerate(
        iter_batches(n, batch_size, shuffle=False, device=index_device), 1
    ):
        batch = _move_batch(dataset.gather_batch(idx), device)
        out = model(batch)
        result = loss_fn(out, batch)

        B = int(batch["family_id"].shape[0])
        total_rows += B
        total_loss += float(result.total.item()) * B

        for k, v in result.components.items():
            n_k = result.n_valid[k]
            head_loss_sum[k] += float(v.item()) * n_k
            head_n_sum[k] += n_k
        for k, top1 in result.top1.items():
            n_k = result.n_valid[k]
            head_correct[k] += int(round(top1 * n_k))
            head_total[k] += n_k
        for k, top3 in result.top3.items():
            n_k = result.n_valid[k]
            head_top3_correct[k] += int(round(top3 * n_k))
            head_total[k] += 0  # already counted above

        # Per-family family-head accuracy.
        family_id = batch["family_id"].long()
        valid = family_id >= 0
        family_logits = out["family_logits"]
        preds = family_logits.argmax(dim=-1)
        k_eff = min(3, int(family_logits.shape[-1]))
        top3_preds = torch.topk(family_logits, k=k_eff, dim=-1).indices
        for b in range(B):
            if not bool(valid[b]):
                continue
            fid = int(family_id[b])
            fam_name = id_to_family[fid] if 0 <= fid < len(id_to_family) else f"id={fid}"
            per_family_total[fam_name] += 1
            if int(preds[b]) == fid:
                per_family_correct[fam_name] += 1
            if bool((top3_preds[b] == fid).any()):
                per_family_top3_correct[fam_name] += 1

        if log_every and it % log_every == 0:
            elapsed = time.time() - t0
            sps = total_rows / max(elapsed, 1e-6)
            print(
                f"  [{label}] {it}/{n_batches}  "
                f"rows={total_rows}/{n}  "
                f"running_loss={total_loss/max(total_rows,1):.4f}  "
                f"sps={sps:7.0f}"
            )

    denom_rows = max(total_rows, 1)
    head_loss_mean = {
        k: head_loss_sum[k] / max(head_n_sum[k], 1) for k in head_loss_sum
    }
    head_top1_mean = {
        k: head_correct[k] / max(head_total[k], 1) for k in head_correct
    }
    head_top3_mean = {
        k: head_top3_correct[k] / max(head_total[k], 1) for k in head_correct
    }
    per_family_top1 = {
        fam: per_family_correct[fam] / per_family_total[fam]
        for fam in per_family_total
    }
    per_family_top3 = {
        fam: per_family_top3_correct[fam] / per_family_total[fam]
        for fam in per_family_total
    }
    macro = _macro_family_metrics(per_family_top1, dict(per_family_total))
    macro_top3 = _macro_family_metrics(per_family_top3, dict(per_family_total))
    return {
        "loss": total_loss / denom_rows,
        "n_rows": total_rows,
        "head_loss": head_loss_mean,
        "head_top1": head_top1_mean,
        "head_top3": head_top3_mean,
        "family_top1": head_top1_mean.get("family", 0.0),
        "family_top3": head_top3_mean.get("family", 0.0),
        "per_family_top1": per_family_top1,
        "per_family_top3": per_family_top3,
        "per_family_total": dict(per_family_total),
        **macro,
        "macro_family_top3": macro_top3["macro_family_top1"],
        "macro_family_top3_min100": macro_top3["macro_family_top1_min100"],
        "macro_family_top3_min500": macro_top3["macro_family_top1_min500"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensorized", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--action-config", type=Path, default=Path("data/action_space_config.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/checkpoints"))
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue training from a branched-policy checkpoint. "
             "--epochs means additional epochs.",
    )
    ap.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Resume model weights but start a fresh optimizer state.",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="If >0, restrict training set to first N samples (for smoke runs).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--history-steps",
        type=int,
        default=None,
        help="Override model history step cap; default comes from feature_config.",
    )
    ap.add_argument(
        "--history-objects-per-step",
        type=int,
        default=None,
        help="Override model history object cap; default comes from feature_config.",
    )
    ap.add_argument(
        "--use-tracked-deck-tokens",
        action="store_true",
        help="Include tracked deck cards as attention tokens for ablation/backcompat.",
    )
    ap.add_argument("--history-step-dropout", type=float, default=0.15)
    ap.add_argument("--history-object-dropout", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=20,
                    help="Print a training-progress line every N optimizer steps.")
    ap.add_argument("--amp", action="store_true",
                    help="Enable mixed-precision (CUDA only).")
    ap.add_argument(
        "--checkpoint-metric",
        choices=("loss", "family_top1", "macro_family_top1"),
        default="loss",
        help="Val metric used to select the best checkpoint.",
    )
    args = ap.parse_args()

    _seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    print(f"device: {device}  amp={use_amp}")

    t_load = time.time()
    print("loading family map + branched caps...")
    action_map = compute_action_map(
        json.loads(args.action_config.read_text(encoding="utf-8"))
    )
    family_map = compute_family_map(action_map)
    branched_caps = derive_branched_caps(action_map, family_map)
    print(
        f"  family_map_version={family_map['family_map_version']} "
        f"n_families={family_map['n_families']}"
    )
    print(f"  branched_caps={branched_caps}")

    print("loading datasets (CPU-resident; minibatches move to device)...")
    train_ds = _build_dataset(
        args.tensorized, args.splits, "train",
        family_map=family_map, branched_caps=branched_caps,
    )
    val_ds = _build_dataset(
        args.tensorized, args.splits, "val",
        family_map=family_map, branched_caps=branched_caps,
    )
    test_ds = _build_dataset(
        args.tensorized, args.splits, "test",
        family_map=family_map, branched_caps=branched_caps,
    )
    n_actions = train_ds.n_actions()
    print(
        f"  train={len(train_ds):>6d}  val={len(val_ds):>5d}  test={len(test_ds):>5d}  "
        f"N_ACTIONS={n_actions}  ({time.time()-t_load:.1f}s)"
    )

    if args.limit_train > 0 and args.limit_train < len(train_ds):
        # Truncate the valid index in place; original tensors stay CPU-resident.
        train_ds._valid = train_ds._valid[: args.limit_train]
        print(f"  limiting train to first {args.limit_train} steps")

    print("building model...")
    resume_ckpt: dict[str, object] | None = None
    if args.resume is not None:
        resume_path = args.resume
        if not resume_path.is_file():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        cfg_payload = dict(resume_ckpt["model_config"])
        cfg_payload["n_actions"] = n_actions
        cfg = ModelConfig(**{
            k: v for k, v in cfg_payload.items()
            if k in ModelConfig.__dataclass_fields__
        })
        if int(resume_ckpt.get("n_actions", n_actions)) != n_actions:
            raise SystemExit(
                f"resume checkpoint n_actions={resume_ckpt.get('n_actions')} "
                f"but tensorized data has n_actions={n_actions}"
            )
    else:
        cfg_overrides = {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "use_tracked_deck_tokens": bool(args.use_tracked_deck_tokens),
            "use_history_tokens": True,
            "history_step_dropout": float(args.history_step_dropout),
            "history_object_dropout": float(args.history_object_dropout),
        }
        if args.history_steps is not None:
            cfg_overrides["history_steps"] = int(args.history_steps)
        if args.history_objects_per_step is not None:
            cfg_overrides["history_objects_per_step"] = int(args.history_objects_per_step)
        cfg = load_model_config(
            n_actions=n_actions,
            **cfg_overrides,
        )
    model = PolicyTransformer(cfg).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(
            f"  resumed model from {args.resume.as_posix()} "
            f"(checkpoint epoch={resume_ckpt.get('epoch')})"
        )
    print(f"  params: {n_params(model):,}")
    print(f"  d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"max_objects={cfg.max_objects} max_deck={cfg.max_deck_cards}")
    print(
        f"  history_steps={cfg.history_steps} "
        f"history_objects_per_step={cfg.history_objects_per_step} "
        f"use_history_tokens={cfg.use_history_tokens} "
        f"use_tracked_deck_tokens={cfg.use_tracked_deck_tokens}"
    )
    print(
        f"  history_dropout step={cfg.history_step_dropout:g} "
        f"object={cfg.history_object_dropout:g}"
    )

    checkpoint_metric = args.checkpoint_metric
    loss_fn = BranchedLoss(family_map).to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if (
        resume_ckpt is not None
        and not args.resume_reset_optimizer
        and "optimizer_state_dict" in resume_ckpt
    ):
        optim.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("  resumed optimizer state")
    elif resume_ckpt is not None and args.resume_reset_optimizer:
        print("  resume-reset-optimizer enabled; using fresh optimizer state")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = float("inf") if checkpoint_metric == "loss" else -float("inf")
    best_path: Path | None = None
    resume_epoch = 0
    if resume_ckpt is not None:
        resume_epoch = int(resume_ckpt.get("epoch") or 0)
        if checkpoint_metric == "loss":
            best_val = float(resume_ckpt.get("val_loss", float("inf")))
        else:
            best_val = float(
                resume_ckpt.get(f"val_{checkpoint_metric}", -float("inf"))
            )
        best_path = args.resume

    n_train = len(train_ds)
    batch_size = args.batch_size
    n_batches_train = (n_train + batch_size - 1) // batch_size
    index_device = train_ds.valid_indices().device
    train_gen = torch.Generator(device=index_device).manual_seed(args.seed)

    print()
    start_epoch = resume_epoch + 1
    end_epoch = resume_epoch + args.epochs
    if resume_ckpt is not None:
        print(
            f"--- resuming at epoch {start_epoch}; running {args.epochs} additional epochs "
            f"through epoch {end_epoch} ({n_batches_train} train batches/epoch, "
            f"batch_size={batch_size}) ---"
        )
    else:
        print(f"--- training {args.epochs} epochs "
              f"({n_batches_train} train batches/epoch, batch_size={batch_size}) ---")
    overall_t0 = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_t0 = time.time()
        window_t0 = epoch_t0
        window_loss_sum = 0.0
        window_rows = 0
        window_fam_correct = 0
        window_fam_total = 0
        running_loss_sum = 0.0
        running_rows = 0
        running_fam_correct = 0
        running_fam_total = 0

        for it, idx in enumerate(
            iter_batches(
                n_train, batch_size, shuffle=True, generator=train_gen,
                device=index_device,
            ),
            1,
        ):
            batch = _move_batch(train_ds.gather_batch(idx), device)
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(batch)
                    result = loss_fn(out, batch)
                    loss = result.total
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                out = model(batch)
                result = loss_fn(out, batch)
                loss = result.total
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            B = int(batch["family_id"].shape[0])
            lt = float(loss.item())
            window_loss_sum += lt * B
            running_loss_sum += lt * B
            window_rows += B
            running_rows += B
            fam_t = result.top1.get("family", 0.0)
            fam_n = result.n_valid.get("family", 0)
            window_fam_correct += int(round(fam_t * fam_n))
            running_fam_correct += int(round(fam_t * fam_n))
            window_fam_total += fam_n
            running_fam_total += fam_n

            if args.log_every and it % args.log_every == 0:
                now = time.time()
                window_secs = now - window_t0
                window_sps = window_rows / max(window_secs, 1e-6)
                avg_loss = running_loss_sum / max(running_rows, 1)
                avg_fam = running_fam_correct / max(running_fam_total, 1)
                w_loss = window_loss_sum / max(window_rows, 1)
                w_fam = window_fam_correct / max(window_fam_total, 1)
                seen_batches = it
                rem_in_epoch = n_batches_train - seen_batches
                rem_total = rem_in_epoch + n_batches_train * (args.epochs - epoch)
                cum_sps = running_rows / max(now - epoch_t0, 1e-6)
                eta = rem_total * batch_size / max(cum_sps, 1e-6)
                print(
                    f"  e{epoch}/{args.epochs} it {it:4d}/{n_batches_train}  "
                    f"loss(win/run)={w_loss:.4f}/{avg_loss:.4f}  "
                    f"fam_top1(win/run)={w_fam:.4f}/{avg_fam:.4f}  "
                    f"sps={window_sps:7.0f}  ETA={_format_eta(eta)}"
                )
                window_t0 = now
                window_loss_sum = 0.0
                window_rows = 0
                window_fam_correct = 0
                window_fam_total = 0

        train_loss = running_loss_sum / max(running_rows, 1)
        train_fam_top1 = running_fam_correct / max(running_fam_total, 1)
        epoch_secs = time.time() - epoch_t0
        epoch_sps = running_rows / max(epoch_secs, 1e-6)
        print(
            f"epoch {epoch}/{args.epochs} TRAIN done  "
            f"loss={train_loss:.4f}  fam_top1={train_fam_top1:.4f}  "
            f"sps={epoch_sps:6.0f}  ({epoch_secs:.1f}s)"
        )

        val_metrics = _evaluate(
            model,
            val_ds,
            loss_fn,
            args.eval_batch_size,
            device,
            label=f"val(e{epoch})",
            log_every=0,
            family_map=family_map,
        )
        macro = val_metrics.get("macro_family_top1")
        print(
            f"epoch {epoch}/{args.epochs} VAL  "
            f"loss={val_metrics['loss']:.4f}  "
            f"fam_top1={val_metrics['family_top1']:.4f}  "
            f"macro_fam={macro:.4f}"
        )
        for head, top1 in sorted(val_metrics["head_top1"].items()):
            n = val_metrics.get("head_n_sum", {}).get(head, "?")
            print(f"    val/head/{head:24s}  top1={top1:.4f}")
        for fam, acc in sorted(val_metrics["per_family_top1"].items(),
                               key=lambda kv: -val_metrics["per_family_total"][kv[0]]):
            n_fam = val_metrics["per_family_total"][fam]
            print(f"    val/family/{fam:55s}  top1={acc:.4f}  n={n_fam}")

        if checkpoint_metric == "loss":
            ck_score = float(val_metrics["loss"])
            improved = ck_score < best_val
        else:
            ck_score = float(val_metrics[checkpoint_metric])
            improved = ck_score > best_val

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_family_top1": train_fam_top1,
            "val_loss": val_metrics["loss"],
            "val_family_top1": val_metrics["family_top1"],
            "val_macro_family_top1": val_metrics["macro_family_top1"],
            "val_macro_family_top1_min100": val_metrics["macro_family_top1_min100"],
            "val_macro_family_top1_min500": val_metrics["macro_family_top1_min500"],
            "val_head_top1": val_metrics["head_top1"],
            "val_head_loss": val_metrics["head_loss"],
            "val_per_family_top1": val_metrics["per_family_top1"],
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_score": ck_score,
            "epoch_seconds": epoch_secs,
        })

        if improved:
            best_val = ck_score
            best_path = args.out / "best.pt"
            torch.save({
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "checkpoint_metric": checkpoint_metric,
                "val_loss": val_metrics["loss"],
                "val_family_top1": val_metrics["family_top1"],
                "val_macro_family_top1": val_metrics["macro_family_top1"],
                "val_head_top1": val_metrics["head_top1"],
                "model_config": cfg.__dict__,
                "n_actions": n_actions,
                "family_map_version": family_map["family_map_version"],
                "branched_caps": branched_caps,
            }, best_path)
            print(f"  -> saved {best_path.as_posix()}")

    elapsed_total = time.time() - overall_t0
    print()
    print(f"training complete in {_format_eta(elapsed_total)}")

    print()
    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded best checkpoint from epoch {ckpt['epoch']}")
    test_metrics = _evaluate(
        model,
        test_ds,
        loss_fn,
        args.eval_batch_size,
        device,
        label="test",
        family_map=family_map,
    )
    print(
        f"TEST  loss={test_metrics['loss']:.4f}  "
        f"fam_top1={test_metrics['family_top1']:.4f}  "
        f"macro_fam={test_metrics['macro_family_top1']:.4f}"
    )
    for head, top1 in sorted(test_metrics["head_top1"].items()):
        print(f"  test/head/{head:24s}  top1={top1:.4f}")
    for fam, acc in sorted(test_metrics["per_family_top1"].items(),
                           key=lambda kv: -test_metrics["per_family_total"][kv[0]]):
        n_fam = test_metrics["per_family_total"][fam]
        print(f"  test/family/{fam:55s}  top1={acc:.4f}  n={n_fam}")

    tensorizer_event_base_counts_reference: dict[str, int] | None = None
    tr_report_path = Path("artifacts/tensorizer_report.json")
    if tr_report_path.is_file():
        try:
            tensorizer_event_base_counts_reference = json.loads(
                tr_report_path.read_text(encoding="utf-8")
            ).get("event_base_counts")
        except (OSError, json.JSONDecodeError):
            tensorizer_event_base_counts_reference = None

    report = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checkpoint_metric": checkpoint_metric,
        "n_params": n_params(model),
        "n_actions": n_actions,
        "n_families": family_map["n_families"],
        "family_map_version": family_map["family_map_version"],
        "branched_caps": branched_caps,
        "model_config": cfg.__dict__,
        "history_config": {
            "history_steps": cfg.history_steps,
            "history_objects_per_step": cfg.history_objects_per_step,
            "use_tracked_deck_tokens": cfg.use_tracked_deck_tokens,
            "use_history_tokens": cfg.use_history_tokens,
            "history_step_dropout": cfg.history_step_dropout,
            "history_object_dropout": cfg.history_object_dropout,
        },
        "device": str(device),
        "amp": use_amp,
        "elapsed_seconds": elapsed_total,
        "history": history,
        "test_metrics": test_metrics,
        "tensorizer_event_base_counts_reference": tensorizer_event_base_counts_reference,
        "best_checkpoint": best_path.as_posix() if best_path else None,
    }
    (args.out / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {(args.out / 'training_report.json').as_posix()}")


if __name__ == "__main__":
    main()
