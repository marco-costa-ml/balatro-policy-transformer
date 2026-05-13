#!/usr/bin/env python3
"""
train.py
========
Behavior-cloning training loop for the Balatro policy transformer.

Loss
----
Mask-aware categorical cross-entropy. We always force ``logits[~mask] = -inf``
before computing the softmax, so gradients only flow through legal actions.
The ``mask_builder`` invariant guarantees ``action_mask[target_action_id] == 1``
for every resolvable step in the corpus, so the masked-CE never sees an
unreachable target.

Metrics
-------
- top-1 accuracy
- top-3 accuracy
- per-family top-1 accuracy (e.g. SelectCard vs PlayHand vs SWAP)
- mean masked-CE loss

Performance
-----------
The entire ~5 MiB tensorized corpus is preloaded onto the training device
(GPU when available) at startup. Training and eval iterate via
``index_select`` on these resident tensors -- no DataLoader workers, no
host->device copies per batch -- which keeps a 4070-class GPU saturated.

Throughput is logged every ``--log-every`` iterations along with rolling
loss / top-1 / steps-per-second / ETA.

Usage
-----
``python train.py
    [--tensorized data/tensorized]
    [--splits artifacts/splits.json]
    [--out artifacts/checkpoints]
    [--epochs 10] [--batch-size 256] [--lr 5e-4]
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
from torch import nn

from action_map import compute_action_map
from dataset import BalatroStepDataset, load_split
from model import PolicyTransformer, load_model_config, n_params


CHECKPOINT_SCHEMA_VERSION = "1.0.0"


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(
    tensorized: Path,
    splits_path: Path,
    name: str,
    device: torch.device,
) -> BalatroStepDataset:
    return BalatroStepDataset(
        tensorized_root=tensorized,
        split_videos=load_split(splits_path, name),
        include_unresolved=False,
        device=device,
    )


def _label_to_family(label: str) -> str:
    """Base family name. ``BuyShopItem_VoucherShopOfferings_0`` -> ``BuyShopItem``."""
    return label.split("_", 1)[0]


def _label_to_subfamily(label: str) -> str:
    """Per-zone subfamily key (drops trailing index).

    Examples:
        ``BuyShopItem_VoucherShopOfferings_0`` -> ``BuyShopItem_VoucherShopOfferings``
        ``SelectCard_CurrentHand_3``           -> ``SelectCard_CurrentHand``
        ``PlayHand``                           -> ``PlayHand``
        ``SWAP_0_1``                           -> ``SWAP``
    """
    if label.startswith("SWAP_"):
        return "SWAP"
    parts = label.rsplit("_", 1)
    if len(parts) == 2:
        try:
            int(parts[1])
            return parts[0]
        except ValueError:
            return label
    return label


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
    batch_size: int,
    device: torch.device,
    index_to_family: list[str],
    *,
    label: str = "eval",
    log_every: int = 0,
) -> dict[str, float | dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    n_top1 = 0
    n_top3 = 0
    per_family_total: collections.Counter = collections.Counter()
    per_family_top1: collections.Counter = collections.Counter()

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    t0 = time.time()
    for it, idx in enumerate(iter_batches(n, batch_size, shuffle=False, device=device), 1):
        batch = dataset.gather_batch(idx)
        target = batch["target_action_id"].long().view(-1)
        logits = model(batch)
        loss = nn.functional.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss.item())

        top3 = logits.topk(3, dim=-1).indices
        preds = top3[:, 0]
        n_top1 += int((preds == target).sum().item())
        n_top3 += int((top3 == target.unsqueeze(-1)).any(dim=-1).sum().item())
        total_steps += int(target.numel())

        # Cheap per-family aggregation done on CPU after batch finishes.
        tgt_np = target.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        for tgt, p in zip(tgt_np, pred_np):
            fam = index_to_family[int(tgt)]
            per_family_total[fam] += 1
            if int(p) == int(tgt):
                per_family_top1[fam] += 1

        if log_every and it % log_every == 0:
            elapsed = time.time() - t0
            sps = total_steps / max(elapsed, 1e-6)
            print(
                f"  [{label}] {it}/{n_batches}  "
                f"steps={total_steps}/{n}  "
                f"running_top1={n_top1/max(total_steps,1):.4f}  "
                f"sps={sps:7.0f}"
            )

    return {
        "loss": total_loss / max(total_steps, 1),
        "top1": n_top1 / max(total_steps, 1),
        "top3": n_top3 / max(total_steps, 1),
        "n_steps": total_steps,
        "per_family_top1": {
            fam: per_family_top1[fam] / per_family_total[fam]
            for fam in per_family_total
        },
        "per_family_total": dict(per_family_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensorized", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--action-config", type=Path, default=Path("data/action_space_config.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/checkpoints"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
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
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=20,
                    help="Print a training-progress line every N optimizer steps.")
    ap.add_argument("--amp", action="store_true",
                    help="Enable mixed-precision (CUDA only).")
    args = ap.parse_args()

    _seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    print(f"device: {device}  amp={use_amp}")

    t_load = time.time()
    print("loading datasets (preloading entire corpus to device)...")
    train_ds = _build_dataset(args.tensorized, args.splits, "train", device)
    val_ds = _build_dataset(args.tensorized, args.splits, "val", device)
    test_ds = _build_dataset(args.tensorized, args.splits, "test", device)
    n_actions = train_ds.n_actions()
    print(
        f"  train={len(train_ds):>6d}  val={len(val_ds):>5d}  test={len(test_ds):>5d}  "
        f"N_ACTIONS={n_actions}  ({time.time()-t_load:.1f}s)"
    )

    if args.limit_train > 0 and args.limit_train < len(train_ds):
        # Truncate the valid index in place; original tensors stay GPU-resident.
        train_ds._valid = train_ds._valid[: args.limit_train]
        print(f"  limiting train to first {args.limit_train} steps")

    print("building model...")
    cfg = load_model_config(
        n_actions=n_actions,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = PolicyTransformer(cfg).to(device)
    print(f"  params: {n_params(model):,}")
    print(f"  d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"max_objects={cfg.max_objects} max_deck={cfg.max_deck_cards}")

    # action label index for per-family metric breakdowns
    action_map = compute_action_map(
        json.loads(args.action_config.read_text(encoding="utf-8"))
    )
    index_to_family = [_label_to_family(label) for label in action_map["index_to_label"]]

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = float("inf")
    best_path: Path | None = None

    n_train = len(train_ds)
    batch_size = args.batch_size
    n_batches_train = (n_train + batch_size - 1) // batch_size
    train_gen = torch.Generator(device=device).manual_seed(args.seed)

    print()
    print(f"--- training {args.epochs} epochs "
          f"({n_batches_train} train batches/epoch, batch_size={batch_size}) ---")
    overall_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()
        window_t0 = epoch_t0
        window_loss = 0.0
        window_steps = 0
        window_correct = 0
        running_loss = 0.0
        running_steps = 0
        running_correct = 0

        for it, idx in enumerate(
            iter_batches(n_train, batch_size, shuffle=True, generator=train_gen, device=device),
            1,
        ):
            batch = train_ds.gather_batch(idx)
            target = batch["target_action_id"].long().view(-1)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(batch)
                    loss = nn.functional.cross_entropy(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(batch)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            with torch.no_grad():
                correct = int((logits.argmax(-1) == target).sum().item())
            n = int(target.numel())
            window_loss += float(loss.item()) * n
            window_steps += n
            window_correct += correct
            running_loss += float(loss.item()) * n
            running_steps += n
            running_correct += correct

            if args.log_every and it % args.log_every == 0:
                now = time.time()
                window_secs = now - window_t0
                window_sps = window_steps / max(window_secs, 1e-6)
                avg_loss = running_loss / max(running_steps, 1)
                avg_top1 = running_correct / max(running_steps, 1)
                w_loss = window_loss / max(window_steps, 1)
                w_top1 = window_correct / max(window_steps, 1)
                # ETA: linear extrapolation across the remaining batches in this
                # epoch + remaining epochs. Excludes eval cost (small relative to
                # train).
                seen_batches = it
                rem_in_epoch = n_batches_train - seen_batches
                rem_total = rem_in_epoch + n_batches_train * (args.epochs - epoch)
                # Use cumulative epoch sps (more stable than window sps).
                cum_sps = running_steps / max(now - epoch_t0, 1e-6)
                eta = rem_total * batch_size / max(cum_sps, 1e-6)
                print(
                    f"  e{epoch}/{args.epochs} it {it:4d}/{n_batches_train}  "
                    f"loss(win/run)={w_loss:.4f}/{avg_loss:.4f}  "
                    f"top1(win/run)={w_top1:.4f}/{avg_top1:.4f}  "
                    f"sps={window_sps:7.0f}  ETA={_format_eta(eta)}"
                )
                window_t0 = now
                window_loss = 0.0
                window_steps = 0
                window_correct = 0

        train_loss = running_loss / max(running_steps, 1)
        train_top1 = running_correct / max(running_steps, 1)
        epoch_secs = time.time() - epoch_t0
        epoch_sps = running_steps / max(epoch_secs, 1e-6)
        print(
            f"epoch {epoch}/{args.epochs} TRAIN done  "
            f"loss={train_loss:.4f}  top1={train_top1:.4f}  "
            f"sps={epoch_sps:6.0f}  ({epoch_secs:.1f}s)"
        )

        val_metrics = _evaluate(model, val_ds, args.eval_batch_size, device, index_to_family,
                                label=f"val(e{epoch})", log_every=0)
        print(
            f"epoch {epoch}/{args.epochs} VAL  "
            f"loss={val_metrics['loss']:.4f}  "
            f"top1={val_metrics['top1']:.4f}  "
            f"top3={val_metrics['top3']:.4f}"
        )
        for fam, acc in sorted(val_metrics["per_family_top1"].items(),
                               key=lambda kv: -val_metrics["per_family_total"][kv[0]]):
            n = val_metrics["per_family_total"][fam]
            print(f"    val/{fam:25s}  top1={acc:.4f}  n={n}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_per_family_top1": val_metrics["per_family_top1"],
            "epoch_seconds": epoch_secs,
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = args.out / "best.pt"
            torch.save({
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "model_config": cfg.__dict__,
                "n_actions": n_actions,
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
    test_metrics = _evaluate(model, test_ds, args.eval_batch_size, device, index_to_family,
                             label="test")
    print(
        f"TEST  loss={test_metrics['loss']:.4f}  "
        f"top1={test_metrics['top1']:.4f}  top3={test_metrics['top3']:.4f}"
    )
    for fam, acc in sorted(test_metrics["per_family_top1"].items(),
                           key=lambda kv: -test_metrics["per_family_total"][kv[0]]):
        n = test_metrics["per_family_total"][fam]
        print(f"  test/{fam:25s}  top1={acc:.4f}  n={n}")

    report = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_params": n_params(model),
        "n_actions": n_actions,
        "device": str(device),
        "amp": use_amp,
        "elapsed_seconds": elapsed_total,
        "history": history,
        "test_metrics": test_metrics,
        "best_checkpoint": best_path.as_posix() if best_path else None,
    }
    (args.out / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {(args.out / 'training_report.json').as_posix()}")


if __name__ == "__main__":
    main()
