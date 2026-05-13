#!/usr/bin/env python3
"""Sweep tensorized steps on pack pages: model SkipPack tendency vs supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import BalatroStepDataset, load_split
from model import ModelConfig, PolicyTransformer, load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("artifacts/checkpoints/best.pt"),
    )
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else (root / args.ckpt)

    with open(root / "data/action_map.json", encoding="utf-8") as f:
        action_map = json.load(f)
    skip_id = action_map["family_offsets"]["SkipPack"]
    sel_start = action_map["family_offsets"]["SelectPackItem"]
    sel_count = action_map["family_sizes"]["SelectPackItem"]
    labels = action_map["index_to_label"]
    n_actions = int(action_map["n_actions"])

    ckpt = torch.load(ckpt_path.resolve(), map_location=DEVICE, weights_only=False)
    cfg_in = ckpt.get("model_config")
    if cfg_in is not None:
        cfg = ModelConfig(**{k: v for k, v in cfg_in.items() if k in ModelConfig.__dataclass_fields__})
        assert cfg.n_actions == n_actions, "action_map / checkpoint mismatch"
    else:
        cfg = load_model_config(n_actions, root / "artifacts/vocab.json")
    model = PolicyTransformer(cfg).to(DEVICE).eval()
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise KeyError("checkpoint missing model_state_dict")
    model.load_state_dict(state)

    ds = BalatroStepDataset(
        tensorized_root=root / "data/tensorized",
        split_videos=load_split(root / "artifacts/splits.json", args.split),
        device=DEVICE,
    )
    tensors = ds.all_tensors()
    valid_flat = ds.valid_indices()

    page = tensors["page_id"].long().squeeze(-1)
    mask_b = tensors["action_mask"].bool()
    tgt = tensors["target_action_id"].long().squeeze(-1)

    page_names = [
        ("In_JokerStandardPlanet_Pack", 5),
        ("In_TarotSpectral_Pack", 7),
    ]

    for page_name, page_id in page_names:
        m = (
            (page == page_id)
            & mask_b[:, skip_id]
            & (mask_b[:, sel_start : sel_start + sel_count].sum(dim=-1) > 0)
        )
        # Corpus rows that are valid (Resolvable target) and match pack-legality predicate.
        m_valid = m[valid_flat.long()]
        rel = torch.nonzero(m_valid, as_tuple=False).squeeze(-1)
        n = int(rel.numel())
        if n == 0:
            print(f"=== {page_name} === no qualifying rows")
            continue

        p_skip_chunks: list[torch.Tensor] = []
        pred_skip_chunks: list[torch.Tensor] = []
        pred_when_tgt_sel: list[bool] = []
        pred_when_tgt_skip: list[bool] = []
        ranked: list[tuple[float, int, str, str]] = []

        for base in range(0, len(rel), args.batch_size):
            sample_idx = rel[base : base + args.batch_size]
            batch = ds.gather_batch(sample_idx)
            with torch.no_grad():
                logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            flats = valid_flat.index_select(0, sample_idx)
            t_batch = tgt[flats]

            p_skip_chunks.append(probs[:, skip_id].detach().cpu())
            pred_skip_chunks.append((pred == skip_id).detach().cpu())

            for i in range(pred.shape[0]):
                ti = int(t_batch[i].item())
                pi = int(pred[i].item())
                is_sel = sel_start <= ti < sel_start + sel_count
                if is_sel:
                    pred_when_tgt_sel.append(pi == skip_id)
                if ti == skip_id:
                    pred_when_tgt_skip.append(pi == skip_id)
                ranked.append(
                    (
                        float(probs[i, skip_id].item()),
                        int(flats[i].item()),
                        labels[ti],
                        labels[pi],
                    ),
                )

        p_skip_np = torch.cat(p_skip_chunks, dim=0).numpy()
        pred_skip_np = torch.cat(pred_skip_chunks, dim=0).numpy().astype(np.float64)

        tgt_slice = tgt[valid_flat.index_select(0, rel)].cpu().numpy().ravel()
        tgt_skip = (tgt_slice == skip_id).mean()
        tgt_sel = ((tgt_slice >= sel_start) & (tgt_slice < sel_start + sel_count)).mean()

        frac_pred_skip = float(pred_skip_np.mean())
        cond_sel = (
            float(np.mean(pred_when_tgt_sel)) if pred_when_tgt_sel else float("nan")
        )
        cond_skip = (
            float(np.mean(pred_when_tgt_skip)) if pred_when_tgt_skip else float("nan")
        )

        ranked.sort(key=lambda r: r[0], reverse=True)

        print(f"=== {page_name} | {args.split} split ===")
        print(
            "rows (SkipPack legal and some SelectPackItem legal):",
            n,
        )
        print("label frac SkipPack:", round(float(tgt_skip), 4))
        print("label frac SelectPackItem_*:", round(float(tgt_sel), 4))
        print("pred argmax frac SkipPack:", round(frac_pred_skip, 4))
        print(
            "conditional: pred_skip | label SelectPackItem_*:",
            round(cond_sel, 4),
        )
        if pred_when_tgt_skip:
            print(
                "conditional: pred_skip | label SkipPack:",
                round(cond_skip, 4),
            )
        print(
            "P(skip) mean / median:",
            round(float(p_skip_np.mean()), 4),
            "/",
            round(float(np.median(p_skip_np)), 4),
        )
        print("examples (highest P(skip), target vs pred argmax):")
        for rank, row in enumerate(ranked[:15], start=1):
            p_skip, fid, tg, predl = row
            print(f"  #{rank}: P(skip)={p_skip:.4f} flat_step={fid} target={tg} pred={predl}")


if __name__ == "__main__":
    main()
