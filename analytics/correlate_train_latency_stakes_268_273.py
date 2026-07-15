#!/usr/bin/env python3
"""
Pearson / Spearman correlation for latency by OCR round between stakes 268 and 273.

Two modes:

- ``forward`` (default): mean **model forward time** (best.pt) per round on the
  **train** split, steps with ``target_action_id >= 0``, valid OCR ``round`` in
  [1, 35]. Correlation is between the two length-35 vectors of **mean ms/step**
  for stake 268 vs 273 (pairwise rounds where **both** have at least one timed
  step).

- ``human_dt``: mean **inter-frame Δt** (s) from granularized ``events`` aligned
  to tensorized rows (same filters). Uses ``data/extracted`` for fps; skips
  ``t==0``; caps Δt at ``--afk``.

Requires: ``artifacts/vocab.json``, ``artifacts/normalization.json``,
``artifacts/feature_config.json``, ``data/action_space_config.json`` for forward
mode (via dataset + checkpoint).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import BalatroStepDataset, load_split
from model import ModelConfig, PolicyTransformer


ROUND_IDX = 4  # tensorize.OCR_NUMERIC_KEYS index of "round"


def _rank_average(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        from scipy import stats as st

        pr, pp = st.pearsonr(x, y)
        sr, sp = st.spearmanr(x, y)
        return float(pr), float(pp), float(sr), float(sp)
    except Exception:
        pr_mat = np.corrcoef(x, y)
        pr = float(pr_mat[0, 1]) if pr_mat.shape == (2, 2) else float("nan")
        sr_mat = np.corrcoef(_rank_average(x), _rank_average(y))
        sr = float(sr_mat[0, 1]) if sr_mat.shape == (2, 2) else float("nan")
        return pr, float("nan"), sr, float("nan")


def decode_stake_vocab(vocab: dict[str, Any], idx: int) -> int | None:
    vals = vocab["vocabularies"]["stake_class_id"]["values"]
    if idx < 0 or idx >= len(vals):
        return None
    v = vals[idx]
    if v == "<PAD>":
        return None
    return int(v)


def denorm_round(ocr_round_norm: float, norm_payload: dict[str, Any]) -> int | None:
    st = norm_payload["stats"].get("ocr.round") or {}
    mn = float(st.get("min", 0.0))
    mx = float(st.get("max", 1.0))
    denom = max(mx - mn, 1.0)
    x = float(ocr_round_norm) * denom + mn
    if not math.isfinite(x):
        return None
    return int(round(x))


def run_forward_mode(
    *,
    tensorized_root: Path,
    splits_path: Path,
    split_name: str,
    checkpoint: Path,
    norm_payload: dict[str, Any],
    vocab_payload: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> None:
    vids = load_split(splits_path, split_name)
    ds = BalatroStepDataset(tensorized_root, vids, include_unresolved=False, device=device)
    if len(ds) == 0:
        raise SystemExit("empty dataset")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = PolicyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tens = ds.all_tensors()
    valid = ds.valid_indices()
    flat = valid.detach().cpu().numpy()
    flat_to_ds = {int(f): i for i, f in enumerate(flat)}

    taid = tens["target_action_id"].detach().cpu().numpy().reshape(-1)
    tstk = tens["stake_class_id"].detach().cpu().numpy().reshape(-1)
    ocrn = tens["ocr_numeric"][:, ROUND_IDX].detach().cpu().numpy().reshape(-1)
    ocrv = tens["ocr_valid"][:, ROUND_IDX].detach().cpu().numpy().reshape(-1)

    cand: list[int] = []
    for f in flat:
        fi = int(f)
        if int(taid[fi]) < 0:
            continue
        if not bool(ocrv[fi]):
            continue
        sid = decode_stake_vocab(vocab_payload, int(tstk[fi]))
        if sid not in (268, 273):
            continue
        ri = denorm_round(float(ocrn[fi]), norm_payload)
        if ri is None or not (1 <= ri <= 35):
            continue
        cand.append(fi)

    sum_ms: dict[tuple[int, int], float] = defaultdict(float)
    cnt: dict[tuple[int, int], int] = defaultdict(int)

    use_cuda = device.type == "cuda"

    def time_batch(batch: dict[str, torch.Tensor]) -> float:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        if use_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in cand:
        ds_i = flat_to_ds[fi]
        sid = decode_stake_vocab(vocab_payload, int(tstk[fi]))
        ri = denorm_round(float(ocrn[fi]), norm_payload)
        assert sid is not None and ri is not None
        buckets[(ri, int(sid))].append(ds_i)

    for (_r, _sid), ds_list in sorted(buckets.items()):
        for i in range(0, len(ds_list), batch_size):
            chunk = torch.tensor(ds_list[i : i + batch_size], dtype=torch.long, device=device)
            batch = ds.gather_batch(chunk)
            ms = time_batch(batch)
            sum_ms[(_r, _sid)] += ms
            cnt[(_r, _sid)] += int(chunk.numel())

    mean_ms_268: list[float] = []
    mean_ms_273: list[float] = []
    rounds_used: list[int] = []
    for r in range(1, 36):
        c268 = cnt.get((r, 268), 0)
        c273 = cnt.get((r, 273), 0)
        if c268 == 0 or c273 == 0:
            continue
        rounds_used.append(r)
        mean_ms_268.append(sum_ms[(r, 268)] / c268)
        mean_ms_273.append(sum_ms[(r, 273)] / c273)

    xa = np.asarray(mean_ms_268, dtype=np.float64)
    ya = np.asarray(mean_ms_273, dtype=np.float64)
    pr, pp, sr, sp = pearson_spearman(xa, ya)

    print(f"mode=forward checkpoint={checkpoint}")
    print(f"split={split_name} pairwise_rounds_n={len(rounds_used)} rounds={rounds_used}")
    print(f"pearson_r={pr:.10g}, pearson_p={pp}")
    print(f"spearman_rho={sr:.10g}, spearman_p={sp}")


def load_fps_map(extracted_root: Path) -> dict[str, float]:
    m: dict[str, float] = {}
    for part in sorted(extracted_root.glob("video_id=*")):
        if not part.is_dir():
            continue
        vid = part.name.split("=", 1)[1]
        en = list(part.glob("*_enriched.json"))
        if not en:
            continue
        try:
            d = json.loads(en[0].read_text(encoding="utf-8"))
            m[vid] = float(d.get("fps") or 30)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            m[vid] = 30.0
    return m


def run_human_dt_mode(
    *,
    tensorized_root: Path,
    granularized_root: Path,
    extracted_root: Path,
    splits_path: Path,
    split_name: str,
    vocab_payload: dict[str, Any],
    norm_payload: dict[str, Any],
    afk_s: float,
) -> None:
    vids = set(load_split(splits_path, split_name))
    fps_map = load_fps_map(extracted_root)
    by_r_stake: dict[tuple[int, int], list[float]] = defaultdict(list)

    for part in sorted(granularized_root.glob("video_id=*")):
        if not part.is_dir() or not part.name.startswith("video_id="):
            continue
        vid = part.name.split("=", 1)[1]
        if vid not in vids:
            continue
        fps = fps_map.get(vid) or 30.0
        npart = tensorized_root / part.name
        if not npart.is_dir():
            continue
        for gpath in sorted(part.glob("run_*.json")):
            npath = npart / gpath.with_suffix(".npz").name
            if not npath.exists():
                continue
            run = json.loads(gpath.read_text(encoding="utf-8"))
            steps = run.get("events") or []
            with np.load(npath) as z:
                tgt = np.array(z["target_action_id"]).reshape(-1)
                stk = np.array(z["stake_class_id"]).reshape(-1)
                ocr = np.array(z["ocr_numeric"])[:, ROUND_IDX]
                ocok = np.array(z["ocr_valid"])[:, ROUND_IDX]
            if len(steps) != len(tgt):
                continue
            for t in range(1, len(steps)):
                if int(tgt[t]) < 0:
                    continue
                if not bool(ocok[t]):
                    continue
                sid = decode_stake_vocab(vocab_payload, int(stk[t]))
                if sid not in (268, 273):
                    continue
                ri = denorm_round(float(ocr[t]), norm_payload)
                if ri is None or not (1 <= ri <= 35):
                    continue
                f0 = steps[t - 1].get("frame_idx")
                f1 = steps[t].get("frame_idx")
                if f0 is None or f1 is None:
                    continue
                dt = (int(f1) - int(f0)) / fps
                if not (0.0 <= dt <= afk_s):
                    continue
                by_r_stake[(ri, int(sid))].append(float(dt))

    mean_268: list[float] = []
    mean_273: list[float] = []
    rounds_used: list[int] = []
    for r in range(1, 36):
        a268 = by_r_stake.get((r, 268), [])
        a273 = by_r_stake.get((r, 273), [])
        if not a268 or not a273:
            continue
        rounds_used.append(r)
        mean_268.append(float(np.mean(a268)))
        mean_273.append(float(np.mean(a273)))

    xa = np.asarray(mean_268, dtype=np.float64)
    ya = np.asarray(mean_273, dtype=np.float64)
    pr, pp, sr, sp = pearson_spearman(xa, ya)
    print(f"mode=human_dt split={split_name} afk_cap={afk_s}")
    print(f"pairwise_rounds_n={len(rounds_used)} rounds={rounds_used}")
    print(f"pearson_r={pr:.10g}, pearson_p={pp}")
    print(f"spearman_rho={sr:.10g}, spearman_p={sp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("forward", "human_dt"), default="forward")
    ap.add_argument("--tensorized", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--granularized", type=Path, default=Path("data/granularized"))
    ap.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/best.pt"))
    ap.add_argument("--normalization", type=Path, default=Path("artifacts/normalization.json"))
    ap.add_argument("--vocab", type=Path, default=Path("artifacts/vocab.json"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--afk", type=float, default=90.0)
    args = ap.parse_args()

    norm_payload = json.loads(args.normalization.read_text(encoding="utf-8"))
    vocab_payload = json.loads(args.vocab.read_text(encoding="utf-8"))

    if args.mode == "forward":
        want_cuda = args.device == "cuda" and torch.cuda.is_available()
        dev = torch.device("cuda" if want_cuda else "cpu")
        run_forward_mode(
            tensorized_root=args.tensorized,
            splits_path=args.splits,
            split_name=args.split,
            checkpoint=args.checkpoint,
            norm_payload=norm_payload,
            vocab_payload=vocab_payload,
            batch_size=int(args.batch_size),
            device=dev,
        )
    else:
        run_human_dt_mode(
            tensorized_root=args.tensorized,
            granularized_root=args.granularized,
            extracted_root=args.extracted,
            splits_path=args.splits,
            split_name=args.split,
            vocab_payload=vocab_payload,
            norm_payload=norm_payload,
            afk_s=float(args.afk),
        )


if __name__ == "__main__":
    main()
