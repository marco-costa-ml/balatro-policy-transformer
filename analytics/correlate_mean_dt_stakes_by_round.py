#!/usr/bin/env python3
"""Pearson / Spearman correlation between mean decision time by round for two stakes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def pearson_spearman(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
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
        rx = _rank_average(x)
        ry = _rank_average(y)
        sr_mat = np.corrcoef(rx, ry)
        sr = float(sr_mat[0, 1]) if sr_mat.shape == (2, 2) else float("nan")
        return pr, float("nan"), sr, float("nan")


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


def pick_stake(objects: list | None) -> int | None:
    stakes = [
        o
        for o in (objects or [])
        if o.get("zone") == "CurrentStake" and o.get("class_id") is not None
    ]
    if not stakes:
        return None
    stakes.sort(key=lambda o: (o.get("position_in_zone", 0), o.get("slot_id", 0)))
    return int(stakes[-1]["class_id"])


def get_round(ev: dict) -> int | None:
    r = (ev.get("state") or {}).get("round")
    if r is None:
        return None
    try:
        return int(r)
    except (TypeError, ValueError):
        return None


def mean_dt_by_round(
    parsed_root: Path,
    extracted_root: Path,
    afk_s: float,
    stake_id: int,
    round_max_exclusive: int,
) -> dict[int, list[float]]:
    fps_map = load_fps_map(extracted_root)
    by_r: dict[int, list[float]] = defaultdict(list)

    for part in sorted(parsed_root.glob("video_id=*")):
        vid = part.name.split("=", 1)[1]
        fps = fps_map.get(vid) or 30.0
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            prev_frame: int | None = None
            prev_round: int | None = None
            for ev in evs:
                frame = ev.get("frame_idx")
                if frame is None:
                    continue
                frame = int(frame)
                curr_round = get_round(ev)
                stake = pick_stake(ev.get("objects"))
                if (
                    prev_frame is not None
                    and curr_round is not None
                    and prev_round is not None
                    and 1 <= curr_round < round_max_exclusive
                    and 1 <= prev_round < round_max_exclusive
                    and stake == stake_id
                ):
                    dt = (frame - prev_frame) / fps
                    if 0.0 <= dt <= afk_s:
                        by_r[curr_round].append(dt)
                prev_frame = frame
                prev_round = curr_round if curr_round is not None else prev_round

    return by_r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    ap.add_argument("--afk", type=float, default=90.0)
    ap.add_argument("--stake-a", type=int, default=268)
    ap.add_argument("--stake-b", type=int, default=273)
    ap.add_argument("--round-max", type=int, default=36, help="Rounds r in [1, R).")
    args = ap.parse_args()

    rmax = int(args.round_max)
    rounds = list(range(1, rmax))
    by_a = mean_dt_by_round(
        args.parsed, args.extracted, float(args.afk), int(args.stake_a), rmax
    )
    by_b = mean_dt_by_round(
        args.parsed, args.extracted, float(args.afk), int(args.stake_b), rmax
    )

    xs: list[float] = []
    ys: list[float] = []
    used_rounds: list[int] = []
    for r in rounds:
        la, lb = by_a.get(r, []), by_b.get(r, [])
        if not la or not lb:
            continue
        xs.append(float(np.mean(la)))
        ys.append(float(np.mean(lb)))
        used_rounds.append(r)

    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    n = len(xa)
    if n < 2:
        print(f"Need at least 2 rounds with both stakes; got {n}")
        return

    pr, pp, sr, sp = pearson_spearman(xa, ya)

    print(f"stakes={args.stake_a}_vs_{args.stake_b} rounds=[1,{rmax - 1}] pairwise_n={n}")
    print(f"rounds_used={used_rounds}")
    print(f"pearson_r={pr:.10g}, pearson_p={pp}")
    print(f"spearman_rho={sr:.10g}, spearman_p={sp}")


if __name__ == "__main__":
    main()
