#!/usr/bin/env python3
"""Conditional loss rate vs mean decision time (stake 268, rounds 1–35).

**Cohort** (same as ``plot_reached_over_lost_by_round_stake268.py``): runs with a
clean start (first ante+round is 1,0) and **first** observed ``CurrentStake``
``class_id`` 268.

For each OCR round r in **[1, ``round_max``)** (default 36 → r = 1..35):

- **Conditional loss rate**: ``p(r) = N_{\\mathrm{lose}}(r) / N_{\\mathrm{reach}}(r)``
  where ``reach`` = runs with ``max round ≥ r``, ``lose at r`` = runs whose **last**
  observed round is ``r``.
- **Mean decision time**: mean inter-event Δt for kept runs only, with the same
  rules as ``plot_mean_decision_time_vs_round_stake268.py`` (both interval
  endpoints with round in **[1, R)**, **current** stake 268 on the event,
  0 ≤ Δt ≤ ``afk_s``).

Plots ``p(r)`` vs mean Δt with **Wilson 95%** intervals on ``p(r)``.
Prints **Pearson** and **Spearman** correlation between (mean Δt , ``p``)
over rounds with ``reached > 0`` and finite mean Δt.

Writes PDF/PNG/CSV under ``artifacts/figures/`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _ieee_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.linewidth": 0.4,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


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


def _get_ante_round(ev: dict) -> tuple[int | None, int | None]:
    s = ev.get("state") or {}
    a, r = s.get("ante"), s.get("round")
    try:
        ai = int(a) if a is not None else None
    except (TypeError, ValueError):
        ai = None
    try:
        ri = int(r) if r is not None else None
    except (TypeError, ValueError):
        ri = None
    return ai, ri


def run_has_clean_start(events: list[dict]) -> bool:
    for ev in events:
        a, r = _get_ante_round(ev)
        if a is None or r is None:
            continue
        return a == 1 and r == 0
    return False


def first_stake_class(events: list[dict]) -> int | None:
    for ev in events:
        c = pick_stake(ev.get("objects"))
        if c is not None:
            return c
    return None


def run_max_round(events: list[dict]) -> int | None:
    m: int | None = None
    for ev in events:
        _, r = _get_ante_round(ev)
        if r is None:
            continue
        m = r if m is None else max(m, r)
    return m


def run_final_round(events: list[dict]) -> int | None:
    for ev in reversed(events):
        _, r = _get_ante_round(ev)
        if r is not None:
            return r
    return None


def get_round(ev: dict) -> int | None:
    _, r = _get_ante_round(ev)
    return r


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for Binomial(n, p); k successes."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    lo, hi = center - rad, center + rad
    return max(0.0, lo), min(1.0, hi)


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based); ties get mean rank."""
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
    """Returns (pearson_r, pearson_p, spearman_rho, spearman_p); p is nan if scipy missing."""
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


def collect_aligned(
    parsed_root: Path,
    extracted_root: Path,
    afk_s: float,
    stake_id: int,
    round_max_exclusive: int,
) -> tuple[dict[int, list[float]], list[tuple[int, int]], int]:
    """Return (dt_bucket by round, (max_round, final_round) per kept run, n_kept)."""
    fps_map = load_fps_map(extracted_root)
    by_r: dict[int, list[float]] = defaultdict(list)
    stats: list[tuple[int, int]] = []

    for part in sorted(parsed_root.glob("video_id=*")):
        vid = part.name.split("=", 1)[1]
        fps = fps_map.get(vid) or 30.0
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            if not evs or not run_has_clean_start(evs):
                continue
            if first_stake_class(evs) != stake_id:
                continue
            mx = run_max_round(evs)
            fin = run_final_round(evs)
            if mx is None or fin is None:
                continue
            stats.append((mx, fin))

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

    return by_r, stats, len(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    ap.add_argument("--afk", type=float, default=90.0, help="Drop Δt above this (seconds).")
    ap.add_argument("--stake", type=int, default=268)
    ap.add_argument("--round-max", type=int, default=36, help="Rounds r in [1, R).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/conditional_loss_vs_mean_dt_stake268"),
    )
    args = ap.parse_args()

    rmax = int(args.round_max)
    by_r, stats, n_kept = collect_aligned(
        args.parsed, args.extracted, float(args.afk), int(args.stake), rmax
    )

    rounds = np.arange(1, rmax, dtype=int)
    reached = np.zeros(len(rounds), dtype=int)
    lost_at = np.zeros(len(rounds), dtype=int)
    mean_dt = np.full(len(rounds), np.nan, dtype=np.float64)
    p_hat = np.full(len(rounds), np.nan, dtype=np.float64)
    p_lo = np.full(len(rounds), np.nan, dtype=np.float64)
    p_hi = np.full(len(rounds), np.nan, dtype=np.float64)

    for j, r in enumerate(rounds):
        rch = sum(1 for mx, _ in stats if mx >= r)
        lost = sum(1 for _, fin in stats if fin == r)
        reached[j] = rch
        lost_at[j] = lost
        dlist = by_r.get(int(r), [])
        if dlist:
            mean_dt[j] = float(np.mean(np.asarray(dlist, dtype=np.float64)))
        if rch > 0:
            p_hat[j] = lost / rch
            lo, hi = wilson_ci(lost, rch)
            p_lo[j], p_hi[j] = lo, hi

    mask_corr = (reached > 0) & np.isfinite(mean_dt) & np.isfinite(p_hat)
    xs = mean_dt[mask_corr]
    ys = p_hat[mask_corr]
    pr, pp, sr, sp = pearson_spearman(xs, ys)

    _ieee_rc()
    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")

    plot_mask = (reached > 0) & np.isfinite(mean_dt) & np.isfinite(p_hat)
    r_plot = rounds[plot_mask]
    x_plot = mean_dt[plot_mask]
    y_plot = p_hat[plot_mask]
    yerr_lo = y_plot - p_lo[plot_mask]
    yerr_hi = p_hi[plot_mask] - y_plot

    ax.errorbar(
        x_plot,
        y_plot,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        color="black",
        ecolor="0.45",
        elinewidth=0.6,
        capsize=2,
        markersize=4,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        clip_on=False,
        zorder=3,
    )

    ax.set_xlabel(r"Mean decision time $\Delta t$ (s)")
    ax.set_ylabel(r"$p(r)=\frac{N_{\mathrm{lose}}(r)}{N_{\mathrm{reach}}(r)}$")
    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    txt = (
        f"$n_{{\\mathrm{{runs}}}}={n_kept}$\n"
        f"$r_{{\\mathrm{{P}}}}={pr:.4f}$\n"
        f"$\\rho_{{\\mathrm{{S}}}}={sr:.4f}$"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)

    csv_lines = [
        "round,reached,lost_at,p_hat,wilson_lo,wilson_hi,mean_dt_s,n_dt_samples"
    ]
    for j, rr in enumerate(rounds):
        ndt = len(by_r.get(int(rr), []))
        mean_s = "" if not np.isfinite(mean_dt[j]) else f"{mean_dt[j]:.8f}"
        ph = p_hat[j]
        pl, phh = p_lo[j], p_hi[j]
        ps = "" if not np.isfinite(ph) else f"{ph:.8f}"
        pls = "" if not np.isfinite(pl) else f"{pl:.8f}"
        phs = "" if not np.isfinite(phh) else f"{phh:.8f}"
        csv_lines.append(
            f"{int(rr)},{int(reached[j])},{int(lost_at[j])},{ps},{pls},{phs},{mean_s},{ndt}"
        )
    csv_path = Path(str(args.out) + ".csv")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"kept_runs_clean_start_first_stake_{args.stake}={n_kept}")
    print(f"rounds_used_correlation={int(np.sum(mask_corr))}")
    print(f"pearson_r={pr:.6g}, pearson_p={pp if np.isfinite(pp) else 'n/a (install scipy)'}")
    print(
        f"spearman_rho={sr:.6g}, spearman_p={sp if np.isfinite(sp) else 'n/a (install scipy)'}"
    )
    print(f"Wrote {args.out}.pdf and {args.out}.png")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
