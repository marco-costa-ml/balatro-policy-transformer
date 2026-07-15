#!/usr/bin/env python3
"""IEEE-style: conditional loss at round r — ``lost_at(r) / reached(r)``, stake 268.

Uses ``data/parsed``. A run is kept iff:

1. First event with both OCR ``ante`` and ``round`` has ``ante==1``, ``round==0``.
2. **First** observed ``CurrentStake`` ``class_id`` (chronological) is **268**.

Over kept runs (OCR extrema):

- **Reached round r**: maximum observed ``round`` is greater than or equal to ``r``.
- **Lost at round r**: **last** observed ``round`` (scan backward) equals ``r``.

For each integer r from **1** to R-1 inclusive (default R=36, so r in 1..35),
plot ``lost / reached`` when ``reached > 0``. Axes only; no caption.
"""

from __future__ import annotations

import argparse
import json
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


def collect_run_stats(parsed_root: Path, stake_id: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for part in sorted(parsed_root.glob("video_id=*")):
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
            rows.append((mx, fin))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--stake", type=int, default=268)
    ap.add_argument(
        "--round-max",
        type=int,
        default=36,
        help="Use OCR rounds r in [1, R) (default 1..35).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/conditional_loss_by_round_stake268"),
    )
    args = ap.parse_args()

    rmax = int(args.round_max)
    stats = collect_run_stats(args.parsed, args.stake)
    n_runs = len(stats)

    rounds = np.arange(1, rmax, dtype=int)
    reached = np.zeros(len(rounds), dtype=int)
    lost_at = np.zeros(len(rounds), dtype=int)
    ratio = np.full(len(rounds), np.nan, dtype=np.float64)

    for j, r in enumerate(rounds):
        rch = sum(1 for mx, _ in stats if mx >= r)
        lost = sum(1 for _, fin in stats if fin == r)
        reached[j] = rch
        lost_at[j] = lost
        if rch > 0:
            ratio[j] = lost / rch

    _ieee_rc()
    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")

    mask = np.isfinite(ratio)
    ax.plot(
        rounds[mask],
        ratio[mask],
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        clip_on=False,
    )

    ax.set_xlabel(r"Round $r$ (OCR)")
    ax.set_ylabel(r"$\frac{N_{\mathrm{lose}}(r)}{N_{\mathrm{reach}}(r)}$")
    step = max(1, (rmax - 1) // 8)
    ax.set_xticks(np.arange(1, rmax, step))
    ax.set_xlim(0.2, rmax - 0.2)
    valid = ratio[mask]
    if valid.size:
        lo, hi = float(np.min(valid)), float(np.max(valid))
        pad = (hi - lo) * 0.08 + 0.02
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    else:
        ax.set_ylim(0.0, 1.0)

    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)

    csv_lines = ["round,reached,lost_at,conditional_loss"]
    for j in range(len(rounds)):
        rj = ratio[j]
        rs = "" if not np.isfinite(rj) else f"{rj:.6f}"
        csv_lines.append(f"{int(rounds[j])},{int(reached[j])},{int(lost_at[j])},{rs}")
    csv_path = Path(str(args.out) + ".csv")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"kept_runs_clean_start_first_stake_{args.stake}={n_runs}")
    print(f"Wrote {args.out}.pdf")
    print(f"Wrote {args.out}.png")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
