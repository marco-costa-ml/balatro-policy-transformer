#!/usr/bin/env python3
"""IEEE-style figure: mean parsed inter-event decision time vs. round (1–35).

Like ``plot_mean_decision_time_vs_ante.py`` but:

- X-axis: OCR ``state.round`` on the **current** event; only **1 ≤ round < 36**.
- **Current** event must have ``CurrentStake`` class_id **268** (same tie-break as
  elsewhere: last stake object by ``position_in_zone``, ``slot_id``).
- Interval counted only if **both** endpoints have round in **[1, 36)** (or missing
  round treated as not counted — we require both ``prev_round`` and
  ``curr_round`` integers in range).
- Still: Δt from consecutive ``frame_idx`` / ``fps``, drop Δt > **90** s (AFK).
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--stake", type=int, default=268)
    ap.add_argument("--round-max", type=int, default=36, help="Use rounds [1, R).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/mean_decision_time_vs_round_stake268"),
    )
    ap.add_argument("--caption", action="store_true")
    args = ap.parse_args()

    rmax = int(args.round_max)
    by_r = mean_dt_by_round(args.parsed, args.extracted, args.afk, args.stake, rmax)
    rounds = np.arange(1, rmax, dtype=int)
    means = np.array(
        [np.mean(by_r[int(r)]) if by_r[int(r)] else np.nan for r in rounds],
        dtype=np.float64,
    )
    ns = np.array([len(by_r[int(r)]) for r in rounds], dtype=int)

    _ieee_rc()
    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")

    mask = np.isfinite(means)
    ax.plot(
        rounds[mask],
        means[mask],
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        clip_on=False,
    )

    ax.set_xlabel(r"Round (OCR, current event)")
    ax.set_ylabel("Mean decision time (s)")
    step = max(1, (rmax - 1) // 8)
    ax.set_xticks(np.arange(1, rmax, step))
    ax.set_xlim(0.2, rmax - 0.2)
    if np.any(mask):
        y0, y1 = float(np.nanmin(means)), float(np.nanmax(means))
        pad = (y1 - y0) * 0.08 + 0.05
        ax.set_ylim(max(0.0, y0 - pad), y1 + pad)
    else:
        ax.set_ylim(0.0, 1.0)

    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    if args.caption:
        note = (
            f"Stake {args.stake}, rounds $[1,{rmax})$, "
            f"$n$ per round min {int(ns[ns>0].min()) if np.any(ns>0) else 0} "
            f"max {int(ns.max())}, AFK $>${args.afk:g}$\\,$s."
        )
        fig.text(0.0, -0.02, note, ha="left", va="top", fontsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)
    print(f"Wrote {args.out}.pdf")
    print(f"Wrote {args.out}.png")


if __name__ == "__main__":
    main()
