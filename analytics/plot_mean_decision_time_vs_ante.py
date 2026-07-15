#!/usr/bin/env python3
"""IEEE-style figure: mean parsed inter-event decision time vs. ante (1–12).

Data: ``data/parsed``; Δt = consecutive ``frame_idx`` gap / per-video ``fps``
from ``data/extracted/.../*_enriched.json``. Excludes intervals when either
endpoint has ``state.ante`` > 12 (missing ante allowed) or Δt > 90 s.
Intervals are bucketed by the **current** event ante when it is in 1…12.
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
    """Matplotlib rcParams loosely aligned with IEEE single-column figures."""
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


def get_ante(ev: dict) -> int | None:
    a = (ev.get("state") or {}).get("ante")
    if a is None:
        return None
    try:
        return int(a)
    except (TypeError, ValueError):
        return None


def ante_ok(a: int | None) -> bool:
    if a is None:
        return True
    return a <= 12


def mean_dt_by_ante(parsed_root: Path, extracted_root: Path, afk_s: float) -> dict[int, list[float]]:
    fps_map = load_fps_map(extracted_root)
    by_ante: dict[int, list[float]] = defaultdict(list)

    for part in sorted(parsed_root.glob("video_id=*")):
        vid = part.name.split("=", 1)[1]
        fps = fps_map.get(vid) or 30.0
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            prev_frame: int | None = None
            prev_ante: int | None = None
            for ev in evs:
                frame = ev.get("frame_idx")
                if frame is None:
                    continue
                frame = int(frame)
                curr_ante = get_ante(ev)
                if prev_frame is not None and curr_ante is not None and 1 <= curr_ante <= 12:
                    if ante_ok(prev_ante) and ante_ok(curr_ante):
                        dt = (frame - prev_frame) / fps
                        if 0.0 <= dt <= afk_s:
                            by_ante[curr_ante].append(dt)
                prev_frame = frame
                prev_ante = curr_ante

    return by_ante


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    ap.add_argument("--afk", type=float, default=90.0, help="Max Δt in seconds.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/mean_decision_time_vs_ante"),
        help="Output path without extension; writes .pdf and .png.",
    )
    ap.add_argument(
        "--caption",
        action="store_true",
        help="If set, add a small footnote with sample counts and filters.",
    )
    args = ap.parse_args()

    by_ante = mean_dt_by_ante(args.parsed, args.extracted, args.afk)
    antes = np.arange(1, 13, dtype=int)
    means = np.array(
        [np.mean(by_ante[int(a)]) if by_ante[int(a)] else np.nan for a in antes],
        dtype=np.float64,
    )
    ns = np.array([len(by_ante[int(a)]) for a in antes], dtype=int)

    _ieee_rc()
    # IEEE single-column width ≈ 3.5 in; height chosen for 4:3 aspect.
    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")

    ax.plot(
        antes,
        means,
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        clip_on=False,
    )

    ax.set_xlabel("Ante (OCR, current event)")
    ax.set_ylabel("Mean decision time (s)")
    ax.set_xticks(antes)
    ax.set_xlim(0.6, 12.4)
    if np.all(np.isnan(means)):
        ax.set_ylim(0.0, 1.0)
    else:
        y0, y1 = float(np.nanmin(means) * 0.98), float(np.nanmax(means) * 1.02)
        ax.set_ylim(y0, y1)

    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    if args.caption:
        note = (
            f"$n$ per ante: min {int(ns.min())}, max {int(ns.max())}. "
            f"Exclude $\\Delta t > {args.afk:g}$\\,s and intervals with ante $>12$ on either end."
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
