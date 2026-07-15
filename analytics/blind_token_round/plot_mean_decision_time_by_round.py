#!/usr/bin/env python3
"""Mean inter-event decision time vs round (1–35), split by BlindToken family."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import (
    BLIND_FAMILY_ORDER,
    collect_mean_dt_by_round_family,
    ieee_rc,
)

_FAMILY_LABEL = {"boss": "Boss", "small": "Small blind", "big": "Big blind"}
_STYLES = {
    "boss": {"color": "0.2", "linestyle": "-", "marker": "o"},
    "small": {"color": "0.45", "linestyle": "--", "marker": "s"},
    "big": {"color": "0.65", "linestyle": "-.", "marker": "^"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    ap.add_argument("--afk", type=float, default=90.0)
    ap.add_argument("--stake", type=int, default=268)
    ap.add_argument("--round-max", type=int, default=36)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/figures/blind_token/mean_decision_time_vs_round_stake268"
        ),
    )
    args = ap.parse_args()

    rmax = int(args.round_max)
    rounds = np.arange(1, rmax, dtype=int)
    by_rf = collect_mean_dt_by_round_family(
        args.parsed,
        args.extracted,
        float(args.afk),
        int(args.stake),
        rmax,
    )

    ieee_rc()
    fig, ax = plt.subplots(figsize=(3.75, 2.625), layout="constrained")

    for fam in BLIND_FAMILY_ORDER:
        ys = []
        for r in rounds:
            dlist = by_rf.get((int(r), fam), [])
            ys.append(float(np.mean(dlist)) if dlist else float("nan"))
        y = np.asarray(ys, dtype=np.float64)
        st = _STYLES[fam]
        ax.plot(
            rounds,
            y,
            label=_FAMILY_LABEL[fam],
            markerfacecolor="white",
            markeredgecolor=st["color"],
            markeredgewidth=0.6,
            clip_on=False,
            **st,
        )

    ax.set_xlabel(r"Round (OCR, current event)")
    ax.set_ylabel("Mean decision time (s)")
    step = max(1, (rmax - 1) // 8)
    ax.set_xticks(np.arange(1, rmax, step))
    ax.set_xlim(0.2, rmax - 0.2)
    all_means: list[float] = []
    for fam in BLIND_FAMILY_ORDER:
        for r in rounds:
            dlist = by_rf.get((int(r), fam), [])
            if dlist:
                all_means.append(float(np.mean(dlist)))
    if all_means:
        y0, y1 = min(all_means), max(all_means)
        pad = (y1 - y0) * 0.08 + 0.05
        ax.set_ylim(max(0.0, y0 - pad), y1 + pad)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)

    lines = ["round,family,mean_dt_s,n_dt_samples"]
    for r in rounds:
        for fam in BLIND_FAMILY_ORDER:
            dlist = by_rf.get((int(r), fam), [])
            mean_s = "" if not dlist else f"{float(np.mean(dlist)):.8f}"
            lines.append(f"{int(r)},{fam},{mean_s},{len(dlist)}")
    Path(str(args.out) + ".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}.pdf / .png / .csv")


if __name__ == "__main__":
    main()
