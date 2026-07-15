#!/usr/bin/env python3
"""Conditional loss vs round (1–35), split by BlindToken family; stake 268 cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import (
    BLIND_FAMILY_ORDER,
    collect_loss_tables,
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
    ap.add_argument("--stake", type=int, default=268)
    ap.add_argument("--round-max", type=int, default=36, help="Rounds r in [1, R).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/figures/blind_token/conditional_loss_by_round_stake268"
        ),
    )
    args = ap.parse_args()

    rmax = int(args.round_max)
    rounds = np.arange(1, rmax, dtype=int)
    reached, lost_at, n_kept = collect_loss_tables(
        args.parsed, int(args.stake), rmax
    )

    ieee_rc()
    fig, ax = plt.subplots(figsize=(3.75, 2.625), layout="constrained")

    for fam in BLIND_FAMILY_ORDER:
        ys = []
        for r in rounds:
            rc = reached.get((int(r), fam), 0)
            lc = lost_at.get((int(r), fam), 0)
            ys.append(lc / rc if rc > 0 else float("nan"))
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

    ax.set_xlabel(r"Round $r$ (OCR)")
    ax.set_ylabel(r"$\frac{N_{\mathrm{lose}}(r,f)}{N_{\mathrm{reach}}(r,f)}$")
    step = max(1, (rmax - 1) // 8)
    ax.set_xticks(np.arange(1, rmax, step))
    ax.set_xlim(0.2, rmax - 0.2)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)

    lines = ["round,family,reached,lost_at,conditional_loss"]
    for r in rounds:
        for fam in BLIND_FAMILY_ORDER:
            rc = reached.get((int(r), fam), 0)
            lc = lost_at.get((int(r), fam), 0)
            p = lc / rc if rc > 0 else float("nan")
            ps = "" if not np.isfinite(p) else f"{p:.8f}"
            lines.append(f"{int(r)},{fam},{rc},{lc},{ps}")
    Path(str(args.out) + ".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"kept_runs={n_kept}")
    print(f"Wrote {args.out}.pdf / .png / .csv")


if __name__ == "__main__":
    main()
