#!/usr/bin/env python3
"""Per blind family: conditional loss vs mean Δt (rounds 1–35), Wilson 95% on p."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import (
    BLIND_FAMILY_ORDER,
    collect_loss_tables,
    collect_mean_dt_by_round_family,
    ieee_rc,
    pearson_spearman,
    wilson_ci,
)

_FAMILY_LABEL = {"boss": "Boss", "small": "Small blind", "big": "Big blind"}
_STYLES = {
    "boss": {"color": "0.2", "marker": "o"},
    "small": {"color": "0.45", "marker": "s"},
    "big": {"color": "0.65", "marker": "^"},
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
            "artifacts/figures/blind_token/conditional_loss_vs_mean_dt_stake268"
        ),
    )
    args = ap.parse_args()

    rmax = int(args.round_max)
    rounds = np.arange(1, rmax, dtype=int)
    reached, lost_at, n_kept = collect_loss_tables(
        args.parsed, int(args.stake), rmax
    )
    by_rf = collect_mean_dt_by_round_family(
        args.parsed,
        args.extracted,
        float(args.afk),
        int(args.stake),
        rmax,
    )

    ieee_rc()
    fig, axes = plt.subplots(
        1, 3, figsize=(6.9, 2.5), layout="constrained", sharex=False, sharey=True
    )

    csv_lines = [
        "round,family,reached,lost_at,p_hat,wilson_lo,wilson_hi,mean_dt_s,n_dt_samples"
    ]

    for ax, fam in zip(axes, BLIND_FAMILY_ORDER):
        xs: list[float] = []
        ys: list[float] = []
        ylo: list[float] = []
        yhi: list[float] = []
        for r in rounds:
            rch = reached.get((int(r), fam), 0)
            lc = lost_at.get((int(r), fam), 0)
            dlist = by_rf.get((int(r), fam), [])
            ph = lc / rch if rch > 0 else float("nan")
            csv_lo, csv_hi = wilson_ci(lc, rch) if rch > 0 else (float("nan"), float("nan"))
            mean_s = "" if not dlist else f"{float(np.mean(dlist)):.8f}"
            ps = "" if not np.isfinite(ph) else f"{ph:.8f}"
            pls = "" if not np.isfinite(csv_lo) else f"{csv_lo:.8f}"
            phs = "" if not np.isfinite(csv_hi) else f"{csv_hi:.8f}"
            csv_lines.append(
                f"{int(r)},{fam},{rch},{lc},{ps},{pls},{phs},{mean_s},{len(dlist)}"
            )
            if rch > 0 and dlist:
                p_hat = lc / rch
                lo, hi = wilson_ci(lc, rch)
                mdt = float(np.mean(dlist))
                xs.append(mdt)
                ys.append(p_hat)
                ylo.append(p_hat - lo)
                yhi.append(hi - p_hat)

        st = _STYLES[fam]
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=[ylo, yhi],
                fmt=st["marker"],
                color=st["color"],
                ecolor="0.45",
                elinewidth=0.6,
                capsize=2,
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=st["color"],
                markeredgewidth=0.6,
                clip_on=False,
            )
        npts = len(xs)
        if npts >= 2:
            pr, pp, sr, sp = pearson_spearman(np.asarray(xs), np.asarray(ys))
            ax.set_title(
                f"{_FAMILY_LABEL[fam]}\n"
                rf"$r_P={pr:.3f}$, $\rho_S={sr:.3f}$",
                fontsize=8,
            )
            print(
                f"{fam}: n_rounds={npts} pearson_r={pr:.6g} p={pp} "
                f"spearman_rho={sr:.6g} p={sp}"
            )
        else:
            ax.set_title(_FAMILY_LABEL[fam], fontsize=8)
            print(f"{fam}: not enough points for correlation (n={npts})")

        ax.set_xlabel(r"Mean $\Delta t$ (s)")
        if fam == "boss":
            ax.set_ylabel(r"$N_{\mathrm{lose}}/N_{\mathrm{reach}}$")
        ax.grid(True, which="major", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle(f"Stake {args.stake}, $n_{{\\mathrm{{runs}}}}={n_kept}$", fontsize=8, y=1.02)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png")
    plt.close(fig)

    Path(str(args.out) + ".csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}.pdf / .png / .csv")


if __name__ == "__main__":
    main()
