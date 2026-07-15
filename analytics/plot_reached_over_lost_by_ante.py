#!/usr/bin/env python3
"""IEEE-style figure: conditional loss at ante a — ``lost_at(a) / reached(a)``, for a < 12.

Uses ``data/parsed`` runs. A run is kept only if the **first** event with both OCR
``state.ante`` and ``state.round`` reads ``ante == 1`` and ``round == 0``
(normal shop-after-death style start).

Definitions (OCR can flicker; we use extrema over the run):

- **Reached ante a**: run's maximum observed ``ante`` is **>= a**.
- **Lost at ante a**: run's **last** observed ``ante`` (last event with non-null
  ante, scanning backward) **equals a**.

For each ante ``a`` in ``1 .. 11`` (strictly less than 12), plot::

    lost_at(a) / reached(a)

when ``reached(a) > 0`` (including zeros when lost_at is 0). Axes only; no footnote by default.
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


def run_max_ante(events: list[dict]) -> int | None:
    m: int | None = None
    for ev in events:
        a, _ = _get_ante_round(ev)
        if a is None:
            continue
        m = a if m is None else max(m, a)
    return m


def run_final_ante(events: list[dict]) -> int | None:
    for ev in reversed(events):
        a, _ = _get_ante_round(ev)
        if a is not None:
            return a
    return None


def collect_run_stats(parsed_root: Path) -> list[tuple[int | None, int | None]]:
    """Return list of (max_ante, final_ante) per kept run."""
    rows: list[tuple[int | None, int | None]] = []
    for part in sorted(parsed_root.glob("video_id=*")):
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            if not evs or not run_has_clean_start(evs):
                continue
            mx = run_max_ante(evs)
            fin = run_final_ante(evs)
            if mx is None or fin is None:
                continue
            rows.append((mx, fin))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/conditional_loss_by_ante"),
        help="Output path without extension; writes .pdf, .png, and .csv.",
    )
    args = ap.parse_args()

    stats = collect_run_stats(args.parsed)
    n_runs = len(stats)

    # Ante a with a < 12 => a = 1 .. 11
    antes = np.arange(1, 12, dtype=int)
    reached = np.zeros(len(antes), dtype=int)
    lost_at = np.zeros(len(antes), dtype=int)
    ratio = np.full(len(antes), np.nan, dtype=np.float64)

    for j, a in enumerate(antes):
        rch = sum(1 for mx, _ in stats if mx >= a)
        lost = sum(1 for _, fin in stats if fin == a)
        reached[j] = rch
        lost_at[j] = lost
        if rch > 0:
            ratio[j] = lost / rch

    _ieee_rc()
    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")

    mask = np.isfinite(ratio)
    ax.plot(
        antes[mask],
        ratio[mask],
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        clip_on=False,
    )

    ax.set_xlabel(r"Ante $a$ (OCR)")
    ax.set_ylabel(r"$\frac{N_{\mathrm{lose}}(a)}{N_{\mathrm{reach}}(a)}$")
    ax.set_xticks(antes)
    ax.set_xlim(0.6, 11.4)
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

    csv_lines = ["ante,reached,lost_at,conditional_loss"]
    for j in range(len(antes)):
        rj = ratio[j]
        rs = "" if not np.isfinite(rj) else f"{rj:.6f}"
        csv_lines.append(f"{int(antes[j])},{int(reached[j])},{int(lost_at[j])},{rs}")
    csv_path = Path(str(args.out) + ".csv")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"kept_runs_first_event_ante1_round0={n_runs}")
    print(f"Wrote {args.out}.pdf")
    print(f"Wrote {args.out}.png")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
