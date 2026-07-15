#!/usr/bin/env python3
"""Grouped horizontal bar chart: raw parsed events vs granularized steps by action family."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from granularize import parse_event_action
from state_reducer import parse_base_action


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
            "savefig.dpi": 300,
            # Avoid "tight" here: log-scale barh with clip_on=False can blow up bbox width.
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.02,
        }
    )


def _short_label(s: str, max_len: int = 28) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _ieee_figure_width_inches(column: str) -> float:
    """Typical IEEE text widths in inches (approx.)."""
    return 7.16 if column == "double" else 3.5


def count_parsed(parsed_root: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for part in sorted(parsed_root.glob("video_id=*")):
        if not part.is_dir():
            continue
        for run_path in sorted(part.glob("run_*.json")):
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for ev in data.get("events") or []:
                if not isinstance(ev, dict):
                    continue
                base, _ = parse_event_action(ev)
                fam = parse_base_action(base) if base else "(none)"
                if not fam:
                    fam = "(none)"
                c[fam] += 1
    return c


def count_granular(granular_root: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for part in sorted(granular_root.glob("video_id=*")):
        if not part.is_dir() or not part.name.startswith("video_id="):
            continue
        for run_path in sorted(part.glob("run_*.json")):
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for step in data.get("events") or []:
                if not isinstance(step, dict):
                    continue
                a = step.get("action")
                if not a:
                    continue
                fam = parse_base_action(str(a))
                if not fam:
                    fam = "(none)"
                c[fam] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--granularized", type=Path, default=Path("data/granularized"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/raw_vs_granular_action_counts"),
    )
    ap.add_argument(
        "--log-ratio",
        type=float,
        default=50.0,
        help="Use log x if max/min among positive counts exceeds this.",
    )
    ap.add_argument(
        "--ieee-column",
        choices=("single", "double"),
        default="single",
        help="Target width: single (~3.5in) or double (~7.16in) IEEE column.",
    )
    args = ap.parse_args()

    raw_c = count_parsed(args.parsed)
    gran_c = count_granular(args.granularized)
    if not raw_c and not gran_c:
        raise SystemExit("No events found under --parsed and --granularized.")

    families = sorted(
        set(raw_c.keys()) | set(gran_c.keys()),
        key=lambda f: max(raw_c.get(f, 0), gran_c.get(f, 0)),
        reverse=True,
    )
    raw_v = np.array([raw_c.get(f, 0) for f in families], dtype=np.float64)
    gran_v = np.array([gran_c.get(f, 0) for f in families], dtype=np.float64)

    pos_all = np.concatenate([raw_v[raw_v > 0], gran_v[gran_v > 0]])
    if pos_all.size:
        ratio = float(np.max(pos_all)) / max(float(np.min(pos_all)), 1.0)
    else:
        ratio = 1.0
    use_log = ratio > float(args.log_ratio)

    _ieee_rc()
    n = len(families)
    fig_w = _ieee_figure_width_inches(args.ieee_column)
    # Readable row height for horizontal bars; cap height for very long lists.
    fig_h = min(9.0, max(2.8, 0.16 * n + 0.9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = np.arange(n, dtype=float)
    h = 0.36
    x_hi = float(max(raw_v.max(), gran_v.max(), 1.0))
    if use_log:
        # Log + barh: bars must not start at x=0 (invalid); use explicit left edge.
        x0 = 1.0
        br = np.maximum(raw_v, x0 * 1.01)
        bg = np.maximum(gran_v, x0 * 1.01)
        ax.barh(
            y - h / 2,
            br - x0,
            left=x0,
            height=h,
            label="Raw events",
            facecolor="white",
            edgecolor="0.15",
            linewidth=0.6,
            clip_on=True,
        )
        ax.barh(
            y + h / 2,
            bg - x0,
            left=x0,
            height=h,
            label="Granularized",
            facecolor="0.78",
            edgecolor="0.15",
            linewidth=0.6,
            clip_on=True,
        )
        ax.set_xscale("log")
        ax.set_xlim(x0, x_hi * 1.35)
    else:
        ax.barh(
            y - h / 2,
            raw_v,
            height=h,
            label="Raw events",
            facecolor="white",
            edgecolor="0.15",
            linewidth=0.6,
            clip_on=True,
        )
        ax.barh(
            y + h / 2,
            gran_v,
            height=h,
            label="Granularized",
            facecolor="0.78",
            edgecolor="0.15",
            linewidth=0.6,
            clip_on=True,
        )
        ax.set_xlim(0.0, x_hi * 1.05)

    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(f) for f in families])
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", linestyle="--", linewidth=mpl.rcParams["grid.linewidth"], alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.34, right=0.98, top=0.98, bottom=0.10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f"{args.out}.pdf",
        dpi=300,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        f"{args.out}.png",
        dpi=300,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

    csv_path = Path(str(args.out) + ".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "raw_events", "granularized_events"])
        for i, fam in enumerate(families):
            w.writerow([fam, int(raw_v[i]), int(gran_v[i])])

    print(f"x_scale={'log' if use_log else 'linear'} (max/min among positive ~ {ratio:.1f})")
    print(f"Wrote {args.out}.pdf, {args.out}.png, {csv_path}")


if __name__ == "__main__":
    main()
