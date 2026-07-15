#!/usr/bin/env python3
"""
Emit two CSV tables (by action family):

1. Non-decomposed: parsed run events (post ``StartNewRun`` split) — one row per
   frame/action in ``data/parsed``, i.e. same cardinality as granularizer input.
2. Decomposed: granularized micro-steps in ``data/granularized``.

Families use ``parse_base_action`` on the primary action label, matching the
raw-vs-granular chart logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from granularize import parse_event_action
from state_reducer import parse_base_action


def _family_from_parsed_event(ev: dict) -> str:
    base, _ = parse_event_action(ev if isinstance(ev, dict) else {})
    fam = parse_base_action(base) if base else "(none)"
    return fam or "(none)"


def _family_from_granular_step(step: dict) -> str | None:
    a = step.get("action")
    if not a:
        return None
    fam = parse_base_action(str(a))
    return fam if fam else "(none)"


def count_parsed_events(parsed_root: Path) -> Counter[str]:
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
                if isinstance(ev, dict):
                    c[_family_from_parsed_event(ev)] += 1
    return c


def count_granular_steps(granular_root: Path) -> Counter[str]:
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
                fam = _family_from_granular_step(step)
                if fam is None:
                    continue
                c[fam] += 1
    return c


def _write_table(path: Path, counts: Counter[str], universe: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_family", "count"])
        total = 0
        for fam in universe:
            n = int(counts.get(fam, 0))
            total += n
            w.writerow([fam, n])
        w.writerow(["TOTAL", total])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--granularized", type=Path, default=Path("data/granularized"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/tables"),
    )
    args = ap.parse_args()

    p = count_parsed_events(args.parsed)
    g = count_granular_steps(args.granularized)
    if not p and not g:
        raise SystemExit("No data under --parsed or --granularized.")

    uni = sorted(
        set(p.keys()) | set(g.keys()),
        key=lambda f: (
            -max(p.get(f, 0), g.get(f, 0)),
            f,
        ),
    )

    out1 = args.out_dir / "extracted_events_by_family.csv"
    out2 = args.out_dir / "decomposed_events_by_family.csv"
    _write_table(out1, p, uni)
    _write_table(out2, g, uni)
    print(f"Wrote {out1} ({sum(p.values())} events)")
    print(f"Wrote {out2} ({sum(g.values())} steps)")


if __name__ == "__main__":
    main()
