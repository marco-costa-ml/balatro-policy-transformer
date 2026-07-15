#!/usr/bin/env python3
"""
Per-action and subevent histograms for **resolved** training steps (tensorized).

Action counts use ``target_action_id`` × ``index_to_label`` (same rows as
``BalatroStepDataset`` with ``include_unresolved=False``).

Subevent bucket:
- If ``action_subtype_id`` maps to a non-PAD vocab entry **and** it is not an
  opaque ``ev_*`` id, use that string (e.g. ``buyvoucher``, ``skipplanetstandardbuffoonpack``).
- Opaque ``ev_*`` ids are treated like missing subtype (same as PAD).
- Otherwise use the flat action’s leading family token, lowercased
  (e.g. ``SkipPack`` → ``skippack``, ``SelectCard_CurrentHand_3`` → ``selectcard``, ``PlayHand`` → ``playhand``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map  # noqa: E402
from dataset import BalatroStepDataset, load_split  # noqa: E402


def load_subtype_lookup(vocab_path: Path) -> list[str]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    values = payload["vocabularies"]["action_subtype"]["values"]
    if not isinstance(values, list):
        raise SystemExit("vocab action_subtype.values must be a list")
    return [str(x) for x in values]


def subevent_bucket(
    action_label: str,
    subtype_id: int,
    subtype_values: list[str],
    *,
    preserve_ev_subtypes: bool = False,
) -> str:
    i = int(subtype_id)
    if 0 < i < len(subtype_values):
        name = subtype_values[i]
        if name and name != "<PAD>":
            if not preserve_ev_subtypes and name.startswith("ev_"):
                return action_label.split("_", 1)[0].lower()
            return name
    return action_label.split("_", 1)[0].lower()


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 6) if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensorized", type=Path, default=_REPO_ROOT / "data" / "tensorized")
    ap.add_argument("--splits", type=Path, default=_REPO_ROOT / "artifacts" / "splits.json")
    ap.add_argument(
        "--action-config",
        type=Path,
        default=_REPO_ROOT / "data" / "action_space_config.json",
    )
    ap.add_argument(
        "--vocab",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "vocab.json",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="Which split’s videos to include (default: train).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    ap.add_argument(
        "--preserve-ev-subtypes",
        action="store_true",
        help="Keep opaque ev_* subtype strings instead of folding to the base action.",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=35,
        help="Print this many rows per section (0 = print all).",
    )
    args = ap.parse_args()

    if not args.tensorized.exists():
        raise SystemExit(f"tensorized dir missing: {args.tensorized}")

    action_map = compute_action_map(
        json.loads(args.action_config.read_text(encoding="utf-8"))
    )
    n_actions = int(action_map["n_actions"])
    index_to_label = list(action_map["index_to_label"])

    ds = BalatroStepDataset(
        tensorized_root=args.tensorized,
        split_videos=load_split(args.splits, args.split),
        include_unresolved=False,
        device=None,
    )
    if ds.n_actions() != n_actions:
        raise SystemExit(
            f"tensorized N_ACTIONS={ds.n_actions()} != action_map {n_actions}"
        )

    flat = ds.valid_indices()
    tensors = ds.all_tensors()
    y = tensors["target_action_id"].index_select(0, flat).long().cpu()
    st = tensors["action_subtype_id"].index_select(0, flat).long().cpu()

    total = int(y.numel())
    counts_action = torch.bincount(y, minlength=n_actions)
    subtype_values = load_subtype_lookup(args.vocab)

    sub_counts: Counter[str] = Counter()
    for i in range(total):
        lab = index_to_label[int(y[i].item())]
        sub = subevent_bucket(
            lab,
            int(st[i].item()),
            subtype_values,
            preserve_ev_subtypes=args.preserve_ev_subtypes,
        )
        sub_counts[sub] += 1

    action_rows = [
        {
            "action_index": i,
            "label": index_to_label[i],
            "count": int(counts_action[i].item()),
            "percent_of_split": pct(int(counts_action[i].item()), total),
        }
        for i in range(n_actions)
    ]
    action_rows.sort(key=lambda r: (-r["count"], r["label"]))

    sub_rows = [
        {"subevent": k, "count": v, "percent_of_split": pct(v, total)}
        for k, v in sorted(sub_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    print(f"split={args.split!r}  resolved_steps={total}")
    print(f"tensorized={args.tensorized.resolve()}")
    print()
    print("=== Per flat action (label) ===")
    lim = len(action_rows) if args.top == 0 else min(args.top, len(action_rows))
    for r in action_rows[:lim]:
        print(
            f"  {r['count']:>8}  {r['percent_of_split']:>9.4f}%  {r['label']}"
        )
    if lim < len(action_rows):
        print(f"  ... ({len(action_rows) - lim} more actions)")
    print()
    print("=== By subevent bucket ===")
    lim2 = len(sub_rows) if args.top == 0 else min(args.top, len(sub_rows))
    for r in sub_rows[:lim2]:
        print(
            f"  {r['count']:>8}  {r['percent_of_split']:>9.4f}%  {r['subevent']}"
        )
    if lim2 < len(sub_rows):
        print(f"  ... ({len(sub_rows) - lim2} more subevents)")

    if args.json_out:
        payload = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "resolved_steps": total,
            "n_actions": n_actions,
            "preserve_ev_subtypes": args.preserve_ev_subtypes,
            "per_action": action_rows,
            "per_subevent": sub_rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Wrote {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
