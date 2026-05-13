#!/usr/bin/env python3
"""
label_resolver.py
=================
Resolve a granularized step's `action` string (e.g.
``SelectCard_CurrentHand_3``, ``BuyShopItem_VoucherShopOfferings_0``,
``SWAP_0_1``, ``CashOut``) into the canonical integer
``target_action_id`` defined by ``data/action_map.json``.

Granularizer guarantees (from ``granularize.py`` schema_version 3.0.0):
- Indexed single-target families emit per-zone ``Base_Zone_i`` labels.
- ``SWAP`` always emits ``SWAP_i_j`` with ``i < j``.
- Indexed steps also expose ``target_zone`` + ``target_position`` (or
  ``swap_pair`` for SWAP) so resolution can be done either via the
  action string OR via the structured fields. Both paths must agree;
  mismatches are reported as errors by ``check_step_consistency``.

Importable helpers
------------------
- ``resolve_label(step, label_to_index) -> int``
- ``check_step_consistency(step, action_map) -> str | None``  (None = ok)
- ``LabelResolutionError``

CLI
---
``python label_resolver.py [--src data/granularized] [--action-map data/action_map.json]``
prints aggregate stats: total / resolved / unresolved counts, plus the first
few unresolved examples per failure class.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


class LabelResolutionError(ValueError):
    """Raised when a step's action string cannot be resolved to an action_id."""


def resolve_label(step: dict[str, Any], label_to_index: dict[str, int]) -> int:
    """
    Resolve a granularized step's ``action`` string to its integer index.

    Raises ``LabelResolutionError`` if the label is missing or unknown.
    """
    label = step.get("action")
    if not isinstance(label, str) or not label:
        raise LabelResolutionError(f"step has no action label: {step!r}")
    try:
        return label_to_index[label]
    except KeyError as exc:
        raise LabelResolutionError(
            f"action label {label!r} not in action map "
            f"(page={step.get('page_name')!r}, source_kind={step.get('source_kind')!r})"
        ) from exc


def _parse_label(label: str) -> tuple[str, str | None, list[int]]:
    """
    Split an action label into ``(base, zone_or_None, [int_suffix...])``.

    Examples
    --------
    - ``"PlayHand"``                        -> ``("PlayHand", None, [])``
    - ``"SelectCard_CurrentHand_3"``        -> ``("SelectCard", "CurrentHand", [3])``
    - ``"BuyShopItem_VoucherShopOfferings_0"`` -> ``("BuyShopItem", "VoucherShopOfferings", [0])``
    - ``"SWAP_0_1"``                        -> ``("SWAP", None, [0, 1])``
    """
    parts = label.split("_")
    if not parts:
        return label, None, []
    base = parts[0]
    if base == "SWAP":
        suffix: list[int] = []
        for part in parts[1:]:
            try:
                suffix.append(int(part))
            except ValueError:
                return base, None, []
        return base, None, suffix
    # For non-SWAP labels: zone is the middle, index is the trailing int.
    if len(parts) == 1:
        return base, None, []
    tail = parts[-1]
    try:
        idx = int(tail)
    except ValueError:
        return base, "_".join(parts[1:]), []
    zone = "_".join(parts[1:-1]) if len(parts) > 2 else None
    return base, zone, [idx]


def check_step_consistency(
    step: dict[str, Any], action_map: dict[str, Any]
) -> str | None:
    """
    Return None if the step is internally consistent, otherwise a short
    diagnostic string describing the inconsistency (does not raise).

    Checks performed
    ----------------
    1. ``action`` resolves to an integer via ``label_to_index``.
    2. The numeric ``target_index`` / ``swap_pair`` fields agree with the
       integer suffix(es) on the action string.
    3. The resolved index falls inside the family's [offset, offset+size).
    """
    label_to_index = action_map["label_to_index"]
    family_offsets = action_map["family_offsets"]
    family_sizes = action_map["family_sizes"]

    label = step.get("action")
    if not isinstance(label, str) or not label:
        return "missing_action_label"

    if label not in label_to_index:
        return f"unknown_label:{label}"

    base, zone, suffix = _parse_label(label)
    if base != "SWAP" and base not in family_offsets and (zone is None or f"{base}_{zone}" not in family_offsets):
        return f"family_unknown:{base}"

    swap_pair = step.get("swap_pair")
    target_position = step.get("target_position")
    target_zone = step.get("target_zone")

    # Numeric / string consistency rules
    # ----------------------------------
    # * SWAP labels must be SWAP_i_j AND swap_pair == [i, j].
    # * Per-zone indexed labels must have target_position == suffix[0]
    #   AND target_zone == zone.
    # * Bare families (DiscardHand, PlayHand, RerollShop, ...) must have
    #   no numeric resolution and no integer suffix.
    if base == "SWAP":
        if len(suffix) != 2:
            return f"swap_label_not_two_indices:{label}"
        if swap_pair is None:
            return f"swap_pair_missing_for_SWAP_label:{label}"
        if [int(swap_pair[0]), int(swap_pair[1])] != suffix:
            return (
                f"swap_pair_vs_label_mismatch:label={label},"
                f"swap_pair={list(swap_pair)}"
            )
    elif suffix and zone is not None:
        if target_position is None:
            return f"target_position_missing_for_indexed_label:{label}"
        if int(target_position) != suffix[0]:
            return (
                f"target_position_vs_label_mismatch:label={label},"
                f"target_position={target_position}"
            )
        if target_zone is not None and target_zone != zone:
            return (
                f"target_zone_vs_label_mismatch:label={label},"
                f"target_zone={target_zone}"
            )
    else:
        if target_position is not None:
            return (
                f"target_position_set_but_bare_label:label={label},"
                f"target_position={target_position}"
            )
        if swap_pair is not None:
            return (
                f"swap_pair_set_but_bare_label:label={label},"
                f"swap_pair={list(swap_pair)}"
            )

    # Family bounds check (use the subfamily key for per-zone labels).
    idx = label_to_index[label]
    if base == "SWAP":
        family_key = "SWAP"
    elif zone is not None and suffix:
        family_key = f"{base}_{zone}"
    else:
        family_key = base
    offset = family_offsets.get(family_key)
    size = family_sizes.get(family_key)
    if offset is None or size is None:
        return f"family_unknown:{family_key}"
    if not (offset <= idx < offset + size):
        return f"index_out_of_family_range:{label}->{idx} not in [{offset},{offset+size})"

    return None


def iter_granularized_steps(src_root: Path):
    """Yield ``(path, step)`` tuples for every granularized step under src_root."""
    for partition in sorted(src_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        for run_file in sorted(partition.glob("run_*.json")):
            with open(run_file, encoding="utf-8") as fh:
                run = json.load(fh)
            for step in run.get("events", []):
                yield run_file, step


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Resolve every granularized step's action label to an action_id and report failures.",
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--action-map", type=Path, default=Path("data/action_map.json"))
    ap.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="How many failure examples to print per failure class.",
    )
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"granularized root not found: {args.src}")
    if not args.action_map.exists():
        raise SystemExit(f"action map not found: {args.action_map}")

    action_map = json.loads(args.action_map.read_text(encoding="utf-8"))

    total = 0
    resolved = 0
    failures = collections.Counter()
    failure_examples: dict[str, list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
    family_counts: collections.Counter[str] = collections.Counter()

    for run_path, step in iter_granularized_steps(args.src):
        total += 1
        diagnostic = check_step_consistency(step, action_map)
        if diagnostic is None:
            resolved += 1
            base = step["action"].split("_", 1)[0]
            family_counts[base] += 1
        else:
            cls = diagnostic.split(":", 1)[0]
            failures[cls] += 1
            if len(failure_examples[cls]) < args.max_examples:
                failure_examples[cls].append((run_path.as_posix(), step))

    print(f"granularized steps scanned: {total}")
    print(f"resolved cleanly:           {resolved}  ({100.0*resolved/max(total,1):.4f}%)")
    print(f"unresolved / inconsistent:  {total - resolved}")
    print()
    print("--- per-family resolved counts ---")
    for fam, n in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {fam:30s}  {n}")

    if failures:
        print()
        print("--- failure classes ---")
        for cls, n in sorted(failures.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cls:40s}  {n}")
        print()
        print("--- failure examples (truncated) ---")
        for cls, exs in failure_examples.items():
            print(f"[{cls}]")
            for path, step in exs:
                print(
                    f"  {path}  step_id={step.get('step_id')}  "
                    f"action={step.get('action')!r}  "
                    f"page={step.get('page_name')!r}  "
                    f"source_kind={step.get('source_kind')!r}  "
                    f"target_zone={step.get('target_zone')}  "
                    f"target_position={step.get('target_position')}  "
                    f"swap_pair={step.get('swap_pair')}"
                )


if __name__ == "__main__":
    main()
