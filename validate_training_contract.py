#!/usr/bin/env python3
"""
validate_training_contract.py
=============================
End-to-end "supervised signal sanity" gate for the Balatro imitation-learning
pipeline.

For every granularized step under ``data/granularized/``:

1. Resolve ``step.action`` -> ``target_action_id`` via ``label_resolver``.
2. Build the v1 action mask via ``mask_builder``.
3. Assert ``action_mask[target_action_id] == 1``.
4. Aggregate pass/fail stats by ``page_name``, ``source_action`` family, and
   ``source_kind``; print a clean report.
5. Persist the report to ``artifacts/training_contract_report.json``.

This is the gating step before any tensorization / model code: until the
pass rate is essentially 100%, the labels and masks disagree and there is
no trustworthy supervised signal to train on.

Usage
-----
``python validate_training_contract.py [--src data/granularized]
   [--action-map data/action_map.json] [--out artifacts/training_contract_report.json]
   [--max-failure-examples 10]``
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from label_resolver import (
    LabelResolutionError,
    check_step_consistency,
    iter_granularized_steps,
    resolve_label,
)
from mask_builder import (
    allowed_families_for_page,
    build_action_mask,
    candidate_count_for_family,
)


def _diagnose_failure(
    step: dict[str, Any],
    action_map: dict[str, Any],
    target_action_id: int,
    mask: np.ndarray,
) -> str:
    """
    Categorize *why* mask[target_action_id] == 0 for a step that we thought
    should be unmasked. Helps point at the exact mask rule to fix.
    """
    label = step["action"]
    base = label.split("_", 1)[0]
    page = step.get("page_name")
    allowed = allowed_families_for_page(page)

    if base not in allowed:
        return f"page_gate_blocked:page={page},base={base}"

    fam_offset = action_map["family_offsets"][base]
    fam_size = action_map["family_sizes"][base]

    if base == "SWAP":
        from mask_builder import _jokers_current  # local import keeps surface clean

        jc = _jokers_current(step)
        if jc < 2:
            return f"swap_jokers_current_lt_2:jokers_current={jc}"
        # parse i_j
        try:
            _, ij = label.split("_", 1)
            i, j = (int(x) for x in ij.split("_"))
        except Exception:
            return f"swap_label_unparseable:{label}"
        if not (i < jc and j < jc):
            return f"swap_index_out_of_bounds:i={i},j={j},jokers_current={jc}"
        return f"swap_unexpected_mask_zero:{label}"

    # Indexed single-target families.
    if base in {
        "UseConsumable",
        "SelectCard",
        "SelectPackItem",
        "BuyAndUseShopConsumable",
        "BuyShopItem",
        "SellItem",
    }:
        idx = target_action_id - fam_offset
        n_candidates = candidate_count_for_family(base, step)
        if not (0 <= idx < fam_size):
            return f"index_out_of_family_range:label={label},idx={idx},size={fam_size}"
        if idx >= n_candidates:
            return (
                f"target_exists_blocked:label={label},idx={idx},"
                f"n_candidates={n_candidates},source_kind={step.get('source_kind')}"
            )
        return f"unexpected_mask_zero:{label}"

    # Fixed families.
    if mask[fam_offset] != True:  # noqa: E712 — explicit for readability
        return f"fixed_family_unexpected_mask_zero:{base}"
    return f"unexpected_mask_zero:{label}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Validate end-to-end (target_action_id, action_mask) contract over granularized data."
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument("--action-map", type=Path, default=Path("data/action_map.json"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/training_contract_report.json"),
        help="Where to write the JSON report.",
    )
    ap.add_argument(
        "--max-failure-examples",
        type=int,
        default=10,
        help="Per failure class, how many concrete examples to keep in the report.",
    )
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"granularized root not found: {args.src}")
    if not args.action_map.exists():
        raise SystemExit(f"action map not found: {args.action_map}")

    action_map = json.loads(args.action_map.read_text(encoding="utf-8"))
    label_to_index = action_map["label_to_index"]
    n_actions = int(action_map["n_actions"])

    total_steps = 0
    label_resolved = 0
    label_consistency_failures = collections.Counter()
    label_consistency_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    mask_pass = 0
    mask_fail = 0
    mask_failure_reasons = collections.Counter()
    mask_failure_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    fail_by_page = collections.Counter()
    fail_by_family = collections.Counter()
    fail_by_source_kind = collections.Counter()
    pass_by_page = collections.Counter()
    pass_by_family = collections.Counter()
    pass_by_source_kind = collections.Counter()

    family_target_distribution: collections.Counter = collections.Counter()
    unmask_count_sum = 0
    unmask_count_min = n_actions + 1
    unmask_count_max = -1

    def _record_example(bucket: dict[str, list[dict[str, Any]]], cls: str, payload: dict[str, Any]) -> None:
        if len(bucket[cls]) < args.max_failure_examples:
            bucket[cls].append(payload)

    for run_path, step in iter_granularized_steps(args.src):
        total_steps += 1
        page = step.get("page_name") or "<none>"
        base = (step.get("action") or "").split("_", 1)[0]
        source_kind = step.get("source_kind") or "<none>"

        # ------------------------------------------------------------------
        # 1) Label resolution + numeric/string consistency check.
        # ------------------------------------------------------------------
        diag = check_step_consistency(step, action_map)
        if diag is not None:
            cls = diag.split(":", 1)[0]
            label_consistency_failures[cls] += 1
            _record_example(
                label_consistency_examples,
                cls,
                {
                    "run_file": run_path.as_posix(),
                    "step_id": step.get("step_id"),
                    "action": step.get("action"),
                    "page_name": page,
                    "source_kind": source_kind,
                    "target_index": step.get("target_index"),
                    "swap_pair": step.get("swap_pair"),
                    "diagnostic": diag,
                },
            )
            continue
        label_resolved += 1

        try:
            target_action_id = resolve_label(step, label_to_index)
        except LabelResolutionError as exc:
            label_consistency_failures["resolve_label_raise"] += 1
            _record_example(
                label_consistency_examples,
                "resolve_label_raise",
                {
                    "run_file": run_path.as_posix(),
                    "step_id": step.get("step_id"),
                    "action": step.get("action"),
                    "error": str(exc),
                },
            )
            continue

        family_target_distribution[(base, target_action_id)] += 1

        # ------------------------------------------------------------------
        # 2) Mask construction + assertion.
        # ------------------------------------------------------------------
        mask = build_action_mask(step, action_map)
        unmask_count = int(mask.sum())
        unmask_count_sum += unmask_count
        if unmask_count < unmask_count_min:
            unmask_count_min = unmask_count
        if unmask_count > unmask_count_max:
            unmask_count_max = unmask_count

        if mask[target_action_id]:
            mask_pass += 1
            pass_by_page[page] += 1
            pass_by_family[base] += 1
            pass_by_source_kind[source_kind] += 1
        else:
            mask_fail += 1
            fail_by_page[page] += 1
            fail_by_family[base] += 1
            fail_by_source_kind[source_kind] += 1
            reason = _diagnose_failure(step, action_map, target_action_id, mask)
            cls = reason.split(":", 1)[0]
            mask_failure_reasons[cls] += 1
            _record_example(
                mask_failure_examples,
                cls,
                {
                    "run_file": run_path.as_posix(),
                    "step_id": step.get("step_id"),
                    "action": step.get("action"),
                    "target_action_id": target_action_id,
                    "page_name": page,
                    "source_kind": source_kind,
                    "target_index": step.get("target_index"),
                    "swap_pair": step.get("swap_pair"),
                    "unmasked_count": unmask_count,
                    "diagnostic": reason,
                },
            )

    pass_rate = mask_pass / max(total_steps, 1)
    label_rate = label_resolved / max(total_steps, 1)
    avg_unmasked = unmask_count_sum / max(label_resolved, 1)

    # ----------------------------------------------------------------------
    # Console report.
    # ----------------------------------------------------------------------
    print(f"granularized steps scanned:   {total_steps}")
    print(f"label resolution clean:        {label_resolved}  ({100.0*label_rate:.4f}%)")
    print(f"mask[target_action_id] == 1:   {mask_pass}  ({100.0*pass_rate:.4f}%)")
    print(f"mask FAILED:                   {mask_fail}")
    print(f"avg #unmasked actions / step:  {avg_unmasked:.2f}  (min={unmask_count_min}, max={unmask_count_max})")
    print()

    if label_consistency_failures:
        print("--- label-consistency failures ---")
        for cls, n in sorted(label_consistency_failures.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cls:40s} {n}")
        print()

    if mask_failure_reasons:
        print("--- mask failure classes ---")
        for cls, n in sorted(mask_failure_reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cls:40s} {n}")
        print()

    print("--- mask pass rate by page ---")
    for page in sorted(set(pass_by_page) | set(fail_by_page)):
        p = pass_by_page[page]
        f = fail_by_page[page]
        tot = p + f
        rate = p / max(tot, 1)
        print(f"  {page:32s} pass={p:6d}  fail={f:6d}  rate={100.0*rate:7.3f}%")
    print()

    print("--- mask pass rate by family ---")
    for fam in sorted(set(pass_by_family) | set(fail_by_family)):
        p = pass_by_family[fam]
        f = fail_by_family[fam]
        tot = p + f
        rate = p / max(tot, 1)
        print(f"  {fam:32s} pass={p:6d}  fail={f:6d}  rate={100.0*rate:7.3f}%")
    print()

    print("--- mask pass rate by source_kind ---")
    for sk in sorted(set(pass_by_source_kind) | set(fail_by_source_kind)):
        p = pass_by_source_kind[sk]
        f = fail_by_source_kind[sk]
        tot = p + f
        rate = p / max(tot, 1)
        print(f"  {sk:32s} pass={p:6d}  fail={f:6d}  rate={100.0*rate:7.3f}%")
    print()

    if mask_failure_examples:
        print("--- mask failure examples (first few per class) ---")
        for cls, exs in mask_failure_examples.items():
            print(f"[{cls}]")
            for ex in exs:
                print(
                    f"  {ex['run_file']}  step_id={ex['step_id']}  "
                    f"action={ex['action']!r}  page={ex['page_name']!r}  "
                    f"src_kind={ex['source_kind']!r}  diag={ex['diagnostic']}"
                )
        print()

    # ----------------------------------------------------------------------
    # Persist machine-readable report.
    # ----------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src": args.src.as_posix(),
        "action_map_path": args.action_map.as_posix(),
        "action_map_version": action_map.get("action_map_version"),
        "n_actions": n_actions,
        "totals": {
            "steps_scanned": total_steps,
            "label_resolved": label_resolved,
            "mask_pass": mask_pass,
            "mask_fail": mask_fail,
            "label_resolution_rate": label_rate,
            "mask_pass_rate": pass_rate,
            "avg_unmasked_actions_per_step": avg_unmasked,
            "min_unmasked_actions_per_step": (
                unmask_count_min if unmask_count_max >= 0 else None
            ),
            "max_unmasked_actions_per_step": (
                unmask_count_max if unmask_count_max >= 0 else None
            ),
        },
        "label_consistency_failures": dict(label_consistency_failures),
        "label_consistency_examples": {k: v for k, v in label_consistency_examples.items()},
        "mask_failure_classes": dict(mask_failure_reasons),
        "mask_failure_examples": {k: v for k, v in mask_failure_examples.items()},
        "pass_by_page": dict(pass_by_page),
        "fail_by_page": dict(fail_by_page),
        "pass_by_family": dict(pass_by_family),
        "fail_by_family": dict(fail_by_family),
        "pass_by_source_kind": dict(pass_by_source_kind),
        "fail_by_source_kind": dict(fail_by_source_kind),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote report -> {args.out.as_posix()}")


if __name__ == "__main__":
    main()
