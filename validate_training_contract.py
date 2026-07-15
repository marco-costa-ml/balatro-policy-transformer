#!/usr/bin/env python3
"""
validate_training_contract.py
=============================
End-to-end "supervised signal sanity" gate for the Balatro imitation-learning
pipeline.

Two complementary modes:

- ``--mode granular`` (default) walks every granularized micro-step:
    1. Resolves ``step.action`` -> ``target_action_id`` via ``label_resolver``.
    2. Builds the v1 action mask via ``mask_builder``.
    3. Asserts ``action_mask[target_action_id] == 1``.
    4. Aggregates pass/fail stats by ``page_name``, ``source_action`` family,
       and ``source_kind``.
- ``--mode tensorized`` walks every per-run ``.npz`` shard under
  ``data/tensorized/`` and enforces the branched-policy invariants emitted
  by tensorize schema 3.0.0:
    * ``0 <= family_id < n_families`` and ``family_mask[family_id] == 1``.
    * v1 back-compat ``action_mask[target_action_id] == 1``.
    * Per decoder-shape pointer / num_cards constraints:
        - card_seq: ``1 <= num_cards <= MAX_CARDS_PER_DECISION``;
          each ``card_ptr_local_seq[i] ∈ [0, MAX_CARD_ZONE_SIZE)`` and
          ``card_pointer_mask[card_ptr_local_seq[i]] == 1``; padding == -1.
        - single_ptr: ``item_ptr_local`` in range and item mask set.
        - chained_cards: item pointer + (optional) card sequence both
          satisfy their per-zone masks.
        - joker_pair (SWAP): ``swap_i_local != swap_j_local``, both within
          ``MAX_JOKER_SLOTS`` and selected by ``swap_joker_mask``.
- ``--mode both`` runs the granular pass then the tensorized pass.

This is the gating step before any model code: until the pass rate is
essentially 100% the supervised signal cannot be trusted.

Usage
-----
``python validate_training_contract.py [--mode granular|tensorized|both]
   [--src data/granularized] [--tensorized data/tensorized]
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

from action_map import compute_action_map
from family_map import compute_family_map
from label_resolver import (
    LabelResolutionError,
    check_step_consistency,
    iter_granularized_steps,
    resolve_label,
)
from mask_builder import (
    allowed_families_for_page,
    build_action_mask,
    candidate_count_for_subfamily,
)


# ---------------------------------------------------------------------------
# Tensorized (branched-policy schema 3.0.0) validation
# ---------------------------------------------------------------------------


_TENSORIZED_CHANNELS = (
    "family_id",
    "num_cards",
    "item_ptr_local",
    "card_ptr_local_seq",
    "swap_i_local",
    "swap_j_local",
    "family_mask",
    "item_pointer_mask",
    "card_pointer_mask",
    "swap_joker_mask",
    "target_action_id",
    "action_mask",
)


def _iter_tensorized_shards(root: Path):
    """Yield ``(video_id, run_path)`` pairs for every ``.npz`` shard."""
    if not root.exists():
        return
    for partition in sorted(root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for shard in sorted(partition.glob("run_*.npz")):
            yield video_id, shard


def _validate_npz_shard(
    npz: dict[str, np.ndarray],
    family_map: dict[str, Any],
    caps: dict[str, int],
    failure_classes: collections.Counter,
    pass_counts: collections.Counter,
    failure_examples: dict[str, list[dict[str, Any]]],
    max_failure_examples: int,
    shard_path: Path,
) -> tuple[int, int, int]:
    """Validate one tensorized run. Returns (n_rows, n_resolved, n_pass)."""
    n_families = int(family_map["n_families"])
    decoder_shapes = family_map["decoder_shapes"]
    id_to_family = family_map["id_to_family"]
    max_cards = int(caps["MAX_CARDS_PER_DECISION"])
    max_item = int(caps["MAX_ITEM_ZONE_SIZE"])
    max_card_zone = int(caps["MAX_CARD_ZONE_SIZE"])
    max_joker = int(caps["MAX_JOKER_SLOTS"])

    family_ids = np.asarray(npz["family_id"]).astype(np.int64)
    num_cards = np.asarray(npz["num_cards"]).astype(np.int64)
    item_ptr_local = np.asarray(npz["item_ptr_local"]).astype(np.int64)
    card_ptr_local_seq = np.asarray(npz["card_ptr_local_seq"]).astype(np.int64)
    swap_i_local = np.asarray(npz["swap_i_local"]).astype(np.int64)
    swap_j_local = np.asarray(npz["swap_j_local"]).astype(np.int64)
    family_mask = np.asarray(npz["family_mask"]).astype(bool)
    item_pointer_mask = np.asarray(npz["item_pointer_mask"]).astype(bool)
    card_pointer_mask = np.asarray(npz["card_pointer_mask"]).astype(bool)
    swap_joker_mask = np.asarray(npz["swap_joker_mask"]).astype(bool)
    target_action_id = np.asarray(npz["target_action_id"]).astype(np.int64)
    action_mask = np.asarray(npz["action_mask"]).astype(bool)

    n_rows = int(family_ids.shape[0])
    n_resolved = 0
    n_pass = 0

    def _record(cls: str, payload: dict[str, Any]) -> None:
        failure_classes[cls] += 1
        if len(failure_examples[cls]) < max_failure_examples:
            failure_examples[cls].append(payload)

    for k in range(n_rows):
        fid = int(family_ids[k])
        # ------------------- v1 back-compat (target_action_id) ------------
        aid = int(target_action_id[k])
        if aid >= 0:
            if not bool(action_mask[k, aid]):
                _record(
                    "v1_action_mask_violation",
                    {"shard": shard_path.as_posix(), "row": k, "target_action_id": aid},
                )

        # ------------------- family-level checks --------------------------
        if fid < 0:
            pass_counts["family_unresolved"] += 1
            continue
        if fid >= n_families:
            _record(
                "family_id_out_of_range",
                {"shard": shard_path.as_posix(), "row": k, "family_id": fid},
            )
            continue
        n_resolved += 1
        if not bool(family_mask[k, fid]):
            _record(
                "family_mask_not_set_at_id",
                {
                    "shard": shard_path.as_posix(),
                    "row": k,
                    "family_id": fid,
                    "family": id_to_family[fid],
                },
            )
            continue

        fname = id_to_family[fid]
        shape = decoder_shapes.get(fname, "unknown")

        # ------------------- shape-specific checks ------------------------
        if shape == "no_args":
            pass_counts["ok_no_args"] += 1
            n_pass += 1
            continue

        if shape == "card_seq":
            nc = int(num_cards[k])
            if not (1 <= nc <= max_cards):
                _record(
                    "card_seq_num_cards_out_of_range",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "num_cards": nc,
                    },
                )
                continue
            ok = True
            for i in range(nc):
                p = int(card_ptr_local_seq[k, i])
                if not (0 <= p < max_card_zone):
                    _record(
                        "card_ptr_out_of_range",
                        {
                            "shard": shard_path.as_posix(),
                            "row": k,
                            "family": fname,
                            "i": i,
                            "p": p,
                        },
                    )
                    ok = False
                    break
                if not bool(card_pointer_mask[k, p]):
                    _record(
                        "card_ptr_mask_violation",
                        {
                            "shard": shard_path.as_posix(),
                            "row": k,
                            "family": fname,
                            "i": i,
                            "p": p,
                        },
                    )
                    ok = False
                    break
            if not ok:
                continue
            for i in range(nc, max_cards):
                if int(card_ptr_local_seq[k, i]) != -1:
                    _record(
                        "card_ptr_padding_not_minus_one",
                        {
                            "shard": shard_path.as_posix(),
                            "row": k,
                            "family": fname,
                            "i": i,
                        },
                    )
                    ok = False
                    break
            if ok:
                pass_counts["ok_card_seq"] += 1
                n_pass += 1
            continue

        if shape == "single_ptr":
            p = int(item_ptr_local[k])
            if not (0 <= p < max_item):
                _record(
                    "item_ptr_out_of_range",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "p": p,
                    },
                )
                continue
            if not bool(item_pointer_mask[k, p]):
                _record(
                    "item_ptr_mask_violation",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "p": p,
                    },
                )
                continue
            pass_counts["ok_single_ptr"] += 1
            n_pass += 1
            continue

        if shape == "chained_cards":
            p = int(item_ptr_local[k])
            if not (0 <= p < max_item):
                _record(
                    "item_ptr_out_of_range",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "p": p,
                    },
                )
                continue
            if not bool(item_pointer_mask[k, p]):
                _record(
                    "item_ptr_mask_violation",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "p": p,
                    },
                )
                continue
            nc = int(num_cards[k])
            if not (0 <= nc <= max_cards):
                _record(
                    "chained_num_cards_out_of_range",
                    {
                        "shard": shard_path.as_posix(),
                        "row": k,
                        "family": fname,
                        "num_cards": nc,
                    },
                )
                continue
            ok = True
            for i in range(nc):
                cp = int(card_ptr_local_seq[k, i])
                if not (0 <= cp < max_card_zone):
                    _record(
                        "card_ptr_out_of_range",
                        {"shard": shard_path.as_posix(), "row": k, "family": fname, "i": i, "p": cp},
                    )
                    ok = False
                    break
                if not bool(card_pointer_mask[k, cp]):
                    _record(
                        "card_ptr_mask_violation",
                        {"shard": shard_path.as_posix(), "row": k, "family": fname, "i": i, "p": cp},
                    )
                    ok = False
                    break
            if not ok:
                continue
            for i in range(nc, max_cards):
                if int(card_ptr_local_seq[k, i]) != -1:
                    _record(
                        "card_ptr_padding_not_minus_one",
                        {"shard": shard_path.as_posix(), "row": k, "family": fname, "i": i},
                    )
                    ok = False
                    break
            if ok:
                pass_counts["ok_chained_cards"] += 1
                n_pass += 1
            continue

        if shape == "joker_pair":
            i_pos = int(swap_i_local[k])
            j_pos = int(swap_j_local[k])
            if i_pos == j_pos:
                _record(
                    "swap_i_eq_j",
                    {"shard": shard_path.as_posix(), "row": k, "swap_i": i_pos, "swap_j": j_pos},
                )
                continue
            if not (0 <= i_pos < max_joker and 0 <= j_pos < max_joker):
                _record(
                    "swap_pos_out_of_range",
                    {"shard": shard_path.as_posix(), "row": k, "swap_i": i_pos, "swap_j": j_pos},
                )
                continue
            if not (bool(swap_joker_mask[k, i_pos]) and bool(swap_joker_mask[k, j_pos])):
                _record(
                    "swap_joker_mask_violation",
                    {"shard": shard_path.as_posix(), "row": k, "swap_i": i_pos, "swap_j": j_pos},
                )
                continue
            pass_counts["ok_joker_pair"] += 1
            n_pass += 1
            continue

        # 'reserved' and unknown shapes
        _record(
            "unknown_decoder_shape",
            {"shard": shard_path.as_posix(), "row": k, "family": fname, "shape": shape},
        )

    return n_rows, n_resolved, n_pass


def run_tensorized_validation(
    tensorized_root: Path,
    family_map: dict[str, Any],
    caps: dict[str, int],
    max_failure_examples: int,
) -> dict[str, Any]:
    """Validate every tensorized shard; return a serializable report dict."""
    failure_classes: collections.Counter = collections.Counter()
    pass_counts: collections.Counter = collections.Counter()
    failure_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    per_family_pass: collections.Counter = collections.Counter()
    per_family_fail: collections.Counter = collections.Counter()

    n_rows_total = 0
    n_resolved_total = 0
    n_pass_total = 0
    shards_seen = 0
    shards_missing_channels = 0

    for video_id, shard in _iter_tensorized_shards(tensorized_root):
        try:
            with np.load(shard, allow_pickle=False) as npz:
                missing = [c for c in _TENSORIZED_CHANNELS if c not in npz.files]
                if missing:
                    shards_missing_channels += 1
                    failure_classes["shard_missing_channels"] += 1
                    if len(failure_examples["shard_missing_channels"]) < max_failure_examples:
                        failure_examples["shard_missing_channels"].append(
                            {"shard": shard.as_posix(), "missing": missing}
                        )
                    continue
                shard_data = {c: np.asarray(npz[c]) for c in _TENSORIZED_CHANNELS}
        except Exception as exc:
            failure_classes["shard_load_error"] += 1
            if len(failure_examples["shard_load_error"]) < max_failure_examples:
                failure_examples["shard_load_error"].append(
                    {"shard": shard.as_posix(), "error": repr(exc)}
                )
            continue

        shards_seen += 1
        n_rows, n_resolved, n_pass = _validate_npz_shard(
            shard_data,
            family_map,
            caps,
            failure_classes,
            pass_counts,
            failure_examples,
            max_failure_examples,
            shard,
        )
        n_rows_total += n_rows
        n_resolved_total += n_resolved
        n_pass_total += n_pass

        # Per-family pass/fail counts derived from raw family_id + family_mask.
        family_ids = shard_data["family_id"].astype(np.int64)
        family_mask = shard_data["family_mask"].astype(bool)
        for k in range(family_ids.shape[0]):
            fid = int(family_ids[k])
            if fid < 0:
                continue
            fname = family_map["id_to_family"][fid] if 0 <= fid < family_map["n_families"] else f"id={fid}"
            if bool(family_mask[k, fid]) and 0 <= fid < family_map["n_families"]:
                per_family_pass[fname] += 1
            else:
                per_family_fail[fname] += 1

    return {
        "tensorized_root": tensorized_root.as_posix(),
        "shards_seen": shards_seen,
        "shards_missing_channels": shards_missing_channels,
        "n_rows": n_rows_total,
        "n_family_resolved": n_resolved_total,
        "n_full_pass": n_pass_total,
        "n_failures": int(sum(failure_classes.values())),
        "full_pass_rate": n_pass_total / max(n_resolved_total, 1),
        "failure_classes": dict(failure_classes),
        "pass_counts": dict(pass_counts),
        "per_family_pass": dict(per_family_pass),
        "per_family_fail": dict(per_family_fail),
        "failure_examples": dict(failure_examples),
    }


def _print_tensorized_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("TENSORIZED-MODE branched-policy contract")
    print("=" * 72)
    print(f"tensorized_root:           {report['tensorized_root']}")
    print(f"shards seen:               {report['shards_seen']}")
    print(f"shards missing channels:   {report['shards_missing_channels']}")
    print(f"rows scanned:              {report['n_rows']}")
    print(f"family-resolved rows:      {report['n_family_resolved']}")
    print(f"full-pass rows:            {report['n_full_pass']}  "
          f"({100.0 * report['full_pass_rate']:.4f}%)")
    print(f"failure rows:              {report['n_failures']}")
    print()
    if report["failure_classes"]:
        print("--- failure classes ---")
        for cls, n in sorted(report["failure_classes"].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cls:40s} {n}")
        print()
    if report["pass_counts"]:
        print("--- pass buckets ---")
        for cls, n in sorted(report["pass_counts"].items()):
            print(f"  {cls:40s} {n}")
        print()
    if report["per_family_pass"] or report["per_family_fail"]:
        print("--- per-family pass/fail ---")
        families = sorted(set(report["per_family_pass"]) | set(report["per_family_fail"]))
        for fam in families:
            p = report["per_family_pass"].get(fam, 0)
            f = report["per_family_fail"].get(fam, 0)
            tot = p + f
            rate = p / max(tot, 1)
            print(f"  {fam:55s} pass={p:6d}  fail={f:6d}  rate={100.0 * rate:7.3f}%")
        print()
    if report["failure_examples"]:
        print("--- failure examples (first few per class) ---")
        for cls, exs in report["failure_examples"].items():
            print(f"[{cls}]")
            for ex in exs[:5]:
                print(f"  {ex}")
        print()


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
    family_offsets = action_map["family_offsets"]
    family_sizes = action_map["family_sizes"]

    if base not in allowed:
        return f"page_gate_blocked:page={page},base={base}"

    if base == "SWAP":
        from mask_builder import _jokers_current  # local import keeps surface clean

        jc = _jokers_current(step)
        if jc < 2:
            return f"swap_jokers_current_lt_2:jokers_current={jc}"
        try:
            _, ij = label.split("_", 1)
            i, j = (int(x) for x in ij.split("_"))
        except Exception:
            return f"swap_label_unparseable:{label}"
        if not (i < jc and j < jc):
            return f"swap_index_out_of_bounds:i={i},j={j},jokers_current={jc}"
        return f"swap_unexpected_mask_zero:{label}"

    # Indexed single-target families. Reconstruct the subfamily key from
    # the label: e.g. ``UseConsumable_CurrentConsumables_0`` -> base
    # ``UseConsumable`` + zone ``CurrentConsumables`` -> subfamily key
    # ``UseConsumable_CurrentConsumables``.
    if base in {
        "UseConsumable",
        "SelectCard",
        "SelectPackItem",
        "BuyAndUseShopConsumable",
        "BuyShopItem",
        "SellItem",
    }:
        parts = label.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
            return f"label_unparseable:{label}"
        subfamily_key = parts[0]
        zone = subfamily_key.split("_", 1)[1] if "_" in subfamily_key else None
        fam_offset = int(family_offsets.get(subfamily_key, -1))
        fam_size = int(family_sizes.get(subfamily_key, 0))
        if fam_offset < 0 or fam_size <= 0 or zone is None:
            return f"unknown_subfamily:{subfamily_key}"
        idx = target_action_id - fam_offset
        n_candidates = candidate_count_for_subfamily(base, zone, step)
        if not (0 <= idx < fam_size):
            return f"index_out_of_family_range:label={label},idx={idx},size={fam_size}"
        if idx >= n_candidates:
            return (
                f"target_exists_blocked:label={label},idx={idx},"
                f"n_candidates={n_candidates},source_kind={step.get('source_kind')}"
            )
        return f"unexpected_mask_zero:{label}"

    # Fixed families.
    fam_offset = int(family_offsets.get(base, -1))
    if fam_offset < 0:
        return f"unknown_fixed_family:{base}"
    if mask[fam_offset] != True:  # noqa: E712 — explicit for readability
        return f"fixed_family_unexpected_mask_zero:{base}"
    return f"unexpected_mask_zero:{label}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validate the supervised-signal contract for the Balatro IL pipeline. "
            "Default mode walks granularized data; --mode tensorized walks the per-run "
            ".npz shards under data/tensorized; --mode both runs both passes."
        )
    )
    ap.add_argument(
        "--mode",
        choices=("granular", "tensorized", "both"),
        default="granular",
        help="Validation mode (default: granular).",
    )
    ap.add_argument("--src", type=Path, default=Path("data/granularized"))
    ap.add_argument(
        "--tensorized",
        type=Path,
        default=Path("data/tensorized"),
        help="Root of tensorized .npz shards (used by tensorized / both modes).",
    )
    ap.add_argument("--action-map", type=Path, default=Path("data/action_map.json"))
    ap.add_argument(
        "--action-config",
        type=Path,
        default=Path("data/action_space_config.json"),
        help="Action space config; used to derive branched caps in tensorized mode.",
    )
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

    run_granular = args.mode in ("granular", "both")
    run_tensorized = args.mode in ("tensorized", "both")

    if run_granular and not args.src.exists():
        raise SystemExit(f"granularized root not found: {args.src}")
    if run_tensorized and not args.tensorized.exists():
        raise SystemExit(f"tensorized root not found: {args.tensorized}")
    if not args.action_map.exists():
        raise SystemExit(f"action map not found: {args.action_map}")

    action_map = json.loads(args.action_map.read_text(encoding="utf-8"))
    label_to_index = action_map["label_to_index"]
    n_actions = int(action_map["n_actions"])

    granular_payload: dict[str, Any] | None = None
    tensorized_payload: dict[str, Any] | None = None

    if not run_granular:
        # Jump straight to the tensorized pass + report writing.
        granular_payload = None
    else:
        granular_payload = _run_granular_validation(
            args=args,
            action_map=action_map,
            label_to_index=label_to_index,
            n_actions=n_actions,
        )

    if run_tensorized:
        action_config = json.loads(args.action_config.read_text(encoding="utf-8"))
        compiled_action_map = compute_action_map(action_config)
        family_map = compute_family_map(compiled_action_map)
        from tensorize import derive_branched_caps

        caps = derive_branched_caps(compiled_action_map, family_map)
        tensorized_payload = run_tensorized_validation(
            args.tensorized, family_map, caps, args.max_failure_examples
        )
        tensorized_payload["family_map_version"] = family_map["family_map_version"]
        tensorized_payload["n_families"] = family_map["n_families"]
        tensorized_payload["branched_caps"] = caps
        _print_tensorized_report(tensorized_payload)

    # Write the combined machine-readable report and return.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "action_map_path": args.action_map.as_posix(),
        "action_map_version": action_map.get("action_map_version"),
        "n_actions": n_actions,
    }
    if granular_payload is not None:
        payload["granular"] = granular_payload
    if tensorized_payload is not None:
        payload["tensorized"] = tensorized_payload
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote report -> {args.out.as_posix()}")


def _run_granular_validation(
    *,
    args: argparse.Namespace,
    action_map: dict[str, Any],
    label_to_index: dict[str, int],
    n_actions: int,
) -> dict[str, Any]:
    """Run the legacy granular-mode validation pass and return a payload dict.

    Prints the human-readable report along the way; the returned dict is
    embedded under ``payload['granular']`` of the combined report.
    """
    total_steps = 0
    label_resolved = 0
    label_consistency_failures: collections.Counter = collections.Counter()
    label_consistency_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    mask_pass = 0
    mask_fail = 0
    mask_failure_reasons: collections.Counter = collections.Counter()
    mask_failure_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    fail_by_page: collections.Counter = collections.Counter()
    fail_by_family: collections.Counter = collections.Counter()
    fail_by_source_kind: collections.Counter = collections.Counter()
    pass_by_page: collections.Counter = collections.Counter()
    pass_by_family: collections.Counter = collections.Counter()
    pass_by_source_kind: collections.Counter = collections.Counter()

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

    print("=" * 72)
    print("GRANULAR-MODE v1 contract")
    print("=" * 72)
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

    return {
        "src": args.src.as_posix(),
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


if __name__ == "__main__":
    main()
