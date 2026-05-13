#!/usr/bin/env python3
"""
diagnose_leakage.py
===================
Three diagnostics in one script.

1. **Split sanity** — verify ``artifacts/splits.json`` partitions video_ids
   without overlap (it should already, by construction in compute_splits).

2. **Observation duplicate rate across splits** — hash every step's
   *observation-only* features (the genuinely model-visible state, with
   no granularizer-derived hints like ``action_subtype_id`` or
   ``source_kind_id``) and report the rate at which val/test hashes
   appear in the train set. High overlap means the policy can shortcut
   "I've seen this exact frame before".

3. **Family-leakage quantification** — for every step, look up just
   ``action_subtype`` (and just ``source_kind``) and ask: "knowing this
   one feature alone, can you guess the action family with the
   majority-vote rule fitted on train?" If yes, the feature is target
   leakage and should be removed from the model.

Outputs a printed report and writes ``artifacts/leakage_report.json``.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS_PATH = Path("artifacts/splits.json")
TENSORIZED_ROOT = Path("data/tensorized")
ACTION_MAP_PATH = Path("data/action_map.json")
OUT_PATH = Path("artifacts/leakage_report.json")

# Features used by the model. We split into (observation-only) and
# (granularizer-derived). This mirrors model.py:GlobalEncoder.
OBSERVATION_KEYS = (
    # global categoricals (game state only)
    "page_id",
    "deck_class_id",
    "stake_class_id",
    "last_tarot_planet_class_id",
    "ante_boss_blind_class_id",
    "small_status_id",
    "big_status_id",
    # global numerics + flags + per-hand + multi-hots
    "ocr_numeric",
    "state_numeric",
    "ocr_valid",
    "flags",
    "hand_levels",
    "hand_played",
    "hand_played_this_round",
    "vouchers_redeemed",
    "bosses_used",
    # objects
    "object_class_id",
    "object_object_type_id",
    "object_zone_id",
    "object_position",
    "object_modifier_id",
    "object_edition_id",
    "object_seal_id",
    "object_rank_id",
    "object_suit_id",
    "object_is_debuffed",
    "object_sticker_rental",
    "object_sticker_perishable",
    "object_sticker_eternal",
    "object_mask",
    # tracked deck
    "deck_card_class_id",
    "deck_card_modifier_id",
    "deck_card_edition_id",
    "deck_card_seal_id",
    "deck_card_rank_id",
    "deck_card_suit_id",
    "deck_card_mask",
    # action mask is observable (derived from page + zones)
    "action_mask",
)

LEAKY_KEYS = ("action_subtype_id", "source_kind_id")


def hash_step(shard: dict[str, np.ndarray], t: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    for key in OBSERVATION_KEYS:
        a = shard[key][t]
        h.update(key.encode())
        h.update(b":")
        # Convert to a stable byte representation; np.ascontiguousarray would
        # promote 0-d to 1-d but that's fine for hashing.
        b = np.ascontiguousarray(a)
        h.update(b.tobytes())
        h.update(b"|")
    return h.hexdigest()


def iter_steps_for_video(video_id: str):
    partition = TENSORIZED_ROOT / f"video_id={video_id}"
    if not partition.exists():
        return
    for run_file in sorted(partition.glob("run_*.npz")):
        with np.load(run_file) as z:
            shard = {k: np.array(z[k]) for k in z.files}
        n = int(shard["target_action_id"].shape[0])
        for t in range(n):
            yield run_file, t, shard


def label_to_family(label: str) -> str:
    return label.split("_", 1)[0]


def main() -> None:
    print("loading splits + action map...")
    splits_payload = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    splits = splits_payload["splits"]
    train_videos = set(splits["train"]["video_ids"])
    val_videos = set(splits["val"]["video_ids"])
    test_videos = set(splits["test"]["video_ids"])

    action_map = json.loads(ACTION_MAP_PATH.read_text(encoding="utf-8"))
    index_to_label = list(action_map["index_to_label"])
    index_to_family = [label_to_family(l) for l in index_to_label]

    # ---------------------------------------------------------------
    # 1) Split sanity check
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("1) SPLIT SANITY")
    print("=" * 70)
    overlap_train_val = train_videos & val_videos
    overlap_train_test = train_videos & test_videos
    overlap_val_test = val_videos & test_videos
    print(f"  videos: train={len(train_videos)}  val={len(val_videos)}  "
          f"test={len(test_videos)}")
    print(f"  train & val:  {len(overlap_train_val)}  ({sorted(overlap_train_val)[:5]})")
    print(f"  train & test: {len(overlap_train_test)} ({sorted(overlap_train_test)[:5]})")
    print(f"  val & test:   {len(overlap_val_test)}  ({sorted(overlap_val_test)[:5]})")
    splits_ok = not (overlap_train_val or overlap_train_test or overlap_val_test)
    print(f"  splits OK?    {splits_ok}")

    # ---------------------------------------------------------------
    # 2) Observation duplicate rate
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("2) OBSERVATION DUPLICATE RATE")
    print("=" * 70)
    print("  hashing observation-only features per step "
          f"({len(OBSERVATION_KEYS)} channels)...")

    train_hashes: set[str] = set()
    train_hash_counts: collections.Counter = collections.Counter()
    train_steps = 0
    for vid in sorted(train_videos):
        for _, t, shard in iter_steps_for_video(vid):
            h = hash_step(shard, t)
            train_hashes.add(h)
            train_hash_counts[h] += 1
            train_steps += 1
    print(f"    train: {train_steps} steps, {len(train_hashes)} unique hashes "
          f"(intra-train dup rate = {1 - len(train_hashes)/max(train_steps,1):.4f})")

    def overlap_rate(videos: set[str], label: str) -> tuple[int, int, int]:
        steps = 0
        in_train = 0
        unique_hashes: set[str] = set()
        for vid in sorted(videos):
            for _, t, shard in iter_steps_for_video(vid):
                steps += 1
                h = hash_step(shard, t)
                unique_hashes.add(h)
                if h in train_hashes:
                    in_train += 1
        rate = in_train / max(steps, 1)
        unique = len(unique_hashes)
        print(
            f"    {label}: {steps} steps, {unique} unique  "
            f"observed in train = {in_train} ({rate:.4f})"
        )
        return steps, in_train, unique

    val_steps, val_in_train, val_unique = overlap_rate(val_videos, "val  ")
    test_steps, test_in_train, test_unique = overlap_rate(test_videos, "test ")

    # ---------------------------------------------------------------
    # 3) Family leakage from a single feature
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("3) FAMILY LEAKAGE FROM A SINGLE FEATURE")
    print("=" * 70)

    # For each leaky feature, fit majority-class-by-feature on train, then
    # evaluate top-1 family accuracy on val/test.
    def fit_and_score(feature_key: str, vocab_name: str) -> dict[str, Any]:
        # Train: feature_value -> Counter of family labels
        co_train: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        for vid in sorted(train_videos):
            for _, t, shard in iter_steps_for_video(vid):
                fv = int(shard[feature_key][t])
                aid = int(shard["target_action_id"][t])
                if aid < 0:
                    continue
                fam = index_to_family[aid]
                co_train[fv][fam] += 1
        # Majority family per feature value.
        rule = {fv: ctr.most_common(1)[0][0] for fv, ctr in co_train.items()}

        # Score on a split.
        def score_split(videos: set[str]) -> dict[str, float]:
            n = 0
            correct = 0
            for vid in sorted(videos):
                for _, t, shard in iter_steps_for_video(vid):
                    aid = int(shard["target_action_id"][t])
                    if aid < 0:
                        continue
                    fv = int(shard[feature_key][t])
                    pred_fam = rule.get(fv)
                    fam = index_to_family[aid]
                    if pred_fam == fam:
                        correct += 1
                    n += 1
            return {"n": n, "top1_family": correct / max(n, 1)}

        train_score = score_split(train_videos)
        val_score = score_split(val_videos)
        test_score = score_split(test_videos)

        # Also report per-family confidence: how concentrated is the mapping?
        top1_purity = []
        for fv, ctr in co_train.items():
            total = sum(ctr.values())
            top = ctr.most_common(1)[0][1]
            top1_purity.append(top / total)

        return {
            "feature": feature_key,
            "vocab_name": vocab_name,
            "n_distinct_values": len(co_train),
            "train": train_score,
            "val": val_score,
            "test": test_score,
            "mean_value_purity": float(np.mean(top1_purity)) if top1_purity else 0.0,
            "median_value_purity": float(np.median(top1_purity)) if top1_purity else 0.0,
        }

    leaky_results = []
    for key, vocab_name in [
        ("action_subtype_id", "action_subtype"),
        ("source_kind_id", "source_kind"),
    ]:
        print(f"  --- {key} ---")
        result = fit_and_score(key, vocab_name)
        leaky_results.append(result)
        print(f"    distinct values: {result['n_distinct_values']}")
        print(
            f"    family top-1 (majority rule):  "
            f"train={result['train']['top1_family']:.4f}  "
            f"val={result['val']['top1_family']:.4f}  "
            f"test={result['test']['top1_family']:.4f}"
        )
        print(
            f"    feature purity (mean/median): "
            f"{result['mean_value_purity']:.4f} / {result['median_value_purity']:.4f}"
        )

    # Random-baseline reference (uniform over 17 families).
    print(
        f"  reference: random-family baseline = {1/17:.4f}  "
        f"(17 action families)"
    )

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------
    report = {
        "splits": {
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "test_videos": len(test_videos),
            "overlap_train_val": sorted(overlap_train_val),
            "overlap_train_test": sorted(overlap_train_test),
            "overlap_val_test": sorted(overlap_val_test),
            "ok": splits_ok,
        },
        "observation_duplicate_rate": {
            "train_steps": train_steps,
            "train_unique": len(train_hashes),
            "val_steps": val_steps,
            "val_unique": val_unique,
            "val_overlap_with_train": val_in_train,
            "val_overlap_rate": val_in_train / max(val_steps, 1),
            "test_steps": test_steps,
            "test_unique": test_unique,
            "test_overlap_with_train": test_in_train,
            "test_overlap_rate": test_in_train / max(test_steps, 1),
        },
        "family_leakage_from_single_feature": leaky_results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"wrote {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
