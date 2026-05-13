#!/usr/bin/env python3
"""
inspect_selectcard.py
=====================
Print N SelectCard examples (default 100) with full context to surface
label-quality issues. For each sampled step we show:

- run path + step_id
- parent action (PlayHand / DiscardHand / UseConsumable / SelectPackItem)
- micro_index (which selection within the sub-sequence)
- candidate list (prev_selected + current_pool), each card rendered as
  ``rank suit (modifier/edition/seal)``
- the labeled ``target_index`` and which card it points at
- model top-5 predictions (action_label + probability) when a checkpoint
  is supplied
- baselines for comparison: leftmost-legal-pick, highest-rank-pick,
  random-legal-pick
- per row: granular ``step["state"]`` (OCR / UI-derived game scalars from the
  granularized step) and ``persistent_state`` **before this step**
  (`data/persistent_state/...`), filtered with ``state_reducer.to_model_visible``

Pulls from ``data/granularized`` for the rich human-readable card text and
from ``data/tensorized`` + the trained checkpoint for the model
distribution. Both sides are joined on ``(video_id, run_index, step_id)``.

Usage
-----
``python inspect_selectcard.py
    [--ckpt artifacts/checkpoints/best.pt]
    [--n 100] [--seed 0]
    [--split test]
    [--out artifacts/selectcard_samples.json]``
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from action_map import compute_action_map
from dataset import BalatroStepDataset, load_split
from model import PolicyTransformer, load_model_config
from tensorize import OCR_NUMERIC_KEYS, STATE_NUMERIC_KEYS
from state_reducer import MODEL_VISIBLE_KEYS, to_model_visible


SUITS = ("Spades", "Hearts", "Diamonds", "Clubs")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K")


SUIT_GLYPH = {"spades": "S", "hearts": "H", "diamonds": "D", "clubs": "C"}


def render_card(obj: dict[str, Any]) -> str:
    """Compact human label for a card-like object."""
    cm = obj.get("card") or {}
    if cm:
        rank = cm.get("rank", "?")
        suit = (cm.get("suit") or "").lower()
        glyph = SUIT_GLYPH.get(suit, "?")
        base = f"{rank}{glyph}"
    elif obj.get("class_id") == 78:
        base = "STONE"
    else:
        base = f"cls={obj.get('class_id')}"
    extras = []
    for key in ("modifier", "edition", "seal"):
        v = obj.get(key)
        if v:
            extras.append(v.replace("m_", "").replace("e_", ""))
    if extras:
        base += "(" + ",".join(extras) + ")"
    return base


def card_signature(obj: dict[str, Any]) -> tuple:
    """Equivalence key for a card. Two cards with the same signature are
    interchangeable from the player's perspective."""
    cm = obj.get("card") or {}
    return (
        obj.get("class_id"),
        obj.get("modifier"),
        obj.get("edition"),
        obj.get("seal"),
        cm.get("rank_index"),
        cm.get("suit_index"),
    )


def _json_safe(x: Any) -> Any:
    """Recursive convert for JSON (numpy scalars, non-finite floats)."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not (x == x):  # NaN
        return None
    if isinstance(x, float) and abs(x) == float("inf"):
        return None
    return x


def render_candidates(prev_selected: list[dict[str, Any]],
                      current_pool: list[dict[str, Any]]) -> list[str]:
    out = []
    for i, c in enumerate(prev_selected):
        out.append(f"[sel {i}] {render_card(c)}")
    for j, c in enumerate(current_pool):
        out.append(f"[cur {j}] {render_card(c)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--granularized", type=Path, default=Path("data/granularized"))
    ap.add_argument("--tensorized", type=Path, default=Path("data/tensorized"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--ckpt", type=Path, default=Path("artifacts/checkpoints/best.pt"))
    ap.add_argument("--action-config", type=Path,
                    default=Path("data/action_space_config.json"))
    ap.add_argument("--split", type=str, default="test",
                    choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/selectcard_samples.json"))
    ap.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("data/persistent_state"),
        help="Directory with persistent_state shards (parallel to granularized)",
    )
    ap.add_argument(
        "--full-tracked-deck",
        action="store_true",
        help=(
            "Export every tracked deck card JSON in each row (large file). "
            "Default exports count + truncated preview."
        ),
    )
    ap.add_argument("--persistent-deck-preview", type=int, default=12)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    action_map = compute_action_map(
        json.loads(args.action_config.read_text(encoding="utf-8"))
    )
    label_to_index = action_map["label_to_index"]
    index_to_label = list(action_map["index_to_label"])
    selectcard_offset = action_map["family_offsets"]["SelectCard"]
    selectcard_size = action_map["family_sizes"]["SelectCard"]

    # ---- Build a granularized index keyed by (video_id, run_index, step_id) ----
    print("indexing granularized SelectCard steps...")
    split_videos = set(load_split(args.splits, args.split))
    select_steps: list[tuple[str, int, dict[str, Any]]] = []
    for partition in sorted(args.granularized.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        vid = partition.name.split("=", 1)[1]
        if vid not in split_videos:
            continue
        for run_file in sorted(partition.glob("run_*.json")):
            run = json.loads(run_file.read_text(encoding="utf-8"))
            run_index = run.get("run_index", 0)
            for step in run.get("events", []):
                action = step.get("action") or ""
                if not action.startswith("SelectCard"):
                    continue
                select_steps.append((vid, run_index, step))
    print(f"  found {len(select_steps)} SelectCard steps in '{args.split}'")

    # ---- Load tensorized dataset for matching predictions ----
    print("loading tensorized split...")
    ds = BalatroStepDataset(
        args.tensorized,
        split_videos=sorted(split_videos),
        include_unresolved=True,
        device=device,
    )
    n_actions = ds.n_actions()

    # Load checkpoint and build model.
    have_model = args.ckpt.exists()
    model: PolicyTransformer | None = None
    if have_model:
        print(f"loading checkpoint {args.ckpt.as_posix()}...")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg_dict = ckpt["model_config"]
        cfg = load_model_config(
            n_actions=ckpt["n_actions"],
            d_model=cfg_dict["d_model"],
            n_layers=cfg_dict["n_layers"],
            n_heads=cfg_dict["n_heads"],
            dim_feedforward=cfg_dict["dim_feedforward"],
            dropout=cfg_dict["dropout"],
        )
        model = PolicyTransformer(cfg).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"  (no checkpoint at {args.ckpt}; skipping predictions)")

    # ---- Sample N steps deterministically ----
    rng = random.Random(args.seed)
    rng.shuffle(select_steps)
    sample = select_steps[: args.n]

    # ---- Build a mapping: for each sampled (vid, run, step_id), find the
    #      flat index inside the dataset (which mirrors the granularized
    #      iteration order). ----
    print("matching tensorized samples by (video_id, run_index, step_id)...")
    # We rebuild the same "valid_indices" iteration order the dataset used
    # so we can index back into ds. The dataset preserves video_id-sorted +
    # run-sorted + per-run step order, but it skips unresolved steps unless
    # include_unresolved=True (which we set above). Build a coordinate map.
    coord_to_flat: dict[tuple[str, int, int], int] = {}
    flat = 0
    for partition in sorted(args.tensorized.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        vid = partition.name.split("=", 1)[1]
        if vid not in split_videos:
            continue
        for run_file in sorted(partition.glob("run_*.npz")):
            with np.load(run_file) as z:
                n = int(z["target_action_id"].shape[0])
            # We don't have step_id in the tensorized file, so we rely on the
            # fact that granularize emits step_id == position-in-events for
            # this run. Verify alignment by also iterating granularized.
            run_index = int(run_file.stem.split("_")[-1])
            for t in range(n):
                coord_to_flat[(vid, run_index, t)] = flat
                flat += 1

    # Materialize predictions in batches.
    flats: list[int] = []
    matched_samples: list[tuple[str, int, dict[str, Any], int]] = []
    for vid, run_idx, step in sample:
        sid = step.get("step_id")
        if sid is None:
            continue
        key = (vid, int(run_idx), int(sid))
        f = coord_to_flat.get(key)
        if f is None:
            continue
        flats.append(f)
        matched_samples.append((vid, int(run_idx), step, f))

    print(f"  matched {len(matched_samples)} of {len(sample)} samples to tensorized rows")

    feature_config_payload: dict[str, Any] | None = None
    fc_path = Path("artifacts/feature_config.json")
    if fc_path.is_file():
        feature_config_payload = json.loads(fc_path.read_text(encoding="utf-8"))

    run_persistent_cache: dict[tuple[str, int], dict[str, Any] | bool] = {}

    def persistent_snapshot_before_step(
        vid_: str, run_idx_: int, step_id_: int,
    ) -> dict[str, Any] | None:
        kk = (vid_, run_idx_)
        if kk not in run_persistent_cache:
            pf = (
                args.persistent_root
                / f"video_id={vid_}"
                / f"run_{int(run_idx_):03d}.json"
            )
            run_persistent_cache[kk] = (
                json.loads(pf.read_text(encoding="utf-8"))
                if pf.is_file()
                else False
            )
        blob = run_persistent_cache[kk]
        if blob is False:
            return None
        states_list = blob.get("states") or []
        if not (0 <= step_id_ < len(states_list)):
            return None
        return states_list[step_id_]

    persistent_missing_steps = 0

    training_sample_included: dict[str, Any] = {
        "persistent_state_alignment": (
            "Each tensorized timestep t aligns granularized run.events[t] with "
            "persistent_state.states[t]: the reducer snapshot BEFORE applying "
            "that step (same convention as tensorize.py)."
        ),
        "granular_game_state": {
            "description": (
                "`step[\"state\"]` from granularized JSON: OCR-derived HUD scalars "
                "tensorized as ocr_numeric (normalized) plus ocr_valid."
            ),
            "canonical_ocr_numeric_keys": list(OCR_NUMERIC_KEYS),
        },
        "persistent_state_used_by_tensorizer": {
            "description": (
                "Persistent reducer dict BEFORE this step contributes state_numeric "
                "counters (skips, hands_played, ...), flags, per-hand arrays, voucher/boss "
                "multi-hots, padded tracked deck channels, deck/stake categorical ids, "
                "etc.; see tensorize.tensorize_step."
            ),
            "subset_exported_below_per_row": "state_reducer.to_model_visible(snapshot)",
            "model_visible_persistent_keys_sorted": sorted(MODEL_VISIBLE_KEYS),
            "persistent_state_numeric_keys_used_in_tensors": list(STATE_NUMERIC_KEYS),
        },
        "per_step_object_tokens": (
            "Padded object_* tensors from granularized objects plus reconstructed "
            "hand/pool (current_hand_or_pack + selected_cards); see "
            "tensorize._merge_hand_into_objects."
        ),
        "action_supervision_channels": ["action_mask", "target_action_id"],
        "tensor_to_policy_model_note": (
            "Tensors store source_kind_id and action_subtype_id for auditing; "
            "PolicyTransformer GlobalEncoder excludes them during forward (leak-free BC)."
        ),
        "selectcard_export_note": (
            "By default `persistent_state_before_step_model_visible` truncates "
            "`tracked_deck_cards` (see --full-tracked-deck / "
            "--persistent-deck-preview) to keep this JSON small; full reducer "
            "snapshots remain on disk under data/persistent_state/."
        ),
    }
    if feature_config_payload:
        training_sample_included["caps_from_feature_config"] = {
            "MAX_OBJECTS_PER_STEP": feature_config_payload.get("MAX_OBJECTS_PER_STEP"),
            "TRACKED_DECK_CAP": feature_config_payload.get("TRACKED_DECK_CAP"),
        }

    # Compute model probabilities for the matched samples in one pass.
    sample_probs: list[np.ndarray] = []
    if model is not None and matched_samples:
        with torch.no_grad():
            tensors = ds.all_tensors()
            flat_tensor = torch.tensor(flats, device=device, dtype=torch.long)
            batch = {k: v.index_select(0, flat_tensor) for k, v in tensors.items()}
            logits = model(batch)  # (N, N_ACTIONS), illegal masked to -inf
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        sample_probs = [probs[i] for i in range(probs.shape[0])]

    # ---- Per-sample analysis + baseline scoring ----
    rows: list[dict[str, Any]] = []
    n_top1 = 0
    n_top3 = 0
    n_top5 = 0
    n_top1_eqv = 0          # equivalence-aware: any card with same signature counts
    n_top3_eqv = 0
    n_baseline_left = 0
    n_baseline_random = 0
    n_baseline_highest_rank = 0
    n_unique_target = 0      # samples where the target card is unique in its pool
    n_ambiguous = 0          # samples with >=2 equivalent candidates
    equiv_count_dist: Counter = Counter()
    family_correct_summary = Counter()
    family_total_summary = Counter()

    for k, (vid, run_idx, step, f) in enumerate(matched_samples):
        action = step["action"]
        target_index = step.get("target_index")
        prev_selected = step.get("selected_cards") or []
        current_pool = step.get("current_hand_or_pack") or []
        candidates = prev_selected + current_pool
        n_cand = len(candidates)
        if target_index is None or not (0 <= target_index < n_cand):
            continue

        target_card = candidates[target_index]
        family = "SelectCard"

        # Equivalence-aware: which candidate indices have the same signature
        # as the target?
        target_sig = card_signature(target_card)
        equivalent_indices = [
            j for j, c in enumerate(candidates) if card_signature(c) == target_sig
        ]
        equiv_count_dist[len(equivalent_indices)] += 1
        if len(equivalent_indices) == 1:
            n_unique_target += 1
        else:
            n_ambiguous += 1

        # Baselines.
        leftmost_pred = 0
        random_pred = random.Random(args.seed + k).randrange(n_cand)
        # Highest-rank baseline among the unselected pool only.
        ranks = []
        for j, c in enumerate(current_pool):
            cm = c.get("card") or {}
            ri = cm.get("rank_index")
            ranks.append((ri if isinstance(ri, int) else -1, j))
        ranks.sort(reverse=True)
        if ranks:
            highest_pred = len(prev_selected) + ranks[0][1]
        else:
            highest_pred = leftmost_pred

        if leftmost_pred == target_index:
            n_baseline_left += 1
        if random_pred == target_index:
            n_baseline_random += 1
        if highest_pred == target_index:
            n_baseline_highest_rank += 1

        # Model top-5.
        top5_labels: list[tuple[str, float]] = []
        model_pred_index: int | None = None
        eqv_top1_hit = False
        eqv_top3_hit = False
        if sample_probs:
            p = sample_probs[k]
            # restrict to SelectCard family slot range for clarity
            sc_p = p[selectcard_offset : selectcard_offset + selectcard_size]
            order = np.argsort(-sc_p)[:5]
            for idx_local in order:
                top5_labels.append((f"SelectCard_{int(idx_local)}", float(sc_p[idx_local])))
            model_pred_index = int(np.argmax(sc_p))
            top3_set = {int(x) for x in order[:3]}
            top5_set = {int(x) for x in order}
            if model_pred_index == target_index:
                n_top1 += 1
                family_correct_summary[family] += 1
            if target_index in top3_set:
                n_top3 += 1
            if target_index in top5_set:
                n_top5 += 1
            # Equivalence-aware: counts as correct if model picks any of the
            # interchangeable cards with the same signature.
            if model_pred_index in equivalent_indices:
                n_top1_eqv += 1
                eqv_top1_hit = True
            if any(j in top3_set for j in equivalent_indices):
                n_top3_eqv += 1
                eqv_top3_hit = True
            family_total_summary[family] += 1

        persistent_before = persistent_snapshot_before_step(vid, int(run_idx), int(step["step_id"]))
        if persistent_before is None:
            persistent_missing_steps += 1
        pv = None
        if persistent_before is not None:
            vis = _json_safe(to_model_visible(persistent_before))
            td = vis.get("tracked_deck_cards")
            if args.full_tracked_deck or not isinstance(td, list):
                pv = vis
            else:
                n_td = len(td)
                prv = args.persistent_deck_preview
                clipped = prv > 0 and n_td > prv
                pv = {**vis, "tracked_deck_card_count": n_td}
                if clipped:
                    pv["tracked_deck_cards_preview_first_n"] = prv
                    pv["tracked_deck_cards"] = td[:prv]
                else:
                    pv["tracked_deck_cards"] = td

        rows.append({
            "video_id": vid,
            "run_index": run_idx,
            "step_id": step.get("step_id"),
            "action": action,
            "persistent_state_reference": (
                f"video_id={vid}/run_{int(run_idx):03d}.json#states[{step.get('step_id')}] "
                "(BEFORE applying this granularized step)"
            ),
            "granular_game_state": _json_safe(step.get("state") or {}),
            "persistent_state_before_step_model_visible": pv,
            "parent_action": step.get("source_action"),
            "parent_subtype": step.get("source_action_subtype"),
            "page_name": step.get("page_name"),
            "micro_index": step.get("micro_index"),
            "n_candidates": n_cand,
            "target_index": int(target_index),
            "target_card": render_card(target_card),
            "candidates": render_candidates(prev_selected, current_pool),
            "equivalent_indices": equivalent_indices,
            "n_equivalent": len(equivalent_indices),
            "model_top5_selectcard": [
                {"label": l, "p": p} for l, p in top5_labels
            ],
            "model_pred_index": model_pred_index,
            "model_top1_eqv_correct": eqv_top1_hit,
            "model_top3_eqv_correct": eqv_top3_hit,
            "baselines": {
                "leftmost_pred": leftmost_pred,
                "leftmost_correct": leftmost_pred == target_index,
                "random_pred": random_pred,
                "random_correct": random_pred == target_index,
                "highest_rank_pred": highest_pred,
                "highest_rank_correct": highest_pred == target_index,
            },
        })

    # ---- Print summary + first 30 examples ----
    n_eval = len(rows)
    print()
    print("=" * 78)
    print(f"SelectCard label-quality sample ({args.split}, n={n_eval})")
    print("=" * 78)
    if n_eval == 0:
        print("no rows to evaluate"); return

    candidate_count_dist = Counter(r["n_candidates"] for r in rows)
    print()
    print("--- candidate-count distribution ---")
    for cnt, n in sorted(candidate_count_dist.items()):
        print(f"  {cnt:2d} candidates  -> {n} samples (random baseline = {1/cnt:.4f})")
    avg_cands = sum(r["n_candidates"] for r in rows) / n_eval
    avg_random = sum(1 / r["n_candidates"] for r in rows) / n_eval
    print(f"  avg candidates: {avg_cands:.2f}   avg random baseline: {avg_random:.4f}")

    print()
    print("--- equivalence-class diagnostic ---")
    print(f"  unique-target samples (only 1 candidate matches signature): "
          f"{n_unique_target}/{n_eval} = {n_unique_target/n_eval:.4f}")
    print(f"  ambiguous samples     (>=2 equivalent candidates):           "
          f"{n_ambiguous}/{n_eval} = {n_ambiguous/n_eval:.4f}")
    for k_, n in sorted(equiv_count_dist.items()):
        print(f"    {k_:2d} equivalent candidates: {n}")

    print()
    print("--- baselines ---")
    print(f"  leftmost-pick   correct: {n_baseline_left}/{n_eval} = {n_baseline_left/n_eval:.4f}")
    print(f"  highest-rank    correct: {n_baseline_highest_rank}/{n_eval} = {n_baseline_highest_rank/n_eval:.4f}")
    print(f"  random-pick     correct: {n_baseline_random}/{n_eval} = {n_baseline_random/n_eval:.4f}")
    if model is not None:
        print(f"  MODEL top-1     correct: {n_top1}/{n_eval} = {n_top1/n_eval:.4f}")
        print(f"  MODEL top-3     correct: {n_top3}/{n_eval} = {n_top3/n_eval:.4f}")
        print(f"  MODEL top-5     correct: {n_top5}/{n_eval} = {n_top5/n_eval:.4f}")
        print(f"  MODEL top-1 EQV correct: {n_top1_eqv}/{n_eval} = {n_top1_eqv/n_eval:.4f}  "
              f"(any equivalent card)")
        print(f"  MODEL top-3 EQV correct: {n_top3_eqv}/{n_eval} = {n_top3_eqv/n_eval:.4f}")

    parent_breakdown = Counter(r["parent_action"] for r in rows)
    print()
    print("--- parent_action breakdown ---")
    for k_, n in parent_breakdown.most_common():
        print(f"  {str(k_):30s} {n} samples")

    print()
    print("--- first 30 rows ---")
    for i, r in enumerate(rows[:30]):
        print(
            f"\n[{i:3d}] {r['video_id']}/run{r['run_index']:02d}/step{r['step_id']}  "
            f"parent={r['parent_action']}({r['parent_subtype']})  "
            f"micro={r['micro_index']}  page={r['page_name']}  "
            f"target_index={r['target_index']}  -> {r['target_card']}  "
            f"({r['n_candidates']} candidates)"
        )
        for c in r["candidates"]:
            print(f"      {c}")
        if r["model_top5_selectcard"]:
            top = ", ".join(f"{x['label']}={x['p']:.3f}" for x in r["model_top5_selectcard"])
            tag = "OK " if r["model_pred_index"] == r["target_index"] else "x  "
            print(f"      MODEL[{tag}] top5: {top}")
        b = r["baselines"]
        print(
            f"      base: leftmost={b['leftmost_pred']}({'+'if b['leftmost_correct'] else '-'})  "
            f"random={b['random_pred']}({'+'if b['random_correct'] else '-'})  "
            f"highest_rank={b['highest_rank_pred']}({'+'if b['highest_rank_correct'] else '-'})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "training_sample_included": training_sample_included,
        "selectcard_persistent_export": {
            "full_tracked_deck": bool(args.full_tracked_deck),
            "persistent_deck_preview_cards": (
                args.persistent_deck_preview if not args.full_tracked_deck else None
            ),
        },
        "persistent_state_missing_in_rows": persistent_missing_steps,
        "persistent_state_root": str(args.persistent_root.as_posix()),
        "split": args.split,
        "n_samples": n_eval,
        "avg_candidates": avg_cands,
        "avg_random_baseline": avg_random,
        "baselines": {
            "leftmost_top1": n_baseline_left / n_eval,
            "highest_rank_top1": n_baseline_highest_rank / n_eval,
            "random_top1": n_baseline_random / n_eval,
        },
        "model": {
            "top1": n_top1 / n_eval if model is not None else None,
            "top3": n_top3 / n_eval if model is not None else None,
            "top5": n_top5 / n_eval if model is not None else None,
            "top1_equivalence_aware": n_top1_eqv / n_eval if model is not None else None,
            "top3_equivalence_aware": n_top3_eqv / n_eval if model is not None else None,
        },
        "ambiguity": {
            "n_unique_target": n_unique_target,
            "n_ambiguous": n_ambiguous,
            "equivalent_count_distribution": dict(sorted(equiv_count_dist.items())),
        },
        "parent_action_breakdown": dict(parent_breakdown),
        "candidate_count_distribution": dict(sorted(candidate_count_dist.items())),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"wrote {args.out.as_posix()}")


if __name__ == "__main__":
    main()
