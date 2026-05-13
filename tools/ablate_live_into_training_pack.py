#!/usr/bin/env python3
"""
Progressively merge a LIVE In_JokerStandardPlanet_Pack snapshot into a fixed
training (SelectPackItem) step, tensorize each blend, run the checkpoint, and
measure P(SkipPack). Uses coarse monotone prefix-search on three buckets
(objects, step_state, persistent_state); optional refinement on
persistent_state top-level JSON keys.

Both halves now share the granularize-3.0 step shape — there is no separate
``current_hand_or_pack`` / ``selected_cards`` projection any more. Pending
playing cards live as ``zone == "PendingCards"`` entries inside ``objects``,
and the per-card ``pending_cards`` view rides along at the top level for
``state_reducer`` compatibility (which the live encoder ignores).

Examples
--------
python tools/ablate_live_into_training_pack.py \\
    --live "%APPDATA%\\\\Balatro\\\\agent_io\\\\snapshots_debug\\\\latest_In_JokerStandardPlanet_Pack.json"

python tools/ablate_live_into_training_pack.py \\
    --granular-video 2426724416 --granular-run 003 --granular-step 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map
from live.live_encoder import _normalize_legacy_snapshot
from model import ModelConfig, PolicyTransformer
from tensorize import Normalizer, VocabLookup, tensorize_step

COARSE_GROUPS = ("objects", "step_state", "persistent_state")


def default_live_snap() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", ""))
        return (
            base
            / "Balatro"
            / "agent_io"
            / "snapshots_debug"
            / "latest_In_JokerStandardPlanet_Pack.json"
        )
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library/Application Support/Balatro/agent_io/snapshots_debug/latest_In_JokerStandardPlanet_Pack.json"
        )
    return (
        Path.home()
        / ".local/share/love/Balatro/agent_io/snapshots_debug/latest_In_JokerStandardPlanet_Pack.json"
    )


def load_artifacts(repo: Path) -> tuple[VocabLookup, Normalizer, dict[str, Any], dict[str, Any]]:
    vocab = VocabLookup(
        json.loads((repo / "artifacts/vocab.json").read_text(encoding="utf-8"))
    )
    norm = Normalizer(
        json.loads((repo / "artifacts/normalization.json").read_text(encoding="utf-8"))
    )
    feat = json.loads((repo / "artifacts/feature_config.json").read_text(encoding="utf-8"))
    action_cfg = json.loads(
        (repo / "data/action_space_config.json").read_text(encoding="utf-8")
    )
    return vocab, norm, feat, action_cfg


def step_from_live_blob(live: dict[str, Any]) -> dict[str, Any]:
    """Project a live snapshot into the canonical granularize-3.0 step shape.

    Accepts both ``live/2.0.0`` (canonical) and legacy ``live/1.0.0`` blobs;
    the latter are first translated through ``_normalize_legacy_snapshot``.
    """
    if live.get("schema_version") != "live/2.0.0":
        live = _normalize_legacy_snapshot(live)
    return {
        "page_name": live.get("page_name"),
        "source_kind": None,
        "action_subtype": None,
        "state": live.get("state") or {},
        "objects": copy.deepcopy(live.get("objects")) or [],
        "pending_cards": copy.deepcopy(live.get("pending_cards")) or [],
        "target_zone": None,
        "target_position": None,
    }


def looks_like_initial_pack_choice(ev: dict[str, Any]) -> bool:
    """Heuristic: first tap in an indexed SelectPack chain (often closer to UI open)."""
    return (ev.get("micro_index") or 0) == 0


def find_training_pick(
    repo: Path,
    *,
    video_id: str | None,
    run_idx: str | None,
    step_id: int | None,
    require_select_pack_item: bool,
    prefer_initial_pack_offer: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    granular_root = repo / "data/granularized"
    pstate_root = repo / "data/persistent_state"

    partitions = sorted(
        p for p in granular_root.iterdir() if p.is_dir() and p.name.startswith("video_id=")
    )
    for partition in partitions:
        vid = partition.name.split("=", 1)[1]
        if video_id is not None and vid != video_id:
            continue
        for run_file in sorted(partition.glob("run_*.json")):
            run_suffix = run_file.name  # run_003.json
            if run_idx is not None and run_file.stem != f"run_{run_idx}":
                continue
            d = json.loads(run_file.read_text(encoding="utf-8"))
            for ev in d.get("events") or []:
                if ev.get("page_name") != "In_JokerStandardPlanet_Pack":
                    continue
                action = ev.get("action") or ""
                if require_select_pack_item and not action.startswith("SelectPackItem"):
                    continue
                if prefer_initial_pack_offer and not looks_like_initial_pack_choice(ev):
                    continue
                if step_id is not None and ev.get("step_id") != step_id:
                    continue
                ps_path = pstate_root / f"video_id={vid}" / run_suffix
                if not ps_path.exists():
                    continue
                states = (json.loads(ps_path.read_text(encoding="utf-8"))).get("states") or []
                sid = ev.get("step_id", 0)
                if 0 <= sid < len(states):
                    ev_out = copy.deepcopy(ev)
                    ev_out["video_id"] = vid
                    return ev_out, copy.deepcopy(states[sid])

    hint = (
        'First pack-pick rows can be labeled SkipPack (previous micro-step); pass a SelectPackItem '
        'step_id. Toggle --relax-pick-filter to include non-picks. Toggle '
        '--prefer-initial-pack-offer to require a fresher OfferingsSelected-free row.'
    )
    raise SystemExit(f"no qualifying training pack-pick step found ({hint})")


def blended_step_persistent(
    train_step: dict[str, Any],
    train_ps: dict[str, Any],
    live_step: dict[str, Any],
    live_ps: dict[str, Any],
    *,
    live_groups: set[str],
    pstate_live_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Start from train; overlay live subtrees listed in ``live_groups``.

    If ``persistent_state`` is live and ``pstate_live_keys`` is a set, copy only
    those top-level keys from ``live_ps`` (rest stay from train). If
    ``pstate_live_keys`` is None and ``persistent_state`` is live, replace the
    entire persistent_state dict.
    """
    step = copy.deepcopy(train_step)
    ps = copy.deepcopy(train_ps)

    if "step_state" in live_groups:
        step["state"] = copy.deepcopy(live_step.get("state") or {})
    if "objects" in live_groups:
        step["objects"] = copy.deepcopy(live_step.get("objects")) or []
        step["pending_cards"] = copy.deepcopy(live_step.get("pending_cards")) or []
        step["target_zone"] = None
        step["target_position"] = None
    if "persistent_state" in live_groups:
        if pstate_live_keys is None:
            ps = copy.deepcopy(live_ps)
        else:
            for k in pstate_live_keys:
                if k in live_ps:
                    ps[k] = copy.deepcopy(live_ps[k])
    return step, ps


def np_record_to_batch(rec: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for k, arr in rec.items():
        if k == "target_action_id":
            continue
        t = torch.as_tensor(np.asarray(arr))
        if t.dtype == torch.float64:
            t = t.float()
        if t.ndim == 0:
            t = t.unsqueeze(0)
        else:
            t = t.unsqueeze(0)
        batch[k] = t.to(device)
    return batch


def load_model(repo: Path, ckpt_path: Path, device: torch.device) -> PolicyTransformer:
    ckpt_path = ckpt_path if ckpt_path.is_absolute() else (repo / ckpt_path)
    ckpt = torch.load(
        ckpt_path.resolve(),
        map_location=device,
        weights_only=False,
    )
    cfg_in = ckpt.get("model_config")
    if cfg_in is None:
        raise RuntimeError("checkpoint missing model_config")
    cfg = ModelConfig(
        **{k: v for k, v in cfg_in.items() if k in ModelConfig.__dataclass_fields__}
    )
    model = PolicyTransformer(cfg).to(device).eval()
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    model.load_state_dict(state)
    return model


class Predictor:
    def __init__(
        self,
        *,
        vocab: VocabLookup,
        norm: Normalizer,
        feat: dict[str, Any],
        amap: dict[str, Any],
        model: PolicyTransformer,
        device: torch.device,
        skip_id: int,
    ) -> None:
        self.vocab = vocab
        self.norm = norm
        self.feat = feat
        self.amap = amap
        self.model = model
        self.device = device
        self.skip_id = skip_id
        lbl = list(amap["index_to_label"])
        self.idx_to_label = lbl

    @torch.no_grad()
    def __call__(self, step: dict[str, Any], ps: dict[str, Any]) -> dict[str, Any]:
        rec = tensorize_step(step, ps, self.amap, self.vocab, self.norm, self.feat)
        bat = np_record_to_batch(rec, self.device)
        logits = self.model(bat)
        probs = torch.softmax(logits, dim=-1)[0]
        sid = self.skip_id
        p_skip = float(probs[sid].item())
        pred = int(torch.argmax(logits, dim=-1)[0].item())
        mask = bat["action_mask"][0].bool()
        denom = probs[mask].sum().clamp_min(1e-12)
        p_skip_masked = float((probs[sid] / denom).item()) if mask[sid] else 0.0
        legal = {"SkipPack"} if mask[sid].item() else set()
        for i in range(sid + 1, min(len(mask), sid + 6)):
            lbl = self.idx_to_label[i] if i < len(self.idx_to_label) else str(i)
            if mask[i]:
                legal.add(lbl)
        out = {
            "p_skip": p_skip,
            "p_skip_masked_cond_legal": p_skip_masked,
            "pred_id": pred,
            "pred_label": self.idx_to_label[pred] if pred < len(self.idx_to_label) else str(pred),
            "argmax_skip": pred == sid,
            "mask_skip_legal": bool(mask[sid].item()),
            "tensorize_target_id": int(rec["target_action_id"].item()),
        }
        pack_items = [(i, self.idx_to_label[i], float(probs[i].item())) for i in range(len(mask)) if mask[i] and self.idx_to_label[i].startswith("SelectPackItem_")]
        out["pack_item_probs_preview"] = sorted(pack_items, key=lambda t: -t[2])[:6]
        return out


def _fmt_pred(d: dict[str, Any]) -> str:
    return (
        f"p_skip={d['p_skip']:.4f} p_skip|(legal)={d['p_skip_masked_cond_legal']:.4f} "
        f"argmax={d['pred_label']} argmax_skip={d['argmax_skip']}"
    )


def _make_hit_fn(args: argparse.Namespace):
    mode = args.hit_mode
    if mode == "thr_argmax":
        return lambda r: r["p_skip"] >= args.thr and r["argmax_skip"]
    if mode == "argmax_skip":
        return lambda r: r["argmax_skip"]
    if mode == "high_p_skip":
        return lambda r: r["p_skip"] >= args.thr
    raise ValueError(mode)


def _describe_object(obj: dict[str, Any]) -> str:
    zone = obj.get("zone")
    pos = obj.get("position_in_zone")
    cid = obj.get("class_id")
    typ = obj.get("object_type")
    stickers = ",".join(obj.get("stickers") or [])
    sticker_part = f" stickers={stickers}" if stickers else ""
    return f"zone={zone} pos={pos} class={cid} type={typ}{sticker_part}"


def _object_zones(objects: list[dict[str, Any]]) -> list[str]:
    zones = sorted({str(o.get("zone")) for o in objects if isinstance(o, dict)})
    return zones


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--live", type=Path, default=None, help="Path to Live JSON snapshot blob")
    ap.add_argument("--ckpt", type=Path, default=Path("artifacts/checkpoints/best.pt"))
    ap.add_argument("--granular-video", type=str, default=None)
    ap.add_argument("--granular-run", type=str, default=None, help='e.g. "003" (no prefix)')
    ap.add_argument("--granular-step", type=int, default=None)
    ap.add_argument(
        "--hit-mode",
        choices=("thr_argmax", "argmax_skip", "high_p_skip"),
        default="thr_argmax",
        help=(
            "thr_argmax: p_skip>=--thr and argmax SkipPack. "
            "argmax_skip: argmax only. high_p_skip: p_skip>=--thr only."
        ),
    )
    ap.add_argument(
        "--thr",
        type=float,
        default=0.9,
        help="Probability threshold used by thr_argmax / high_p_skip hit-modes.",
    )
    ap.add_argument(
        "--prefer-initial-pack-offer",
        action="store_true",
        help=(
            "Prefer the first granular micro-step in a SelectPack decomposition "
            "(micro_index==0). Still often includes PackOfferingsSelected rows from the dataset."
        ),
    )
    ap.add_argument(
        "--relax-pick-filter",
        action="store_true",
        help=(
            "Allow non-SelectPackItem granular rows. Default requires action "
            "SelectPackItem_* so the baseline is not a SkipPack micro-step."
        ),
    )
    ap.add_argument(
        "--prefix-order",
        type=str,
        default="objects,step_state,persistent_state",
        help="Comma order used for monotone prefix search",
    )
    args = ap.parse_args(argv)
    hit_fn = _make_hit_fn(args)

    repo = args.root.resolve()
    live_path = (args.live or default_live_snap()).expanduser().resolve()

    vocab, norm, feat, action_cfg = load_artifacts(repo)
    amap = compute_action_map(action_cfg)
    skip_id = int(amap["family_offsets"]["SkipPack"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = load_model(repo, args.ckpt, device)
    pred = Predictor(
        vocab=vocab,
        norm=norm,
        feat=feat,
        amap=amap,
        model=model,
        device=device,
        skip_id=skip_id,
    )

    train_ev, train_ps = find_training_pick(
        repo,
        video_id=args.granular_video,
        run_idx=args.granular_run,
        step_id=args.granular_step,
        require_select_pack_item=not args.relax_pick_filter,
        prefer_initial_pack_offer=args.prefer_initial_pack_offer,
    )

    if not live_path.exists():
        raise SystemExit(f"live snapshot missing: {live_path}")

    live_snap = json.loads(live_path.read_text(encoding="utf-8"))
    live_step_tpl = step_from_live_blob(live_snap)
    live_ps_full = live_snap.get("persistent_state") or {}

    baseline_step = copy.deepcopy(train_ev)
    baseline_ps = copy.deepcopy(train_ps)

    print("=" * 78)
    print("BASELINE - training pick (full granular step + persistent BEFORE)")
    print(f"  granular: video_id={train_ev.get('video_id', '?')} "
          f"step_id={train_ev.get('step_id')} action={train_ev.get('action')}")
    print("=" * 78)
    b0 = pred(baseline_step, baseline_ps)
    print(_fmt_pred(b0))

    lone_step = {
        **step_from_live_blob(live_snap),
        "action": "",
        "source_kind": live_snap.get("source_kind"),
        "action_subtype": live_snap.get("action_subtype"),
    }
    print()
    print("PURE LIVE (tensorize live snapshot standalone, action cleared)")
    rl = pred(lone_step, copy.deepcopy(live_ps_full))
    print(_fmt_pred(rl))

    lf_step, lf_ps = blended_step_persistent(
        baseline_step,
        baseline_ps,
        live_step_tpl,
        live_ps_full,
        live_groups={"objects", "step_state", "persistent_state"},
    )
    print()
    print("=" * 78)
    print("FULL LIVE BLEND - objects, step_state, persistent_state from live")
    print("=" * 78)
    b_all = pred(lf_step, lf_ps)
    print(_fmt_pred(b_all))

    coarse_order = tuple(s.strip() for s in args.prefix_order.split(",") if s.strip())
    unknown = set(coarse_order) - set(COARSE_GROUPS)
    if unknown:
        raise SystemExit(f"unknown coarse groups {unknown}")

    def coarse_mask(prefix_len: int) -> set[str]:
        return set(coarse_order[: max(0, min(prefix_len, len(coarse_order))) ])

    def evaluate_prefix(k: int, *, label: str = "") -> tuple[dict[str, Any], set[str]]:
        live_g = coarse_mask(k)
        step_x, ps_x = blended_step_persistent(
            baseline_step,
            baseline_ps,
            live_step_tpl,
            live_ps_full,
            live_groups=live_g,
        )
        r = pred(step_x, ps_x)
        tag = hit_fn(r)
        extra = f" {label}" if label else ""
        print(
            f"  prefix_len={k} live_groups={sorted(live_g)!r}: {_fmt_pred(r)} "
            f"[{'HIT' if tag else '.'}]{extra}"
        )
        return r, live_g

    print()
    print("=" * 78)
    print(f"COARSE PREFIX SEARCH (order={coarse_order!r})")
    print(f"  HIT predicate: hit_mode={args.hit_mode} thr={args.thr}")
    print("=" * 78)

    r_full, g_full = evaluate_prefix(len(coarse_order), label="(full prefix)")
    if not hit_fn(r_full):
        print()
        print("[warn] Full live blend on this prefix order does not HIT; singleton probes:")
        for g in coarse_order:
            step_x, ps_x = blended_step_persistent(
                baseline_step,
                baseline_ps,
                live_step_tpl,
                live_ps_full,
                live_groups={g},
            )
            rr = pred(step_x, ps_x)
            print(f"    only {g!r}: {_fmt_pred(rr)} [{'HIT' if hit_fn(rr) else '.'}]")
        print("[note] Lower --thr or try --granular-* to pick another training step.")
        return

    r_zero, _ = evaluate_prefix(0, label="(all train obs)")

    if hit_fn(r_zero):
        print()
        print("[note] Baseline already strongly prefers SkipPack; pick a different training row.")
        return

    cached: dict[int, tuple[dict[str, Any], set[str]]] = {
        0: (r_zero, coarse_mask(0)),
        len(coarse_order): (r_full, g_full),
    }

    def peek(k: int) -> tuple[dict[str, Any], set[str]]:
        if k not in cached:
            cached[k] = evaluate_prefix(k)
        return cached[k]

    hi = len(coarse_order)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if hit_fn(peek(mid)[0]):
            hi = mid
        else:
            lo = mid + 1

    k_star = lo
    groups_hit = coarse_mask(k_star)
    culprit = coarse_order[k_star - 1] if k_star > 0 else None
    groups_prev = coarse_mask(max(0, k_star - 1))

    print()
    print(
        f"Smallest prefix_len={k_star} live_groups={sorted(groups_hit)!r} "
        f"(newest group in prefix: {culprit!r})"
    )

    if culprit == "objects":
        print()
        print("-" * 78)
        print("Refine objects with everything earlier in the prefix already applied")
        print("-" * 78)

        def eval_with_objects(label: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
            step_x, ps_x = blended_step_persistent(
                baseline_step,
                baseline_ps,
                live_step_tpl,
                live_ps_full,
                live_groups=groups_prev,
            )
            step_x["objects"] = copy.deepcopy(objects)
            r = pred(step_x, ps_x)
            print(f"  {label:<48} {_fmt_pred(r)} [{'HIT' if hit_fn(r) else '.'}]")
            return r

        train_objects = copy.deepcopy(baseline_step.get("objects") or [])
        live_objects = copy.deepcopy(live_step_tpl.get("objects") or [])
        print(f"  train object zones: {_object_zones(train_objects)}")
        print(f"  live  object zones: {_object_zones(live_objects)}")

        eval_with_objects("train objects", train_objects)
        eval_with_objects("live objects",  live_objects)
        eval_with_objects("empty objects", [])

        for zone in _object_zones(train_objects):
            kept = [o for o in train_objects if o.get("zone") != zone]
            eval_with_objects(f"train objects minus {zone}", kept)

        print()
        print("Live object singletons:")
        for i, obj in enumerate(live_objects):
            eval_with_objects(f"  live object[{i}] {_describe_object(obj)}", [obj])

        print()
        print("Live object prefixes:")
        for n in range(1, len(live_objects) + 1):
            eval_with_objects(f"  first {n} live object(s)", live_objects[:n])

        return

    if culprit != "persistent_state" or k_star == 0:
        return

    keys_sorted = sorted(set(baseline_ps.keys()) | set(live_ps_full.keys()))
    print()
    print("-" * 78)
    print(
        "Refine persistent_state: binary search on SORTED key-prefix "
        f"({len(keys_sorted)} keys); other coarse fields follow groups_prev."
    )

    def evaluate_pstate_keys(n_live: int) -> dict[str, Any]:
        step_b, ps_b = blended_step_persistent(
            baseline_step,
            baseline_ps,
            live_step_tpl,
            live_ps_full,
            live_groups=groups_prev | {"persistent_state"},
            pstate_live_keys=set(keys_sorted[:n_live]),
        )
        r = pred(step_b, ps_b)
        ok = hit_fn(r)
        preview = keys_sorted[: min(6, n_live)]
        print(
            f"  pstate live keys n={n_live:3d}/{len(keys_sorted)} preview={preview!r}: "
            f"{_fmt_pred(r)} [{'HIT' if ok else '.'}]"
        )
        return r

    nk = len(keys_sorted)
    end_r = evaluate_pstate_keys(nk)
    if not hit_fn(end_r):
        print("[warn] All live pstate keys still do not reproduce HIT under this blend; abort refine.")
        return
    start_r = evaluate_pstate_keys(0)
    if hit_fn(start_r):
        print("[warn] Zero live pstate keys already HIT — prefix order may not isolate persistent_state.")
        return

    lo_k, hi_k = 0, nk
    while lo_k < hi_k:
        mid_k = (lo_k + hi_k) // 2
        if hit_fn(evaluate_pstate_keys(mid_k)):
            hi_k = mid_k
        else:
            lo_k = mid_k + 1

    print()
    print(f"Smallest |pstate live key prefix| ≈ {lo_k} keys (see key order in log above).")


if __name__ == "__main__":
    main()
