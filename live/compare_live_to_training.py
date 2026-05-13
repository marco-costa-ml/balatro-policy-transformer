#!/usr/bin/env python3
"""
compare_live_to_training.py
===========================
Diff the latest live snapshot (saved by agent_server.py to
``agent_io/snapshots_debug/latest_<Page>.json``) against a representative
training Blind_Select / In_Blind / In_Shop / etc. sample.

Reports:
  - Object zone histograms (live vs training)
  - OCR field validity (which keys are present/absent in each)
  - Per-tensor diff after running both through ``tensorize_step``
    (focus on object_class_id / object_zone_id / object_position /
     state vector validity / persistent_state-derived ids).

Usage:
    python live/compare_live_to_training.py --page Blind_Select
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from action_map import compute_action_map
from tensorize import Normalizer, VocabLookup, tensorize_step
from live.live_encoder import _normalize_legacy_snapshot


def default_io_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Balatro" / "agent_io"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Balatro" / "agent_io"
    return Path.home() / ".local" / "share" / "love" / "Balatro" / "agent_io"


def find_training_sample(page: str, repo_root: Path) -> tuple[dict, dict] | None:
    """Find a granularized step + persistent_state for the given page."""
    for fp in sorted((repo_root / "data" / "granularized").rglob("run_*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        for ev in d.get("events", []):
            if ev.get("page_name") == page:
                # Load the matching persistent_state for this run.
                vid = d["video_id"]
                ps_path = repo_root / "data" / "persistent_state" / f"video_id={vid}" / fp.name
                if ps_path.exists():
                    ps = json.loads(ps_path.read_text(encoding="utf-8"))
                    states = ps.get("states") or []
                    step_id = ev.get("step_id", 0)
                    if 0 <= step_id < len(states):
                        return ev, states[step_id]
    return None


def histogram(name: str, snap: dict, key: str = "objects") -> dict:
    items = snap.get(key) or []
    return collections.Counter((o.get("zone"), o.get("position_in_zone")) for o in items)


def ocr_present(snap: dict) -> set[str]:
    state = snap.get("state") or {}
    return {k for k, v in state.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="Blind_Select")
    ap.add_argument("--io-dir", type=Path, default=default_io_dir())
    args = ap.parse_args()

    live_path = args.io_dir / "snapshots_debug" / f"latest_{args.page}.json"
    if not live_path.exists():
        print(f"[!] live snapshot missing: {live_path}")
        print(f"    Make sure agent_server.py captured one (page={args.page}) since the last edit.")
        sys.exit(1)

    live_snap = json.loads(live_path.read_text(encoding="utf-8"))
    if live_snap.get("schema_version") != "live/2.0.0":
        # Legacy snapshot — route through the same shim agent_server uses so
        # this comparison stays meaningful when an older Lua build is still
        # writing snapshots.
        live_snap = _normalize_legacy_snapshot(live_snap)
    training = find_training_sample(args.page, _REPO_ROOT)
    if training is None:
        print(f"[!] no training sample with page={args.page}")
        sys.exit(2)
    train_step, train_pstate = training

    # Live snapshot is already in the (step + persistent_state) shape.
    live_step = {
        "page_name": live_snap.get("page_name"),
        "source_kind": None,
        "action_subtype": None,
        "state": live_snap.get("state") or {},
        "objects": live_snap.get("objects") or [],
        "pending_cards": live_snap.get("pending_cards") or [],
        "target_zone": None,
        "target_position": None,
        "action": "",
    }
    live_pstate = live_snap.get("persistent_state") or {}

    print("=" * 78)
    print(f"PAGE: {args.page}")
    print("=" * 78)
    print()

    print("--- Object zone histogram ---")
    print(f"{'(zone, pos)':<48} {'live':>6} {'train':>6}")
    live_hist = histogram(args.page, {"objects": live_step["objects"] or []})
    train_hist = histogram(args.page, {"objects": train_step.get("objects") or []})
    keys = sorted(set(live_hist) | set(train_hist), key=lambda k: (k[0] or "", k[1] or 0))
    for k in keys:
        l = live_hist.get(k, 0)
        t = train_hist.get(k, 0)
        flag = "  " if l == t else " *"
        print(f"{flag} {str(k):<46} {l:>6} {t:>6}")
    print()

    print("--- OCR field presence ---")
    lp = ocr_present(live_step)
    tp = ocr_present(train_step)
    only_live = lp - tp
    only_train = tp - lp
    common = lp & tp
    print(f"common  : {sorted(common)}")
    print(f"only live: {sorted(only_live)}")
    print(f"only train: {sorted(only_train)}")
    print()

    print("--- Persistent state spot-check ---")
    keys_to_check = (
        "stake", "first_hand", "first_discard", "small_status", "big_status",
        "ante_boss_blind", "ecto_minus", "skips", "hands_played",
        "is_boss_blind_rerolled",
    )
    print(f"{'key':<24} {'live':>20} {'train':>20}")
    for k in keys_to_check:
        l = live_pstate.get(k)
        t = train_pstate.get(k)
        flag = "  " if l == t else " *"
        print(f"{flag} {k:<22} {repr(l):>20} {repr(t):>20}")

    # Compare nested dicts.
    for nk in ("deck", "deck_modifiers"):
        print(f"\n  {nk}:")
        ld = live_pstate.get(nk) or {}
        td = train_pstate.get(nk) or {}
        for k in sorted(set(ld) | set(td)):
            l = ld.get(k); t = td.get(k)
            flag = "  " if l == t else " *"
            print(f"   {flag} {k:<28} {repr(l):>16} {repr(t):>16}")
    print()

    # Tensor diff
    vocab = VocabLookup(json.loads((_REPO_ROOT / "artifacts" / "vocab.json").read_text(encoding="utf-8")))
    norm = Normalizer(json.loads((_REPO_ROOT / "artifacts" / "normalization.json").read_text(encoding="utf-8")))
    feat = json.loads((_REPO_ROOT / "artifacts" / "feature_config.json").read_text(encoding="utf-8"))
    action_cfg = json.loads((_REPO_ROOT / "data" / "action_space_config.json").read_text(encoding="utf-8"))
    amap = compute_action_map(action_cfg)

    rec_live = tensorize_step(live_step, live_pstate, amap, vocab, norm, feat)
    rec_train = tensorize_step(train_step, train_pstate, amap, vocab, norm, feat)

    print("--- Tensor diff ---")
    print(f"{'channel':<32} {'shape':<14} {'identical':<10} {'first-mismatch'}")
    for k in sorted(rec_live):
        a = np.asarray(rec_live[k])
        b = np.asarray(rec_train[k])
        if a.shape != b.shape:
            print(f"  {k:<30} {str(a.shape):<14} SHAPE_DIFF")
            continue
        eq = np.array_equal(a, b)
        if eq:
            print(f"  {k:<30} {str(a.shape):<14} yes")
        else:
            diff_idx = np.argwhere(a != b)
            n = len(diff_idx)
            first = diff_idx[0]
            try:
                la = a[tuple(first)]
                lb = b[tuple(first)]
            except IndexError:
                la = lb = "?"
            print(f"  {k:<30} {str(a.shape):<14} no   ({n} diffs, first @ {tuple(first)} live={la!r} train={lb!r})")


if __name__ == "__main__":
    main()
