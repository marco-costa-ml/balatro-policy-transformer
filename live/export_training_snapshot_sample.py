#!/usr/bin/env python3
"""
Export one granularized training step + matching persistent_state BEFORE that step
into the same JSON envelope Lua's agent_bridge uses for snapshots (what
live_encoder / tensorize_step consume).

This is the closest on-disk representation of "model input before tensorizing"
for training data: same top-level keys as ``snapshots_debug/latest_*.json``,
minus runtime-only fields (``meta.game_state_id``, etc.) unless you add them.

Usage:
  python live/export_training_snapshot_sample.py \\
    --page In_JokerStandardPlanet_Pack \\
    --out artifacts/training_snapshot_samples/In_JokerStandardPlanet_Pack_training.json

Optional: --video 2512428128 --run 003 --step-id 13  (defaults: first match for page)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _find_step(
    repo: Path, page: str, video_id: str | None, run_index: str | None, step_id: int | None
) -> tuple[dict, dict, str, str]:
    """Return (granularized_run, step_event, video_id, run_file_stem)."""
    gdir = repo / "data" / "granularized"
    for part in sorted(gdir.glob("video_id=*")):
        vid = part.name.split("=", 1)[1]
        if video_id is not None and vid != str(video_id):
            continue
        for grun in sorted(part.glob("run_*.json")):
            stem = grun.stem
            if run_index is not None and stem != f"run_{run_index}":
                continue
            run = json.loads(grun.read_text(encoding="utf-8"))
            for ev in run.get("events") or []:
                if ev.get("page_name") != page:
                    continue
                if step_id is not None and ev.get("step_id") != step_id:
                    continue
                return run, ev, vid, stem
    raise SystemExit(
        f"No event found: page={page!r} video={video_id!r} run={run_index!r} step_id={step_id!r}"
    )


def _persistent_before(repo: Path, video_id: str, run_stem: str, step_id: int) -> dict:
    path = repo / "data" / "persistent_state" / f"video_id={video_id}" / f"{run_stem}.json"
    if not path.exists():
        raise SystemExit(f"persistent_state missing: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    states = doc.get("states") or []
    if not (0 <= step_id < len(states)):
        raise SystemExit(f"step_id {step_id} out of range for {path} (n={len(states)})")
    return states[step_id]


def build_live_shaped_snapshot(repo: Path, page: str, video_id: str | None, run_index: str | None, step_id: int | None) -> dict:
    run, ev, vid, run_stem = _find_step(repo, page, video_id, run_index, step_id)
    sid = int(ev["step_id"])
    p_before = _persistent_before(repo, vid, run_stem, sid)

    target_position = ev.get("target_position")
    if target_position is None:
        target_position = -1

    return {
        "schema_version": "training_export/2.0.0",
        "request_id": sid,
        "page_name": ev.get("page_name"),
        "source_kind": ev.get("source_kind"),
        "action_subtype": ev.get("action_subtype"),
        "state": ev.get("state") or {},
        "objects": ev.get("objects") or [],
        "pending_cards": ev.get("pending_cards") or [],
        "target_zone": ev.get("target_zone"),
        "target_position": target_position,
        "persistent_state": p_before,
        # Training corpus has no precomputed legality strings; mask_builder builds masks offline.
        "legal_actions": None,
        "meta": {
            "provenance": "data/granularized + data/persistent_state (state BEFORE step)",
            "video_id": vid,
            "run_index": run.get("run_index"),
            "run_file": f"video_id={vid}/{run_stem}.json",
            "step_id": sid,
            "frame_idx": ev.get("frame_idx"),
            "recorded_action": ev.get("action"),
            "source_action": ev.get("source_action"),
            "micro_index": ev.get("micro_index"),
            "swap_pair": ev.get("swap_pair"),
        },
        "_granularizer_extras": {
            "selected_object": ev.get("selected_object"),
            "source_event_index": ev.get("source_event_index"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--video", default=None, help="video_id string (default: first file with a match)")
    ap.add_argument("--run", default=None, help="run stem e.g. 003 for run_003.json")
    ap.add_argument("--step-id", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--repo",
        type=Path,
        default=_REPO_ROOT,
    )
    args = ap.parse_args()

    snap = build_live_shaped_snapshot(args.repo, args.page, args.video, args.run, args.step_id)
    out = args.out
    if out is None:
        out = args.repo / "artifacts" / "training_snapshot_samples" / f"{args.page}_training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
