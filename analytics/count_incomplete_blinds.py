#!/usr/bin/env python3
"""
Count incomplete blinds in parsed runs.

A blind is a contiguous block of events with ``page_name == "In_Blind"`` and the
same non-null ``state.round``. A blind is incomplete when ``hands_left`` or
``discards_left`` does not decrement smoothly vs labeled ``PlayHand`` /
``DiscardHand`` actions (likely missed classifier events).

Important timing: the PlayHand / DiscardHand screenshot is taken **after** the
game has already decremented ``hands_left`` / ``discards_left``. The labeled
frame therefore shows post-action OCR (delta 0 vs the previous frame is normal
when the HUD updated one frame earlier). Unlabeled drops of exactly 1 are
forgiven when the same counter value appears on a nearby labeled frame.

Usage::

    python analytics/count_incomplete_blinds.py [--parsed data/parsed]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from granularize import parse_event_action


def _ocr_int(ev: dict[str, Any], key: str) -> int | None:
    s = ev.get("state") or {}
    v = s.get(key)
    return int(v) if isinstance(v, int) else None


def _iter_run_files(parsed_root: Path):
    for partition in sorted(parsed_root.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("video_id="):
            continue
        video_id = partition.name.split("=", 1)[1]
        for run_file in sorted(partition.glob("run_*.json")):
            yield video_id, run_file


@dataclass
class BlindSegment:
    video_id: str
    run_index: int
    round: int
    start_frame_idx: int | None
    event_indices: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        return bool(self.reasons)


def _segment_blinds(
    video_id: str,
    run_index: int,
    events: list[dict[str, Any]],
) -> list[BlindSegment]:
    segments: list[BlindSegment] = []
    current: BlindSegment | None = None

    for i, ev in enumerate(events):
        if ev.get("page_name") != "In_Blind":
            current = None
            continue
        rnd = _ocr_int(ev, "round")
        if rnd is None:
            current = None
            continue
        if current is None or current.round != rnd:
            current = BlindSegment(
                video_id=video_id,
                run_index=run_index,
                round=rnd,
                start_frame_idx=ev.get("frame_idx"),
            )
            segments.append(current)
        current.event_indices.append(i)

    return [s for s in segments if s.event_indices]


def _event_action(ev: dict[str, Any]) -> str:
    base, _ = parse_event_action(ev)
    return base


def _labeled_frame_has_post_action_ocr(
    *,
    delta: int,
) -> bool:
    """Return True when OCR on a labeled PlayHand/DiscardHand frame looks valid.

    The screenshot is captured after the counter has already decremented, so
    ``delta`` vs the immediately previous event may be 0 (HUD updated earlier)
    or 1 (HUD updated on this frame). Anything else is suspicious.
    """
    return 0 <= delta <= 1


def _find_nearby_labeled_frame(
    events: list[dict[str, Any]],
    indices: list[int],
    start_pos: int,
    *,
    action_name: str,
    counter_key: str,
    counter_value: int,
    look_ahead: int = 1,
) -> bool:
    """True if a labeled frame within ``look_ahead`` shares ``counter_value``."""
    end = min(len(indices), start_pos + look_ahead + 1)
    for pos in range(start_pos, end):
        ev = events[indices[pos]]
        if _event_action(ev) != action_name:
            continue
        curr = _ocr_int(ev, counter_key)
        if curr == counter_value:
            return True
    return False


def _check_counter_smoothness(
    seg: BlindSegment,
    events: list[dict[str, Any]],
    *,
    counter_key: str,
    action_name: str,
    skip_action_on_frame: str,
) -> None:
    indices = seg.event_indices

    for pos, idx in enumerate(indices):
        ev = events[idx]
        curr = _ocr_int(ev, counter_key)
        base = _event_action(ev)

        if base == skip_action_on_frame:
            continue

        prev: int | None = None
        if pos > 0:
            prev = _ocr_int(events[indices[pos - 1]], counter_key)

        if prev is None or curr is None:
            continue

        delta = prev - curr

        if base == action_name:
            if not _labeled_frame_has_post_action_ocr(delta=delta):
                if delta < 0:
                    seg.reasons.append(
                        f"{action_name} at frame {ev.get('frame_idx')} but "
                        f"{counter_key} increased ({prev} -> {curr})"
                    )
                else:
                    seg.reasons.append(
                        f"{action_name} at frame {ev.get('frame_idx')} but "
                        f"{counter_key} dropped by {delta} (expected 0 or 1 "
                        f"vs previous frame)"
                    )
            continue

        if delta > 1:
            seg.reasons.append(
                f"{counter_key} dropped by {delta} at frame {ev.get('frame_idx')} "
                f"(expected <=1)"
            )
        elif delta == 1:
            if not _find_nearby_labeled_frame(
                events,
                indices,
                pos,
                action_name=action_name,
                counter_key=counter_key,
                counter_value=curr,
            ):
                seg.reasons.append(
                    f"{counter_key} dropped by 1 at frame {ev.get('frame_idx')} "
                    f"without nearby {action_name} (action={base})"
                )


def analyze_run(run: dict[str, Any], video_id: str) -> list[BlindSegment]:
    events = run.get("events") or []
    run_index = int(run.get("run_index") or 0)
    segments = _segment_blinds(video_id, run_index, events)
    for seg in segments:
        _check_counter_smoothness(
            seg,
            events,
            counter_key="hands_left",
            action_name="PlayHand",
            skip_action_on_frame="DiscardHand",
        )
        _check_counter_smoothness(
            seg,
            events,
            counter_key="discards_left",
            action_name="DiscardHand",
            skip_action_on_frame="PlayHand",
        )
    return segments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/incomplete_blinds_report.json"),
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/tables/incomplete_blinds.csv"),
    )
    args = ap.parse_args()

    if not args.parsed.exists():
        raise SystemExit(f"parsed root not found: {args.parsed}")

    all_segments: list[BlindSegment] = []
    runs_processed = 0

    for video_id, run_file in _iter_run_files(args.parsed):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        runs_processed += 1
        all_segments.extend(analyze_run(run, video_id))

    total_blinds = len(all_segments)
    incomplete = [s for s in all_segments if s.incomplete]
    incomplete_count = len(incomplete)
    rate = incomplete_count / total_blinds if total_blinds else 0.0

    print(f"runs processed:     {runs_processed}")
    print(f"blinds segmented:   {total_blinds}")
    print(f"incomplete blinds:  {incomplete_count}")
    print(f"incomplete rate:    {rate:.4f}")

    incomplete_records = [
        {
            "video_id": s.video_id,
            "run_index": s.run_index,
            "round": s.round,
            "start_frame_idx": s.start_frame_idx,
            "reasons": s.reasons,
        }
        for s in incomplete
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parsed_root": args.parsed.as_posix(),
        "runs_processed": runs_processed,
        "total_blinds": total_blinds,
        "incomplete_blinds": incomplete_count,
        "incomplete_rate": rate,
        "incomplete_blind_records": incomplete_records,
    }
    args.report.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote report -> {args.report.as_posix()}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "run_index", "round", "start_frame_idx", "reasons"])
        for rec in incomplete_records:
            w.writerow(
                [
                    rec["video_id"],
                    rec["run_index"],
                    rec["round"],
                    rec["start_frame_idx"],
                    " | ".join(rec["reasons"]),
                ]
            )
    print(f"wrote csv -> {args.csv.as_posix()}")


if __name__ == "__main__":
    main()
