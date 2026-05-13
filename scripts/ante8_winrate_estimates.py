#!/usr/bin/env python3
"""Estimate ante-8 → ante-9 (win) rate from granularized OCR ante with debouncing."""

from __future__ import annotations

import json
from collections import Counter
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import analyze_granularized_decks_and_ante as granular  # noqa: E402


STREAK = 3


def ordered_transitions_for_video(
    paths: list[tuple[int, Path]],
    streak_required: int,
) -> list[tuple[int, int]]:
    seq: list[tuple[int, int]] = []
    confirmed: int | None = None
    streak_val: int | None = None
    streak_count = 0
    paths = sorted(paths, key=lambda t: t[0])
    for _ridx, path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for ev in data.get("events", []):
            raw = ev.get("state", {}).get("ante")
            if raw is None:
                continue
            raw = int(raw)
            if raw == streak_val:
                streak_count += 1
            else:
                streak_val = raw
                streak_count = 1
            if streak_count != streak_required:
                continue
            if streak_val is None or streak_val == confirmed:
                continue
            prev = confirmed
            confirmed = streak_val
            if prev is None:
                continue
            seq.append((prev, confirmed))
    return seq


def first_exit_after_seven_to_eight(
    transitions: list[tuple[int, int]],
) -> tuple[int, Counter[int]]:
    """Each debounced 7->8: take first subsequent edge (8, y); count y."""
    out = Counter()
    attempts = 0
    i = 0
    while i < len(transitions):
        if transitions[i] != (7, 8):
            i += 1
            continue
        attempts += 1
        j = i + 1
        while j < len(transitions) and transitions[j][0] != 8:
            j += 1
        if j < len(transitions) and transitions[j][0] == 8:
            out[transitions[j][1]] += 1
        i = j + 1 if j < len(transitions) else len(transitions)
    return attempts, out


def main() -> None:
    root = Path("data/granularized")
    files = granular.iter_run_files(root)
    by_video: dict[str, list[tuple[int, Path]]] = {}
    for p in files:
        vid, ridx = granular.parse_video_run_index(p)
        by_video.setdefault(vid, []).append((ridx, p))

    trans = Counter()
    from8 = Counter()
    for vid in by_video:
        t = ordered_transitions_for_video(
            by_video[vid],
            streak_required=STREAK,
        )
        for a, b in t:
            trans[(a, b)] += 1
        i = 0
        while i < len(t):
            if t[i][0] == 8:
                from8[t[i][1]] += 1
            i += 1

    attempts = 0
    paired_exit = Counter()
    for vid in by_video:
        t = ordered_transitions_for_video(by_video[vid], STREAK)
        a, c = first_exit_after_seven_to_eight(t)
        attempts += a
        paired_exit.update(c)

    wins = paired_exit[9]
    to_one = paired_exit[1]
    weird = sum(n for y, n in paired_exit.items() if y not in (1, 9))

    g8 = sum(from8.values())
    w89 = trans[(8, 9)]

    print("streak_required:", STREAK)
    print()
    print("=== A) Pair each debounced 7->8 with first later (8->y) [recommended]")
    print("  attempts (7->8):", attempts)
    print("  first exit counts:", dict(sorted(paired_exit.items())))
    if attempts:
        print(f"  winrate = P(y=9 | attempt): {wins / attempts:.4f}  ({wins}/{attempts})")
    if wins + to_one:
        print(
            f"  wins / (wins + y=1): {wins / (wins + to_one):.4f}  ({wins}/{wins + to_one})"
        )
    print(f"  ambiguous first exits (y not in {{1,9}}): {weird}")
    if attempts and weird:
        print(
            f"  range if ambiguous all wins: {(wins + weird) / attempts:.4f}"
        )
        print(f"  range if ambiguous all losses: {wins / attempts:.4f}")
        print(
            f"  midpoint if ambiguous 50/50: {(wins + 0.5 * weird) / attempts:.4f}"
        )
    print()
    print("=== B) Global debounced transition counts (can double-count OCR 8<->7)")
    print("  transition 7->8:", trans[(7, 8)])
    print("  transitions with prev==8 (total):", g8)
    print("  8->9:", w89, " 8->1:", from8[1], " other:", g8 - w89 - from8[1])
    if g8:
        print(f"  naive wins / all 8 exits: {w89 / g8:.4f}")
    if w89 + from8[1]:
        print(
            f"  wins / (8->9 + 8->1): {w89 / (w89 + from8[1]):.4f}"
        )


if __name__ == "__main__":
    main()
