#!/usr/bin/env python3
"""
super_step.py
=============
Group granularized events into parent-level "super-steps" for the branched
autoregressive policy.

A granularized run interleaves three kinds of micro-steps (per
``granularization_schema.md`` section 7):

- ``swap_synth`` — one micro-step per joker swap, before the parent commit
  it belongs to. Each swap_synth is its own atomic decision.
- ``select`` — one micro-step per highlighted card in a decomposed
  parent (PlayHand / DiscardHand / UseConsumable-with-cards /
  SelectPackItem-tarot). Multiple selects share ``source_event_index``.
- ``commit`` / ``pass_through`` — the parent action's emission. ``commit``
  is for decomposed parents, ``pass_through`` for non-decomposed ones.

For the branched policy:

- Each ``swap_synth`` micro-step becomes one super-step (``family == SWAP``).
- Each ``(select*, commit | pass_through)`` block becomes ONE super-step
  whose family / arguments are derived from the commit step. The encoder
  conditions on the FIRST micro-step in the block (the state before any
  card is highlighted; ``pending_cards == []``).

Importable
----------
- ``SuperStep`` dataclass
- ``iter_super_steps(events) -> Iterator[SuperStep]``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


__all__ = [
    "SuperStep",
    "iter_super_steps",
    "make_history_record",
]


@dataclass(frozen=True)
class SuperStep:
    """A parent-level decision derived from one or more granularized steps."""

    kind: str                          # "swap" | "regular"
    encoder_step: dict[str, Any]       # state the model conditions on
    commit_step: dict[str, Any]        # supervised target source
    select_steps: list[dict[str, Any]] = field(default_factory=list)
    encoder_step_idx: int = -1         # index of encoder_step in the events list


def iter_super_steps(events: list[dict[str, Any]]) -> Iterator[SuperStep]:
    """Walk granularized events and yield parent-level super-steps.

    Walks in order. Per granularize schema, swap_synth steps for an event
    are emitted BEFORE that event's select-then-commit block, so we can
    treat the stream as: ``(swap_synth*, (select*, commit|pass_through))*``.

    Returns:
        Iterator of ``SuperStep`` instances. Each instance covers either
        one swap_synth (kind == "swap") or one select-then-commit block
        (kind == "regular"). Malformed runs of orphan selects are
        skipped silently; callers can detect them via empty output.
    """
    n = len(events)
    i = 0
    while i < n:
        step = events[i]
        kind = step.get("source_kind")

        if kind == "swap_synth":
            yield SuperStep(
                kind="swap",
                encoder_step=step,
                commit_step=step,
                select_steps=[],
                encoder_step_idx=i,
            )
            i += 1
            continue

        if kind == "select":
            source_idx = step.get("source_event_index")
            select_steps: list[dict[str, Any]] = []
            j = i
            while (
                j < n
                and events[j].get("source_kind") == "select"
                and events[j].get("source_event_index") == source_idx
            ):
                select_steps.append(events[j])
                j += 1
            if (
                j < n
                and events[j].get("source_kind") in ("commit", "pass_through")
                and events[j].get("source_event_index") == source_idx
            ):
                yield SuperStep(
                    kind="regular",
                    encoder_step=select_steps[0],
                    commit_step=events[j],
                    select_steps=select_steps,
                    encoder_step_idx=i,
                )
                i = j + 1
            else:
                # Orphan selects without a paired commit/pass_through; skip
                # past them to avoid an infinite loop on malformed data.
                i = j
            continue

        if kind in ("commit", "pass_through"):
            yield SuperStep(
                kind="regular",
                encoder_step=step,
                commit_step=step,
                select_steps=[],
                encoder_step_idx=i,
            )
            i += 1
            continue

        # Unknown / missing source_kind — advance and skip.
        i += 1


def make_history_record(ss: SuperStep) -> dict[str, Any]:
    """Build a history record describing the executed parent action.

    For ``card_seq`` and ``chained_cards`` parents the commit step itself
    has dropped ``pending_cards`` and the PendingCards object slice (the
    cards have already been played / used). To preserve "which cards
    were touched" in the history view we lift those fields from the LAST
    select step, while keeping the commit's action label, target zone /
    position, and primary ``selected_object`` (consumable / pack item).

    For card_seq parents (PlayHand / DiscardHand) where the commit has
    ``selected_object == None``, we surface the LAST selected card so
    history can place at least one "target object" in its per-record
    object slice; the rest of the played cards appear via PendingCards.
    """
    base: dict[str, Any] = dict(ss.commit_step)
    if ss.select_steps:
        last_sel = ss.select_steps[-1]
        if last_sel.get("objects") is not None:
            base["objects"] = last_sel.get("objects")
        if last_sel.get("pending_cards") is not None:
            base["pending_cards"] = last_sel.get("pending_cards")
        if base.get("selected_object") is None:
            base["selected_object"] = last_sel.get("selected_object")
    return base
