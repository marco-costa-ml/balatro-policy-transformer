#!/usr/bin/env python3
"""
emission_policy.py
==================
Convert a parent-commit decision from the branched policy into the
sequence of granular IPC labels that ``Balatro/agent_bridge.lua``
expects.

Per ``Balatro/agent_bridge.lua`` ZONE_DISPATCH + fixed dispatch table:

- ``no_args`` families (SelectBlind / SkipBlind / RerollBossBlind /
  CashOut / LeaveShop / SkipPack / RerollShop) emit ONE label equal to
  the family name.
- ``card_seq`` families (PlayHand / DiscardHand) emit ``num_cards``
  ``SelectCard_CurrentHand_<dyn_idx>`` labels followed by the commit
  name (``PlayHand`` / ``DiscardHand``).
- ``single_ptr`` families (Buy*, Sell*, BuyAndUseShopConsumable)
  emit ONE label ``<family_name>_<item_idx>``.
- ``chained_cards`` families (UseConsumable / SelectPackItem) emit
  ``num_cards`` ``SelectCard_<card_zone>_<dyn_idx>`` labels first,
  then the commit ``<family_name>_<item_idx>``. The card zone is
  CurrentHand in ``In_Blind`` and TarotSpectralHand in pack pages.
- ``joker_pair`` (SWAP) emits ONE label ``SWAP_<min(i,j)>_<max(i,j)>``
  matching the canonical ordering Balatro's dispatcher expects.

Index space translation (card_seq + chained_cards)
--------------------------------------------------
The model's ``card_ptr_local_seq`` predicts **original** positions in
the encoder snapshot's hand (the position_in_zone seen on the encoder
step, which is also the dynamic pool at step 0 since nothing has been
picked yet). The supervised target in ``tensorize.py`` is also the
original position — see ``_selected_position`` and ``_fill_branched_targets``.

Lua's ``_select_card_in_pool(area, idx)``, however, indexes into the
**dynamic pool** at the moment of dispatch — i.e. the set of currently
unhighlighted cards renumbered ``0..n_pool-1``. After each select Lua
moves that card into highlighted, so the pool shrinks and indices shift.

So at emission time we convert each original-position pick ``p_t`` to
the dynamic-pool index Lua expects with one line of arithmetic:

    dyn_idx_t = p_t - count(p_i for i < t such that p_i < p_t)

i.e. subtract the number of earlier picks numerically smaller than this
one (because each of those was already pulled out of the pool before
this label fires). The model's ``already_picked`` mask guarantees no
position is repeated, so the resulting ``dyn_idx_t`` is always in
``[0, n_pool_at_step_t - 1]`` and Lua will accept it.

``EmissionPolicy`` is stateful: it caches the unrolled plan between
ticks so each Lua poll consumes the next label until the plan is
drained. Plans are invalidated when the snapshot's page changes or
when the next queued label is not in Lua's legal set (defensive
sanity check; in practice the model should not produce a plan that
diverges from Lua's legality).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from family_map import DEFAULT_CARD_ZONE_FOR_FAMILY


__all__ = [
    "ExpansionResult",
    "EmissionPolicy",
    "card_zone_for_emission",
    "original_to_dynamic_pool_indices",
]


# ---------------------------------------------------------------------------
# Original-position -> dynamic-pool-index translation
# ---------------------------------------------------------------------------

def original_to_dynamic_pool_indices(
    original_positions: list[int] | tuple[int, ...],
) -> list[int]:
    """Translate a sequence of original-position card pointers to the
    dynamic-pool indices Lua's ``_select_card_in_pool`` expects.

    ``original_positions[t]`` is the model's predicted position_in_zone
    of the t-th picked card, in the encoder snapshot's hand. After
    Lua executes the first ``t`` labels the pool has those ``t`` cards
    removed; the t-th label needs an index relative to the *current*
    (shrunken) pool, which is

        dyn_idx_t = p_t - #{ p_i : i < t, p_i < p_t }

    Negative / ``None`` entries (padding from ``card_ptr_local_seq``)
    pass through as ``-1`` so the caller can drop them.
    """
    out: list[int] = []
    for t, p in enumerate(original_positions):
        if p is None or int(p) < 0:
            out.append(-1)
            continue
        p_int = int(p)
        shift = 0
        for q in original_positions[:t]:
            if q is None:
                continue
            q_int = int(q)
            if 0 <= q_int < p_int:
                shift += 1
        out.append(p_int - shift)
    return out


# ---------------------------------------------------------------------------
# Card-zone resolution at emission time
# ---------------------------------------------------------------------------
#
# Mirrors ``argument_spec.card_zone_for_use_consumable`` /
# ``card_zone_for_select_pack_item`` but driven by ``page_name`` only so
# we don't need the original step dict. The granular ``SelectCard_*``
# label keyed off the same zone names Lua's ZONE_DISPATCH knows about.

def card_zone_for_emission(family_name: str, page_name: str | None) -> str | None:
    """Return the canonical card-zone name for ``SelectCard_<zone>_<idx>``."""
    if family_name in {"PlayHand", "DiscardHand"}:
        return "CurrentHand"
    if family_name == "UseConsumable_CurrentConsumables":
        if page_name == "In_TarotSpectral_Pack":
            return "TarotSpectralHand"
        return "CurrentHand"
    if family_name == "SelectPackItem_PackOfferings":
        # SelectPackItem only takes card targets in tarot/spectral packs.
        # Joker / planet / standard packs have num_cards == 0.
        return "TarotSpectralHand"
    return DEFAULT_CARD_ZONE_FOR_FAMILY.get(family_name)


# ---------------------------------------------------------------------------
# Expansion (decision -> ordered label list)
# ---------------------------------------------------------------------------

@dataclass
class ExpansionResult:
    """Result of expanding one decision into IPC labels.

    ``labels`` may be empty when the decision was malformed (e.g. SWAP
    with i == j, or chained_cards with no valid item pick); callers
    should fall back to a safe label in that case.
    """

    labels: list[str]
    family_name: str
    decoder_shape: str


def expand_decision(
    *,
    family_name: str,
    decoder_shape: str,
    page_name: str | None,
    num_cards: int,
    card_ptr_local_seq: list[int],
    item_ptr_local: int | None,
    swap_i_local: int | None,
    swap_j_local: int | None,
) -> ExpansionResult:
    """Convert decoded arguments into the ordered IPC label sequence."""
    if decoder_shape == "no_args":
        return ExpansionResult([family_name], family_name, decoder_shape)

    if decoder_shape == "card_seq":
        zone = card_zone_for_emission(family_name, page_name) or "CurrentHand"
        labels: list[str] = []
        picks = list(card_ptr_local_seq[: max(0, int(num_cards))])
        # Model predicts ORIGINAL positions in the encoder hand; Lua's
        # ``_select_card_in_pool`` expects DYNAMIC-POOL indices that
        # shift after each select. Translate before emitting (see
        # module docstring for the formula).
        dyn_picks = original_to_dynamic_pool_indices(picks)
        for dyn in dyn_picks:
            if dyn < 0:
                continue
            labels.append(f"SelectCard_{zone}_{dyn}")
        labels.append(family_name)
        return ExpansionResult(labels, family_name, decoder_shape)

    if decoder_shape == "single_ptr":
        if item_ptr_local is None or int(item_ptr_local) < 0:
            return ExpansionResult([], family_name, decoder_shape)
        return ExpansionResult(
            [f"{family_name}_{int(item_ptr_local)}"], family_name, decoder_shape
        )

    if decoder_shape == "chained_cards":
        if item_ptr_local is None or int(item_ptr_local) < 0:
            return ExpansionResult([], family_name, decoder_shape)
        labels = []
        zone = card_zone_for_emission(family_name, page_name)
        if zone and int(num_cards) > 0:
            picks = list(card_ptr_local_seq[: int(num_cards)])
            dyn_picks = original_to_dynamic_pool_indices(picks)
            for dyn in dyn_picks:
                if dyn < 0:
                    continue
                labels.append(f"SelectCard_{zone}_{dyn}")
        labels.append(f"{family_name}_{int(item_ptr_local)}")
        return ExpansionResult(labels, family_name, decoder_shape)

    if decoder_shape == "joker_pair":
        if swap_i_local is None or swap_j_local is None:
            return ExpansionResult([], family_name, decoder_shape)
        i, j = int(swap_i_local), int(swap_j_local)
        if i < 0 or j < 0 or i == j:
            return ExpansionResult([], family_name, decoder_shape)
        lo, hi = (i, j) if i < j else (j, i)
        return ExpansionResult(
            [f"SWAP_{lo}_{hi}"], family_name, decoder_shape
        )

    # ``reserved`` (StartNewRun) or unknown — no emission.
    return ExpansionResult([], family_name, decoder_shape)


# ---------------------------------------------------------------------------
# Stateful queue between Lua polls
# ---------------------------------------------------------------------------

@dataclass
class _PendingPlan:
    """State for an active (in-flight) emission plan."""

    labels: list[str]
    family_name: str
    decoder_shape: str
    page_name: str | None
    decision_snapshot: dict[str, Any]
    decision: dict[str, Any]
    emitted: list[str] = field(default_factory=list)


class EmissionPolicy:
    """Stateful adapter between the model's parent-commit decision and Lua's
    per-tick IPC.

    Lifecycle:
        1. ``set_plan(labels, ...)`` after the model decodes a super-step.
        2. ``pop_next(legal_set, page_name)`` each Lua poll until the plan
           is drained; returns ``None`` once the queue is empty.
        3. ``finished()`` becomes True after the LAST label is popped;
           caller should commit a history record at that point.
        4. Bad plans (page change mid-plan, queued label not in Lua's
           legal set) auto-invalidate so the caller falls back to model
           re-query / fallback label.
    """

    def __init__(self) -> None:
        self._pending: _PendingPlan | None = None
        self._invalidated: bool = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_pending(self) -> bool:
        return self._pending is not None and bool(self._pending.labels)

    def last_committed(self) -> _PendingPlan | None:
        """Return the just-completed plan (between last pop and next set).

        Cleared on the next ``set_plan`` call.
        """
        if self._pending is None:
            return None
        if self._pending.labels:
            # Plan still in progress.
            return None
        return self._pending

    def finished(self) -> bool:
        """True iff the most recently set plan has been fully emitted."""
        return (
            self._pending is not None
            and not self._pending.labels
            and not self._invalidated
        )

    def invalidated(self) -> bool:
        return self._invalidated

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_plan(
        self,
        labels: list[str],
        *,
        family_name: str,
        decoder_shape: str,
        page_name: str | None,
        decision_snapshot: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        """Install a new emission plan, replacing any prior one."""
        self._pending = _PendingPlan(
            labels=list(labels),
            family_name=family_name,
            decoder_shape=decoder_shape,
            page_name=page_name,
            decision_snapshot=decision_snapshot,
            decision=decision,
            emitted=[],
        )
        self._invalidated = False

    def clear(self) -> None:
        self._pending = None
        self._invalidated = False

    def pop_next(
        self,
        legal_set: set[str] | None,
        page_name: str | None,
    ) -> str | None:
        """Pop the next IPC label, validating against the live snapshot.

        Returns ``None`` when no plan is active, or when the plan was
        invalidated (page changed, next label not legal). On
        invalidation the pending plan is dropped and the caller is
        expected to fall back to model re-query.
        """
        if self._pending is None or not self._pending.labels:
            return None
        if self._pending.page_name is not None and page_name != self._pending.page_name:
            # Page transitioned during the plan — treat as invalid.
            self._pending = None
            self._invalidated = True
            return None
        next_label = self._pending.labels[0]
        if legal_set is not None and next_label not in legal_set:
            # Lua doesn't accept this label any more — abandon plan.
            self._pending = None
            self._invalidated = True
            return None
        # Pop it.
        self._pending.labels.pop(0)
        self._pending.emitted.append(next_label)
        return next_label
