#!/usr/bin/env python3
"""
state_reducer.py
================
Persistent state reducer for granularized Balatro runs.

This is the executable port of ``state_reducer_pseudocode.md`` (which itself
is the implementation companion to ``state_schema.md``). It exposes the
following surface to the rest of the pipeline:

- ``default_state()`` -> dict
- ``apply_step(state, step)`` -> dict  (pure: same inputs -> same output)
- ``reduce_run(run)``         -> list[dict]  (state BEFORE each event)
- ``to_model_visible(state)`` -> dict        (model input subset)
- ``to_full_state(state)``    -> dict        (model + internal; for masking)
- ``MODEL_VISIBLE_KEYS``, ``INTERNAL_KEYS``

Reducer contract
----------------
- ``state`` is reset to ``default_state()`` at every ``StartNewRun`` event.
- ``apply_step(state, step)`` returns the persistent state AFTER ``step``;
  the model input for step ``t`` uses the state BEFORE that step.
- Internal fields (``swap_count``, ``last_swap``, ``deck_detected``,
  ``prev_jokers_all``, ``hand_and_level_unparsed_count``) MUST NOT be passed
  to the model. ``to_model_visible`` filters them out.

The ``ante_boss_blind`` field is updated as an observation-driven side
effect on every step (matching the pseudocode note that the field is set
"when BlindOfferings becomes visible, not [in SelectBlind]").
"""

from __future__ import annotations

import copy
from typing import Any

from card_effects import (
    BLACK_HOLE_CLASS,
    CARD_CONSUMABLE_HANDLERS,
    PLANET_TO_HAND,
    STANDARD_CARD_CLASS_RANGE,
    STONE_CARD_CLASS,
    TRACKED_DECK_CAP,
    UNHANDLED_RANDOM_CONSUMABLES,
    apply_card_consumable,
    build_abandoned_deck,
    build_checkered_deck,
    build_standard_deck,
    normalize_pack_card,
    trim_tracked_deck,
)


# ---------------------------------------------------------------------------
# 2) Visibility constants
# ---------------------------------------------------------------------------

MODEL_VISIBLE_KEYS: frozenset[str] = frozenset(
    {
        "deck",
        "stake",
        "tracked_deck_cards",
        "deck_modifiers",
        "last_tarot_planet",
        "ecto_minus",
        "skips",
        "hands_played",
        "unused_discards",
        "first_hand",
        "first_discard",
        "vouchers_redeemed",
        "bosses_used",
        "ante_boss_blind",
        "small_status",
        "big_status",
        "is_boss_blind_rerolled",
        "hands",
    }
)

INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "swap_count",
        "last_swap",
        "deck_detected",
        "prev_jokers_all",
        "hand_and_level_unparsed_count",
        "unhandled_random_consumable_count",
    }
)


# ---------------------------------------------------------------------------
# 3) Class ID constants
# ---------------------------------------------------------------------------

# decks
DECK_ABANDONED = 52
DECK_CHECKERED = 57
DECK_ERRATIC = 58
DECK_MAGIC = 61
DECK_NEBULA = 62
DECK_ZODIAC = 67

DECK_CLASS_RANGE = range(52, 68)  # 52..67 inclusive

# vouchers
VOUCHER_CLASS_RANGE = range(320, 352)
V_CLEARANCE_SALE = 322
V_CRYSTAL_BALL = 323
V_DIRECTORS_CUT = 324
V_LIQUIDATION = 330
V_OVERSTOCK_NORM = 336
V_PLANET_MERCHANT = 341
V_RETCON = 346
V_TAROT_MERCHANT = 348
V_TELESCOPE = 350

# blinds
BIG_BLIND = 371
SMALL_BLIND = 394
BOSS_BLIND_RANGE = range(370, 400)  # 370..399; exclude BIG/SMALL


def is_boss_blind(class_id: int) -> bool:
    return class_id in BOSS_BLIND_RANGE and class_id not in (BIG_BLIND, SMALL_BLIND)


# consumables
PLANET_RANGE = range(236, 248)
SPECTRAL_RANGE = range(248, 266)
TAROT_RANGE = range(298, 320)
ECTO_CLASS = 253
FOOL_CLASS = 303


def is_planet_or_tarot(class_id: int) -> bool:
    return class_id in PLANET_RANGE or class_id in TAROT_RANGE


# stickers
STICKER_RENTAL = 367
STICKER_PERISHABLE = 368
STICKER_ETERNAL = 369

# default stake when CurrentStake[0] is missing
DEFAULT_STAKE = 268


# ---------------------------------------------------------------------------
# 4) Initial / Default state
# ---------------------------------------------------------------------------

POKER_HANDS: list[str] = [
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]


def default_state() -> dict[str, Any]:
    """Reset state used at run start (and as fallback when StartNewRun is missing)."""
    return {
        # [MODEL-VISIBLE]
        "deck": {
            "class_id": None,
            "name": None,
            "is_magic": False,
            "is_nebula": False,
            "is_abandoned": False,
            "is_checkered": False,
            "is_zodiac": False,
            "is_erratic": False,
        },
        "stake": DEFAULT_STAKE,
        # tracked_deck_cards is a list of card objects (see card_effects.py).
        # Capped at TRACKED_DECK_CAP entries. Defaults to a standard 52-card
        # playing deck; apply_deck_initialization replaces it for the special
        # decks (abandoned/checkered).
        "tracked_deck_cards": build_standard_deck(),
        "deck_modifiers": {
            "no_face_cards_start": False,
            "spades_hearts_only_start": False,
            "randomized_starting_deck": False,
        },
        "last_tarot_planet": None,
        "ecto_minus": 0,
        "skips": 0,
        "hands_played": 0,
        "unused_discards": 0,
        "first_hand": True,
        "first_discard": True,
        "vouchers_redeemed": [],
        "bosses_used": [],
        "ante_boss_blind": None,
        "small_status": 1,
        "big_status": 1,
        "is_boss_blind_rerolled": False,
        "hands": {
            name: {"level": 1, "played": 0, "played_this_round": 0}
            for name in POKER_HANDS
        },
        # [INTERNAL]
        "swap_count": 0,
        "last_swap": None,
        "deck_detected": False,
        "prev_jokers_all": None,
        "hand_and_level_unparsed_count": 0,
        "unhandled_random_consumable_count": 0,
    }


# ---------------------------------------------------------------------------
# 9) Helpers (declared early so reducer can use them)
# ---------------------------------------------------------------------------

def first_or_none(seq):
    """Return the first element of a sequence, or None."""
    if not seq:
        return None
    return seq[0]


def parse_base_action(action: str) -> str:
    """``'BuyShopItem_2'`` -> ``'BuyShopItem'``; ``'SWAP_0_1'`` -> ``'SWAP'``."""
    return action.split("_", 1)[0]


def extract_target_class_id(selected_object) -> int | None:
    """Pull ``selected_object.object.class_id`` if it exists and is an int."""
    if not isinstance(selected_object, dict):
        return None
    obj = selected_object.get("object")
    if not isinstance(obj, dict):
        return None
    cid = obj.get("class_id")
    return int(cid) if isinstance(cid, int) else None


def extract_target_object(selected_object) -> dict[str, Any] | None:
    """Pull the ``selected_object.object`` payload if present."""
    if not isinstance(selected_object, dict):
        return None
    obj = selected_object.get("object")
    return obj if isinstance(obj, dict) else None


def add_voucher_if_absent(state: dict[str, Any], voucher_class_id: int) -> None:
    """Append voucher class id to ``vouchers_redeemed`` only if not present."""
    if voucher_class_id not in state["vouchers_redeemed"]:
        state["vouchers_redeemed"].append(voucher_class_id)


# ---------------------------------------------------------------------------
# 6) StartNewRun initialization
# ---------------------------------------------------------------------------

def apply_deck_initialization(state: dict[str, Any], deck_class_id: int) -> None:
    """Apply per-deck side effects (class flags, tracked-deck contents, starting vouchers)."""
    if deck_class_id == DECK_ABANDONED:
        state["deck"]["is_abandoned"] = True
        state["deck_modifiers"]["no_face_cards_start"] = True
        state["tracked_deck_cards"] = build_abandoned_deck()

    elif deck_class_id == DECK_CHECKERED:
        state["deck"]["is_checkered"] = True
        state["deck_modifiers"]["spades_hearts_only_start"] = True
        state["tracked_deck_cards"] = build_checkered_deck()

    elif deck_class_id == DECK_ERRATIC:
        state["deck"]["is_erratic"] = True
        state["deck_modifiers"]["randomized_starting_deck"] = True
        # Keep the default standard deck; do not hallucinate randomized contents.

    elif deck_class_id == DECK_MAGIC:
        state["deck"]["is_magic"] = True
        add_voucher_if_absent(state, V_CRYSTAL_BALL)

    elif deck_class_id == DECK_NEBULA:
        state["deck"]["is_nebula"] = True
        add_voucher_if_absent(state, V_TELESCOPE)

    elif deck_class_id == DECK_ZODIAC:
        state["deck"]["is_zodiac"] = True
        add_voucher_if_absent(state, V_TAROT_MERCHANT)
        add_voucher_if_absent(state, V_PLANET_MERCHANT)
        add_voucher_if_absent(state, V_OVERSTOCK_NORM)

    # All other decks (red/blue/yellow/green/black/painted/anaglyph/plasma/
    # ghost/challenge): keep the default standard deck.


def on_start_new_run(
    prev_state: dict[str, Any],
    step: dict[str, Any],
    zones: dict[str, Any],
) -> dict[str, Any]:
    state = default_state()
    state["deck_detected"] = True

    # Granularizer's canonical zone snapshot does NOT include CurrentDeck/CurrentStake
    # (those zones aren't in the snapshot map). Look in the raw event objects too.
    deck_obj = first_or_none(zones.get("CurrentDeck") or zones.get("current_deck"))
    stake_obj = first_or_none(zones.get("CurrentStake") or zones.get("current_stake"))

    if deck_obj is None or stake_obj is None:
        # Fall back to scanning the raw step.objects for these zones.
        for obj in step.get("objects") or []:
            zone = obj.get("zone")
            if deck_obj is None and zone == "CurrentDeck":
                deck_obj = obj
            elif stake_obj is None and zone == "CurrentStake":
                stake_obj = obj
            if deck_obj is not None and stake_obj is not None:
                break

    if stake_obj is not None:
        try:
            state["stake"] = int(stake_obj.get("class_id", DEFAULT_STAKE))
        except (TypeError, ValueError):
            state["stake"] = DEFAULT_STAKE

    if deck_obj is None:
        state["deck_detected"] = False
        return state

    try:
        deck_class_id = int(deck_obj["class_id"])
    except (KeyError, TypeError, ValueError):
        state["deck_detected"] = False
        return state

    state["deck"]["class_id"] = deck_class_id
    state["deck"]["name"] = deck_obj.get("name")  # optional metadata, often absent

    apply_deck_initialization(state, deck_class_id)
    return state


# ---------------------------------------------------------------------------
# 7) Action update functions
# ---------------------------------------------------------------------------

def on_select_blind(
    state: dict[str, Any],
    target_class_id: int | None,
    zones: dict[str, Any],
) -> dict[str, Any]:
    state["first_hand"] = True
    state["first_discard"] = True
    for hand_name in state["hands"]:
        state["hands"][hand_name]["played_this_round"] = 0
    state["swap_count"] = 0  # [INTERNAL]

    if target_class_id is None:
        return state

    if target_class_id == SMALL_BLIND:
        state["small_status"] = 0
    elif target_class_id == BIG_BLIND:
        state["big_status"] = 0
    elif is_boss_blind(target_class_id):
        if target_class_id not in state["bosses_used"]:
            state["bosses_used"].append(target_class_id)
        state["small_status"] = 1
        state["big_status"] = 1
        state["is_boss_blind_rerolled"] = False
        # ante_boss_blind is updated separately by _observe_ante_boss_blind.
    return state


def on_skip_blind(state: dict[str, Any], target_class_id: int | None) -> dict[str, Any]:
    state["skips"] += 1
    if target_class_id == SMALL_BLIND:
        state["small_status"] = 2
    elif target_class_id == BIG_BLIND:
        state["big_status"] = 2
    return state


def on_reroll_boss_blind(state: dict[str, Any]) -> dict[str, Any]:
    state["is_boss_blind_rerolled"] = True
    return state


def _apply_hand_level_consumable(state: dict[str, Any], target_class_id: int) -> bool:
    """
    Apply planet / black-hole hand-level effects.

    Returns True iff the consumable was a planet or Black Hole (hand level
    actually changed), False otherwise. Used by on_use_consumable and
    on_select_pack_item.
    """
    if target_class_id == BLACK_HOLE_CLASS:
        for hand_name in state["hands"]:
            state["hands"][hand_name]["level"] += 1
        return True
    hand_name = PLANET_TO_HAND.get(target_class_id)
    if hand_name is None:
        return False
    state["hands"][hand_name]["level"] += 1
    return True


def on_use_consumable(
    state: dict[str, Any],
    target_class_id: int | None,
    selected_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Resolve a UseConsumable / BuyAndUseShopConsumable / auto-used pack
    consumable. Updates:

    - ``last_tarot_planet`` for planets and tarots.
    - ``hands[*].level`` for planets and Black Hole.
    - ``ecto_minus`` for Ectoplasm.
    - ``tracked_deck_cards`` for card-targeting tarots/spectrals
      (chariot, death, devil, empress, hanged_man, heirophant, justice,
      lovers, magician, moon, star, strength, sun, tower, world,
      aura, cryptid, deja_vu, medium, talisman, trance).
    - ``unhandled_random_consumable_count`` (internal) for
      familiar/grim/immolate/incantation/ouija/sigil, whose random
      effects we cannot reconstruct deterministically.
    """
    if target_class_id is None:
        return state

    # Track the most recently used planet/tarot regardless of effect.
    if is_planet_or_tarot(target_class_id):
        state["last_tarot_planet"] = target_class_id

    # Hand-level bumps (planets + Black Hole). Black Hole is a spectral
    # so we set last_tarot_planet for the planets only (above).
    _apply_hand_level_consumable(state, target_class_id)

    if target_class_id == ECTO_CLASS:
        state["ecto_minus"] += 1

    # Card-targeting tarot/spectral effects mutate tracked_deck_cards.
    if target_class_id in CARD_CONSUMABLE_HANDLERS:
        apply_card_consumable(
            state["tracked_deck_cards"],
            target_class_id,
            selected_cards or [],
        )
    elif target_class_id in UNHANDLED_RANDOM_CONSUMABLES:
        # familiar/grim/immolate/incantation/ouija/sigil — destroy random
        # cards / convert hand cards to single random rank/suit. We can't
        # reconstruct the outcome from this stream alone; leave the deck
        # imprecise and bump the diagnostic counter.
        state["unhandled_random_consumable_count"] += 1

    return state


def on_select_pack_item(
    state: dict[str, Any],
    target_class_id: int | None,
    selected_object_payload: dict[str, Any] | None,
    selected_cards: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Resolve a SelectPackItem step.

    Three cases (resolved by target class id):

    1. Standard playing card (class_id 0..51, ``object_type='card'``):
       Append a normalized copy of the card (with its modifier/edition/seal
       intact) to ``tracked_deck_cards`` and trim to ``TRACKED_DECK_CAP``.

    2. Stone card (class_id == 78, ``object_type='modifier'``):
       Append a canonicalized stone-card entry (no rank/suit, modifier=m_stone).

    3. Anything else (planet, tarot, spectral, joker):
       Treated as auto-used: route through ``on_use_consumable`` so planets
       still bump hand levels, tarots/spectrals can mutate the deck via
       ``selected_cards``, and jokers are no-ops on persistent state.
    """
    if target_class_id is None:
        return state

    if target_class_id in STANDARD_CARD_CLASS_RANGE or target_class_id == STONE_CARD_CLASS:
        new_card = normalize_pack_card(selected_object_payload)
        if new_card is not None:
            state["tracked_deck_cards"].append(new_card)
            trim_tracked_deck(state["tracked_deck_cards"])
        return state

    return on_use_consumable(state, target_class_id, selected_cards)


def on_buy_shop_item(state: dict[str, Any], target_class_id: int | None) -> dict[str, Any]:
    if target_class_id is None:
        return state
    if target_class_id in VOUCHER_CLASS_RANGE:
        add_voucher_if_absent(state, target_class_id)
    return state


def on_cash_out(state: dict[str, Any], ocr: dict[str, Any]) -> dict[str, Any]:
    discards_left = ocr.get("discards_left")
    if isinstance(discards_left, int):
        state["unused_discards"] += discards_left
    # else: data-quality warning; leave unchanged.
    return state


def on_play_hand(state: dict[str, Any], _ocr: dict[str, Any]) -> dict[str, Any]:
    """Apply PlayHand without reading ``hand_and_level`` OCR.

    Live play and the policy do not receive the HUD ``hand_and_level`` string.
    Per-hand ``played`` / ``played_this_round`` / level bumps from scoring are
    therefore not inferred from OCR here; planets and Black Hole still update
    ``hands[*].level`` via ``on_use_consumable``.
    """
    state["hands_played"] += 1
    state["first_hand"] = False
    state["swap_count"] = 0  # [INTERNAL]
    return state


def on_discard_hand(state: dict[str, Any]) -> dict[str, Any]:
    state["first_discard"] = False
    state["swap_count"] = 0  # [INTERNAL]
    return state


def on_swap(state: dict[str, Any], action_label: str) -> dict[str, Any]:
    state["swap_count"] += 1  # [INTERNAL]
    state["last_swap"] = action_label  # [INTERNAL] e.g. "SWAP_1_5"
    return state


# ---------------------------------------------------------------------------
# Observation-driven update (NOT tied to any specific action)
# ---------------------------------------------------------------------------

def _observe_ante_boss_blind(state: dict[str, Any], zones: dict[str, Any]) -> None:
    """
    Track the most-recently observed boss blind in the BlindOffering zone.

    Implements the pseudocode note: "ante_boss_blind is set when
    BlindOfferings becomes visible, not [in SelectBlind]". We update on
    every step so the field reflects the current ante's boss as soon as
    the offering UI appears.
    """
    # New granularize schema uses canonical zone names ("BlindOffering");
    # legacy snapshots used the lowercase alias "blind_offerings".
    candidates = (
        zones.get("BlindOffering")
        or zones.get("blind_offerings")
        or []
    )
    for obj in candidates:
        cid = obj.get("class_id")
        if isinstance(cid, int) and is_boss_blind(cid):
            state["ante_boss_blind"] = cid
            return


# ---------------------------------------------------------------------------
# 5) Reducer entry point
# ---------------------------------------------------------------------------

def _zones_from_step(step: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Build a ``{zone_name: [objects]}`` dict from ``step.objects``.

    The new granularize schema (3.0.0) no longer emits a top-level
    ``zones`` dict, so callers reconstruct it from ``step.objects`` whose
    members carry canonical zone names (``CurrentDeck``, ``BlindOffering``,
    ``CurrentJokers``, ``PendingCards``, ...). Legacy snapshots that
    still carry a top-level ``zones`` dict pass through unchanged.
    """
    legacy = step.get("zones")
    if isinstance(legacy, dict) and legacy:
        return legacy
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in step.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        zone = obj.get("zone")
        if isinstance(zone, str):
            grouped.setdefault(zone, []).append(obj)
    return grouped


def apply_step(state: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    """
    Take the persistent state BEFORE a step and return the persistent state
    AFTER the step has been applied. This is what the model sees on step t+1.
    """
    action = step.get("action") or ""
    base_action = parse_base_action(action)
    selected = step.get("selected_object")
    target_class_id = extract_target_class_id(selected)
    target_object = extract_target_object(selected)
    # New granularize uses ``pending_cards``; older snapshots used ``selected_cards``.
    selected_cards = (
        step.get("pending_cards")
        or step.get("selected_cards")
        or []
    )
    ocr = step.get("state") or {}
    zones = _zones_from_step(step)

    if base_action == "StartNewRun":
        state = on_start_new_run(state, step, zones)
        # Run the observation update on the freshly-initialized state.
        _observe_ante_boss_blind(state, zones)
        return state

    # Observation-driven update happens before the per-action update so that
    # SelectBlind sees the correct ante_boss_blind in the same step.
    _observe_ante_boss_blind(state, zones)

    if base_action == "SelectBlind":
        state = on_select_blind(state, target_class_id, zones)
    elif base_action == "SkipBlind":
        state = on_skip_blind(state, target_class_id)
    elif base_action == "RerollBossBlind":
        state = on_reroll_boss_blind(state)
    elif base_action == "PlayHand":
        state = on_play_hand(state, ocr)
    elif base_action == "DiscardHand":
        state = on_discard_hand(state)
    elif base_action == "UseConsumable":
        state = on_use_consumable(state, target_class_id, selected_cards)
    elif base_action == "BuyAndUseShopConsumable":
        state = on_use_consumable(state, target_class_id, selected_cards)
    elif base_action == "SelectPackItem":
        state = on_select_pack_item(state, target_class_id, target_object, selected_cards)
    elif base_action == "BuyShopItem":
        state = on_buy_shop_item(state, target_class_id)
    elif base_action == "CashOut":
        state = on_cash_out(state, ocr)
    elif base_action == "SWAP":
        state = on_swap(state, action)
    # SelectCard, SkipPack, LeaveShop, RerollShop, SellItem have no
    # persistent reducer effects in this version.
    return state


# ---------------------------------------------------------------------------
# Convenience: per-run reduction
# ---------------------------------------------------------------------------

def reduce_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Walk every event in a granularized run and return the per-step
    state-BEFORE-action snapshots.

    Returns ``len(events)`` snapshots: ``snapshots[t]`` is the state the model
    sees as input for step ``t``. After applying the last step, the
    final post-state is discarded (it's never used as a model input within
    this run; downstream callers can recompute it via ``apply_step`` if
    they need it).
    """
    events = run.get("events") or []
    if not events:
        return []

    snapshots: list[dict[str, Any]] = []
    state = default_state()
    for step in events:
        snapshots.append(copy.deepcopy(state))
        state = apply_step(state, step)
    return snapshots


# ---------------------------------------------------------------------------
# Visibility filters used by tensorization & masking
# ---------------------------------------------------------------------------

def to_model_visible(state: dict[str, Any]) -> dict[str, Any]:
    """
    Return only the [MODEL-VISIBLE] subset of the state. This is the
    persistent feature payload tensorization should serialize.
    """
    return {k: state[k] for k in MODEL_VISIBLE_KEYS if k in state}


def to_full_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Return the full state (model-visible + internal). This is what the
    mask builder is allowed to read from (it may also use ``swap_count``,
    ``last_swap`` for the SWAP cap rules).
    """
    return dict(state)
