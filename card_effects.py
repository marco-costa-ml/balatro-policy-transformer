#!/usr/bin/env python3
"""
card_effects.py
===============
Card-encoding helpers, deck builders, and per-consumable card-modification
handlers used by ``state_reducer.py`` to maintain ``tracked_deck_cards``.

Design
------
``tracked_deck_cards`` is a list (max length ``TRACKED_DECK_CAP``) of card
objects in the same shape the granularizer emits, minus volatile fields:

    {
        "class_id":   int,                 # 0..51 standard, 78 stone
        "object_type": "card",             # always "card" for tracked entries
        "card":       {rank, rank_index, suit, suit_index, is_ace, is_face} | None,
        "modifier":   "m_bonus" | "m_glass" | "m_gold" | "m_lucky"
                    | "m_mult" | "m_steel" | "m_stone" | "m_wild" | None,
        "edition":    "e_foil" | "e_holo" | "e_negative" | "e_polychrome" | None,
        "seal":       "blue_seal" | "gold_seal" | "purple_seal" | "red_seal" | None,
        "stickers":   list[str],
    }

A stone card emitted from a pack arrives with ``class_id == 78`` and
``object_type == "modifier"``; we canonicalize it to
``object_type == "card"`` with ``modifier="m_stone"`` and ``card=None``.

Consumable effects fall into two categories:

1. **Targeted** (selected by player; granularized as commit step with
   ``selected_cards``): tarots and a subset of spectrals. We resolve each
   selected card to its closest match in the tracked deck and mutate.
   (See ``CARD_CONSUMABLE_HANDLERS``.)
2. **Random / hand-wide** (no selected_cards in the granularized step):
   familiar, grim, immolate, incantation, ouija, sigil. These are
   intentionally NOT handled in v1 because we cannot reconstruct the
   random outcome from the granularized stream alone; they're listed in
   ``UNHANDLED_RANDOM_CONSUMABLES`` for downstream reporting.
"""

from __future__ import annotations

import copy
from typing import Any


# ---------------------------------------------------------------------------
# Card encoding (mirrors data/metadata_map.csv class_ids 0..51)
# ---------------------------------------------------------------------------

SUIT_NAMES: list[str] = ["Spades", "Hearts", "Diamonds", "Clubs"]
RANK_NAMES: list[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
FACE_RANK_INDICES: frozenset[int] = frozenset({10, 11, 12})  # J, Q, K
ACE_RANK_INDEX: int = 0

STANDARD_CARD_CLASS_RANGE = range(0, 52)
STONE_CARD_CLASS = 78  # also the class_id for the m_stone enhancement itself

# Abandoned-deck face-card class IDs (J/Q/K of each suit).
ABANDONED_FACE_CARD_CLASS_IDS: frozenset[int] = frozenset(
    {10, 11, 12, 23, 24, 25, 36, 37, 38, 49, 50, 51}
)

# How many tracked cards we retain per state. Initial decks are 52; cards
# can be created via cryptid / packs / random spectrals so we cap to keep
# the eventual tensor input bounded.
TRACKED_DECK_CAP: int = 75


# ---------------------------------------------------------------------------
# Hand-level consumables (planets + black hole)
# ---------------------------------------------------------------------------

PLANET_TO_HAND: dict[int, str] = {
    236: "Flush House",      # c_ceres
    237: "Full House",       # c_earth
    238: "Flush Five",       # c_eris
    239: "Flush",            # c_jupiter
    240: "Four of a Kind",   # c_mars
    241: "Pair",             # c_mercury
    242: "Straight Flush",   # c_neptune
    243: "Five of a Kind",   # c_planet_x
    244: "High Card",        # c_pluto
    245: "Straight",         # c_saturn
    246: "Two Pair",         # c_uranus
    247: "Three of a Kind",  # c_venus
}

BLACK_HOLE_CLASS = 250  # increments level of every poker hand by 1


# Random/hand-wide spectrals we acknowledge but don't model in v1.
# Their outcomes (random destroyed cards, random new cards, single-rank /
# single-suit hand conversion) aren't reconstructible from the granularized
# stream alone.
UNHANDLED_RANDOM_CONSUMABLES: frozenset[int] = frozenset(
    {
        254,  # c_familiar    - destroy 1 random + add 3 enhanced face cards
        255,  # c_grim        - destroy 1 random + add 2 enhanced aces
        257,  # c_immolate    - destroy 5 random cards in hand
        258,  # c_incantation - destroy 1 random + add 4 enhanced numbered cards
        260,  # c_ouija       - convert all hand cards to one random rank
        261,  # c_sigil       - convert all hand cards to one random suit
    }
)


# ---------------------------------------------------------------------------
# Card factories
# ---------------------------------------------------------------------------

def class_id_to_card_meta(class_id: int) -> dict[str, Any] | None:
    """Return the nested ``card`` sub-dict for a standard playing card class id."""
    if class_id not in STANDARD_CARD_CLASS_RANGE:
        return None
    rank_index = class_id % 13
    suit_index = class_id // 13
    return {
        "rank": RANK_NAMES[rank_index],
        "rank_index": rank_index,
        "suit": SUIT_NAMES[suit_index],
        "suit_index": suit_index,
        "is_ace": rank_index == ACE_RANK_INDEX,
        "is_face": rank_index in FACE_RANK_INDICES,
    }


def make_standard_card(class_id: int) -> dict[str, Any]:
    """Construct a default playing-card entry (no modifier/edition/seal)."""
    return {
        "class_id": class_id,
        "object_type": "card",
        "card": class_id_to_card_meta(class_id),
        "modifier": None,
        "edition": None,
        "seal": None,
        "stickers": [],
    }


def make_stone_card() -> dict[str, Any]:
    """Construct a bare stone card (class_id=78, no rank/suit, modifier=m_stone)."""
    return {
        "class_id": STONE_CARD_CLASS,
        "object_type": "card",
        "card": None,
        "modifier": "m_stone",
        "edition": None,
        "seal": None,
        "stickers": [],
    }


def normalize_pack_card(selected_object: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Normalize a SelectPackItem ``selected_object.object`` payload into the
    canonical tracked-deck card shape.

    Accepts either:
    - a standard playing card (``class_id 0..51, object_type='card'``)
    - a stone card (``class_id == 78, object_type='modifier'``)
    Returns None for any other shape.
    """
    if not isinstance(selected_object, dict):
        return None
    cid = selected_object.get("class_id")
    if not isinstance(cid, int):
        return None

    if cid in STANDARD_CARD_CLASS_RANGE:
        card_meta = selected_object.get("card") or class_id_to_card_meta(cid)
        return {
            "class_id": cid,
            "object_type": "card",
            "card": copy.deepcopy(card_meta) if card_meta else class_id_to_card_meta(cid),
            "modifier": selected_object.get("modifier"),
            "edition": selected_object.get("edition"),
            "seal": selected_object.get("seal"),
            "stickers": list(selected_object.get("stickers") or []),
        }

    if cid == STONE_CARD_CLASS:
        # Pack-spawned stone card has object_type='modifier' and no rank/suit;
        # canonicalize to a card entry with m_stone modifier.
        return {
            "class_id": STONE_CARD_CLASS,
            "object_type": "card",
            "card": None,
            "modifier": "m_stone",
            "edition": selected_object.get("edition"),
            "seal": selected_object.get("seal"),
            "stickers": list(selected_object.get("stickers") or []),
        }

    return None


# ---------------------------------------------------------------------------
# Initial deck builders
# ---------------------------------------------------------------------------

def build_standard_deck() -> list[dict[str, Any]]:
    """Default 52-card playing deck."""
    return [make_standard_card(cid) for cid in range(0, 52)]


def build_abandoned_deck() -> list[dict[str, Any]]:
    """Standard 52 minus J/Q/K of every suit -> 40 cards."""
    return [
        make_standard_card(cid)
        for cid in range(0, 52)
        if cid not in ABANDONED_FACE_CARD_CLASS_IDS
    ]


def build_checkered_deck() -> list[dict[str, Any]]:
    """Two copies each of every spade and every heart -> 52 cards."""
    deck: list[dict[str, Any]] = []
    for cid in range(0, 13):  # spades
        deck.append(make_standard_card(cid))
        deck.append(make_standard_card(cid))
    for cid in range(13, 26):  # hearts
        deck.append(make_standard_card(cid))
        deck.append(make_standard_card(cid))
    return deck


def trim_tracked_deck(deck: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Truncate the tracked-deck list in place to ``TRACKED_DECK_CAP`` (FIFO)."""
    if len(deck) > TRACKED_DECK_CAP:
        del deck[TRACKED_DECK_CAP:]
    return deck


# ---------------------------------------------------------------------------
# Card matching for targeted-consumable effects
# ---------------------------------------------------------------------------

def _score_match(tracked: dict[str, Any], target: dict[str, Any]) -> int:
    """Higher = better match; -1 if class_id disagrees."""
    if tracked.get("class_id") != target.get("class_id"):
        return -1
    score = 0
    for key in ("modifier", "edition", "seal"):
        if tracked.get(key) == target.get(key):
            score += 1
    return score


def find_closest_in_deck_index(
    deck: list[dict[str, Any]],
    target_card: dict[str, Any],
) -> int | None:
    """
    Return the index of the best-matching card in the deck.

    Strategy:
    - Prefer same class_id with most matching of {modifier, edition, seal}.
    - Tie-breaks deterministically: first occurrence wins.
    - Falls back to same rank+suit (via the nested ``card`` dict) if no
      class_id match (handles stone-converted cards whose class_id changed).
    - Returns None if nothing remotely matches.
    """
    if not deck or not isinstance(target_card, dict):
        return None

    best_idx: int | None = None
    best_score = -1
    for i, card in enumerate(deck):
        score = _score_match(card, target_card)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx is not None and best_score >= 0:
        return best_idx

    # Fallback: same rank + suit (for tower-converted cards that became
    # stone but whose deck entry still references the original class_id).
    target_meta = target_card.get("card") or {}
    t_rank_index = target_meta.get("rank_index")
    t_suit_index = target_meta.get("suit_index")
    if t_rank_index is None or t_suit_index is None:
        return None
    for i, card in enumerate(deck):
        cm = card.get("card") or {}
        if cm.get("rank_index") == t_rank_index and cm.get("suit_index") == t_suit_index:
            return i
    return None


def find_closest_in_deck(
    deck: list[dict[str, Any]],
    target_card: dict[str, Any],
) -> dict[str, Any] | None:
    """Convenience wrapper around ``find_closest_in_deck_index``."""
    idx = find_closest_in_deck_index(deck, target_card)
    return deck[idx] if idx is not None else None


# ---------------------------------------------------------------------------
# Per-consumable card-modification handlers
# ---------------------------------------------------------------------------

def _set_modifier(deck: list[dict[str, Any]], target: dict[str, Any], modifier: str) -> None:
    """Set ``modifier`` on the closest deck match."""
    match = find_closest_in_deck(deck, target)
    if match is not None:
        match["modifier"] = modifier


def _set_seal(deck: list[dict[str, Any]], target: dict[str, Any], seal: str) -> None:
    """Set ``seal`` on the closest deck match."""
    match = find_closest_in_deck(deck, target)
    if match is not None:
        match["seal"] = seal


def _change_suit(deck: list[dict[str, Any]], target: dict[str, Any], new_suit: str) -> None:
    """
    Re-encode the closest matching card to ``new_suit``, preserving rank.
    Stone cards (no rank/suit) are skipped silently.
    """
    match = find_closest_in_deck(deck, target)
    if match is None or match.get("card") is None:
        return
    new_suit_index = SUIT_NAMES.index(new_suit)
    rank_index = match["card"]["rank_index"]
    new_class_id = new_suit_index * 13 + rank_index
    match["class_id"] = new_class_id
    match["card"]["suit"] = new_suit
    match["card"]["suit_index"] = new_suit_index


def _increase_rank(deck: list[dict[str, Any]], target: dict[str, Any]) -> None:
    """Bump rank_index by 1 (mod 13) on the closest matching card; preserve suit."""
    match = find_closest_in_deck(deck, target)
    if match is None or match.get("card") is None:
        return
    old_rank_index = match["card"]["rank_index"]
    new_rank_index = (old_rank_index + 1) % 13
    suit_index = match["card"]["suit_index"]
    new_class_id = suit_index * 13 + new_rank_index
    match["class_id"] = new_class_id
    match["card"]["rank_index"] = new_rank_index
    match["card"]["rank"] = RANK_NAMES[new_rank_index]
    match["card"]["is_ace"] = new_rank_index == ACE_RANK_INDEX
    match["card"]["is_face"] = new_rank_index in FACE_RANK_INDICES


# Per-tarot/spectral handlers. Each takes (deck, selected_cards) and mutates
# the deck in place. The granularizer guarantees selected_cards order matches
# in-game player selection order.

def _on_chariot(deck, selected_cards):
    for c in selected_cards[:1]:
        _set_modifier(deck, c, "m_steel")


def _on_devil(deck, selected_cards):
    for c in selected_cards[:1]:
        _set_modifier(deck, c, "m_gold")


def _on_empress(deck, selected_cards):
    for c in selected_cards[:2]:
        _set_modifier(deck, c, "m_mult")


def _on_heirophant(deck, selected_cards):
    for c in selected_cards[:2]:
        _set_modifier(deck, c, "m_bonus")


def _on_justice(deck, selected_cards):
    for c in selected_cards[:1]:
        _set_modifier(deck, c, "m_glass")


def _on_lovers(deck, selected_cards):
    for c in selected_cards[:1]:
        _set_modifier(deck, c, "m_wild")


def _on_magician(deck, selected_cards):
    for c in selected_cards[:2]:
        _set_modifier(deck, c, "m_lucky")


def _on_tower(deck, selected_cards):
    for c in selected_cards[:1]:
        _set_modifier(deck, c, "m_stone")


def _on_moon(deck, selected_cards):
    for c in selected_cards[:3]:
        _change_suit(deck, c, "Clubs")


def _on_star(deck, selected_cards):
    for c in selected_cards[:3]:
        _change_suit(deck, c, "Diamonds")


def _on_sun(deck, selected_cards):
    for c in selected_cards[:3]:
        _change_suit(deck, c, "Hearts")


def _on_world(deck, selected_cards):
    for c in selected_cards[:3]:
        _change_suit(deck, c, "Spades")


def _on_strength(deck, selected_cards):
    for c in selected_cards[:2]:
        _increase_rank(deck, c)


def _on_hanged_man(deck, selected_cards):
    """Destroy up to 2 selected cards from the deck."""
    # Destroy by class+attr match; iterate from highest index to avoid shift bugs.
    indices: list[int] = []
    for c in selected_cards[:2]:
        idx = find_closest_in_deck_index(deck, c)
        if idx is not None and idx not in indices:
            indices.append(idx)
    for idx in sorted(indices, reverse=True):
        del deck[idx]


def _on_death(deck, selected_cards):
    """Death: convert the LEFT (first) selected card to a copy of the RIGHT (second)."""
    if len(selected_cards) < 2:
        return
    left, right = selected_cards[0], selected_cards[1]
    left_match = find_closest_in_deck(deck, left)
    if left_match is None:
        return
    left_match["class_id"] = right.get("class_id", left_match["class_id"])
    left_match["card"] = copy.deepcopy(right.get("card"))
    left_match["modifier"] = right.get("modifier")
    left_match["edition"] = right.get("edition")
    left_match["seal"] = right.get("seal")
    left_match["stickers"] = list(right.get("stickers") or [])


def _on_aura(deck, selected_cards):
    """
    Adds Foil/Holographic/Polychrome edition to 1 selected card.
    The exact edition is randomized in-game; default to e_foil deterministically.
    """
    if not selected_cards:
        return
    match = find_closest_in_deck(deck, selected_cards[0])
    if match is not None and match.get("edition") is None:
        match["edition"] = "e_foil"


def _on_cryptid(deck, selected_cards):
    """Create 2 exact copies of the 1 selected card (including all attributes)."""
    if not selected_cards:
        return
    match = find_closest_in_deck(deck, selected_cards[0])
    if match is None:
        return
    for _ in range(2):
        deck.append(copy.deepcopy(match))
    trim_tracked_deck(deck)


def _on_deja_vu(deck, selected_cards):
    if selected_cards:
        _set_seal(deck, selected_cards[0], "red_seal")


def _on_medium(deck, selected_cards):
    if selected_cards:
        _set_seal(deck, selected_cards[0], "purple_seal")


def _on_talisman(deck, selected_cards):
    if selected_cards:
        _set_seal(deck, selected_cards[0], "gold_seal")


def _on_trance(deck, selected_cards):
    if selected_cards:
        _set_seal(deck, selected_cards[0], "blue_seal")


# Dispatch table. Class ids match REQUIRES_CARD_SELECTION in granularize.py.
CARD_CONSUMABLE_HANDLERS: dict[int, Any] = {
    # Tarots
    298: _on_chariot,
    299: _on_death,
    300: _on_devil,
    302: _on_empress,
    304: _on_hanged_man,
    305: _on_heirophant,
    309: _on_justice,
    310: _on_lovers,
    311: _on_magician,
    312: _on_moon,
    313: _on_star,
    314: _on_strength,
    315: _on_sun,
    317: _on_tower,
    319: _on_world,
    # Spectrals
    249: _on_aura,
    251: _on_cryptid,
    252: _on_deja_vu,
    259: _on_medium,
    263: _on_talisman,
    264: _on_trance,
}


def apply_card_consumable(
    deck: list[dict[str, Any]],
    consumable_class_id: int,
    selected_cards: list[dict[str, Any]],
) -> bool:
    """
    Dispatch a card-targeting consumable. Mutates ``deck`` in place.

    Returns True if a handler ran (regardless of whether the deck changed),
    False if the consumable id has no card-targeting handler. Random/hand-wide
    spectrals (familiar/grim/immolate/incantation/ouija/sigil) intentionally
    return False; the caller should account for the deck imprecision.
    """
    handler = CARD_CONSUMABLE_HANDLERS.get(consumable_class_id)
    if handler is None:
        return False
    handler(deck, selected_cards or [])
    trim_tracked_deck(deck)
    return True
