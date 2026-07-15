#!/usr/bin/env python3
"""
argument_spec.py
================
Canonical per-class_id constraints for consumable / pack-item targeting in
the branched autoregressive policy.

One source of truth consumed by:

- ``tensorize.py`` (validate that ``num_cards`` and target zone match the
  ground-truth label),
- ``model.py`` (clamp ``num_cards_logits`` to ``[min_cards, max_cards]`` and
  apply per-card attribute masks once the consumable / pack-item pointer is
  teacher-forced or predicted),
- ``live/agent_server.py`` (mirror the same masks at inference),
- ``validate_training_contract.py`` (assert dataset is internally consistent).

Schema
------
``ARGUMENT_SPEC[class_id] -> {
    "min_cards": int,
    "max_cards": int,
    "attribute_mask": None | "seal_blue" | "seal_gold" | "seal_purple"
                          | "seal_red" | "edition_any",
}``

Items not in the table default to ``{min_cards: 0, max_cards: 0,
attribute_mask: None}`` (most spectrals/planets/etc. do not consume cards).

Per-page card zone resolution
-----------------------------
For ``UseConsumable``, the card zone depends on ``page_name``:

- ``In_Blind`` -> ``CurrentHand``
- ``In_TarotSpectral_Pack`` -> ``TarotSpectralHand``

For ``SelectPackItem``, the card zone is ``TarotSpectralHand`` and applies
only when the parent subtype is ``selectpackitemtarot``; other pack items
(joker / planet / standard card) take ``min_cards == max_cards == 0``.

Inventory / global gates (e.g. ``c_hex`` needs a non-editioned joker;
``c_fool`` needs ``last_tarot_planet != 303``) live alongside this table
in ``INVENTORY_GATES`` so a single import gives the model all the per-item
legality information.

Sources: ``mask_schema.md`` sections 4.1-4.4 and
``granularize.REQUIRES_AT_LEAST_ONE_CARD``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARGUMENT_SPEC_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Per-class_id targeting rules
# ---------------------------------------------------------------------------

# Spectrals and tarots that consume playing cards. Cardinality follows
# mask_schema.md section 4.1. ``class_id`` 309 (c_justice) is present in
# granularize.REQUIRES_AT_LEAST_ONE_CARD but absent from mask_schema.md
# cardinality lists; gameplay-confirmed to require exactly one target card.
ARGUMENT_SPEC: dict[int, dict[str, Any]] = {
    # exactly 1 card
    249: {"min_cards": 1, "max_cards": 1, "attribute_mask": "edition_any"},   # c_aura
    251: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_cryptid
    252: {"min_cards": 1, "max_cards": 1, "attribute_mask": "seal_red"},      # c_deja_vu
    259: {"min_cards": 1, "max_cards": 1, "attribute_mask": "seal_purple"},   # c_medium
    263: {"min_cards": 1, "max_cards": 1, "attribute_mask": "seal_gold"},     # c_talisman
    264: {"min_cards": 1, "max_cards": 1, "attribute_mask": "seal_blue"},     # c_trance
    298: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_chariot
    300: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_devil
    309: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_justice
    310: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_lovers
    317: {"min_cards": 1, "max_cards": 1, "attribute_mask": None},            # c_tower
    # exactly 2 cards
    299: {"min_cards": 2, "max_cards": 2, "attribute_mask": None},            # c_death
    # up to 2 cards
    302: {"min_cards": 1, "max_cards": 2, "attribute_mask": None},            # c_empress
    304: {"min_cards": 1, "max_cards": 2, "attribute_mask": None},            # c_hanged_man
    305: {"min_cards": 1, "max_cards": 2, "attribute_mask": None},            # c_heirophant
    311: {"min_cards": 1, "max_cards": 2, "attribute_mask": None},            # c_magician
    314: {"min_cards": 1, "max_cards": 2, "attribute_mask": None},            # c_strength
    # up to 3 cards
    312: {"min_cards": 1, "max_cards": 3, "attribute_mask": None},            # c_moon
    313: {"min_cards": 1, "max_cards": 3, "attribute_mask": None},            # c_star
    315: {"min_cards": 1, "max_cards": 3, "attribute_mask": None},            # c_sun
    319: {"min_cards": 1, "max_cards": 3, "attribute_mask": None},            # c_world
}


# Global / inventory gates per consumable class_id. These do NOT affect
# card targeting but determine whether the family + item pointer pair is
# legal at all. ``mask_schema.md`` sections 4.2-4.4.
INVENTORY_GATES: dict[int, str] = {
    248: "joker_count_>=_1",          # c_ankh
    256: "hex_no_editioned_jokers",   # c_hex (compound rule)
    318: "joker_count_>=_1",          # c_wheel_of_fortune
    262: "needs_joker_slot",          # c_soul
    265: "needs_joker_slot",          # c_wraith
    308: "needs_joker_slot",          # c_judgement
    303: "fool_last_planet_not_303",  # c_fool
    301: "needs_consumable_slot",     # c_emperor
    307: "needs_consumable_slot",     # c_high_priestess
}


# Mapping from attribute_mask name -> object field comparison rule. The
# model / live server use this to mask hand-card tokens whose attribute
# would collide with the consumable's effect.
ATTRIBUTE_MASK_RULES: dict[str, dict[str, Any]] = {
    "seal_blue": {"field": "seal", "value": "blue_seal"},
    "seal_gold": {"field": "seal", "value": "gold_seal"},
    "seal_purple": {"field": "seal", "value": "purple_seal"},
    "seal_red": {"field": "seal", "value": "red_seal"},
    # ``edition_any`` masks any card whose edition is not None / not 0.
    "edition_any": {"field": "edition", "value": "__any_non_null__"},
}


# Sentinel returned for class_ids not in ``ARGUMENT_SPEC``.
_DEFAULT_SPEC: dict[str, Any] = {
    "min_cards": 0,
    "max_cards": 0,
    "attribute_mask": None,
}


def spec_for_class_id(class_id: int | None) -> dict[str, Any]:
    """Return the ``ArgumentSpec`` row for a consumable/pack-item class.

    Returns the no-card default for unknown / None ``class_id``.
    """
    if class_id is None:
        return dict(_DEFAULT_SPEC)
    return dict(ARGUMENT_SPEC.get(int(class_id), _DEFAULT_SPEC))


def inventory_gate_for_class_id(class_id: int | None) -> str | None:
    """Return the ``INVENTORY_GATES`` entry for a consumable class, if any."""
    if class_id is None:
        return None
    return INVENTORY_GATES.get(int(class_id))


# ---------------------------------------------------------------------------
# Per-page card-zone resolution
# ---------------------------------------------------------------------------

# UseConsumable resolves its card pool from the page name. Mirrors
# granularize.CARD_SELECTION_POOL (which is documented in
# granularization_schema.md section 5).
USE_CONSUMABLE_CARD_ZONE_BY_PAGE: dict[str, str] = {
    "In_Blind": "CurrentHand",
    "In_TarotSpectral_Pack": "TarotSpectralHand",
}

# SelectPackItem only uses card targets when the parent subtype is the
# "selectpackitemtarot" variant. The pool is always TarotSpectralHand.
SELECT_PACK_ITEM_TAROT_SUBTYPE = "selectpackitemtarot"
SELECT_PACK_ITEM_TAROT_CARD_ZONE = "TarotSpectralHand"


def card_zone_for_use_consumable(page_name: str | None) -> str | None:
    """Return the card pool zone for a ``UseConsumable`` at this page."""
    if page_name is None:
        return None
    return USE_CONSUMABLE_CARD_ZONE_BY_PAGE.get(page_name)


def card_zone_for_select_pack_item(subtype: str | None) -> str | None:
    """Return the card pool zone for a ``SelectPackItem`` of this subtype."""
    if subtype == SELECT_PACK_ITEM_TAROT_SUBTYPE:
        return SELECT_PACK_ITEM_TAROT_CARD_ZONE
    return None


# ---------------------------------------------------------------------------
# Cardinality bounds (global)
# ---------------------------------------------------------------------------

# MAX_CARDS_PER_DECISION is the union over all families: PlayHand /
# DiscardHand can sample up to 5 cards; tarot/spectral capped at 3. One
# shared tensor buffer is sized to the max.
MAX_CARDS_PER_DECISION_PLAY_DISCARD = 5
MAX_CARDS_PER_DECISION_TAROT_SPECTRAL = 3
MAX_CARDS_PER_DECISION = MAX_CARDS_PER_DECISION_PLAY_DISCARD


@dataclass(frozen=True)
class CardCardinality:
    """Per-family card count constraints."""

    family_kind: str
    min_cards: int
    max_cards: int
    card_zone: str | None


def cardinality_for_play_discard() -> CardCardinality:
    return CardCardinality(
        family_kind="card_seq",
        min_cards=1,
        max_cards=MAX_CARDS_PER_DECISION_PLAY_DISCARD,
        card_zone="CurrentHand",
    )


def cardinality_for_use_consumable(
    class_id: int | None, page_name: str | None
) -> CardCardinality:
    spec = spec_for_class_id(class_id)
    zone = card_zone_for_use_consumable(page_name)
    return CardCardinality(
        family_kind="chained_cards",
        min_cards=int(spec["min_cards"]),
        max_cards=int(spec["max_cards"]),
        card_zone=zone if spec["max_cards"] > 0 else None,
    )


def cardinality_for_select_pack_item(
    class_id: int | None, subtype: str | None
) -> CardCardinality:
    spec = spec_for_class_id(class_id)
    zone = card_zone_for_select_pack_item(subtype)
    return CardCardinality(
        family_kind="chained_cards",
        min_cards=int(spec["min_cards"]) if zone else 0,
        max_cards=int(spec["max_cards"]) if zone else 0,
        card_zone=zone,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def build_argument_spec_artifact() -> dict[str, Any]:
    """Return a fully-serializable snapshot of all argument-spec tables."""
    return {
        "schema_version": ARGUMENT_SPEC_SCHEMA_VERSION,
        "max_cards_per_decision": MAX_CARDS_PER_DECISION,
        "max_cards_per_decision_play_discard": MAX_CARDS_PER_DECISION_PLAY_DISCARD,
        "max_cards_per_decision_tarot_spectral": MAX_CARDS_PER_DECISION_TAROT_SPECTRAL,
        "argument_spec": {
            str(cid): {
                "min_cards": int(row["min_cards"]),
                "max_cards": int(row["max_cards"]),
                "attribute_mask": row["attribute_mask"],
            }
            for cid, row in sorted(ARGUMENT_SPEC.items())
        },
        "inventory_gates": {str(k): v for k, v in sorted(INVENTORY_GATES.items())},
        "attribute_mask_rules": ATTRIBUTE_MASK_RULES,
        "use_consumable_card_zone_by_page": USE_CONSUMABLE_CARD_ZONE_BY_PAGE,
        "select_pack_item_tarot_subtype": SELECT_PACK_ITEM_TAROT_SUBTYPE,
        "select_pack_item_tarot_card_zone": SELECT_PACK_ITEM_TAROT_CARD_ZONE,
    }


def write_argument_spec_artifact(path: Path) -> dict[str, Any]:
    payload = build_argument_spec_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_argument_spec_artifact(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Write data/argument_spec.json (canonical ArgumentSpec snapshot).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/argument_spec.json"),
        help="Output artifact path.",
    )
    args = ap.parse_args()
    payload = write_argument_spec_artifact(args.out)
    print(f"wrote {args.out.as_posix()}")
    print(f"  schema_version: {payload['schema_version']}")
    print(f"  argument_spec entries: {len(payload['argument_spec'])}")
    print(f"  inventory_gates entries: {len(payload['inventory_gates'])}")
    print(f"  max_cards_per_decision: {payload['max_cards_per_decision']}")


if __name__ == "__main__":
    main()
