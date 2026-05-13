# Balatro State Reducer Pseudocode

## 1) Purpose and Source of Truth

This file is the implementation-level companion to `state_schema.md`.

It specifies, in Python-flavored pseudocode:
- the persistent state reducer (`apply_step`),
- StartNewRun initialization,
- per-action update functions,
- the `hand_and_level` OCR parser,
- visibility-tag enforcement (`[MODEL-VISIBLE]`, `[OBSERVATION]`, `[INTERNAL]`).

The contract:
- `apply_step(state, step)` is pure: same `state` and `step` produce the same returned state.
- Reducer state is reset at every `StartNewRun`.
- Internal fields (`swap_count`, `last_swap`, `prev_jokers_all`, `deck_detected`, `hand_and_level_unparsed_count`, `unhandled_random_consumable_count`) must NOT be passed to tensorization.

Card-encoding helpers, deck builders, and per-consumable card-modification logic live in `card_effects.py` (referenced throughout this doc).

---

## 2) Visibility Constants

```python
# Used by tensorization to filter which fields it serializes.
MODEL_VISIBLE_KEYS = {
    "deck", "stake",
    "tracked_deck_cards", "deck_modifiers",
    "last_tarot_planet", "ecto_minus", "skips",
    "hands_played", "unused_discards",
    "first_hand", "first_discard",
    "vouchers_redeemed", "bosses_used",
    "ante_boss_blind", "small_status", "big_status",
    "is_boss_blind_rerolled",
    "hands",
}

INTERNAL_KEYS = {
    "swap_count",
    "last_swap",
    "deck_detected",
    "prev_jokers_all",
    "hand_and_level_unparsed_count",
    "unhandled_random_consumable_count",
}

# Per-step OBSERVATION fields are not stored in the reducer; they are
# read directly from the granularized step's `state` and `zones`.
```

---

## 3) Class ID Constants

```python
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
BOSS_BLIND_RANGE = range(370, 400)  # 370..399; exclude 371 and 394

def is_boss_blind(class_id: int) -> bool:
    return class_id in BOSS_BLIND_RANGE and class_id not in (BIG_BLIND, SMALL_BLIND)

# consumables
PLANET_RANGE = range(236, 248)
SPECTRAL_RANGE = range(248, 266)
TAROT_RANGE = range(298, 320)
ECTO_CLASS = 253
FOOL_CLASS = 303
BLACK_HOLE_CLASS = 250

def is_planet_or_tarot(class_id: int) -> bool:
    return class_id in PLANET_RANGE or class_id in TAROT_RANGE

# Hand-level consumables (planet -> poker hand). Black Hole bumps them all.
PLANET_TO_HAND = {
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

# tracked_deck_cards
STANDARD_CARD_CLASS_RANGE = range(0, 52)
STONE_CARD_CLASS = 78
TRACKED_DECK_CAP = 75

# Card-targeting consumables that mutate tracked_deck_cards via the
# closest-match rule. (See card_effects.CARD_CONSUMABLE_HANDLERS.)
CARD_CONSUMABLE_CLASSES = {
    # Tarots
    298,  # c_chariot      m_steel
    299,  # c_death        left becomes copy of right
    300,  # c_devil        m_gold
    302,  # c_empress      m_mult x2
    304,  # c_hanged_man   destroy up to 2
    305,  # c_heirophant   m_bonus x2
    309,  # c_justice      m_glass
    310,  # c_lovers       m_wild
    311,  # c_magician     m_lucky x2
    312,  # c_moon         suit -> Clubs (up to 3)
    313,  # c_star         suit -> Diamonds (up to 3)
    314,  # c_strength     rank +1 (up to 2)
    315,  # c_sun          suit -> Hearts (up to 3)
    317,  # c_tower        m_stone
    319,  # c_world        suit -> Spades (up to 3)
    # Spectrals
    249,  # c_aura         e_foil placeholder
    251,  # c_cryptid      duplicate target x2
    252,  # c_deja_vu      red_seal
    259,  # c_medium       purple_seal
    263,  # c_talisman     gold_seal
    264,  # c_trance       blue_seal
}

# Random / hand-wide spectrals NOT modeled in v1.
UNHANDLED_RANDOM_CONSUMABLES = {254, 255, 257, 258, 260, 261}

# stickers
STICKER_RENTAL = 367
STICKER_PERISHABLE = 368
STICKER_ETERNAL = 369

# default stake when CurrentStake[0] is missing
DEFAULT_STAKE = 268
```

---

## 4) Initial / Default State

```python
POKER_HANDS = [
    "Flush Five", "Flush House", "Five of a Kind",
    "Straight Flush", "Four of a Kind", "Full House",
    "Flush", "Straight", "Three of a Kind",
    "Two Pair", "Pair", "High Card",
]

def make_standard_card(class_id: int) -> dict:
    """Build a default playing-card entry (no modifier/edition/seal).
    Each entry mirrors the granularizer's card schema (minus volatile fields):

        {
            "class_id":    int,
            "object_type": "card",
            "card":        {rank, rank_index, suit, suit_index, is_ace, is_face} | None,
            "modifier":    str | None,
            "edition":     str | None,
            "seal":        str | None,
            "stickers":    list[str],
        }

    See card_effects.make_standard_card for the executable version."""
    rank_index = class_id % 13
    suit_index = class_id // 13
    rank = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"][rank_index]
    suit = ["Spades","Hearts","Diamonds","Clubs"][suit_index]
    return {
        "class_id": class_id,
        "object_type": "card",
        "card": {
            "rank": rank, "rank_index": rank_index,
            "suit": suit, "suit_index": suit_index,
            "is_ace": rank_index == 0,
            "is_face": rank_index in {10, 11, 12},
        },
        "modifier": None, "edition": None, "seal": None, "stickers": [],
    }

def build_standard_deck() -> list[dict]:
    return [make_standard_card(cid) for cid in range(0, 52)]

def default_state() -> dict:
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
        # FIFO list of card objects, capped at TRACKED_DECK_CAP=75 entries.
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
```

---

## 5) Reducer Entry Point

```python
def apply_step(state: dict, step: dict) -> dict:
    """
    Take the persistent state BEFORE a step and return the persistent state
    AFTER the step has been applied. This is what the model sees on step t+1.

    `step` is a granularized step (one action). The model input for step t
    uses `state` BEFORE this call.
    """
    action = step["action"]                      # e.g. "PlayHand", "SWAP_1_3"
    base_action = parse_base_action(action)      # "PlayHand", "SWAP", ...
    selected = step.get("selected_object")
    target_class_id = extract_target_class_id(selected)
    target_object = extract_target_object(selected)        # full object payload
    selected_cards = step.get("selected_cards") or []      # in-hand targets, if any
    ocr = step.get("state", {}) or {}
    zones = step.get("zones", {}) or {}

    if base_action == "StartNewRun":
        return on_start_new_run(state, step, zones)

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
    # SelectCard, SkipPack, LeaveShop, RerollShop, SellItem
    # have no persistent reducer effects in this version.

    return state
```

---

## 6) StartNewRun Initialization

```python
def on_start_new_run(prev_state: dict, step: dict, zones: dict) -> dict:
    state = default_state()
    state["deck_detected"] = True

    deck_obj = first_or_none(zones.get("CurrentDeck") or zones.get("current_deck"))
    stake_obj = first_or_none(zones.get("CurrentStake") or zones.get("current_stake"))

    if stake_obj is not None:
        state["stake"] = int(stake_obj.get("class_id", DEFAULT_STAKE))

    if deck_obj is None:
        state["deck_detected"] = False
        return state  # standard defaults

    deck_class_id = int(deck_obj["class_id"])
    state["deck"]["class_id"] = deck_class_id
    state["deck"]["name"] = deck_obj.get("name")  # optional metadata

    apply_deck_initialization(state, deck_class_id)
    return state


def build_abandoned_deck() -> list[dict]:
    """Standard 52 minus J/Q/K of every suit -> 40 cards."""
    face_class_ids = {10, 11, 12, 23, 24, 25, 36, 37, 38, 49, 50, 51}
    return [make_standard_card(cid) for cid in range(0, 52) if cid not in face_class_ids]

def build_checkered_deck() -> list[dict]:
    """Two copies each of every spade and every heart -> 52 cards."""
    deck = []
    for cid in range(0, 13):    # spades
        deck.append(make_standard_card(cid))
        deck.append(make_standard_card(cid))
    for cid in range(13, 26):   # hearts
        deck.append(make_standard_card(cid))
        deck.append(make_standard_card(cid))
    return deck

def apply_deck_initialization(state: dict, deck_class_id: int) -> None:
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


def add_voucher_if_absent(state: dict, voucher_class_id: int) -> None:
    if voucher_class_id not in state["vouchers_redeemed"]:
        state["vouchers_redeemed"].append(voucher_class_id)
```

---

## 7) Action Update Functions

```python
def on_select_blind(state: dict, target_class_id: int | None, zones: dict) -> dict:
    state["first_hand"] = True
    state["first_discard"] = True
    for hand_name in state["hands"]:
        state["hands"][hand_name]["played_this_round"] = 0
    state["swap_count"] = 0  # [INTERNAL] reset per masking_schema_disorganized

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
        # ante_boss_blind is set when BlindOfferings becomes visible, not here.
    return state


def on_skip_blind(state: dict, target_class_id: int | None) -> dict:
    state["skips"] += 1
    if target_class_id == SMALL_BLIND:
        state["small_status"] = 2
    elif target_class_id == BIG_BLIND:
        state["big_status"] = 2
    return state


def on_reroll_boss_blind(state: dict) -> dict:
    state["is_boss_blind_rerolled"] = True
    return state


def _apply_hand_level_consumable(state: dict, target_class_id: int) -> bool:
    """Planet -> bump the corresponding poker hand by 1.
    Black Hole (250) -> bump every poker hand by 1.
    Returns True iff this consumable is a hand-level effect."""
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
    state: dict,
    target_class_id: int | None,
    selected_cards: list[dict],
) -> dict:
    """
    Resolve UseConsumable / BuyAndUseShopConsumable / auto-used pack consumable.

    Effects (applied in order):
      1. last_tarot_planet for planets and tarots.
      2. hands[*].level for planets and Black Hole.
      3. ecto_minus for Ectoplasm (253).
      4. tracked_deck_cards mutation for card-targeting tarots/spectrals
         (see CARD_CONSUMABLE_CLASSES + card_effects.CARD_CONSUMABLE_HANDLERS).
      5. unhandled_random_consumable_count++ for random/hand-wide spectrals
         (UNHANDLED_RANDOM_CONSUMABLES) — deck stays unchanged in v1.
    """
    if target_class_id is None:
        return state

    if is_planet_or_tarot(target_class_id):
        state["last_tarot_planet"] = target_class_id

    _apply_hand_level_consumable(state, target_class_id)

    if target_class_id == ECTO_CLASS:
        state["ecto_minus"] += 1

    if target_class_id in CARD_CONSUMABLE_CLASSES:
        # Dispatched to per-consumable handlers in card_effects.py. Each handler
        # walks selected_cards, finds the closest match in tracked_deck_cards,
        # and mutates (or destroys, or duplicates) that entry.
        apply_card_consumable(state["tracked_deck_cards"], target_class_id, selected_cards)
    elif target_class_id in UNHANDLED_RANDOM_CONSUMABLES:
        state["unhandled_random_consumable_count"] += 1

    return state


def on_select_pack_item(
    state: dict,
    target_class_id: int | None,
    target_object: dict | None,
    selected_cards: list[dict],
) -> dict:
    """
    Three cases by target class:

    1. Standard playing card (class_id 0..51, object_type='card'):
       Append a normalized card object (preserving modifier/edition/seal) to
       tracked_deck_cards and trim to TRACKED_DECK_CAP.

    2. Stone card (class_id == 78, object_type='modifier'):
       Append a canonicalized stone-card entry (no rank/suit, modifier=m_stone).

    3. Anything else (planet/tarot/spectral/joker):
       Treat as auto-used and route to on_use_consumable.
    """
    if target_class_id is None:
        return state

    if target_class_id in STANDARD_CARD_CLASS_RANGE or target_class_id == STONE_CARD_CLASS:
        new_card = normalize_pack_card(target_object)  # see card_effects.py
        if new_card is not None:
            state["tracked_deck_cards"].append(new_card)
            trim_tracked_deck(state["tracked_deck_cards"])
        return state

    return on_use_consumable(state, target_class_id, selected_cards)


def on_buy_shop_item(state: dict, target_class_id: int | None) -> dict:
    if target_class_id is None:
        return state
    if target_class_id in VOUCHER_CLASS_RANGE:
        add_voucher_if_absent(state, target_class_id)
    return state


def on_cash_out(state: dict, ocr: dict) -> dict:
    discards_left = ocr.get("discards_left")
    if isinstance(discards_left, int):
        state["unused_discards"] += discards_left
    # else: data-quality warning; leave unchanged
    return state


def on_play_hand(state: dict, ocr: dict) -> dict:
    state["hands_played"] += 1
    state["first_hand"] = False
    state["swap_count"] = 0  # [INTERNAL] reset per masking_schema_disorganized

    raw = ocr.get("hand_and_level_raw") or ocr.get("hand_and_level")
    parsed = parse_hand_and_level(raw)
    if parsed is None:
        state["hand_and_level_unparsed_count"] += 1  # [INTERNAL]
        return state

    hand_name, level = parsed
    entry = state["hands"][hand_name]
    entry["level"] = max(entry["level"], level)
    entry["played"] += 1
    entry["played_this_round"] += 1
    return state


def on_discard_hand(state: dict) -> dict:
    state["first_discard"] = False
    state["swap_count"] = 0  # [INTERNAL]
    return state


def on_swap(state: dict, action_label: str) -> dict:
    state["swap_count"] += 1            # [INTERNAL]
    state["last_swap"] = action_label   # [INTERNAL] e.g. "SWAP_1_5"
    return state
```

---

## 7.5) Card-Effect Handlers (`card_effects.py`)

The card-modifying tarot/spectral logic lives in `card_effects.py`. Every
handler takes `(deck, selected_cards)` and mutates the deck in place via
the closest-match rule:

```python
def find_closest_in_deck(deck: list[dict], target_card: dict) -> dict | None:
    """Pick the entry with the same class_id and the most-matching of
    {modifier, edition, seal}. First occurrence wins on ties.
    Falls back to (rank_index, suit_index) match when class_id disagrees
    (handles tower-converted cards). Returns None if nothing matches."""
```

Per-consumable handler summary (full effect table in `state_schema.md` §5.5):

| Class ID | Effect on closest match |
| --- | --- |
| `298 c_chariot` / `300 c_devil` / `309 c_justice` / `310 c_lovers` / `317 c_tower` | set `modifier = m_steel/m_gold/m_glass/m_wild/m_stone` (1 card) |
| `302 c_empress` / `305 c_heirophant` / `311 c_magician` | set `modifier = m_mult/m_bonus/m_lucky` (up to 2 cards) |
| `312 c_moon` / `313 c_star` / `315 c_sun` / `319 c_world` | re-encode suit to Clubs/Diamonds/Hearts/Spades (up to 3 cards) |
| `314 c_strength` | bump rank_index by 1 (mod 13), recompute class_id (up to 2 cards) |
| `304 c_hanged_man` | delete the matching entries (up to 2 cards) |
| `299 c_death` | left selected card becomes a copy of right selected card |
| `249 c_aura` | set `edition = e_foil` if currently None (placeholder for the random Foil/Holo/Polychrome roll) |
| `251 c_cryptid` | append two `deepcopy(matched_card)` entries |
| `252 c_deja_vu` / `259 c_medium` / `263 c_talisman` / `264 c_trance` | set `seal = red_seal/purple_seal/gold_seal/blue_seal` |

After every effect (and after every append from `on_select_pack_item`), the
deck is truncated to at most `TRACKED_DECK_CAP = 75` entries (FIFO: keep the
first 75).

The random / hand-wide spectrals (`254 c_familiar`, `255 c_grim`, `257 c_immolate`,
`258 c_incantation`, `260 c_ouija`, `261 c_sigil`) are NOT modeled in v1: we cannot
reconstruct their random outcomes from the granularized stream alone. The
reducer increments `unhandled_random_consumable_count` (`[INTERNAL]`) and leaves
`tracked_deck_cards` unchanged for these.

---

## 8) hand_and_level OCR Parser

The OCR field appears in two equivalent forms across schemas:
- raw extracted: `"hand_and_level": "highcardlvl.36"`
- parsed pass-through: `"hand_and_level_raw": "highcardlvl.36"`

Observed values include:
- `"highcardlvl.36"`, `"highcardlvl.96"`, `"highcardlvl.97"`
- `"pairlvl.1"`
- `"fourofakindlvl.6"`
- `"fiveofakindlvl.1"`
- `"flushfivelvl.1"`
- `"????lvl.?"` (unparseable sentinel)
- `null` (missing)

```python
import re

# Canonical hand names use the keys defined in state_schema.md section 10.
HAND_NAME_BY_TOKEN = {
    "highcard":      "High Card",
    "pair":          "Pair",
    "twopair":       "Two Pair",
    "threeofakind":  "Three of a Kind",
    "straight":      "Straight",
    "flush":         "Flush",
    "fullhouse":     "Full House",
    "fourofakind":   "Four of a Kind",
    "straightflush": "Straight Flush",
    "fiveofakind":   "Five of a Kind",
    "flushhouse":    "Flush House",
    "flushfive":     "Flush Five",
}

HAND_AND_LEVEL_RE = re.compile(r"^([a-z?]+)lvl\.(\d+|\?)$")


def parse_hand_and_level(raw: str | None) -> tuple[str, int] | None:
    """
    Return (canonical_hand_name, level) if parseable, else None.

    None outputs:
      - raw is None or empty
      - regex fails to match (unexpected OCR shape)
      - name token is "????" or unknown
      - level token is "?" or non-numeric
    """
    if not raw:
        return None
    s = raw.strip().lower().replace(" ", "")
    m = HAND_AND_LEVEL_RE.match(s)
    if not m:
        return None

    name_token, level_token = m.group(1), m.group(2)
    if name_token == "????" or level_token == "?":
        return None

    hand_name = HAND_NAME_BY_TOKEN.get(name_token)
    if hand_name is None:
        return None

    try:
        level = int(level_token)
    except ValueError:
        return None
    if level < 1:
        return None
    return hand_name, level
```

Notes:
- Level is taken at face value when parseable, and combined with prior knowledge via `max(...)` in `on_play_hand` so transient OCR drops cannot regress a hand's level.
- Unparseable hands still increment `hands_played` (which is the run-wide play count) but do not pollute per-hand counters.
- The `[INTERNAL] hand_and_level_unparsed_count` is a diagnostic for downstream OCR quality reports; it is never part of the model input.

---

## 9) Helpers

```python
def first_or_none(seq):
    if not seq:
        return None
    return seq[0]


def parse_base_action(action: str) -> str:
    """
    'PlayHand'        -> 'PlayHand'
    'BuyShopItem_2'   -> 'BuyShopItem'
    'SWAP_0_1'        -> 'SWAP'
    'SelectCard_3'    -> 'SelectCard'
    """
    return action.split("_", 1)[0]


def extract_target_class_id(selected_object) -> int | None:
    """
    Granularized step's `selected_object` shapes (per granularization_schema.md):
      None
      {"role": ..., "slot_id": ..., "object": {... "class_id": int ...}}
      {"role": "joker_slot_pair", "pair": [i, j], "object": {...}}  # no class
    """
    if not isinstance(selected_object, dict):
        return None
    obj = selected_object.get("object")
    if not isinstance(obj, dict):
        return None
    cid = obj.get("class_id")
    return int(cid) if isinstance(cid, int) else None


def extract_target_object(selected_object) -> dict | None:
    """Pull the full `selected_object.object` payload (with class_id, card,
    modifier, edition, seal, stickers, ...) when present. Used by
    on_select_pack_item to canonicalize a pack-spawned card before appending
    it to tracked_deck_cards."""
    if not isinstance(selected_object, dict):
        return None
    obj = selected_object.get("object")
    return obj if isinstance(obj, dict) else None
```

---

## 10) Tensorization Contract

When tensorization serializes the state for the model at step `t`:

```python
def to_model_input(state: dict, step: dict) -> dict:
    persistent_visible = {k: state[k] for k in MODEL_VISIBLE_KEYS}
    observations = read_observation_fields(step)  # OCR + zones, [OBSERVATION]
    return {**persistent_visible, **observations}

def to_mask_input(state: dict, step: dict) -> dict:
    # Masking is allowed to read everything: persistent visible, observations,
    # and INTERNAL bookkeeping (swap_count, last_swap, etc.).
    return {**state, **read_observation_fields(step)}
```

This is the only place `[INTERNAL]` fields may flow downstream — into mask generation, never into the model input vector.
