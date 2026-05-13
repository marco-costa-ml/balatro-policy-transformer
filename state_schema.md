# Balatro State Schema

## 1) Scope and Source of Truth

This document defines internal persistent state for tensorization and masking.

Primary source:
- `data/masking_schema_disorganized.md`

Related schemas:
- `parse_schema.md` defines extracted -> parsed event fields.
- `granularization_schema.md` defines one-action micro-step emission.
- `mask_schema.md` defines legality gates consuming this state.
- `tensorization_schema.md` defines final tensor encoding.

Persistent state is recomputed by walking each run in order. The state snapshot fed to the model for step `t` must represent all completed actions before step `t`; after processing step `t`, update the reducer state for step `t + 1`.

Reducer pseudocode lives in [state_reducer_pseudocode.md](state_reducer_pseudocode.md).

---

## 1.1) Field Visibility Tags

Every field in this document is tagged with one of:

- `[MODEL-VISIBLE]` — must be encoded into the model input vector by `tensorization_schema.md`.
- `[INTERNAL]` — used only by mask generation, granularization, or diagnostics; must not be fed to the model.
- `[OBSERVATION]` — model-visible, but not part of persistent reducer state; comes from current-step OCR/zones each tick.

A field's visibility is part of its contract. Promoting an `[INTERNAL]` field to `[MODEL-VISIBLE]` requires a schema version bump.

---

## 2) Class ID Ranges and Named Constants

Cards:
- standard playing cards: `0-51` (rank_index = `class_id % 13`, suit_index = `class_id // 13`; rank order `A, 2..9, T, J, Q, K`; suit order `Spades, Hearts, Diamonds, Clubs`)
- stone card placeholder (no rank/suit): `78` (also the class_id for the `m_stone` enhancement itself)
- facedown/unknown playing-card-like object: `231`

Card modifiers / editions / seals (used by `tracked_deck_cards` entries):
- modifiers (`m_*`): `m_bonus`, `m_glass`, `m_gold`, `m_lucky`, `m_mult`, `m_steel`, `m_stone`, `m_wild`
- editions (`e_*`): `e_foil`, `e_holo`, `e_negative`, `e_polychrome`
- seals: `red_seal`, `blue_seal`, `gold_seal`, `purple_seal`

Decks (`CurrentDeck[0]` on `StartNewRun`):
- `52 b_abandoned`
- `57 b_checkered`
- `58 b_erratic`
- `61 b_magic`
- `62 b_nebula`
- `67 b_zodiac`

Vouchers:
- voucher class range: `320-351`
- `323 v_crystal_ball`
- `336 v_overstock_norm`
- `341 v_planet_merchant`
- `348 v_tarot_merchant`
- `350 v_telescope`
- discount vouchers:
  - `322 v_clearance_sale`
  - `330 v_liquidation`
- boss reroll vouchers:
  - `324 v_directors_cut`
  - `346 v_retcon`

Consumables:
- planets: `236-247`
- spectrals: `248-265`
- tarots: `298-319`

Hand-level consumables (planet ↔ poker hand):
- `236 c_ceres`    → Flush House
- `237 c_earth`    → Full House
- `238 c_eris`     → Flush Five
- `239 c_jupiter`  → Flush
- `240 c_mars`     → Four of a Kind
- `241 c_mercury`  → Pair
- `242 c_neptune`  → Straight Flush
- `243 c_planet_x` → Five of a Kind
- `244 c_pluto`    → High Card
- `245 c_saturn`   → Straight
- `246 c_uranus`   → Two Pair
- `247 c_venus`    → Three of a Kind
- `250 c_black_hole` (spectral): increments the level of every poker hand by 1

Card-targeting consumables (mutate `tracked_deck_cards` via the closest-match rule in §5.5):
- Modifier-applying tarots: `298 c_chariot` (m_steel), `300 c_devil` (m_gold), `302 c_empress` (m_mult, ×2), `305 c_heirophant` (m_bonus, ×2), `309 c_justice` (m_glass), `310 c_lovers` (m_wild), `311 c_magician` (m_lucky, ×2), `317 c_tower` (m_stone)
- Suit-changing tarots (up to 3 cards): `312 c_moon` (Clubs), `313 c_star` (Diamonds), `315 c_sun` (Hearts), `319 c_world` (Spades)
- Rank-bumping tarot: `314 c_strength` (+1 rank, up to 2 cards)
- Destroy tarot: `304 c_hanged_man` (up to 2 cards)
- Convert tarot: `299 c_death` (left card becomes a copy of the right card)
- Edition-applying spectral: `249 c_aura` (Foil/Holo/Polychrome; reducer applies `e_foil` deterministically)
- Copy-creating spectral: `251 c_cryptid` (creates 2 exact copies of the selected card)
- Seal-applying spectrals: `252 c_deja_vu` (red_seal), `259 c_medium` (purple_seal), `263 c_talisman` (gold_seal), `264 c_trance` (blue_seal)

Random / hand-wide spectrals (NOT modeled in v1; see §5.5):
- `254 c_familiar`, `255 c_grim`, `257 c_immolate`, `258 c_incantation`, `260 c_ouija`, `261 c_sigil`

Blinds:
- boss blind range: `370-399`, excluding `371` and `394`
- `371` is big blind
- `394` is small blind

Stickers:
- `367 rental`
- `368 perishable`
- `369 eternal`

---

## 3) Persistent Reducer State (Model-Visible)

Every field in this section is `[MODEL-VISIBLE]`. Tensorization may encode arrays as dense masks, counts, or embeddings, but the semantic fields below are required.

```python
persistent_state = {
    # run identity / setup -- [MODEL-VISIBLE]
    "deck": {
        "class_id": int | None,        # [MODEL-VISIBLE]
        "name": str | None,            # [MODEL-VISIBLE]
        "is_magic": bool,              # [MODEL-VISIBLE]
        "is_nebula": bool,             # [MODEL-VISIBLE]
        "is_abandoned": bool,          # [MODEL-VISIBLE]
        "is_checkered": bool,          # [MODEL-VISIBLE]
        "is_zodiac": bool,             # [MODEL-VISIBLE]
        "is_erratic": bool,            # [MODEL-VISIBLE]
    },
    "stake": int,                      # [MODEL-VISIBLE]

    # deck-derived state -- [MODEL-VISIBLE]
    # tracked_deck_cards is a list of card objects (FIFO, capped at TRACKED_DECK_CAP=75).
    # Each entry mirrors the granularizer's card schema minus volatile fields:
    "tracked_deck_cards": list[
        {
            "class_id": int,             # 0..51 standard, 78 stone
            "object_type": "card",
            "card": {                    # None for stone cards (class_id == 78)
                "rank": str,             # "A","2",..."9","T","J","Q","K"
                "rank_index": int,       # 0..12 (A=0, K=12)
                "suit": str,             # "Spades"|"Hearts"|"Diamonds"|"Clubs"
                "suit_index": int,       # 0..3 (Spades=0)
                "is_ace": bool,
                "is_face": bool,
            } | None,
            "modifier": str | None,      # m_bonus|m_glass|m_gold|m_lucky|m_mult|m_steel|m_stone|m_wild|None
            "edition": str | None,       # e_foil|e_holo|e_negative|e_polychrome|None
            "seal": str | None,          # red_seal|blue_seal|gold_seal|purple_seal|None
            "stickers": list[str],
        }
    ],
    "deck_modifiers": {
        "no_face_cards_start": bool,
        "spades_hearts_only_start": bool,
        "randomized_starting_deck": bool,
    },

    # action history counters -- [MODEL-VISIBLE]
    "last_tarot_planet": int | None,
    "ecto_minus": int,
    "skips": int,
    "hands_played": int,
    "unused_discards": int,
    "first_hand": bool,
    "first_discard": bool,

    # inventory / unlock-like persistent sets -- [MODEL-VISIBLE]
    "vouchers_redeemed": list[int],
    "bosses_used": list[int],

    # blind state -- [MODEL-VISIBLE]
    "ante_boss_blind": int | None,
    "small_status": int,
    "big_status": int,
    "is_boss_blind_rerolled": bool,

    # poker hand progression -- [MODEL-VISIBLE]
    "hands": {
        "<hand_name>": {
            "level": int,
            "played": int,
            "played_this_round": int,
        }
    },
}
```

Defaults before the first `StartNewRun`:
- `deck.class_id = None`
- all deck flags false
- `stake = 268`
- `tracked_deck_cards = build_standard_deck()` (one entry for each class_id 0..51, all attributes default to `None`/`[]`)
- `deck_modifiers.* = false`
- `last_tarot_planet = None`
- `ecto_minus = 0`
- `skips = 0`
- `hands_played = 0`
- `unused_discards = 0`
- `first_hand = True`
- `first_discard = True`
- `vouchers_redeemed = []`
- `bosses_used = []`
- `ante_boss_blind = None`
- `small_status = 1`
- `big_status = 1`
- `is_boss_blind_rerolled = False`
- every poker hand starts `{level: 1, played: 0, played_this_round: 0}`

---

## 4) StartNewRun Initialization

On each `StartNewRun`, reset the full persistent state to defaults, then apply run-start initialization from zones.

Source zones:
- `CurrentDeck[0]` gives deck class ID.
- `CurrentStake[0]` gives stake class ID.

If `CurrentStake[0]` is missing, keep default `stake = 268`.

If `CurrentDeck[0]` is missing, keep `deck.class_id = None`, the default standard `tracked_deck_cards`, and no deck flags. This is a data-quality warning, not a hard failure.

## 4.1 Deck-Derived Initialization Rules

Default standard deck (used as the baseline for every deck except Abandoned and Checkered):
- `tracked_deck_cards` contains one entry per class ID `0-51` (52 cards), each with `modifier=None, edition=None, seal=None, stickers=[]`.

Abandoned Deck (`52 b_abandoned`):
- set `deck.is_abandoned = True`
- set `deck_modifiers.no_face_cards_start = True`
- replace `tracked_deck_cards` with the standard deck minus all face cards:
  - rank indices `10, 11, 12` for each suit
  - class IDs: `10, 11, 12, 23, 24, 25, 36, 37, 38, 49, 50, 51`
- result: 40 entries.

Checkered Deck (`57 b_checkered`):
- set `deck.is_checkered = True`
- set `deck_modifiers.spades_hearts_only_start = True`
- replace `tracked_deck_cards` with two copies of every spade and every heart:
  - class IDs `0-12` (spades): two entries each
  - class IDs `13-25` (hearts): two entries each
  - class IDs `26-51` (diamonds/clubs): no entries
- result: 52 entries.

Erratic Deck (`58 b_erratic`):
- set `deck.is_erratic = True`
- set `deck_modifiers.randomized_starting_deck = True`
- keep `tracked_deck_cards` as the default standard deck unless parsed data later provides concrete per-card deck contents.
- Rationale: deck ID alone says ranks/suits are randomized, but does not identify the concrete randomized card multiset. Do not hallucinate a random deck in tensorization.

Magic Deck (`61 b_magic`):
- set `deck.is_magic = True`
- append `323 v_crystal_ball` to `vouchers_redeemed` at run start if absent.

Nebula Deck (`62 b_nebula`):
- set `deck.is_nebula = True`
- append `350 v_telescope` to `vouchers_redeemed` at run start if absent.

Zodiac Deck (`67 b_zodiac`):
- set `deck.is_zodiac = True`
- append these vouchers to `vouchers_redeemed` at run start if absent:
  - `348 v_tarot_merchant`
  - `341 v_planet_merchant`
  - `336 v_overstock_norm`

Other deck class IDs `53, 54, 55, 56, 59, 60, 63, 64, 65, 66`:
- store `deck.class_id` and `deck.name`,
- set corresponding known flags false unless a specific field is added later,
- keep the default standard `tracked_deck_cards`,
- do not infer unlisted deck effects in this schema.

---

## 5) Action Update Rules

All updates below happen after the action is consumed as the current step target.

## 5.1 Consumable Updates

Actions:
- `UseConsumable`
- `BuyAndUseShopConsumable`
- `SelectPackItem` (when the pack item itself is a planet/tarot/spectral; the game auto-uses it)

Target source:
- `UseConsumable`: `CurrentConsumablesSelected[0]`
- `BuyAndUseShopConsumable`: `TopShelfShopOfferingsSelected[0]` or equivalent selected shop-consumable target from granularized `selected_object`
- `SelectPackItem`: `selected_object.object` (the pack item)

The granularized commit step also carries `selected_cards` — the in-hand cards the player targeted before the consumable resolves. This list is empty for non-card-targeting consumables.

Update rules (apply in this order):

1. **Last-used tracker.** If the target class ID is in `236-247` (planets) or `298-319` (tarots), set `last_tarot_planet = target.class_id`. (Black Hole is a spectral and intentionally does NOT update this field.)

2. **Hand-level bumps** (planets and Black Hole — see §2 for the planet ↔ hand mapping):
   - If the target class ID is in `236-247`, increment `hands[<mapped_hand>].level += 1`.
   - If the target class ID is `250` (Black Hole), increment `hands[<every hand>].level += 1`.

3. **Ectoplasm penalty.** If the target class ID is `253`, increment `ecto_minus += 1`.

4. **Card-targeting effects.** If the target class ID is in the card-targeting set (see §5.5), apply the corresponding mutation to `tracked_deck_cards`. If the target class ID is in the random / hand-wide set (see §5.5), increment `unhandled_random_consumable_count` (`[INTERNAL]`) and leave the deck unchanged.

5. **Vouchers.** No direct `vouchers_redeemed` update from consumable actions.

## 5.2 Blind and Round Flow Updates

`SkipBlind`:
- increment `skips += 1`
- if target blind class ID is `394`, set `small_status = 2`
- if target blind class ID is `371`, set `big_status = 2`

`SelectBlind`:
- reset `first_hand = True`
- reset `first_discard = True`
- reset every `hands[hand_name].played_this_round = 0`
- if target blind class ID is `394`, set `small_status = 0`
- if target blind class ID is `371`, set `big_status = 0`
- if target blind class ID is a boss blind (`370-399`, excluding `371`, `394`):
  - append target class ID to `bosses_used` if absent
  - set `small_status = 1`
  - set `big_status = 1`
  - set `is_boss_blind_rerolled = False`

`RerollBossBlind`:
- set `is_boss_blind_rerolled = True`

`CashOut`:
- add current OCR `discards_left` to `unused_discards`.
- If `discards_left` is null/missing, add `0` and emit a data-quality warning.

## 5.3 Hand and Discard Updates

`PlayHand`:
- increment `hands_played += 1`
- set `first_hand = False`
- parse OCR `hand_and_level` (format `<handname>lvl.<level>`, e.g. `highcardlvl.36`):
  - if parseable to `(hand_name, level)`:
    - set `hands[hand_name].level = max(hands[hand_name].level, level)`
    - increment `hands[hand_name].played += 1`
    - increment `hands[hand_name].played_this_round += 1`
  - if unparseable (null, `????lvl.?`, or non-matching token):
    - increment `hands_played` (already done above) but do not touch any specific hand entry,
    - increment `hand_and_level_unparsed_count` ([INTERNAL]).
- Full parsing pseudocode lives in [state_reducer_pseudocode.md](state_reducer_pseudocode.md).

`DiscardHand`:
- set `first_discard = False`

## 5.4 Shop and Voucher Updates

`BuyShopItem`:
- if target class ID is in `320-351`, append it to `vouchers_redeemed` if absent.
- if buying duplicate vouchers appears in data, keep `vouchers_redeemed` set-like for model state; duplicate purchase observations should be treated as data-quality warnings.

Voucher effects consumed by masking/cost logic:
- `322 v_clearance_sale`: 25% discount
- `330 v_liquidation`: 50% discount
- `324 v_directors_cut`: allows one boss reroll while `is_boss_blind_rerolled == False`
- `346 v_retcon`: allows boss reroll independent of `is_boss_blind_rerolled`
- `323 v_crystal_ball`: increases consumable slots through game rules; tensorization should still use OCR slot counts as observed.
- `350 v_telescope`: affects pack behavior; keep as redeemed voucher state.
- `348 v_tarot_merchant`, `341 v_planet_merchant`, `336 v_overstock_norm`: affect shop offerings; keep as redeemed voucher state.

## 5.5 Card-Targeting Consumable Effects on `tracked_deck_cards`

When a consumable in the targeted set resolves, the reducer mutates `tracked_deck_cards` in place. For each entry in `selected_cards`, the reducer finds the **closest match** in the deck and applies the effect.

**Closest-match rule.** Pick the deck entry that:
1. has the same `class_id`, AND
2. matches the most fields among `{modifier, edition, seal}`.

Tie-broken by first occurrence. If no entry shares the target's `class_id` (e.g. a tower-converted card whose class_id changed), fall back to matching by `(rank_index, suit_index)` from the nested `card` sub-dict. If still nothing matches, the effect is a no-op for that selected card.

**Per-consumable effects:**

| Class ID | Name | Effect on matched deck entry |
| --- | --- | --- |
| `298` | c_chariot      | `modifier = "m_steel"` (1 card) |
| `300` | c_devil        | `modifier = "m_gold"` (1 card) |
| `302` | c_empress      | `modifier = "m_mult"` (up to 2 cards) |
| `305` | c_heirophant   | `modifier = "m_bonus"` (up to 2 cards) |
| `309` | c_justice      | `modifier = "m_glass"` (1 card) |
| `310` | c_lovers       | `modifier = "m_wild"` (1 card) |
| `311` | c_magician     | `modifier = "m_lucky"` (up to 2 cards) |
| `317` | c_tower        | `modifier = "m_stone"` (1 card) |
| `312` | c_moon         | re-encode suit to `Clubs` (preserve rank; up to 3 cards) |
| `313` | c_star         | re-encode suit to `Diamonds` (up to 3 cards) |
| `315` | c_sun          | re-encode suit to `Hearts` (up to 3 cards) |
| `319` | c_world        | re-encode suit to `Spades` (up to 3 cards) |
| `314` | c_strength     | `rank_index = (rank_index + 1) mod 13`, recompute `class_id`, `is_face`, `is_ace` (up to 2 cards) |
| `304` | c_hanged_man   | delete from `tracked_deck_cards` (up to 2 cards) |
| `299` | c_death        | left card (selected_cards[0]) becomes a copy of right card (selected_cards[1]); copy class_id, card sub-dict, modifier, edition, seal, stickers |
| `249` | c_aura         | set `edition = "e_foil"` if currently None (placeholder for the random Foil/Holo/Polychrome roll) |
| `251` | c_cryptid      | append two `deepcopy(matched_card)` entries; trim to `TRACKED_DECK_CAP` |
| `252` | c_deja_vu      | `seal = "red_seal"` (1 card) |
| `259` | c_medium       | `seal = "purple_seal"` (1 card) |
| `263` | c_talisman     | `seal = "gold_seal"` (1 card) |
| `264` | c_trance       | `seal = "blue_seal"` (1 card) |

**Random / hand-wide spectrals (NOT modeled in v1):**
- `254 c_familiar` — destroy 1 random card, add 3 random Enhanced face cards
- `255 c_grim` — destroy 1 random card, add 2 random Enhanced Aces
- `257 c_immolate` — destroy 5 random cards in hand
- `258 c_incantation` — destroy 1 random card, add 4 random Enhanced numbered cards
- `260 c_ouija` — convert all hand cards to one random rank
- `261 c_sigil` — convert all hand cards to one random suit

These outcomes cannot be reconstructed deterministically from the granularized stream. The reducer leaves `tracked_deck_cards` unchanged and increments `unhandled_random_consumable_count` (`[INTERNAL]`) for diagnostics.

**Cap.** After every card-targeting effect (and after every card append from §5.6), the reducer truncates `tracked_deck_cards` to at most `TRACKED_DECK_CAP = 75` entries (FIFO: keep the first 75).

## 5.6 SelectPackItem Updates

`SelectPackItem` is dispatched by the class of its target object:

1. **Playing card target** (`class_id` in `0..51`, `object_type == "card"`): append a normalized copy of the target to `tracked_deck_cards`, preserving its `modifier`, `edition`, `seal`, and `stickers`. Trim to `TRACKED_DECK_CAP`.
2. **Stone card target** (`class_id == 78`, `object_type == "modifier"`): append a canonicalized stone-card entry — `class_id=78`, `object_type="card"`, `card=None`, `modifier="m_stone"`. Trim to `TRACKED_DECK_CAP`.
3. **Planet / tarot / spectral target**: route to §5.1 (treated as auto-used; the granularized commit step's `selected_cards` carries the in-hand targets when the pack item is a card-modifying tarot, e.g. `selectpackitemtarot` → c_chariot).
4. **Joker target**: no persistent state changes (jokers occupy joker slots, not the deck).

---

## 6) Per-Step Observations (Model-Visible, Not Persistent)

Every field in this section is `[OBSERVATION]`. These are read from the current step each tick and fed to the model alongside persistent state. They are not part of the reducer.

OCR-derived `[OBSERVATION]`:
- `ante`
- `round`
- `dollars`
- `hands_left`
- `discards_left`
- `hand_size_current`
- `hand_size_total`
- `jokers_current`
- `jokers_total`
- `consumables_current`
- `consumables_total`
- `reroll_price`
- `cash_out`

Zone-derived `[OBSERVATION]`:
- current page/phase (`page_name`)
- current joker inventory from `CurrentJokersAll` or canonicalized joker zone
- current consumables from `CurrentConsumables`
- shop offerings from shop zones
- current hand / pack offerings from their active zones
- selected zones for target resolution

Reducer state and current-step observations are both available to mask generation. Mask generation may also read `[INTERNAL]` fields (section 7); the model never does.

---

## 7) Internal Bookkeeping State (Not Model-Visible)

Every field in this section is `[INTERNAL]`. These are required for granularization, masking, or diagnostics, but must not be tensorized as model inputs.

`swap_count: int` — `[INTERNAL]`
- default `0`
- increment after every `SWAP` action
- reset to `0` after `SelectBlind`, `PlayHand`, or `DiscardHand`
- consumed by `SWAP` mask cap (`mask_schema.md`)

`last_swap: str | None` — `[INTERNAL]`
- default `None`
- set to canonical label such as `SWAP_1_5` after each `SWAP`
- consumed by repeat-swap mask suppression

`deck_detected: bool` — `[INTERNAL]`
- default `False`
- set `True` if `CurrentDeck[0]` is observed at `StartNewRun`
- diagnostic only; model uses `deck.class_id` and deck flags instead.

`prev_jokers_all: list[int] | None` — `[INTERNAL]`
- default `None`
- snapshot of `CurrentJokersAll` slot ordering used for synthetic `SWAP` generation per `granularization_schema.md`.
- consumed only by granularization, never by tensorization.

`hand_and_level_unparsed_count: int` — `[INTERNAL]`
- default `0`
- increment when `PlayHand` occurs and `hand_and_level` cannot be parsed
- diagnostic only.

`unhandled_random_consumable_count: int` — `[INTERNAL]`
- default `0`
- increment whenever the reducer sees a UseConsumable / BuyAndUseShopConsumable / SelectPackItem (auto-use) targeting a random / hand-wide spectral that v1 does not model (`254`, `255`, `257`, `258`, `260`, `261` — see §5.5)
- diagnostic only; signals that `tracked_deck_cards` is imprecise for that run.

---

## 8) Object Metadata Requirements

Metadata defaults come from `data/metadata_map.csv` for:
- cards `0-51`
- jokers `80-229`
- planets `236-247`
- spectrals `248-265`
- tarots `298-319`

Special handling:
- class `231` must be treated as a playing-card-like object with unknown suit/rank/related card metadata.
- `tracked_deck_cards` is a FIFO list of card objects (NOT a multiset of class-id counts). Duplicates are supported by simply having multiple entries with the same `class_id` (required for Checkered Deck, Cryptid copies, and pack-spawned cards). The list is capped at `TRACKED_DECK_CAP = 75` entries to bound the eventual tensorization input.

---

## 9) Cost and Sell-Value Derivation

Cost derivation:
1. start from parent metadata cost,
2. add edition cost if object has edition (`68-71`),
3. apply voucher discounts:
   - class `322`: 25% discount,
   - class `330`: 50% discount,
4. round half down,
5. clamp minimum buy cost to `1`,
6. apply overrides:
   - if joker class `85` is in current jokers, set cost to `0` for classes `358-360` and `236-247`,
   - if object has sticker `367`, set cost and sell price to `1`.

Sell value:
- `floor(total_buy_cost / 2)`, minimum `1`.

Eternal sell mask:
- if an item has sticker `369`, `SellItem(i)` is masked.

---

## 10) Poker Hand Names

Initialize exactly these hands:
- `Flush Five`
- `Flush House`
- `Five of a Kind`
- `Straight Flush`
- `Four of a Kind`
- `Full House`
- `Flush`
- `Straight`
- `Three of a Kind`
- `Two Pair`
- `Pair`
- `High Card`

Each hand has:
- `level: int = 1`
- `played: int = 0`
- `played_this_round: int = 0`

---

## 11) Implementation References and Open Items

Implementation is specified as pseudocode in [state_reducer_pseudocode.md](state_reducer_pseudocode.md). That file is the source of truth for:
- StartNewRun reset and deck-derived initialization,
- per-action update functions,
- `hand_and_level` OCR parser with the canonical token map,
- visibility separation between persistent (model-visible), per-step observation (model-visible), and internal bookkeeping (not model-visible).

Open items:
- Erratic Deck cannot be fully reconstructed from `CurrentDeck[0]` alone. Treat `b_erratic` as a deck flag and randomized-start modifier unless concrete card composition is observed elsewhere.
- `hand_and_level` token map must be kept in sync with any future poker hand additions; unknown tokens are bucketed into `[INTERNAL] hand_and_level_unparsed_count` rather than guessed.
- Voucher-driven slot caps (`v_crystal_ball`, etc.) are reflected through OCR slot counts; no separate persistent override is currently maintained.
- Random / hand-wide spectrals (familiar/grim/immolate/incantation/ouija/sigil) are acknowledged but not modeled in v1 — they leave `tracked_deck_cards` unchanged and bump `unhandled_random_consumable_count`. A future iteration can reconstruct their effects by diffing the BEFORE/AFTER `CurrentHand` zone snapshots.
- Aura's edition roll (Foil/Holo/Polychrome) is collapsed to `e_foil` deterministically. A future iteration can read the AFTER zone snapshot to recover the actual edition.

