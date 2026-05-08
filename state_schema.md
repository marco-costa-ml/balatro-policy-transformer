# Balatro State Schema

## 1) Scope and Source of Truth

This document captures state and metadata requirements defined in `data/masking_schema.py`.

It specifies:
- persistent values fed to the model,
- internal values not fed to the model,
- update rules driven by actions,
- object metadata defaults and economy derivation.

---

## 2) Persistent Information Fed to the Model

Core persistent fields:
- `last_tarot_planet: int | null` (set from consumable targets in `236-247` or `298-319`)
- `stake: int` (from first `StartNewRun` `CurrentStake[0]`; default `268`)
- `ecto_minus: int` (increment on class `253` use)
- `skips: int` (increment on `SkipBlind`)
- `hands_played: int` (increment on `PlayHand`)
- `unused_discards: int` (add `discards_left` on `CashOut`)
- `vouchers_redeemed: int[]` (append target voucher class `320-351` on voucher buys)
- `ante_boss_blind: int | null`
- `small_status: int` (0 selected / 1 available / 2 skipped)
- `big_status: int` (0 selected / 1 available / 2 skipped)
- `bosses_used: int[]`
- `is_boss_blind_rerolled: bool`
- `hands`: poker-hand tracker with fields `{level, played, played_this_round}` per hand type
- `cards_in_deck`: object map seeded from class IDs `0-51`

## 2.1 Blind-State Update Rules

- `ante_boss_blind`: last class in `BlindOfferings` between `370-399`, excluding `371`, `394`
- `small_status`:
  - set `0` when `SelectBlind` target is `394`
  - set `2` when `SkipBlind` target is `394`
  - set `1` when a boss blind (`370-399` excluding `371`, `394`) is selected
- `big_status`:
  - set `0` when `SelectBlind` target is `371`
  - set `2` when `SkipBlind` target is `371`
  - set `1` when a boss blind (`370-399` excluding `371`, `394`) is selected
- `bosses_used`: append class ID for each selected boss blind (`370-399`, excluding `371`, `394`)
- `is_boss_blind_rerolled`:
  - set `true` on `RerollBossBlind`
  - reset `false` when selecting a boss blind (`370-399`, excluding `371`, `394`)

---

## 3) Information Not Fed to the Model

- `swap_count`
  - increment after every `SWAP` action
  - reset on `SelectBlind`, `PlayHand`, or `DiscardHand`
- `deck_detected` (placeholder bool)
- `last_swap` (e.g., `SWAP_1_5`)

---

## 4) Object Metadata Requirements

Metadata defaults come from `data/metadata_map.csv` for:
- cards `0-51`
- jokers `80-229`
- planets `236-247`
- spectrals `248-265`
- tarots `298-319`

Special handling:
- class `231` must be treated as a playing-card-like object with unknown suit/rank/related card metadata.

---

## 5) Cost and Sell-Value Derivation

Cost derivation:
1. start from parent metadata cost
2. add edition cost if object has edition (`68-71`)
3. apply voucher discounts:
   - class `322`: 25% discount
   - class `330`: 50% discount
4. round half down
5. clamp minimum buy cost to `1`
6. overrides:
   - if joker class `85` is in `CurrentJokers`, set cost to `0` for classes `358-360` and `236-247`
   - if object has sticker `367`, set cost and sell price to `1`

Sell value:
- `floor(total_buy_cost / 2)`, minimum `1`

---

## 6) Global-State Structure (Model-Facing Shape)

`data/masking_schema.py` defines a global structure containing:
- page/phase,
- run counters and slot capacities,
- persistent counters,
- blind/shop state,
- compact set-like masks (e.g. vouchers/bosses used),
- poker hand progression.

This doc treats that structure as required semantic content; exact tensor encoding is defined in `tensorization_schema.md`.

