# Balatro Action Mask Schema

## 1) Scope and Source of Truth

This document is synchronized to `data/masking_schema.py` and defines legality masking over candidate actions.

Inputs:
- granularized step context,
- derived state (`state_schema.md`),
- zone/object snapshots.

Output:
- action mask where valid=1 and masked=0.

---

## 2) Page Gating Rules

- `SelectBlind` / `SkipBlind` / `RerollBossBlind`: only `Blind_Select`
- `DiscardHand` / `PlayHand`: only `In_Blind`
- `UseConsumable`: masked if `CurrentConsumablesSelected.length <= 0`
- `SelectCard(i)`: only `In_Blind` or `In_TarotSpectral_Pack`
- `CashOut`: only `Cash_Out`
- `BuyShopItem(i)`: only `In_Shop`
- `BuyAndUseShopConsumable(i)`: only `In_Shop`
- `LeaveShop`: only `In_Shop`
- `RerollShop`: only `In_Shop`
- `SelectPackItem(i)`: only `In_JokerStandardPlanet_Pack` (source wording)
- `SkipPack`: only `In_TarotSpectral_Pack` and `In_JokerStandardPlanet_Pack`
- `SellItem(i)`: globally available (then item-gated)

---

## 3) Baseline Action Gating

- `SkipBlind`: mask if `OfferedTag.length == 0`
- `RerollBossBlind`:
  - allow iff voucher `346` exists, or voucher `324` exists and `is_boss_blind_rerolled == false`
  - else mask
- `DiscardHand` / `PlayHand`: mask unless `selected_cards.length > 0`
- `BuyShopItem(i)`:
  - mask if `dollars < shop_offerings(i).cost`
  - mask joker buys when `jokers_current >= jokers_total` and edition is not negative
  - mask consumable buys when `consumables_current >= consumables_total`
- `SellItem(i)`: mask if selected item has sticker `class_id 369` (eternal)
- `RerollShop`: mask if `reroll_price > dollars`
- `SelectPackItem(i)`: mask when selected pack item is joker (`80-229`) and `jokers_current >= jokers_total`

---

## 4) UseConsumable Family Constraints

## 4.1 Card-Selection Cardinality

Exactly 1 card selected:
- `263 c_talisman`
- `249 c_aura`
- `252 c_deja_vu`
- `264 c_trance`
- `259 c_medium`
- `251 c_cryptid`
- `298 c_chariot`
- `310 c_lovers`
- `317 c_tower`
- `300 c_devil`

Up to 2 cards selected:
- `302 c_empress`
- `304 c_hanged_man`
- `305 c_heirophant`
- `311 c_magician`
- `314 c_strength`

Up to 3 cards selected:
- `312 c_moon`
- `313 c_star`
- `315 c_sun`
- `319 c_world`

Exactly 2 cards selected:
- `299 c_death`

## 4.2 Joker/Inventory Constraints

Require `jokers_current >= 1`:
- `248 c_ankh`
- `256 c_hex`
- `318 c_wheel_of_fortune`

Require `jokers_current < jokers_total`:
- `265 c_wraith`
- `262 c_soul`
- `308 c_judgement`

Require `last_tarot_planet != 303`:
- `303 c_fool`

Require `consumables_current < consumables_total`:
- `307 c_high_priestess`
- `301 c_emperor`

## 4.3 Shop Use Notes from Source

Tarots without extra requirements:
- `316 c_temperance`
- `306 c_hermit`

Tarots/spectrals listed as shop-usable in source:
- `248 c_ankh`
- `256 c_hex`
- `265 c_wraith`
- `301 c_emperor`
- `303 c_fool`
- `306 c_hermit`
- `307 c_high_priestess`
- `308 c_judgement`
- `316 c_temperance`
- `318 c_wheel_of_fortune`

Source contains one truncated note (`THE H`) near shop-usable tarots; treat as unresolved TODO.

## 4.4 Additional Card/Joker Attribute Constraints

These constraints are required by game logic and should be applied during `UseConsumable` masking.

Seal consumables cannot target a card that already has the same seal:
- `263 c_talisman` (Gold Seal): mask target card if `target.seal == gold_seal`
- `252 c_deja_vu` (Red Seal): mask target card if `target.seal == red_seal`
- `264 c_trance` (Blue Seal): mask target card if `target.seal == blue_seal`
- `259 c_medium` (Purple Seal): mask target card if `target.seal == purple_seal`

Edition consumable cannot target already-editioned cards:
- `249 c_aura`: mask target card if `target.edition != null`

Hex special joker-edition constraint:
- `256 c_hex`: mask action if exactly one joker currently has any edition.
- Equivalently, action is only legal when the current joker count without editions is `>= 1`.

---

## 5) BuyAndUseShopConsumable and SelectPackItem

- `BuyAndUseShopConsumable(i)`:
  - generated for consumables in shop offerings,
  - uses same logic family as `UseConsumable`,
  - source note: selected card index is always 0.

- `SelectPackItem(i)`:
  - generated for each pack offering candidate,
  - source marks older overloaded-zone wording as deprecated,
  - selected-card intermediary context should come from `TarotSpectralHandSelected` for tarot path.

---

## 6) SWAP Mask Rules

`SWAP_i_j` definitions:
- valid only for `0 <= i < j < N`, where `N = jokers_current` slot count basis.
- number of pair actions is `C(N,2) = N(N-1)/2`.

Mask conditions:
- mask if `swap_count >= (max(0, jokers_current - 1)) * 2.5`
- mask if `last_swap == SWAP_i_j`
- mask if `i >= jokers_current`
- mask if `j >= jokers_current`
- mask if `jokers_current < 2`

---

## 7) Invariants and Diagnostics

- mask computation must be deterministic.
- target-required actions should not be unmasked without resolvable targets.
- include diagnostics for all-masked states and unresolved target cases.
- keep zone alias handling compatible with both overloaded and split zone naming schemes.

