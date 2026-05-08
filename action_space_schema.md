# Balatro Action Space Schema

## 1) Purpose

This file is the canonical definition of the flattened action space used by masking, tensorization, and training targets.

Source-aligned inputs:
- `data/masking_schema.py` (authoritative action/mask intent)
- `granularization_schema.md` (micro-step emission semantics)
- `mask_schema.md` (legality rules)

---

## 2) Canonical Action Families

Base macro actions:
- `StartNewRun`
- `SelectBlind`
- `SkipBlind`
- `RerollBossBlind`
- `DiscardHand`
- `PlayHand`
- `UseConsumable(i)`
- `SelectCard(i)`
- `CashOut`
- `SelectPackItem(i)`
- `BuyAndUseShopConsumable(i)`
- `BuyShopItem(i)`
- `LeaveShop`
- `SkipPack`
- `SellItem(i)`
- `RerollShop`
- `SWAP(i, j)`

Subtype families (metadata on actions, not separate top-level families):
- `BuyShopItem`: `buyvoucher`, `buyandopenplanetstandardbuffoonpack`, `buytopshelfjoker`, `buyandopentarotspectralpack`, `buytopshelfconsumable`
- `SelectPackItem`: `selectpackitemtarot`, `selectpackitemcard`, `selectpackitemjoker`, `selectpackitemplanet`
- `SellItem`: `selljoker`, `sellconsumable`
- `SkipPack`: `skipplanetstandardbuffoonpack`, `skiptarotspectralpack`, `skipplanetstandardbuffoonpackblind`, `skiptarotspectralpackblind`

---

## 3) Flat Index Construction

## 3.1 Deterministic Ordering

Global action-map ordering must be fixed and versioned:

1. Fixed non-index actions (in this exact order):
   - `StartNewRun`
   - `SelectBlind`
   - `SkipBlind`
   - `RerollBossBlind`
   - `DiscardHand`
   - `PlayHand`
   - `CashOut`
   - `LeaveShop`
   - `SkipPack`
   - `RerollShop`

2. Indexed single-slot families:
   - `UseConsumable_0 ... UseConsumable_(MAX_CONSUMABLE_TARGETS-1)`
   - `SelectCard_0 ... SelectCard_(MAX_SELECT_CARD_TARGETS-1)`
   - `SelectPackItem_0 ... SelectPackItem_(MAX_PACK_ITEM_TARGETS-1)`
   - `BuyAndUseShopConsumable_0 ... BuyAndUseShopConsumable_(MAX_SHOP_CONSUMABLE_TARGETS-1)`
   - `BuyShopItem_0 ... BuyShopItem_(MAX_BUYSHOPITEM_TARGETS-1)`
   - `SellItem_0 ... SellItem_(MAX_SELLITEM_TARGETS-1)`

3. Pair-index swap family:
   - `SWAP_i_j` for all `0 <= i < j < MAX_JOKER_SLOTS`, ordered lexicographically by `(i, j)`.

This yields one fixed global map used across all runs.

## 3.2 Action Count Formula

Let:
- `F = 10` fixed non-index actions
- `U = MAX_CONSUMABLE_TARGETS`
- `C = MAX_SELECT_CARD_TARGETS`
- `P = MAX_PACK_ITEM_TARGETS`
- `B = MAX_SHOP_CONSUMABLE_TARGETS`
- `S = MAX_BUYSHOPITEM_TARGETS`
- `L = MAX_SELLITEM_TARGETS`
- `J = MAX_JOKER_SLOTS`

Then total action-space size:

`N_ACTIONS = F + U + C + P + B + S + L + J*(J-1)/2`

---

## 4) Index Semantics by Family

- `UseConsumable_i`: i-th consumable candidate in canonical consumable target ordering.
- `SelectCard_i`: i-th selectable card candidate in current select-card pool ordering.
- `SelectPackItem_i`: i-th pack candidate.
- `BuyAndUseShopConsumable_i`: i-th buy-and-use consumable candidate.
- `BuyShopItem_i`: i-th shop buy candidate.
- `SellItem_i`: i-th inventory sell candidate.
- `SWAP_i_j`: swap joker slots `i` and `j` (unordered pair encoded with `i < j`).

All non-index fixed actions map to a single index each.

---

## 5) Candidate Ordering Rules (Per Step)

Per-step candidate ordering must be deterministic and tied to zone ordering (`position_in_zone`, then `slot_id`):

- consumables: selected zone first, then visible candidate zone fallback
- cards: selected-card zone first for granularized intermediary generation; otherwise visible pool
- pack/shop/inventory candidates: canonical zone list order then object order within zone

Recommended canonical zone priority:

- `BuyShopItem_i`: `VoucherShopOfferings` -> `PackShopOfferings` -> `TopShelfShopOfferings`
- `SelectPackItem_i`: `PackOfferings` (or legacy overloaded equivalent)
- `SellItem_i`: `CurrentJokers` then `CurrentConsumables` unless explicit selected zones are available
- `BuyAndUseShopConsumable_i`: shop offerings filtered to consumables only

---

## 6) Mask Alignment Contract

For each step:
- emit `action_mask` length exactly `N_ACTIONS`
- set index valid only if action is legal under `mask_schema.md`
- executed action must map to `target_action_id` with:
  - `0 <= target_action_id < N_ACTIONS`
  - `action_mask[target_action_id] == 1`

---

## 7) SWAP Family Rules

Swap family inclusion:
- include all possible `SWAP_i_j` entries up to `MAX_JOKER_SLOTS` in global map.

Per-step legality:
- constrained by current `jokers_current`
- `i < j < jokers_current`
- additional mask gates from `mask_schema.md`:
  - `swap_count` cap (`>= (max(0, jokers_current - 1)) * 2.5`)
  - repeat-swap suppression via `last_swap`
  - minimum two jokers

---

## 8) Versioning and Artifacts

Persist with tensorized data:
- `action_map.json` (`index -> action_label`)
- `action_space_config.json` (all MAX_* constants + ordering policy)
- `action_map_version` (semantic version string)

Any change to:
- family ordering,
- MAX constants,
- label format,
- candidate ordering policy,

must bump `action_map_version`.

---

## 9) Required Decisions Before Locking v1

These values must be finalized in config before tensorization starts:
- `MAX_CONSUMABLE_TARGETS`
- `MAX_SELECT_CARD_TARGETS`
- `MAX_PACK_ITEM_TARGETS`
- `MAX_SHOP_CONSUMABLE_TARGETS`
- `MAX_BUYSHOPITEM_TARGETS`
- `MAX_SELLITEM_TARGETS`
- `MAX_JOKER_SLOTS`

Recommended approach:
- compute maxima from full granularized corpus,
- then add small headroom buffer (for robustness),
- lock in versioned config.

