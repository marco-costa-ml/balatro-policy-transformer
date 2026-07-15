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

2. Indexed single-slot families: -- depricated
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

## 4) Index Semantics by Family -- depricated

- `UseConsumable_i`: i-th consumable candidate in canonical consumable target ordering.
- `SelectCard_i`: i-th selectable card candidate in current select-card pool ordering.
- `SelectPackItem_i`: i-th pack candidate.
- `BuyAndUseShopConsumable_i`: i-th buy-and-use consumable candidate.
- `BuyShopItem_i`: i-th shop buy candidate.
- `SellItem_i`: i-th inventory sell candidate.
- `SWAP_i_j`: swap joker slots `i` and `j` (unordered pair encoded with `i < j`).

All non-index fixed actions map to a single index each.

---

## 5) Candidate Ordering Rules (Per Step) -- depricated

Per-step candidate ordering must be deterministic and tied to zone ordering (`position_in_zone`, then `slot_id`).

Universal rule: selected zones first (in canonical order), then visible zones (in canonical order). Object order within a zone follows `(position_in_zone, slot_id)`.

Per-family canonical concatenation (mirrors `granularize.py:canonical_candidates_for` and the formulas in §9): -- depricated

- `UseConsumable_i`: `CurrentConsumablesSelected` -> `CurrentConsumables`
- `SelectCard_i` (page = `In_Blind` or default): `CurrentHandSelected` -> `CurrentHandOrPackOfferingsSelected` (legacy) -> `CurrentHand` -> `CurrentHandOrPackOfferings` (legacy)
- `SelectCard_i` (page = `In_TarotSpectral_Pack`): `TarotSpectralHandSelected` -> `TarotSpectralHand`
- `SelectPackItem_i`: `PackOfferingsSelected` -> `CurrentPackSelected` (legacy) -> `CurrentHandOrPackOfferingsSelected` (legacy) -> `PackOfferings` -> `CurrentPack` (legacy) -> `CurrentHandOrPackOfferings` (legacy)
- `BuyAndUseShopConsumable_i`: (`TopShelfShopOfferingsSelected` -> `ShopOfferingsSelected`, filtered to `object_type == consumable`) -> (`TopShelfShopOfferings` -> `ShopOfferings`, filtered to `object_type == consumable`)
- `BuyShopItem_i`: `VoucherShopOfferingsSelected` -> `PackShopOfferingsSelected` -> `TopShelfShopOfferingsSelected` -> `ShopOfferingsSelected` -> `VoucherShopOfferings` -> `PackShopOfferings` -> `TopShelfShopOfferings` -> `ShopOfferings`
- `SellItem_i`: `CurrentJokersSelected` -> `CurrentConsumablesSelected` -> `CurrentJokers` (or `CurrentJokersAll` as fallback when `CurrentJokers` is absent) -> `CurrentConsumables`

`SelectCard_i` during granularized intermediary micro-steps uses a dynamic candidate list `prev_selected + current_pool`, where:
- `prev_selected` = cards already committed earlier in this granularized sub-sequence,
- `current_pool` = `unselected_base + remaining_to_select` shuffled by the granularizer's RNG and including the target.

The integer `i` written into the action label is the position of the target's `slot_id` in this list.

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

## 9) Required Decisions Before Locking v1 -- depricated

Canonical counting rules (from `data/extracted`, no headroom):
- `MAX_CONSUMABLE_TARGETS`: `len(CurrentConsumablesSelected) + len(CurrentConsumables)`
- `MAX_SELECT_CARD_TARGETS`: `max(len(CurrentHandSelected) + len(CurrentHand), len(TarotSpectralHandSelected) + len(TarotSpectralHand), len(PackOfferingsSelected) + len(PackOfferings))`
- `MAX_PACK_ITEM_TARGETS`: `len(PackOfferingsSelected) + len(PackOfferings)`
- `MAX_SHOP_CONSUMABLE_TARGETS`: consumable-filtered count from `TopShelfShopOfferingsSelected + TopShelfShopOfferings + ShopOfferingsSelected + ShopOfferings`
- `MAX_BUYSHOPITEM_TARGETS`: `len(VoucherShopOfferingsSelected + PackShopOfferingsSelected + TopShelfShopOfferingsSelected + ShopOfferingsSelected) + len(VoucherShopOfferings + PackShopOfferings + TopShelfShopOfferings + ShopOfferings)`
- `MAX_SELLITEM_TARGETS`: `len(CurrentJokersSelected) + len(CurrentConsumablesSelected) + len(CurrentJokers) + len(CurrentConsumables)`, with `CurrentJokersAll` used as compatibility source when `CurrentJokers` is absent in extracted data
- `MAX_JOKER_SLOTS`: `max(len(CurrentJokersAll), len(CurrentJokers), jokers_total)`

Joker exception:
- selected/unselected disjoint-sum rules apply to non-joker target families,
- do not sum `CurrentJokersAll` with `CurrentJokers`; treat `CurrentJokersAll` as the canonical all-joker slot basis.

Finalize these values in config before tensorization starts:
- `MAX_CONSUMABLE_TARGETS = 4` (observed max from extracted corpus)
- `MAX_SELECT_CARD_TARGETS = 14` (observed max from extracted corpus)
- `MAX_PACK_ITEM_TARGETS = 6` (observed max from extracted corpus)
- `MAX_SHOP_CONSUMABLE_TARGETS = 4` (observed max from extracted corpus)
- `MAX_BUYSHOPITEM_TARGETS = 13` (observed max from extracted corpus)
- `MAX_SELLITEM_TARGETS = 12` (observed max from extracted corpus)
- `MAX_JOKER_SLOTS = 9` (observed max from extracted corpus)

Corpus note for reproducibility:
- maxima are computed from `data/extracted/video_id=*/*.json` only,
- finalized config values equal observed maxima (no headroom),
- validation guard: `python compute_action_space_config.py --check-locked data/action_space_config.json --no-write` must pass in CI/data-refresh workflows,
- if future extracted data exceeds any locked value, bump config and `action_map_version`.

