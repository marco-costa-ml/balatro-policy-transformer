# Balatro Granularization Schema

## 1) Scope and Source of Truth

This document describes the behavior of `granularize.py` (schema version
`3.0.0`). The authoritative requirement source is
`data/masking_schema_disorganized.md` lines 264-374. Masking policy
details live in `mask_schema.md`. Action-label / index conventions are
documented in `action_space_schema.md`.

---

## 2) Output Step Schema

Each emitted step (row in `data/granularized/video_id=*/run_*.json`):

```json
{
  "step_id": 0,
  "frame_idx": 0,
  "page_name": "Blind_Select",
  "source_event_index": 0,
  "micro_index": 0,
  "source_kind": "pass_through | select | commit | swap_synth",
  "source_action": "PlayHand",
  "source_action_subtype": "buyvoucher | null",
  "action": "PlayHand | BuyShopItem_VoucherShopOfferings_0 | SWAP_0_1 | ...",
  "action_subtype": "buyvoucher | null",
  "target_zone": "CurrentHand | VoucherShopOfferings | ... | null",
  "target_position": 0,
  "swap_pair": [0, 1],
  "selected_object": {"object": "<copy of target object>"},
  "pending_cards": ["<card>", "..."],
  "state": {"hands_left": 4, "...": "..."},
  "objects": ["<obj>", "..."]
}
```

- `selected_object` is non-null only when `target_zone` / `target_position`
  are set; its `object` field equals `objects[target_zone][target_position]`.
- `swap_pair` is non-null only for `source_kind == "swap_synth"`.
- `pending_cards` is empty for most steps. It accumulates during a
  `SelectCard` micro-sequence and contains the final selected-card set
  on the parent `commit` step.

Removed vs schema 2.x: `target_index`, `zones` (canonical-alias dict),
`current_hand_or_pack`, `current_hand`, `selected_cards`. All replaced
by `target_zone` / `target_position` / `pending_cards` / inlined
`objects`.

---

## 3) Zone Normalization

Applied when reading parsed `event.objects[i].zone`:

- `FooAll` -> `Foo` (e.g. `CurrentHandAll` -> `CurrentHand`,
  `PackOfferingsAll` -> `PackOfferings`, `CurrentJokersAll` -> `CurrentJokers`).
- `FooSelected` -> dropped from output entirely (read only to identify
  targets and selected cards).
- A bare zone `Foo` that has a paired `FooAll` or `FooSelected`
  elsewhere is collapsed into the same `Foo` group (deduped by
  `slot_id`).
- All other zones pass through unchanged (`BlindOffering`, `CurrentDeck`,
  `CurrentTags`, `OfferedTag`, `BigBlindTag`, `BlindToken`, `BlindOfferingsNext`,
  `CurrentStake`, `PackConsumableUse`, `VoucherConsumableRedeemUse`).

Plus a synthetic zone `PendingCards` (script-local) that holds playing
cards selected so far within the current parent sequence and is emitted
into `objects` for every step.

---

## 4) Target Resolution

For each event, look up the target zone base (and subtype if applicable):

| Base / Subtype                                | Selected source (target = `Selected[0]`) | All zone used for `target_position` |
|-----------------------------------------------|-------------------------------------------|--------------------------------------|
| `BuyShopItem/buyvoucher`                      | `VoucherShopOfferingsSelected`            | `VoucherShopOfferings`               |
| `BuyShopItem/buyandopenplanetstandardbuffoonpack` | `PackShopOfferingsSelected`           | `PackShopOfferings`                  |
| `BuyShopItem/buyandopentarotspectralpack`     | `PackShopOfferingsSelected`               | `PackShopOfferings`                  |
| `BuyShopItem/buytopshelfjoker`                | `TopShelfShopOfferingsSelected`           | `TopShelfShopOfferings`              |
| `BuyShopItem/buytopshelfconsumable`           | `TopShelfShopOfferingsSelected`           | `TopShelfShopOfferings`              |
| `SelectPackItem/*`                            | `PackOfferingsSelected`                   | `PackOfferings`                      |
| `SellItem/selljoker`                          | `CurrentJokersSelected`                   | `CurrentJokers`                      |
| `SellItem/sellconsumable`                     | `CurrentConsumablesSelected`              | `CurrentConsumables`                 |
| `BuyAndUseShopConsumable`                     | `TopShelfShopOfferingsSelected`           | `TopShelfShopOfferings`              |
| `UseConsumable`                               | `CurrentConsumablesSelected`              | `CurrentConsumables`                 |

`target_position` = position of the target object's `slot_id` in the
matching All zone. If the All-zone lookup fails (data inconsistency),
`target_zone` is preserved but `target_position` is null and the action
label degrades to bare (handled downstream as an unresolved label).

`SelectBlind`, `SkipBlind`, `RerollBossBlind`, `RerollShop`,
`DiscardHand`, `PlayHand`, `CashOut`, `LeaveShop`, `SkipPack`, and
`StartNewRun` have no target object.

---

## 5) SelectCard Decomposition

Triggered for parent events with at least one selected playing card:

| Parent event                                  | Selected source        | Pool / target zone for SelectCard |
|-----------------------------------------------|------------------------|------------------------------------|
| `PlayHand`                                    | `CurrentHandSelected`  | `CurrentHand`                      |
| `DiscardHand`                                 | `CurrentHandSelected`  | `CurrentHand`                      |
| `UseConsumable` on `In_Blind`                 | `CurrentHandSelected`  | `CurrentHand`                      |
| `UseConsumable` on `In_TarotSpectral_Pack`    | `TarotSpectralHandSelected` | `TarotSpectralHand`           |
| `SelectPackItem/selectpackitemtarot`          | `TarotSpectralHandSelected` | `TarotSpectralHand`           |

`UseConsumable` decomposes only when its target consumable's `class_id`
is in `REQUIRES_AT_LEAST_ONE_CARD` (see `granularize.py`):

```
{249, 251, 252, 259, 263, 264, 298, 299, 300, 302, 304, 305,
 309, 310, 311, 312, 313, 314, 315, 317, 319}
```

Iteration order: selected cards sorted by `position_in_zone` within
their Selected zone.

Per micro-step:

1. Find the target card's current position in the **dynamic** pool.
2. Emit `SelectCard_<pool>_<pos>` with `target_zone=<pool>`,
   `target_position=<pos>`, `source_kind="select"`.
3. Snapshot `pending_cards` BEFORE adding this card; embed in `objects`
   as `zone="PendingCards"`.
4. Pop the target from the dynamic pool, append its key to
   `pending_keys`.

After all selects, emit the parent step with `source_kind="commit"`:
the dynamic pool has all selected cards removed; `pending_cards`
contains all selected cards. `pending_cards` is cleared between parent
events (next event starts with an empty `PendingCards`).

---

## 6) SWAP Synthesis

`SWAP` is the only synthesized base action. Generated before any event
whose base is in:

```
{DiscardHand, PlayHand, UseConsumable, CashOut, SelectPackItem,
 BuyAndUseShopConsumable, BuyShopItem, LeaveShop, SellItem}
```

Algorithm:

1. Compare `last_jokers` (previous parent's `CurrentJokers` snapshot)
   with the current event's `CurrentJokers`.
2. Reconcile the set difference: only operate on jokers present in both.
3. Greedy in-place transform of `current` into `target`:
   ```python
   for i in range(len(current)):
       if current[i] != target[i]:
           j = current.index(target[i], i + 1)
           emit SWAP(i, j); current[i], current[j] = current[j], current[i]
   ```
4. For each emitted swap:
   - Step copies the parent event's OCR and all populated zones, but
     `CurrentJokers` is replaced by the snapshot taken IMMEDIATELY
     BEFORE this swap is applied (so the model sees the state it would
     need to act on to produce that swap).
   - `action = "SWAP_i_j"`, `swap_pair = [i, j]`,
     `source_kind = "swap_synth"`, no target.

After all swaps for this event have been emitted, `last_jokers` is
updated to the in-progress reordered list. Once the parent event itself
is recorded, `last_jokers` is reset to the parent's `CurrentJokers`.

Snapshot exclusions (events that do NOT update `last_jokers` and do NOT
trigger swap comparison):

```
{StartNewRun, SkipBlind, SkipPack, RerollShop, RerollBossBlind, SelectCard}
```

---

## 7) Emission Ordering and `step_id`

Within a parsed event:

```
SWAP_synth_steps (micro_index = -K .. -1)
SelectCard_steps  (micro_index =  0 .. K)
parent_step       (micro_index =  K) - "commit" or "pass_through"
```

Across the run, all emitted steps receive a monotonic `step_id`
starting at 0. Negative `micro_index` values for `swap_synth` are
preserved purely as metadata and do not affect ordering since the SWAP
steps are emitted into the output list before their parent.

---

## 8) Action Label Format

| Family group                | Label format                              | Example                                   |
|-----------------------------|-------------------------------------------|-------------------------------------------|
| Bare (no target)            | `Base`                                    | `PlayHand`, `LeaveShop`, `StartNewRun`    |
| Per-zone indexed            | `Base_Zone_i`                             | `BuyShopItem_VoucherShopOfferings_0`      |
| Pair-indexed                | `SWAP_i_j` (i < j over `CurrentJokers`)   | `SWAP_0_1`                                |

The per-zone subfamilies are exactly those listed in
`action_space_schema.md` and `data/action_map.json`. The integer index
`i` is identical to `target_position` (or to `swap_pair` indices), so
downstream consumers can use the structured fields without re-parsing
the action string.
