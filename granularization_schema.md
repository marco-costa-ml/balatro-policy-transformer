# Balatro Granularization Schema

## 1) Scope and Source of Truth

This document defines granularization behavior using `data/masking_schema.py` as the authoritative requirement source.

Granularization converts parsed events into micro-steps with:
- one action per emitted step,
- explicit target-object semantics,
- intermediary `SelectCard(i)` generation where required,
- synthetic `SWAP(i, j)` generation from joker order changes.

Masking policy details live in `mask_schema.md`.
Persistent/derived state details live in `state_schema.md`.
Tensor-space flattening details live in `tensorization_schema.md`.

---

## 2) Canonical Action Set

Base actions:
1. `StartNewRun`
2. `SelectBlind`
3. `SkipBlind`
4. `RerollBossBlind`
5. `DiscardHand`
6. `PlayHand`
7. `UseConsumable(i)`
8. `SelectCard(i)`
9. `CashOut`
10. `SelectPackItem(i)`
11. `BuyAndUseShopConsumable(i)`
12. `BuyShopItem(i)`
13. `LeaveShop`
14. `SkipPack`
15. `SellItem(i)`
16. `RerollShop`
17. `SWAP(i, j)`

Action subtype families:
- `BuyShopItem`: `buyvoucher`, `buyandopenplanetstandardbuffoonpack`, `buytopshelfjoker`, `buyandopentarotspectralpack`, `buytopshelfconsumable`
- `SelectPackItem`: `selectpackitemtarot`, `selectpackitemcard`, `selectpackitemjoker`, `selectpackitemplanet`
- `SellItem`: `selljoker`, `sellconsumable`
- `SkipPack`: `skipplanetstandardbuffoonpack`, `skiptarotspectralpack`, `skipplanetstandardbuffoonpackblind`, `skiptarotspectralpackblind`

---

## 3) Intermediary Event Generation

## 3.1 Always Granularized

- `DiscardHand`:
  - emit `SelectCard(i)` for each selected card in order,
  - emit final `DiscardHand`.
- `PlayHand`:
  - emit `SelectCard(i)` for each selected card in order,
  - emit final `PlayHand`.
- `SWAP(i, j)`:
  - synthetic only; never directly present in extracted events.

## 3.2 Conditionally Granularized

- `UseConsumable`
- `SelectPackItem`

If target parent class is in:
`{249, 251, 252, 259, 263, 264, 298, 299, 300, 302, 304, 305, 309, 310, 311, 312, 313, 314, 315, 317, 319}`,
emit intermediary `SelectCard(i)` events before final action.

If target is not in the set, do not emit intermediary select-card events for that action.

---

## 4) Target Resolution Contract

Default target rules:
- `DiscardHand`: intermediary targets are `CurrentHandOrPackOfferingsSelected[i]`
- `PlayHand`: intermediary targets are `CurrentHandOrPackOfferingsSelected[i]`
- `UseConsumable`: `CurrentConsumablesSelected[0]`
- `BuyAndUseShopConsumable`: `TopShelfShopOfferingsSelected[0]` (plus mask constraints)
- `SellItem/selljoker`: `CurrentJokersSelected[0]`
- `SellItem/sellconsumable`: `CurrentConsumablesSelected[0]`
- `SkipPack`: no target
- `SWAP(i, j)`: target is pair `(i, j)` over joker slots

Subtype-specific target rules:
- `BuyShopItem/buyvoucher`: `VoucherShopOfferingsSelected[0]`
- `BuyShopItem/buyandopenplanetstandardbuffoonpack`: `PackShopOfferingsSelected[0]`
- `BuyShopItem/buytopshelfjoker`: `TopShelfShopOfferingsSelected[0]`
- `BuyShopItem/buyandopentarotspectralpack`: `PackShopOfferingsSelected[0]`
- `BuyShopItem/buytopshelfconsumable`: `TopShelfShopOfferingsSelected[0]`
- `SelectPackItem/selectpackitemtarot`:
  - final target: `CurrentHandOrPackOfferingsSelected[0]` (legacy naming in source spec),
  - intermediary select-card targets: `TarotSpectralHandSelected[i]`
- `SelectPackItem/selectpackitemcard|joker|planet`: `CurrentHandOrPackOfferingsSelected[0]`

---

## 5) SWAP Synthesis

`SWAP` generation is based on joker-order deltas in `CurrentJokersAll`.

Snapshot exclusion actions:
- `StartNewRun`
- `SkipBlind`
- `SelectCard`
- `RerollShop`
- `RerollBossBlind`
- `SkipPack`

Algorithm (from source spec):

```python
current = start.copy()
actions = []
for i in range(len(current)):
    if current[i] != end[i]:
        j = current.index(end[i])
        actions.append(SWAP(i, j))
        current[i], current[j] = current[j], current[i]
```

Set reconciliation rule:
- if jokers were added/removed, reconcile differences first,
- generate swaps over shared joker identities only.

Additional invariant:
- `CurrentJokersAll` intentionally duplicates `CurrentJokersSelected[0]`;
  normalize duplicate identity entries before computing permutation deltas.

---

## 6) Zone Naming and Migration Note

`data/masking_schema.py` still references overloaded zones:
- `CurrentHandOrPackOfferings`
- `CurrentHandOrPackOfferingsSelected`

It also marks parts of this logic as deprecated/confusing for select-pack behavior and references `TarotSpectralHand*` for selected-card context.

Implementation should support both:
- legacy overloaded zones (for compatibility with existing parsed/granularized data),
- split hand/pack zones in newer extractor outputs.

---

## 7) Output Schema (Granularized Step)

Each output step should include:

```json
{
  "video_id": "string",
  "run_index": 0,
  "frame_idx": 0,
  "page_name": "string",
  "action": "string",
  "action_subtype": "string|null",
  "source_action": "string",
  "source_action_subtype": "string|null",
  "source_event_index": 0,
  "micro_index": 0,
  "source_kind": "pass_through | select | commit | swap_synth",
  "selected_object": "object|null",
  "state": "parsed OCR state snapshot",
  "objects": "[object] parsed object snapshot",
  "zones": "normalized zone projections",
  "step_id": 0
}
```

---

## 8) Open Items from Source Spec

- `SWAP` is specified but marked as needing implementation integration.
- `SelectPackItem` still contains deprecated wording around overloaded zones.
- Shop-usable tarot list has a truncated line in source (`THE H`); treat as unresolved TODO until clarified.

