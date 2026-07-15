# Balatro Tensorization Schema

## 1) Scope and Source Alignment

This document defines model-input tensorization, aligned to:
- `parse_schema.md`
- `granularization_schema.md`
- `state_schema.md`
- `mask_schema.md`
- authoritative rules in `data/masking_schema.py`

Output target:
- tensor artifacts (`.pt`) plus reproducibility artifacts (vocabularies, action map, stats).

---

## 2) Tensorization Responsibilities

Tensorization must:
1. convert each granularized step into model features,
2. construct and version the flattened action map,
3. align per-step legality masks to that action map,
4. encode state/object/page/subtype signals into deterministic integer/float channels,
5. emit train/val/test tensor shards and metadata.

---

## 3) Action-Space Contract (Masking-Schema Derived)

Flattened action space includes:
- non-indexed actions:
  - `StartNewRun`, `SelectBlind`, `SkipBlind`, `RerollBossBlind`, `DiscardHand`, `PlayHand`, `CashOut`, `LeaveShop`, `SkipPack`, `RerollShop`
- indexed actions:
  - `UseConsumable_i`
  - `SelectCard_i`
  - `SelectPackItem_i`
  - `BuyAndUseShopConsumable_i`
  - `BuyShopItem_i`
  - `SellItem_i`
- pair-indexed swap actions:
  - `SWAP_i_j` for `0 <= i < j < N` where `N` is configured joker-slot space

Constraints from source spec:
- include `SWAP` family in flat action indexing
- preserve canonical pair uniqueness (`SWAP_i_j == SWAP_j_i`, so only `i < j`)
- expected pair count: `C(N,2)=N(N-1)/2`

Action map artifacts:
- `action_map.json` (index -> label)
- `action_map_version` in dataset metadata
- `data/action_space_config.json` as the source-of-truth MAX constant artifact used to build `action_map.json`

---

## 4) Input Feature Groups

Per-step input channels should include:
- action-context fields (`action`, `action_subtype`, source provenance)
- parsed OCR state
- derived persistent/global state (from `state_schema.md`)
- object-zone projections (`zones`, `objects`)
- selected object payload (if present)

Recommended encoded groups:
- categorical IDs (page, action subtype, object types, zones, enums)
- numeric vectors (economy, counters, slot usage, per-object numeric attrs)
- padded object matrices
- optional sequence context windows

---

## 5) Mask and Target Alignment

For each step:
- build `action_mask` using `mask_schema.md` rules,
- map executed action to `target_action_id` using the same `action_map`,
- enforce invariant: `action_mask[target_action_id] == 1`.

Additional invariants:
- mask length equals action-map length
- deterministic mask generation for identical input state

---

## 6) Shape, Padding, and Limits

Define and freeze configuration:
- `max_objects_per_step`
- `max_selected_cards`
- `max_shop_offerings` (or per-offering-zone caps)
- `max_joker_slots` (drives swap action family size)
- sequence truncation/windowing limits

Important separation:
- input-shape limits above define model input tensor dimensions,
- action-space MAX constants (`MAX_*_TARGETS`, `MAX_JOKER_SLOTS`) define policy output/mask dimensions and are sourced from `data/action_space_config.json`.

Padding policy:
- PAD index for categorical channels
- zero-fill numeric/object matrices
- explicit validity masks for padded rows/elements

---

## 7) Normalization and Stats

For each numeric feature, define transform type:
- raw
- clipped
- log
- min-max
- z-score

Persist transform metadata and fitted statistics in artifacts for reproducible training/inference.

---

## 8) Export Artifacts

Recommended outputs:
- `train.pt`
- `val.pt`
- `test.pt`
- `artifacts/action_map.json`
- `artifacts/*_vocab.json`
- `artifacts/normalization.json`
- `artifacts/schema_versions.json`

Split policy should be deterministic and run/video partition aware to avoid leakage.

---

## 9) Known Ambiguities from Source File

`data/masking_schema.py` contains a truncated note near shop-usable tarots (`THE H`), so tensorization should:
- keep explicit schema versioning,
- record unresolved-source flags in metadata,
- avoid hardcoding behavior not explicitly captured in finalized mask/state rules.

