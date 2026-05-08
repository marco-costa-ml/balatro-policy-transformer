# Balatro Parse Schema

## 1) Scope and Source Alignment

This document defines parse-stage behavior (`extracted -> parsed`) and aligns terminology to `data/masking_schema.py`.

Parse responsibilities:
1. normalize OCR fields,
2. flatten zone objects into typed object rows,
3. preserve action `type/id/subtype` metadata,
4. split runs on `StartNewRun`.

Parse does not perform:
- decomposition (`SelectCard`, `SWAP`),
- action masking,
- persistent-state transitions.

---

## 2) Input and Output Contracts

Input:
- `data/extracted/video_id=*/events.json`

Output:
- `data/parsed/video_id=*/run_###.json`
- `data/parsed/config.json` (generated schema manifest)

Top-level input may be:
- object with `events`,
- or event list directly.

---

## 3) Parsed Event Shape

```json
{
  "frame_idx": 128,
  "page_name": "In_TarotSpectral_Pack",
  "state": {
    "hands_left": 5,
    "discards_left": 4,
    "dollars": 57,
    "ante": 19,
    "round": 54,
    "deck_remaining": 20,
    "deck_total": 29,
    "round_score": null
  },
  "objects": [parsed_object, ...],
  "actions": ["SkipPack"],
  "action_details": [
    {"type": "SkipPack", "id": "pred_skiptarotspectralpack_128_0", "subtype": "skiptarotspectralpack"}
  ]
}
```

Run envelope:

```json
{
  "video_id": "2726526327",
  "run_index": 0,
  "events": [...]
}
```

---

## 4) OCR Normalization

Normalized `state` fields:
- `hands_left`, `discards_left`, `dollars`, `ante`, `round`, `deck_remaining`, `deck_total`, `round_score`

Transforms:
- strip non-digits and cast to int,
- `ante` uses numerator of `X/Y`,
- `deck_values` maps to `deck_remaining` + `deck_total`.

---

## 5) Object Parsing

Flatten `state.zones` into `objects`.

Parsed object fields:
- `slot_id`, `zone`, `position_in_zone`, `class_id`, `object_type`,
- `card`, `edition`, `modifier`, `seal`, `stickers`, `is_debuffed`.

Object type inference follows class IDs and class-name prefixes.

Note:
- parse currently keeps class-based typing behavior from code,
- higher-level facedown/metadata semantics are defined in `state_schema.md`.

---

## 6) Action Parsing

`actions` normalization accepts:
- string,
- object with `type`,
- list of either.

`action_details` stores:
- `type`,
- `id`,
- `subtype` parsed with `pred_(.+)_\d+_\d+`.

---

## 7) Run Splitting

Split action: `StartNewRun`.

Rules:
1. each event containing `StartNewRun` starts a run,
2. split event is included in that run,
3. events before first split are discarded,
4. fallback: if no split is present, emit one run from all events.

---

## 8) Zone Naming Compatibility

`data/masking_schema.py` contains overloaded legacy zone names:
- `CurrentHandOrPackOfferings`
- `CurrentHandOrPackOfferingsSelected`

Current extractor variants may also emit split zones (`CurrentHand*`, `PackOfferings*`) and shop-specific zones.

Parse must preserve zone names as observed (pass-through semantics) and avoid collapsing by assumption.

---

## 9) Validation Checklist

After parse:
1. every partition emits run files,
2. `run_index` is contiguous per video,
3. `frame_idx` is non-decreasing per run,
4. `actions` and `action_details` remain aligned,
5. slot identities are stable across adjacent frames where objects persist,
6. generated `config.json` zone list matches observed zones.

---

## 10) Pipeline Risk Notes (Parse-Relevant)

- schema drift between extracted zones and parse config can break downstream target resolution.
- permissive parse defaults can hide extraction errors unless warnings are emitted.
- stale parsed outputs (old zone conventions) can silently conflict with current masking/granularization assumptions.
