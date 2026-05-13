# Training Data Pipeline

This document explains how Balatro gameplay data moves from parsed events into
granularized steps, then into tensorized `.npz` shards, and finally into the
policy transformer used during training and live inference.

The emphasis is on:

- how actions become per-step labels,
- how zones are converted into model-visible objects,
- which fields are encoded but not actually consumed by the model,
- how selected cards differ from selected target leaks,
- where shuffling happens,
- how masks and targets line up with the flattened action space.

Code references:

- `granularize.py`
- `tensorize.py`
- `dataset.py`
- `model.py`
- `mask_builder.py`
- `live/live_encoder.py`

Related schema docs:

- `granularization_schema.md`
- `tensorization_schema.md`
- `action_space_schema.md`
- `mask_schema.md`
- `state_schema.md`

---

## 1. Pipeline Overview

The high-level flow is:

```text
data/parsed
  -> granularize.py
data/granularized
  -> compute_persistent_state / state reducer outputs
data/persistent_state
  -> tensorize.py
data/tensorized
  -> dataset.py
BalatroStepDataset
  -> train.py
PolicyTransformer
```

At live time, `live/live_encoder.py` skips the on-disk dataset and feeds a Lua
snapshot through the same `tensorize_step()` code path used for training.

```text
Lua snapshot
  -> LiveEncoder._build_step()
  -> tensorize_step()
  -> model batch
```

The goal is for training and live inference to use the same feature contract:
same vocab IDs, same normalization, same object channels, same action map, and
same mask width.

---

## 2. Source Data: Parsed Events

The parsed layer is not the model's direct input. It is a rich record of what
the parser/OCR/object extractor saw at a game frame.

A parsed event generally has:

- `frame_idx`
- `page_name`
- `action` or `actions`
- `action_details`
- `state`: OCR-like scalar game state
- `objects`: a list of visible objects with zones, class IDs, card metadata,
  stickers, modifiers, editions, seals, debuff flags, etc.

Objects are still in raw UI zones at this point. Example zone concepts:

- `CurrentHand`
- `CurrentHandSelected`
- `PackOfferings`
- `PackOfferingsSelected`
- `CurrentConsumables`
- `CurrentConsumablesSelected`
- `CurrentJokersAll`
- `CurrentJokersSelected`
- `TarotSpectralHand`
- `TarotSpectralHandSelected`
- shop offering zones
- blind offering zones

Parsed events can have an action that represents multiple physical or logical
decisions, such as selecting several cards and then playing/discarding the hand.
That is why the next stage exists.

---

## 3. Granularization

`granularize.py` converts parsed events into one-action-per-step records under
`data/granularized/video_id=*/run_*.json`.

Granularization responsibilities:

- emit exactly one supervised action label per output step,
- resolve indexed action labels like `SelectCard_3` or `BuyShopItem_2`,
- synthesize intermediate `SelectCard_i` micro-steps when an action required
  choosing cards first,
- synthesize `SWAP_i_j` steps by comparing joker order changes,
- preserve enough source context to tensorize later without re-reading parsed
  events.

Each granularized step contains fields like:

```json
{
  "frame_idx": 29100,
  "page_name": "In_JokerStandardPlanet_Pack",
  "action": "SelectPackItem_0",
  "action_subtype": "selectpackitemjoker",
  "source_action": "SelectPackItem",
  "source_action_subtype": "selectpackitemjoker",
  "source_event_index": 7,
  "micro_index": 0,
  "source_kind": "pass_through",
  "selected_object": { "...": "..." },
  "target_index": 0,
  "swap_pair": null,
  "state": { "...": "..." },
  "objects": [ "... context/raw objects ..." ],
  "zones": { "... canonical zone projections ..." },
  "step_id": 20
}
```

### 3.1 Action Labels

The flattened action space has:

- bare actions, e.g. `PlayHand`, `SkipPack`, `CashOut`,
- single-index actions, e.g. `SelectCard_0`, `SelectPackItem_1`,
- pair-index actions, e.g. `SWAP_0_2`.

Granularization writes explicit action labels. `label_resolver.py` later maps
those strings to `target_action_id`.

Examples:

```text
SelectPackItem_0
SelectCard_4
SellItem_1
SWAP_0_2
PlayHand
SkipPack
```

For indexed single-target actions, `target_index` is also saved. For swaps,
`swap_pair` is saved. The string label is the source of the training target,
but the extra fields let consistency checks catch broken labels.

### 3.2 Candidate Ordering

Indexed action labels use a canonical candidate ordering:

```text
selected zones first, then visible/base zones
```

For example, `SelectPackItem` candidates are resolved from selected pack zones
followed by visible pack zones. This is useful for label construction, but it
can leak the current target if passed directly to the model. Tensorization
therefore treats label-resolution zones differently from model-visible zones.

### 3.3 Decomposed Card Selection

Actions like `PlayHand` and `DiscardHand` are decomposed into:

```text
SelectCard_i
SelectCard_j
...
PlayHand or DiscardHand
```

For each select-card micro-step:

- `selected_cards` contains cards already chosen in earlier micro-steps,
- `current_hand_or_pack` contains the currently visible selectable pool,
- the current target is inside `current_hand_or_pack`,
- the pool is shuffled by `granularize.py` with the granularizer RNG.

This is important: previous selections are observable state. If the player has
already selected two cards in a five-card hand, the model should see which two
cards are selected before predicting the next selection or commit action.

### 3.4 Granularizer Shuffling

`granularize.py` shuffles card selection pools during decomposition.

In `build_select_card_sequence()`:

```python
pool = copy.deepcopy(unselected_base + remaining_to_select)
rng.shuffle(pool)
```

Then each card in that shuffled pool is assigned:

```python
{"zone": pool_zone_name, "position_in_zone": i}
```

This prevents the decomposed micro-step data from always placing the target at
a predictable location. The target label is recomputed after the shuffle by
matching `slot_id`.

This shuffling happens at granularization time. It changes the generated JSON
step itself, not just tensor order.

---

## 4. Persistent State

Persistent state is the game memory carried across steps. Tensorization needs
the state before each action.

The dataset stores a parallel tree:

```text
data/persistent_state/video_id=*/run_*.json
```

Each run has a `states` array aligned with granularized step order. In
`tensorize.py`, `_persistent_state_for_step(pstates, t)` returns the
persistent state before granularized step `t`.

If the persistent-state file is missing, `tensorize.py` can rederive it by
starting from `default_state()` and applying each granularized step with
`apply_step()`.

Persistent state contributes:

- deck identity and deck flags,
- stake,
- boss blind and boss history,
- redeemed vouchers,
- tracked deck cards,
- poker-hand levels and play counts,
- persistent counters such as `skips`, `hands_played`, `unused_discards`,
  `ecto_minus`,
- boolean flags such as `first_hand`, `first_discard`,
  `is_boss_blind_rerolled`.

---

## 5. Tensorization

`tensorize.py` converts each granularized step plus its persistent-before state
into a dictionary of NumPy arrays. Those per-step dictionaries are stacked into
per-run `.npz` shards under:

```text
data/tensorized/video_id=*/run_*.npz
```

The public entry point is:

```python
tensorize_step(step, persistent_state, action_map, vocab, norm, feature_config)
```

Tensorization does five major things:

1. canonicalize the step shape,
2. build global scalar features,
3. build object and tracked-deck token matrices,
4. build the action legality mask,
5. resolve the supervised target action ID.

---

## 6. Step Canonicalization

Before feature building, `tensorize_step()` calls:

```python
canonicalize_step_for_tensorization(step)
```

This function normalizes the many possible representations of dynamic objects
into a consistent shape:

- stable context objects in `objects`,
- active selectable candidates in `current_hand_or_pack`,
- previous selected cards in `selected_cards`.

Training data can carry active pools in several places:

- top-level `current_hand_or_pack`,
- legacy top-level `current_hand`,
- `zones.current_hand_or_pack`,
- raw `objects` with dynamic zones like `CurrentHand` or `PackOfferings`.

Canonicalization chooses the first available source in that order.

### 6.1 Dynamic Zones

`tensorize.py` defines dynamic pool zones:

```text
CurrentHand
CurrentHandOrPackOfferings
PackOfferings
CurrentPack
TarotSpectralHand
```

and dynamic selected zones:

```text
CurrentHandSelected
CurrentHandOrPackOfferingsSelected
PackOfferingsSelected
CurrentPackSelected
TarotSpectralHandSelected
```

Together these form `DYNAMIC_OBJECT_ZONES`.

During canonicalization:

- objects in dynamic zones are removed from the stable context object list,
- dynamic pool objects go into `current_hand_or_pack`,
- dynamic selected objects go into `selected_cards` unless a canonical
  `selected_cards` field already exists.

This separation lets tensorization later rebuild the object token list in a
controlled, leak-aware order.

### 6.2 Context Objects

Context objects are raw objects not in dynamic pool/selected zones. Examples:

- `CurrentDeck`
- `CurrentStake`
- `CurrentJokersAll`
- `CurrentConsumables`
- shop zones when they are not being pulled into a dynamic hand/pack pool

These are kept in `objects`.

Before they become model-visible tokens, context objects are passed through
selected-zone leak normalization, described later.

---

## 7. Zones and What the Model Sees

The policy transformer sees zones only through object tokens:

```text
object_zone_id
```

Each object token contains:

- `object_class_id`
- `object_object_type_id`
- `object_zone_id`
- `object_position`
- `object_modifier_id`
- `object_edition_id`
- `object_seal_id`
- `object_rank_id`
- `object_suit_id`
- `object_is_debuffed`
- `object_sticker_rental`
- `object_sticker_perishable`
- `object_sticker_eternal`
- `object_mask`

The zone ID is categorical and comes from `artifacts/vocab.json`.

### 7.1 Model-Visible Selected Cards

Prior selected playing cards are intentionally visible to the model.

These are cards already selected by previous decisions in a multi-card sequence.
They are not the current target. They are part of the observable game state.

Examples:

- during a `PlayHand` decomposition, after the first selected card,
  `selected_cards` contains that chosen card before the next `SelectCard_i`,
- during tarot/spectral card targeting, already-selected playing cards may
  remain visible before the next selection.

These remain as selected zones:

```text
CurrentHandSelected
TarotSpectralHandSelected
```

So the model can learn things like:

- "these two cards are already selected,"
- "the next selection should complement the already-selected card,"
- "commit action is appropriate once enough cards have been selected."

### 7.2 Current-Target Selected-Zone Leaks

Some raw selected zones encode what the player clicked in the current action.
Those should not be visible to the model, because they reveal the label.

The classic leak is:

```text
PackOfferingsSelected
```

If a training row labeled `SelectPackItem_0` contains the chosen pack item in
`PackOfferingsSelected`, the model can learn:

```text
if PackOfferingsSelected exists, choose SelectPackItem
if it does not exist, choose SkipPack
```

That is not available at live pre-action time and causes train/live mismatch.

To prevent this, tensorization merges raw selected-zone target artifacts into
their non-selected counterpart.

Examples:

```text
PackOfferingsSelected       -> PackOfferings
CurrentPackSelected         -> CurrentPack
CurrentConsumablesSelected  -> CurrentConsumables
TopShelfShopOfferingsSelected -> TopShelfShopOfferings
ShopOfferingsSelected       -> ShopOfferings
```

`CurrentJokersSelected` is the exception. It is intentionally preserved because
`CurrentJokersAll` already contains the selected joker copy while preserving
joker order. The selected-zone copy is not used to reconstruct joker slot order.

### 7.3 How the Code Decides What to Preserve

The distinction is source-aware.

In `canonicalize_step_for_tensorization()`:

- if `selected_cards` came from the canonical top-level `selected_cards` field
  and its selected zone is a legitimate playing-card selected zone, it is
  considered observable;
- if selected objects are recovered from raw dynamic selected zones, they are
  treated as target-leak artifacts.

Currently observable selected-card zones are:

```text
CurrentHandSelected
TarotSpectralHandSelected
```

Legacy `CurrentHandOrPackOfferingsSelected` can be treated as observable only
on non-pack pages where it represents hand-card selection, not pack target
selection.

---

## 8. Object List Reconstruction

The final model-visible object list is built in `_merge_hand_into_objects()`.

The broad structure is:

```text
context objects
  + selected_cards
  + current_hand_or_pack
```

But the exact handling depends on whether `selected_cards` are observable prior
state or target-leak artifacts.

### 8.1 Observable Prior Selected Cards

If `selected_cards` are observable:

```text
context objects
  + selected_cards tagged as selected zone
  + pool cards tagged as base zone
```

Example:

```text
CurrentHandSelected:  previously selected card
CurrentHand:          remaining selectable hand cards
```

Those selected-card tokens stay visible as selected tokens.

### 8.2 Non-Observable Selected Target Artifacts

If selected objects came from target-leak zones:

```text
selected target artifact + base pool
  -> merge into base pool
  -> deterministic shuffle inside the merged zone
  -> reassign position_in_zone from 0..N-1
```

Example:

```text
PackOfferingsSelected + PackOfferings
  -> PackOfferings
```

The selected object remains visible as an object, but not as "the selected one."
It becomes just another pack offering.

This preserves object content while removing the label leak.

---

## 9. Shuffling

There are two shuffling mechanisms in the data path.

### 9.1 Granularizer Pool Shuffling

`granularize.py` shuffles decomposed selection pools with a seeded
`random.Random` instance.

This applies to selection-heavy actions such as hand selection. The purpose is
to avoid always placing the target at a predictable location when synthesizing
micro-steps.

This shuffle modifies the emitted granularized step.

### 9.2 Tensorizer Selected-Zone Merge Shuffling

`tensorize.py` uses deterministic hash-based shuffling when it merges raw
selected-zone target artifacts into their base zone.

The function is:

```python
merge_selected_zones_for_model(...)
```

It builds a stable shuffle key from:

- step identity seed (`frame_idx`, `source_event_index`, `micro_index`,
  `step_id`, `page_name`),
- object identity-ish fields (`slot_id`, `class_id`, `object_type`, zone,
  position, card metadata),
- the original object index.

Then it:

1. normalizes `*Selected` zone names to base zones,
2. groups objects in zones that had selected objects merged,
3. sorts each group by the stable hash,
4. assigns fresh `position_in_zone` values.

The shuffle is deterministic: the same input produces the same tensors. It is
not random each epoch.

The purpose is to avoid teaching the model that the de-selected target always
lands at the front or back after merge.

---

## 10. Global Feature Channels

Tensorization emits several global channels.

### 10.1 Global Categorical IDs

Emitted by `_build_global_categoricals()`:

```text
page_id
source_kind_id
action_subtype_id
deck_class_id
stake_class_id
last_tarot_planet_class_id
ante_boss_blind_class_id
small_status_id
big_status_id
```

Important: `source_kind_id` and `action_subtype_id` are tensorized and stored
for diagnostics/compatibility, but the model deliberately does not consume
them. They are leakage-prone because they are derived from the granularizer or
the labeled action.

`model.py` consumes only these global categoricals:

```text
page_id
deck_class_id
stake_class_id
last_tarot_planet_class_id
ante_boss_blind_class_id
small_status_id
big_status_id
```

### 10.2 OCR Numeric State

Emitted as `ocr_numeric` with a parallel validity bit vector `ocr_valid`.

Keys:

```text
hands_left
discards_left
dollars
ante
round
deck_remaining
deck_total
round_score
cash_out
reroll_price
consumables_current
consumables_total
jokers_current
jokers_total
hand_size_current
hand_size_total
```

Missing or null numeric values encode as zero in `ocr_numeric`, with the
corresponding `ocr_valid` bit set to false.

Legacy parser fields removed before numeric encoding:

```text
hand_and_level_raw
ocr_extra
```

### 10.3 Persistent Numeric State

Emitted as `state_numeric`:

```text
skips
hands_played
unused_discards
ecto_minus
```

These come from persistent state, not the OCR `state` dict.

### 10.4 Boolean Flags

Emitted as `flags`:

```text
first_hand
first_discard
is_boss_blind_rerolled
deck.is_magic
deck.is_nebula
deck.is_abandoned
deck.is_checkered
deck.is_zodiac
deck.is_erratic
deck_modifiers.no_face_cards_start
deck_modifiers.spades_hearts_only_start
deck_modifiers.randomized_starting_deck
```

### 10.5 Per-Hand Arrays

The model receives per-poker-hand arrays:

```text
hand_levels
hand_played
hand_played_this_round
```

Order is fixed by `artifacts/feature_config.json`.

### 10.6 Multi-Hot Inventories

The model receives:

```text
vouchers_redeemed
bosses_used
```

These are boolean vectors over fixed class-ID lists in
`artifacts/feature_config.json`.

---

## 11. Tracked Deck Tokens

Tracked deck cards are read from persistent state:

```python
pstate["tracked_deck_cards"]
```

They become padded deck-card token channels:

```text
deck_card_class_id
deck_card_modifier_id
deck_card_edition_id
deck_card_seal_id
deck_card_rank_id
deck_card_suit_id
deck_card_mask
```

Tracked deck tokens do not include:

- zone,
- position,
- object type,
- debuff,
- stickers.

The model treats tracked deck cards as their own token stream with a separate
type embedding.

---

## 12. Object Tokens

Object tokens are built from the reconstructed object list described earlier.

For each object, tensorization emits:

```text
object_class_id
object_object_type_id
object_zone_id
object_position
object_modifier_id
object_edition_id
object_seal_id
object_rank_id
object_suit_id
object_is_debuffed
object_sticker_rental
object_sticker_perishable
object_sticker_eternal
object_mask
```

Objects are capped by `MAX_OBJECTS_PER_STEP` from `artifacts/feature_config.json`.
Current reports print this value as 48.

Padding:

- categorical channels: zero,
- numeric/bool channels: zero/false,
- `object_mask`: false for padded rows.

Only rows with `object_mask == true` participate in transformer attention.

---

## 13. What Is Not Fed to the Model

Some data exists in granularized JSON or tensorized shards but is not consumed
by `PolicyTransformer`.

### 13.1 Stored but Not Used by the Model

Tensorized but intentionally excluded in `GlobalEncoder`:

```text
source_kind_id
action_subtype_id
```

Reason: leakage.

- `source_kind_id` can say whether a row is a synthetic select step, commit
  step, swap step, etc.
- `action_subtype_id` often directly reveals the action family.

They remain in `.npz` records, but `model.py` does not embed them.

### 13.2 Label-Only / Supervision Fields

These are not model inputs:

```text
action
source_action
source_action_subtype
selected_object
target_index
swap_pair
target_action_id
```

`target_action_id` is the supervised label used by the loss. It is not an
input feature.

`selected_object` describes the action target for bookkeeping/state updates;
it is not encoded into model features.

### 13.3 Raw Zone Snapshots

The raw `zones` dictionary is not tensorized directly. It is only a fallback
source used during canonicalization to reconstruct:

- `current_hand_or_pack`,
- `selected_cards`,
- dynamic pools.

After canonicalization, model inputs are object tokens, not the raw `zones`
dictionary.

### 13.4 Live Legal Actions

Live snapshots include:

```text
legal_actions
```

This is not a learned feature. `live/live_encoder.py` converts it into
`action_mask` and overrides the tensorizer's approximate mask.

The model sees the mask only by having illegal logits set to `-inf`, not as an
ordinary feature vector.

---

## 14. Action Masks

`mask_builder.py` creates `action_mask`.

The mask has length `N_ACTIONS`, which is currently 98.

Mask rules include:

- page gating: only action families legal on the current page are allowed,
- target-exists gating: indexed actions are unmasked only for existing
  candidates,
- swap gating: `SWAP_i_j` requires enough joker slots.

For dynamic select families (`SelectCard`, `SelectPackItem`), candidate count
uses:

```text
len(selected_cards) + len(current_hand_or_pack)
```

when those canonical fields are present.

At live time, Lua emits exact legal actions. `LiveEncoder` converts those labels
to a mask and overwrites the tensorizer-built mask because Lua has the most
accurate legality information.

---

## 15. Target Resolution

`label_resolver.py` maps the granularized `action` string to the integer
`target_action_id`.

Examples:

```text
SkipPack          -> index 8
SelectPackItem_0  -> index 28
SelectCard_3      -> index 17
SWAP_0_2          -> one of the SWAP family indices
```

If the action label is missing or unknown, tensorization stores:

```text
target_action_id = -1
```

`dataset.py` excludes those unresolved rows by default.

### 15.1 Spurious Pack Skip Filter

Before tensorizing each run, `_process_run()` filters a known bad pattern:

```text
SkipPack
followed immediately by SelectPackItem_* or SelectCard_* on the same page
```

A real `SkipPack` closes the pack, so those rows are treated as parser/UI
artifacts. The filter increments:

```text
filtered.spurious_skippack_before_selection
```

The recent tensorizer report shows 326 such rows filtered.

---

## 16. Dataset Loading

`dataset.py` loads tensorized `.npz` files for a split.

Splits come from:

```text
artifacts/splits.json
```

The split unit is video ID, so all runs for a video stay together.

`BalatroStepDataset`:

- loads all matching run shards,
- concatenates each channel into one tensor,
- builds a `_valid` index of rows with `target_action_id >= 0`,
- optionally moves all tensors to the training device.

During training, batches are gathered by index:

```python
batch = train_ds.gather_batch(idx)
target = batch["target_action_id"]
```

---

## 17. Model Inputs

`PolicyTransformer` consumes the batch dict and builds three token groups:

```text
CLS token
global token
object tokens
tracked deck tokens
```

### 17.1 Global Token

`GlobalEncoder` embeds:

```text
page_id
deck_class_id
stake_class_id
last_tarot_planet_class_id
ante_boss_blind_class_id
small_status_id
big_status_id
```

and concatenates numeric/bool arrays:

```text
ocr_numeric
state_numeric
ocr_valid
flags
hand_levels
hand_played
hand_played_this_round
vouchers_redeemed
bosses_used
```

It deliberately does not use:

```text
source_kind_id
action_subtype_id
```

### 17.2 Object Tokens

`CardLikeTokenEncoder` encodes object tokens with:

- class,
- modifier,
- edition,
- seal,
- rank,
- suit,
- object type,
- zone,
- position,
- debuff flag,
- sticker flags.

Object tokens use `object_mask` as the transformer padding mask.

### 17.3 Tracked Deck Tokens

Tracked deck tokens use the same card-like encoder class but without:

- zone,
- position,
- object type,
- debuff,
- stickers.

They use `deck_card_mask` as the padding mask.

### 17.4 Transformer and Masked Logits

The final sequence is:

```text
[CLS, global, object_0, ..., object_N, deck_0, ..., deck_M]
```

The transformer encodes the sequence, the CLS output goes through the policy
head, and then:

```python
logits = logits.masked_fill(~action_mask, -inf)
```

Training loss is cross entropy over masked logits. Inference softmax is also
over legal actions only.

---

## 18. Training

`train.py` trains behavior cloning:

- loads train/val/test `BalatroStepDataset`,
- builds `PolicyTransformer`,
- optimizes cross entropy against `target_action_id`,
- evaluates top-1/top-3 and per-family accuracy,
- saves the best checkpoint by validation loss to:

```text
artifacts/checkpoints/best.pt
```

The checkpoint stores:

- `model_state_dict`,
- `optimizer_state_dict`,
- `model_config`,
- `n_actions`,
- validation metrics.

Training report is written to:

```text
artifacts/checkpoints/training_report.json
```

---

## 19. Live Encoding

`live/live_encoder.py` bridges Lua snapshots into the same tensor path.

Lua snapshot fields:

```text
page_name
state
objects
current_hand_or_pack
selected_cards
persistent_state
legal_actions
```

`LiveEncoder._build_step()` creates a tensorizer-compatible step:

```python
{
    "page_name": snapshot.get("page_name"),
    "source_kind": snapshot.get("source_kind"),
    "action_subtype": snapshot.get("action_subtype"),
    "state": snapshot.get("state") or {},
    "objects": snapshot.get("objects") or [],
    "current_hand_or_pack": snapshot.get("current_hand_or_pack") or [],
    "selected_cards": snapshot.get("selected_cards") or [],
    "target_index": None,
    "action": "",
}
```

Then it calls `tensorize_step()`.

After tensorization, it replaces `record["action_mask"]` with the Lua
`legal_actions` mask. This keeps live legality faithful to the game engine.

---

## 20. Concrete Zone Examples

### 20.1 Hand Selection

Suppose a hand-selection micro-step has:

```text
selected_cards = [7 of Hearts]
current_hand_or_pack = [7 of Spades, 8 of Hearts, 9 of Clubs]
```

Tensorization emits object tokens like:

```text
CurrentHandSelected: 7 of Hearts
CurrentHand:         7 of Spades
CurrentHand:         8 of Hearts
CurrentHand:         9 of Clubs
```

This is intentional. The selected card is previous state.

### 20.2 Pack Selection Leak

Suppose a raw training row has:

```text
PackOfferingsSelected: chosen joker
PackOfferings:         other joker
```

Tensorization should not emit `PackOfferingsSelected`. It emits:

```text
PackOfferings: chosen joker
PackOfferings: other joker
```

with deterministic shuffled ordering and fresh positions.

The chosen joker is still visible as an offering, but it is not marked as the
chosen one.

### 20.3 Tarot/Spectral Multi-Card Selection

For tarot/spectral selection where previous selected playing cards matter,
`TarotSpectralHandSelected` remains visible:

```text
TarotSpectralHandSelected: already selected card
TarotSpectralHand:         remaining candidate card
```

This is observable state and is not treated as an action-target leak when it
comes from canonical `selected_cards`.

### 20.4 CurrentJokersSelected

`CurrentJokersSelected` is exempt from selected-zone merge because joker order
logic uses `CurrentJokersAll`, which intentionally includes the selected joker
copy while preserving slot order.

---

## 21. Current Invariants Worth Checking

Useful checks after tensorization:

```text
No PackOfferingsSelected in object_zone_id
No CurrentPackSelected in object_zone_id
No CurrentConsumablesSelected leak unless intentionally allowed
CurrentHandSelected exists for prior selected hand cards
TarotSpectralHandSelected exists for prior tarot/spectral selected cards
CurrentJokersSelected may exist
target_action_id >= 0 for training rows
action_mask[target_action_id] should be true for resolvable rows
```

A quick selected-zone scan can reveal whether a leak reappeared:

```python
import json, numpy as np, pathlib
vocab = json.load(open("artifacts/vocab.json", encoding="utf-8"))["vocabularies"]["zone"]["values"]
counts = {}
for fp in pathlib.Path("data/tensorized").glob("video_id=*/run_*.npz"):
    z = np.load(fp)
    vals = z["object_zone_id"][z["object_mask"]]
    for i in vals.tolist():
        name = vocab[int(i)]
        if "Selected" in name:
            counts[name] = counts.get(name, 0) + 1
print(counts)
```

Expected selected-zone categories after the current fix:

```text
CurrentHandSelected
TarotSpectralHandSelected
CurrentJokersSelected
```

`PackOfferingsSelected` should not appear in tensorized object inputs.

---

## 22. Summary

The core contract is:

- granularization creates supervised one-action steps and labels,
- tensorization converts only observable state into model features,
- target labels and current-action selected zones must not leak into inputs,
- previous selected playing cards are observable and should remain visible,
- raw selected target artifacts are merged into base zones and shuffled,
- tensorized data may include diagnostic/leaky fields, but the model excludes
  known leaky global channels,
- live inference reuses the same tensorizer and overrides legality with
  engine-provided `legal_actions`.

This separation lets the model learn from the same kind of state it will see
at live decision time while preserving enough supervision to train a
behavior-cloned policy.
