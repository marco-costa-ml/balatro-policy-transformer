# Source extract

Core modules from the abandoned policy stack, kept here for reading and reference.

| Module | Role |
|--------|------|
| `model.py` | Branched autoregressive policy transformer |
| `family_map.py` / `argument_spec.py` | Family taxonomy + decoder argument shapes |
| `branched_loss.py` | Multi-head training loss |
| `mask_builder.py` | Legality masks |
| `logit_adjust.py` | Rare-action logit adjustment |
| `dataset.py` / `train.py` | Data loading and training loop |
| `eval_branched.py` | Offline correctness evaluation |
| `super_step.py` / `history_features.py` / `supplement_features.py` | Step construction extras |

Paths inside these files still refer to the original flat layout (`data/`, `artifacts/`). The complete pipeline, live agent bridge, analytics, tests, and datasets live only in the local `_local/` workspace (not published).
