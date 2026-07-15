# Design docs

These notes describe how expert Balatro runs were turned into a training signal for a factorized policy.

| Doc | What it covers |
|-----|----------------|
| [action_space.md](action_space.md) | Flat labels vs. family + argument factorization |
| [state.md](state.md) | Snapshot fields the model conditions on |
| [masking.md](masking.md) | Legal-action masks at train and inference time |
| [training_pipeline.md](training_pipeline.md) | End-to-end extract → train flow |
| [granularization.md](granularization.md) | How raw events become decision steps |
| [tensorization.md](tensorization.md) | On-disk tensor layout |

Companion machine-readable configs live in [`../configs`](../configs).
