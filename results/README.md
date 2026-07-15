# Results snapshots

Lightweight reports from the final training / eval runs. They support the accuracy numbers in the root README; they are not a full experiment log.

| File | Role |
|------|------|
| `train_dt_model_correctness.json` | Train-set top-1 / top-3 action match vs. expert |
| `train_dt_model_correctness_by_family.json` | Same, broken down by action family |
| `tensorizer_report.json` | Dataset scale (runs, steps, family counts) |
| `training_contract_report.json` | Schema / contract validation summary |
| `training_distribution_report.json` | Label distribution diagnostics |

**Headline (train split, filtered):** top-1 ≈ **74%**, top-3 ≈ **93%** on kept steps — strong imitation metrics that did not translate to reliable live play.
