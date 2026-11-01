# Current adopted baseline: dense ensemble with D-logit temperature 4

The current package is `joint_gru_sn_dense_calibrated_ensemble_v1`, architecture kind `sn_dense_calibrated_ensemble`, explicitly adopted by the user on 1 November 2026. Its default prediction is `data/experiments/current_baseline/joint_gru_sn_dense_calibrated_ensemble.csv`. The [promotion report](CALIBRATED_DENSE_PROMOTION.md), [active manifest](baseline_manifest.json), [active policy](model_decision_policy.json) and [promotion receipt](results/baseline_calibrated_dense_promotion.json) own the current version and evidence.

The same two width-96 conditional H/A GRUs, fine-tuned hidden-24 D paths and paired teachers from the dense-SoccerNet D2 recipe remain in use. Four checkpoint hashes, input features, normalizers, context and training are unchanged. Native member probabilities are normalized and averaged equally; frozen arm f062 softens D log odds with temperature 4 and uses the original cost-3 decoder. This is a versioned decoder promotion, with no retraining.

Matched clean/feed-like spell F1 is **0.658625 / 0.518527**, versus **0.647215 / 0.505811** for the previous dense package. Both seeds and every provider improve spell in both views. All declared provider guards and the historical +0.01 clean screen pass. The saved public diagnostic ties spell at 0.630303, with macro 0.864181 and D-F1 0.873502. It was known before promotion authorization, followed external selection, and did not select the rule. The five repeatedly exposed development matches, synthetic stress and 17-possession public window do not establish independent or full-match performance.

## Private cache restore

Use private branch `research/calibrated-dense-baseline-v1` until its delivery merges. Fetch this exact versioned manifest, and use the matching public inference code:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_calibrated_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/calibrated_dense_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/calibrated_dense_restored.csv
```

Restore the existing organizer input cache separately if missing. The four checkpoint hashes must match the active manifest. Choose a fresh output: existing generated CSVs are never overwritten. This public checkout has matching inference code; no source overlay is needed. A later incompatible shared-code checkout must use the versioned private source snapshot in an isolated restore. See the [restore reference](../.agents/skills/possession-research/references/restore.md).

Promotion replayed the new baseline CLI from feature-only tracking and reproduced the sealed f062 CSV byte for byte, with 199,319 valid rows. No organizer reference labels were loaded or rescored; all historical gates remain closed.

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_sn_dense_ensemble_v1` | [Dense training](SOCCERNET_DENSE.md), [original promotion](results/baseline_sn_dense_promotion.json), [decoder successor](CALIBRATED_DENSE_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_ensemble_v1.json) |
| `joint_gru_soccernet_capacity_ensemble_v1` | [Capacity study](SOCCERNET_CAPACITY.md), [promotion](results/baseline_soccernet_capacity_promotion.json), [verified remote delivery](results/baseline_soccernet_capacity_remote_artifacts.json) | [Package](baselines/joint_gru_soccernet_capacity_ensemble_v1.json) |
| `joint_gru_combined_ensemble_v1` | [Objective ensemble](OBJECTIVE_ENSEMBLE.md), [deployment/restore receipt](results/baseline_combined_objective_artifacts.json), [closed public diagnostic](results/combined_objective_public_check.json) | [Package](baselines/joint_gru_combined_ensemble_v1.json) |
| `joint_gru_seed_ensemble_v1` | [Seed/context study](SEED_GAP_CONTEXT.md), [promotion](results/baseline_seed_ensemble_promotion.json) | [Package](baselines/joint_gru_seed_ensemble_v1.json) |
| `joint_gru_slotdrop_v1` | [Stress components and promotion rationale](STRESS_COMPONENT_ABLATION.md), [promotion](results/baseline_joint_gru_promotion.json) | [Package](baselines/joint_gru_slotdrop_v1.json) |
| `gru_hgb_ballgap25_v1` | [Light corruption](LIGHT_CORRUPTION.md), [promotion](results/baseline_ballgap25_promotion.json) | [Package](baselines/gru_hgb_ballgap25_v1.json) |
| `gru_hgb_v1` | [Temporal ensemble](TEMPORAL_ENSEMBLE.md), [historical selection](MODEL_SELECTION.md) | [Package](baselines/gru_hgb_v1.json) |

For `<version>`, use private `baseline_cache_manifest_<version>.json` and its
matching public `experiments/baselines/<version>.json`. Archived policies are
`experiments/baselines/model_decision_policy_<version>.json`. For example:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_soccernet_capacity_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --manifest experiments/baselines/joint_gru_soccernet_capacity_ensemble_v1.json --output data/experiments/current_baseline/capacity_replay.csv
```

If reproducing a frozen study rather than package inference, use its original
source snapshot and sealed inputs. Never rewrite its source hashes to accept
current code. The older baseline descriptions and initial setup are available
in Git history; the evidence above remains the reproducible record.
