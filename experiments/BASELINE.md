# Current adopted baseline: original-possession diversity ensemble

The current package is `joint_gru_sn_dense_original_diversity_ensemble_v1`, architecture kind `sn_dense_original_diversity_ensemble`, explicitly adopted by the user on 1 November 2026 after consultation with Astra. It uses the already-sealed `original_full` pooled seed-2026/2027 checkpoints from the fixed public diagnostic. The [promotion report](ORIGINAL_FULL_PROMOTION.md), [active manifest](baseline_manifest.json), [active policy](model_decision_policy.json) and [promotion receipt](results/baseline_original_full_promotion.json) own the current package and evidence.

The width-96 conditional H/A GRUs and fine-tuned D paths use the original-possession diversity sampler, raw-feed training views and unchanged paired teacher normalization. Average the native full H/A/D probability vectors equally, then apply the frozen temperature-4 D-logit transform and cost-3 decoder. Phase averaging and stacking are not part of this package. Promotion restores existing checkpoints; it does no fitting or organizer scoring.

Matched clean/feed-like spell F1 is **0.662590 / 0.528180**, versus the previous calibrated dense package's **0.658625 / 0.518527**. All feed-first provider guards pass. The historical +0.01 clean screen **fails**; explicit user adoption records an exception and preserves that failed result. SkillCorner feed spell declines by .003720 against the predecessor, within its guard. The five exposed development matches and synthetic views do not establish independent or full-match performance.

The already-known public diagnostic is **0.611765 spell / 0.851742 macro / 0.862420 D-F1**, versus predecessor .630303/.864181/.873502 on 17 reference possessions. These saved results are diagnostic only and do not select the recipe. The public gate remains closed.

## Private cache restore

Use private delivery branch `research/promote-original-full-v1` until its PR is merged; the versioned manifest is authoritative even if a generic default or private main is older. The four-checkpoint weight cache and matching source cache are versioned separately. See the promotion report for current remote verification and site delivery state.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_original_diversity_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/original_full_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/original_full_restored.csv
```

Restore the existing organizer input cache separately if missing. Model hashes must match the active manifest. Use a fresh output path. Matching public code is required; the private source snapshot supports an isolated version restore after incompatible shared-code changes. See the [restore reference](../.agents/skills/possession-research/references/restore.md).

Fresh feature-only baseline inference reproduces the sealed **199,319-row** candidate CSV byte for byte, SHA-256 `7fe3da71b0cbd66bee046597611aac016fd714ddfaec6ba924dc400d7466bd1f`. No organizer reference labels are loaded or rescored. The default output is `data/experiments/current_baseline/joint_gru_sn_dense_original_diversity_ensemble.csv`.

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_sn_dense_calibrated_ensemble_v1` | [Calibrated promotion](CALIBRATED_DENSE_PROMOTION.md), [successor adoption](ORIGINAL_FULL_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json) |
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
