# Current adopted baseline: spell-frontier blend

The user authorized `joint_gru_sn_dense_spell_frontier_ensemble_v1` after the
frozen frontier diagnostic. The [promotion report](SPELL_FRONTIER_PROMOTION.md),
[active manifest](baseline_manifest.json), [policy](model_decision_policy.json)
and [receipt](results/baseline_spell_frontier_promotion.json) identify the package.
The separate [extra-data follow-up](V2_EXTRA_DATA.md) does not replace this choice.

Conditional H/A combines the stage-1024 CE model (0.5), calibrated model (0.375)
and predecessor (0.125), with arithmetic pooling and H/A transition cost 3.
Average paired seeds within each native phase, then phases. The predecessor's
decoded D mask stays exact. All ten checkpoint roles, including paired teachers,
are packaged. Promotion performs no fitting or organizer rescoring.

Original exposed development clean/v1/v2 spell F1 is
**.670137/.532128/.593619**. All three spell means improve over the predecessor;
v1 macro declines .000699 within the accepted .001 allowance. The >.60 v2 goal
and historical +.01 clean screen remain failed under an explicit adoption
exception. Provider regressions remain disclosed. The exact GAR TEST check improves spell
and macro but reduces short-spell recall by .008267 clean and .005504 v2; see
the promotion report for the complete separate cohorts and paired intervals.
Independent continuous-match
gold confirmation is pending. The saved public diagnostic is **.611765 spell /
.851285 macro** on 17 possessions, with public spell unchanged.

## Inference and local restore

The complete package is verified locally; no new private-cache upload is claimed.
The old private-cache default still identifies the historical phase-average
package. Use the [local archive receipt](results/baseline_spell_frontier_local_artifacts.json)
for exact model/source hashes and the archive location. It contains all ten
models, matching source, active manifest/policy and aggregate provenance, with
repository-relative paths verified through fresh empty-directory extraction.
Restore organizer tracking inputs separately; they are not included.

```sh
# Extract the local archive into an empty work directory or matching checkout.
tar -xzf data/models/joint_gru_sn_dense_spell_frontier_ensemble_v1_local.tar.gz -C /path/to/restored-workspace
# Install the pinned requirements and restore tracking inputs separately.
python -m experiments.current_baseline predict --output data/experiments/current_baseline/frontier_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/frontier_restored.csv
```

The CLI checks all checkpoint hashes and refuses to overwrite an output.
Expected full prediction: 199,319 rows, SHA-256
`26af7348940626da1cbe05a0c73b2a9db41ad9cedcb3ae6adc5de2771b6ce5dc`.
The normal CLI reproduces the previously sealed candidate exactly without
reading public labels. See the [restore reference](../.agents/skills/possession-research/references/restore.md).

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_sn_dense_original_diversity_phase_ensemble_v1` | [Phase-average promotion](ORIGINAL_FULL_PHASE_PROMOTION.md), [verified historical remote restore](results/baseline_original_full_phase_remote_verification.json) | [Package](baselines/joint_gru_sn_dense_original_diversity_phase_ensemble_v1.json) |
| `joint_gru_sn_dense_original_diversity_ensemble_v1` | [Phase0 promotion](ORIGINAL_FULL_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_original_diversity_ensemble_v1.json) |
| `joint_gru_sn_dense_calibrated_ensemble_v1` | [Calibrated promotion](CALIBRATED_DENSE_PROMOTION.md), [successor adoption](ORIGINAL_FULL_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json) |
| `joint_gru_sn_dense_ensemble_v1` | [Dense training](SOCCERNET_DENSE.md), [original promotion](results/baseline_sn_dense_promotion.json), [decoder successor](CALIBRATED_DENSE_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_ensemble_v1.json) |
| `joint_gru_soccernet_capacity_ensemble_v1` | [Capacity study](SOCCERNET_CAPACITY.md), [promotion](results/baseline_soccernet_capacity_promotion.json), [verified remote delivery](results/baseline_soccernet_capacity_remote_artifacts.json) | [Package](baselines/joint_gru_soccernet_capacity_ensemble_v1.json) |
| `joint_gru_combined_ensemble_v1` | [Objective ensemble](OBJECTIVE_ENSEMBLE.md), [deployment/restore receipt](results/baseline_combined_objective_artifacts.json), [closed public diagnostic](results/combined_objective_public_check.json) | [Package](baselines/joint_gru_combined_ensemble_v1.json) |
| `joint_gru_seed_ensemble_v1` | [Seed/context study](SEED_GAP_CONTEXT.md), [promotion](results/baseline_seed_ensemble_promotion.json) | [Package](baselines/joint_gru_seed_ensemble_v1.json) |
| `joint_gru_slotdrop_v1` | [Stress components and promotion rationale](STRESS_COMPONENT_ABLATION.md), [promotion](results/baseline_joint_gru_promotion.json) | [Package](baselines/joint_gru_slotdrop_v1.json) |
| `gru_hgb_ballgap25_v1` | [Light corruption](LIGHT_CORRUPTION.md), [promotion](results/baseline_ballgap25_promotion.json) | [Package](baselines/gru_hgb_ballgap25_v1.json) |
| `gru_hgb_v1` | [Temporal ensemble](TEMPORAL_ENSEMBLE.md), [historical selection](MODEL_SELECTION.md) | [Package](baselines/gru_hgb_v1.json) |

## Private-cache restore

For archived `<version>` packages, use private `baseline_cache_manifest_<version>.json` and its
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
