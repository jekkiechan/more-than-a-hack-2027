# Current adopted baseline: original-full with phase averaging

The user authorized `joint_gru_sn_dense_original_diversity_phase_ensemble_v1`
on 1 November 2026. It enables uniform sampling-phase averaging for the exact
previous original_full two-seed pooled models. The [promotion report](ORIGINAL_FULL_PHASE_PROMOTION.md),
[active manifest](baseline_manifest.json), [policy](model_decision_policy.json)
and [receipt](results/baseline_original_full_phase_promotion.json) identify the package.

At 25 Hz, evaluate all five 5 Hz sampling offsets. Average the native two-seed
probabilities within each phase, then average phases, then apply the unchanged
f062 D-logit temperature 4 and cost-3 decoder. The four checkpoint hashes,
paired teacher normalization, widths 96/24 and five-second context are unchanged.
There is no fitting or organizer rescoring during promotion.

Original exposed development clean/v1/v2 spell F1 is **.667556/.531576/.583571**.
GAR TEST (ten games, separate from nine selected-on VALID games) clean/v2 spell
is **.701641/.701289**. All seven historical active guards pass; the +.01 clean
screen remains failed with an explicit user adoption exception. Calibrated plus
phases remains the TEST/v2 spell point leader. The TEST phase gain interval
includes zero; silver segments do not establish independent continuous-match
accuracy. The saved public diagnostic is .611765 spell / .849473 macro on 17
possessions, diagnostic only.

## Private cache restore

The code promotion is merged in [PR #74](https://github.com/jekkiechan/more-than-a-hack-2027/pull/74).
Private [delivery PR #18](https://github.com/jekkiechan/mtah-data/pull/18) is
merged after explicit upload approval on 2 November 2026. All 399 package/source
entries (7,978,448 bytes) pass fresh remote download and empty-directory restore
checks with an independent LFS store. Generic and versioned cache defaults agree.
The [remote verification receipt](results/baseline_original_full_phase_remote_verification.json)
records the observed delivery state. The matching source snapshot preserves
inference code; the phase0 package stays separately restorable.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_original_diversity_phase_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/phase_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/phase_restored.csv
```

Restore organizer tracking inputs separately if missing. The default CLI checks
all checkpoint hashes and refuses to overwrite an output. Expected sealed output:
199,319 rows, SHA-256 `eb011268af14b4de77b4b1c823916631442ae2090efbb9526314007ddf27ceea`.
See the [restore reference](../.agents/skills/possession-research/references/restore.md).

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_sn_dense_original_diversity_ensemble_v1` | [Phase0 promotion](ORIGINAL_FULL_PROMOTION.md) | [Package](baselines/joint_gru_sn_dense_original_diversity_ensemble_v1.json) |
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
