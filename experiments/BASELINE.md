# Current baseline: Fable ball-exit correction

The selected recipe is `joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1`:
Fable’s unchanged `snap_m0.0` correction on the previous spell-frontier H/A
blend. The [manifest](baseline_manifest.json), [policy](model_decision_policy.json),
[adoption receipt](results/ball_exit_adoption_v1.json) and
[promotion report](BALL_EXIT_PROMOTION.md) identify the recipe. The exact
historical pooled CE files remain unavailable; an independently verified
four-fit reconstruction supplies the complete package. It is a new checkpoint
lineage, not byte restoration of the missing files. The reconstruction passed
reference-free full-CSV parity, the fixed public diagnostic and GAR TEST10
replay. The guarded promotion runner installed the reconstructed manifest and
policy as the working default on this research branch. Its
[promotion receipt](results/ball_exit_reconstructed_v1.json) binds the active
package. PR #92 is open and unmerged; the default has not reached main.
See the [reconstruction report](FRONTIER_POOLED_RECONSTRUCTION.md) for evidence.

Fable adjusts dead-ball starts toward observed pitch exits, using its frozen
zero-margin, five-second delay and three-second advance rule. The ball-exit rule
adds no learned component; the two pooled CE weights were reconstructed as
documented below. Conditional H/A
still combines CE/calibrated/predecessor at 0.5/0.375/0.125, pools seeds within
native phases then phases, and uses H/A cost 3. The ball-exit rule itself adds
no learned component; pooled weight reconstruction is documented below.

Matched exposed development clean/v1/v2 spell is **.681508/.534731/.598690**,
with **+.011371/+.002602/+.005071** versus the previous frontier. Macro F1 is
.876204/.827310/.854915. Short recall improves in each view; SkillCorner regresses
under its earlier silver dead-ball convention. The >.60 v2 target remains unmet;
independent continuous-match gold confirmation remains pending.

The complementary disjoint turnover mixture leads eligible development at
.598780 v2, only +.000090 over Fable. Choosing Fable is an explicit simplicity
and deployment-cost tradeoff: the complement requires additional pooled models
and shows a tiny gain on the same exposed data. It remains research-only.

## Inference and local restore

Restore the reconstructed ten-role package from its versioned private cache
manifest, following the commands in the
[reconstruction report](FRONTIER_POOLED_RECONSTRUCTION.md).
The package is independently verified in a fresh remote clone with empty LFS;
the separate fit artifacts and deployment roles have recorded hashes. Restore
tracking inputs separately. The two historical original CE identities remain
documented in the [promotion report](BALL_EXIT_PROMOTION.md#exact-deployment-artifact-blocker)
for provenance and must not be attributed to reconstructed files.

The active default CLI reproduces the sealed Fable CSV byte-for-byte at SHA-256
`56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f`. The
post-promotion experiment suite passes 977 tests with one skip; the default
package is installed on the research branch, while PR #92 remains unmerged.

```sh
# After the reconstructed model package and tracking inputs have been restored:
python -m experiments.current_baseline predict --output data/experiments/current_baseline/fable_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/fable_restored.csv
```

The reconstructed deployment run matches the saved 199,319-row reference-free
frontier CSV byte-for-byte. On the already-closed 7,500-frame/17-possession
public diagnostic, Fable reaches .623529 spell, .864234 macro and .889372 D-F1,
up from .611765/.851285/.862870. GAR TEST10 equal-game spell means are .709186
clean and .709176 v2, versus .708496/.708619 control; all aggregate macro, D-F1
and short-recall means also improve. This is repeated TEST exposure with silver
labels, and two games decline in each view, so it is not independent gold
confirmation or a uniform per-game guarantee. The predecessor’s saved public
numbers remain historical comparators. To use that archived recipe explicitly, pass
`--manifest experiments/baselines/joint_gru_sn_dense_spell_frontier_ensemble_v1.json`.
It needs the same two missing CE files. The
[promotion report](BALL_EXIT_PROMOTION.md#implementation-and-continuation)
provides the separate exact-package regression workflow.

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_sn_dense_spell_frontier_ensemble_v1` | [Frontier promotion](SPELL_FRONTIER_PROMOTION.md), [local archive](results/baseline_spell_frontier_local_artifacts.json) | [Package](baselines/joint_gru_sn_dense_spell_frontier_ensemble_v1.json) |
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
