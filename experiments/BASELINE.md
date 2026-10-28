# Combined-objective two-seed GRU baseline

The working baseline is `joint_gru_combined_ensemble_v1`: equal native H/A/D
probabilities from combined-objective seeds 2026 and 2027, followed by the
unchanged cost-3 Viterbi decoder. Both networks retain hidden24, player-only
D inputs, ten-second output cores and five-second context on each side.

| Matched full provider holdout, five external development matches | Prior original-loss ensemble | Combined-loss ensemble |
| --- | ---: | ---: |
| Clean spell F1 | .626531 | **.633205** |
| Feed-like spell F1 | .487419 | **.498830** |
| Clean macro F1 | .854559 | .854226 |
| Feed-like macro F1 | .811127 | .811387 |

The fixed comparison improves primary spell F1 by .006674 clean and .011410
feed-like. The declared practical provider guards pass. The gain is uneven:
SkillCorner spell F1 declines .007075 clean and .007120 feed-like, while
Metrica improves .025024/.030917. Equal-provider D-F1 declines .002220/.001915;
Metrica clean D recall falls .026597 while precision rises .018437. These
tradeoffs are recorded in the [comparison report](OBJECTIVE_ENSEMBLE.md).

The user requested a comparison and, if it worked, consideration of a baseline
bump. This is a working development adoption under a new practical screen
frozen before the ensemble comparison, after inspecting the earlier loss
study. It does not change that study's failed +.01 both-seed screen. All five
development matches were previously exposed; feed-like stress is synthetic.
Independent broadcast confirmation remains pending.

## Deployment package and restore

The comparison used saved full-provider-holdout models, with no new fitting.
After it qualified, two pooled deployment members were fitted on exactly the
same thirteen external TRAIN matches, excluding all five selection matches.
Each uses 52 augmented records, 62,643 parameters, 30,560 windows per epoch and
3,824 optimizer updates. Normalization and feature schema match the prior
pooled controls exactly. The separately frozen [deployment declaration](results/combined_objective_deployment_declaration.json)
and fit receipts record these two fits; promotion only packages their outputs.

The [verified private cache](results/baseline_combined_objective_artifacts.json) default `baseline_cache_manifest.json` restores the matching
manifest and two weights under `models/joint_gru_combined_ensemble_v1/`.
The versioned copy is `baseline_cache_manifest_joint_gru_combined_ensemble_v1.json`;
`baseline_source_cache_manifest.json` restores the matching inference source.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --data-dir data
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest.json --data-dir data
python -m experiments.current_baseline predict
python evaluation/validate_submission.py data/experiments/current_baseline/joint_gru_combined_ensemble.csv
```

The full 199,319-row label-free candidate is format-validated. The preceding
original-loss ensemble reproduces its sealed CSV byte-for-byte. The adopted
package also reproduces its candidate through the actual baseline entrypoint.
Use `--output` when replaying in a workspace where a candidate already exists.

The separately requested public diagnostic was run after the package and adoption were sealed. Combined and preceding ensembles both scored **0.557576 spell F1**; combined macro was **0.854088** versus **0.862375**. This does not confirm an improvement on the 17-possession public window. The [saved report](results/combined_objective_public_check.json) contains the fixed comparison; its gate is closed and no selection or tuning followed. Full-match performance remains unknown.

Historical manifests, policies and closed scoring gates remain preserved. The preceding ensemble is archived under `experiments/baselines/` for rollback.

## Previous original-loss two-seed ensemble

`joint_gru_seed_ensemble_v1` was adopted earlier on 2026-10-28, with matched clean/feed-like spell .626531/.487419. It averaged the original slot-dropout seeds and retained five-second context. Seed 2026 was restored and the pooled 2027 member fitted on the same 13 TRAIN matches with exact normalization/budget parity. See the [original study](SEED_GAP_CONTEXT.md), [promotion receipt](results/baseline_seed_ensemble_promotion.json), and [archived manifest](baselines/joint_gru_seed_ensemble_v1.json). The versioned private restore manifest remains `baseline_cache_manifest_joint_gru_seed_ensemble_v1.json`. Its first public score is now recorded in the separate fixed diagnostic above; it was unknown at its original adoption.

## Previous single seed GRU package

`joint_gru_slotdrop_v1` remains archived under
[its original manifest](baselines/joint_gru_slotdrop_v1.json) and
[policy](baselines/model_decision_policy_joint_gru_slotdrop_v1.json).
Its private restore manifest is `baseline_cache_manifest_joint_gru_slotdrop_v1.json`.
It can still be replayed with `--manifest experiments/baselines/joint_gru_slotdrop_v1.json`.
Promotion preserves those bytes and all earlier evidence.

## Previous GRU/HGB ball-gap baseline

Its manifest and policy are preserved at `baselines/gru_hgb_ballgap25_v1.json`
and `baselines/model_decision_policy_gru_hgb_ballgap25_v1.json`. Restore it with
the private `baseline_cache_manifest_gru_hgb_ballgap25_v1.json`. Its replay through
the current runner reproduces its validated CSV (`c12854d2…83fc`):

```sh
python -m experiments.current_baseline predict --manifest experiments/baselines/gru_hgb_ballgap25_v1.json --output data/experiments/current_baseline/previous_ballgap25_replay.csv
```

The closed competition-check directory binds the pre-promotion
`current_baseline.py` hash. Re-running that closed gate is neither needed nor
allowed; its sealed outputs remain the record.

<details><summary>Previous package description</summary>


The current user-authorized working baseline is **GRU/HGB with 25% ball-gap-only
replacement in team-model training**, version `gru_hgb_ballgap25_v1`. Its GRU
and ordered HGB dead-ball heads are byte-identical to the previous package;
their P(D) probabilities still average 50/50. Viterbi switch cost remains 3.
Only the pooled contextual HGB team checkpoint changes.

The user explicitly requested this promotion after reviewing the lighter
corruption experiment. It adopts an external robustness tradeoff; it does not
claim a win under the earlier clean spell-first rule. That frozen external
selection and every completed public gate remain historical records.

| D-provider-holdout development | Previous spell / macro | Current spell / macro |
| --- | ---: | ---: |
| Clean inputs | .5746 / .8449 | .5740 / .8439 |
| Composite-corrupted inputs | .4205 / .7699 | .4315 / .7866 |

Clean spell is slightly lower, by 0.06 percentage points; corrupted spell rises
by 1.10 points. These are the same five historical development matches across
three providers, with equal provider weights and pooled team models. They are
not fresh independent confirmation. The packaged deployment heads use pooled
training; provider-holdout results describe the corresponding held-out D heads.

`baseline_manifest.json` records the user's authorization, model hashes, exact
tradeoff, unchanged D heads, previous manifest, and existing diagnostic evidence.
`results/baseline_ballgap25_promotion.json` records the package and byte-identical
inference verification. No training or organizer scoring occurs during promotion.

### Recorded competition check

| Candidate | Spell F1 | Macro F1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| Previous GRU/HGB | .4889 | .8568 | .8630 |
| Current ball-gap team model | .5257 | .8588 | .8568 |

These are existing scores on the identical public five minutes: 7,500 graded
frames and 17 reference possessions. The new candidate predicts 18 possessions,
versus 19 previously. Its official 90% spell interval is [.3657,.6743]. Full-match
competition performance remains unknown. These diagnostics were already known
when the user requested promotion; they are not independent confirmation.
Promotion reads no organizer reference labels and does not repeat either check.

The newer check is stored in `results/light_corruption_public_check.json`; the
previous check remains in `results/current_baseline_public_check.json`. Both
exclusive gates stay closed. The historical PathCRF hybrid scored .5067 spell /
.6795 macro on this window and remains an archived comparator.

### Restore and run

The private cache's default `baseline_cache_manifest.json` now restores
`models/gru_hgb_ballgap25_v1/`: the unchanged GRU and ordered HGB, new team model,
and current promotion manifest. No retraining is required.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --data-dir data
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest.json --data-dir data
python -m experiments.current_baseline predict
```

The default output is now
`data/experiments/current_baseline/gru_hgb_ballgap25.csv`. The runner validates
all 199,319 rows and records the active model/manifest hashes without reading
reference labels. Its CSV reproduces the already tested diagnostic exactly.
Existing candidate files are never overwritten; use `--output` for another path.
This is a reproducible working candidate, not an organizer-submitted entry.

The matching code/metadata snapshot can be restored using the private cache's
`baseline_source_cache_manifest.json`; its metadata records the public base
commit to overlay. Use the matching current manifest and runtime when restoring
from a checkout that predates this promotion. Input and model manifests remain
separate, and no restore step authorizes resetting a completed scoring gate.

## Older GRU/HGB baseline and frozen experiments

The previous model package remains at `models/gru_hgb_v1/`. Its exact manifest
and decision policy are preserved at `baselines/gru_hgb_v1.json` and
`baselines/model_decision_policy_gru_hgb_v1.json`. The private historical cache
manifest is `baseline_cache_manifest_gru_hgb_v1.json`.

To generate the former baseline without changing the current default:

```sh
python -m experiments.current_baseline predict --manifest experiments/baselines/gru_hgb_v1.json --output data/experiments/current_baseline/previous_gru_hgb_replay.csv
```

Historical experiment sources deliberately bind the old active manifest/policy
hashes. Reproduce them with their original snapshot and archived metadata; do
not rewrite their declarations to accept the new baseline. Future experiments
should freeze a new output directory and compare against the newly adopted
working package. See [LIGHT_CORRUPTION.md](LIGHT_CORRUPTION.md) for the training
recipe and all four experimental arms.

</details>
