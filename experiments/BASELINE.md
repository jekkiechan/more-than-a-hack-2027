# Current two seed GRU ensemble baseline

The user-authorized working baseline is **the equal 2026/2027 probability
ensemble**, package `joint_gru_seed_ensemble_v1`, adopted on 2026-10-28.
Both networks are the original slot-dropout joint GRU with a player-only D
path: hidden24, ten-second output cores and five seconds of context per side.
The runner shares one feature pass, averages their native H/A/D probabilities
50/50, then uses the unchanged Viterbi switch cost 3.

| Full provider holdout on the same five external matches | Former seed 2026 | Seed 2027 | Adopted ensemble |
| --- | ---: | ---: | ---: |
| Clean spell F1 | .608763 | .617600 | **.626531** |
| Feed-like spell F1 | .474237 | .485302 | **.487419** |
| Clean macro F1 | .843432 | .854336 | **.854559** |
| Feed-like macro F1 | .801255 | .810874 | **.811127** |

The ensemble improves clean/feed-like spell by .01777/.01318 over the former
working seed and .00893/.00212 over seed 2027. It passes the same clean screen,
feed-like floor and provider D/macro guards relative to the former baseline.
These remain previously exposed development matches with synthetic stress;
independent confirmation is pending. See the [matched study](SEED_GAP_CONTEXT.md)
and [promotion receipt](results/baseline_seed_ensemble_promotion.json).

The fifteen-second-context recipe is **not** adopted. It improves some long-gap
frames but does not give consistent spell gains, and its seed 2026 fails a
SkillCorner D guardrail. The ensemble retains the original five-second context.

## Deployment package and restore

Seed 2026 was restored from the sealed former package. The archive had only
provider-held-out 2027 models, so one pooled deployment checkpoint was fitted
on the same thirteen TRAIN matches, excluding all five selection matches. Its
normalization, feature schema, 62,643 parameters, 30,560 windows per epoch and
3,824 optimizer updates match pooled seed 2026 exactly. No ensemble-comparison
control was retrained; no organizer labels were read or scored. The separate
[deployment declaration](results/seed_ensemble_deployment_declaration.json)
records that one fit; the promotion operation only packages and predicts.

The private cache's default `baseline_cache_manifest.json` restores the two
weights and matching manifest under `models/joint_gru_seed_ensemble_v1/`.
The versioned copy is `baseline_cache_manifest_joint_gru_seed_ensemble_v1.json`.
Use `baseline_source_cache_manifest.json` for the matching code snapshot.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --data-dir data
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest.json --data-dir data
python -m experiments.current_baseline predict
python evaluation/validate_submission.py data/experiments/current_baseline/joint_gru_seed_ensemble.csv
```

The label-free 199,319-row CSV was format-validated and replayed byte-for-byte
through the actual baseline entrypoint. Original seed-2026 prediction parity
also reproduces its previously sealed CSV. Generated candidates are never
overwritten; use `--output` when replaying in an existing workspace.

The new ensemble has **no organizer public score**. The saved .5818 spell /
.8689 macro result belongs to the former single-seed GRU on only 17 reference
possessions. That historical result is retained separately in the manifest,
and every public scoring gate stays closed.

[Experiment progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9)
shows dated external milestones and the separately qualified organizer checks.

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
