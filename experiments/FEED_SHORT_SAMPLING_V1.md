# Short-possession sampling and selected-model calibration

Status: completed, 1 November 2026. Baseline unchanged.

The current experimental leader is the two-seed raw-feed augmented model:
clean/feed-like spell F1 .654740/.525240. The adopted calibrated baseline is
.658625/.518527. The next stretch remains .55 on the original frozen benchmark.

## Hypothesis and intervention

The earlier subsecond weighting study changed the loss contribution of rare
examples but did not change their exposure. This study replaces 25% of window
draws with an existing window containing a complete H/A possession lasting at
most three seconds, within the same already-selected record. It uses a separate
seeded RNG and falls back to the original window when no eligible pool exists.
Native unknown spans, time breaks and record-edge censoring remain intact.
This is explicitly label-conditioned TRAIN sampling; inference never uses it.

Preserving the record preserves source, match, clean/defective variant and
paired H/A orientation selection. Selected windows retain their original
alternate-feature flags. GAR swap and batch-shuffle RNG states are unchanged.
The new window distribution trains both H/A and D; D is not held fixed.
The audit replays every batch and compares source/match/game exposure with the
archived augmented-control receipts before any candidate fit.

Six fits use the same provider-excluded teachers and normalizers, seeds
2026/2027, width96, 2,048 steps, batch32, half original/half GAR, unchanged
losses and raw-feed views. No new labels, training data, architecture or decoder.
The decoder remains D temperature4 followed by cost3.

## Frozen comparison and stopping rule

Two fixed arms: the new two-seed probability ensemble and an equal native
probability blend with the two-seed raw-feed control. All six checkpoints are
sealed before scoring. Rank by original feed-like spell F1; macro breaks ties.
A candidate must exceed .525240 and pass the promoted-baseline guards: clean
spell within .005; each provider clean macro and D-F1 within .02; each provider
feed spell within .01. Report the historical +.01 clean screen separately.
There is no replacement-rate, duration-threshold, blend-weight or seed sweep.

If an eligible arm improves the original benchmark, evaluate that fixed winner
on the previous ten extra corruption views. Those views and their results are
already exposed; this is an additional development diagnostic, not independent
confirmation and not a new selection set. No organizer reference scoring or
promotion occurs. The same five exposed development matches, synthetic stress
and silver-label limitations continue to apply.

## Reproduction

Restore the parent raw-feed study and its pinned original dependencies first.
New frozen source/input bindings, TRAIN audit, checkpoints and score locks must
not be overwritten. From repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/audit_feed_short_sampling_v1.py
.venv/bin/python -m experiments.feed_short_sampling_v1 declare
.venv/bin/python -m experiments.feed_short_sampling_v1 preflight
.venv/bin/python -m experiments.feed_short_sampling_v1 fit --fold leave_out_metrica --seed 2026
# Complete the other five fixed pairs, then:
.venv/bin/python -m experiments.feed_short_sampling_v1 seal
.venv/bin/python -m experiments.feed_short_sampling_v1 evaluate
```

The target-mask, same-record/RNG, disabled-training parity and exception cleanup
contracts are in `experiments/tests/test_feed_short_sampler.py`.

## Predeclared component check

Before parent scores, `feed_short_head_v1` fixes one conditional follow-up if
resampling misses .55: use resampled conditional H/A with the existing raw-feed
D probabilities, cross within each seed and then average. This adds zero fits
and 45 original-cohort cells. It isolates the team-attribution component from
D-distribution changes caused by sampling. Same guards; no other crosses,
weights or decoding settings. No new independent confirmation is claimed.

## Conditional game-level access experiment

Before any within-record scores, `feed_short_game_v1` declares six additional
fits only if that comparison and its fixed H/A cross both miss .55 under the
guards. The hypothesis is that GAR record fragmentation blocks access to useful
short examples: within-record replacement actually fires on only about 4% of
GAR draws, despite a 25% request rate.

Keep the original-provider windows exactly matched to the first study. For
GAR, use a separate RNG and select eligible windows across records from the
same already-selected game. Consume the prior within-record RNG identically so
old-provider sampling remains exact. Game/source balance, labels, native
eligibility, all training settings and decoder stay fixed. The TRAIN audit
finds eligible windows in all 45 games and raises GAR short-row exposure to
about 7.8%; it verifies identical original-provider counts and source/game draws.

The two fixed candidates are the complete game-sampled ensemble and its
conditional H/A crossed with raw-feed control D, within each seed before
averaging. This is a change in access to existing windows, not a new threshold
or rate sweep. There is no extra-corruption selection; all evidence remains
original exposed development. All models are sealed before scores, and the
same promoted-baseline guards and .55 stretch apply.

## Verified first-round finding

The within-record full model is nearly flat in feed spell (.525090 versus
.525240 control) and improves clean spell to .657930. The equal blend reaches
.524193 feed spell. Both pass guards but do not improve the primary metric.
The predeclared H/A cross reaches **.527396 feed / .657215 clean**, passes all
guards and improves short-spell recall from .199402 to .212521. Feed macro F1
falls from .826937 to .824460, a tradeoff retained in the report rather than
hidden by the spell gain. Metrica and Sportec feed spell improve; SkillCorner
falls slightly, .457600 to .456672. The whole-model and component comparisons
pass independent official-score, native-grading and probability-algebra replay.

The game-level continuation remains a separate, predeclared comparison. Its
condition is met because neither first-round arm reaches .55 under the guards.

The first-round gain belongs to the probability ensemble: the H/A cross's
individual feed scores are .514093 and .513095, below the augmented control's
.518567 and .516938 respectively. Its ensemble improves .525240 → .527396.
Thus the evidence supports this fixed combination, not a claim that each seed
became more robust. An average of individual scores would miss this behavior.

## Bounded calibration after model selection

Before game-level results, `feed_short_calibration_v1` freezes one additional
check if the final eligible model leader improves .525240 but still misses .55.
Lock that model using the original fixed-decoder scores before any grid scores.
The 27 fixed combinations are D temperatures {2,4,6}, conditional H/A
temperatures {.75,1,1.25}, and shared transition costs {2,3,4}. The inherited
setting T(D)=4/T(H/A)=1/cost=3 must replay its saved result exactly.

This calibrates the changed probability model; it does not rerun the old
522-configuration search on the old model. Evaluate one fixed ensemble only,
retain the original guards and spell-first ranking, and run the selected
setting on both individual seeds as a diagnostic without reselection. Stop
after this grid: no local refinement, new weights or decoder families. The
435 score cells comprise 405 grid cells and 30 seed diagnostics, all on the
same exposed development cohort. This is additional tuning, not confirmation.

## Results

| Fixed arm | Clean spell | Feed-like spell | Guards |
| --- | ---: | ---: | --- |
| Promoted baseline | 0.658625 | 0.518527 | Pass |
| Raw-feed augmented control | 0.654740 | 0.525240 | Pass |
| Short-window resampling | 0.657930 | 0.525090 | Pass |
| Equal resampling / control blend | 0.655531 | 0.524193 | Pass |
| Resampled H/A + control D | 0.657215 | 0.527396 | Pass |
| Within-game GAR resampling | 0.657623 | 0.517577 | Pass |
| Within-game H/A + control D | 0.656474 | 0.518174 | Pass |
| Selected model + calibration | 0.657215 | 0.527396 | Pass |

Best eligible original-benchmark arm: **Resampled H/A + control D, 0.527396**. The .55 stretch is **not reached**. No model promotion occurs.

All declared fits completed at 2,048 updates. The training receipts exactly match the pre-fit sampling audit; original source/match/game exposure counters remain identical to the augmented controls. The 45 augmented-control score cells and live inference reproduce exactly (maximum probability difference zero). All 539 tests pass. Independent official-score, aggregation, native-grading and ensemble/component checks pass.

## Exposure and interpretation

Short H/A row exposure rises from 3.7–7.8% to 8.3–12.6% for original-provider records, and from 1.8–1.9% to 2.4% for SoccerNet. About 8,200 original-provider draws and 1,200 GAR draws per fit are replaced out of 32,768 draws per source group. The remaining GAR replacement requests fall back because the selected record lacks an eligible window. Counts describe repeated TRAIN exposure; the sampler adds no new examples or independent labels.

| Fixed arm | Feed macro | Feed D-F1 | Feed short recall (<3s) | Feed predicted spells |
| --- | ---: | ---: | ---: | ---: |
| Promoted baseline | 0.823343 | 0.859048 | 0.198820 | 282.333 |
| Raw-feed augmented control | 0.826937 | 0.857715 | 0.199402 | 282.778 |
| Short-window resampling | 0.824739 | 0.859151 | 0.210253 | 292.833 |
| Equal resampling / control blend | 0.825744 | 0.858801 | 0.198790 | 284.778 |
| Resampled H/A + control D | 0.824460 | 0.858212 | 0.212521 | 295.389 |
| Within-game GAR resampling | 0.821552 | 0.860122 | 0.216312 | 299.889 |
| Within-game H/A + control D | 0.821908 | 0.859609 | 0.215572 | 301.944 |
| Selected model + calibration | 0.824460 | 0.858212 | 0.212521 | 295.389 |

Counts are equal-provider means. Short recall is one-to-one spell matching at IoU .5; training eligibility is complete H/A spans at most three seconds, so these are different denominators.

| Provider | Control feed spell | Resampled feed spell |
| --- | ---: | ---: |
| Metrica | 0.617628 | 0.614027 |
| SkillCorner | 0.457600 | 0.456921 |
| Sportec | 0.500494 | 0.504322 |

| Resampled seed | Clean spell | Feed spell |
| --- | ---: | ---: |
| 2026 | 0.653965 | 0.516011 |
| 2027 | 0.653575 | 0.514064 |

The historical +.01 clean screen results are {"fixed_half_blend": false, "short_sampling": false}. The fixed H/A cross historical screen is False. Individual seed scores are descriptive and are not averaged to obtain ensemble performance.

## Additional corruption diagnostic

No additional corruption diagnostic was run because neither parent arm provided an eligible improvement.

Evidence: [declaration](results/feed_short_sampling_v1_declaration.json), [TRAIN exposure](results/feed_short_sampling_v1_exposure.json), [control replay](results/feed_short_sampling_v1_preflight.json), [aggregate](results/feed_short_sampling_v1_aggregate.json), [independent verification](results/feed_short_sampling_v1_verification.json), [component declaration](results/feed_short_head_v1_declaration.json).
[Component result](results/feed_short_head_v1_aggregate.json) and [component verification](results/feed_short_head_v1_verification.json).

## Game-level results

The additional six fits completed. GAR short-frame exposure rises to about 7.8%, while original-provider sampling remains exactly matched to the within-record fits. The table above includes both fixed game-level arms. Their historical clean screens are {"game_ha": false, "game_sampling": false}.

[Game declaration](results/feed_short_game_v1_declaration.json), [TRAIN audit](results/feed_short_game_v1_exposure.json), [game aggregate](results/feed_short_game_v1_aggregate.json), [independent replay](results/feed_short_game_v1_verification.json).

## Selected-model calibration results

The model was locked as **targeted_ha** before evaluating the 27 fixed settings. Selected setting: `{"dead": 3, "dt": 4, "family": "calibration", "ha": 3, "ht": 1, "id": "c13"}`. This adds tuning on the same development cohort and is not independent confirmation. The inherited decoder reproduces the selected model exactly; all 435 score cells pass independent replay.

| ID | D temperature | H/A temperature | Cost | Clean spell | Feed spell | Guards |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c00 | 2 | 0.75 | 2 | 0.649575 | 0.515054 | False |
| c01 | 2 | 0.75 | 3 | 0.653356 | 0.519871 | False |
| c02 | 2 | 0.75 | 4 | 0.655240 | 0.521264 | False |
| c03 | 2 | 1 | 2 | 0.653067 | 0.518391 | False |
| c04 | 2 | 1 | 3 | 0.653368 | 0.521051 | False |
| c05 | 2 | 1 | 4 | 0.653782 | 0.520891 | False |
| c06 | 2 | 1.25 | 2 | 0.654855 | 0.520032 | True |
| c07 | 2 | 1.25 | 3 | 0.656589 | 0.521227 | True |
| c08 | 2 | 1.25 | 4 | 0.652830 | 0.515615 | False |
| c09 | 4 | 0.75 | 2 | 0.660969 | 0.519461 | False |
| c10 | 4 | 0.75 | 3 | 0.660540 | 0.525472 | True |
| c11 | 4 | 0.75 | 4 | 0.658239 | 0.525626 | False |
| c12 | 4 | 1 | 2 | 0.659479 | 0.524459 | True |
| c13 | 4 | 1 | 3 | 0.657215 | 0.527396 | True |
| c14 | 4 | 1 | 4 | 0.654481 | 0.525331 | True |
| c15 | 4 | 1.25 | 2 | 0.654123 | 0.523997 | True |
| c16 | 4 | 1.25 | 3 | 0.656006 | 0.524847 | True |
| c17 | 4 | 1.25 | 4 | 0.646378 | 0.520755 | False |
| c18 | 6 | 0.75 | 2 | 0.656244 | 0.520885 | False |
| c19 | 6 | 0.75 | 3 | 0.650937 | 0.520248 | False |
| c20 | 6 | 0.75 | 4 | 0.642911 | 0.517375 | False |
| c21 | 6 | 1 | 2 | 0.650306 | 0.518714 | False |
| c22 | 6 | 1 | 3 | 0.645429 | 0.516693 | False |
| c23 | 6 | 1 | 4 | 0.638892 | 0.512336 | False |
| c24 | 6 | 1.25 | 2 | 0.643656 | 0.514516 | False |
| c25 | 6 | 1.25 | 3 | 0.640279 | 0.512204 | False |
| c26 | 6 | 1.25 | 4 | 0.634107 | 0.504753 | False |

Selected-setting seed diagnostics: {"2026": {"clean": 0.6527463101099896, "feed_like": 0.5140926688656763}, "2027": {"clean": 0.6549431527860169, "feed_like": 0.5130946585787409}}. These diagnostics did not reselect the setting.

[Calibration declaration](results/feed_short_calibration_v1_declaration.json), [all grid results](results/feed_short_calibration_v1_readout_v2.json), [independent replay](results/feed_short_calibration_v1_verification.json).

## Preserved readout serialization failure

After all 435 calibration outputs were saved, the frozen writer failed on a NumPy boolean and left a partial JSON file. The partial file, failure log, lock and native outputs are preserved. A separately declared readout recovery normalizes scalar types before serialization and rebuilds metrics only from those saved outputs. It introduces no inference, arms, thresholds or new score population. The unchanged independent verifier reads the recovered v2 output through an explicit adapter; the originally selected setting is required to match.

[Recovery amendment](results/feed_short_calibration_v1_readout_recovery_v2.json). The original partial output remains in private storage at its original restore path.

## Decision and next experiment

The eligible and unconstrained grid winner is the inherited setting, so
calibration adds no gain. Retain the within-record H/A cross as the numerical
research leader: **.527396**, +.002155 over the raw-feed candidate and +.008868
over the adopted baseline. Clean spell is .657215. The .55 stretch remains
.022604 away; adoption is unchanged.

More GAR short-window exposure reduced feed performance even though the audit
confirmed the intended increase. That does not establish a universal benefit
from oversampling, or prove that the labels are wrong. The next useful work is
a TRAIN-only audit of short-span label fidelity and diversity, alongside new
independently labeled evaluation when available. Stop this declared sampling
comparison and calibration grid; no post-result rates or settings are added.
