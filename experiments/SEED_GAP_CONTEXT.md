# Two-seed GRU ensemble and long-gap context

Status: completed. Date: 2026-10-28. Branch:
`possession/seed-ensemble-gap-context`, based on `7615dd5`.

This study tests the adopted slot-dropout joint GRU on the same five previously
exposed external development matches. All learned H/A, D and normalization
components exclude the evaluation provider. It does not establish independent
broadcast or full organizer-match accuracy.

## Ensemble result and current gap diagnosis

The actual probability ensemble improves both headline views. Its extra gain
over seed 2027 is modest: +.0089 clean and +.0021 feed-like spell F1. The adopted package stays unchanged; the ensemble is a modest external
development improvement, while longer context is not a consistent replacement.

![Matched clean and feed-like spell scores](results/seed_gap_context_scores.png)

| Recipe / view | Spell F1 | Macro F1 | Auxiliary 70/30 | D precision | D recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 2026 / clean | .608763 | .843432 | .679164 | .894689 | .786376 |
| Seed 2027 / clean | .617600 | .854336 | .688621 | .886197 | .827713 |
| Equal probability ensemble / clean | .626531 | .854559 | .694940 | .897016 | .814340 |
| Seed 2026 / feed-like | .474237 | .801255 | .572343 | .890351 | .785583 |
| Seed 2027 / feed-like | .485302 | .810874 | .582973 | .882069 | .823803 |
| Equal probability ensemble / feed-like | .487419 | .811127 | .584532 | .890409 | .810993 |

All 30 individual-seed parity cells reproduce the saved results. The ensemble
is 15 actual new decodes, not the mean of the two score rows. The ensemble
beats both seeds on Sportec and at the provider-averaged SkillCorner level,
but seed 2027 remains stronger on Metrica in both views. On feed-like live
frames, only seed 2026 is correct on 3.81%, only seed 2027 on 2.94%, and both
are wrong on 16.30%, using the same provider-balanced weighting. Complementary
errors exist, but most residual wrong frames are shared.

| Provider / view | Seed 2026 spell | Seed 2027 spell | Ensemble spell |
| --- | ---: | ---: | ---: |
| Sportec / clean | .61915 | .60335 | .64091 |
| Metrica / clean | .74785 | .78596 | .77220 |
| SkillCorner / clean | .45929 | .46350 | .46649 |
| Sportec / feed-like | .44286 | .43344 | .46263 |
| Metrica / feed-like | .55506 | .59647 | .57112 |
| SkillCorner / feed-like | .42479 | .42600 | .42850 |

The adopted seed's feed-like long-gap wrong-team rate is **24.32%**, versus
**5.10% false D**, compared with **9.28% wrong team on observed-ball rows**.
This satisfies the frozen condition for the longer-context fits. It supports
the H/A diagnosis on the synthetic feed-like views, with a material false-D
component still present.
Including reference-dead errors, the provider-balanced long-gap error partition
is 57.92% wrong team, 12.21% false D and 29.88% missed D. The diagnosis is a
majority H/A problem in this view, not an absence of dead-ball errors.

Native clean data does **not** support a broad long-live-gap claim: across all
five matches there are only **108 graded live frames** inside gaps >=10 s,
all from SkillCorner under the unchanged >=12-visible-player grading filter.
Their errors are 7 wrong-team and 81 false-D frames;
20 frames have the correct team. Sportec has no such gap rows; Metrica's long
native gaps cover dead play. This is a missing evaluation population, not
evidence that the baseline's real organizer-gap bottleneck is established.

On feed-like views, team accuracy declines from 87.31% within one second of a
reliable contact proxy to 69.17% at 5–<10 s and 53.31% at >=20 s. Short-spell
miss rates are about 92% in 10–<40 s gaps versus 73% on no-gap rows; direct
turnover recall within one second is about 4–6% versus 44%. Fragmentation is
about 2.6–2.9% in those gap bins versus 2.7% on no-gap rows. The diagnostic
points to misses and timing rather than a large rise in fragmentation.
SkillCorner has 784/1,125 reference spells touching graded-run edges (69.69%),
versus 1/456 Sportec and 4/382 Metrica. Short-spell counts therefore include
many censored observations; keep the official population, but avoid treating
all of these snippets as complete real possessions.

![Wrong-team and false-D errors by gap duration and contact distance](results/seed_gap_context_errors.png)

## Completed longer-context result

Decision: retain the adopted 5s-context package. The actual seed ensemble is a
candidate for confirmation; do not promote the 15s-context recipe. All six fits
and all 30 context cells completed, with exact control normalization, feature
schema, parameter, window and optimizer-update parity. No controls were refit.

| Context recipe / view | Spell F1 | Macro F1 | Auxiliary 70/30 | D precision | D recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| context15_2026 / clean | 0.605830 | 0.829878 | 0.673045 | 0.916399 | 0.731580 |
| context15_2026 / feed_like | 0.473933 | 0.789466 | 0.568593 | 0.913189 | 0.729990 |
| context15_2027 / clean | 0.615305 | 0.854462 | 0.687052 | 0.898431 | 0.815616 |
| context15_2027 / feed_like | 0.489945 | 0.812794 | 0.586800 | 0.892913 | 0.811328 |

| Seed | Clean spell delta | Feed-like spell delta | Worst provider clean D/macro delta | Guardrails |
| --- | ---: | ---: | ---: | --- |
| 2026 | -0.002932 | -0.000304 | -0.057947 | fail |
| 2027 | -0.002296 | +0.004643 | -0.008303 | pass |

Both seeds lose a little clean spell F1. Feed-like spell is effectively flat
for seed 2026 and rises .00464 for seed 2027. Provider effects differ: seed 2027
gains .02239 clean/.02550 feed-like on Sportec but loses .02733/.01240 on
Metrica. Seed 2026 loses .01042/.01090 on SkillCorner. Its clean SkillCorner
D F1 falls .71639→.65844 and D recall .62359→.52309; macro falls .02399.
That violates the .02 clean provider guardrail. Seed 2027 passes the guards
but does not produce a clean spell gain. Per-provider P/R and all cell counts
are retained in [context results](results/seed_gap_context_long.json).

### Does the additional window help gap recovery?

Partially. In feed-like live gaps >=10s, nominal windows containing both
surrounding contact proxies increase from **7.02% to 47.20%** of frames, with
the same replicate/match/provider weighting. All-long-gap team accuracy rises
in both seeds, and gains are larger on the pre-existing subset newly bracketed
by the longer window. The partition is descriptive and was summarized after
the frozen experiment; it is not a randomized intervention on contacts.

| Seed / long-gap subset | 5s team accuracy | 15s team accuracy | Delta (pp) | 5s→15s wrong-team rate | 5s→15s false-D rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026 / all_long | 70.59% | 72.43% | +1.85 | 24.32%→23.73% | 5.10%→3.84% |
| 2026 / newly_bracketed | 70.88% | 73.90% | +3.02 | 24.00%→22.54% | 5.12%→3.56% |
| 2026 / still_outside_window | 69.68% | 70.37% | +0.68 | 25.27%→25.52% | 5.05%→4.11% |
| 2027 / all_long | 70.23% | 71.59% | +1.36 | 23.54%→23.00% | 6.23%→5.41% |
| 2027 / newly_bracketed | 69.77% | 72.75% | +2.97 | 24.25%→22.28% | 5.98%→4.97% |
| 2027 / still_outside_window | 69.80% | 70.02% | +0.22 | 23.81%→24.45% | 6.39%→5.53% |

The newly bracketed subset has 66,169 repeated frame exposures across the two
feed-like realizations, versus 163,799 all-long-gap exposures; these are not
independent samples. All three providers contribute. Wrong-team and false-D
rates both improve there. Some improvements also occur without a bracketing
pair, so the entire gain cannot be attributed to seeing contacts. This is
consistent with the proposed local recovery mechanism, without establishing
a net possession-spell benefit or a causal contact-only effect.

Short possessions remain difficult: in 10–<20s gaps, the missed-short-spell
rate changes 92.13%→93.11% for seed 2026 and stays 94.52% for seed 2027.
At 20–<40s it changes 92.09%→91.89% and 92.02%→90.65%. Fragmentation stays
low. One-second turnover recall improves in 10–<20s gaps (5.56%→9.16% and
5.57%→8.63%), with mixed effects in longer gaps. These modest timing/frame
improvements do not consistently recover short spells at the official IoUs.
Full gap/contact/joint strata, support counts, misses, fragmentation and
timing with unmatched events are in the [readout](results/seed_gap_context_readout.json).

### All official matching thresholds

Replicates are averaged within match, then matches within provider and
providers equally. These are F1 values, not a different matching objective.

| Recipe / view | IoU .3 | .4 | .5 | .6 | .7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed_2026 / clean | 0.70827 | 0.67405 | 0.61886 | 0.55928 | 0.48336 |
| seed_2026 / feed_like | 0.59206 | 0.54134 | 0.47760 | 0.41294 | 0.34724 |
| seed_2027 / clean | 0.71117 | 0.67693 | 0.62667 | 0.57319 | 0.50004 |
| seed_2027 / feed_like | 0.59688 | 0.55038 | 0.48738 | 0.43018 | 0.36169 |
| equal_probability_ensemble / clean | 0.71444 | 0.68453 | 0.63865 | 0.58325 | 0.51179 |
| equal_probability_ensemble / feed_like | 0.59804 | 0.55880 | 0.49033 | 0.42937 | 0.36056 |
| context15_2026 / clean | 0.70449 | 0.66907 | 0.61512 | 0.55911 | 0.48136 |
| context15_2026 / feed_like | 0.58691 | 0.54133 | 0.47260 | 0.41501 | 0.35382 |
| context15_2027 / clean | 0.71621 | 0.67676 | 0.61943 | 0.56499 | 0.49914 |
| context15_2027 / feed_like | 0.59943 | 0.54473 | 0.49607 | 0.43397 | 0.37552 |

The cohort has 1,963 reference spells across the five native matches. Reference
spells and grading counts stay unchanged in every matched comparison. The
experiment contains 45 original/ensemble cells plus 30 context cells; the
independent verifier checks 15 bit-exact full-vector ensembles, all 30 original
probability hashes and replays 45 ensemble/context score cells through the
official scorer. The other 30 individual cells reproduce the archived scores.

Next: obtain independent labeled broadcast evidence containing genuine long
live-ball gaps. The present result supports a local context mechanism under
synthetic missingness, but does not justify a larger sweep or deployment change
from these exposed development matches alone.

## Frozen comparison

[Declaration](results/seed_gap_context_declaration.json). All 253 files in the
archived `gru_slot_robustness_cache_manifest.json` were restored and verified
against the existing size/SHA-256 receipt before evaluation. No control was
retrained. Each of three held-out providers has the original seeds 2026/2027.

For each match/view, average the two **native H/A/D probability vectors equally**,
then run the unchanged Viterbi decoder at switch cost 3. Compare this inference
ensemble with each individual seed. Clean is native tracking; feed-like uses
the existing corruption realizations `corrupt_1`/`corrupt_2`, preserving ball
gaps, coordinates, player-slot gaps and ID swaps. Every individual clean and
feed-like score must reproduce the saved stress-component ablation. Aggregate
replicates within match, matches within provider, then providers equally.

Spell F1 averaged across the five official IoUs remains primary; macro F1 breaks
exact ties. The 70/30 auxiliary is reported without ranking on it. Scores,
per-provider D precision/recall, all IoUs, spell counts and diagnostics are saved
in [aggregate results](results/seed_gap_context.json).

## Gap and contact definitions

Missing-ball intervals respect native observation masks and input half/time
breaks. Duration bins are <3, 3–<10, 10–<20, 20–<40 and >=40 seconds; observed
frames are a separate no-gap group. A gap's length includes unknown-label rows;
only reference-graded rows contribute accuracy or spell outcomes.

The nearest observed contact is a **tracking proxy**, reusing the existing
`possession_geometry` criteria: best control >=.2, absolute team margin >=.2,
stable for .2 seconds, with native player/ball masks. The diagnostic measures
distance in seconds within the same input block, without the feature's 10-second
age cap. Bins are <1, 1–<3, 3–<5, 5–<10, 10–<20 and >=20 seconds; absence of any
contact in the block is separate. This does not infer new possession labels.

On reference-live rows, correct team, wrong team, false D and X are disjoint.
Team accuracy counts false D as an error. Reference-D misses are reported
separately. Joint gap/contact bins accompany the marginal bins.

Spell misses retain the official global one-to-one IoU .5 assignment.
Short means <3 seconds; fragmentation means at least two same-team predicted
spells overlap a reference spell by any positive duration. Spells are located
by their midpoint; a second view groups them by maximum intersected gap.
Graded-run edge censoring is reported. Turnover timing uses direct H/A changes
within graded runs, greedily matched one-to-one to same-direction predictions
within 5 seconds. Misses and matches within 1 second accompany signed/absolute
timing; timing of matched events alone must not hide missed turnovers.

## Conditional longer-context experiment

The frozen trigger checks adopted seed 2026 on feed-like tracking: within live
gaps >=10 seconds, provider-balanced wrong-team rate must exceed false-D rate
and wrong-team rate on no-gap rows. This is a descriptive development condition.

If it holds, fit exactly six models: two seeds x three full provider holdouts,
using **15 seconds of context on each side of the same 10-second output cores**.
The 5-second-context controls are restored. Features, hidden24 capacity, masks,
normalization population, whole-window augmentation choices, source/match
weights, eight epochs, batch64, AdamW settings, update/window count and decoder
stay fixed. Exact normalization and budget parity are checked against each
original checkpoint. Seal all six models before any longer-context scoring.

The shared training/inference libraries are reused verbatim through a serial,
process-local context adapter that restores the original constant on exit.
Window counts and crop alignment have synthetic contracts. No old runner,
baseline manifest, policy, historical declaration or public gate is modified.

Runtime scheduling/cache adapters were added during fitting, before any context
scoring; [runtime note](results/seed_gap_context_runtime.json) records them.
The two initial fits use the in-memory runner. Later fits call that same
mathematical routine with byte-verified file-backed raw arrays, allowing the OS
to evict unused raw pages after normalization. A synthetic contract reproduces
every fitted state value and training-history entry exactly. This changes cache
storage and scheduling; all declared numerical mechanisms remain fixed.

For each seed the clean and feed-like spell deltas must stay >=-.01 and every
provider's clean macro/D-F1 delta >=-.02 versus its own original control. The
mechanistic hypothesis requires positive spell and long-gap team-accuracy deltas
in both seeds. Seeing both contacts inside the nominal input window is measured
at 5 and 15 seconds, independently of target labels.
These are nominal sampled-window bounds: the unchanged centered feature
aggregates also have their existing short temporal halos. A bracketing count
describes contact availability, not proof that the recurrent network used it.

## Reproduction and artifact handling

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest gru_slot_robustness_cache_manifest.json --data-dir data
python -m unittest discover -s experiments/tests -v
python -m experiments.seed_gap_context freeze
python -m experiments.seed_gap_context evaluate
python -m experiments.seed_gap_context report
# Only if the frozen diagnosis holds:
python -m experiments.seed_gap_context prepare
python scripts/fit_context_filebacked.py --contract
python scripts/fit_context_filebacked.py --source sportec --seed 2026
# Repeat the remaining five declared source/seed pairs, then:
python -m experiments.seed_gap_context seal
python -m experiments.seed_gap_context evaluate-context
python -m experiments.seed_gap_context context-report
```

Use the pinned CPU research dependencies. Training caches and per-frame arrays
stay in ignored `data/experiments/seed_gap_context_v1/` and the private cache.
No organizer labels are read and no full-length organizer CSV is generated by
this external study. The adopted baseline remains `joint_gru_slotdrop_v1`.

## Verification and archived state

- All 154 repository synthetic/contract tests passed, plus exact plain/file-backed
  fitted-state and training-history parity. CI runs both contracts.
- [Independent replay receipt](results/seed_gap_context_verification.json): 15
  exact probability ensembles, 30 individual probability hashes, and 45 official
  ensemble/context score replays passed. The frozen sources and inputs still match.
- The bundled 199,319-row CSV format check passed. This external study generated
  no organizer CSV and read no organizer ground truth; every old gate stays closed.
- [Initial ensemble archive](results/seed_gap_context_ensemble_artifacts.json)
  was independently downloaded and verified. The [complete artifact receipt](results/seed_gap_context_artifacts.json)
  records the final six checkpoints, outputs, evaluation caches and source snapshot.
  Regenerable training caches and temporary file-backed raw arrays are omitted.
- Restore the original `gru_slot_robustness_cache_manifest.json` dependency and
  then `seed_gap_context_cache_manifest.json` from private branch
  `experiments/seed-gap-context-v1`. Keep all arrays/checkpoints private.

Standalone PNG/SVG figures are saved beside the JSON results. To regenerate
post-hoc tables, plots and the independent check after restoring the cache:

```sh
python scripts/summarize_seed_gap_context.py
python scripts/plot_seed_gap_context.py
python scripts/verify_seed_gap_context.py
```

The report/readout/verifier use exclusive result creation. Read existing frozen
outputs rather than overwriting them; reproduce in a separate checkout/output
copy when needed. No adoption or full-match accuracy claim follows this study.
