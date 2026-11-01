# Decoder refinement and public diagnostic

Status: completed, 2026-11-01. Base fork main `866d86a`.
The user requested iteration on the [134-arm study](DECODER_SEARCH_V2.md),
explicitly welcomed non-Viterbi methods, and requested a public-score diagnostic.

## Frozen comparison

One question: can a focused refinement improve official possession spell F1
over the previous H/A-cost-3, team/dead-cost-12 development winner?
Fix every model, input array, corruption realization, grading mask and metric.
No fitting, new labels or organizer-driven selection. The declaration predates
candidate scoring and fixes 118 configurations:

| Family | Arms | Intervention |
| --- | ---: | --- |
| Transition matrix | 30 | H/A costs 2/2.5/3/3.5/4; team/dead costs 6/9/12/18/24/36 |
| Dead duration | 12 | Exact segment decoding, min D .25/.5/1/2 s, dead costs 6/12/24; H/A min one frame |
| Selective D closing | 24 | Fill short D islands only between matching predicted teams, with mean D confidence < .6/.8 |
| D temperature | 12 | Recalibrate D logits with temperatures .5/.75/1.25/1.5, preserving conditional H/A |
| D-only filtering | 12 | Smooth D probabilities with .15/.3/.6/1.2 s Gaussian kernels, preserving conditional H/A |
| Confident-anchor segments | 18 | No Viterbi: require sustained high-confidence anchors, grow same-state intervals, place one evidence-maximizing boundary between different anchors |
| Total variation | 10 | No Viterbi: 100 projected-dual iterations of continuous temporal denoising, componentwise or joint edge norm, then direct argmax |

Separate constant D-entry/D-exit penalties were deliberately omitted: away from
recording endpoints their sum controls complete D runs, so this would mostly be
a redundant parameterization. The new duration, calibration and selective
closing mechanisms change the actual inference behavior.

The 28 non-Viterbi arms never call the Viterbi implementation, verified by a test
that replaces it with a failing stub. Other tests cover team swaps, unavailable
classes, period isolation, input immutability, exact prior-winner replay, and an
exclusive public gate that stays closed after a reference-hash failure.

## Protocol and selection

Use the same hash-bound reconstructed `control_D2` probabilities: five exposed
matches (Metrica 3, Sportec J03WOY, SkillCorner 1996435/1996436/2006229), clean and
two frozen synthetic feed-like views, two component seeds and their native
probability ensemble. The source excludes each provider from both heads and
normalization. Original fold checkpoint byte identity is not claimed; matching
reconstructed-control provenance remains explicit. All development matches have
repeated prior exposure; some supervision is silver and stress is synthetic.

Verify all 45 exact previous-winner paths and saved metric cells before scoring.
Complete 1,770 ensemble cells and 90 component-seed cells for three finalists.
Rank by clean equal-provider official spell F1, macro for exact ties. Average
replicates within match, matches within provider, then providers equally. Report
the fixed 70/30 auxiliary score without ranking on it. Require aggregate feed
spell >= adopted-.01, each clean provider macro/D-F1 >= adopted-.02, and now
each provider feed spell >= adopted-.01. This added provider stress guard is
frozen before this study; it does not rewrite the previous study's guards.
The +.01 clean screen stays explicit. Seed checks are descriptive and do not
reselect the finalist. Also report selection on two providers and application to
the omitted third, still as exposed development evidence.

## Public diagnostic requested by the user

Only after external selection finishes, generate two fixed full CSVs: the
previous H/A3/dead12 winner and the selected refinement. Deduplicate identical
CSV hashes. Verify the current cost-3 CSV byte-for-byte against its sealed
adopted-deployment hash, and the old split-cost CSV against the earlier candidate.
Validate all 199,319 rows, then seal source/model/input/prediction and development
selection hashes before reference access.

Claim a new exclusive gate, then read the public reference once and run the
unchanged official evaluator with 1,000 bootstrap draws. Reuse the saved adopted
baseline score, never rescore it. Expected population: 7,500 graded frames and
17 reference possessions. Every older gate stays closed. A failed scoring call
also leaves this gate permanently closed. No fitting, threshold selection,
repair, seed choice or promotion follows the public result. Full-match organizer
performance remains unknown.

## Reproduction and evidence paths

```sh
.venv/bin/python -m unittest experiments.tests.test_decoder_refinement -v
.venv/bin/python -m experiments.decoder_refinement_run freeze
.venv/bin/python -m experiments.decoder_refinement_run run
.venv/bin/python -m experiments.decoder_refinement_public prepare
.venv/bin/python -m experiments.decoder_refinement_public score
```

All stages use exclusive creation and must not overwrite this completed study.
Inference takes probabilities, period/time-break identifiers and fps, never
truth labels, truth-derived masks or provider identifiers. Detailed frame
outputs remain under ignored `data/experiments/decoder_refinement_v3`.

## External results and decision

All 118 configurations completed 1,770 ensemble cells; three finalists completed 90 component-seed cells plus complete-wrapper ensemble replays. All 45 previous-winner paths and metric cells reproduced, and an independent scalar hierarchy/rank replay verified every arm. No public labels entered selection.

**Best selected refinement: dead-probability temperature 1.5, H/A cost 3, team/dead cost 6 (`r069`).** The transformation divides D log odds by 1.5, moves D probabilities toward .5, and retains the conditional H/A probabilities. It improves both spell and macro F1 versus the adopted baseline and the previous split-cost winner. This is an empirical decoder/calibration gain, not proof of calibrated probability estimates.

| Decoder | Clean spell | Feed-like spell | Clean macro | Feed-like macro |
| --- | ---: | ---: | ---: | ---: |
| Adopted cost 3 | 0.647215 | 0.505811 | 0.857677 | 0.815535 |
| Previous H/A3, dead12 | 0.651341 | 0.508182 | 0.856791 | 0.814835 |
| Selected D-temperature 1.5 | 0.652958 | 0.511065 | 0.860730 | 0.820064 |
| Best non-Viterbi anchors | 0.649962 | 0.508516 | 0.859075 | 0.817892 |

The winner gains +0.005743/+0.005254 spell versus adopted, and +0.001616/+0.002883 versus the previous winner. The +.01 clean screen remains failed. All declared aggregate and provider guards pass. No adoption is performed.

| Provider | Clean spell delta vs adopted | Feed-like spell delta | Clean macro delta | Clean D-F1 delta |
| --- | ---: | ---: | ---: | ---: |
| Metrica | +0.011220 | +0.003702 | +0.003071 | +0.004921 |
| Sportec | +0.004337 | +0.005176 | +0.004039 | +0.005664 |
| SkillCorner | +0.001673 | +0.006883 | +0.002049 | +0.004369 |

Each provider improves clean and stressed spell F1, clean macro, and clean D-F1. The new provider stress guard is satisfied without an exception.

| Seed | Adopted clean | Selected clean | Adopted feed-like | Selected feed-like |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 0.635521 | 0.641160 | 0.501847 | 0.503769 |
| 2027 | 0.637245 | 0.642854 | 0.495759 | 0.502487 |
| ensemble | 0.647215 | 0.652958 | 0.505811 | 0.511065 |

Both component seeds improve in both views. These checks use the same exposed matches and are not independent confirmation.

The provider-selection check is now positive: **0.648840 clean / 0.506357 feed-like**, +0.001625/+0.000546 versus adopted. Metrica is omitted while selecting `r001` on the other providers; SkillCorner uses `r069`; Sportec uses `r073`. The rule uses only selection-provider scores/guards. This is more encouraging than the previous negative check, but neither its magnitude nor the reused cohort establishes transfer to genuinely new matches.

### Non-Viterbi and other alternatives

| Family | Its best clean spell | Same configuration feed-like | Guards pass |
| --- | ---: | ---: | --- |
| dead_temperature | 0.652958 | 0.511065 | True |
| matrix | 0.652551 | 0.508352 | True |
| dead_filter | 0.651643 | 0.508916 | True |
| dead_closing | 0.651341 | 0.508182 | True |
| dead_duration | 0.651341 | 0.508182 | True |
| anchor_segments | 0.649962 | 0.508516 | True |
| total_variation | 0.628050 | 0.480540 | False |

**The anchor segmenter provides a useful non-Viterbi alternative:** confidence .55 sustained for .3 seconds, no probability filtering, continuous interval growth and one soft-evidence boundary between opposing anchors. It scores .649962/.508516 and passes all guards. It beats the adopted decoder by +.002748/+.002705 while trailing the selected calibrated decoder. It was not selected for the public diagnostic; no public score is claimed for it.

Total-variation denoising loses substantially. Its fixed 100 iterations approximate a continuous denoising objective; this is not an exhaustive rejection of every TV solver or regularization scheme. Dead-duration and selective-D-closing winners reproduce the previous split-cost result, adding no improvement.

### Diagnostics and tests

clean: D precision/recall/F1 0.917604/0.806578/0.855161; predicted spell count 360.667 → 352.444; short-spell (<3s, IoU .5) recall 0.402475 → 0.394295. The counts/recalls are equal-provider match means.

feed_like: D precision/recall/F1 0.906347/0.815106/0.855691; predicted spell count 318.444 → 308.889; short-spell (<3s, IoU .5) recall 0.217097 → 0.208906. The counts/recalls are equal-provider match means.

All **505 experiment tests**, 26 integrity/gate tests and the synthetic file-backed fit contract pass. Both frozen public CSVs contain 199,319 valid rows. The baseline and previous-candidate full CSVs replay byte-for-byte; exact hashes, the full source/configuration/model/input seal and the exclusive new public gate are retained privately.

[Declaration](results/decoder_refinement_v3_declaration.json), [control replay](results/decoder_refinement_v3_control_replay.json), [all-arm aggregate](results/decoder_refinement_v3_aggregate.json), [independent scalar readout](results/decoder_refinement_v3_readout.json).

## One-time public diagnostic: spell gain does not transfer

**Both frozen candidates score 0.575000 public spell F1, below the adopted
baseline's saved 0.630303 (−0.055303).** The selected refinement raises macro
F1 by 0.012029 and D-F1 by 0.021422, but spell F1 is the headline metric.
This outcome must accompany the positive development comparison.

| Fixed prediction | Public spell F1 | Public macro F1 | D-F1 | Predicted possessions |
| --- | ---: | ---: | ---: | ---: |
| Adopted baseline (saved score) | 0.630303 | 0.851387 | 0.843424 | 16 |
| Original H/A3, dead12 | 0.575000 | 0.862259 | 0.862327 | 15 |
| Selected refinement | 0.575000 | 0.863416 | 0.864846 | 15 |

The two candidates have different sealed CSV hashes and different frame/macro
scores; their spell scores happen to tie at every IoU threshold. They predict
15 possessions against 17 references, versus the baseline's 16 predictions.
The candidates match 11 references at IoU .3/.4/.5, 7 at .6 and 6 at .7.
These aggregate counts do not identify a single cause or authorize any
reference-based boundary repair.

Exactly two unique sealed CSVs were scored after the exclusive gate was claimed.
The reference hash and 7,500-frame/17-possession grading population match the
saved protocol. The baseline score was read from its saved result and not
recomputed. All earlier gates remain closed; the new gate is permanently closed.
No training, seed/parameter selection, repair or adoption followed public access.
[Public result](results/decoder_refinement_v3_public_check.json).

The official 5th–95th percentile spell intervals for both new predictions are
[0.40, 0.75], conditional on this single small public window. Separate intervals
are not paired significance and do not estimate full-match performance. The
non-Viterbi anchor arm was not part of this fixed diagnostic and has no new
public score.

Decision: retain the adopted package; preserve the calibrated and anchor
methods as experimental development findings. The +.01 external screen already
failed before the public diagnostic. No public-driven reselection follows the
negative result. Any stronger next efficacy claim needs independent dense
possession evidence, not additional tuning on this organizer window.
