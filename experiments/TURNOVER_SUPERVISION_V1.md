# Turnover supervision from the approved frontier

## Status

The 8 November 2026 comparison finds one eligible development candidate:
**75% hard-turnover / 25% fresh-continuation CE reaches .594002443007756 v2
spell F1**, versus the approved **.5936187198896373**. Clean/v1 spell and
short recall in all three views improve, and every declared retention check
passes. Independent mixture replay verifies all saved outputs.

None of the three training arms alone passes every check. Hard-turnover
sampling improves clean/v1 spell and retains all short-recall means but loses
.000153 v2 spell; ordinary continuation improves v2 but loses v1 short recall.
The separately declared blend combines these complementary results.
The adopted/deployed package remains unchanged: this study produces provider
holdout research candidates, not fitted pooled deployment weights.
The [frozen plan](TURNOVER_SUPERVISION_V1_PLAN.md) owns the original hypothesis,
budget and retention checks.

## Training results

All values below use the same 35 development cells and equal-provider protocol.
The [saved evaluation](results/turnover_supervision_eval.json) contains the
unrounded official scores, provider results and every guard.

| Arm | Clean spell | V1 spell | V2 spell | Failed retention checks |
| --- | ---: | ---: | ---: | --- |
| Approved frontier | .670137 | .532128 | .593619 | Reference |
| Fresh continuation | .670446 | .532146 | .594751 | V1 short recall |
| Uniform turnovers | .672104 | .533222 | .593464 | V2 spell and short recall |
| Hard turnovers | .672138 | .533505 | .593466 | V2 spell |

Relative to matched fresh continuation, hard sampling gains .001692 clean and
.001359 v1 spell but loses .001285 v2 spell. Uniform sampling gains .001658
clean and .001077 v1 but loses .001286 v2. Neither event sampler improves the
primary v2 metric over the matched training control.

| Arm | Clean macro | V1 macro | V2 macro | Clean short recall | V1 short recall | V2 short recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Approved frontier | .871864 | .826154 | .852204 | .408597 | .220367 | .288604 |
| Fresh continuation | .871950 | .826789 | .852458 | .408621 | .218434 | .288619 |
| Uniform turnovers | .872043 | .827124 | .851975 | .410313 | .223023 | .288084 |
| Hard turnovers | .872083 | .827220 | .852068 | .411030 | .221331 | .288924 |

All macro and provider clean-macro/D guards pass. The fixed D precision/recall
are .884844/.861920 clean, .843006/.881217 v1 and .873034/.862980 v2 for every
arm. The auxiliary .7 spell + .3 macro summaries, in clean/v1/v2 order, are
.730655/.620336/.671194 approved, .730897/.620539/.672063 fresh,
.732085/.621393/.671018 uniform and .732121/.621620/.671046 hard; these are
not the ranking criterion.

Hard sampling's v2 spell change varies by provider: Metrica -.001215,
SkillCorner -.000271 and Sportec +.001027 versus approved. Fresh continuation
gains .003893 in Metrica and .000437 in Sportec but loses .000933 in
SkillCorner. This does not establish a provider-wide improvement.

Separate CE-seed v2 spell scores span .591070–.591910 fresh,
.590610–.592857 uniform and .589819–.593356 hard. The two-seed ensembles
score higher than either individual-CE diagnostic in each arm, and the seed
spread exceeds the small candidate differences. These are sensitivity
diagnostics on the same exposed cohort, not confidence intervals.

Reference spell counts remain 404.333 per provider-weighted match in every
view. Predicted clean/v1/v2 counts are 346.333/299.500/329.333 approved,
346.111/298.167/328.417 fresh, 346.778/301.111/330.306 uniform and
347.222/300.667/330.417 hard. More predicted switches alone do not establish
better possession matching.

## What the diagnostic found

The [current-model audit](results/turnover_error_audit.json) examines direct
H-to-A and A-to-H transitions with at least one second of stable labels on
each side. It matches predicted switches in the same direction and continuous
time block within one second, one-to-one.

Across the five clean matches, 936 transitions qualify; corruption realizations
repeat those physical transitions, yielding 6,552 eligible cell-events across
35 cells. There are 4,008 misses across those views. Of the misses, 3,632
(90.6%) have mean correct-team conditional probability at or below .5 on at
least one side. The largest category is weak/reversed probability with no
predicted D within one second: 3,139 misses (78.3%). Predicted D appears at the
boundary for 538 misses and within one second for 781, including those 538.
These associations support testing H/A supervision; they do not show that
training can fix every miss.

| Provider | Clean eligible transitions | Clean recovered within 1 s | V2 eligible cell-events | V2 recovered within 1 s |
| --- | ---: | ---: | ---: | ---: |
| Metrica | 240 | 210 | 960 | 660 |
| SkillCorner | 410 | 94 | 1,640 | 342 |
| Sportec | 286 | 157 | 1,144 | 509 |

V2 counts sum four realizations of each match; they are not additional
independent events. Matching is maximum-cardinality chronological matching,
not minimum timing-error matching. The initial exploratory H-to-A-only audit
is preserved in [its scope-error receipt](results/turnover_error_audit_scope_error.json);
the corrected bidirectional audit supersedes it.

The [TRAIN inventory](results/turnover_supervision_audit.json) found 2,139
original-data transitions with stable one-second context, plus 5,083 in GAR
TRAIN. These are continuity counts from existing labels, not newly annotated
or independently verified possession boundaries. This experiment mines only
the original-data events; it leaves the GAR supervision recipe unchanged.

## Matched training and inference

Luna implementations compare ordinary continuation, uniform turnover sampling,
and hard-turnover sampling. Each receives 256 updates from the same
reconstructed stage-1024 CE checkpoint in three provider holdouts and two seeds,
for 18 fits. The source mix remains 16 original plus 16 GAR windows per batch;
this **50/50 mix is not a random train/test split**. Entire providers remain
excluded from every learned component used for their evaluation.

Both event arms replace 25% of original window draws using existing geometry
within the already selected record. Record, team orientation, alternate feature
choice, four-v2-view schedule, and every GAR draw remain paired. Separate
random streams govern override requests and event choices. Physical-event
bags are shared across paired views so those views do not masquerade as
additional turnovers.

Hardness is the smaller of mean correct-team H/A probability during one second
before and after the event. It is computed on clean TRAIN features by the
frozen fold/seed CE checkpoint, with D probability removed from that measurement.
The harder half is selected within each physical match. These are in-sample
training errors, not out-of-fold uncertainty estimates.

The model architecture, loss, normalization and D component remain unchanged.
Evaluation replaces only the CE half of the approved .5/.375/.125
CE/calibrated/predecessor ensemble. It retains seed-then-phase averaging,
H/A cost 3 and exact predecessor D/X masks. The five exposed development
matches supply clean, two v1 and four v2 realizations: 35 paired cells.
Aggregation averages realizations within match, matches within provider, and
providers equally.

The [training declaration](results/turnover_supervision_train_declaration.json)
binds the plan, 47 source files, original/GAR TRAIN inputs, teacher identities,
and six recovered stage-1024 checkpoints. The checkpoints are reconstructed,
not byte-identical to missing historical files. The baseline phase smoke
reproduces its cached probabilities exactly; the full 35-cell approved-score
replay gates candidate inference.

Separate CE-seed diagnostics retain the full two-seed calibrated/predecessor
experts. They measure sensitivity to the CE seed; their score average is not
the two-seed inference ensemble.

## Training exposure

All 18 fits finished. Independent checks verified all 18 exact source/match/view
and request schedules, all 18 frozen D/normalization states, and the six
fold/seed hardness catalogs. Each arm supplies 24,576 original-data draws
across its six fits; GAR draws remain identical.

| Arm | Turnover-containing original draws | Share | Known H/A sampled rows | Short-spell sampled rows |
| --- | ---: | ---: | ---: | ---: |
| Fresh continuation | 9,515 | 38.72% | 917,155 | 117,348 |
| Uniform turnovers | 13,298 | 54.11% | 965,536 | 130,250 |
| Hard turnovers | 13,298 | 54.11% | 969,820 | 128,800 |

Both event arms perform exactly 6,237 replacements on the same requested
draws. Uniform sampling also covers more distinct eligible events than control
in every fold/seed; hard sampling repeatedly emphasizes its selected half.
These counts describe the 256-update continuation. They do not establish that
the baseline had never seen the events during earlier training. No new labels
were added, and greater known-label exposure accompanies the changed event
distribution. Sampled short-spell rows use the existing TRAIN sampler's
at-most-three-second definition; evaluation recall uses its frozen
less-than-three-second definition.

A separate turnover diagnostic is frozen before candidate inference. It
replays the original 35 baseline audit rows and applies the same event definition
to the 105 ensemble candidate outputs. This checks whether supervision affects
the intended errors without adding a selection guard. Predicted edge counts
are descriptive, not precision: the eligible reference events form a subset
of all transitions.

The [completed turnover readout](results/turnover_supervision_turnover_readout.json)
exactly replays all 35 baseline audit rows and checks 105 candidate rows.
Recovered turnovers out of 6,552 repeated cell-events are **2,544 approved,
2,539 fresh, 2,548 uniform and 2,550 hard**. Hard sampling therefore recovers
six more than approved and eleven more than fresh, despite a much larger
increase in training exposure. Clean SkillCorner recovery stays at 94/410 in
every arm. Hard sampling's v2 recoveries change from 660 to 657 in Metrica,
stay at 342 in SkillCorner and rise from 509 to 513 in Sportec. The intended
mechanism moves only slightly and unevenly; more event exposure has not
resolved the main pool of weak/reversed H/A errors.

## Cached-mixture follow-up

After inspecting the training results, we separately declare
[three fixed mixtures](TURNOVER_SUPERVISION_MIX_V1_PLAN.md): 25%, 50% and 75%
hard-turnover CE, with the remaining CE weight from fresh continuation.
The approved .5/.375/.125 family weights and all decoder/retention settings
stay fixed. This is a post-result complementarity test, not new independent
evidence. It adds no fitting or inference, omits the weaker uniform arm, and
does not allocate a finer search, provider-specific weights or seed selection.

All three original training endpoints must replay before evaluating mixtures.
The [endpoint replay](results/turnover_supervision_mix_v1_endpoint_replay.json)
passes all 105 original-arm scorecards before mixture scoring.
The [saved mixture results](results/turnover_supervision_mix_v1.json) are:

| Hard CE fraction | Clean spell | V1 spell | V2 spell | Retention |
| --- | ---: | ---: | ---: | --- |
| .25 | .670851 | .531568 | .594679 | Fails v1 spell and short recall |
| .50 | .670678 | .533007 | .594342 | Fails v1 short recall |
| .75 | .670848 | .533247 | .594002 | Passes every check |

The .75 mixture improves spell by **.000712/.001119/.000384** and short
recall by **.000741/.000606/.000506**, in clean/v1/v2 order, versus approved.
Its macro F1 is .871997/.827122/.852111; v2 macro declines .000093,
within the frozen .001 allowance. Its auxiliary 70/30 summaries are
.731193/.621410/.671435. D metrics, D/X masks and grading/reference counts
remain unchanged. Predicted spell counts are 346.111/300.611/329.972.

The candidate's v2 provider spell deltas are +.000237 Metrica, -.000246
SkillCorner and +.001160 Sportec. Clean spell is unchanged in Metrica and
improves in the other two providers; v1 spell improves in all three.
The CE-seed v2 diagnostics are .593255 and .590136, both below the approved
two-seed ensemble. They are not comparable independent estimates of that
ensemble's uncertainty; they do show that its small gain depends on pooling.

Retain `.75` as an experimental development candidate. The
[independent mixture readout](results/turnover_supervision_mix_v1_readout.json)
verifies all 105 pooled and 210 CE-seed outputs, including reconstructed
probabilities, decoding, official scores, counts and guards.
The numerical v2 leaders `.25` and `.50` remain
rejected under their failed guards; this report does not relax those checks.
Stop this declared weight search. The gain is small on repeatedly exposed
data, and further fractions would provide little new evidence. The concrete
next validation is to construct the exact pooled counterpart and evaluate it
without retuning on the separate GAR protocol, followed by independently
annotated continuous-match gold data. Neither step is performed by this study,
and there is no automatic promotion or organizer rescore.

## Evidence limits

The same five development matches have been examined repeatedly. Feed-like v2
is model-conditioned synthetic corruption. GAR is silver supervision on
discontinuous segments, and its TEST/VALID labels are not used here. Original
TRAIN native timestamp arrays are unavailable, so native label context relies
on prepared period/record boundaries; the shared feature pipeline preserves
sampled time breaks. Fixed D masks constrain the improvements possible from
H/A training. No organizer labels are read or scored, no gate is reopened,
and no pooled deployment candidate is produced by this comparison.

## Validation and delivery

All 32 focused turnover and mixture tests pass. The final required experiment
suite runs 925 tests: 919 pass, one skips, and five inherited public-contract tests error
because the historical `v2_spell_balanced_v1/public_prediction_seal.json`
and associated candidate artifact are absent. The frozen tests are preserved;
no organizer scoring is used to recreate their fixture. The 26 root tests,
file-backed fit contract and existing 199,319-row submission format check pass.

The [independent readout](results/turnover_supervision_readout.json) verifies
all 35 baseline, 105 ensemble candidate and 210 CE-seed diagnostic scorecards.
It reconstructs raw phase/seed pooling, family mixtures and decoding, checks
exact D/X masks and grading/reference counts, and reproduces aggregate scores,
guards and training exposure receipts. The turnover-specific readout separately
verifies its 35 baseline and 105 candidate audit rows.

The [draft PR #90](https://github.com/jekkiechan/more-than-a-hack-2027/pull/90)
targets the fork's `main` and includes the prior recovery/postprocessing work
from PR #88. It stays draft because of the inherited missing-fixture test
errors. Documentation links and frozen-source preservation pass.

Private artifact availability and research-site publication are recorded in
the delivery receipts added after this source/result snapshot. Weights,
event-level details and per-frame outputs remain private; the public report
contains aggregate evidence only.

The [final private delivery receipt](results/turnover_supervision_private_delivery.json)
records 159 verified members (18 checkpoints, both rounds' 70 cell caches,
catalogs, receipts and logs) in a 2,506,037,599-byte archive. Delivery uses two
byte-preserving Git LFS parts because the whole archive exceeds GitHub's
per-object limit. The authorized private branch is
`jekkiechan/mtah-data:research/turnover-supervision-v1-parts` at `f1de198`.
A fresh shallow clone with isolated LFS storage verifies each downloaded part,
the complete concatenated archive hash, and every member's path, size and
SHA-256 without extraction. This supersedes the preserved initial local-only
receipt. The existing frontier-recovery archive remains a separate dependency.

The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#turnover-supervision)
is published as version 37 from source `26d6f0d`; see the
[publication receipt](results/turnover_supervision_site_publication.json).
All nine metric/view combinations pass desktop/mobile interaction checks,
and all 63 chart values exactly match the published aggregate evidence.
The custom audience, access revision 2 and one external visitor are preserved.
