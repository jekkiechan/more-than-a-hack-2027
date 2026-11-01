# Short-possession label corroboration and exposure diversity

Study `short_quality_diversity_v1`, 1 November 2026. Twelve fits are running;
candidate results are not yet available. The adopted model is unchanged.

## Question and matched controls

Can stronger corroboration of short GAR silver labels and more even exposure
to distinct possessions improve original feed-like spell F1 beyond **.527396**?
The matched numerical leader combines within-record short-sampled H/A with the
raw-feed model's D probabilities. Its clean spell F1 is .657215. The raw-feed
control reaches .654740/.525240 clean/feed-like; the adopted calibrated model
reaches .658625/.518527. The stretch target remains .55.

The preceding [within-record and within-game sampling study](FEED_SHORT_SAMPLING_V1.md)
increased exposure without adding examples. Broader game-level GAR access
reduced performance. This study tests a specific label-consistency and exposure
mechanism, without repeating its rate, duration or decoder searches.

## TRAIN-only label audit

The restored dense archive contains 11,249 records from 45 GAR TRAIN games.
There are 1,427 complete native H/A possessions lasting at most three seconds.
Every span has an overlapping same-team controlled, uncensored touch flag;
that existing flag alone does not distinguish reliable short labels.

| Diagnostic | Complete short spans |
| --- | ---: |
| Total | 1,427 |
| No sampled observed-ball evidence | 583 |
| Conflicting sparse H/A anchor inside span | 72 |
| No sampled row at all | 10 |
| Held-uncontrolled provenance somewhere inside | 110 |
| Meets the conservative corroboration rule | 403 |

These diagnostic groups overlap. Sparse anchors, native event-control flags
and geometry are imperfect silver-label proxies. An anchor conflict is a
consistency warning, not proof of incorrect possession. Retaining the last
controller through contested touches remains the label rule. No labels are
rewritten, and no native unknowns or grading masks are filled.

A span qualifies for additional sampling when it is complete at native rate,
lasts at most three seconds, has an overlapping same-team uncensored controlled
touch, contains no conflicting sparse H/A anchor, and has at least two sampled
observed-ball rows where its team's nearest player is at least .5 m closer to
the ball than the opponent's. The 403 retained spans cover every TRAIN game,
with 4–18 per game. This can favor easy, observable cases; it does not establish
gold-standard fidelity or create new supervision.

The [duration/provenance audit](results/short_quality_diversity_v1_label_fidelity_audit.json)
finds just **38 corroborated subsecond spans out of 198**, versus 159/687 for
one-to-two seconds and 206/542 for two-to-three seconds. Only 25,520 of the
74,889 native short-possession frames have an observed ball. In 1,209 spans,
no individual controlled touch lasts .2 seconds; brief touch flags can be
legitimate one-touch actions, so that duration is reported, not used to reject
labels. The audit does not justify making every touch into a possession.

## Two fixed training interventions

Both variants preserve the numerical leader's original-provider draws exactly.
For each GAR draw, retain the original game selection, then independently
request a replacement with probability .25. Native complete-span eligibility
preserves record edges, unknowns and time-block boundaries. Every replacement
uses an existing window from the same game and retains its feature/context
metadata. Ordinary draws still train on all the original labels.

**Quality:** choose uniformly from existing windows overlapping a corroborated
span. **Diversity:** shuffle distinct native possession IDs within each game,
exhaust each bag before reuse, and choose an overlapping existing window for
the selected possession. This balances repeated exposure to existing examples;
it does not add new games or independent labels.

The exact pre-fit sampling audit covers all twelve declared fits:

| Exposure measure per fit | Quality | Diversity |
| --- | ---: | ---: |
| GAR replacement draws | 8,026–8,100 | 8,034–8,109 |
| Distinct corroborated possessions covered | 403 | 403 |
| Distinct replacement windows | 457 | 457 |
| Effective possession count, inverse concentration | 310.6–316.6 | 328.7–342.3 |
| Short H/A rows / all sampled GAR H/A rows | 8.12–8.25% | 8.66–8.74% |

Effective count is `(sum exposures)^2 / sum(exposures^2)` over corroborated
possessions overlapping replacement windows. A window can cover multiple spans;
counts include each overlap. The bag sampler separately records nominated IDs.
Both recipes already cover all 403 spans, so diversity means more even reuse,
not a larger set of observed examples.

Two seeds (2026/2027) and three full provider exclusions yield six fits per
variant. Every learned H/A, D and normalization component excludes the evaluated
provider. Teachers, frozen normalizers, raw-feed views, dense labels, losses,
width96, batch32, half original/half GAR, H/A learning rate .0003, D learning
rate .0001 and 2,048 updates match the preceding study. Both heads see the
changed window distribution. There is no early stopping.

## Frozen evaluation and stopping rule

Seal all twelve checkpoints before evaluating any candidate. Five fixed arms:
each variant's full probability ensemble; each variant's conditional H/A
crossed with raw-feed control D within seed before averaging; and one equal
native-probability blend of the diversity H/A cross with the numerical leader.
Every arm uses the unchanged D-temperature4/cost3 decoder.

Compare the same five exposed development matches, native grading, clean view
and two original feed-like corruption replicas. Average replicas within match,
matches within provider, then providers equally. Rank by official feed-like
spell F1, macro F1 for exact ties; report the auxiliary 70/30 summary separately.
Report both individual seeds, rather than averaging their metric scores.

Guards against the promoted model: clean aggregate spell within .005; each
provider's clean macro/D-F1 within .02; each provider's feed spell within .01.
Report the historical +.01 clean screen separately. A useful new leader must
also exceed .527396. No rates, thresholds, seeds, decoder settings or blend
weights are added after observing results. If a fixed eligible arm improves
the leader, lock it and compare it with that leader on two newly generated
target-independent corruption realizations per match, without reselection.
New synthetic views still do not provide independent-match confirmation.

## Conditional original-provider diversity continuation

Before any parent scores, `short_original_diversity_v1` declares six additional
fits only if the five parent arms miss .55 under guards. Keep the GAR quality
and distinct-possession schedule exactly fixed. Change only the original
training windows: instead of uniform eligible-window selection, cycle distinct
complete native short possessions within the same already-selected record,
then choose one of its existing overlapping windows. The replacement request
rate remains .25; source, match, paired feature view and H/A orientation stay
fixed. A separate RNG leaves the entire GAR draw schedule unchanged.

Original-provider labels are not rewritten or newly corroborated in this
continuation. Its hypothesis concerns exposure diversity on the other half
of the batch. Counts retain paired-record duplication explicitly; they are
not presented as additional physical examples. Six checkpoints must be sealed
before the three fixed full/H/A-cross/equal-leader-blend arms are scored.
The primary metric, decoder, guards, seeds and update budget remain fixed.
No additional rates, durations or decoder settings follow this continuation.

## Verification and reproduction

The preflight reproduces all 90 archived leader/raw-feed control cells exactly.
A restored raw-feed model's inference differs by at most 7.45 × 10⁻⁷ in
probabilities and changes zero decoded labels. All 543 experiment contracts and
26 root tests pass, along with file-backed fit parity and bundled CSV format
validation. The first full test attempt identified missing pinned PyArrow;
installing `requirements-soccernet.txt` resolved all three dependency errors.

Restore the exact parent research archives and dependencies using their private
manifests. The [declaration](results/short_quality_diversity_v1_declaration.json)
binds source/input hashes; the [training audit](results/short_quality_diversity_v1_training_audit.json)
records all fixed draw sequences and the unchanged source/game exposures.
Restore completed artifacts instead of repeating frozen commands.

```sh
.venv/bin/python -m experiments.short_quality_diversity_v1 audit
.venv/bin/python -m experiments.short_quality_diversity_v1 declare
.venv/bin/python -m experiments.short_quality_diversity_v1 preflight
.venv/bin/python -m experiments.short_quality_diversity_v1 fit --variant quality --fold leave_out_metrica --seed 2026
# Complete the other eleven fixed variant/fold/seed combinations, then:
.venv/bin/python -m experiments.short_quality_diversity_v1 seal
.venv/bin/python -m experiments.short_quality_diversity_v1 evaluate
.venv/bin/python scripts/verify_short_quality_diversity_v1.py
```

No organizer references, GAR VALID/TEST labels, public scoring or model
promotion are part of this study. All tracking arrays, checkpoints and frame
outputs remain in ignored/private storage. Results are development evidence
on previously exposed matches, some with event-derived silver references.
