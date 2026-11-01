# Short-possession label corroboration and exposure diversity

Study `short_quality_diversity_v1`, 1 November 2026. 18 fixed fits are complete. The adopted model is unchanged.
Best eligible original feed-like spell F1 across controls and candidates: **0.528180**.

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

The H/A crosses retain the raw-feed control's D probabilities within seed.
Joint decoding can still change D labels when the H/A emissions change;
probability preservation is not a claim of identical decoded D output.

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

The [physical-span audit](results/short_original_diversity_v1_physical_span_audit.json)
deduplicates match/native endpoints across paired feature views and H/A swaps.
The original pools contain 515, 948 and 853 physical short spans for the
Sportec-, Metrica- and SkillCorner-excluded folds respectively. Their training
records contain exactly four copies each (2,060/3,792/3,412 paired-record
entries). The sampler audit reaches every eligible paired-record entry in both
seeds. These copies do not increase the diversity of physical supervision.

## Verification and reproduction

The preflight reproduces all 90 archived leader/raw-feed control cells exactly.
A restored raw-feed model's inference differs by at most 7.45 × 10⁻⁷ in
probabilities and changes zero decoded labels. All 545 experiment contracts and
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

## Completed original-benchmark results

| Fixed arm | Clean spell | Feed-like spell | Feed macro | Auxiliary 70/30 | Guards |
| --- | ---: | ---: | ---: | ---: | --- |
| Promoted baseline | 0.658625 | 0.518527 | 0.823343 | 0.609972 | Pass |
| Previous numerical leader | 0.657215 | 0.527396 | 0.824460 | 0.616515 | Pass |
| Distinct GAR possessions, full | 0.657582 | 0.524750 | 0.822643 | 0.614118 | Pass |
| Distinct GAR possessions, H/A cross | 0.660452 | 0.526037 | 0.822418 | 0.614951 | Pass |
| GAR diversity + leader, equal blend | 0.658372 | 0.526877 | 0.824015 | 0.616018 | Pass |
| Corroborated windows, full | 0.658638 | 0.526944 | 0.822801 | 0.615701 | Pass |
| Corroborated windows, H/A cross | 0.660532 | 0.525462 | 0.823383 | 0.614838 | Pass |
| Distinct original possessions, full | 0.662590 | 0.528180 | 0.825370 | 0.617337 | Pass |
| Distinct original possessions, H/A cross | 0.660243 | 0.526386 | 0.824209 | 0.615733 | Pass |
| Original diversity + leader, equal blend | 0.660162 | 0.527613 | 0.823926 | 0.616507 | Pass |

Best eligible arm: **Distinct original possessions, full**, 0.528180 feed-like spell F1, +0.000784 versus the previous numerical leader. The .55 stretch is not reached; the remaining gap is 0.021820. The adopted package is unchanged.

| Arm | Feed D precision | Feed D recall | Feed D-F1 | Short recall (<3s, IoU .5) | Predicted spells | Reference spells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | 0.843412 | 0.877506 | 0.859048 | 0.198820 | 282.333 | 404.333 |
| Previous numerical leader | 0.842867 | 0.876875 | 0.858212 | 0.212521 | 295.389 | 404.333 |
| Distinct GAR possessions, full | 0.833569 | 0.886069 | 0.858283 | 0.213543 | 301.611 | 404.333 |
| Distinct GAR possessions, H/A cross | 0.840955 | 0.876678 | 0.857174 | 0.215604 | 303.667 | 404.333 |
| GAR diversity + leader, equal blend | 0.843354 | 0.877789 | 0.858863 | 0.213525 | 299.611 | 404.333 |
| Corroborated windows, full | 0.826716 | 0.888078 | 0.855924 | 0.217287 | 301.833 | 404.333 |
| Corroborated windows, H/A cross | 0.840588 | 0.877158 | 0.857303 | 0.218984 | 304.833 | 404.333 |
| Distinct original possessions, full | 0.842902 | 0.880688 | 0.860203 | 0.220039 | 305.556 | 404.333 |
| Distinct original possessions, H/A cross | 0.841913 | 0.877545 | 0.858185 | 0.218251 | 303.556 | 404.333 |
| Original diversity + leader, equal blend | 0.842626 | 0.876256 | 0.857777 | 0.214837 | 299.444 | 404.333 |

Spell counts are equal-provider means, not raw corpus totals. Evaluation short recall uses <3s and global one-to-one IoU .5 matching; TRAIN eligibility uses complete <=3s native spans.

### Provider and seed spread

| Arm | Metrica feed spell | SkillCorner feed spell | Sportec feed spell |
| --- | ---: | ---: | ---: |
| Promoted baseline | 0.606749 | 0.454592 | 0.494242 |
| Previous numerical leader | 0.620941 | 0.456672 | 0.504574 |
| Distinct GAR possessions, full | 0.612926 | 0.447949 | 0.513374 |
| Distinct GAR possessions, H/A cross | 0.617184 | 0.447080 | 0.513847 |
| GAR diversity + leader, equal blend | 0.618229 | 0.453327 | 0.509075 |
| Corroborated windows, full | 0.610523 | 0.455595 | 0.514714 |
| Corroborated windows, H/A cross | 0.614234 | 0.451554 | 0.510598 |
| Distinct original possessions, full | 0.616220 | 0.450872 | 0.517448 |
| Distinct original possessions, H/A cross | 0.612726 | 0.450515 | 0.515917 |
| Original diversity + leader, equal blend | 0.619667 | 0.453519 | 0.509653 |

| Arm / seed | Clean spell | Feed-like spell |
| --- | ---: | ---: |
| Previous numerical leader / 2026 | 0.652746 | 0.514093 |
| Previous numerical leader / 2027 | 0.654943 | 0.513095 |
| Distinct GAR possessions, full / 2026 | 0.645801 | 0.515758 |
| Distinct GAR possessions, full / 2027 | 0.648864 | 0.512738 |
| Distinct GAR possessions, H/A cross / 2026 | 0.648299 | 0.515831 |
| Distinct GAR possessions, H/A cross / 2027 | 0.650412 | 0.514197 |
| GAR diversity + leader, equal blend / 2026 | 0.650308 | 0.515226 |
| GAR diversity + leader, equal blend / 2027 | 0.655669 | 0.518044 |
| Corroborated windows, full / 2026 | 0.650866 | 0.512594 |
| Corroborated windows, full / 2027 | 0.651480 | 0.515021 |
| Corroborated windows, H/A cross / 2026 | 0.649429 | 0.512300 |
| Corroborated windows, H/A cross / 2027 | 0.651931 | 0.516155 |
| Distinct original possessions, full / 2026 | 0.650750 | 0.516486 |
| Distinct original possessions, full / 2027 | 0.660130 | 0.518873 |
| Distinct original possessions, H/A cross / 2026 | 0.651532 | 0.516255 |
| Distinct original possessions, H/A cross / 2027 | 0.659210 | 0.518758 |
| Original diversity + leader, equal blend / 2026 | 0.647245 | 0.514857 |
| Original diversity + leader, equal blend / 2027 | 0.655841 | 0.519475 |

Individual seed scores are descriptive; their average is not the probability ensemble. Both checkpoints contribute equally before decoding. These two seeds do not establish independent-match uncertainty.

The aggregate gain comes from Sportec (+.012874 feed spell); Metrica (−.004721) and SkillCorner (−.005801) decline against the preceding leader. Both individual seeds improve aggregate feed spell, while clean seed outcomes are mixed. Feed short-spell recall rises .212521 → .220039. The separate H/A cross does not retain the full model’s gain, so the result cannot be attributed to team sampling alone.

### Declared guards and historical screen

- **Distinct GAR possessions, full:** passes feed-first guards; historical +.01 clean screen fails.
- **Distinct GAR possessions, H/A cross:** passes feed-first guards; historical +.01 clean screen fails.
- **GAR diversity + leader, equal blend:** passes feed-first guards; historical +.01 clean screen fails.
- **Corroborated windows, full:** passes feed-first guards; historical +.01 clean screen fails.
- **Corroborated windows, H/A cross:** passes feed-first guards; historical +.01 clean screen fails.
- **Distinct original possessions, full:** passes feed-first guards; historical +.01 clean screen fails.
- **Distinct original possessions, H/A cross:** passes feed-first guards; historical +.01 clean screen fails.
- **Original diversity + leader, equal blend:** passes feed-first guards; historical +.01 clean screen fails.

### Artifact and verification evidence

- `short_quality_diversity_v1`: [aggregate](results/short_quality_diversity_v1_aggregate.json), [model seal](results/short_quality_diversity_v1_model_seal.json), [separate official/algebra replay](results/short_quality_diversity_v1_verification.json).
- `short_original_diversity_v1`: [aggregate](results/short_original_diversity_v1_aggregate.json), [model seal](results/short_original_diversity_v1_model_seal.json), [separate official/algebra replay](results/short_original_diversity_v1_verification.json).

## Fixed new-corruption diagnostic

The winner was frozen as `original_full` before these scores. Two new target-independent corruption realizations per existing match use separately frozen seeds and inputs. The candidate was not reselected.

| Arm | New-view feed spell | New-view feed macro |
| --- | ---: | ---: |
| fixed_candidate | 0.547221 | 0.827410 |
| leader | 0.546583 | 0.827986 |

These numbers use different synthetic realizations from the original-benchmark table; compare only the matched rows here. They remain exposed-match development evidence and do not provide independent-match or gold-label confirmation.

[New-view declaration](results/short_quality_fresh_v1_declaration.json), [aggregate](results/short_quality_fresh_v1_aggregate.json), [separate official/component replay](results/short_quality_fresh_v1_verification.json).

## Decision and limits

The completed comparison improves the numerical development leader at 0.528180. Corroboration is a training-sampling proxy, and all recipes reuse existing labels and examples. Even balanced reuse leaves only 38 corroborated subsecond GAR possessions across 45 games. This does not establish that rejected labels are wrong, that observable-only sampling is universally useful, or that synthetic stress gains transfer to real broadcast tracking.

Stop the declared comparisons. The next evidence priority is a larger, more varied set of externally corroborated short possessions and independently labeled dense evaluation. Do not turn uncontrolled touches into possessions to manufacture supervision. No automatic promotion or organizer-reference diagnostic follows these scores.
