# V2 spell frontier: mixtures and fresh training windows

Study: `v2_spell_frontier_v1` · 7 November 2026 · complete; final public diagnostic recorded.

The user requested further iteration before the final public spell-F1 diagnostic.
This separate follow-up extends [spell balancing](V2_SPELL_BALANCED.md), whose
frozen candidate reaches 0.591174 v2 spell F1 while retaining all three spell
means and short-spell recall above adopted. Its full reference-free public CSV
is sealed but remains unscored. Every prior declaration, result and gate is
preserved.

## Results and decision

**The final eligible candidate reaches 0.593619 v2 spell F1 (+0.010048 versus adopted).** The >0.60 goal remains unmet.

The final pipeline is `base__ce0.5_cal0.375_ad0.125_arithmetic_ha3`, after 1024 continuation updates. It replaces the balanced study’s frozen choice. No adoption change is made.

| Pipeline | Clean spell | V1 spell | V2 spell | V2 macro |
| --- | ---: | ---: | ---: | ---: |
| Adopted baseline · 0 | 0.667556 | 0.531576 | 0.583571 | 0.849975 |
| Original promising blend · 512 | 0.668249 | 0.534925 | 0.585615 | 0.850181 |
| balanced · base · 50/25/25 mean · HA 3 · 512 | 0.669461 | 0.537144 | 0.590454 | 0.851303 |
| balanced · base · 50/25/25 mean · HA 3 · 1024 | 0.668533 | 0.533876 | 0.591174 | 0.851496 |
| Final selection · frontier · base · 50/37.5/12.5 mean · HA 3 · 1024 | 0.670137 | 0.532128 | 0.593619 | 0.852204 |

Relative to the original promising arm, the final v1 spell change is **-0.002797**. The shorter 512-update 50/25/25 arithmetic blend improves all three spell scores relative to that arm, reaching **0.590454 v2**. The frozen rule ranks retained v2 performance against adopted; the final choice is a tradeoff, not dominance over every earlier blend.

### What each round found

Round 1: **5/30 eligible**. Eligible leader `ce0.5_cal0.375_ad0.125_arithmetic_ha3` gives clean/v1/v2 spell **0.670137/0.532128/0.593619**.

Round 2: **1/4 eligible**. Eligible leader `fresh__ce0.5_cal0.25_ad0.25_arithmetic_ha3` gives clean/v1/v2 spell **0.670968/0.533503/0.590891**.


The longer-training arms do not beat the wider stage-1024 mixture under the fixed retention rule. The highest longer-training v2 score fails v1 spell and short-spell recall. Further continuation therefore does not justify replacing the retained leader.

| Continuation pipeline | Clean spell | V1 spell | V2 spell | Failed checks |
| --- | ---: | ---: | ---: | --- |
| `repeat__ce0.5_cal0.375_ad0.125_arithmetic_ha3` | 0.669021 | 0.530852 | 0.593962 | feed_like_v1/short_spell_recall_iou_0_5_lt3s, feed_like_v1/spell_f1 |
| `fresh__ce0.5_cal0.375_ad0.125_arithmetic_ha3` | 0.670552 | 0.530120 | 0.594697 | feed_like_v1/short_spell_recall_iou_0_5_lt3s, feed_like_v1/spell_f1 |
| `repeat__ce0.5_cal0.25_ad0.25_arithmetic_ha3` | 0.668321 | 0.531326 | 0.590154 | feed_like_v1/spell_f1 |
| `fresh__ce0.5_cal0.25_ad0.25_arithmetic_ha3` | 0.670968 | 0.533503 | 0.590891 | None |

The overall numerical maximum is **0.594761 v2**, from the stage-1024 50/50 continuation/calibrated log mixture. It fails v1 short-spell recall and is not selected. No tested pipeline exceeds 0.60.

### Retention and provider spread

| View | Δ spell | Δ macro | Δ short recall | D F1 | D precision | D recall | 70/30 auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | +0.002581 | +0.000593 | +0.002246 | 0.870926 | 0.884844 | 0.861920 | 0.730655 |
| feed_like_v1 | +0.000553 | -0.000699 | +0.000078 | 0.860563 | 0.843006 | 0.881217 | 0.620336 |
| feed_like_v2 | +0.010048 | +0.002229 | +0.004450 | 0.866138 | 0.873034 | 0.862980 | 0.671194 |

| View / provider | Spell | Δ spell | Δ macro | Δ short recall |
| --- | ---: | ---: | ---: | ---: |
| clean / Metrica | 0.848418 | +0.002201 | +0.000350 | +0.000000 |
| clean / SkillCorner | 0.484069 | +0.007101 | +0.001690 | +0.006738 |
| clean / Sportec | 0.677922 | -0.001558 | -0.000261 | +0.000000 |
| feed_like_v1 / Metrica | 0.644307 | +0.015025 | +0.003741 | +0.011905 |
| feed_like_v1 / SkillCorner | 0.448655 | -0.002345 | -0.000225 | -0.006595 |
| feed_like_v1 / Sportec | 0.503422 | -0.011021 | -0.005612 | -0.005076 |
| feed_like_v2 / Metrica | 0.705783 | +0.024877 | +0.003860 | +0.009921 |
| feed_like_v2 / SkillCorner | 0.483972 | -0.000166 | +0.000697 | -0.000376 |
| feed_like_v2 / Sportec | 0.591101 | +0.005432 | +0.002130 | +0.003807 |

Aggregate retention does not imply improvement for every provider. Provider losses above remain disclosed; the frozen rule is applied to equal-provider view means.

### Paired seed diagnostics

| View | Seed 2026 spell | Seed 2027 spell | Inference ensemble spell |
| --- | ---: | ---: | ---: |
| clean | 0.672159 | 0.666558 | 0.670137 |
| feed_like_v1 | 0.532667 | 0.525340 | 0.532128 |
| feed_like_v2 | 0.590726 | 0.587623 | 0.593619 |

These seed diagnostics hold the pooled adopted D mask fixed. They describe H/A seed spread; they are not independent repetitions of the complete D pipeline. The ensemble averages probabilities before decoding, not scores.

The three-expert public pipeline performs 30 seed/phase model passes versus 10 for adopted (three expert families × two seeds × five phases); no latency-neutral claim is made.

### Actual data exposure

| Fold / seed | Repeat unique windows | Fresh unique windows | Shared windows |
| --- | ---: | ---: | ---: |
| leave_out_metrica/2026 | 5635 | 5622 | 3683 |
| leave_out_metrica/2027 | 5667 | 5664 | 3720 |
| leave_out_skillcorner/2026 | 4469 | 4473 | 3336 |
| leave_out_skillcorner/2027 | 4495 | 4464 | 3333 |
| leave_out_sportec/2026 | 5374 | 5392 | 3547 |
| leave_out_sportec/2027 | 5392 | 5330 | 3499 |

All six paired comparisons use different recorded draw-sequence digests. Window identities include source/game, start, size and the original-source alternate flag (GAR identities exclude the later team-swap draw). Both arms retain the paired D parameters and normalization. These are additional draws from existing TRAIN data, not new annotations.

### Final public diagnostic

| Public pipeline | Spell F1 | Macro F1 | D F1 |
| --- | ---: | ---: | ---: |
| Saved adopted | 0.611765 | 0.849473 | 0.862870 |
| Frozen final candidate | 0.611765 | 0.851285 | 0.862870 |

Public spell changes by **+0.000000**, macro by **+0.001812**. The adopted full-CSV hash replays exactly before sealing. One final 199,319-row candidate is format-validated and scored only after development selection; the balanced gate remains unused. The reference grades 7,500 frames and 17 possessions. No public-driven tuning or reselection follows.

### Verification and artifacts

All four development rounds pass their independent saved-output audits. Every candidate scorecard is recomputed with the shared official evaluator; shortlisted mixtures and both seeds are independently reconstructed. Native phase pooling is checked in the trained rounds. Full cohort identities, reference counts and exact D masks are retained. The first study’s stage-label and audit-order execution corrections remain recorded in its report; no frozen evidence was rewritten.

Research weights, sample receipts and full-frame outputs remain in ignored local storage. The local model archive is hash-verified; no new remote weight availability is claimed. The retained adopted model and policy are unchanged.

- [Round one](results/v2_spell_frontier_v1_round1.json) and [round two](results/v2_spell_frontier_v1_round2.json) preserve every candidate and failed check.
- [Final selection](results/v2_spell_frontier_v1_selection.json) and [combined readout](results/v2_spell_frontier_v1_readout.json).
- [Independent R1 audit](results/v2_spell_frontier_v1_independent_round1.json) and [R2 audit](results/v2_spell_frontier_v1_independent_round2.json).
- [Training exposure seal](results/v2_spell_frontier_v1_training_seal.json) and [local artifact receipt](results/v2_spell_frontier_v1_artifacts.json).
- [Final public result](results/v2_spell_frontier_v1_public_result.json).

## Question and fixed design

The best retained mixture used calibrated weight 0.25, the edge of the earlier
grid. Thirty fixed stage-1024 mixtures extend calibrated weights to 0.75,
with nonnegative adopted residual, arithmetic/log pooling and H/A cost 3.
Both prior leader mixtures replay before new scoring. The two best eligible
mixtures plus the current arithmetic anchor form a deduplicated shortlist.

A paired continuation compares repeated versus fresh TRAIN-window draws:
twelve provider-excluded fits, two seeds, 512 further updates at learning rate
0.0001, warm-started from exactly paired stage-1024 checkpoints. Both use the
same loss, optimizer reset, batch size 32, 50% GAR, four fixed v2 TRAIN views,
and Torch/dropout seeds. The fresh arm offsets the original distinct-possession
sampler and trainer draw RNG by 104729; actual window receipts verify changed
exposure. D parameters, normalization and teacher identity remain fixed.
The shortlist is evaluated with both new stage-1536 arms (at most six pipelines).

The original TRAIN pool has 13 matches; each development fold excludes its
whole provider from all learned components and normalization. GAR contributes
45 TRAIN games with generated silver targets. No GAR VALID or TEST evaluation
is part of this declaration. The five exposed development matches produce 35
cells: clean, two historical v1 realizations and four fixed v2 realizations.
Replicates are averaged within match, matches within provider, then providers
equally. Seeds are pooled within native phase before averaging phases.

## Selection and evidence limits

Eligibility remains fixed against adopted: strictly higher spell F1 in all
three views, macro F1 at least adopted minus 0.001 per view, short-spell recall
at least adopted, exact adopted D predictions and unchanged reference counts.
Rank eligible candidates by v2 spell F1, macro for exact ties, then pipeline ID.
A new candidate replaces the balanced selection only with strictly higher v2
spell F1; otherwise retain it. Seed diagnostics never rerank candidates.
The aspiration remains **v2 spell F1 above 0.60**. No automatic adoption follows.

Independent audits rescore saved predictions and verify hashes, identities,
provider reduction, gates, rankings and shortlisted seed/ensemble decoding.
Round two also verifies native phase pooling. Selection precedes pooled TRAIN
reproduction, full-match prediction, sealing and one final public score across
both studies. The old balanced public gate stays unused. Public labels cannot
choose mixtures, seeds, repairs or thresholds.

These are repeatedly exposed development matches and model-conditioned
synthetic views. Additional training draws do not create independent evaluation
evidence or new gold labels. The public window has only 7,500 graded frames and
17 possessions and is a reused pipeline diagnostic.

## Evidence

- [Frozen parent plan](results/v2_spell_frontier_v1_plan.json)
- [Previous development selection](results/v2_spell_balanced_v1_selection.json)
- [PR #86](https://github.com/jekkiechan/more-than-a-hack-2027/pull/86)
