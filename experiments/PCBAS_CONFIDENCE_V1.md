# PCBAS confidence check: sparse event anchors without ball tracking

Date: 2 November 2026. Status: completed; adopted model retained.

## Question and scope

Can the frozen original-full phase-average package identify the controlling team
at independently annotated action moments in a new broadcast source when measured
ball tracking is unavailable? This is a deliberately narrow portability check,
not a possession-spell evaluation or a model-selection study.

The current package, decoder and four model checkpoints remain unchanged. There
are no training fits, threshold searches, offset searches or organizer-label
accesses. The original development scores and GAR TEST scores are separate
protocols and cannot be subtracted from these event-anchor metrics.

## Acquisition and provenance

Access to [SoccerNet/SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026)
was verified after approval. The tactical VAL archive is pinned to revision
`a16e373c8c8d26b1af4874f4208810baf337e709`. Files, source metadata, hashes and
reproducible acquisition/audit scripts remain under ignored
`data/independent/sn_pcbas_2026/`; no raw or reconstructive data is published.

The [FOOTPASS paper](https://arxiv.org/html/2511.16183v1) describes a 2023–24
European cohort at 25 Hz. The three validation match identifiers are 18, 24 and
47, each split into two halves. The temporal cohort differs from GAR World Cup
2022 and the organizer's 2024–25 match. Anonymized identifiers do not establish
complete fixture-level deduplication against every historical source; this is
not certified full-pipeline independent gold confirmation.

The [official validation annotations](https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/playbyplay_GT/playbyplay_val.json)
are pinned separately. Each event provides frame, direction-based team, shirt
number, action class, box availability and replay flag. Team identity is
half-local attacking direction, not a persistent home/away identifier.

The [aggregate acquisition receipt](results/pcbas_confidence_v1_acquisition.json)
records a 156,773,777-byte ZIP and 555,386,336-byte extracted HDF5, verified by
SHA-256. The six halves contain 451,161 consecutive within-half frames and
9,917,540 player rows. All 6,070 annotated action rows join to matching tracking
frame/shirt/class records; 614 events are marked as replay. No ball entity is
present. Finite image boxes cover only about 31–49% of player rows by half;
remaining finite pitch positions are not independent observed tracks.

## What these labels mean

Action classes are background, Drive, Pass, Cross, Throw-in, Shot, Header,
Tackle and Block. Background means no annotated action; it never means dead
play. Drive moments are the primary control-team anchors. Pass, Cross and Shot
are secondary classes reported separately. Header, Tackle and Block are not
reliable possession-control labels and are excluded from the primary question.
Replay events are excluded using the supplied replay flag, not inferred from
box availability. No continuous H/A/D targets or spell F1 can be derived without
additional assumptions or independent annotation.

## Frozen diagnostic design

Use the adopted two-seed, five-phase package unchanged. Ball coordinates are
missing throughout and the ball observation mask is false; never reconstruct
ball position from the annotated actor. Only allowed tracking fields enter the
inference path. Event classes, actors and targets are excluded from features.

Coordinates are normalized. Mapping them to a standard 105 by 68 metre pitch
is an explicit pre-score adapter assumption, not verified physical dimensions.
Keep outliers and native frame/half boundaries, and distinguish source-imputed
player coordinates from image-observed players. This calibration assumption,
source imputation and whole-match ball absence limit transfer claims.

At each fixed non-replay Drive frame, score exact decoded-team agreement,
counting D/X predictions as wrong. Report per-game equal means and pooled
numerators; conditional team accuracy must remain secondary so abstention/dead
predictions cannot hide failures. Secondary class-specific results and live
anchor predicted-D rates do not measure dead-ball precision or recall. There is
one arm and no post-result offset or threshold search.

## Results and verification

The [declaration](results/pcbas_event_anchor_v1_declaration.json) is frozen before
inference. A 750-frame real-checkpoint replay matches the deployed API exactly.
Nine focused adapter/metric contracts pass, including target exclusion, replay
filtering, mask handling, half/time breaks, team symmetry and D/X penalties.
All 596 experiment tests and 26 root tests pass. The first full-suite attempt
had three environment errors from missing pinned `pyarrow`; after installing
`pyarrow==20.0.0`, the unchanged suite passed. Documentation checks pass.

All six halves completed: 451,161 native frames in 263.81 seconds. Predictions
were sealed before the single event-score pass. All 5,158 eligible live anchors
are included; no selected anchors were lost to missing tracking, duplicate
records or opposing-team collisions.

| Action | Correct / anchors | Pooled agreement | Equal-game agreement | Predicted D at live anchors (pooled) |
| --- | ---: | ---: | ---: | ---: |
| Drive (primary) | 1,616 / 2,239 | 72.18% | **71.66%** | 7.28% |
| Pass | 1,970 / 2,755 | 71.51% | 70.96% | 6.86% |
| Cross | 60 / 99 | 60.61% | 60.98% | 35.35% |
| Shot | 37 / 65 | 56.92% | 56.84% | 40.00% |

Primary Drive agreement by game is 65.11%, 69.16% and 80.72%, compared with
52.34%, 52.03% and 54.02% for each game's majority direction alias. These are
half-local direction aliases, not persistent home/away teams. The result supports
useful team-attribution transfer even with all ball input missing. It does not
show improved model weights or calibrated qualification odds. Three games and
correlated action samples do not justify a narrow confidence interval.

The live/dead gate is a material weakness at crosses and shots under this input
regime: 35/99 crosses and 26/65 shots receive D. This could reflect missing-ball
shift, tracking/scale limitations, boundary timing or a learned error; this study
does not isolate the cause. Do not tune the gate on these three now-exposed
validation matches or treat action background as negative/dead supervision.

[Full aggregate result](results/pcbas_event_anchor_v1_result.json) preserves
half-level counts, exclusions, team distributions and source/seal hashes.
Independent recomputation from sealed predictions is recorded separately in
[the verification receipt](results/pcbas_event_anchor_v1_verification.json).

## Decision and next experiment

Retain the adopted package. This is positive but narrow evidence for player-only
team attribution, with an actionable live/dead failure pattern. Full possession
spell confidence remains limited by missing continuous gold and measured ball.
The next experiment should use a separately frozen set with observed ball input
and independent continuous H/A/D intervals, or separately declared live/dead
boundary annotation around crosses/shots on new matches. Keep these VAL matches
as exposed diagnostic evidence from now on. No promotion or score-driven
parameter changes follow this result.

## Reproduction

Restore the exact adopted package as described in [BASELINE](BASELINE.md),
acquire the pinned validation archive and official event file into the ignored
source directory, and install the adapter dependency:

```sh
.venv/bin/python -m pip install -r experiments/requirements-pcbas.txt
.venv/bin/python -m unittest experiments.tests.test_pcbas_event_anchor_v1 -v
```

The one-time `freeze`, `predict`, and `score` stages are implemented in
[the study runner](pcbas_event_anchor_v1.py). Closed gates and declarations must
not be overwritten to rerun a completed study. Use the saved aggregate result
and independent verification for readout; reproduction of inference belongs in
an isolated copy preserving the exact source/input/model hashes and original
receipts. Raw inputs are access-controlled and are not included in the public
repository.

## Artifact and delivery state

Raw tactical data and annotation files are locally acquired under ignored data.
No new raw-data upload or independent remote restore is claimed. The adopted
package and frozen historical studies are unchanged. Repository and progress-site
delivery will record the completed aggregate evidence separately.
