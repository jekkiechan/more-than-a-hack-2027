# Iterative v2 model experiments

Study date: 6 November 2026. Status: running. The adopted package remains
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.

## Question and prior evidence

Can combining archived models and continuing training on the additional
SoccerNet-GAR supervision raise matched v2 spell F1 above **0.60**, with no
regression in the other protected metrics?

This study builds on open PRs #82–84 at `30ddcbc`. Their best v2 point estimate,
0.594110, did not retain all clean/v1 metrics. Larger H/A capacity and a mixed
v1/v2 training recipe also failed the strict retention rule. Those historical
results and failures remain unchanged. Luna assisted with restoration,
experimental implementation, contract tests, and independent verification.

The [parent plan](results/v2_iterative_v1_plan.json) limits the work to three
separately frozen rounds: four archived-model combinations, twelve paired
continuation fits, and a final bounded combination follow-up. Each round has its
own declaration, sources, artifact hashes and results.

## Data and evaluation

The development cohort is the same five previously exposed matches: Sportec
J03WOY, Metrica Game 3 and SkillCorner 1996435, 1996436 and 2006229. Clean input,
two fixed historical v1 realizations and four fixed v2 realizations produce
35 matched cells. Replicas are averaged within match, then matches within
provider, then providers equally. Each prediction averages the two fixed seeds
within each native sampling phase, then averages phases before decoding.
Individual-seed scores are reported separately; their metric mean is not the
ensemble score.

All models, teachers, normalizers and original training examples exclude the
held-out provider. The additional training corpus is the archived 45-game GAR
TRAIN set, combined with the original 13 TRAIN matches. These GAR records carry
generated dense silver labels and discontinuous segments. This study reuses
existing additional supervision; it does not acquire new independently labelled
matches. Native unknowns, missingness, half/time breaks and team symmetry are
preserved.

The evaluator must reproduce all 24 adopted aggregate metrics within the
predeclared 1e-8 replay tolerance before any candidate scoring. Success uses
**zero regression tolerance**: spell F1, macro F1, D-F1, D precision, D recall
and under-three-second spell recall must each be at least the matched adopted
value on clean, v1 and v2. V2 spell must be strictly greater than 0.60. Ranking
uses unrounded v2 spell, with macro F1 breaking exact ties. The 70/30 combined
score is descriptive. Reference spell counts must remain invariant; predicted
counts and provider spread are diagnostics.

## Round one: combine archived models

Four fixed pipelines combine adopted original-full and archived calibrated
probabilities: a uniform categorical blend with the joint cost-3 decoder;
a uniform conditional H/A blend under the adopted D mask with H/A cost 3;
the same conditional blend at cost 1.5; and an equal conditional logit blend
under the adopted D mask at cost 3. The fixed-D adopted-only conditional
replay is a diagnostic, since conditional and joint H/A decoding need not
produce identical labels.

## Round two: paired boundary-loss continuation

Two recipes, three provider exclusions and seeds 2026/2027 give twelve fits.
Each starts from the corresponding verified adopted H/A checkpoint, retains
its exact normalizer and D weights, and runs 512 updates with batch size 32,
a 50/50 original/GAR mixture and learning rate 0.0003. Both recipes use the
shared distinct-possession sampler and the same four round-robin v2 TRAIN
views. The paired arms must have identical source/game/window draw schedules.

The control retains the existing dense duration-weighted H/A cross-entropy
and live-transition objective. The treatment adds a signed H/A logit-margin
loss with coefficient 0.1 and margin 1. It covers the two rows adjacent to a
known H/A transition plus one neighboring model sample on either side. Shared
valid-pair masks, period identity and known H/A targets exclude false edges
across gaps, unknowns, dead play or padding. Labels enter this loss only while
fitting. Inference receives tracking-derived features alone.

Each recipe is evaluated on its own and as a fixed equal conditional-H/A blend
with the adopted model, always retaining the adopted decoded D mask and H/A
cost 3. Preserving D probabilities alone would not guarantee unchanged decoded
D labels, which is why these arms explicitly retain the decoded mask.

## Limitations and delivery

These are further development experiments on selected-on matches. The v2
simulator places corruption using development/training states and has a
historical model-conditioned calibration limitation. Four realizations reduce
corruption-draw sensitivity; they do not create independent-match evidence.
No organizer-reference scoring, GAR VALID/TEST selection, pooled deployment
fitting or promotion is part of this experiment. Weights and native outputs
remain in ignored local storage; remote availability is recorded separately.
