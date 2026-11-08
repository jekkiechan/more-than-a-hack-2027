# Current research state

Reviewed 8 November 2026 against fork `main` at `8919726`,
[PR #87](https://github.com/jekkiechan/more-than-a-hack-2027/pull/87) at `b12eab3`,
[draft PR #88](https://github.com/jekkiechan/more-than-a-hack-2027/pull/88), and
Claude's `fervent-curie-fcqsta` at `61f8182`. The active manifest
now records the user-authorized spell-frontier promotion. Check newer merged/open
PRs before current-results advice. Historical study evidence is indexed in
[EXPERIMENTS](EXPERIMENTS.md); package restore/inference lives in
[BASELINE](../../experiments/BASELINE.md).

## Adopted package

`joint_gru_sn_dense_spell_frontier_ensemble_v1` is the provisional working
baseline, explicitly approved by the user after the frozen frontier diagnostic.
It combines CE/calibrated/predecessor conditional H/A probabilities at
0.5/0.375/0.125, averages seeds within phases then phases, and keeps the exact
predecessor decoded D mask. Promotion adds no fitting or organizer scoring.

Original exposed development clean/v1/v2 spell is **.670137/.532128/.593619**,
with v2 +.010048 versus the previous phase-average package. All three spell
means improve; v1 macro declines .000699 within the accepted .001 allowance.
**The >.60 v2 target and historical +.01 clean screen remain unmet.** Provider
regressions remain disclosed; user approval changes adoption, not those outcomes.

The exact deployment GAR TEST10 clean/v2 spell is
**0.708496/0.708619**.
TEST short-spell recall declines by .008267 clean and .005504 v2 despite better
spell and macro means; the deployment is not an all-metric improvement on GAR.
Selected-on VALID9 stays separate. All 95 paired cells pass independent official
replay; all predecessor per-game controls, D masks and grading populations agree.
This is reused silver-segment evidence, not independent continuous-match gold
confirmation. See [promotion and GAR results](../../experiments/SPELL_FRONTIER_PROMOTION.md).

The unchanged saved public diagnostic is **.611765 spell / .851285 macro** on
7,500 graded frames and 17 possessions. The normal CLI reproduces the exact
199,319-row sealed CSV without reference access. The public gate remains closed;
the earlier balanced gate remains unused. The historical public score is not
validation of the new extra-data experimental arms.

## Extra-data continuation — complete

[The two-round follow-up](../../experiments/V2_EXTRA_DATA.md) uses Luna to test
18 provider-excluded continuation fits and four cached probability blends.
The training arms isolate GAR objective weight and corroborated short-possession
sampling against a paired fresh-draw control, retaining the same 13 eligible
original TRAIN matches and 45 GAR TRAIN games.

Fresh continuation raises clean/v1/v2 spell to **.670446/.532146/.594751** but
loses v1 short-spell recall. Higher GAR weight reaches **.595084 v2** but loses
v1 spell and short recall. Quality sampling preserves short recall but loses
clean and v2 spell. All macro retention checks pass and D remains exact.
**None of the three training arms or four cached blends retains every check. Keep the approved frontier package.**

The quality control already nominates 402–403 of 403 qualified short spans;
increasing the rate mainly repeats existing supervision. Training lineage,
paired exposure and all saved candidate/seed predictions are independently
audited. The four-mixture control replay has zero error across 105 cell/arm
pairs. No additional organizer or GAR scoring is used to choose these arms.

## Next work

The [8 November complementary pilot](../../experiments/COMPLEMENTARY_SPELL_V1.md)
is complete on the exact historical original-full phase fold models. Equal
seed weighting remains best among five tested ratios on v2. A count-preserving
posterior-median boundary rule gains only .000207 v2 spell (with clean/v1 gains
and all pilot checks passing), mainly in SkillCorner; 280 saved outputs replay
independently. This is a small exposed-development signal, not a verified
upgrade to the current frontier package. The original research caches were
unavailable; the [current-frontier follow-up](../../experiments/FRONTIER_POSTPROCESS_V1.md)
restores source inputs, reconstructs twelve CE fold fits and reproduces every
saved selected-frontier metric across 35 development cells, including exact
.5936187198896373 v2 spell. Checkpoint files are reconstructed, not byte-identical
to the missing originals.

The first round tests nine seed-weight, boundary, H/A-cost and disagreement
changes against this verified control. All 350 saved predictions pass independent
replay; none passes every retention guard. Disagreement strength 1 reaches
.593689 v2 but loses v1/v2 short recall; boundary .8 s improves short recall but
loses .000025 v2 spell. The separately declared composition reaches .593532 v2
and loses v2 short recall; all 35 new predictions and 105 controls replay
independently. No candidate is retained. The adopted package remains unchanged.

Claude's requested branch became visible during this work. Its pinned
[197-arm study](https://github.com/jekkiechan/more-than-a-hack-2027/blob/61f8182f86f2faca7873edd4db4c060a93285ec2/experiments/V2_POSTPROC_V1.md)
also reports no retained candidate; its .594992 v2 leader loses clean spell and
short recall. We inspected its saved report without independently replaying its
197 predictions or modifying that branch.

Stop the declared local rate/weight/blend comparisons. Their results do not
establish a hard performance ceiling, but further fine searches on the same five
exposed matches are weaker evidence than new data. Prioritize independent,
continuous-match H/A/D/X annotation and more varied corroborated short
possessions. GAR silver clips, sparse PCBAS action anchors and model-conditioned
synthetic views do not replace that evaluation.

Keep development clean plus four or more v2 realizations, equal-provider means,
all spell views, short recall and fixed D/count checks. V1/composite remain
historical diagnostics. Do not reopen closed public gates or use the reused
17-possession window for tuning.

## Artifacts and delivery

All ten deployment checkpoint roles, matching source, manifests/policy and
aggregate provenance are verified in a local archive, including fresh
empty-directory extraction and an isolated ten-model load. No new private-cache
upload is claimed; the old remote default still describes the historical
phase-average package. [Current package instructions](../../experiments/BASELINE.md)
and [local archive receipt](../../experiments/results/baseline_spell_frontier_local_artifacts.json)
own exact availability. New research weights, sampling sequences and frame
outputs remain local; the full training seal stays unchanged at its ignored
frozen path and its public aggregate inventory binds its hash.

The promotion report preserves aborted GAR attempts, the seed-description
erratum, six-phase compatibility correction and exact runtime guards. Old
runners, declarations, decisions and scoring gates remain intact. Use the
recorded pre-activation revision to replay historical hash-bound studies.

The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
is published as version 34 with the same restricted audience. Desktop/mobile
checks pass across all six pages and all 12 new chart combinations.
[PR #87](https://github.com/jekkiechan/more-than-a-hack-2027/pull/87) carries the
completed experiments and promotion; merging follows the existing CI policy.
[Delivery receipt](../../experiments/results/v2_extra_data_v1_delivery.json).
