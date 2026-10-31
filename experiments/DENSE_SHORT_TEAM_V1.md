# Subsecond possession team attribution

Status: both fixed comparisons completed; retain the adopted baseline,
31 October 2026.
Study: `dense_short_team_v1`, based on fork main `09cbce3`.

## Question and comparison

Does increasing the training contribution of complete H/A possessions lasting
at most one second improve team attribution and overall possession spell F1?

The adopted dense model already uses capped square-root duration weights for
complete possessions. This study adds one fixed multiplier of two specifically
to complete subsecond H/A spans, on both original-provider and SoccerNet TRAIN
records. The multiplier acts only on conditional H/A cross-entropy. Each
window divides by the multiplier mass over known H/A frames; the existing
duration factors stay in the numerator. With the multiplier disabled, the
training path must reproduce D2 loss and model updates.

Spans touching a record boundary, time discontinuity or unknown target are
censored and receive no extra weight. Eligibility uses native labels and frame
rates before mapping to the sampled training grid. H and A follow identical
rules. Evaluation and inference receive no eligibility or duration targets.

The architecture, sampler, training budget, D loss, transition loss and cost-3
decoder stay fixed. Six candidate fits cover three full-provider exclusions
and seeds 2026/2027, each with 2,048 updates. The six reconstructed D2 controls
are restored from the previous targeted-study archive, without refitting.
Seeds are also averaged as native probabilities before decoding.

## Evaluation and stop rule

The paired cohort contains five previously exposed development matches and
three views per match: clean and two fixed synthetic feed-like corruptions.
All learned components and normalization exclude the evaluated provider.
Official spell F1 averages matching F1 across IoU thresholds 0.3–0.7. Macro F1
breaks exact ties; the 70/30 summary is descriptive.

The existing +0.01 clean screen, feed-like floor of control minus 0.01, and
provider clean macro/D-F1 floors of control minus 0.02 remain unchanged.
Diagnostics distinguish wrong-team frame errors within ≤1-second reference
spells from complete ≤1-second spell recall at IoU 0.5 and the established
<3-second recall metric. Better short recall alone is not an upgrade.

No weight, threshold or seed sweep follows a negative result under this study
ID. Another experiment requires a distinct hypothesis and declaration. Prior
decoder, contact, native-resolution and boundary-weighting failures are recorded
in the [experiment index](../docs/research/EXPERIMENTS.md#settled-directions).

## Evidence limits

This is development evidence on reused matches, with synthetic stress and
SoccerNet-derived silver labels. The controls are the verified reconstruction
described in [the targeted follow-ups](DENSE_TARGETED_FOLLOWUPS.md), not the
unavailable original research checkpoint bytes. No organizer labels are read,
no public scoring gate is reopened, and this study does not promote a model.

## Training coverage and verification

Eligible subsecond labels are scarce even before fitting. Counts below describe
the loaded TRAIN records and views, not unique frames in a new evaluation set.

| Training contribution | Known sampled H/A frames | Eligible ≤1s sampled frames | Eligible windows / all windows |
| --- | ---: | ---: | ---: |
| Original providers, Metrica excluded | 574,080 | 4,092 | 1,168 / 25,976 |
| Original providers, SkillCorner excluded | 314,932 | 3,196 | 904 / 11,396 |
| Original providers, Sportec excluded | 527,948 | 2,056 | 632 / 23,748 |
| SoccerNet TRAIN, each fold | 688,426 | 638 | 190 / 22,238 |

The added multiplier cannot supply missing short-possession examples. This
coverage limits the interpretation of a negative result and the likely value
of repeatedly tuning a duration weight on the same corpus.

The [frozen declaration](results/dense_short_team_v1_declaration.json) binds
training inputs, six teacher checkpoint/receipt pairs, reconstructed controls,
evaluation caches, source, baseline manifest and decision policy. All 30 saved
control members and 15 ensembles passed hash, native-array, probability-mean
and decoder checks. The restored diagnostics reproduce 36.71% clean and 44.29%
feed-like wrong-team rates within ≤1-second possessions.

With the multiplier disabled, a two-update real-data contract reproduces H/A
and D weights exactly. A synthetic nontrivial short-spell contract confirms
that the multiplier changes H/A updates while leaving the D path identical.
All 479 experiment tests, 26 integrity tests, file-backed fitting parity,
documentation checks and the bundled 199,319-row CSV format check pass.
The initial broad test run lacked pinned PyArrow; installing `pyarrow==20.0.0`
resolved its three dependency errors. No organizer scoring was involved.

## Results and delivery

Retain `joint_gru_sn_dense_ensemble_v1`. All six candidate fits completed at
2,048 updates, taking about four minutes each on this runtime. The 90 paired
score rows cover both arms, both seeds and their probability ensembles, with
15 cells per arm/member. The direct candidate loses overall spell F1 in both
views and misses the clean improvement screen; the feed-like floor and all
provider clean macro/D guards pass.

| Probability ensemble | Clean spell | Clean macro | Feed-like spell | Feed-like macro |
| --- | ---: | ---: | ---: | ---: |
| Reused D2 control | 0.647215 | 0.857677 | 0.505811 | 0.815535 |
| Subsecond H/A weight ×2 | 0.645165 | 0.857879 | 0.503646 | 0.814516 |
| Candidate minus control | −0.002050 | +0.000202 | −0.002164 | −0.001020 |
| Fixed 50/50 ensemble blend | 0.646614 | 0.857883 | 0.504806 | 0.815048 |
| Blend minus control | −0.000601 | +0.000206 | −0.001005 | −0.000488 |

The [readout](results/dense_short_team_v1_readout.json) retains unrounded
scores, provider guards, seed ranges, counts and prediction hashes. Its
[evaluation seal](results/dense_short_team_v1_evaluation_seal.json) binds the
six candidate checkpoints before scoring.

| Diagnostic, equal-provider mean | Control clean | Candidate clean | Control feed-like | Candidate feed-like |
| --- | ---: | ---: | ---: | ---: |
| Wrong-team rate within ≤1s reference possessions | 36.7118% | 36.4586% | 44.2935% | 44.8222% |
| False-D rate within ≤1s reference possessions | 15.1527% | 15.1527% | 16.4291% | 16.2324% |
| Recall of reference possessions <3s, IoU 0.5 | 0.402475 | 0.396971 | 0.217097 | 0.218803 |
| Predicted possessions | 360.667 | 364.222 | 318.444 | 321.222 |

These duration definitions and denominators differ: the first two rows are
frame-error rates, while recall uses one-to-one spell matching. Prediction
counts are provider-balanced means, not raw corpus totals. Clean complete
≤1s matching loses two Metrica possessions (13/33 to 11/33), remains 0/62 on
SkillCorner and 3/44 on Sportec. Under the two feed-like replicas combined,
Metrica gains one match (7/66 to 8/66), while SkillCorner remains 0/124 and
Sportec 5/88. This is not a broad improvement in short-possession recovery.

The ensemble clean spell deltas are −0.006058 Metrica, −0.001953 SkillCorner
and +0.001862 Sportec. Every provider's feed-like spell score declines.
Individual seed deltas are +0.002395/+0.001951 clean/feed-like for 2026 and
+0.000528/−0.002232 for 2027. Both individual clean scores improve, but averaging
their probabilities produces a worse ensemble. An average of seed scores
would conceal this result.

The [sampling exposure audit](results/dense_short_team_exposure_v1.json)
replays all 2,048 batches for each fit and exactly matches every recorded
source/match/game exposure counter. Only 301–321 of 32,768 GAR draws per fit
contain eligible short frames, contributing about 0.09–0.10% of sampled H/A
rows. Original-provider contributions are about 0.39–0.98%. Reweighting those
same examples does not increase their diversity.

The [runtime control replay](results/dense_short_team_v1_runtime_control_replay.json)
reruns one restored control through current inference: maximum probability
difference 3.58 × 10⁻⁷, with zero changed labels. The transient two-update
real-data training check was not saved as an artifact; the committed synthetic
parity test is the durable executable contract. Full-fit D arrays differ
slightly from the archived controls, so D preservation is a loss/architecture
claim, not bit-identical checkpoint bytes. Joint decoding can also change D
labels when H/A emissions change.
The [independent review](results/dense_short_team_v1_independent_review.json)
verifies all 90 native archive hashes and recomputes all 15 candidate ensemble
score cells to 10⁻¹² tolerance. Across all six fits, the maximum D-parameter
difference is 5.51 × 10⁻⁷; native P(D) differs by at most 1.85 × 10⁻⁶.

## Final bounded follow-up

After seeing the direct comparison, `dense_short_team_blend_v1` asks whether
one fixed 50/50 probability blend of the control and candidate ensembles can
retain their complementary evidence. It uses the same 15 cells and decoder,
adds no fits, and keeps the existing clean/stress/provider screens. This is
an explicitly post-result hypothesis with a new declaration, not another arm
added to the completed weighting study. No blend-weight sweep follows it.

The direct result, sparse training exposure and previous settled short-spell
failures do not justify another duration-weight or decoder sweep. The blend
was the final low-cost check before stopping this line on the current data.

The [blend declaration](results/dense_short_team_blend_v1_declaration.json)
was frozen before its 15-cell comparison. Its
[readout](results/dense_short_team_blend_v1_readout.json) shows clean/feed-like
spell F1 of **0.646614/0.504806**, both below control. It fails the +0.01 clean
screen; the other guards pass. The ≤1s wrong-team rate is 36.5906% clean and
44.9416% feed-like: a small clean improvement and a larger stress regression.
Clean complete ≤1s matching remains 14/139 versus control's 16/139; stressed
matching is 12/278 for both. These raw counts do not average provider rates.
The blend's separate `complete_short_spell_readout` pools counts within each
provider; its `<3s` ratios differ from the equal-match rates in `summaries`.
Use `summaries` for protocol-matched comparisons.
The [independent blend review](results/dense_short_team_blend_v1_independent_review.json)
verifies all 30 source and 15 output archive hashes, exact probability blending,
decoder replay and all 15 official score cells. It also supplies the balanced
short-spell diagnostic alongside the frozen pooled-count diagnostic.

No more fits, weights or decoder settings were tried. Both fixed comparisons
lose overall spell F1, while the short-attribution diagnostic remains mixed.
Sparse supervision is a plausible limit, not a proven cause of the negative
result. Another sweep would reuse the same exposed matches and examples;
targeted external short-possession supervision and independent dense labels
offer a stronger next research direction. The adopted manifest, decision
policy, deployment outputs and organizer scoring gates remain unchanged.

## Reproduction and artifact state

The immutable source and input bindings are in the two declarations and the
first study's evaluation seal. Commands run from the repository root with the
pinned experiment environment:

```sh
python -m experiments.dense_short_team_v1 preflight
python -m experiments.dense_short_team_v1 freeze
# Once per declared fold and seed; existing fits are never overwritten.
python -m experiments.dense_short_team_v1 fit --fold leave_out_metrica --seed 2026
python -m experiments.dense_short_team_eval_v1 freeze
python -m experiments.dense_short_team_eval_v1 evaluate
python -m experiments.dense_short_team_exposure_v1 run
python -m experiments.dense_short_team_blend_v1 declare
python -m experiments.dense_short_team_blend_v1 evaluate
```

The other fit folds are `leave_out_skillcorner` and `leave_out_sportec`; each
uses seeds 2026 and 2027. Restore the sealed artifacts for readout instead of
rerunning completed commands. Code and aggregate evidence are merged in
[PR #50](https://github.com/jekkiechan/more-than-a-hack-2027/pull/50).

Private [mtah-data PR #9](https://github.com/jekkiechan/mtah-data/pull/9) holds
six candidate models, six fit receipts, 45 candidate and 15 blend probability
archives, logs, diagnostics and the source/results snapshot at public commit
`c674648`. A fresh remote checkout with a separate LFS store verified all
122 archive entries. Restoration to an empty directory recovered 118 mapped
files (311,153,361 bytes) with zero missing files or size/hash mismatches.
The [remote verification receipt](results/dense_short_team_v1_remote_verification.json)
pins archive commit `89a7c47`, manifest hashes and dependency selections.
Large unchanged inputs remain pinned to the earlier private manifests; these
were revalidated against the existing restore receipt, not downloaded again.
Use the archive's README and `scripts/dense_short_team_v1_cache.py` for restore.

The restricted [research site](https://more-than-a-hack-progress.buffhen.chatgpt.site/experiments.html#subsecond-team)
is published as version 11. Desktop/mobile layout, navigation, metric selectors,
copy links and aggregate downloads passed checks; its audience is unchanged.
The [publication receipt](results/dense_short_team_v1_site_publication.json)
records source/version/deployment identity and the QA font-loading limitation.
Delivery receipts follow the pinned research snapshot so archive hashes remain
non-circular. No new deployment model is created.
