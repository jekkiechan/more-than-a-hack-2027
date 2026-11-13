# Current research state

## Gated boundary refiner — 13 November 2026

[The follow-up](../../experiments/LGBM_BOUNDARY_GATE_V1.md) scores 43 fixed
deployment rules for the LightGBM dead-ball refiner on development; 15 pass all
guards. Confident boundary shifts (threshold .6) keep 63% of the v2 spell gain
(**+.0093**) and leave public spell F1 unchanged at **.623529** in one sealed
user-requested regression check. On exact-package GAR TEST10 it lowers spell F1
in nine of ten games: clean **−.0039**, v2 **−.0051**, both intervals below
zero. The predeclared GAR rule fails.

Keep reconstructed Fable adopted; manifest and policy are unchanged. The
development v2 gain did not transfer: the refiner adds dead ball on GAR and
removes it on the challenge match. A D-to-live-only gated arm on GAR is the one
open follow-up and needs its own declaration. The progress site has not been
updated for this study.

## LightGBM spell retention — 12 November 2026

The [bounded development follow-up](../../experiments/LGBM_SPELL_RETENTION_V1.md)
completes 18 provider-held-out refiner fits and six fixed strategies. Conservative
D-end updates improve spell and D-F1 in clean/v1/v2: v2 spell **.598690 → .600820**,
D-F1 **.869770 → .875536**. This retains only 21.95% of the unrestricted refiner’s
v2 D gain; historical-v1 short recall falls .001576. The two-ended adapter also
fails only v1 short recall. Full/equal-view refiners retain v1 macro/D failures;
missingness ablation loses clean/v2 short recall. Every ensemble is ineligible.

Keep reconstructed Fable adopted. No public/GAR scoring or pooled deployment fit
occurs in this follow-up; public spell retention remains unestablished. The five
matches are exposed development data with non-nested upstream features. All 630
saved candidate rows and control aggregates pass independent replay. All 1,000
experiment tests pass apart from one skip, plus 26 root tests and file-backed
parity. Next localize short spells harmed by D-end movements and test missingness
transfer with fixed label-independent masks before another bounded correction.

Delivery: the 646-member private archive passes a fresh isolated remote restore.
The [site publication receipt](../../experiments/results/lgbm_spell_retention_v1_site_publication.json)
records version 45 with unchanged restricted audience and passing desktop/mobile checks.

## Pooled LightGBM public diagnostic — 12 November 2026

The [user-requested public check](../../experiments/LGBM_PUBLIC_V1.md) is complete.
One fixed pooled, block-safe adaptation of `R2_full_w0.5` was fitted on 532,637
rows from the 35 exposed development cells. Baseline full-CSV replay is exact;
source/model/input/prediction seals precede the exclusive public gate. Public
spell changes **.623529 → .569697**, macro **.864234 → .886569**, and D-F1
**.889372 → .924716**. D precision improves but recall falls; predicted
possessions are 16 versus 17 references. The same 17-possession window is a
diagnostic only. No tuning, repeat scoring or promotion follows.

Keep reconstructed Fable adopted. The original development gain justifies a
label-independent missingness stress test, recommended by the parallel Luna
reviews; v1 error localization is the second proposed experiment. Neither was
run during the public-first task. This pooled adaptation has no matched
development/GAR evaluation, and its upstream training/deployment probabilities
differ in fold scope. The original v1 macro guard remains failed.

Delivery: [PR #94](https://github.com/jekkiechan/more-than-a-hack-2027/pull/94),
private 369-member archive verified from a fresh isolated remote restore, and
research site version 44 published with unchanged restricted audience.

Current-main correction: PR #92 and PR #93 are merged as of this review
(`81d013e`); historical sections below preserve their earlier branch statuses.

## V2 target push — 12 November 2026

[The v2 target push study](../../experiments/V2_TARGET_PUSH_V1.md) scores 37
declared arms against the adopted Fable `snap_m0.0` control. Decoder re-tuning,
H/A cost and nearest-player H/A evidence give no eligible arm. A provider-held-out
LightGBM dead-ball refiner blended 50/50 with the existing P(D) reaches
clean/v1/v2 spell **.6932/.5392/.6135** with short recall up in every view; it
fails only the v1 macro retention check (−.0050 against a .001 allowance), so no
arm is eligible under the repository guards. Ablations show about two thirds of
the v2 gain comes from ball-missingness features, which `feed_like_v2` places
with development labels; without them v2 is .6036 and short recall fails.
The adopted package is unchanged. The refiner is fit on the exposed development
matches themselves; no pooled deployment fit, GAR or public check was run, and
the research site was not updated from this session.

## Reconstructed deployment verification — 11 November 2026

The originally missing pooled CE checkpoints were not recovered. Instead, the
frozen pooled recipe was rerun as a separately declared four-fit reconstruction.
Sampler/exposure/view checks and D/normalization state match exactly; maximum
loss-interval difference is `1.816079021e-08` against the frozen `1e-7` bound.
The independent receipt checks 220 input entries and 48 source entries with no
mismatches. All 52 regenerated v2 feature files and 52 sidecars match historical
bytes. Its preparation receipt records one worker rather than two; normalizing
only that metadata field reproduces the historical receipt hash, while the
actual receipt remains unchanged. The reconstructed weights are not byte-
identical to the originals. See the
[pooled reconstruction report](../../experiments/FRONTIER_POOLED_RECONSTRUCTION.md).

The reference-free 199,319-row control CSV matches byte-for-byte. On the fixed
public diagnostic, Fable changes spell/macro/D-F1 from
`.611765/.851285/.862870` to `.623529/.864234/.889372`; this reused
7,500-frame/17-possession window was not used for tuning. The fixed GAR TEST10
replay covers 50 paired cells, 100 arrays and 800 metric comparisons. Clean
spell/macro/D-F1/short recall are `.709186/.784717/.593243/.264521` versus
`.708496/.780580/.583474/.263398` control; v2 is
`.709176/.784865/.596512/.263570` versus
`.708619/.782179/.589851/.262769`. All aggregate means improve, but per-game
spell means include two declines in each view. This is repeated TEST exposure
with silver labels, not independent gold confirmation or a uniform per-game
guarantee. VALID remains separate; no tuning followed.

The guarded default CLI replay also matches the sealed Fable CSV byte-for-byte
(SHA-256 `56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f`).
Post-promotion checks pass: 978 experiment tests, 977 passed and one skipped;
all 26 root tests, file-backed fit parity, submission format and the independent
reconstruction audit pass.

All ten deployment roles in the reconstructed package were hash-verified from a
fresh private remote clone with empty LFS; only the two pooled CE files differ
from the earlier package. After the fit, CSV, public, GAR and archive guards
passed, the guarded promotion runner installed the reconstructed Fable manifest
and policy on this research branch. The
[promotion receipt](../../experiments/results/ball_exit_reconstructed_v1.json)
binds those records; the [final verification](../../experiments/results/ball_exit_reconstructed_v1_verification.json)
records the default CLI replay and post-promotion checks. PR #92 is not merged;
this is the working default on the research branch, not a change to main.

The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
is published as version 41 with the same restricted audience. Its
[publication receipt](../../experiments/results/ball_exit_reconstructed_v1_site_publication.json)
records the source, deployment and desktop/mobile checks.

## Earlier software and source verification — 11 November 2026

The user requested verification before final promotion, with Luna subagents.
[The new verification receipt](../../experiments/results/ball_exit_verification_v2.json)
records 964 experiment tests: 963 pass and one skips, with no failures/errors.
The former five errors are fixed by isolated synthetic contract fixtures;
frozen production artifacts and public gates remain intact. All 26 root tests,
file-backed parity and CSV format pass. Luna independently replays 140 saved
development outputs / 840 metrics with zero error and verifies all source hashes.
All four pinned GAR TEST raw files are restored and hash-verified.

At the time of this earlier receipt, exact inference and GAR/public checks were
blocked on the original pooled stage-1024 CE checkpoints. Their identities
remain in the historical handoff below. The subsequent reconstruction and
fixed-check results above resolve the executable-package blocker without
restoring those original bytes.

## Complementary live-team improvement — 11 November 2026

[The complementary comparison](../../experiments/BALL_EXIT_COMPLEMENT.md)
retains a small development candidate: apply the existing .75 hard-turnover /
.25 fresh-continuation H/A blend only on live intervals whose boundaries snap
leaves unchanged. Clean/v1/v2 spell becomes **.681677/.534829/.598780**;
v2 gain over Fable is **+.000090**. All declared retention checks pass, with the
small v2 mean gain concentrated in SkillCorner. Direct composition reaches a
higher .599644 but fails clean/short-recall checks. Both rounds are independently
verified; neither is a pooled deployment or GAR/public evaluation. Retain this
as research evidence and stop the bounded composition search. Fable was provisionally adopted at that point; its reconstructed pooled package
and final research-branch promotion are recorded above.
The user approved delivery. [PR #92](https://github.com/jekkiechan/more-than-a-hack-2027/pull/92)
publishes the implementation and aggregate evidence. The restricted research site
was previously updated to version 40; the [verification publication receipt](../../experiments/results/ball_exit_verification_v2_site_publication.json)
records that earlier publication separately from the later reconstruction checks.

## Ball-exit reconstructed promotion — 11 November 2026

[The ball-exit follow-up](../../experiments/BALL_EXIT_PROMOTION.md) independently
reproduces Fable's snap result: clean/v1/v2 spell **.681508/.534731/.598690**.
Two predeclared crossing-time extensions fail v2 spell and short-recall retention;
retain unchanged snap. All 140 saved prediction sets and 840 metrics replay
with zero error. This remains exposed development evidence, not GAR/public
validation.

The user delegated the choice: “promote the version you think is best.”
**The active manifest and policy now select reconstructed Fable snap_m0.0.**
The [adoption receipt](../../experiments/results/ball_exit_adoption_v1.json)
records an explicit simplicity/deployment-cost tradeoff against the complementary
candidate’s tiny +.000090 exposed-development gain and extra pooled-model needs.

At the time of adoption, the exact pooled `ce_2026.joblib` and `ce_2027.joblib`
were missing; the reconstruction now supplies a separately identified package.
The new SkillCorner fold upload is a different experiment. The guarded runner
installed the reconstructed model package and its manifest/policy on the
research branch after fixed checks passed. No fitting on TEST/public labels or
closed-gate reset occurred.

Reviewed 11 November against main `8919726`, Fable PR #91 at `9a4e7f0`, and
[PR #92](https://github.com/jekkiechan/more-than-a-hack-2027/pull/92).
The predecessor manifest/policy and all frozen reports remain archived.

## Adopted package

`joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1` is the
working default on the research branch. It retains CE/calibrated/predecessor H/A
at 0.5/0.375/0.125, paired-seed
native-phase pooling and H/A cost 3. Fable corrects D starts toward observed
pitch exits using the unchanged zero-margin snap rule. Exposed development
clean/v1/v2 spell is **.681508/.534731/.598690**; macro is
**.876204/.827310/.854915**. Clean gains +.011371 over frontier and passes the
+.01 screen. The >.60 v2 target remains unmet; SkillCorner regresses, and no
independent continuous-match gold confirmation is claimed.

The predecessor’s GAR TEST10 `.708496/.708619` clean/v2 spell and public
`.611765` spell/`.851285` macro are controls. Reconstructed GAR means are
`.709186/.709176` clean/v2 spell and the public diagnostic is `.623529` spell /
`.864234` macro; see the verification above for full metrics and limitations.
The predecessor’s TEST short-recall declines remain disclosed in its
[promotion report](../../experiments/SPELL_FRONTIER_PROMOTION.md).
Restore and continuation commands are in [BASELINE](../../experiments/BASELINE.md)
and the [Fable report](../../experiments/BALL_EXIT_PROMOTION.md).

## Turnover supervision — one retained development candidate

[The turnover study](../../experiments/TURNOVER_SUPERVISION_V1.md) continues
from the approved frontier with 18 matched provider-excluded fits. Stable
turnover sampling raises original-window event exposure from 38.72% to 54.11%.
Hard-event sampling improves clean/v1 spell to .672138/.533505 and retains
all three short-recall means, but v2 .593466 fails the strict improvement
check. Uniform sampling also loses v2 short recall. Fresh continuation reaches
.594751 v2 but loses v1 short recall. No training arm is retained.

The baseline audit finds weak/reversed team probabilities on at least one
side of 90.6% of missed eligible turnovers. This motivates supervision work,
but the sampling comparison shows that greater exposure alone does not
improve the primary stress metric. A separately declared three-mixture
hard/fresh follow-up retains **75% hard / 25% fresh CE**: clean/v1/v2 spell
**.670848/.533247/.594002**, with all short-recall means improving and all
retention checks passing. V2 macro falls .000093 within the .001 allowance;
SkillCorner v2 spell falls .000246. The gain is small on exposed development
matches, and CE-seed diagnostics show sensitivity to pooling.

Both rounds independently replay all saved scores and probabilities; turnover
recovery itself improves only six repeated cell-events versus approved
(2,550 vs 2,544 of 6,552). Retain the blend as an experimental candidate; it is
not part of the adopted package. This earlier turnover study had no pooled
counterpart or GAR VALID/TEST allocation. Stop the declared weight search and
seek independent continuous-match gold evidence.

## Ball-exit dead-ball correction — one retained development candidate

[The 11 November study](../../experiments/BALL_EXIT_DMASK_V1.md) edits only the
frozen D mask: a predicted dead-ball run starts where the ball is observed
leaving the pitch. `snap_m0.0` reaches clean/v1/v2 spell
**.681508/.534731/.598690** (+.011371/+.002602/+.005071) with short recall up
in every view and all retention checks passing. Sportec and Metrica improve in
every v2 cell; SkillCorner declines because its silver D starts before the ball
leaves. **The >.60 v2 target remains unmet by .0013.** Its fixed GAR/public
checks are now reported in the reconstructed-promotion evidence above; seek
independent continuous-match gold confirmation.

## SkillCorner 2019/20 real-feed windows — complete, negative

[The 11 November study](../../experiments/SC2020_REALFEED_V1.md) adds six unused
SkillCorner 2019/20 broadcast-tracking TRAIN matches to the frontier CE
continuation. Across 18 fits and 14 declared candidates v2 spell stays at
.591410–.594575 versus .593619; none passes retention and the data effect is
within draw noise. The adopted package and the unmet >.60 target are unchanged.

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

The [turnover archive](../../experiments/results/turnover_supervision_private_delivery.json)
is uploaded to private `mtah-data:research/turnover-supervision-v1-parts`
at `f1de198`. A fresh isolated LFS download verifies both byte-preserving
parts, the combined archive hash and all 159 members, including 18 checkpoints
and 70 saved evaluation cells. Parts avoid GitHub's per-object size limit.
The [site publication](../../experiments/results/turnover_supervision_site_publication.json)
is complete as version 37, with its restricted audience preserved.

The prior promotion recorded all ten deployment checkpoint roles, matching
source, manifests/policy and aggregate provenance in a verified local archive,
including fresh extraction and an isolated ten-model load. That historical
receipt does not establish availability in every later workspace. The remote
default still describes the historical phase-average package.
[Current package instructions](../../experiments/BASELINE.md)
and [local archive receipt](../../experiments/results/baseline_spell_frontier_local_artifacts.json)
own that package's availability. The current-frontier postprocessing report
separately records reconstructed research fold checkpoints, exact control replay,
and native prediction-cache delivery. These are research holdout artifacts,
not a replacement pooled deployment package. Frozen training seals remain intact.

The 160-file research archive is now uploaded to private `mtah-data` branch
`research/frontier-recovery-v1` at `ff3d8a1`, bound to public source `c2f8761`.
A fresh isolated LFS download verifies the archive and all member hashes.
[Remote availability receipt](../../experiments/results/frontier_postprocess_private_delivery.json).

The promotion report preserves aborted GAR attempts, the seed-description
erratum, six-phase compatibility correction and exact runtime guards. Old
runners, declarations, decisions and scoring gates remain intact. Use the
recorded pre-activation revision to replay historical hash-bound studies.

The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
was published as version 36 with the same restricted audience. That historical update
passes desktop/mobile checks for Overview and Experiments and all 36 first-round
chart combinations; the second-round failure is shown separately.
[PR #87](https://github.com/jekkiechan/more-than-a-hack-2027/pull/87) carries the
completed experiments and promotion; merging follows the existing CI policy.
[Draft PR #88](https://github.com/jekkiechan/more-than-a-hack-2027/pull/88) carries
the subsequent matched postprocessing results and remains stacked on #87.
[Historical publication receipt](../../experiments/results/frontier_postprocess_site_publication.json);
[prior delivery receipt](../../experiments/results/v2_extra_data_v1_delivery.json).
