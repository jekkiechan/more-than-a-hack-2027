# Current research state

Reviewed 4 November 2026 against fork `main` through
[PR #82](https://github.com/jekkiechan/more-than-a-hack-2027/pull/82), plus the
completed v2 target and weight-edit studies below. The user-authorized phase-average promotion
remains the adopted package.
Check newer merged/open PRs before current-results advice. Detailed historical
results belong in [EXPERIMENTS](EXPERIMENTS.md) and the linked reports.

## Adopted package

The user authorized **original_full with phase averaging** on 1 November 2026:
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1`. This supersedes the
phase0 promotion on PR #70. The [promotion report](../../experiments/ORIGINAL_FULL_PHASE_PROMOTION.md)
and [active manifest](../../experiments/baseline_manifest.json) own delivery and
inference. All four checkpoint hashes are unchanged; five-phase averaging is
now enabled. No fitting or organizer rescoring is part of promotion.

Original clean/v1/v2 spell: **.667556/.531576/.583571**. Exact-package GAR TEST
clean/v2 spell: **.701641/.701289** on ten games. All seven historical active
guards pass; the +.01 clean screen remains failed with a user-authorized
exception. Calibrated+phase leads the earlier exact-package TEST comparison;
the paired TEST phase-gain interval includes zero. Independent continuous-match
gold confirmation remains pending.

The evaluation policy now uses clean plus `feed_like_v2` with 4+ realizations
for original-provider development. Keep GAR TEST separate from selected-on
VALID and record future reuse. V1 and composite remain historical diagnostics.

## V2 spell target — 3 November 2026

The completed [v2 target studies](../../experiments/V2_SPELL_TARGET.md) reach
**0.592749 eligible v2 spell F1**, versus adopted **0.583571** (+0.009178).
Clean spell is **0.665355** versus **0.667556** (-0.002201). The **0.60 target
remains unmet**. The winner blends the new four-view original-possession model
and archived calibrated model 50/50, averages native phases, and uses H/A
transition cost 1.5. Common clean/provider retention checks pass; the historical
clean-gain screen remains failed and adoption is unchanged.

Six fixed provider-excluded fits use the eligible original TRAIN pool and
45 GAR TRAIN games. Nine consensus arms, 109 calibration arms, three new-model
arms, three fixed decoder-transfer arms and a final 20-arm boundary check use
the same five exposed development matches, clean plus four fixed v2 views.
All eight archived C0 spell/macro aggregates replay with zero error. Final
local validation passes 639 experiment tests and 26 root tests. The H/A-cost-2
numerical maximum is 0.592926 but fails the fixed clean floor; it is not chosen.

New views, research weights and native outputs are locally verified, not newly
uploaded. No pooled deployment fit, organizer scoring, GAR TEST reuse or
default change occurs. An exact pooled-package check and independent
continuous-match gold remain outstanding evidence needs. The report preserves
failed attempts, mechanical runtime corrections and the disclosed recovery
receipt preservation exception.

Research code and aggregate evidence are on
[PR #82](https://github.com/jekkiechan/more-than-a-hack-2027/pull/82).
Research site **version 29** is published with desktop/mobile checks,
all six chart-control combinations and seven comparison rows. Restricted
access remains at revision 2 with the same allowlist. The
[delivery receipt](../../experiments/results/v2_spell_target_delivery.json)
records the native publication result and local-only artifact availability.

## Direct GRU weight-edit follow-up — 4 November 2026

The [weight-edit study](../../experiments/V2_WEIGHT_EDIT.md) in open [PR #83](https://github.com/jekkiechan/more-than-a-hack-2027/pull/83) reconstructed
six provider-held-out folds from PR82’s frozen recipe because the prior local
checkpoints were not available in the reachable archive. Recipe-level parity is
declared; byte identity is not claimed. A TRAIN-only, match-balanced delayed
transition direction and bounded update-gate edits were compared against the
unedited reconstruction on matched clean, v1, and v2 development views. The
unedited control reproduces the historical .665355 clean / .592749 v2 blend
(v2 exact to its archived aggregate); the 25-cell full-cohort control
reproduces all four saved clean/v2 spell and macro aggregates with zero error,
and independent official replay computes metrics from every new cell artifact. The primary seven-arm comparison selects
`update_bias_p025` at .664316 clean / .594110 v2 spell, +.001360 v2 and -.001039
clean versus unedited; all fixed guards pass. The .60 target remains .005890
away. This is a fixed development selection, not an adoption.

The post-selection v1 diagnostic is .518850 spell / .823766 macro, versus
.518637 / .823302 unedited and .531576 / .826853 adopted. The edit itself gains
+.000213 spell versus its matched unedited reconstruction; the -.012726 adopted
gap is a whole-pipeline comparison, not an isolated edit effect. Two pooled fits
are sealed. The one-use public diagnostic returns .594286 spell / .852078 macro
versus adopted .611765 / .849473 on the fixed window (199,319 rows; 7,500
graded; 17 reference spells). No unedited-new-pooled public control was scored,
so this external difference cannot be attributed specifically to the tensor
edit; it is diagnostic only and not full-match confirmation. Neither v1 nor
public results rerank the frozen v2 winner. The ten-game GAR readout is .703713
clean / .700806 v2 spell versus adopted .701641 / .701289; its v2 paired
interval crosses zero. All 50 GAR prediction sets and aggregates pass final
independent replay. This external comparison is whole-pipeline, with no
unedited-new-pooled control. The study is complete; adoption is unchanged. PR82’s
historical .592749 v2 / .665355 clean result remains unchanged, the .60 target
remains unmet, and adoption is unchanged.

## V2 data and capacity follow-up — 4 November 2026

The [data/capacity study](../../experiments/V2_DATA_CAPACITY.md) completed all
18 provider-held-out fits and 105 scoring jobs across 35 exposed development
cells. The width-96 PR82-style blend reproduces the eligible v2 spell score
(.592749), but falls below adopted clean and v1 spell. Width 192 reaches
.592533 v2 spell (−.000216 versus its matched width-96 blend); the balanced
v1/v2 width-96 recipe reaches .591496 (−.001253). Across the 12 candidate arms,
**0 pass all 19 strict checks**. The D-mask arm preserves D metrics but does
not solve the H/A and all-view retention tradeoffs. The archived controls
replay all 35 cells and 24 official aggregate checks, with maximum error
2.22e-16; the independent audit verified the original 35-cell result.

A separately declared final check uniformly averaged the three already-trained
experts across the same four pipeline types, without new fits or inference. It
also has **0/4 strict passes**; the best v2 spell is .592585, .000165 below the
PR82 threshold, and clean/v1 regressions remain. Across the original and final
checks, **0/16 candidate pipelines qualify**. Verification of this final
uniform result replayed all 140 arm/cell scalars, recomputed all four means and
the 19 gates, and confirmed no inference was repeated. Both audits pass; no
further search is planned.

No package or manifest changed. The .60 stretch target remains unmet. Evidence
is limited to the five previously exposed matches and synthetic feed-like
views; there is no GAR TEST, organizer or independent continuous-match gold
result. New private weights remain local. Automatic review initially rejected both
publication attempts before any remote mutation; that historical state is
preserved in the [delivery receipt](../../experiments/results/v2_data_capacity_delivery.json).
The user has since explicitly authorized both destinations. GitHub branch
`research/v2-data-capacity-v1` is pushed at `2764bba` and [PR #84](https://github.com/jekkiechan/more-than-a-hack-2027/pull/84)
is open; no CI checks had started at Site deployment. The Site deployment
succeeded as version 31 at the [research progress site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/),
from source commit `c9d66174510a7a2415648caa5b937b6f8fda6d3a`. It retains the
custom audience, revision 2 and one external viewer. The
[publication receipt](../../experiments/results/v2_data_capacity_publication_v1.json)
records deployment and static checks. Browser QA was unavailable; no
desktop/mobile visual check is claimed. New private weights and frame outputs
were not published; model adoption is unchanged.

## New external confidence check — 2 November 2026

[PCBAS validation](../../experiments/PCBAS_CONFIDENCE_V1.md) acquired and audited
three broadcast validation matches (six halves), then evaluated the frozen
adopted package with all ball input missing. Primary Drive team agreement is
**71.66% equal-game** (1,616/2,239 pooled correct), above each game's majority
direction-alias reference. Pass/Cross/Shot agreement is 70.96%/60.98%/56.84%.
Predicted dead play at live cross/shot anchors is 35.35%/40.00% pooled, a material
failure pattern in this severe missing-ball regime. No fitting or adoption change.

These are sparse action anchors, not continuous possession gold or spell F1.
Coordinates use a declared 105×68 m standardization; imputed off-screen tracks
are masked, replay events excluded, and full fixture identities remain unknown.
The three VAL matches are now exposed. New independent continuous labels and
observed-ball inputs remain the priority; no top-20 probability is established.
Raw assets/predictions are locally verified only, with no new remote upload.

## Expanded PCBAS evaluation — 2 November 2026

The [twelve-game continuation](../../experiments/PCBAS_EXPANDED_ANCHOR_V2.md)
is complete on twelve deterministic additional TRAIN games, with 36 games kept
unscored. The unchanged submission package reaches **71.57% equal-game Drive
agreement**, 95% whole-game bootstrap interval **68.95–74.10%**, across 8,071
primary anchors. All 96 class/half cells and 23,039 source-event joins independently
verify. No fitting, parameter search, organizer scoring or promotion occurred.

This corroborates the earlier three-game diagnostic on more matches; it does
not establish continuous possession spell accuracy. Cross/shot live-anchor
predicted-D rates remain 29.60%/33.45%; pooled low-visibility performance is weak
and class mixtures differ. Independent continuous H/A/D/X labels and measured
ball inputs remain the priority. The twelve games are now exposed diagnostics,
not a fresh source for future selection claims. Source anonymity limits exhaustive
fixture deduplication and intervals describe this source only.

Code and aggregate evidence are published in
[PR #78](https://github.com/jekkiechan/more-than-a-hack-2027/pull/78).
Research site **version 28** is published after explicit user approval, with all
12 desktop/mobile page checks and 16 chart-control combinations passing. Access
revision 2 and the same two external viewers are unchanged. Native scoped Git
source access and Sites deployment resolve the earlier missing-helper blocker.
The [final publication receipt](../../experiments/results/pcbas_expanded_anchor_v2_publication_v2.json)
records the observed state. Raw data and native predictions remain locally
verified only; no raw-data upload is claimed.

## Bridge and package check on new PCBAS/GAR evidence — 2 November 2026

Two frozen, fit-free studies close the main questions the PCBAS results raised.

- [GAR no-ball bridge](../../experiments/GAR_NOBALL_BRIDGE_V1.md): removing the
  ball on the ten GAR TEST games drops adopted pass-anchor agreement to **67.5%**
  versus PCBAS **68.8%**, consistent by the declared rule (within 5 points). The
  PCBAS cross/shot dead-play pattern reproduces on GAR without ball (shot predicted
  D 20%→38%). Conclusion: PCBAS measures player-only inference, not a 2023–24
  cohort transfer penalty. The ball is worth **+.211 spell F1 (95% CI +.180 to
  +.238)** on GAR. Clean parity with `gar_deployment_eval_v2` is exact. Per the
  declared rule, PCBAS H/A training is not prioritized.
- [PCBAS package comparison](../../experiments/PCBAS_PACKAGE_COMPARISON_V1.md):
  on 18 never-scored reserve games, adopted Drive agreement is **71.97%**
  (69.40–74.43%), replicating the 3- and 12-game results. Calibrated+phase minus
  adopted is **−0.45 points (−0.86, −0.02)**; the adopted package wins in 12 of 18
  games. No case for switching packages. The last 18 PCBAS TRAIN games stay sealed.

## Latest delivery

The phase-average promotion and eval-policy update are merged in
[PR #74](https://github.com/jekkiechan/more-than-a-hack-2027/pull/74) with passing
CI. Research site **version 25** is published; all 12 desktop/mobile page checks
and 18 chart-control combinations pass. The audience remains restricted at
revision 2. Following explicit user upload approval on 2 November, private
[PR #18](https://github.com/jekkiechan/mtah-data/pull/18) is merged. All 399 entries /
7,978,448 bytes pass independent remote download and empty-directory restore;
generic and versioned defaults select the promoted phase-average package. The
[baseline guide](../../experiments/BASELINE.md#private-cache-restore) gives direct
restore/inference commands without the previous directory override.
The [promotion report](../../experiments/ORIGINAL_FULL_PHASE_PROMOTION.md#delivery-state)
owns the latest delivery state; earlier version receipts below remain historical.

The exact-package GAR comparison and original-provider phase guard audit are
complete on PR #73: 19 match evaluations, 470 official scalar replays with zero
error, all seven active guards passing and both component seeds improving.
The [complete scorecard](../../experiments/ORIGINAL_FULL_PHASE_AVERAGE.md) records
Astra's provisional phase-upgrade recommendation and the still-failed historical
clean screen. Research site **version 24** is published with desktop/mobile
interaction checks and unchanged restricted access revision 2. New native
prediction artifacts are locally verified; no new remote-native upload is
claimed. No model, default or adoption policy changes.

The original-full default CLI replays the already-sealed candidate exactly, with
199,319 valid rows. Private delivery PR #17 is merged; all **213 entries /
6,417,817 bytes** pass independent remote fetch and empty restore. Research site
**version 22** is published with desktop/mobile interaction checks and unchanged
restricted access revision 2. The [promotion report](../../experiments/ORIGINAL_FULL_PROMOTION.md)
links the separate verification and publication receipts.

The [stacking study](../../experiments/SMART_STACKING_V1.md) is merged on
[PR #68](https://github.com/jekkiechan/more-than-a-hack-2027/pull/68) with passing
CI. All 495 independent replays pass; its combined suite has 568 tests. Research
site version **21** was published with desktop/mobile checks and restricted
access revision 2 unchanged. The stack improves its historical adopted control,
but original diversity is now selected for adoption. No pooled stack is installed.
The [publication receipt](../../experiments/results/smart_stacking_v1_publication.json)
supersedes the preserved blocked snapshot. New stack native artifacts remain locally
verified; no new remote stack-native upload or organizer scoring occurred.

The fixed public diagnostic is published on [PR #66](https://github.com/jekkiechan/more-than-a-hack-2027/pull/66)
and research site version **20**, with desktop/mobile checks and restricted
access revision 2 unchanged. The two pooled models, full CSVs and sealed native
outputs are preserved in a locally verified 22-file archive (14,774,217 bytes).
The user subsequently authorized private upload. All 22 entries pass an
independent remote fetch and empty-directory restore with exact hashes and a
separate LFS store. The
[remote verification receipt](../../experiments/results/short_quality_public_v1_remote_verification.json)
supersedes the earlier
[local-only delivery snapshot](../../experiments/results/short_quality_public_v1_delivery.json),
which is preserved unchanged.
The previous research archive below remains remotely verified.


Research code and aggregate evidence are on [PR #65](https://github.com/jekkiechan/more-than-a-hack-2027/pull/65).
All **559 new archive/source entries / 1,714,557,333 bytes** pass an independent
remote fetch and empty-directory restore with exact hashes and a separate LFS
store. Research site version **19** is published; desktop/mobile chart, data,
navigation and download checks pass, with restricted access revision 2 unchanged.
The [delivery receipt](../../experiments/results/short_quality_diversity_v1_delivery.json)
and [publication receipt](../../experiments/results/short_quality_diversity_v1_site_publication.json)
record the observed states. Archive identifiers and exact restore instructions
remain private. No model promotion or organizer scoring occurred.

## Next work

The user-requested [phase-average readout](../../experiments/ORIGINAL_FULL_PHASE_AVERAGE.md)
contains the completed exact-package comparison and original-provider guard
audit. Astra now supports provisional phase averaging as an incremental upgrade
to original_full; all seven active guards pass and both component seeds improve.
TEST clean/v2 spell is .701641/.701289 versus .698836/.699133 phase0; v2 gain is
positive in 8/10 games, but its paired 95% interval includes zero. Calibrated+phase
still leads TEST spell (.704846/.704118). VALID's nine selected-on games stay
separate. All 380 GAR and 90 guard cells replay officially with zero scalar
error. No additional experiment is required before the adoption decision.
The historical +.01 clean screen failed. The subsequent user-authorized promotion
records the new exception and enables phase averaging; no new public scoring.
Research site version 24 publishes the completed scorecard, TEST/VALID results,
guard/seed checks and final Astra review. Desktop/mobile controls and layouts pass.
Restricted access revision 2 and the same two external viewers are preserved.
The user explicitly approved the scoped-credential fallback after the initial
automatic-review rejection. The [publication receipt](../../experiments/results/original_full_phase_average_publication_v2.json)
records the resolved delivery state.

The [challenge-calibrated feed study](../../experiments/FEED_LIKE_V2.md) replaces the
state-blind feed-like view with `feed_like_v2` (fidelity error .72 → .09 on full-pitch
providers). Under v2 the previous calibrated-dense baseline scores .5836 (.5880 with
5 Hz phase TTA) versus .5785 for the now-adopted `original_full` recipe; raw-feed
training (.5784) also does not transfer. The v1-feed advantage behind the promotion
does not hold on v2. Use v2 with 4+ realizations as the stress guardrail.

Prioritize independent continuous-match gold labels and corroborated short-possession
supervision. Phase TTA's fixed follow-up is complete and the user has authorized its
versioned promotion; prioritize new independent gold evidence next.
The numerical comparisons below retain each completed study's historical
baseline. The adopted recipe is now original diversity; PR #68's stack remains
an archived comparator, with no pooled deployment package.


The [stacking comparison](../../experiments/SMART_STACKING_V1.md) completes
24 meta-models, zero base refits and 495 paired original score cells.
Best guard-passing learned stack: **.661814/.525872 clean/feed-like spell**,
versus adopted .658625/.518527 and original-diversity .662590/.528180.
It retains raw-feed D; learning the D combination instead reduces clean/feed
spell to .638640/.513886 and fails guards. Uniform averaging reaches .523795
feed. No stack beats the strongest single recipe, so the predeclared new-view
and expert-fallback checks are skipped. It remains a credible replacement
candidate for adopted: higher clean/feed spell, both seeds improve feed, and
feed-first guards pass. Original diversity is now adopted instead; failing the stricter study-win rule
does not negate the stack's measured gains against the historical baseline.
No adoption or organizer scoring.
All base, teacher, normalization and meta-label components exclude the target
provider; other-provider OOF arrays are deliberately not reused for training.
Stop this bounded weight/context comparison. Stronger error-complementarity
evidence and independent match evaluation are more informative than another
sweep on these five exposed matches.

The [feed-like benchmark audit](../../experiments/FEED_BENCHMARK_AUDIT_V1.md) finds
small decoder/arm-combination ceilings (+.005/+.0076 per-cell oracles) and shows that
synthetic feed-like gaps ignore game state, unlike the real feed. Under
challenge-matched gaps the `original_full` candidate beats the adopted model in 12/12
realizations (SkillCorner trails); 5 Hz phase TTA is a small, consistent gain. Prefer a
pre-declared multi-realization benchmark over the two original replicas for selection.

The separately requested [fixed public diagnostic](../../experiments/SHORT_QUALITY_PUBLIC_V1.md)
is complete after two pooled fits. Public spell is **.611765** versus adopted
.630303; macro **.851742** versus .864181. The full CSV replays/validation and
reference-free artifact checks pass. The one-use gate is closed, with no
public-driven tuning, repairs or model promotion. External selection results
below remain unchanged; this is a 17-reference diagnostic, not new independent
or full-match confirmation.


The [short-label quality and diversity comparison](../../experiments/SHORT_QUALITY_DIVERSITY_V1.md)
is complete: **18 fixed fits** and eight fixed arms. Best eligible original
feed-like spell is **.528180**, clean **.662590**, versus the previous numerical
leader .527396/.657215. All guards pass; the historical +.01 clean screen fails.
The aggregate feed gain comes from Sportec; Metrica and SkillCorner decline.
The fixed new-corruption check reaches .547221 versus matched leader .546583;
these are different realizations, not a replacement for the original benchmark
or independent matches. All 420 new score cells pass separate official/algebra
replay. The .55 original-benchmark target remains unmet; adoption is unchanged.

The TRAIN audit finds only **38 corroborated subsecond GAR spans out of 198**,
and 403 complete <=3s spans across all 45 games. Original pools contain 515–948
physical short spans per fold; paired views do not add examples. Prioritize more
varied, externally corroborated short supervision and independent dense
evaluation. Stop the declared sampling comparisons without new rate, duration,
blend-weight or decoder settings. The earlier
[short-window sampling and calibration study](../../experiments/FEED_SHORT_SAMPLING_V1.md)
remains the .527396 comparator.

The completed [raw-feed training study](../../experiments/FEED_ROBUSTNESS_V1.md) reaches **0.525240** original feed-like spell F1 versus .518527 promoted. Twelve fits, fixed blends/component checks and fresh corruption checks are complete; 0.55 remains unmet and adoption is unchanged. The two-seed fresh score is .533871 versus matched baseline .524936. Prioritize short-possession supervision/exposure and independent dense evaluation; fresh synthetic views are not new matches.

1. The [feed-first calibration and pooling study](../../experiments/FEED_SPELL_PUSH.md)
   is complete: five rounds, 522 declared configurations (including controls).
   D-logit temperature 4 with the original cost-3 decoder (`f062` / `g001`)
   leads feed-like spell F1 at **.518527**, with clean **.658625**, versus adopted
   .505811/.647215. Every provider and both seeds improve spell in both views;
   all guards and the historical +.01 clean screen pass. Two-provider selection
   reaches .513611 feed-like in the broad round. Pooling alternatives, filtering,
   uncertainty transforms and the local refinement do not beat the simple rule.
   This is tuned evidence from the same exposed matches and synthetic stress.
   After selection, the user-requested fixed public diagnostic ties adopted spell
   at .630303 and raises macro .851387 → .864181, D-F1 .843424 → .873502
   on 17 references. The user subsequently authorized promotion; f062 is now
   the default package under a versioned manifest. Promotion replays the fixed
   candidate without public-driven tuning or rescoring. Fresh dense evaluation or new
   corruption realizations are more informative than another local temperature sweep.
   The [earlier refinement](../../experiments/DECODER_REFINEMENT_V3.md) and its
   separately frozen public diagnostic remain historical evidence for those candidates.
2. The [subsecond-team follow-up](../../experiments/DENSE_SHORT_TEAM_V1.md)
   is complete: six fixed 2× H/A-weighted fits and a fixed 50/50 ensemble blend
   both reduce clean/feed-like spell F1. Short-team errors improve marginally
   clean but worsen under stress. Exact sampler replay shows only 0.09–0.10%
   of sampled GAR H/A rows qualify. Stop duration/blend-weight sweeps on these
   exposed matches; prioritize targeted external supervision and independent
   dense evaluation. This follows the negative
   [targeted dense comparisons](../../experiments/DENSE_TARGETED_FOLLOWUPS.md).
   The adopted model and closed organizer gates remain unchanged.
3. The source-specific head comparison on open PR #47 is completed and negative:
   retain the shared dense heads. Its provider-definition audit supports holding
   possession through contested touches; it does not justify rebuilding GAR
   labels. Whole-stoppage model comparison remains blocked on continuous aligned
   tracking. See the [PR #47 report](https://github.com/jekkiechan/more-than-a-hack-2027/blob/6e93eb2218a533b8017ff50f284f655021eec9e6/experiments/SOURCE_CONDITIONED_HEADS_REPORT.md).
4. Fresh external dense possession evidence remains optional stronger
   confirmation. The [independent-data search](../../experiments/INDEPENDENT_DATA_SEARCH.md)
   acquired a SoccerTrack panorama sample; the
   [blind annotation proposal](../../experiments/SHORT_SPELL_NEXT.md) still needs
   independent H/A/D/X labels. Event-interpolated ball paths are not measured
   observations. Sample acquisition and GAR silver labels do not supply
   independent confirmation.

Before another local sweep, consult the index's
[settled directions](EXPERIMENTS.md#settled-directions).

## Artifacts and open work

- **Historical calibrated dense promotion:** its manifest/policy were version 8; the four checkpoint hashes are unchanged. The baseline CLI matches the sealed candidate byte for byte and format validation passes. Public promotion PR #60 and private cache PR #14 are merged; all 191 package/source entries pass a fresh remote restore. Site version 16 is published with the new adoption timeline, pipeline and desktop/mobile QA. The [promotion report](../../experiments/CALIBRATED_DENSE_PROMOTION.md) owns current delivery and restore state.


- **Feed-first iteration:** five rounds / 522 declared configurations, 7,830 ensemble cells
  and 360 seed cells complete. The fixed public diagnostic scores one unique CSV
  once and ties adopted spell F1 at .630303, with higher macro and D-F1. Separate scalar replays pass. The refinement's
  unsupported aggregate-field lookup failed before candidate scoring; its
  frozen runner/lock are preserved with a bounded execution amendment.
  Code/results are merged in public PR #58; private PR #13 has 1,176 entries
  verified by fresh remote download and empty-directory restore. Research site
  version 15 is published with unchanged restricted access and desktop/mobile QA.
  [Report, checks and artifact state](../../experiments/FEED_SPELL_PUSH.md).


- **Decoder refinement:** 118 arms and the fixed public diagnostic are complete;
  `r069` was its strongest external candidate, superseded by the adopted temperature-4 rule. Private PR #12 has 432
  fresh-restore hash-verified entries. Site version 14 is published with external-first
  interpretation and desktop/mobile QA. The user authorized this publication and
  future experiment-report updates to the same existing restricted audience.
  [Report and receipts](../../experiments/DECODER_REFINEMENT_V3.md#delivery-and-reproduction-state).

- **Decoder search delivery:** code/results merged in PR #54; private PR #11
  has 295 remotely restored, hash-verified entries. The experimental full CSV
  passes format validation. Research site version 12 was published after explicit
  user approval; the previous approval blocker is resolved. [Report and receipts](../../experiments/DECODER_SEARCH_V2.md#artifact-and-site-delivery).

- **Spell-F1 postprocessing (2026-11-01):** fixed posterior expected-spell-F1
  decoding completed all 45 paired cells. Ensemble clean/feed-like changes
  −.004235/−.002601; both individual seeds also decline. Retain cost-3 Viterbi;
  no sample-budget or seed sweep follows. The result is included in published research site version 12. [Report](../../experiments/SPELL_MBR_V1.md).

- **Postprocessing runtime (2026-11-01):** optional decoder reduces synthetic
  199,319-frame decoding time by 16%, with independent numerical parity tests.
  Not integrated or promoted; real-input parity and end-to-end timing remain.
  Included in published research site version 12.
  [Report](../../experiments/POSTPROCESSING_RUNTIME_V1.md).

- **Subsecond team attribution:** six fits and the fixed blend are complete;
  neither improves the adopted ensemble. Code/results are merged in PR #50.
  Private [mtah-data PR #9](https://github.com/jekkiechan/mtah-data/pull/9)
  delivers 122 verified archive entries and 118 successfully restored mapped
  files. The restricted research site is published as version 11.
  [Report and delivery receipts](../../experiments/DENSE_SHORT_TEAM_V1.md#reproduction-and-artifact-state).

- **Targeted dense follow-ups:** all 12 fits and three comparisons are complete;
  reconstructed controls reproduce saved clean/feed-like scores within 3 × 10⁻¹¹.
  Input byte-hash differences remain documented. Private archive delivery is complete: all 338 entries verified remotely and
  235 unique original-path files restored without mismatches.
  [mtah-data PR #8](https://github.com/jekkiechan/mtah-data/pull/8);
  [coordinated report](../../experiments/DENSE_TARGETED_FOLLOWUPS.md).

- **Previous dense deployment:** private [mtah-data PR #7](https://github.com/jekkiechan/mtah-data/pull/7)
  is merged at `19ee280`. This session restored and verified all four versioned
  checkpoint hashes and reproduced the sealed full CSV from raw ATD inputs.
  [Submission receipt](../../experiments/dense_submission_audit_v1/report.json).
  The deployment package does not include the original dense research fold
  checkpoints or their probability arrays.
- **Previous capacity deployment:** verified remote recovery is complete;
  [receipt](../../experiments/results/baseline_soccernet_capacity_remote_artifacts.json).
  This does not establish remote availability of every research fold checkpoint.
- **Matched training follow-up:** completed, no upgrade;
  [report](../../experiments/SOCCERNET_FOLLOWUP.md) owns replay and artifact state.
  Its reconstructed reference is distinct from the later exact deployment restore.
- **Open evidence:** [PR #35](https://github.com/jekkiechan/more-than-a-hack-2027/pull/35)
  contains completed label-free gap, possession-rate and adaptation comparisons,
  all reported no go against the former combined-objective baseline. It is
  unmerged (head `143d010`), not evidence against the new dense recipe.
- Draft PRs #8, #11, #13, #16, #18 and #25 remain open at review. Find pinned
  evidence in the index and inspect live heads before resuming. A draft's
  presence does not imply its code is absent from later merged dependencies.

Update this file when adoption, priorities or blockers change. Put scorecards,
per-study receipts and historical decisions in their reports/index, rather than
appending another completed-study handoff here.
