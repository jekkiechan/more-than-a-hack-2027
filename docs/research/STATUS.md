# Current research state

Reviewed 2026-11-01 against fork `main` through
[PR #64](https://github.com/jekkiechan/more-than-a-hack-2027/pull/64), plus the
completed [short-label quality and diversity study](../../experiments/SHORT_QUALITY_DIVERSITY_V1.md)
on [PR #65](https://github.com/jekkiechan/more-than-a-hack-2027/pull/65).
Check newer merged/open PRs before current-results advice. Detailed historical
results belong in [EXPERIMENTS](EXPERIMENTS.md) and the linked reports.

## Adopted package

`joint_gru_sn_dense_calibrated_ensemble_v1`: the same two dense-SoccerNet H/A GRUs and fine-tuned D paths, now using the frozen temperature-4 D-logit transform and cost-3 decoder. Explicitly user-adopted on 1 November 2026. The [baseline guide](../../experiments/BASELINE.md) owns restore/inference; the [promotion report](../../experiments/CALIBRATED_DENSE_PROMOTION.md) and receipt bind the change.

Matched clean/feed-like spell is .658625/.518527 versus previous .647215/.505811. Both seeds and every provider improve spell in both views; all guards and +.01 clean screen pass. Public spell ties .630303, macro improves to .864181 and D-F1 to .873502. Promotion performs no fitting or organizer rescoring and reproduces the sealed full CSV exactly. Historical failed screens and closed gates are preserved. Independent dense confirmation and full-match organizer performance remain unknown.

## Latest delivery

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

- **Calibrated dense promotion:** active manifest/policy are version 8; the four checkpoint hashes are unchanged. The baseline CLI matches the sealed candidate byte for byte and format validation passes. Public promotion PR #60 and private cache PR #14 are merged; all 191 package/source entries pass a fresh remote restore. Site version 16 is published with the new adoption timeline, pipeline and desktop/mobile QA. The [promotion report](../../experiments/CALIBRATED_DENSE_PROMOTION.md) owns current delivery and restore state.


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
