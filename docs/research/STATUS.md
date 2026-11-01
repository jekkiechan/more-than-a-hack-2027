# Current research state

Reviewed 2026-11-01 against fork `main` through
[PR #57](https://github.com/jekkiechan/more-than-a-hack-2027/pull/57), plus the completed
[feed-first calibration and pooling study](../../experiments/FEED_SPELL_PUSH.md).
Check newer merged/open PRs before current-results advice. Detailed historical
results belong in [EXPERIMENTS](EXPERIMENTS.md) and the linked reports.

## Adopted package

`joint_gru_sn_dense_ensemble_v1`: two width-96 conditional H/A GRUs with
fine-tuned D paths, trained with dense SoccerNet-GAR touch-stream silver labels.
The [baseline guide](../../experiments/BASELINE.md) owns the package, evidence,
restore and inference commands; the active manifest and decision policy are
linked there. The user explicitly approved promotion despite missing the +.01
clean spell screen. The frozen failed screen and closed public diagnostic remain
unchanged. Full-match organizer performance and independent broadcast confirmation
are unknown. All five development matches have prior exposure; feed-like stress
is synthetic, and dense GAR labels are derived silver labels.

## Next work

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
   on 17 references. No public-driven tuning or promotion follows; the manifest
   stays unchanged. An opt-in full CSV
   generator and frozen evidence are supplied. Fresh dense evaluation or new
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
  `r069` remains the strongest external candidate. Private PR #12 has 432
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

- **Dense deployment:** private [mtah-data PR #7](https://github.com/jekkiechan/mtah-data/pull/7)
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
