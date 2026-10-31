# Current research state

Reviewed 2026-10-31 against fork `main` through
[PR #48](https://github.com/jekkiechan/more-than-a-hack-2027/pull/48), with the
completed source-head evidence on open [PR #47](https://github.com/jekkiechan/more-than-a-hack-2027/pull/47).
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

1. Execute the [targeted dense follow-ups](../../experiments/DENSE_TARGETED_FOLLOWUPS.md):
   current error audit, fixed missing-ball team memory, confidence-dependent
   decoding, and silver-label transition weighting. These comparisons are
   running; no improvement or promotion is established. Exact submission replay
   has passed. Missing dense research checkpoints require explicitly declared
   matched control reconstruction, not pooled deployment substitutes.
2. The source-specific head comparison on open PR #47 is completed and negative:
   retain the shared dense heads. Its provider-definition audit supports holding
   possession through contested touches; it does not justify rebuilding GAR
   labels. Whole-stoppage model comparison remains blocked on continuous aligned
   tracking. See the [PR #47 report](https://github.com/jekkiechan/more-than-a-hack-2027/blob/6e93eb2218a533b8017ff50f284f655021eec9e6/experiments/SOURCE_CONDITIONED_HEADS_REPORT.md).
3. Fresh external dense possession evidence remains optional stronger
   confirmation. The [independent-data search](../../experiments/INDEPENDENT_DATA_SEARCH.md)
   acquired a SoccerTrack panorama sample; the
   [blind annotation proposal](../../experiments/SHORT_SPELL_NEXT.md) still needs
   independent H/A/D/X labels. Event-interpolated ball paths are not measured
   observations. Sample acquisition and GAR silver labels do not supply
   independent confirmation.

Before another local sweep, consult the index's
[settled directions](EXPERIMENTS.md#settled-directions).

## Artifacts and open work

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
