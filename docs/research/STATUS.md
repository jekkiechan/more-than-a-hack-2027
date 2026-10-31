# Current research state

Reviewed 2026-10-31 against fork `main` at `1b06133` (through PR #48),
plus the completed supervision experiments in [PR #47](https://github.com/jekkiechan/more-than-a-hack-2027/pull/47).
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

1. Retain the adopted dense shared-head package. The completed
   [source-head comparison](../../experiments/SOURCE_CONDITIONED_HEADS_REPORT.md)
   fitted all 18 declared models. Shared control C scores .647215/.505811
   clean/feed-like spell F1; separate H/A heads score .634885/.494700 and
   separate H/A+D heads .635602/.495254. Both miss the feed-like floor and
   clean +.01 screen. Slightly better D frame F1 does not recover possession
   spells. Do not repeat this fixed head split as a promising upgrade.
   The [Sportec TRAIN rule diagnostic](../../experiments/PROVIDER_TOUCH_CALIBRATION.md)
   is complete: hold-through-contest beats literal challenger attribution, while
   a zero-second gap trades more unknown predictions for a small spell gain.
   Keep the GAR rule; no full-provider label calibration or model upgrade is
   established. The incorrect-clock v1 is invalid; corrected v2 uses raw timestamps.
   The [coverage audit](../../experiments/FULL_MATCH_COVERAGE_EXECUTION.md)
   is complete: only 369/4,352 observed OUT runs have a bounded same-record
   restart, at most 4.805 seconds later. The full-match/long-stoppage arm is
   blocked on continuous tracking and complete event timelines, not a model failure.
2. Pursue fresh external dense possession evidence and authorized true-broadcast
   access. The [independent-data search](../../experiments/INDEPENDENT_DATA_SEARCH.md)
   acquired a SoccerTrack panorama sample; the
   [blind annotation proposal](../../experiments/SHORT_SPELL_NEXT.md) still needs
   independent H/A/D/X labels. Event-interpolated ball paths are not measured
   observations. Neither sample acquisition nor GAR silver labels supplies that
   independent confirmation.

Before another local sweep, consult the index's
[settled directions](EXPERIMENTS.md#settled-directions).

## Artifacts and open work

- **Dense deployment:** private [mtah-data PR #7](https://github.com/jekkiechan/mtah-data/pull/7)
  is merged (verified 2026-10-31, delivery head `33bcf56`). The versioned dense
  manifest is delivered; verify the selected manifest rather than assuming
  the generic default names the dense package.
  Verify the package hashes after restore. See BASELINE for commands.
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
