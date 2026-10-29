# Does the current GRU benefit from SkillCorner training data?

Status: completed; retain the active dataset and baseline. Date: 29 October 2026.

This study tests removing the original eight SkillCorner training matches from
the current combined-objective GRU. It is separate from the completed
[expansion study](SKILLCORNER_EXPANSION.md), which compared eight versus fourteen
SkillCorner matches. The active thirteen-match dataset and model remain unchanged
throughout this comparison. There is no automatic promotion or new organizer
public scoring in this task.

## Question and fixed comparison

The control is `joint_gru_combined_ensemble_v1`, trained on Sportec three matches,
Metrica two matches and SkillCorner eight matches. The candidate removes all
eight SkillCorner training matches. Both use the same hidden24 player-only-D
architecture, combined loss, augmentation, seeds 2026/2027, normalization method,
eight epochs, batch size 64, ten-second cores, five-second context and cost-3
decoder. Each learned normalization is fitted on its own training records only.

| Evaluation protocol | Control training | Candidate training | Draws per epoch / updates |
| --- | --- | --- | ---: |
| Sportec provider held out | Metrica 2 + SkillCorner 8 | Metrica 2 | 23,748 / 2,976 |
| Metrica provider held out | Sportec 3 + SkillCorner 8 | Sportec 3 | 25,976 / 3,248 |
| SkillCorner provider held out | Sportec 3 + Metrica 2 | Same saved control | Unchanged parity |
| All providers available, separate evaluation matches | Original 13 | Sportec 3 + Metrica 2 | 30,560 / 3,824 |

Six candidate fits are declared: two seeds for each changed provider fold and
the all-provider training setup. The existing SkillCorner-held-out checkpoints
already exclude SkillCorner and are reused exactly. Their natural training
budget is 11,396 windows per epoch and 1,432 updates; they cannot replace the new
matched-budget five-match fits in the final row.

The original five evaluation matches stay fixed: Sportec J03WOY, Metrica Game 3,
and SkillCorner 1996435/1996436/2006229. Each has its original clean view and two
fixed `feed_like` stress replicates. Both arms use identical inputs, grading,
unknown labels and time breaks. All five evaluation matches are excluded from
every training setup. The three historical Sportec test exclusions remain fixed.

The main scorecard preserves the original five-match, full-provider protocol and
its control spell F1 .633205 clean / .498830 feed-like. Source contribution is
also reported separately over Sportec and Metrica: only these provider-holdout
models change. SkillCorner primary scores are parity and cannot establish a
benefit from SkillCorner training. The all-provider supplement uses the same five
held-out matches and is reported separately from full provider exclusion.

## Budget and interpretation

The smaller training sets repeat complete seeded shuffled window permutations,
then a fresh shuffled remainder, until each epoch reaches the original control
budget. For unique-window count N and budget B, each window appears either
floor(B/N) or floor(B/N)+1 times per epoch. The equality path preserves the
original DataLoader and its random-number use exactly. Actual draws, cycles,
remainder, batch counts and optimizer updates are audited, rather than assuming
that equal epoch counts imply equal compute. Frozen training libraries and the
previous expansion sampler are not modified.

This tests the practical effect of removing a source at fixed optimizer cost.
It necessarily changes source diversity, repeated exposure and the normalized
loss mass of retained providers. For example, Sportec holdout changes from
equal Metrica/SkillCorner mass to Metrica alone; all-provider training changes
from thirds to equal Sportec/Metrica halves. The existing nominal source/match weighting
formula remains fixed. Each optimizer step normalizes its sampled batch, so
nominal balance is not a guarantee of exactly equal realized loss mass per epoch. This is not a marginal label-quality experiment holding
the remaining providers' loss contributions constant.

## Frozen screen and evidence limits

All deltas mean candidate without SkillCorner minus current model. A practical
case to recommend removal requires original-five full-provider clean spell gain
at least .01, feed-like loss no worse than .01, and each provider's clean macro
and D-F1 loss no worse than .02. Because the SkillCorner provider is unchanged,
the +.01 five-match equal-provider screen implies +.015 on the two changed
providers' mean. The treated Sportec/Metrica effect is always shown directly.

The all-provider supplement must additionally improve clean spell strictly on
the Sportec/Metrica equal-provider mean, with feed-like loss no worse than .01
and each of those providers' clean macro/D-F1 loss no worse than .02. Its
SkillCorner and complete-five scores are descriptive: lost specialization to
broadcast data is not itself a veto on a competition-focused removal proposal.

Failing this screen does not prove SkillCorner helps. Raw effects, provider
spread, both seeds and the actual probability ensemble determine whether the
development evidence favours inclusion, removal or remains mixed. Existing
matches are previously exposed development data; synthetic feed-like stress is
not fresh tactical-feed confirmation. Each tactical provider contributes only
one evaluation match. Full-match competition performance remains unknown.

SkillCorner has 10 Hz broadcast tracking, partial visibility, event-derived
silver targets and visibility filtering that removes most dead-ball frames.
Sportec and Metrica also have source-specific dead-ball shortcuts. All native
masks, unknown-target margins and period/time boundaries remain intact.

## Verification and delivery

Before fitting, freeze source/configuration/input/cache/control hashes, exact
cohorts, dependency versions, sampling and criteria. Verify cache receipts,
source exclusions and contracts for sampler cycles, equality parity, target
independence and error restoration. Seal all six fitted checkpoints before
evaluating either protocol.

The declared comparison comprises thirty paired cells and 180 seed/ensemble
score rows. Independent verification will check identities, hashes, training-only
normalization, actual budgets, native probabilities, equal probability means,
cost-3 predictions, official spell/frame metrics, historical control parity and
aggregation. The final readout will record tests actually run, all failed guards
and actual local/remote artifact availability. No prior private-upload rejection
will be bypassed.

The frozen execution commit is `c4dcbe3decf50d29e7d722c20ca379aeff45ce5c`;
its declaration SHA-256 is
`dcd2e6c4518599d8d3aa95aba8ddfceefc5fa9a32a74f5fbf0445b9e8f91a72c`.
All six checkpoints were sealed before scoring (seal SHA-256
`348a880e8b31dd14e5226e956d20b34139f935ad4818fbc5eac56a9988b94fc8`).
The [pre-fit](results/skillcorner_removal_prefit_verification.json) and
[fit verification](results/skillcorner_removal_fit_verification.json) receipts
confirm the cache/control hashes, actual optimizer budgets, model shapes and
bitwise training-only normalizers. The standalone verifier's initial function-
boundary/null-return error was corrected before accepting any full result;
its [correction audit](results/skillcorner_removal_verifier_corrections.json)
preserves the failed invocation and records that frozen code, data and weights
were unchanged.

## Measured outcome

**Retain the active dataset and baseline.** The current recipe with SkillCorner scores better on these excluded-provider holdouts, while removing it benefits some matches when their provider is represented in training. These development results do not establish a competition improvement from deleting SkillCorner. The frozen removal screen fails; no model or dataset promotion occurs.

### Original five-match benchmark

These are the same five matches and full-provider protocol behind the current .6332 headline. Each row compares matched arms; the model family, inference ensemble and decoder are unchanged.

| View | Current recipe | Remove original 8 SkillCorner TRAIN matches | Removal delta |
| --- | ---: | ---: | ---: |
| clean | 0.633205 | 0.625786 | -0.007419 |
| feed_like | 0.498830 | 0.496092 | -0.002737 |

Both component seeds also lose spell F1 in both primary views after removal. On the two changed provider holdouts (Sportec and Metrica), the ensemble removal effect is -.011128 clean and -.004106 feed-like. The three SkillCorner evaluation matches are exact parity and are excluded from that source-contribution mean. This development evidence is consistent with a transfer benefit from the existing SkillCorner corpus under this recipe; it does not establish its effect on the organizer match.

### Separate training setup with providers represented

The current supplement model trains on all three sources; the removal model trains on Sportec and Metrica only. Every evaluation match is excluded from both arms. Its absolute scores belong to a different training protocol and a two-provider aggregation; they do not replace the .6332 benchmark.

| Tactical Sportec/Metrica mean | Current recipe | Remove SkillCorner | Removal delta |
| --- | ---: | ---: | ---: |
| clean | 0.774753 | 0.787806 | +0.013053 |
| feed_like | 0.580867 | 0.593287 | +0.012420 |

The positive tactical mean is concentrated in Sportec. The Metrica match, which is especially relevant to the competition provider, gains only .002720 clean and loses .007082 under feed-like stress. Removing SkillCorner therefore does not consistently improve even this small target-provider check.

| Provider | Full-provider clean / feed-like removal delta | Pooled-protocol clean / feed-like removal delta |
| --- | ---: | ---: |
| sportec | -0.014339 / +0.008319 | +0.023387 / +0.031922 |
| metrica | -0.007917 / -0.016530 | +0.002720 / -0.007082 |
| skillcorner | +0.000000 / +0.000000 | -0.059460 / -0.049990 |

Sportec and Metrica are represented in both pooled arms. SkillCorner is represented only in the current control; its pooled row is a diagnostic loss of that specialization. The all-five pooled diagnostic declines .686720 → .675603 clean and .546481 → .538098 feed-like; this includes lost SkillCorner specialization. The predeclared recommendation does not use that decline as a veto. It remains visible alongside the tactical result.

### Seed and frame-quality diagnostics

| Protocol / cohort | Member | Clean removal delta | Feed-like removal delta |
| --- | --- | ---: | ---: |
| primary / all | 2026 | -0.014672 | -0.006765 |
| primary / all | 2027 | -0.028346 | -0.018093 |
| primary / all | ensemble | -0.007419 | -0.002737 |
| pooled / tactical | 2026 | +0.009129 | +0.007380 |
| pooled / tactical | 2027 | +0.002588 | -0.004822 |
| pooled / tactical | ensemble | +0.013053 | +0.012420 |

Primary Metrica D-F1 improves .010270 clean and .003906 feed-like after removal, despite lower spell F1. Its D recall rises .027041/.026797 while precision falls .009016/.021694. Short-spell recall also rises there, demonstrating that one favorable component does not determine the interval-ranking result. These diagnostics do not isolate a causal mechanism.

Only the original-five primary +.01 clean-spell screen fails; the other thirteen checks pass. A failed deletion recommendation alone would not prove SkillCorner helps. Here the separate raw primary comparisons favor inclusion in both views and both seeds; the provider-represented comparisons show a mixed, provider-dependent effect.

All five IoUs, macro F1, the auxiliary 70/30 summary, D precision/recall/F1, short-spell recall, reference/predicted spell counts, fragmentation and per-match/seed rows are preserved in the [180 rows](results/skillcorner_removal_rows.json) and [readout](results/skillcorner_removal.json).

### Practical conclusion and limits

Keep the original eight SkillCorner training matches in the current baseline. The earlier failed expansion test still says that six additional SkillCorner matches did not earn adoption; it did not establish that the original source was useless. This removal study supplies that missing direct comparison.

Competition usefulness remains uncertain because these are repeatedly exposed development matches, only one tactical match per provider, and synthetic feed-like gaps. The next decisive data investment is an independently labeled match with tactical-camera visibility and real live-ball gaps. The current evidence does not justify automatically deleting SkillCorner or acquiring more of the same source. No new organizer inputs, labels, predictions or scoring gates were used in this study. Historical public diagnostics were already known and were not used to select this candidate.

Verification status: all six model seals passed; the [independent external verification](results/skillcorner_removal_external_verification.json) completed in two documented stages. The repository passed 265 experiment tests and 18 root integrity tests (283 total), exact file-backed synthetic-fit parity and the bundled submission-format check. Two additional root integrity tests were added after the executable freeze; no runtime Python file or candidate manifest changed.

## Independent replay completion

The recorded replay completed all thirty paired cells, 180 decoded/official
metric rows, sixty candidate/parity seed-probability checks, thirty archived
primary-control comparisons, 180 period-reset decoder checks and fifteen
historical primary ensemble rows. A missing constant in the standalone checker
stopped that run at aggregation. After checking that the sealed data, model
and output hashes were unchanged, a separate readout completed all twenty-four
provider/seed summaries and the fourteen frozen guards from the same 180 rows,
without further inference or scoring. This is a resumed verification, not an
uninterrupted full run.

Saved/recomputed probability parity uses absolute tolerance 1e-7; the historical
archive guard uses 1e-6. No numerical tolerance was loosened. The corrected
verifier and routing/CLI contracts are preserved. Its initial control/candidate
routing mistake, function-boundary issues and final readout correction changed
only the standalone verifier; the frozen experiment code, models, predictions
and criteria were unchanged. The [external replay audit](results/skillcorner_removal_external_replay_audit.json)
records the exact traceback and captured checker hash.

Exact source bytes for that captured checker version were not recovered after
its edits. The non-matching reconstruction is explicitly marked non-exact; the
audit preserves this limitation, the captured execution hash and log, and the
corrected reproducible checker. The complete frozen training/inference source
snapshot is intact and all 160 hashes match.

## Local artifact availability

The [artifact receipt](results/skillcorner_removal_artifacts.json) records the
local private-cache commit `6abe1fc20090265c75b3246baf05657d7e3474e5` on
`archive/skillcorner-removal-v1`: 261 new payload files / 233,525,134 bytes and
109 reusable base files / 2,013,304,812 bytes. All 37 new LFS pointer objects
verify by size and SHA-256. The official conflict-safe restore helper verified
all 370 unique files / 2,246,829,946 bytes into an empty local directory.

No private upload was attempted for this study, and no new upload rejection
occurred. The earlier destination-specific authorization restriction remains;
remote availability and remote restore are not claimed. The current baseline
manifest, model files, evaluation manifest and decision policy are unchanged.
From the local archive checkout, restore with:

```sh
python scripts/competition_cache.py restore --manifest skillcorner_removal_v1_cache_manifest.json --data-dir /path/to/code/data
python scripts/competition_cache.py restore --manifest skillcorner_removal_v1_base_dependencies_manifest.json --data-dir /path/to/code/data
```

Use the frozen public execution checkout or the archived 160-file source
snapshot and set `MTAH_DATA_DIR` to that restored data directory. Future
unrelated Python modules change the runner's full-source seal; do not rewrite
the saved declaration to accommodate them. No organizer scoring is needed
to reproduce this external comparison.
