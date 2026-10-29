# Does the current GRU benefit from SkillCorner training data?

Status: implementation and pre-fit verification. Date: 29 October 2026.

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
from thirds to equal Sportec/Metrica halves. The existing source/match weighting
formula remains fixed. This is not a marginal label-quality experiment holding
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
