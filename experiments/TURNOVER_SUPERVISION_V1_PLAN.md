# Turnover supervision from the approved spell frontier

## Question and status

This 8 November 2026 study is in preparation. Can TRAIN-only sampling around
stable H/A turnovers improve the approved **.5936187198896373 v2 spell F1**
package, relative to an equally long ordinary continuation?

This is a training-supervision experiment. It changes which original-data
windows receive additional exposure; it does not create or correct labels.
The current package remains adopted until a completed matched comparison
supports a change. No organizer or GAR VALID/TEST labels enter this study.

## Why this experiment

Previous boundary-margin losses, learned turnover decoder costs, and increased
short-possession sampling did not produce a retained upgrade. The
[current-frontier postprocessing study](FRONTIER_POSTPROCESS_V1.md) also found
no retained change across seed weights, boundary adjustment, switch costs and
expert disagreement. These outcomes motivate examining the training signal.

The [TRAIN inventory](results/turnover_supervision_audit.json) finds 2,906
adjacent H/A transitions in 13 original-provider matches. Of these, 2,139 have
at least one second of stable labels on each side, and 1,701 also have that
context inside an existing training core. The 45 GAR TRAIN games contain
5,911 adjacent transitions, including 5,083 with stable one-second context.
These are label-continuity counts, not independently verified possession
boundaries. GAR remains generated silver supervision. We initially target
original-data events to avoid treating more silver transitions as stronger
evidence.

## Bounded design

Three arms start from the same reconstructed stage-1024 CE checkpoint for
each held-out provider and seed:

| Arm | Additional original-data sampling |
| --- | --- |
| `fresh_control` | None; ordinary continuation |
| `uniform_turnover` | Cycle through eligible stable turnovers |
| `hard_turnover` | Cycle through the harder half of eligible turnovers within each match |

Each arm receives 256 updates at learning rate .0001 for three provider
folds and seeds 2026/2027: **18 fits**. Batches retain 16 original and 16 GAR
windows. Existing short-possession sampling, GAR quality sampling, four v2
training realizations, normalization, architecture and loss remain shared.
Both turnover arms independently replace 25% of original window draws with
an event-containing window from the already selected record. The control's
sampler still executes in every arm, preserving its random stream and the GAR
draw schedule. Exposure receipts distinguish its proposals from actual draws.
An independent request stream keeps the override requests paired between the
two turnover arms; event choices use a separate stream. A replacement keeps
the selected record, team orientation and base draw's alternate-feature flag.
It can supersede a proposed short-spell window, so realized short-spell
exposure is measured separately from the unchanged short-sampler proposal rate.

An eligible event is a direct H-to-A or A-to-H transition with uninterrupted
same-team labels for at least one second on either side, within the same
native period/record. Its selected core must contain both contexts. D,
unknown labels and period breaks invalidate context. Eligible transitions
include long spells; they are not restricted to spells shorter than three
seconds. Stable context does not imply that the complete surrounding
possessions are observed.
The implemented eligibility check conservatively requires sampled core
endpoints to cover the entire native context. Its eligible catalog can therefore
be smaller than the inventory's nominal 50-sample time-interval count above.

Hardness is computed once from the frozen fold/seed CE checkpoint on each
physical match's clean TRAIN features. Its score is the smaller of the mean
correct-team conditional probabilities during the preceding and following
one-second windows. Lower scores are harder. The harder half is selected
within each match, with deterministic ties. Paired orientations and corrupted
views share physical event identity and ranking. No development targets enter
mining, and the held-out provider is excluded from every learned component.
This is mining of training errors, not out-of-sample uncertainty estimation.

## Evaluation and retention

The comparison uses the existing five exposed development matches: clean,
two fixed v1 realizations and four fixed v2 realizations, totaling 35 cells.
Realizations are averaged within match, then matches within provider, then
providers equally. All 35 saved approved-baseline scorecards must replay
before candidate scoring. Reconstructed checkpoint serialization differs from
the missing originals; the recovered current package reproduces every saved
baseline metric.

Only the CE expert changes in the approved .5 CE / .375 calibrated / .125
predecessor arithmetic ensemble. Seed and native-phase pooling, H/A switch
cost 3, and exact predecessor D/X masks stay fixed. The official scorer and
grading population remain unchanged.

Retention requires strictly higher v2 spell F1, no clean/v1 spell loss, no
short-spell recall loss in any view, macro F1 within .001 of baseline in every
view, and no provider clean macro/D F1 loss greater than .02. Exact D/X masks,
graded-frame counts and reference-spell counts are invariants. We also compare
each supervision arm against `fresh_control` to separate the sampling effect
from additional updates. Individual-seed results are diagnostics, not an
inference ensemble or independent confirmation.

Source, configuration, input and checkpoint hashes will be frozen before
fitting/scoring. No post-result parameter search is part of this declaration.
The study does not allocate pooled deployment fits, GAR TEST evaluation,
organizer scoring or automatic promotion.

## Limits

The five development matches are repeatedly exposed; synthetic v2 corruption
is model-conditioned. Original TRAIN caches do not retain native timestamps,
so native label context relies on prepared period/record boundaries; sampled
feature records retain the shared time-break validation. Clean TRAIN hardness
may miss corruption-specific failures. Fixed D masks also limit what better
H/A emissions can recover. New independent continuous-match annotations remain
necessary for stronger evidence.
