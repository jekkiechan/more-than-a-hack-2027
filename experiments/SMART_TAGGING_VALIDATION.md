# Smart Tagging validation protocol

**Status: completed, 2026-10-29. Retain the current baseline.** The fixed
Smart Tagging candidate loses spell F1 on the separately requested public
diagnostic. This small-window result does not establish full-match efficacy.

## Question and evidence boundary

The question is whether Metrica SportsCode Smart Tagging can provide useful
inference-time evidence for H/A/D possession prediction. Smart Tagging is an
automatic annotation of tactical phases and set pieces, not possession ground
truth. The official challenge documentation says its tags are generated from
the same video/tracking pipeline, can contain false positives and negatives,
and describe phases that can cover part of, one, or multiple possessions.
`Team=N/A` is present, including for throw-ins. Agreement with these tags alone
cannot establish possession accuracy or a real Metrica tag gain.

The working baseline is `joint_gru_combined_ensemble_v1`: the equal probability
ensemble of combined-objective joint GRUs (seeds 2026/2027), decoded with
Viterbi cost 3. Its development evidence is previously exposed provider
holdouts: clean/feed-like spell F1 .633205/.498830. The prior organizer public
diagnostic is closed; its saved spell F1 is .557576 and is reused without
rescoring. It cannot be used to choose tags, thresholds, features, seeds, or
models. The user separately requested one fixed Smart Tagging diagnostic on
2026-10-29, with predictions sealed before a distinct exclusive public gate.
That descriptive comparison does not reopen any historical gate.
No manual labels or corrections may be made for the organizer match. See
`docs/research/STATUS.md`, `experiments/model_decision_policy.json`, and the
repository `AGENTS.md` for current state and constraints.

## Evidence tiers

### 1. Synthetic contract checks

Test parser and feature contracts using fabricated XML fixtures only: interval
rounding and half-open frame boundaries; interval overlap and ordering; absent,
unknown, and `N/A` team tags; multiple tag groups; halftime/time breaks; and
missing or malformed fields. Verify tag-feature construction never reads H/A/D
targets or the evaluation labels, is deterministic, preserves missingness, and
has a defined output when no tag covers a frame. These checks can establish
implementation behavior, not tag quality or model improvement. Do not create
gold-derived artificial tags and score them as validation; that is circular.

### 2. Genuine XML, label-free coverage and contradiction audit

When the authorized XML is available, record its source identifier, retrieval
status, byte count, and SHA-256 before parsing. Keep the XML and frame-level
derivatives in ignored/private data storage; publish only aggregate counts.
Check the declared clock against documented match/half boundaries, and report
per code and half: interval count and duration, frame coverage, overlap,
missing/unknown team rate, team-labeled coverage, and set-piece coverage. Do not
impute a missing team or turn phase intervals into H/A/D labels.

Separately define a short list of **high-confidence local tracking anchors**
from label-free observations (for example, a clearly observed restart or a
stable same-side controlled sequence). Freeze anchor rules without organizer
reference labels. Report how often tag intervals agree or contradict each
anchor, with counts and unassigned cases. These are consistency diagnostics,
not accuracy estimates: the anchors are sparse, operationally defined, and
share the tracking/video pipeline. Also report agreement between frozen
baseline predictions and tags as model agreement only, never as correctness.
Do not inspect public labels to resolve disagreements.

### 3. Fixed label-free prediction outputs

After the input and feature recipe are frozen, produce one saved output for the
unchanged baseline and, if an implementation exists, a separately identified
tag-informed candidate. Seal code, model, XML, tracking, feature, and prediction
hashes before any reference-label access. Report coverage and prediction
changes by tag availability/type and native ball-observation status. This tier
shows behavior and reproducibility only; it cannot rank models or establish
possession gains without independent H/A/D labels.

Report native clean tracking and the repository's declared `feed_like` stress
as separate views, with identical frames and transformations for matched arms.
Keep `composite` and component ablations diagnostic only under the current
policy. Stratify by observed-ball versus missing-ball frames as a diagnostic;
never treat ball absence as D. Missingness is source-specific: Metrica sample
data historically blanks the ball mostly during dead play, unlike the
challenge, where ball gaps often occur during live play. Thus Metrica-only
tag/ball results can reward a shortcut. `feed_like` is synthetic stress, not
real broadcast confirmation. If no independent labels exist, report no
accuracy metric or efficacy conclusion.

### 4. Efficacy test on a genuinely held-out tagged match

The decisive test requires a separate match with (a) genuine Smart Tagging
created by the same system, (b) tracking on a compatible timebase, and (c)
continuous H/A/D/X intervals independently annotated by two qualified
annotators from video alone. Annotators must be blind to tags, model outputs,
and event/possession feeds. Freeze raw annotations, uncertainty, adjudication,
and the resulting interval labels before exposing the tags or any predictions.
Record annotation agreement and uncertain/ungradable coverage.

Freeze one matched comparison before scoring: the unchanged baseline versus the
same pipeline with a precisely specified tag feature/prior, holding model
budget, training data, seeds, decoder and evaluation frames fixed. For a
whole-pipeline transfer claim, exclude the held-out provider/match from every
learned H/A/D component and normalization; audit prior checkpoint and tagger
exposure. Evaluate once on the held-out match with the official scorer. Report
unrounded spell F1 and macro F1, all five IoU thresholds, per-team results,
reference/predicted spell counts, boundary/short-spell diagnostics, and
stratification by tag coverage and ball missingness. Include clean and
`feed_like` views where the tracking supports the same frozen corruption
protocol; do not characterize either synthetic stress as broadcast evidence.

One independent match supports a pilot only. A general efficacy or adoption
claim needs multiple independent matches and an appropriate match-level
uncertainty analysis, with the current predeclared score ordering and guards.
The active policy ranks unrounded spell F1 first and macro F1 only for an exact
spell tie; the weighted 70/30 summary is auxiliary. Do not reuse the closed
organizer gate as a holdout or alter the policy after observing results.

## Potential external analogues

Tracking- or event-derived phase proxies can help exercise parser interfaces,
missingness handling, and anticipated failure modes. If their boundaries or
team values are constructed from H/A/D gold labels, they are circular and must
not be presented as validation, tagger agreement, or evidence of efficacy.
At most, such inputs belong in synthetic contract fixtures or an explicitly
label-conditioned mechanism stress test with no accuracy claim. Genuine tags
from a different match and independent labels are required for the efficacy
test above.

## Separately requested public diagnostic

The user explicitly asked to see the public-set impact while acknowledging the
no-tuning constraint. One fixed candidate uses the current model and cost-3
decoder, with conditional H/A odds tilted by at most two on missing-ball
frames, a one-second taper, exact official phase codes, neutral unknown/conflict
frames, and bitwise unchanged native P(D). Goals, player-box counts and set
pieces stay neutral. The odds cap and taxonomy were finalized before any new
reference access. No fitting, parameter sweep or baseline promotion occurs.

Commands use `python -m experiments.smart_tagging_check freeze|predict|check
--out data/experiments/smart_tagging_validation_v1_final`. The check verifies
the declaration and prediction seal, then claims the fixed global
`experiments/results/smart_tagging_public_check_started.json` before loading
reference labels. The marker remains closed on failure and blocks alternate
output directories. The baseline full CSV must reproduce its saved hash before
scoring; its saved result is reused. The candidate is scored exactly once with
the official evaluator. Report spell F1, macro F1, predicted-spell count, D
metrics and all five IoUs. This is a direct check on the provided 17-spell
window, not independent confirmation or an estimate of full-match performance.

## Next evidence

An independent separate tagged match is still needed for a credible efficacy
claim and any learned calibration. Continuous blind possession labels and
compatible tracking/tag timebases remain the main external-data requirement.

## Completed readout — 29 October 2026

The single fixed candidate lowers the competition primary metric, so it is not
adopted. No model, prior, cap, taper, taxonomy, decoder, policy or adoption
change follows the public result. This tests one conservative missing-ball
prior; it does not rule out every use of Smart Tagging.

| Public five-minute diagnostic | Saved baseline | Tag candidate | Delta |
| --- | ---: | ---: | ---: |
| Official mean spell F1 | .557576 | .537143 | -.020433 |
| Macro F1 | .854088 | .866346 | +.012258 |
| Auxiliary 70/30 | .646529 | .635904 | -.010625 |
| Predicted possession spells (reference: 17) | 16 | 18 | +2 |
| Dead-ball F1 | .854810 | .854810 | 0 |

The spell loss is consistent with additional unmatched intervals: at IoU .5
both recover ten reference possessions, but spurious predictions rise from
six to eight. IoU .3 improves slightly (.666667 → .685714); IoU .4/.5/.6/.7
fall from .666667/.606061/.545455/.303030 to
.628571/.571429/.514286/.285714. Better per-frame team classification does not
establish better possession recovery. Native P(D) is bitwise preserved; the
public decoded D precision/recall/F1 also happen to remain identical. In general
an H/A emission change can move D boundaries through joint Viterbi decoding.

The official candidate 90% spell bootstrap interval is [.388571, .685714].
The saved baseline interval is [.387879, .727273]. These separate intervals
are not a paired significance test. Only 7,500 frames and 17 reference spells
are graded; the full-match score and independent confirmation remain unknown.
This is a user-requested, previously exposed public diagnostic, not an external
clean/feed-like comparison. No new feed-like corruption or external accuracy
study ran, and no provider-robustness claim follows.

### Genuine-tag audit

The authorized XML has 638 intervals. The fixed taxonomy maps 334 attacking
and 140 defending intervals; 36 have unmapped teams and 128 are deliberately
neutral/ignored. Mapped phases cover 93,391 of 158,857 graded frames (58.79%).
They intersect 22,718 of 64,643 missing-ball frames (35.14%). After taper and
conflict neutralization, directional evidence remains on 21,322 missing-ball
frames (32.98%). There are 1,163 opposing-tag overlap frames among missing-ball
frames. The prior changes 1,378 raw H/A/D argmax decisions on the full graded
match; this is not the count of decoded changes or correct decisions.

The separately declared raw-phase shadow, without the missing-ball eligibility
gate, agrees with clear observed tracking-contact anchors on 15,446 of 19,009
supported frames (81.26%). These are correlated frames from the same automatic
pipeline, and anchors describe proximity rather than independently adjudicated
control. This consistency statistic is not accuracy or gain evidence. Native
missing-ball frames have no observed contact anchor by construction.

### Seals, verification and artifacts

Luna subagents reviewed integration semantics and validation, implemented the
adapter/audit and their adversarial tests, and independently reviewed the
preflight checker. The baseline's full 199,319-row replay reproduces the saved
scored CSV SHA-256 `a9beb9ddee36182f926e4d693f648c19f4a19c55afd1653ea93e19f3dd32634b`.
The actual input and both member weight hashes match their existing private
cache manifests. Both full-length CSVs validate before reference access.

Before any new reference access, review found two operational issues: the
readout needed to map saved `predicted_spells` to evaluator `n_pred_spells`, and
the CLI needed to resolve relative output paths before sealing. Original and
intermediate declarations, partial outputs and original checker snapshots are
preserved. The final declaration is a replacement preparation record, not a
new candidate. All four original/final probability and CSV artifacts are
byte-identical. The model, prior settings, inputs and weights never changed.
The global scoring gate was consumed only once, after the final prediction
seal; no earlier scoring attempt or reference read occurred.

- [Final declaration](results/smart_tagging_declaration.json),
  [prediction seal](results/smart_tagging_prediction_seal.json),
  [pre-reference amendments](results/smart_tagging_pre_reference_amendment.json).
- [Official public result](results/smart_tagging_public_check.json),
  [closed global gate](results/smart_tagging_public_check_started.json).
- [Label-free audit](results/smart_tagging_label_free_audit.json),
  [audit declaration](results/smart_tagging_audit_declaration.json).
- [Independent verification](results/smart_tagging_verification.json),
  [local artifact inventory](results/smart_tagging_artifacts.json).

The private XML, source tracking, weights and frame-level outputs stay under
ignored `data/` or the existing private cache. New output artifacts are locally
verified only; no new private upload or fresh remote restore is claimed.
The artifact inventory includes operational preparation attempts as well as
the final sealed output. Historical public reports, gates, baseline manifest
and decision policy are unchanged.

Validation completed: 291 experiment contracts, 18 root dataset/gate tests,
exact file-backed synthetic-fit parity, and the bundled submission format
check. The new modules contribute 26 synthetic contracts, including target
independence, actual code taxonomy, team symmetry, masks and discontinuities,
probability invariants, source/input tampering, exclusive writes and blocking
alternate-directory retries after a failed reference read. A standalone Luna
verifier replays the prior and decoder from saved arrays without opening or
rescoring public reference labels.
