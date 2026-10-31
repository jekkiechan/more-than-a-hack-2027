# Confidence-based abstention

**Retain `gru_hgb_ballgap25_v1`.** All 135 external evaluations completed and
all 15 baseline cells reproduced the archived spell/macro scores within 1e-12.
The primary 60% abstention arm improves clean spell F1 by .00837, but loses
.00545 under corruption and loses .01787/.05183 clean/stress macro F1.
Every tested arm fails at least one predeclared guardrail. No baseline,
organizer gate, or model checkpoint changed.

| Arm | Clean spell | Stress spell | Clean macro | Stress macro |
| --- | ---: | ---: | ---: | ---: |
| Existing decoder | .57402 | .43152 | .84394 | .78659 |
| Abstain below .50 | .57866 | .42973 | .83939 | .77101 |
| **Abstain below .60 — primary** | **.58240** | .42608 | .82607 | .73476 |
| Abstain below .70 | .57500 | .40591 | .80383 | .66560 |
| Abstain below .80 | .56011 | .34230 | .76240 | .54912 |
| Hold below .50 | .57730 | .42831 | .84253 | .78499 |
| Hold below .60 | .57750 | .42087 | .83990 | .77580 |
| Hold below .70 | .56704 | .39823 | .83318 | .75511 |
| Hold below .80 | .55020 | .32926 | .81859 | .71031 |

These are five historically exposed development matches, with **D-only provider
holdout and pooled H/A**. They are neither full-pipeline holdouts nor independent
confirmation. The numerical clean leader is the primary abstain-.60 arm;
the adopted baseline remains unchanged. There is no clean/stress winner among
the tested gates.

The primary arm abstains on 13.15% of clean and 23.79% of corrupted graded
frames, using equal-provider aggregation. It fails the macro-loss guardrail
on both views, the stress-spell guardrail, and the Sportec stress guardrail.

| Provider | Clean spell delta, abstain .60 | Stress spell delta, abstain .60 |
| --- | ---: | ---: |
| Sportec | +.00937 | −.02947 |
| Metrica | −.00269 | −.00173 |
| SkillCorner | +.01844 | +.01486 |

The abstain-versus-hold spell difference is confined to SkillCorner, where
native unknown-label gaps reset the scorer's fill. Explicit .60 holding gains
only .00348 clean spell and loses .01066 stress spell overall. Thus the larger
X gain cannot be interpreted as the benefit of an equivalent deployable
state-holding rule. The distinction is tested below and in the contract suite.

The primary arm reduces clean predicted possessions from 2,023 to 1,831
against 1,963 reference spells. However, subsecond recall falls from 18.78%
to 15.06% clean, and 10.59% to 7.59% under corruption. Clean typed boundary
precision rises from 34.07% to 36.19%, while recall falls from 29.82% to 29.41%;
stress recall falls from 17.03% to 15.62%. Fewer predictions include missed
real short possessions, not just removed noise. Latencies in the JSON apply
only to boundaries that remain matched.

Dead-ball precision rises but recall falls: clean equal-provider D
precision/recall changes from .89538/.79682 to .92918/.72100; stress changes
from .80600/.84391 to .88705/.76025. The full per-match/provider diagnostics
and fixed 70/30 auxiliary summaries are in the
[aggregate report](results/confidence_abstention.json).

The conditional paired-match bootstrap puts the primary spell delta at
[+.00736, +.00939] clean and [−.00682, −.00445] stress (95%). These intervals
condition on the existing sources; Sportec and Metrica each contain only one
evaluation match and contribute no sampling variation. They do not establish
unseen-provider or full-match competition performance. The two corruption
replicates are stress views, not training seeds or an inference ensemble.
Both replicates lose primary-arm spell F1: −.00480 for `corrupt_1` and −.00609
for `corrupt_2` under the same equal-provider aggregation.

Study ID: `confidence_abstention_v1`. This compares post-decoder abstention
against the adopted `gru_hgb_ballgap25_v1` package. No model is fitted, no
organizer labels are accessed, and no baseline promotion is part of this study.

## Frozen question and protocol

Does rejecting a low-confidence decoded state improve official spell F1 over
the existing Viterbi decoder, without material clean/stress regressions?

The [frozen declaration](results/confidence_abstention_declaration.json) is the
authority for arms, thresholds, cohort, hashes, seeds and guardrails. The primary
arm is `abstain_60`: after unchanged Viterbi decoding at switch cost 3, write `X`
where the joint H/A/D probability of the **decoded class** is below .60. These
are uncalibrated local model probabilities, not guaranteed correctness
probabilities or confidence in the complete Viterbi sequence.
Thresholds .50, .70 and .80 are predeclared sensitivity checks, not additional
independent tests. Equality to the threshold accepts the label.

Each threshold also has a `hold` control: replace rejected labels with the last
accepted state inside an input time block. As this is offline inference, leading
gaps use the first accepted state. If a whole block has no accepted state, hold
retains the original decoder while abstain produces all X. Inference uses no
reference labels or reference-derived boundaries.

The comparison restores the original D-provider-holdout GRU/HGB checkpoints
and the adopted pooled ball-gap H/A checkpoint. Provider exclusions apply to
both D heads and their normalization, **not** to H/A training. The external
development games are Sportec J03WOY, Metrica 3, and SkillCorner 1996435,
1996436 and 2006229. All have prior research exposure. The 13 training matches
and all input/checkpoint hashes are listed in the declaration. There is no new
training, independent confirmation match, or pretrained-checkpoint change.

Each match has clean inputs and the two original deterministic composite
corruption views. The saved label-free gap calibration and assumed coordinate
errors are reused; no organizer inputs are downloaded or recalibrated. Features
are recomputed after corrupting raw tracking. Native observation masks,
unknown targets, period/time breaks, and the original SkillCorner filter of
at least 12 observed players are preserved. Grading is fixed before corruption.

Sportec supplies provider H/A/D labels but has frozen-ball dead-state shortcuts.
Metrica uses event-derived silver labels and ball-missingness shortcuts.
SkillCorner uses provider possession and interruption-derived D cores with
unknown margins; visibility filtering further limits its graded population.
These source differences limit transfer and boundary claims.

Stress replicates average within match, then matches within provider, then the
three providers receive equal weight. Official spell F1 is primary, macro F1
breaks exact ties, and the .7 spell + .3 macro summary is auxiliary only.
The primary arm must improve clean spell F1, lose no stress spell F1, lose at
most .005 macro F1 on either view and at most .02 spell F1 on any provider/view.
No arm is automatically adopted.

The budget is 15 prediction cells × 9 arms = 135 evaluations, with zero fits.
All predictions are sealed before scoring. Every baseline cell must reproduce
the archived `ball_team25` spell/macro scores within 1e-12 before any candidate
is scored. Interrupted scoring resumes from saved aggregate cell results.

## Scoring interpretation

The official scorer fills X within **reference-graded runs**, which may be
shorter than input halves/time blocks when native targets are unknown. Explicit
model-side holding cannot use those hidden boundaries. Therefore abstain and
hold can differ on external data with native X gaps, despite matching temporal
behavior on uninterrupted graded runs. This is a scorer/coverage distinction,
not evidence that X estimates a better possession state.

Short-spell recall uses official one-to-one matches at IoU .5. Unknown margins
may censor possessions; short spells are observed graded fragments, not always
complete real-world possessions. Typed transition diagnostics use .5-second
tolerance; reported latencies are in seconds and only cover matched boundaries.
No claim of timing quality on censored boundaries follows from these numbers.

## Reproduction

Use Python 3.12, [pinned experiment dependencies](requirements.txt), and CPU
PyTorch 2.8.0. Restore the five external input files and adopted model package
from private `jekkiechan/mtah-data` main. Restore
`corruption_cache_manifest.json` from its `experiments/realistic-corruption`
branch using `scripts/competition_cache.py`; this supplies the held-out D heads
and original corruption declaration. Their bytes are checked before inference.

```sh
python -m unittest discover -s experiments/tests -v
python -m experiments.confidence_abstention freeze
python scripts/prepare_abstention_cache.py
python -m experiments.confidence_abstention evaluate
```

The optional four-worker preparer changes scheduling only and records its own
source hash and the declaration hash in `cache_execution.json` before running.
The main module independently seals every prediction cell. The commands above
describe the original run; a completed run's aggregate report is intentionally
exclusive and cannot be overwritten. Restore/read the saved results for review.
Serial preparation
is also available with `python -m experiments.confidence_abstention prepare`.
Use the exact recorded source snapshot to reproduce a frozen study; do not edit
the declaration to accommodate a newer checkout. Existing result files are
immutable; read them instead of rescoring a completed study.

Frame-level caches remain private under
`data/experiments/confidence_abstention_v1/`. Public outputs contain declarations,
aggregate metrics and artifact receipts only. No organizer submission CSV is
generated by this external-only experiment.

All 213 artifact files (34,682,668 bytes), including the sealed probabilities,
cached cell scores and source snapshot, are uploaded to private
`jekkiechan/mtah-data`, branch `experiments/confidence-abstention-v1`, commit
`f24085861e020cf3f4772998e2e103a4a907cfc4`. A fresh separate clone downloaded
and verified every size/SHA-256 against
`confidence_abstention_cache_manifest.json`. See the
[artifact receipt](results/confidence_abstention_artifacts.json).

From that private branch, restore the completed experiment with:

```sh
python scripts/competition_cache.py fetch --manifest confidence_abstention_cache_manifest.json --data-dir /path/to/more-than-a-hack-2027/data
```

Reaggregation from cached cell scores was byte-identical to the published JSON,
with `score_cell` patched to raise if called: zero additional cell scoring.
Restored probabilities avoid repeating the CPU feature-generation work.

## Validation and next action

All 80 synthetic/contract tests passed with the pinned dependencies, including
CPU PyTorch 2.8.0. Checks cover team swaps, half/time discontinuities, all-X
blocks, threshold equality, official fill behavior, suppressed false splits,
delayed true turnovers, input immutability and serial/parallel cache parity.
All 135 cohort cells preserve the same per-match grading and reference-spell
counts. The bundled 199,319-row CSV passed format validation; no new organizer
CSV was produced or scored.

Retain the current baseline. Do not extend this threshold sweep on the same
matches. A further reliability-gating study needs a distinct hypothesis and
independent, more completely labeled broadcast evidence, particularly to
separate actual state-estimation gains from native-X scoring effects.
