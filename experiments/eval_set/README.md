# Temporal-model evaluation set

This is a frozen **development benchmark**, using the existing external
whole-match splits. It does not manufacture a fresh test by reshuffling matches
that have already been used. `manifest.json` records match paths, SHA-256 hashes,
source/label provenance, previous exposure, pooled and leave-one-source-out
folds, and the missing independent-confirmation set. No organizer tracking or
public-window labels are read by this package.

| Source | Train | Selection | Label provenance |
| --- | --- | --- | --- |
| Sportec | J03WMX, J03WN1, J03WOH | J03WOY | Provider BallStatus/BallPossession |
| Metrica | Games 1, 2 | Game 3 | Human-event-derived H/A/D, including inferred D intervals |
| SkillCorner | First 8 IDs in `dead_transfer.SKILLCORNER_IDS` | Last 3 IDs | Provider possession and interruption-derived D cores; silver |

All frames, halves, windows, masks and team swaps from a match inherit its role.
Neither randomly split frames nor overlapping training/selection windows are
valid. Fit temporal normalization on training matches only and split sequence
context at period and timestamp discontinuities. The complete finite candidate
family, epochs, seeds and decoder settings must be frozen by the experiment
runner before selection. This manifest fixes evaluation data, not model weights.

Use the existing official metric through
`experiments.metrics.evaluate_tracking`: greedy one-to-one H/A spell matching,
F1 averaged across IoU .3/.4/.5/.6/.7. Do not substitute maximum-cardinality
interval matching, a single IoU threshold, or D intervals as possession spells.
Report every match, per-source means, and an equal-source mean; millions of
correlated frames do not give millions of independent observations.

For a D-head experiment, retain the same conditional H/A predictions and decoder
for every candidate. Choose using external development results, primarily
equal-source official spell F1 on leave-one-source-out folds, with D
precision/recall/F1 and boundary errors showing whether the proposed mechanism
helps. Retain pooled results as a complementary within-source check. If a pooled
H/A teacher has seen the held-out provider, a LOSO D-head fold measures **D-head
transfer**; the entire H/A/D system is not provider-unseen. Do not adapt candidate
grids after seeing the organizer check.

`summarize_d_boundaries(labels, pred, period, fps, timestamps=None)` collapses
H/A into live and uses the official transition matcher. This prevents a team
switch from becoming a stoppage boundary and distinguishes entering/leaving D.
It reports precision/recall and signed/absolute latency in seconds at roughly
0.5 and 1 second tolerances (rounded to the input frame rate). Matching is
one-to-one, with the official earliest-feasible policy; matched latency is
conditional on a match, so always inspect recall too. The official X abstention
handling, reference X gaps, period boundaries and timestamp gaps are preserved.

`paired_source_bootstrap` compares rows with identical `(eval_source, match)`
keys. It resamples paired matches within each source, then gives each source
equal weight. Its interval is conditional on these sources and historical
matches. A source with only one match contributes no sampling variation; these
five matches cannot justify a strong unseen-provider significance claim.

## Shortcut and coverage audit

`shortcut_audit.json` contains aggregate statistics on the existing selection
matches. It confirms why a single-provider result is insufficient:

| Selection source | Live cue | D cue |
| --- | --- | --- |
| Sportec J03WOY | 0.66% stationary ball | 100% stationary ball |
| Metrica game 3 | 0.33% ball missing | 99.77% ball missing |
| SkillCorner, three matches | 78.0–89.8% retained by visibility filter | Only 24.0–28.9% retained |

Stationary here means at most 0.05 m/s between valid consecutive observed
samples within a half/time block. The audit uses tracking availability flags,
not interpolated positions, where available. SkillCorner's existing >=12-visible
player filter removes camera-cut shortcuts but also discards most dead-ball
frames. Its scores therefore describe visible-player stoppages, not every
broadcast stoppage. Preserve reference X gaps rather than concatenating surviving
frames: gaps must not create artificial possessions or matchable boundaries.

Player-only D features avoid Sportec/Metrica ball shortcuts. They can still learn
provider-specific player-detection patterns; LOSO comparisons and visibility
coverage remain necessary. For additional stress probes, freeze missing-ball
seeds and rates externally, distinguish synthetic gaps from real occlusions,
and never use ball missingness to derive reference labels.

## Confirmation and public five minutes

No available match is a fresh confirmation set: earlier experiments have
already trained on or scored all listed matches. Historical Sportec tests
J03WPY/J03WR9/J03WQQ remain excluded. The next independent test needs a new
external match, representative stoppages and real live-play missing-ball
behavior, with accessible label provenance. An external manual/adjudicated
annotation project would be allowed under the challenge rules, but existing
Metrica/SkillCorner silver labels must not be renamed manual gold.

The organizer's public five minutes are a pipeline check only under `RULES.md`.
After external selection, freeze the selected model/configuration and make one
check against the previous candidate for alignment and a descriptive score.
Do not use that result to change a threshold, choose a candidate, tune a model,
or claim full-match improvement. No hand labels or manual corrections on the
organizer match are allowed.

## Run

From the repository root:

```sh
# Already executed; these commands deliberately refuse to overwrite artifacts.
python -m experiments.eval_set.protocol freeze
python -m experiments.eval_set.protocol audit

# Assets-free contracts; no torch or LightGBM dependency.
python -m unittest experiments.eval_set.test_protocol -v
```

Use a new `--output` path for a new protocol rather than overwriting a frozen
manifest. A new data configuration needs a new manifest and declared exposure
history. Frame predictions and tracking arrays remain under ignored `data/`;
only aggregate audit results and file hashes are tracked here.
