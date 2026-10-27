# Live-play corruption and a three-class GRU

Keep `gru_hgb_ballgap25_v1`. This bounded experiment evaluates two proposed
directions on the five existing development matches. A +0.01 equal-provider
clean spell-F1 gain is a screening threshold, not permission to promote a model.
Independent labeled broadcast confirmation is still required.

## Measured development results

| Arm | Clean spell F1 | Clean macro F1 | Stress spell F1 | Stress macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.5338 | 0.8300 | 0.4129 | 0.7779 |
| Live-gap pipeline | 0.5240 | 0.8252 | 0.4277 | 0.7870 |
| H/A/D GRU | 0.6272 | 0.7883 | 0.3976 | 0.6722 |
| Live-gap H/A/D GRU | 0.6002 | 0.7289 | 0.4278 | 0.6238 |

These are full H/A **and** D provider holdouts, averaged equally across three
providers. They must not be mixed with pooled-package or D-only holdout scores.
All 60 paired evaluations completed. The archived baseline reproduces exactly
on all 15 match/view cases for spell F1, macro F1, D F1 and spell counts.

| Arm | Clean spell delta | Passes +.01 screen |
| --- | ---: | --- |
| Live-gap pipeline | -0.0098 | No |
| H/A/D GRU | +0.0934 | Yes |
| Live-gap H/A/D GRU | +0.0664 | Yes |

Highest clean development score: **H/A/D GRU**. This is a development
screen, not independent broadcast confirmation or an automatic baseline change.

The targeted corruption fails the clean screen (-.0098), although synthetic
stress spell F1 improves by .0148. Keep it as robustness evidence, not a baseline
replacement. The plain joint GRU passes the clean screen (+.0934), but stress
spell F1 falls by .0153 and clean macro F1 falls by .0417. Its clean spell gains
are concentrated in Sportec and Metrica; SkillCorner declines by .0104.

The main failure is dead-ball recall under provider shift. On SkillCorner,
the plain GRU's mean D F1 falls from .7022 to .1773; D recall is only 7.6–13.0%
across its three matches. The live-gap GRU also misses most SkillCorner dead
time and reduces Metrica D F1 from .8893 to .4810. Its overall clean macro
F1 falls by .1011. The spell gains therefore do not demonstrate a reliable
three-class replacement. The next bounded GRU study should address this
dead-ball transfer failure before any fresh confirmation or component removal;
no further fitting or decoder adjustment was performed after this readout.

### Clean behavior and runtime

| Arm | Recall <3 s, IoU .5 | False possessions / graded min | Predicted spells | Fragmented reference spells | Seconds / match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 29.2% | 2.917 | 2242 | 124 | 52.9 |
| Live-gap pipeline | 27.9% | 2.955 | 2243 | 118 | 51.7 |
| H/A/D GRU | 38.0% | 2.753 | 2348 | 45 | 22.9 |
| Live-gap H/A/D GRU | 37.4% | 2.948 | 2397 | 60 | 22.8 |

There are 1,963 reference spells across the five matches. Counts sum matches;
recall, false possessions per minute and latency use equal-provider means.
Latency includes common feature extraction and the fixed decoder, with one CPU
thread per model. These are single-run timings, not latency confidence intervals.
The joint GRU has 62,643 parameters and a roughly 0.26 MB checkpoint, with two
team-swapped inference passes. This simplifies model components while retaining
the existing feature engineering and decoder. No pooled deployment GRU was fitted.

### Provider tradeoffs

| Provider | Baseline clean spell | Live-gap pipeline delta | H/A/D GRU delta | Live-gap H/A/D GRU delta |
| --- | ---: | ---: | ---: | ---: |
| sportec | 0.5332 | -0.0191 | +0.1297 | +0.1373 |
| metrica | 0.6457 | -0.0159 | +0.1609 | +0.0689 |
| skillcorner | 0.4225 | +0.0056 | -0.0104 | -0.0069 |

| Provider | Baseline clean macro / D F1 | H/A/D GRU clean macro / D F1 | Live-gap H/A/D GRU clean macro / D F1 |
| --- | ---: | ---: | ---: |
| Sportec | .8500 / .9098 | .8995 / .9458 | .9074 / .9558 |
| Metrica | .8912 / .8893 | .9176 / .8880 | .7229 / .4810 |
| SkillCorner | .7488 / .7022 | .5479 / .1773 | .5565 / .1986 |

The full aggregate evidence and all per-match/seed results are in
[live_gap_had_development.json](results/live_gap_had_development.json).
[Provenance and corruption audit](results/live_gap_had_audit.json) records
runtime versions, checkpoint verification, matched GRU budgets, and realized
training gaps. Model weights and frame-level predictions are archived in the verified
private cache on branch `experiments/live-gap-had-v1`, with
`live_gap_had_cache_manifest.json`; the user-approved upload is complete.
The current baseline manifest, default
inference entrypoint and organizer gates are unchanged.

Validation: 77 synthetic/contract tests and GitHub CI passed. The bundled
submission format check passed; no new organizer submission or public score
was generated.

## Frozen comparison

| Arm | Training change | Inference |
| --- | --- | --- |
| baseline | Archived full-provider-holdout counterpart of the current recipe | HGB teams + 50/50 GRU/ordered HGB D |
| live_gap_pipeline | Replace 25% of team rows and D windows with live-gap/fragmentation/uncertainty views | Same three components |
| had_gru | Original clean/coarse-ball views, with 25% whole-window ball-gap replacement | One small H/A/D GRU |
| live_gap_had_gru | Same GRU budget; replace the 25% ball-gap views with live-gap/fragmentation/uncertainty views | One small H/A/D GRU |

The held-out provider is excluded from **every** learned component and all
normalization. Training uses only the original training matches; all derived
views remain grouped with their original match. The archived baseline checkpoints
are verified by SHA-256 and their exact training-match lists. No pooled baseline
checkpoint is substituted into a full-provider-holdout comparison.

The new corruption places renewal gaps on the external training match's live
time clock within each continuous tracking block. It preserves dead/unknown
ball observations, all native missing coordinates and every target. Gaps that
meet dead/unknown intervals are split there. Durations use the previously frozen
unlabeled organizer gap histogram (median 10.72 s, capped at 30 s), **not** a
measured live-play-only distribution. Existing fragmentation, player dropout and
coordinate-error assumptions are reused. Ball localization error is independently
smooth with 0.2 m SD; player camera drift is not assumed to move the ball identically.
Realized live/dead missingness is reported by training match.

The completed training-input audit finds realized median live gaps of 7.76–8.20 s
on Sportec, 5.68–5.76 s on Metrica, and 3.20–5.40 s on SkillCorner. The corresponding
90th percentiles span 11.24–22.96 s. Clipping at dead/unknown intervals prevents
an exact match to the 10.72 s calibration median; this is a limitation of the
tested corruption, not evidence that competition gaps were reproduced exactly.

This tests a targeted training change, not a faithful simulator of competition
tracking. Conditioning on external training states is explicit supervision for
augmentation; neither evaluation corruption nor inference uses those states.
The two evaluation stress views remain the prior label-free composite-corruption
seeds, so they do not favor the new live-conditioned generator by construction.
Every arm receives the same raw clean and corrupted evaluation inputs.

The GRU takes the union of existing team-specific ball/player geometry,
relative-motion/contact evidence and temporal summaries, plus player-only
collective-motion D features. Finite-value masks distinguish missing data from
zero motion. Normalization is learned from training core rows only. There are
24 hidden units in each direction, three output logits, eight epochs, 5 Hz
sampling, 10 s output cores, 5 s context on either side, batch size 64, and
AdamW with learning rate .001 and weight decay .0001. Windows never cross a
match, period or timestamp gap. H/A-swapped training and inference enforce team
symmetry. Unknown targets carry zero loss weight.

The GRU comparison is a whole-pipeline replacement: it has a joint cross-entropy
loss and whole-window augmentation instead of separate H/A and D losses. It is
not merely a changed output layer or an equal-optimizer-update comparison to
the three-component pipeline. Both GRU arms have identical training budgets.
The reference GRU uses the team model's clean/coarse-ball observation recipe
for its entire feature union; it does not have a separate legacy player-slot
dropout copy for a separate D loss. Consequently any difference measures this
complete joint-model recipe, not architecture alone with every training view
held identical to each of the old independently trained heads.
All arms retain the existing Viterbi decoder and switch cost 3.

## Labels and confirmation

The existing Metrica converter already ignores BALL LOST and CHALLENGE as
control changes; a subsequent RECOVERY establishes the opposing team's control.
The SkillCorner converter carries the last known team across live null-possession
intervals, retaining unknown intervals before first control and the one-second
uncertain stoppage margins. These rules were audited and their existing tests
rerun. Targets and native visibility grading masks are unchanged.

This is code-level semantic alignment, not an independent audit of every provider
annotation. Sportec provider possession/status and event-derived Metrica and
SkillCorner silver labels have no new adjudicated proof of equivalence for
deflections. Filling unknown labels from nearest-player geometry would invent
ground truth and is not part of this experiment.

The earlier acquired Belmont–Wakefield recording has audio but lacks calibrated
tracking and independent H/A/D labels. The different-producer MRVTV recording
remains a reserved confirmation candidate. Neither becomes a labeled validation
match through synthetic corruption or model predictions. Independent annotations,
tracking and alignment remain an unmet data dependency; no fresh-broadcast gain
is claimed. Organizer five-minute labels and completed public gates are unused.

The actionable access, blind-annotation, alignment and sealed-confirmation
checklist is [private data issue #4](https://github.com/jekkiechan/mtah-data/issues/4).
The recommended access route is the Gradient/PFF World Cup dataset; matching
video rights and independent dense labels must still be established. The issue
lives in the private data repository because Issues are disabled in the public
code repository. SoccerTrack's event-reconstructed auxiliary ball track is not
an independent observed-ball input and is not substituted into this benchmark.

## Readout

Primary: unchanged official spell F1, averaged over IoU .3/.4/.5/.6/.7, with
macro F1 only for an exact tie. Match means are averaged within provider, then
providers receive equal weight. Stress seeds average within match. The +.01
screen applies to the clean full-provider-holdout comparison. The auxiliary
70/30 spell/macro summary does not rank candidates.

Additional readouts are under-three-second spell recall at IoU .5, unmatched
predicted possessions per graded minute at IoU .5, reference spells overlapped
by multiple same-team predictions, predicted/reference spell counts, D F1,
per-provider changes and one-thread runtime. Unmatched predictions are an
operational false-possession measure, not proof that every such error is a
fragmentation error. Short-spell counts can include censored intervals under the
unchanged official grading policy; SkillCorner boundary completeness is unknown.

Timing includes both feature families, model inference and decoding, excludes
scoring and disk I/O, and shares feature extraction between arms. Runtime is
measured on the same machine and is descriptive; concurrent training can affect
wall-clock latency. Checkpoint sizes and parameter counts describe simplification
separately from speed.

## Reproduction

The completed run is uploaded to the private cache. Check out private
`mtah-data` branch `experiments/live-gap-had-v1` and run from that cache checkout:

```sh
python scripts/competition_cache.py fetch --manifest live_gap_had_cache_manifest.json --data-dir /path/to/code/data
```

The manifest includes the 18 external arrays, nine archived comparators, 15 new
checkpoints, 60 prediction files, frozen receipts and aggregate results. Completed
reports are immutable; use a fresh `--out` directory to repeat training.

Install `experiments/requirements-temporal.txt`, using CPU PyTorch 2.8.0. Restore
the 18 external arrays listed in `experiments/eval_set/manifest.json` from the
private cache. Restore only the nine comparator checkpoints and their JSON
receipts:

- private branch `experiments/possession-gaps-v1`,
  `experiments/new_provider_transfer_v1/without_*_full_holdout_equal_sources_team.*`
  to `data/experiments/archived_full_holdout/`;
- private branch `experiments/realistic-corruption`,
  `experiments/realistic_corruption_v1/without_*_baseline_{bigru,ordered_hgb}.*`
  to `data/experiments/archived_d_holdout/`.

Fetch the LFS objects before running; pointer files fail the hash checks. Keep
private cache remotes separate from the public code repository.

```sh
python -m experiments.live_gap_experiment freeze
python -m experiments.live_gap_experiment prepare
python -m experiments.live_gap_experiment fit --arm live_gap_pipeline
python -m experiments.live_gap_experiment fit --arm had_gru
python -m experiments.live_gap_experiment fit --arm live_gap_had_gru
python -m experiments.live_gap_experiment evaluate
python -m experiments.live_gap_experiment report
python -m unittest discover -s experiments/tests -v
```

Use `--source sportec|metrica|skillcorner` for individual preparation or holdout
jobs; for `fit`/`evaluate`, it names the excluded/evaluated provider. All three
models must be frozen before scoring that fold. `--out` selects a fresh run
directory; source fingerprints reject changes to an already frozen experiment.
Aggregate reports contain provenance; weights, caches and frame-level predictions
remain under ignored `data/`.
