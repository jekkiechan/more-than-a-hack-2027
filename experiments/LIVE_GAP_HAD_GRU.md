# Live-play corruption and a three-class GRU

Keep `gru_hgb_ballgap25_v1`. This bounded experiment evaluates two proposed
directions on the five existing development matches. A +0.01 equal-provider
clean spell-F1 gain is a screening threshold, not permission to promote a model.
Independent labeled broadcast confirmation is still required.

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
