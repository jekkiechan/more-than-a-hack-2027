# Which composite-stress components cause the stress loss?

The GRU's stress failure comes almost entirely from **per-frame sampling loss**: every
object is dropped on 2–8% of frames. The organizer feed does not have this defect.
Under feed-like stress, every GRU seed beats the adopted baseline by +0.04 to +0.05
spell F1. The Full127 LightGBM blend adds nothing over its GRU (+0.002 / −0.002).
These are descriptive results on five previously exposed external matches. No
model is adopted, and no organizer labels were read.

## Why this test

The [feed defect profile](results/feed_defect_profile.json) compares the organizer
feed with the composite stress, using tracking only:

| Statistic | BAR-ATH feed | Sportec / Metrica, composite stress |
| --- | ---: | ---: |
| Frames with no tracked players | 0% | ~5% |
| Frames losing ≥25% of players at once, per minute | 0.25 | ~70 |
| Ball gaps per minute / median length | 1.4 / 10.7 s | 31–44 / 0.04 s |
| Gaps inside a track, per track-minute / median length | 0.13 / 7.5 s | 27–28 / 0.04 s |

`stress_components.corrupt` draws every random event exactly as the frozen composite
does, but applies only the components it is asked to keep. With all components kept
it reproduces the composite exactly. Every view below uses the composite's own two
seeds, so the views differ only in which defects are applied.

| View | Components kept |
| --- | --- |
| `ball_gaps` | calibrated renewal ball gaps |
| `feed_like` | ball gaps, coordinate drift/noise/jumps, per-slot gaps, same-team ID swaps |
| `no_sampling` | composite minus per-frame sampling loss |
| `no_regional` | composite minus regional player blanking |

The study re-scored the frozen provider-holdout checkpoints without retraining. All 15
clean and composite cells reproduce the archived scores, with no parity issues.
[Declaration and full results](results/stress_component_ablation.json).

## Results

Spell F1 averaged with providers weighted equally; replicates are averaged within each match.

| Recipe | Clean | Ball gaps | Feed-like | No sampling | No regional | Composite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adopted baseline | 0.5338 | 0.4373 | 0.4328 | 0.4292 | 0.4183 | 0.4129 |
| Slot-dropout GRU 2026 | 0.6088 | 0.4842 | 0.4742 | 0.4712 | 0.3230 | 0.3180 |
| Slot-dropout GRU 2027 | 0.6176 | 0.4954 | 0.4853 | 0.4833 | 0.3110 | 0.3101 |
| Full127 standalone | 0.5555 | 0.4461 | 0.4421 | 0.4398 | 0.4364 | 0.4345 |
| Full127 + slot GRU 2026 | 0.6088 | 0.4874 | 0.4758 | 0.4755 | 0.4178 | 0.4164 |
| Full127 + slot GRU 2027 | 0.6113 | 0.4888 | 0.4833 | 0.4816 | 0.4297 | 0.4289 |
| Geometry63 + control GRU 2026 | 0.6257 | 0.4924 | 0.4803 | 0.4752 | 0.3758 | 0.3718 |
| Geometry63 + control GRU 2027 | 0.6184 | 0.4923 | 0.4760 | 0.4744 | 0.4022 | 0.3999 |

- **Attribution.** Removing per-frame sampling loss alone recovers 53–57% of the
  GRU's clean-to-composite loss. Removing regional blanking recovers 0–2%. The baseline and
  Full127 lose only 4–13% of their composite loss when sampling is removed: trees
  are insensitive to one-frame holes, and the GRU is not.
- **Dead-ball overprediction is a sampling artifact.** Under the composite, GRU D precision
  falls to 0.48–0.69 at recall ≥0.98. Under feed-like stress it stays at its clean
  level (0.83–0.94) on every provider.
- **GRU stress guardrail on feed-like stress.** GRU minus baseline is +0.040 / +0.049
  for control and +0.042 / +0.053 for slot dropout. All four pass, and the GRU beats the
  baseline on every provider in both seeds.
- **LightGBM value.** Full127 + slot GRU minus slot GRU on feed-like stress is
  +0.0016 / −0.0020, well short of the declared +0.02, so the tree's robustness value does
  not survive realistic stress. On clean data the blend is also within noise of the GRU,
  and on SkillCorner it lowers clean D recall (0.47–0.50 vs 0.62–0.70).

## Implication

Composite stress overstates the GRU's fragility, because per-frame dropout of every
object at once is not a defect of the organizer feed. Long ball gaps and
track fragmentation are the defects to test against. This matches the earlier public pipeline
check, where the GRU scored 0.6194 vs the baseline's 0.5257. These are still five previously exposed
matches, and differences below ~0.04 spell F1 are noise.

```sh
python -m experiments.stress_component_ablation freeze
python -m experiments.stress_component_ablation run --match <match> --view <view>
python -m experiments.stress_component_ablation report
```
