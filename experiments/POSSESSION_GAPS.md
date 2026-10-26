# Possession-gap experiment results

**Results only. Keep `gru_hgb_ballgap25_v1` as the current working baseline.**
No tested configuration establishes an overall improvement. This publication
contains aggregate evidence and research recommendations. It changes no model,
training or inference code, checkpoint, baseline manifest, decision policy or
organizer evaluation gate.

## Matched development results

The current baseline was reproduced exactly: spell F1 .57402 clean / .43152
corrupted, macro F1 .84394 / .78659. These are five historically exposed
development matches, not fresh independent confirmation.

The following comparisons hold out the provider from D training while H/A
training remains pooled. Scores use the unchanged official greedy spell metric,
averaged over IoU .3/.4/.5/.6/.7. Corruption seeds average within match, matches
within provider, then providers receive equal weight. Grading masks stay fixed.

| Fixed arm | Clean spell F1 | Corrupted spell F1 | Clean macro F1 | Corrupted macro F1 |
|---|---:|---:|---:|---:|
| Current baseline | .57402 | .43152 | .84394 | .78659 |
| Short-spell loss weighting | .56942 | .43726 | .84310 | .78615 |
| Directional player-only stoppage/restart heads | .57422 | .43142 | .84430 | .78678 |
| 25% ball-hidden conditional-team ensemble | .56613 | .43201 | .84126 | .78666 |
| Duration prior, strength .15 | .49524 | .35241 | .84142 | .78487 |
| Duration prior, strength .35 | .53193 | .39394 | .84274 | .78593 |

Short-spell weighting raises clean subsecond recall from 18.78% to 20.61% and
1–3-second recall from 35.20% to 37.04%, but loses .460 percentage points of clean
spell F1. Its stress gain is .573 points. This is a tradeoff, not a promotion.

The boundary head raises one-second boundary F1 from 44.45% to 45.70% and recall
from 40.55% to 41.70% on timing-eligible Metrica/Sportec. Downstream spell F1 is
essentially unchanged. A control that halves every D/live switch penalty scores
.57247 / .42959. The tested head adds player-context evidence, not audio.

Duration priors recover some short spells but overproduce possessions: 2,695 and
2,356 predictions versus the baseline's 2,023 and 1,963 reference spells across
the five clean holdout cases. Ball-hidden averaging loses clean short-spell
recall with negligible stress gain.

## Stricter provider transfer

Both H/A and D training exclude the evaluation provider in this comparison.
Its more demanding training split must be kept separate from the table above.

| Fixed training arm | Clean spell / macro | Corrupted spell / macro |
|---|---:|---:|
| Equal source weights | .53380 / .83002 | .41293 / .77793 |
| SkillCorner source weight ×2 when present | .53006 / .83026 | .40910 / .77660 |

The fixed 2× weighting does not help. Closing the pooled-H/A gap lowers the clean
spell estimate from .57402 to .53380; SkillCorner drops from .46511 to .42250.
This improves evaluation coverage, not independence: all five games are reused.
SkillCorner's censored dead frames and boundary timing remain unassessed.

## Whistle evidence and acquired audio

A label-using diagnostic reaches .69094 spell F1 with perfect observed D/live
labels and baseline team evidence. Perfect team evidence with baseline P(D)
reaches .85165. These are error-source diagnostics, not achievable guarantees.
The weak player-only boundary result does not rule out a genuinely new audio cue.

Three full-match-length community football recordings have verified audio and
public reuse terms. The acquired development source is Belmont Media Center's
[Belmont–Wakefield, 9 September 2024](https://archive.org/details/snafuinfinitysoccer-boys_090924-vs-wakefield):
92m59s, CC0, with 51 MB of native 48 kHz stereo AAC retained privately. This is
community/high-school footage, not measured professional-broadcast transfer.

A fixed pretrained whistle detector processed the entire recording in 7m42s.
It exported **335 raw candidate fragments**, 428 unlabeled development review
tasks, and a separate 93-part whole-timeline annotation template without detector
hints. These are not 335 verified whistles. No whistle accuracy or possession
gain has been measured. Tracking and independent D/H/A labels are still needed.

MRVTV's Lake Region–Harwood 2022 recording is a different-producer confirmation
candidate with metadata-only exposure; Belmont–Wellesley 2022 is a backup.
Full video coverage, edits, audio quality and timing still require audit.

## Evidence and publication scope

- [Combined scorecard](results/new_gap_experiments.json) and
  [exact baseline replay](results/new_gap_baseline_reproduction.json).
- Individual reports: [short spells](results/new_short_spell_development.json),
  [boundaries](results/new_boundary_development.json),
  [occlusion](results/new_occlusion_development.json),
  [duration](results/new_duration_development.json), and
  [full provider transfer](results/new_provider_transfer_development.json).
- [Full audio pass](results/new_full_match_whistle_inference.json),
  [verified audio sources](results/new_audio_source_search.json),
  [whistle research](results/new_whistle_audio_research.json), and
  [broader data research](results/new_broadcast_data_research.json).
- [Next research directions](NEXT_RESEARCH.md).

Source fingerprints in the reports describe the local experimental checkout,
including local numerical freeze `4fe2fab`. The implementation and its 107 local
tests are excluded from this results-only publication at the user's request.
Compact checkpoints, audio and provenance remain in private `mtah-data` branch
`experiments/possession-gaps-v1`. No new baseline is adopted.
