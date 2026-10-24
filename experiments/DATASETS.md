# External development data

The Barcelona–Athletic challenge is excluded from external training and model
selection. Its public five-minute labels remain a pipeline check only.

## SkillCorner auxiliary benchmark

Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), pinned to
commit `4340d274572876239c154c90bc507a9b3250a656`.

The repository [licence](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE)
is MIT, copyright (c) 2020 SkillCorner. Credit: SkillCorner and PySport's open-data
initiative. The downloaded data and generated NPZ files stay in the gitignored
`data/external/` directory. The source README asks users to acknowledge SkillCorner.

The following match split was fixed by ascending available match ID before model
training or label inspection. There is no random frame split and no split seed.

| Split | Match IDs |
| --- | --- |
| Train | 1874553, 1886347, 1899585 |
| Validation | 1925299 |
| Held-out test | 1927964 |

These are distinct 2024/25 Australian A-League matches. They are unrelated to the
provided LaLiga challenge. The five tracking files contain approximately 300,000
timestamps at 10 Hz; the raw tracking/metadata total about 454 MB, while the
normalized arrays are about 29 MB compressed. No video is needed.

### Target semantics and limits

Targets come directly from `possession.group` in SkillCorner's tracking files:

- `home team` -> `H`
- `away team` -> `A`
- null -> `X`, meaning **unlabelled/unknown**, not a known dead ball

There are **no explicit D labels in this source**. A missing possession annotation
must not be relabelled D. This dataset can support team-attribution experiments,
but cannot establish dead-ball accuracy or a complete H/A/D benchmark. A two-class
H/A macro F1 must be named as such; the hackathon's official three-class macro F1
would be capped at 2/3 when D is absent from both target and prediction. Spell
scores restricted to labelled runs also omit unknown intervals and are not
directly comparable to the challenge's full-match possession metric.

SkillCorner's possession targets are **provider-generated silver labels**, not
independent human annotations. Its tracking coordinates are also model outputs,
including extrapolated player and ball positions. A high score measures agreement
with that provider on unseen matches; it does not prove accuracy against reality
or the hackathon's corrected tracking. The source documents erroneous data points
and approximately 97% player-identity accuracy. That identity figure is not a
possession accuracy claim.

### Input/target separation

`experiments/datasets.py` reads the possession field only when constructing target
labels. It never supplies source ownership, possessor player ID, team playing-time
statistics, dynamic events, or phases of play to the input arrays. Team membership
comes from the match's roster, not from possession. Detection flags describe
visibility and are distinct from ball-in-play status.

Coordinates are translated from pitch-centred metres to metres relative to a
corner, `[0, pitch_length] x [0, pitch_width]`. Extrapolated positions are retained;
their visibility flags remain available so models can choose whether to use them.
Ball extrapolation and provider-generated targets may share upstream processing,
which is another reason to treat this as an auxiliary benchmark.

The provider match clock is missing on 19 unassigned rows around the halves in
each downloaded match. These rows are retained, with unknown targets preserved.
Model timestamps use the verified contiguous source frame index divided by 10 Hz,
and explicit period changes delimit halves. This yields a finite sampling clock
without inventing labels or joining possessions across unknown intervals; it is
not elapsed wall-clock time across halftime. The converter also checks that known
source match-clock increments within each half agree with 10 Hz.

### Reproduce

```bash
python experiments/datasets.py --download
```

The downloader uses the pinned commit, including Git LFS media for tracking.
Tracking SHA-256 checksums were verified against the pinned LFS pointers. Per-match
JSON manifests store the actual metadata and tracking hashes, pitch dimensions,
label mapping, source, and provenance. `data/external/split.json` stores the split.

Outputs are `data/external/processed/skillcorner_<match_id>.npz`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `ball` | N x 2 | Ball x/y metres, float32; NaNs preserved |
| `players` | N x P x 2 | Player x/y metres, float32; absent slots are NaN |
| `teams` | P | 0 home, 1 away, int8 |
| `labels` | N | H/A/X Unicode targets; isolate from input features |
| `period` | N | 1/2 for match periods, 0 when unassigned |
| `fps` | scalar | 10 Hz |
| `timestamps` | N | Contiguous source frame index / 10 Hz; finite sampling clock |
| `pitch` | 2 | Pitch length/width in metres |
| `ball_observed` | N | Whether the ball was detected rather than extrapolated |
| `player_observed` | N x P | Whether each player was detected |

No player names, jersey numbers, source possession flags, or ball-live flags are
saved as features. `load_match(name)` loads a dictionary from this format. Models
must select explicit input keys and keep `labels` separate.

## Complementary benchmark

See `BENCHMARK_AUDIT.md` for the independently investigated Sportec Open DFL data
with explicit ball status and possession, and for the distinction between that
primary H/A/D benchmark and this auxiliary H/A-only silver-label source.
