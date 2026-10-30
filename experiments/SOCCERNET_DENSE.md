# SoccerNet-GAR dense silver H/A/D labels (recipe v1)

Study `sn_dense_v1`. The adopted SoccerNet training (PRs #36-#40) used one H/A
label per PASS/CROSS/SHOT clip centre. This module builds dense per-frame
H/A/D labels from the event stream already present in the `tracking-full`
parquet columns, for GAR train, valid and test, on the same stride-6 (about
5 Hz) grid and the same 204-feature schema. **Silver** means derived by rule
from event metadata, not hand-annotated per frame. The GAR test split was only
downloaded and labelled; nothing was scored on it. Organizer public ground
truth was never read.

Code: `soccernet_dense_labels.py` (build, `load_dense_records`),
`soccernet_dense_qa.py` (QA), tests in `tests/test_soccernet_dense_labels.py`.
Frozen modules are untouched.

## Data acquisition

Revision `32c840210b7d08d06976e7f9457dfff79fb2da99`, branch `tracking-full`,
all 14 files (7.1 GB) into the ignored
`data/independent/soccernet_gar/tracking-full/`. The seven TRAIN file SHA-256
values equal `raw_source_sha256.file_sha256` in
`results/soccernet_capacity_declaration.json`. Valid/test hashes (equal to the
Hugging Face LFS hashes) are in
[`results/soccernet_dense_v1_download_receipt.json`](results/soccernet_dense_v1_download_receipt.json).
Splits: train 62,159 clips / 45 games, valid 12,091 / 9, test 13,689 / 10.

## Reuse of the frozen path

`_patched_assemble()` takes the source of the frozen
`soccernet_segments._assemble_game` and applies four asserted textual hooks in
memory (it raises if the frozen text ever changes): carry the event columns on
each parsed row, count event-column conflicts between duplicate frames, emit
every contiguous segment (the frozen code only keeps segments containing a
sparse anchor), and expose per-frame events. Merging of overlapping clips,
exact-duplicate dedup, breaks at invalid/missing frames, jersey slots, the 204
features at stride 6 and the sparse anchors are exactly the frozen code.
Parity check on valid game 3841: all 217 frozen anchor-bearing records have
bit-identical `x`, `indices`, `y`, `full_y`, `full_timestamps` (plus 27 extra
records the frozen code drops for lacking anchors). Split-aware sample reading
is new because the frozen reader is train-only. Event-column conflicts between
duplicate frames were zero.

## Recipe v1 (per merged contiguous record, native frames)

Touch run = maximal consecutive frames with `game_event_type` `OTB` (or any
`*KICKOFF`) and the same `game_event_id`. Team from `home_team` ('1' H, '0' A;
none or conflicting = unknown team, 0.5% of runs). End type = last non-empty
`possession_event_type` in the run. A run is **uncontrolled** if it ends in CH
or RE, or is a single frame ending in CH, RE or CL. A run cut by the record end
with no end type is treated as controlled (6.6% of runs).

An *owner* state is tracked:

1. Inside a run of the owner's team: that team (provenance `touch`).
2. Gap to a next run of the same team: owner throughout (`propagated`).
3. Next run is the opponent and controlled: gap stays owner, switch at the
   opponent run start (`touch`).
4. Next run is an uncontrolled opponent touch: possession stays with the owner
   until a later controlled opponent touch. Frames inside that uncontrolled
   run are labelled with the owner (`held_uncontrolled`).
5. `OUT` game event: the pre-OUT gap keeps the owner only if the OUT is within
   3 s of the last touch (else unknown); D (`dead`) from the OUT frame until the
   next touch run or record end. The next toucher after a dead ball becomes the
   owner whether or not that run is controlled.
6. Unknown before the first controlled touch of a record, after the last
   touch (unless an OUT follows within 3 s), and across a touch run of
   unknown team. Tracking-validity gaps already split records, so nothing
   propagates across them.
7. Sparse centre-anchor labels are kept (`anchor_y`, `anchor_full_y`) and fill
   only frames the recipe left unknown (`anchor_only`, 0.01% of frames).

Deviation from the literal brief wording ("inside a run: that team"): inside
an *uncontrolled* opponent run we hold the owner (rule 4), because that touch is
by definition not a change of possession. The literal alternative is stored as
`full_y_literal`. It would double the H/A switches (13,922 vs 5,911 on train)
and make 65% of complete spells shorter than 3 s, i.e. flicker.

## Record format

`load_dense_records(split)` returns a list of dicts sorted by (game, half,
first frame), one per contiguous segment that has at least one known label.
Files: `data/experiments/sn_dense_v1/records/{train,valid,test}/games/<game>.joblib`
plus a hash-sealed `manifest.json` (ignored by git). All frozen-record keys are
kept: `x` float32 (T,204), `names` (204), `native_fps`, `fps` (=native/6, about
4.995 Hz), `indices` (T,), `native_n`, `full_frame_numbers`, `full_timestamps`,
`full_player_observed` (N,slots), `full_ball_observed`, `sampled_*`, `source`,
`match`, `segment_id`, `anchor_frame_mapping`. Changed or new:

| key | shape | meaning |
| --- | --- | --- |
| `y` | (T,) int8 | dense label on the stride-6 grid, **0=H, 1=A, 2=D, -1=unknown** (same coding as all other records; D is `y==2`) |
| `full_y` | (N,) int8 | dense label at native rate; `y == full_y[indices]` |
| `prov`, `full_prov` | int8 | 0 unknown, 1 touch, 2 propagated, 3 dead, 4 held_uncontrolled, 5 anchor_only |
| `period` / `full_period` | (T,) / (N,) int32 | one time-block id per record, with `period == full_period[indices]` as `transition_targets` requires. This differs from the frozen sparse records (which stored the half in `full_period`) |
| `game_period` | int | the real half (1, 2, 3...) |
| `anchor_y`, `anchor_full_y` | (T,), (N,) | original sparse H/A anchors (-1 elsewhere) |
| `full_y_literal` | (N,) | sensitivity labelling (literal in-run rule) |
| `touch_runs` | (R,6) int32 | columns `touch_run_columns` = start,end,team(-1 unknown),end_code,controlled,censored; `out_frames` OUT indices |

`objective_losses.spell_duration_factors` and `transition_targets` accept these
records directly (covered in the unit test). The 3 s grace and all codes are
module constants.

## Results (`results/soccernet_dense_v1_labels_report.json`)

| | train | valid | test (sealed) |
| --- | ---: | ---: | ---: |
| games / records | 45 / 11,249 | 9 / 2,191 | 10 / 2,439 |
| hours (labelled records) | 46.9 | 9.06 | 10.3 |
| 5 Hz rows | 847,712 | 163,564 | 186,026 |
| native H / A / D / unknown | .431 / .381 / .063 / .126 | .404 / .404 / .064 / .129 | .320 / .492 / .060 / .128 |
| provenance: touch / propagated / held / dead | .373 / .434 / .005 / .063 | .366 / .437 / .005 / .064 | .376 / .431 / .005 / .060 |
| H/A switches (per hour) | 5,911 (126) | 1,218 (135) | 1,284 (125) |
| complete spells (n), median, mean | 9,100; 4.1 s; 6.3 s | 1,827; 4.0; 6.0 | 2,063; 4.0; 6.3 |
| complete spells under 3 s / under 1 s | 37.6% / 12.3% | 38.1% / 13.1% | 38.5% / 12.4% |

Sparse centre anchors label about 0.9% of native train frames; dense labels cover 87.4%.
Organizer-style references in this repo: Metrica spells mean 9.6-10.4 s, median
5.0-5.9 s, 30-33% under 3 s ([OPEN_DATA](OPEN_DATA.md)); the Sportec reference
has 41% under 3 s, mean 7.7 s ([ERROR_ANALYSIS](ERROR_ANALYSIS.md)). Our
complete (fully observed) spells are shorter than the open-data means because
"complete" excludes the longest ones, which touch record edges (all-spell
median 5.5 s, 27% under 3 s, p75 10.5 s), so the distribution is in the same
range as organizer-style possessions.

**Anchor agreement.** On the 43,406 train anchor frames (valid 8,313, test
9,667): dense equals the anchor on 98.6% of frames where dense is known
(valid 98.4, test 98.3). 1.0% of anchors fall where dense is unknown (filled by
the anchor), 0.01% where dense says D. The 1.3-1.6% H/A disagreements are
almost all (487 of 570 on train) a timing offset: the opposite team's touch run
starts 1-3 native frames after the anchor frame (the annotated pass time is
slightly before the receiver's first touch); all are inside `propagated` gaps.

**TACKLE semantics** (metadata `team` of `PLAYER SUCCESSFUL TACKLE` against
the dense label 0.5 s before and 1 s after the event; train n=6,178 with both
known): 62% already equal the metadata team before the event and stay (3,815),
6% flip away (353), 12% flip toward (724), 21% stay opposite (1,286). So the
metadata team is the tackling (winning) side more often than not (flips toward
it are 2:1), but the event time is noisy and the dense label usually switched
before it, or never switched within 1 s. It is not a clean possession-change
anchor, so TACKLE stays excluded from anchors and any use should take the
frame stream. HEADER behaves like a receiver event (before: 65% opposite,
after: 74% same); PASS/CROSS/SHOT team is the actor and the dense label agrees
at about 98%. The metadata team of OUT is a placeholder and was never used.

**Switch spot check.** 20 random switches per split are in the local
`labels_report.json` (not committed). Typical pattern: owner run `PA`, then a
1-frame or longer controlled opposing run `PA` at the switch. Uncontrolled `CH`/`RE`
or 1-frame `CL` runs of the opponent before a switch are correctly skipped.

## Known limits

- Silver labels: a possession that changes only after a long uncontrolled
  contest is switched at the next controlled touch, not at the physical
  turnover. Runs with a missing end type at record edges are assumed
  controlled.
- Unknown (12.6% of frames) is left as -1 and should get zero loss weight.
- Games differ in side balance (test A .49 vs H .32), a property of the games.
- Record boundaries are the merged clip coverage; records are short (median
  about 11 s) because source clips are sampled sparsely.

Reproduce: `python -m experiments.soccernet_dense_labels --splits valid test train`
then `python -m experiments.soccernet_dense_qa`.
