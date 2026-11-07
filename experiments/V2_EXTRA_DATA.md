# Extra GAR data: targeted continuation

Study `v2_extra_data_v1` · 7 November 2026 · execution in progress.

The user requested further tests after approving the frozen
[spell-frontier blend](V2_SPELL_FRONTIER.md) for promotion. This study uses that
blend as its development control. The previous public diagnostic is closed;
no organizer reference access or public scoring is authorized by this study.
The separate promotion retains the exact previously approved weights.

## Question and declared design

Can a targeted change in how existing GAR TRAIN data contributes improve the
frontier blend while retaining clean, v1 and v2 spell F1? Prior scaling,
capacity, v1/v2 mixing and longer continuation results remain preserved.
This is a shorter, paired continuation with two isolated data interventions.

| Arm | GAR loss share | GAR quality replacement rate |
| --- | ---: | ---: |
| Fresh continuation control | 0.500 | 0.25 |
| More GAR objective weight | 0.625 | 0.25 |
| More corroborated short GAR possessions | 0.500 | 0.50 |

All 18 fits start from the exact balanced stage-1024 checkpoint: three excluded
providers × two fixed seeds × three arms. Each gets 256 updates at learning rate
0.0001, ending at stage 1280. Batches retain 16 original-provider and 16 GAR
windows; the objective weight is not a sampling fraction. Original short-span
replacement remains 0.25. Fresh schedules offset the original distinct-span
sampler and trainer draw RNG by 104729, with paired Torch/dropout seeds.
The original clean mass and four fixed v2 TRAIN views, teacher, normalization,
D parameters, targets and architecture are unchanged.

The eligible pool is the same 13 original TRAIN matches (whole-provider
exclusion in each fold) and 45 GAR TRAIN games. Corroboration uses the existing
TRAIN-only generated-label and geometric checks. It changes sampling, not
labels or inference. These are additional draws from existing silver data,
not new annotations. GAR VALID/TEST are excluded from fitting.

## Comparison and selection

The deployed recipe stays fixed: CE 0.5, calibrated 0.375 and predecessor 0.125,
arithmetic conditional H/A pooling and H/A transition cost 3, with the exact
predecessor decoded D mask. Average the two seeds within each native phase,
then phases before decoding. Score the same 35 development cells: five clean,
ten v1 and twenty v2. Average realizations within match, matches within
provider, and providers equally. Per-seed scores are diagnostics, not separate
candidate pipelines.

The control has clean/v1/v2 spell F1 **0.670137/0.532128/0.593619**. Each candidate
must equal or improve all three spell means, strictly improve v2, retain
short-spell recall in every view, and retain macro F1 within 0.001 per view.
D masks and reference counts must be identical. Rank eligible candidates by
unrounded v2 spell F1, macro F1 for exact ties, then ID. The aspiration remains
v2 spell F1 above 0.60. No automatic promotion of a new arm follows.

Control replay precedes fitting/scoring. Candidate weights, evaluation code,
cohort identities and recipe are sealed before candidate evaluation. Reports
will distinguish the paired continuation effect from each data intervention,
including every failed guard and provider/seed spread.

## Evidence and limitations

- [Frozen plan](results/v2_extra_data_v1_plan.json).
- [Previous frozen selection](results/v2_spell_frontier_v1_selection.json).

These five matches have been repeatedly exposed to development. The v2 views
are synthetic and model-conditioned. Any gain here needs further confirmation;
GAR silver segments and the reused 17-possession public window do not establish
independent continuous-match gold performance. The exact previously approved
package's GAR TEST check is a separate deployment diagnostic, not a source of
training choices for this study.
