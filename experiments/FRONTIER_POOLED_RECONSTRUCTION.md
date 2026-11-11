# Pooled CE checkpoint reconstruction

**Status: four-fit reconstruction, fixed deployment checks and guarded
research-branch promotion passed.** The original pooled CE stage-1024
checkpoints are unavailable. This study recreates their frozen training recipe
so a reconstructed package can be evaluated as the Fable input. The fits are
reconstructions, not restores of the original checkpoint files.

The [frozen plan](results/frontier_pooled_reconstruction_v1_plan.json) bounded
the work to four serial CPU fits: seeds 2026 and 2027, each trained for 512
updates at learning rate `3e-4`, then 512 more at `1e-4`. Stage 512 starts from
the byte-identical adopted pooled H/A model and paired teacher; stage 1024
starts from the reconstructed stage-512 checkpoint. The frozen trainer,
32-window batches, 50/50 original/GAR mix, distinct-possession sampler,
four-view round-robin schedule, repeated seed draws, D weights and
normalization were retained. See the [runner](frontier_pooled_reconstruction.py),
[declaration](results/frontier_pooled_reconstruction_v1_declaration.json),
[fit seal](results/frontier_pooled_reconstruction_v1_seal.json), and
[independent verification](results/frontier_pooled_reconstruction_v1_independent_verification.json).

The pool contains the same 13 original TRAIN matches and 45 GAR TRAIN games.
Raw inputs, prepared caches, receipts and the v1 views were checked against
their historical declarations. All 52 regenerated v2 feature files and all 52
sidecars match the historical training declaration byte-for-byte. The actual
preparation receipt records one worker rather than the historical two; changing
only that metadata field reproduces the historical receipt hash. GAR VALID/TEST
data and organizer labels were excluded from fitting.

All four fits passed the frozen parity checks. Sampling, exposures, view draws,
D state and normalizers match exactly. Each of the 16 loss intervals per fit is
within the declared `1e-7` tolerance:

| Seed | Updates | Maximum loss-interval difference |
| --- | ---: | ---: |
| 2026 | 512 | 1.210719347e-08 |
| 2026 | 1024 | 1.303851604e-08 |
| 2027 | 512 | 1.816079021e-08 |
| 2027 | 1024 | 1.583248377e-08 |

The largest observed difference is `1.816079021e-08`; independent checks covered
220 training-input entries and 48 source entries with zero mismatches. The
reconstructed stage-1024 SHA-256 values are
`da1f62e8d309397a7280fa12e1a79a88a8c9f25012eaea5b8333a400299ef066` (2026) and
`fdc3b02dc997abc0b164ff8991c5a32b80ff52b1078efe1072e2e862985b065b` (2027).
They differ from the historical originals
`d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2` and
`4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa`.
The historical hashes are not represented as recovered.

Deployment inference matched the saved reference-free 199,319-row frontier
CSV byte-for-byte (SHA-256
`26af7348940626da1cbe05a0c73b2a9db41ad9cedcb3ae6adc5de2771b6ce5dc`). The
fixed Fable CSV is sealed separately as
`56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f`. Both used
the normal deployment CLI without reference access.

The fixed public diagnostic used the already-closed window of 7,500 graded
frames and 17 possessions. Frontier to Fable spell F1 is `.611765` → `.623529`
(+.011765), macro F1 `.851285` → `.864234` (+.012949), and D-F1 `.862870` →
`.889372` (+.026502). No tuning or candidate selection followed; the small,
reused public window is a diagnostic, not independent gold confirmation.

The fixed GAR TEST10 replay compares clean and four v2 views for ten games.
Across the 50 paired game/view cells, 100 prediction arrays and 800 scalar
comparisons replayed independently. Equal-game means were:

| View | Metric | Frontier control | Fable snap | Change |
| --- | --- | ---: | ---: | ---: |
| Clean | Spell F1 | .708496 | .709186 | +.000690 |
| Clean | Macro F1 | .780580 | .784717 | +.004138 |
| Clean | D-F1 | .583474 | .593243 | +.009768 |
| Clean | Short-spell recall | .263398 | .264521 | +.001124 |
| V2 | Spell F1 | .708619 | .709176 | +.000557 |
| V2 | Macro F1 | .782179 | .784865 | +.002686 |
| V2 | D-F1 | .589851 | .596512 | +.006661 |
| V2 | Short-spell recall | .262769 | .263570 | +.000801 |

All four aggregate metrics improve in both views. Per-game spell means do not
improve uniformly: clean has 7 improvements, 1 tie and 2 declines; v2 has 8
improvements and 2 declines. Per-game changes range from −.002232 to +.003204
clean and −.001121 to +.002133 v2. These are small means on repeated TEST
exposure with silver labels, not a fresh or gold evaluation. VALID remains
separate; there was no tuning or candidate selection from these results.
The [GAR declaration](results/ball_exit_reconstructed_v1_gar_declaration.json)
and [result](results/ball_exit_reconstructed_v1_gar.json) retain the full
per-game evidence.

The reconstructed package contains all ten deployment roles and the four fit
artifacts in private model storage. Its versioned cache manifest is
`baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1.json`.
A fresh remote clone with empty LFS and no local references verified all 23
manifest files. The [archive receipt](results/ball_exit_reconstructed_v1_archive_verification.json)
records that verification. The tested package branch is
`research/frontier-pooled-reconstruction-v1` at `d5d93aa`; no private PR was
created. The staged package includes models and manifest, not raw inputs or
frame predictions. The [promotion receipt](results/ball_exit_reconstructed_v1.json)
records activation on this research branch. Main-repository PR #92 is open and unmerged.

After activation, the normal default CLI replay reproduced the sealed Fable
CSV byte-for-byte (SHA-256
`56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f`). The
post-promotion suite passed 978 tests (977 passed, one skipped), alongside the
26 root tests, file-backed parity, submission-format check and independent
reconstruction audit. The consolidated [deployment verification receipt](results/ball_exit_reconstructed_v1_verification.json)
binds the archive check, default CLI parity and post-promotion test results.

To restore the complete deployment package from an authorized `mtah-data`
checkout, fetch and verify its versioned cache manifest, then restore it into
the competition repository's `data` directory:

```sh
python scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1.json
python scripts/competition_cache.py verify --manifest baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1.json
python scripts/competition_cache.py restore --manifest baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1.json --data-dir ../more-than-a-hack-2027/data
```

The training CLI below records how this bounded fit was executed. It expects
the pinned TRAIN input bundle materialized at
`data/experiments/frontier_pooled_reconstruction_v1/inputs/` and the frozen
environment. The deployment cache contains models, not those training inputs.
For a replay, set the data and archive roots before `prepare`:

```sh
export MTAH_DATA_DIR=data/experiments/frontier_pooled_reconstruction_v1/inputs/original
export MTAH_RF_ARCHIVE_DIR=data/experiments/frontier_pooled_reconstruction_v1/inputs/rf/experiments/feed_robustness_v1/artifacts
export V2_ITERATIVE_GAR_ARCHIVE=data/experiments/frontier_pooled_reconstruction_v1/inputs/gar/gar/train
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Rerunning these commands would create another reconstruction and must not
overwrite the sealed study artifacts.

```sh
.venv/bin/python -m experiments.frontier_pooled_reconstruction prepare
.venv/bin/python -m experiments.frontier_pooled_reconstruction freeze
.venv/bin/python -m experiments.frontier_pooled_reconstruction fit-one --seed 2026 --stage 512
.venv/bin/python -m experiments.frontier_pooled_reconstruction fit-one --seed 2026 --stage 1024
.venv/bin/python -m experiments.frontier_pooled_reconstruction fit-one --seed 2027 --stage 512
.venv/bin/python -m experiments.frontier_pooled_reconstruction fit-one --seed 2027 --stage 1024
.venv/bin/python -m experiments.frontier_pooled_reconstruction seal
.venv/bin/python -m experiments.frontier_pooled_reconstruction verify
```
