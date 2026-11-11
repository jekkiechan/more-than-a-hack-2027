# Pooled CE checkpoint reconstruction

**Status: training reconstruction verified; deployment checks in progress.** The original pooled CE stage-1024
checkpoints are unavailable. This bounded study recreates their frozen training
recipe so the reconstructed package can be checked as the Fable input. A new
fit remains a reconstruction, even when its training trace and predictions
match the historical run; it is not a restore of the original checkpoint file.

The [plan](results/frontier_pooled_reconstruction_v1_plan.json) fixes four
serial CPU fits: seeds 2026 and 2027, each at 512 updates with learning rate
`3e-4`, then 512 more at `1e-4`. Stage 512 starts from the exact adopted pooled
H/A model and its paired teacher. Stage 1024 starts from the reconstructed
stage-512 checkpoint. The trainer, 32-row batches, 50/50 original/GAR mix,
distinct-possession sampler, four-view round-robin schedule and repeated seed
draw schedule follow the historical pooled CE recipe. D weights and
normalization stay fixed.

The training pool contains the same 13 original TRAIN matches and 45 GAR TRAIN
games. Raw inputs, prepared caches, cache receipts and the v1 views are checked
against their original manifests. All 52 regenerated v2 feature files and all
52 sidecars match the historical training declaration byte-for-byte. The
preparation receipt records one worker instead of the historical two; changing
only that field for comparison reproduces the historical receipt hash exactly.
GAR VALID/TEST data and organizer labels are excluded.

All four fits completed and passed the declared reconstruction checks. Sampling,
exposures, view draws, D state and normalizers match exactly. The 16 loss
intervals per fit remain within the frozen `1e-7` tolerance:

| Seed | Updates | Maximum loss-interval difference |
| --- | ---: | ---: |
| 2026 | 512 | 1.210719347e-08 |
| 2026 | 1024 | 1.303851604e-08 |
| 2027 | 512 | 1.816079021e-08 |
| 2027 | 1024 | 1.583248377e-08 |

All four checkpoint file hashes differ from the originals. The
[training seal](results/frontier_pooled_reconstruction_v1_seal.json) and
[verification receipt](results/frontier_pooled_reconstruction_v1_verification.json)
record the new identities separately. Deployment prediction parity and fixed
Fable regression checks remain pending.

The [frozen declaration](results/frontier_pooled_reconstruction_v1_declaration.json)
and [runner](frontier_pooled_reconstruction.py) keep these artifacts separate
from the historical declaration and seal. The original stage-1024 file hashes
are `d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2`
(2026) and `4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa`
(2027). Neither hash will be presented as restored unless the reconstructed
file itself matches it.

After the four fits, the saved reference-free 199,319-row frontier CSV must
match SHA-256
`26af7348940626da1cbe05a0c73b2a9db41ad9cedcb3ae6adc5de2771b6ce5dc`
before the package is described as prediction-equivalent. The fixed Fable GAR
checks and the single planned public diagnostic remain pending. They use the
exact reconstructed pooled package and do not authorize additional candidate
search or tuning. No organizer reference labels have been used.

To reproduce the bounded fit sequence after restoring the pinned private TRAIN
inputs, use the pinned experiment environment and run each fit in its own
process, in order:

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
