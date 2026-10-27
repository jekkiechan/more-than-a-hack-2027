# Superseded D-only LightGBM companion study

Status: superseded by the user's clarification that LightGBM must directly
predict all H/A/D classes. Twelve binary D-head models were fitted and sealed;
no evaluation scores were opened. The planned 240-cell D-only evaluation was
cancelled for scope, not because of a result. The corrected active study is
[Full H A D LightGBM](JOINT_LGBM.md).

The original source and frozen declaration remain unchanged under
`data/experiments/temporal_lgbm_v1`, alongside a `scope_correction_receipt.json`.
These checkpoints are historical D-head artifacts, not a completed answer to
the requested full classifier. The four arms compared 67 original player
features with 625 bidirectional player features, holding a GRU's conditional
team attribution fixed. That restricted scope was the reason for replacement.

## Completed training cost

All 12 fits completed without changing the frozen specification. Training rows
were 141,392 with SkillCorner excluded, 179,452 with Sportec excluded, and
207,964 with Metrica excluded. The mean fitting time across folds was 39.1
seconds for raw_slow, 408.7 for context_slow, 187.8 for context_shallow and
228.9 for context_regularized. These are elapsed one-thread fit times while
other research jobs shared the machine. Feature-cache preparation is separate.

In the matched context_slow arm, new temporal columns account for 46.5% of
training split gain averaged across folds: future standard deviation 12.2%,
future mean 10.1%, future lag values 7.1%, past standard deviation 4.9%, and
smaller shares for the other context families. This establishes that the trees
use the expanded representation. Training gain is not held-out permutation
importance and cannot establish that a feature causes a generalization gain.
