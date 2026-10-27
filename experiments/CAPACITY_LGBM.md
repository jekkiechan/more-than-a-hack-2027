# Cancelled D-only LightGBM capacity extension

Status: cancelled during fitting after the user clarified that LightGBM must
predict the full H/A/D distribution. Both running workers were stopped. No
capacity checkpoint completed and no evaluation score was opened; this was a
scope correction, not a result-based selection. The corrected active study is
[Full H A D LightGBM](JOINT_LGBM.md).

The original declaration and source snapshot remain unchanged in
`data/experiments/capacity_lgbm_v1`, with a `scope_correction_receipt.json`.
The abandoned plan increased a binary player-only D head from 31 leaves/depth7
to 63/depth8 and 127/depth10, with learning rate .01 and 1280 binary rounds.
Those settings did not constitute a full H/A/D LightGBM classifier or a full
three-class GRU ensemble. They are not reported as tested capacity improvements.
