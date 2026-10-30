# sn_dense_v1 training harness

`experiments/sn_dense_train.py` generalizes `soccernet_capacity_model.train` (frozen, unedited).
Defaults reproduce it bit-for-bit (tested). Tests: `experiments/tests/test_sn_dense_train.py`.

```python
from experiments import sn_dense_train as sn
games = sn.nested_games(all_game_ids, 22)          # SHA-256 ranked, nested across n
out = sn.train(teacher, old_records, gar_records, width=96, lr=3e-4, seed=2026, steps=2048,
               fraction=0.5,                        # 0.0 and 1.0 both valid
               ha_target="dense",                   # or "anchors"
               train_d=True, d_lr=1e-4,             # fine-tune D path from teacher weights
               soccernet_games=games,
               ball_dropout=0.0,                    # >0: feature-level ball-gap view (feed_like gap process)
               eval_every=256, patience=3, validate=sn.make_gar_validator(valid_records, teacher))
model = out["model"]                                # best validated (else final) snapshot
probability = sn.predict(record, model, teacher)    # q * (1 - P(D)); P(D) fine-tuned if train_d
```

GAR record contract: `x, names, y, period, timestamps, fps` (sampled fps), `native_fps`, `indices,
native_n, full_y, full_period, source, match` (dense: `y` in {0 H, 1 A, 2 D, -1}). Spell duration uses
`native_fps`. With `anchors` and `train_d`, D is trained on original windows only. No resume support.
The ball view is an approximation (NaN on ball-dependent columns); pass `ball_view_fn` for an exact
raw-tracking recompute. Never run the validator on the sealed GAR test split.
