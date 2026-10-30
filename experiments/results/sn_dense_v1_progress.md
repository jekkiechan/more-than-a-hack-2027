# sn_dense_v1 progress (aggregates only; seed 2026 unless stated)

Dev = 15 development cells (provider-held-out fold model on its provider, equal-provider mean). Valid = GAR-valid dense silver, mean of the 3 fold models. Aggregates only.

### GAR-valid references (dense silver)

| model | spell F1 | macro | D-F1 | short-rec |
|---|---|---|---|---|
| pooled teacher_2026 | 0.6387 | 0.7282 | 0.4737 | 0.2462 |
| pooled capacity_2026 | 0.6642 | 0.7401 | 0.4763 | 0.2553 |
| pooled teacher_2027 | 0.6363 | 0.7263 | 0.4634 | 0.2513 |
| pooled capacity_2027 | 0.6680 | 0.7390 | 0.4698 | 0.2594 |
| pooled teacher_ensemble | 0.6481 | 0.7364 | 0.4846 | 0.2462 |
| pooled capacity_ensemble | 0.6751 | 0.7446 | 0.4846 | 0.2523 |

### C1 arms (fraction .5)

| tag | steps (best) | dev clean spell | d vs R | d vs .626425 | dev clean macro | dev clean D-F1 | dev feed spell | feed d vs R | valid spell | valid macro | valid D-F1 | valid short-rec | guard | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher (fold) | 0 | 0.6264 | | | 0.8461 | 0.8354 | 0.4872 | | 0.6218 | 0.6970 | 0.3975 | 0.2764 | | |
| R_f050_g45_s2026 | 2048/2048/2048 | 0.6288 | +0.0000 | +0.0024 | 0.8472 | 0.8351 | 0.4946 | +0.0000 | 0.6628 | 0.7112 | 0.3954 | 0.2787 | ok | 21 |
| D1_f050_g45_s2026 | 2048/2048/2048 | 0.6272 | -0.0016 | +0.0007 | 0.8478 | 0.8334 | 0.4939 | -0.0007 | 0.6744 | 0.7140 | 0.3944 | 0.2737 | ok | 21 |
| D2_f050_g45_s2026 | 2048/2048/2048 | 0.6355 | +0.0067 | +0.0091 | 0.8506 | 0.8364 | 0.5018 | +0.0073 | 0.6965 | 0.7791 | 0.5686 | 0.3377 | ok | 28 |
| D3_f050_g45_s2026 | 8192/7168/7168 (8192/7168/7168) | 0.6192 | -0.0096 | -0.0072 | 0.8390 | 0.8158 | 0.4985 | +0.0039 | 0.7249 | 0.8078 | 0.6280 | 0.3662 | FAIL | 151 |


