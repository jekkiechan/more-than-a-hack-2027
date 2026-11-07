# Spell-frontier package promotion

The user explicitly requested promotion of the frozen
`base__ce0.5_cal0.375_ad0.125_arithmetic_ha3` blend after its final diagnostic.
The package is staged as `joint_gru_sn_dense_spell_frontier_ensemble_v1`.
Activation follows the exact deployment check and normal-CLI replay.
The separate [extra-data study](V2_EXTRA_DATA.md) does not change this choice.

## Approved behavior

The H/A mixture uses continuation weight 0.5, calibrated weight 0.375 and
predecessor weight 0.125, arithmetic pooling and transition cost 3. Average the
two seeds within each native phase, then average phases. D predictions remain
exactly the predecessor's decoded D mask. The two CE models are the already
sealed pooled stage-1024 weights; promotion performs no fitting.

| Exposed development view | Predecessor spell | Approved blend spell | Δ spell | Δ macro |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.667556 | 0.670137 | +0.002581 | +0.000593 |
| Historical v1 | 0.531576 | 0.532128 | +0.000553 | −0.000699 |
| V2 | 0.583571 | 0.593619 | +0.010048 | +0.002229 |

The >0.60 v2 aspiration and historical +0.01 clean improvement screen remain
unmet. User authorization changes adoption, not those historical outcomes.
Provider regressions and the lower v1 spell score relative to the original
promising arm remain in the [frozen frontier report](V2_SPELL_FRONTIER.md).
Independent continuous-match gold confirmation remains pending.

The already-recorded public diagnostic is **0.611765 spell / 0.851285 macro**
on 7,500 graded frames and 17 possessions. Public spell is unchanged versus the
predecessor. The closed gate is preserved; there is no reference access or
public rescoring during promotion or the extra-data follow-up.

## Reproduction and availability

The new package contains all ten named checkpoint roles, including paired
teachers. Its feature-only adapter reproduces the sealed full prediction
exactly: 199,319 rows, SHA-256
`26af7348940626da1cbe05a0c73b2a9db41ad9cedcb3ae6adc5de2771b6ce5dc`.
The [adapter replay receipt](results/baseline_spell_frontier_adapter_replay.json)
binds the staged manifest and output. Models and predictions remain in ignored
local storage. No new private-cache upload is claimed.

The predecessor manifest and policy are archived without changing their bytes.
Historical declarations keep their original hashes; replay a historical study
with its recorded source revision. The active manifest, policy and default CLI
are changed only after the staged replay and deployment check complete.
