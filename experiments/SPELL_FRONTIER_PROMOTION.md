# Spell-frontier package promotion

The user explicitly requested promotion of the frozen
`base__ce0.5_cal0.375_ad0.125_arithmetic_ha3` blend after its final diagnostic.
The package is now adopted as `joint_gru_sn_dense_spell_frontier_ensemble_v1`.
Activation completed after the exact deployment check and normal-CLI replay.
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
with its recorded source revision. The active manifest, policy and default CLI now route to the approved package.
The normal CLI reproduces the exact sealed CSV; the full format check passes.

## Exact deployment diagnostic and execution record

The fixed package is checked on GAR TEST10 and selected-on VALID9 separately,
using clean inputs and four exact predecessor v2 realizations. These are reused
silver segments, not new independent gold confirmation. The normal 29.97 Hz
GAR input requires six native phases. All models and the mixture were fixed
before this diagnostic; the results do not select the extra-data arms.

The frozen execution records preserve the following corrections:

- An early one-game probe was stopped before a declaration; no game output or
  metrics were retained. The [aborted-attempt receipt](results/spell_frontier_gar_v1_aborted_predeclaration.json)
  records the deviation. The original plan and receipt were restored to their
  exact bytes after an accidental edit; the separate
  [source-guard amendment](results/spell_frontier_gar_v1_sourceguard_amendment.json)
  adds the native grading-population check.
- The declaration's seed description omitted the heldout helper prefix. The
  executable path already uses the exact predecessor seed; the
  [prose erratum](results/spell_frontier_gar_v1_seed_description_erratum.json)
  records the full expression and expected seeds without changing execution.
- The first declared game failed before decoding/scoring because the reused
  public pooling helper accepts only two/five phases. The
  [six-phase amendment](results/spell_frontier_gar_v1_sixphase_amendment.json)
  preserves the traceback and adds six-phase support with identical float64
  seed-then-phase averaging. Two/five-phase deployment behavior is unchanged.
- The [capture runtime amendment](results/spell_frontier_gar_v1_capture_runtime.json)
  applies the existing, exactly tested quantile acceleration during native
  input capture as well as feature preparation. Phase-0 feature equality and
  all per-game predecessor score guards remain unchanged at 1e-12 tolerance.

The original frozen runner, package adapter, public helper, declarations and
failed attempts remain intact. Separate wrappers apply the bounded execution
corrections. The [audit plan](results/spell_frontier_gar_v1_audit_plan.json)
requires independent official rescoring of all 95 paired game/view cells and
equal-game reductions, with descriptive paired bootstrap intervals.

## Completed GAR deployment results

| Split / view | Frontier spell | Predecessor spell | Paired Δ spell | 90% paired game interval | Δ macro | Δ short recall | Games improving spell |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| TEST / clean | 0.708496 | 0.701641 | +0.006855 | [+0.003107, +0.010824] | +0.001507 | -0.008267 | 8/10 |
| TEST / v2_mean_over_four_reps | 0.708619 | 0.701289 | +0.007330 | [+0.004671, +0.009839] | +0.001678 | -0.005504 | 9/10 |
| VALID / clean | 0.705900 | 0.700777 | +0.005123 | [+0.001150, +0.009515] | +0.001263 | +0.004292 | 6/9 |
| VALID / v2_mean_over_four_reps | 0.700924 | 0.697917 | +0.003007 | [+0.000529, +0.005494] | +0.000860 | -0.006289 | 7/9 |

GAR TEST short-spell recall declines despite improved spell and macro F1: -0.008267 clean and -0.005504 v2. Thus the approved package is not an all-metric improvement on this silver cohort. Original-provider short-recall retention and this separate deployment tradeoff must not be conflated.

Four v2 realizations are averaged within each game before equal-game aggregation. TEST10 and previously selected-on VALID9 stay separate. Intervals resample whole paired games (2,000 draws, fixed seed), are descriptive 90% intervals, and do not supply fresh independent confirmation. All 95 paired cells pass independent official rescoring, for 190 scorecards; D masks, grading populations and reference spell counts are exact. All actual corruption seeds match the predecessor contract.

- [Complete GAR result](results/spell_frontier_gar_v1.json), [independent score audit](results/spell_frontier_gar_v1_verification.json), [runtime and seed verification](results/spell_frontier_gar_v1_execution_verification.json).
- [Pre-activation provenance checks](results/baseline_spell_frontier_pre_activation.json), [completed promotion receipt](results/baseline_spell_frontier_promotion.json), [local archive verification](results/baseline_spell_frontier_local_artifacts.json).

## Delivery

[PR #87](https://github.com/jekkiechan/more-than-a-hack-2027/pull/87) includes the
completed promotion and follow-up studies. All 841 experiment and 26 root tests
pass, alongside the exact file-backed fit contract, CSV format and documentation
checks. The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
is published as version 34 with its existing audience unchanged.
[Validation](results/baseline_spell_frontier_validation.json) and
[delivery receipt](results/v2_extra_data_v1_delivery.json).
