# Complementing Fable with turnover supervision

11 November 2026. **Retain one small development improvement:** use the existing
75% hard-turnover / 25% fresh-continuation H/A blend only on live stretches whose
boundaries Fable's snap correction leaves unchanged. V2 spell rises from
**.5986897355 to .5987797447**, a gain of **.0000900092**. Clean/v1 spell and
short recall in every view also improve; every declared retention check passes.
This is a small point gain on exposed development data, not established
generalization or a new deployment. The .60 target remains unmet.

The user requested a complementary approach instead of further extensions to
the exit rule. The [earlier crossing-time extensions](BALL_EXIT_PROMOTION.md)
remain failed. Both studies below keep Fable's `snap_m0.0` mask exactly fixed;
only H/A labels within live play can change. Fable's unchanged implementation
is now the user-authorized provisional default, with its
[exact deployment artifact blocker](BALL_EXIT_PROMOTION.md#exact-deployment-artifact-blocker).

## First comparison: direct composition

The [turnover-supervision study](TURNOVER_SUPERVISION_V1.md) already produced
three useful fixed H/A endpoints: fresh continuation, hard-turnover continuation,
and its retained .75/.25 CE blend. They are complementary in mechanism to a
dead-ball mask correction. This study adds no fits, mixture-weight search,
decoder search, or provider-specific switches.

| H/A predictor with snap | Clean spell | V1 spell | V2 spell | Failed retention checks vs snap |
| --- | ---: | ---: | ---: | --- |
| Existing frontier | .681508 | .534731 | .598690 | Control |
| Fresh continuation | .681343 | .534856 | .599644 | Clean spell; v1/v2 short recall |
| Hard-turnover continuation | .682619 | .535980 | .598553 | V2 spell; v1/v2 short recall |
| Retained .75 hard / .25 fresh | .681442 | .535779 | .599075 | Clean spell; v1 short recall |

The numerical leader's +.000954 v2 gain is not an eligible improvement.
The retained mixture's old clean gain did not carry through: its clean
interaction is −.000778 relative to adding the separately measured gains.
The original-mask endpoints remain controls, not new selection candidates.

The design crosses two D masks (original and snap) with four H/A predictors,
including frontier. All **175 control scorecards** replay before any of the
three new snap compositions is scored. All eight arms' **280 saved outputs /
1,680 metric values** independently replay with zero error; exact D masks,
aggregation, guards and selection also verify.

- [Declaration](results/ball_exit_turnover_complement_v1_declaration.json), committed before scoring at `d0bb5ca`.
- [Full factorial result](results/ball_exit_turnover_complement_v1.json), including per-view interaction terms.
- [Independent verification](results/ball_exit_turnover_complement_v1_verification.json).

## Second comparison: keep the changes on separate live stretches

This is explicitly a post-result design, declared separately at `0ad8453`.
The hypothesis is that H/A changes should not disturb the intervals Fable
has recovered. Use the already-scored complementary predictions only where
an entire live interval has **exactly the same start and end** under the
original and snap masks, within the same model time block. Else retain snap's
frontier H/A labels. No duration threshold or confidence parameter is added.

| H/A predictor on unchanged intervals | Clean spell | V1 spell | V2 spell | Retention |
| --- | ---: | ---: | ---: | --- |
| Snap frontier | .681508 | .534731 | .598690 | Control |
| Fresh continuation | .681566 | .535047 | .598844 | Fails v1/v2 short recall |
| Hard-turnover continuation | .681677 | .535183 | .598421 | Fails v2 spell/short recall |
| **Retained .75 hard / .25 fresh** | **.681677** | **.534829** | **.598780** | **Passes every check** |

The eligible composition's clean/v1/v2 spell changes are
**+.000169395 / +.000098055 / +.000090009**. Short recall changes are
+.000740741 / +.000593958 / +.000137980; macro changes are
+.000044869 / +.000045752 / +.000009029. D metrics are unchanged by construction.
The v2 provider changes are −.000211 Metrica, +.000564 SkillCorner, and
−.000084 Sportec: the small mean gain comes from SkillCorner, not every provider.

All **140 saved outputs / 840 metric values** independently replay with zero
error. An additional check reconstructs the interval eligibility using interior
live states and both endpoint transitions, then verifies all **105 candidate
outputs** and their exact D masks.

- [Declaration](results/ball_exit_turnover_disjoint_v1_declaration.json).
- [Result](results/ball_exit_turnover_disjoint_v1.json).
- [Metric/selection verification](results/ball_exit_turnover_disjoint_v1_verification.json).
- [Independent composition/mask verification](results/ball_exit_turnover_disjoint_v1_mask_verification.json).

## Scope, costs and next step

Both rounds use the same five exposed provider-held-out matches, clean, two v1
realizations and four fixed v2 realizations. Average realizations within match,
matches within provider, and providers equally. All existing weights and
two-seed/native-phase pooling are frozen. V2 gaps are label-conditioned synthetic
stress; calibration is model-conditioned. These are not independent matches or
continuous gold confirmation. No new seed selection or component-seed inference
is performed; the historical CE-seed diagnostics remain descriptive only.

The 2,506,037,599-byte turnover archive was downloaded from its pinned private
branch, and both parts, the concatenated archive, and all 159 members verified
([restore receipt](results/ball_exit_turnover_complement_v1_restore.json)).
The new experiment uses cached probabilities; it does not retrain or substitute
provider-fold weights for pooled deployment weights. Its frame predictions
remain local and ignored; no new raw-data or weight upload occurred.

Retain the eligible composition as a research candidate, and stop this bounded
composition comparison without finer gates or blend ratios. The measured gain
is too small to establish that additional deployment complexity is worthwhile.
It would require the pooled fresh/hard counterparts, plus the frontier fallback;
no such package was fitted or timed here. The user subsequently delegated the choice; Fable is now the provisional default
as an explicit simplicity/cost tradeoff, preserving this composition as the
eligible numerical leader. Complete Fable’s exact deployment/regression checks
once its two missing pooled CE files are restored.
No GAR TEST or organizer scoring occurred in either complementary study.

Run from the repository root after restoring the declared private archives:

```sh
python -m experiments.ball_exit_complement run
python -m experiments.ball_exit_disjoint run
python -m experiments.ball_exit_verify experiments/results/ball_exit_turnover_disjoint_v1.json /tmp/disjoint-replay.json
```

The runners refuse to overwrite completed/partial runs. Existing results should
be verified with the last command, rather than rerun under a closed study ID.
The 28 focused ball-exit tests pass. The full suite has 958 tests: 952 pass,
one skips, and five inherited public-contract tests lack historical local-only
seal/CSV fixtures. Root tests, documentation, file-backed fit parity, and the
bundled submission format check pass. The required research-site update is published as version 38; see
[delivery checks](BALL_EXIT_PROMOTION.md#delivery-checks-and-site-update).

## Publication status

The user approved the prepared delivery. The implementation and aggregate reports
are published in [draft PR #92](https://github.com/jekkiechan/more-than-a-hack-2027/pull/92)
on `research/ball-exit-promotion-v2`, stacked on Fable’s PR #91 and targeting the
fork’s main. No weights, raw inputs or frame predictions were published.
The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
now includes both complementary rounds, the negative crossing extensions and
the approved Fable promotion status. Version 38 preserves its restricted audience.
The [publication receipt](results/ball_exit_delivery_publication_v1.json) records
this delivery; it does not constitute deployment verification or model activation.
