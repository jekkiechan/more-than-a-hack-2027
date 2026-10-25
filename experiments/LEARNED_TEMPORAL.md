# Ordered temporal context for dead-ball detection

This experiment isolates the D head. Conditional H/A probabilities come from
one unchanged pooled multi-source contextual HGB, and every D head uses the
same Viterbi switch cost of 3. Differences therefore measure how changing
dead/live probabilities affects complete possession spells.

Six configurations were fixed before fitting: the player-only HGB baseline,
the same HGB with ordered features, matched LightGBM heads at learning rates
0.08/160 trees and 0.02/640 trees, a small noncausal TCN, and a bidirectional GRU.
The new LightGBM pair concerns **D detection**, rather than the full H/A/D
classifier. The previous full-model `lgbm_kaggle_b` already used learning rate
0.02 and 1,500 trees; its historical Sportec validation spell F1 was 0.71614.
See [REFINE_LGBM.md](REFINE_LGBM.md). This experiment does not relabel that
earlier score as a new result or run an additional open-ended tuning sweep.

Both sequence models use collective player motion/shape features from
`refine_domain.features`, sampled on a common 5 Hz grid after computing native
tracking derivatives. Neural inputs have explicit observation masks; missing
features are imputed only after fitting medians/scales on training matches.
No ball, event, possession-owner, or label field reaches the D features.
Player-only inputs intentionally avoid the frozen-ball and blank-ball shortcuts.
Provider-specific player-detection patterns can still be learned.

The ordered HGB adds preceding/following values at 1, 3, and 5 seconds, plus
future-minus-past differences. It separates slowing down from speeding up
without requiring a recurrent model. The neural models use 20-second input
windows, outputting the central 10 seconds, with five seconds of context on
each side. Windows reset at match, half, and timestamp discontinuities. Edges
are repeated inside their own input block. The 24-channel TCN has four
kernel-5 convolutions at dilations 1/2/4/8; the GRU has 24 hidden units per
direction. Each trains for eight fixed epochs with AdamW, seed 2026, and no
validation-based checkpoint/threshold sweep.

The TCN receptive field is 61 samples (about 12 seconds). Its five-second halo
is slightly shorter than the receptive-field radius, so some core-edge outputs
can depend on convolution padding. This frozen comparison retains that modest
chunk-edge limitation; it is not adjusted after looking at validation scores.

All heads use the same clean/player-slot-dropout training copies and equal
total weights by provider, then match. The HGB baseline is refitted at 5 Hz;
it is a matched ablation baseline, not an exact replay of the historical 4 Hz
`dead_transfer.py` run. All D probabilities are interpolated within continuous
tracking blocks to the native frame rate before decoding and official scoring.

## Evaluation data

The subagent-created [evaluation manifest](eval_set/manifest.json) fixes the
existing whole-match train/development assignments, file hashes, and label
provenance. [The audit](eval_set/README.md) explains each provider's shortcut and
the visibility coverage. Existing Sportec tests J03WPY/J03WR9/J03WQQ are excluded.
These are historical development matches; none is a fresh confirmation set.

Every configuration runs in pooled training and three folds withholding a
provider from **D-head training**. The H/A backbone stays fixed and has seen
training matches from all providers, so this is not an unseen-provider test
of the complete system. The five development matches are Sportec J03WOY,
Metrica game 3, and SkillCorner 1996435/1996436/2006229. The SkillCorner
>=12-visible-player filter discards most D frames; its scores apply to the
retained visible-player subset. Silver labels and censored boundaries remain
explicitly distinguished from independent manual ground truth.

The SkillCorner D-core labels have X margins at every stoppage transition, so
there are **zero eligible reference D boundaries** on those matches. Boundary
timing is assessable on Sportec and Metrica only. The unchanged official scorer
returns zero boundary F1 when reference counts are zero; the report marks those
SkillCorner rows unsupported rather than interpreting the zeros as timing
failures or averaging them with the two eligible providers.

The official spell F1 averages greedy same-team matches at IoU .3/.4/.5/.6/.7.
Report each match and equal-provider means, alongside frame macro F1, D
precision/recall, raw D AUROC, and entering/leaving-D boundary timing. Paired
bootstrap deltas resample whole matches within providers; singleton-provider
matches cannot contribute uncertainty, so these intervals do not establish
unseen-provider significance. The fixed selection rule is highest
equal-provider spell F1 on provider-holdout D rows, with macro F1 as a tie-break;
the original HGB remains eligible. No organizer labels enter selection.

## Results

All 24 fits and 60 match-level evaluations completed. Values below are equal
means across providers, after averaging matches within each provider. The full
[aggregate report](results/learned_temporal_development.json) preserves every
match, D precision/recall, raw AUROC, boundary counts/timing, paired deltas,
package versions, and the frozen selection/model hashes.

| D head | Pooled spell F1 | Provider-holdout D spell F1 | Provider-holdout D F1 |
| --- | ---: | ---: | ---: |
| Matched HGB baseline | .5748 | .5588 | .8197 |
| **Ordered features + HGB** | .5853 | **.5652** | .8223 |
| LightGBM .08 / 160 trees | .5727 | .5577 | .8195 |
| LightGBM .02 / 640 trees | .5744 | .5613 | .8187 |
| TCN | .5858 | .5605 | **.8402** |
| Bidirectional GRU | **.5891** | .5649 | .8303 |

Ordered HGB is selected by the predeclared provider-holdout spell criterion,
improving by .00635 over the matched baseline. Its .00027 lead over the GRU is
effectively a tie on this small suite, not reliable evidence of superiority.
The GRU has the strongest pooled spell score, but the provider-holdout criterion
was fixed before fitting and is not changed after seeing that result.

The TCN raises provider-holdout D F1 by .02056 and D recall from .7830 to .8207,
but spell F1 increases only .00171. Its spell change is -.00175 on Sportec,
-.00712 on Metrica, and +.01400 on SkillCorner. Better D classification alone
does not consistently recover more complete possessions. The GRU gains .00608
spell F1 over baseline, but does not clearly beat simpler ordered features.

The lower-rate LightGBM head improves spell F1 by .00360 over the .08 head, with
essentially unchanged D F1. There is little evidence here for a larger
hyperparameter search. The older full H/A/D .02 experiment remains a separate
historical Sportec result, not multi-provider confirmation.

Ordered HGB's paired match-bootstrap spell delta interval is [.00310, .01071];
the TCN interval is [-.00115, .00420]. These intervals condition on the existing
matches and cannot establish cross-provider significance: Sportec and Metrica
each have only one development match, and repeated model comparisons add
selection optimism. A new independently labeled match remains necessary.

At a 1-second tolerance, ordered HGB improves D-boundary F1 from .4464 to .4805
on Sportec and .2901 to .3945 on Metrica. Many boundaries are still missed.
SkillCorner timing cannot be assessed because of its X margins. No new organizer
score is reported; candidate generation confirmed that the tactical ATD files
are missing before it could read any public target labels.

## Reproduce

The external NPZs must be available in the existing `data/external/processed`
and `data/external_audit/sportec/processed` directories. See [OPEN_DATA.md](OPEN_DATA.md)
for source download/conversion commands. Install the existing experiment
requirements plus the optional `torch` and `lightgbm` packages. CPU-only PyTorch
is sufficient; the recorded results include actual package versions.

```sh
python -m experiments.temporal_attribution
python -m experiments.learned_temporal --prepare
python -m experiments.learned_temporal --fold pooled
python -m experiments.learned_temporal --fold without_sportec
python -m experiments.learned_temporal --fold without_metrica
python -m experiments.learned_temporal --fold without_skillcorner
python -m experiments.temporal_report
python -m unittest discover -s experiments/tests -v
python -m unittest experiments.eval_set.test_protocol -v
```

The attribution training and native feature caches take several minutes on CPU.
Training checkpoints, probabilities, predictions, and frozen source/config
hashes remain under ignored `data/experiments/learned_temporal/`. Completed
fits resume; a changed temporal source/spec requires a new output directory.
The report refuses partial candidate comparisons and freezes the selected
pooled model before any organizer pipeline check.

## Organizer five-minute check

Organizer tracking and prior model artifacts were absent from the initial
workspace and are not in the external-data cache. Once the tactical ATD assets
are under `data/tactical` (or `MTAH_DATA_DIR`), or a previously generated
`data/challenge/inputs.npz` is supplied:

```sh
python -m experiments.temporal_challenge generate
python evaluation/validate_submission.py data/experiments/learned_temporal/challenge/selected.csv
python -m experiments.temporal_challenge check --previous-csv path/to/previous_candidate.csv
```

Generation reads no organizer labels, writes complete 199,319-row files, and
validates them with the official validator. The optional previous candidate
must be supplied for an exact file-to-file comparison. Both the selected head
and matched HGB baseline are frozen before the check. An exclusive marker
prevents repeated public-window checks and regeneration after checking starts.
Under `RULES.md`, this is one descriptive pipeline check, never a tuning or
selection step. Public-window improvement does not establish full-match gain.
