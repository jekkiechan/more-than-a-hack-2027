# Direct weight edits after v2 model reconstruction

Status: running. The six provider-held-out checkpoints have been reconstructed
under a new study ID from the frozen recipe and verified sources. The seven-arm primary and post-selection v1 development readouts are complete.
Two pooled fits are sealed. The authorized one-use public diagnostic has run;
GAR TEST deployment remains pending. This report will be updated from frozen
receipts as those stages finish.

Date / branch / source: 4 November 2026; `research/weight-edit-v2-spell`.
The active baseline remains `joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.
The reconstruction declaration is
[`v2_weight_edit_reconstruction_v1_declaration.json`](results/v2_weight_edit_reconstruction_v1_declaration.json).
Its private-data restoration and source hashes are recorded in the ignored
`data/experiments/v2_weight_edit_reconstruction_v1/data_ready_receipt.json`.

## Why test weight edits here?

PR82's four-view short-possession model reached **.592749 v2 spell F1** and
**.665355 clean spell F1**. The adopted package was **.583571 v2** and
**.667556 clean**. Thus the prior candidate improved the exposed v2 development
mean by .009178, lost .002201 on clean input, and remained below the .60 target.
That winner had no matched v1, GAR TEST, or public result for this version.
These numbers describe PR82's original model, not the current reconstruction or
any weight edit.

The exact six PR82 checkpoints and their 52 prepared v2 training views were not
available from the reachable private archive. The reconstruction therefore
reuses PR82's frozen recipe and verified TRAIN sources to fit six new
provider-held-out models. New weights and hashes are recorded; byte identity to
the PR82-local models is not claimed. This reconstruction stage uses its
declared optimizer recipe. It is distinct from the later edits, which change
selected tensors directly and do not run an optimizer.

## Question and method

Can a TRAIN-only diagnosis of delayed H/A transitions identify a direction
whose fixed edit ranks best on v2 after passing all predeclared clean, provider,
and v2 eligibility guards? The matched v1 result is reported as a
post-selection diagnostic; it is not part of the candidate ranking rule.

The inspiration is the reported Qwen weight-space edit: locate an
activation direction associated with a repeated behavior, then remove that
component from selected weights. The analogy is limited. The possession model
is a bidirectional GRU with a conditional H/A path and a separate D path, not a
Qwen transformer or its feed-forward blocks. The rank-one edit targets the
GRU's H/A output projection using an activation direction measured only on
eligible TRAIN matches. A second bounded mechanism shifts the GRU update-gate
input bias to test recurrent persistence; this is specific to GRUs and has no
Qwen counterpart. Neither mechanism edits the dead-play path.

For the rank-one direction, TRAIN-only H/A boundaries are grouped into
one-sample-late predictions (one 0.2-second step at the model’s 5 Hz inference
rate) and correctly timed controls. The transition must remain within one
inference sequence; sample-sequence edges and transitions spanning multiple
delayed samples are excluded. This model-chunk boundary rule limits what the
diagnostic represents. Each eligible match contributes its own group means,
so a match with more frames cannot dominate the direction. Each reconstructed conditional H/A state has 291,842 scalar parameters across
10 tensors; the separate D state has 63,051 parameters and is untouched. The
direct-edit arms remove 25%, 50%, or 100% of the rank-one response from the
conditional H/A `output.weight` matrix (shape 2×192), preserving its common-mode
component. This changes at most 384 matrix values in one tensor per fold/seed
model. The D path and all other tensors remain unchanged. The update-gate arms add
−0.25 or −0.50 to, or +0.25 as an opposite-sign control on, the update slices
of both forward and reverse GRU input-bias tensors; recurrent biases and all
other tensors remain unchanged; each bias arm changes two input-bias tensors and 192 scalar values per
fold/seed model (96 update-gate entries in each direction). These seven arms, including an unedited control,
change tensors directly and run no optimizer. Candidate strengths and decisions must
be frozen before held-out labels are read.

The six reconstructed models now have TRAIN-only edit diagnostics. The eligible
original-provider pool has 13 distinct matches (3 Sportec, 2 Metrica, 8
SkillCorner); each provider-excluded fold uses all eligible matches from the two
remaining providers: 10 for Sportec holdout, 11 for Metrica holdout, and 5 for
SkillCorner holdout. Each model’s diagnostic has 40, 44, or 20
match×view×orientation rows for those folds (clean and first fixed v2 view,
native and swapped orientations). In fit-receipt
order (Sportec seeds 2026/2027, Metrica 2026/2027, SkillCorner 2026/2027), the
late/ontime activation-row counts are 556/1,356; 618/1,400; 758/1,954;
708/1,962; 630/2,442; and 540/2,404. These counts combine clean and first-v2
views and native/swapped orientations; they are observations, not unique event
counts.

The reconstruction uses six provider-held-out fits: seeds 2026 and 2027, with
Metrica, SkillCorner, or Sportec excluded from each fold's learned inputs.
Verified original-provider TRAIN arrays/records and the archived 45-game GAR
TRAIN records support reconstruction. The exact package also restores the
five clean development inputs and ten saved v1 stress inputs. Development
comparisons are paired against the unedited reconstruction on the same matches
and views. The exposed development cohort is five original-provider matches;
v2 uses four fixed realizations, averaged within match before provider and
match-level reporting.

For the selected edit, the deployment stage has fit only the two pooled seeds,
blending each edited model's probabilities 50/50 with its same-seed
calibrated-dense counterpart and retaining the selected decoder. The fits are
sealed. GAR TEST is a separate ten-game check with clean and four v2 views per
game. It is not pooled with development or selected-on VALID. The user authorized a separate one-use organizer public-window diagnostic; it
ran only after candidate selection was frozen and was not used for tuning or
selection. The public score is a sparse diagnostic, not full-match confirmation. The candidate rule ranks unrounded v2 spell among
arms passing the fixed clean spell floor (0.662555533), v2 spell floor
(0.573571208), and each-provider macro/D floor (baseline minus 0.02).

## Seven-arm development comparison

The auxiliary 70/30 value is **0.7 × spell F1 + 0.3 × macro F1** and is
descriptive only; arm selection uses unrounded v2 spell subject to all fixed
guards. The table shows each direct-edit arm against the unedited
reconstruction and the adopted package separately. Seed values are diagnostics;
selection is on the declared equal-seed/equal-phase aggregate.

| Fixed arm | Clean spell / macro / 70-30 | v2 spell / macro / 70-30 | Δ clean / v2 spell vs unedited | Δ clean / v2 spell vs adopted | Guard |
| --- | --- | --- | --- | --- | --- |
| Unedited blend control | 0.665355/0.872661/0.727547 | 0.592749/0.854043/0.671137 | +0.000000 / +0.000000 | -0.002201 / +0.009178 | PASS |
| Projection α=.25 | 0.663150/0.872566/0.725975 | 0.592601/0.854049/0.671036 | -0.002205 / -0.000148 | -0.004406 / +0.009030 | PASS |
| Projection α=.50 | 0.664134/0.872692/0.726701 | 0.592814/0.854067/0.671190 | -0.001221 / +0.000065 | -0.003422 / +0.009243 | PASS |
| Projection α=1.00 | 0.664063/0.872705/0.726656 | 0.592761/0.854143/0.671176 | -0.001292 / +0.000012 | -0.003492 / +0.009190 | PASS |
| Update-bias δ=−.25 | 0.663012/0.872337/0.725810 | 0.591711/0.853893/0.670365 | -0.002343 / -0.001038 | -0.004543 / +0.008140 | PASS |
| Update-bias δ=−.50 | 0.662455/0.872287/0.725404 | 0.591630/0.853981/0.670335 | -0.002900 / -0.001119 | -0.005101 / +0.008059 | FAIL clean spell floor |
| Update-bias δ=+.25 | 0.664316/0.872560/0.726789 | 0.594110/0.854513/0.672231 | -0.001039 / +0.001360 | -0.003240 / +0.010538 | PASS · selected |

The 70/30 auxiliary is computed as 0.7 × equal-provider spell F1 + 0.3 × equal-provider macro F1; it is not the ranking metric. `update_bias_m050` fails only the fixed clean spell floor (.662455 versus .662556); the other six arms pass every guard.

### Component-seed spread

| Fixed arm | Clean spell seed 2026 / 2027 | v2 spell seed 2026 / 2027 | v2 seed gap (2026−2027) |
| --- | --- | --- | --- |
| Unedited blend control | 0.663826 / 0.660904 | 0.586460 / 0.585790 | +0.000671 |
| Projection α=.25 | 0.662403 / 0.660847 | 0.586222 / 0.585970 | +0.000252 |
| Projection α=.50 | 0.662403 / 0.661100 | 0.586056 / 0.586249 | -0.000193 |
| Projection α=1.00 | 0.663145 / 0.662184 | 0.586469 / 0.587210 | -0.000742 |
| Update-bias δ=−.25 | 0.663542 / 0.659991 | 0.586549 / 0.585402 | +0.001147 |
| Update-bias δ=−.50 | 0.659957 / 0.658832 | 0.583417 / 0.585766 | -0.002349 |
| Update-bias δ=+.25 | 0.660983 / 0.660033 | 0.586136 / 0.584926 | +0.001209 |

### Per-provider clean and v2 diagnostics

Scores and spell counts are equal-match means within each provider; v2 values
average the four fixed realizations within match.

| Arm | View | Provider | D precision | D recall | Short-spell recall (<3s, IoU .5) | Reference / predicted spells |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Unedited blend control | Clean | Metrica | 0.893110 | 0.932132 | 0.603175 | 382.0 / 347.0 |
| Unedited blend control | Clean | SkillCorner | 0.854180 | 0.718687 | 0.280215 | 375.0 / 381.7 |
| Unedited blend control | Clean | Sportec | 0.909394 | 0.945811 | 0.360406 | 456.0 / 329.0 |
| Unedited blend control | v2 | Metrica | 0.873980 | 0.922352 | 0.367063 | 382.0 / 331.8 |
| Unedited blend control | v2 | SkillCorner | 0.852960 | 0.745750 | 0.274355 | 375.0 / 370.8 |
| Unedited blend control | v2 | Sportec | 0.893619 | 0.933434 | 0.243655 | 456.0 / 302.8 |
| Projection α=.25 | Clean | Metrica | 0.893129 | 0.932132 | 0.595238 | 382.0 / 345.0 |
| Projection α=.25 | Clean | SkillCorner | 0.854180 | 0.718687 | 0.280215 | 375.0 / 381.7 |
| Projection α=.25 | Clean | Sportec | 0.909371 | 0.945842 | 0.355330 | 456.0 / 327.0 |
| Projection α=.25 | v2 | Metrica | 0.874002 | 0.922352 | 0.367063 | 382.0 / 331.8 |
| Projection α=.25 | v2 | SkillCorner | 0.852930 | 0.745776 | 0.274355 | 375.0 / 370.8 |
| Projection α=.25 | v2 | Sportec | 0.893591 | 0.933460 | 0.243655 | 456.0 / 302.2 |
| Projection α=.50 | Clean | Metrica | 0.895396 | 0.932132 | 0.595238 | 382.0 / 344.0 |
| Projection α=.50 | Clean | SkillCorner | 0.854119 | 0.718818 | 0.280215 | 375.0 / 381.7 |
| Projection α=.50 | Clean | Sportec | 0.909310 | 0.944694 | 0.355330 | 456.0 / 328.0 |
| Projection α=.50 | v2 | Metrica | 0.874008 | 0.922252 | 0.367063 | 382.0 / 331.5 |
| Projection α=.50 | v2 | SkillCorner | 0.852427 | 0.745959 | 0.274355 | 375.0 / 370.5 |
| Projection α=.50 | v2 | Sportec | 0.893594 | 0.933494 | 0.243655 | 456.0 / 302.2 |
| Projection α=1.00 | Clean | Metrica | 0.895410 | 0.932087 | 0.595238 | 382.0 / 344.0 |
| Projection α=1.00 | Clean | SkillCorner | 0.853984 | 0.718563 | 0.280215 | 375.0 / 382.0 |
| Projection α=1.00 | Clean | Sportec | 0.909317 | 0.944921 | 0.355330 | 456.0 / 328.0 |
| Projection α=1.00 | v2 | Metrica | 0.874068 | 0.922263 | 0.367063 | 382.0 / 332.0 |
| Projection α=1.00 | v2 | SkillCorner | 0.852313 | 0.746205 | 0.274910 | 375.0 / 370.5 |
| Projection α=1.00 | v2 | Sportec | 0.893605 | 0.933566 | 0.243655 | 456.0 / 302.2 |
| Update-bias δ=−.25 | Clean | Metrica | 0.893241 | 0.932487 | 0.603175 | 382.0 / 351.0 |
| Update-bias δ=−.25 | Clean | SkillCorner | 0.851095 | 0.719000 | 0.284588 | 375.0 / 385.0 |
| Update-bias δ=−.25 | Clean | Sportec | 0.909092 | 0.945826 | 0.360406 | 456.0 / 329.0 |
| Update-bias δ=−.25 | v2 | Metrica | 0.874083 | 0.923201 | 0.367063 | 382.0 / 335.2 |
| Update-bias δ=−.25 | v2 | SkillCorner | 0.853411 | 0.746014 | 0.277043 | 375.0 / 373.8 |
| Update-bias δ=−.25 | v2 | Sportec | 0.893348 | 0.933408 | 0.242386 | 456.0 / 303.5 |
| Update-bias δ=−.50 | Clean | Metrica | 0.893437 | 0.932731 | 0.595238 | 382.0 / 350.0 |
| Update-bias δ=−.50 | Clean | SkillCorner | 0.851206 | 0.719231 | 0.288961 | 375.0 / 388.3 |
| Update-bias δ=−.50 | Clean | Sportec | 0.909266 | 0.945857 | 0.365482 | 456.0 / 333.0 |
| Update-bias δ=−.50 | v2 | Metrica | 0.874395 | 0.923323 | 0.373016 | 382.0 / 340.0 |
| Update-bias δ=−.50 | v2 | SkillCorner | 0.853034 | 0.746800 | 0.280860 | 375.0 / 376.7 |
| Update-bias δ=−.50 | v2 | Sportec | 0.894307 | 0.933747 | 0.246193 | 456.0 / 305.5 |
| Update-bias δ=+.25 | Clean | Metrica | 0.892932 | 0.931687 | 0.603175 | 382.0 / 347.0 |
| Update-bias δ=+.25 | Clean | SkillCorner | 0.853826 | 0.720314 | 0.271541 | 375.0 / 378.3 |
| Update-bias δ=+.25 | Clean | Sportec | 0.909559 | 0.944664 | 0.355330 | 456.0 / 328.0 |
| Update-bias δ=+.25 | v2 | Metrica | 0.874250 | 0.921952 | 0.355159 | 382.0 / 327.2 |
| Update-bias δ=+.25 | v2 | SkillCorner | 0.854218 | 0.744726 | 0.269982 | 375.0 / 369.3 |
| Update-bias δ=+.25 | v2 | Sportec | 0.893194 | 0.932898 | 0.239848 | 456.0 / 298.0 |
| Adopted original_full | Clean | Metrica | 0.905239 | 0.927225 | 0.611111 | 382.0 / 345.0 |
| Adopted original_full | Clean | SkillCorner | 0.851616 | 0.709991 | 0.277993 | 375.0 / 385.3 |
| Adopted original_full | Clean | Sportec | 0.897677 | 0.948543 | 0.329949 | 456.0 / 314.0 |
| Adopted original_full | v2 | Metrica | 0.881886 | 0.912445 | 0.349206 | 382.0 / 336.5 |
| Adopted original_full | v2 | SkillCorner | 0.855016 | 0.737707 | 0.277366 | 375.0 / 374.1 |
| Adopted original_full | v2 | Sportec | 0.882200 | 0.938789 | 0.225888 | 456.0 / 297.8 |

## Matched readout

The six-fit unedited reconstruction control reproduces the prior selected blend’s
clean spell F1 at **0.665354983** (reported previously as 0.665355) and v2 at
**0.5927491913**, exactly matching the archived eligible-v2 aggregate. The
separate adopted original-full control replays all 25 clean/v2 cells for spell
and macro F1 with zero error. Phase averaging uses every offset supported by
each source’s native sample rate: five at 25 Hz (Sportec/Metrica), two at 10 Hz
(SkillCorner), and six at 29.97 Hz for GAR. The frozen declaration’s “five
native phases” phrase is shorthand and is not universal; runtime uses a
sample-rate-derived stride. This documentation clarification does not alter the
frozen declaration, model tensors, or score cells. This validates the comparison reference; it is not an edit result. The fixed primary development winner is `update_bias_p025`: clean spell **0.664316**, v2 spell **0.594110**, macro **0.872560 / 0.854513** and auxiliary 70/30 **0.726789 / 0.672231**. It gains **.001360** v2 spell (+0.136 pp) over the unedited reconstruction while losing **.001039** clean spell (−0.104 pp); versus adopted, clean is −.003240 and v2 is +.010538. All fixed development guards pass, but the .60 target remains **.005890** away. The opposite-sign update-bias control won; the hypothesis that reducing recurrent persistence would help was not confirmed. The post-selection v1 diagnostic is **0.518850 spell / 0.823766 macro**, versus unedited **0.518637 / 0.823302** and adopted **0.531576 / 0.826853**. The edit itself gains **.000213 v1 spell** versus the matched unedited reconstruction; the **.012726** adopted-package gap is a whole-pipeline comparison, not an isolated edit effect. v1 did not participate in selection. The frozen primary choice remains the best eligible v2 arm; later v1/GAR/public readouts do not rerank it. Two pooled deployment fits are sealed; GAR TEST remains pending. External GAR/public comparisons are against the adopted whole pipeline, with no unedited-new-pooled control. Adoption remains unchanged because promotion was not requested.

| Cohort and protocol | Clean | v1 | v2 | Status |
| --- | --- | --- | --- | --- |
| Five original-provider development matches; paired to unedited reconstruction | See seven-arm table | .518850 spell / .823766 macro (post-selection) | Primary clean/v2 in table | v1 diagnostic complete |
| Ten GAR TEST games; exact selected deployment package, games averaged equally | Pending | Outside declared deployment check | Pending | Not started |
| Organizer public window; one authorized fixed diagnostic | Not applicable | Not applicable | .594286 spell / .852078 macro | Complete; diagnostic only |

## One-use public diagnostic

After the edit was frozen, the user-authorized one-use diagnostic scored the
selected output on the fixed public window. Spell F1 is **.594286** and macro
F1 **.852078**, compared with the saved adopted result **.611765 / .849473**:
spell falls **.017479**, while macro rises **.002605**. The scored CSV has 199,319
rows, of which 7,500 are graded; there are 17 reference and 18 predicted
spells. The single-match bootstrap 5th–95th percentile ranges are .434286–.742857
for spell and .697750–.887613 for macro; these are conditional on this window,
not intervals for the paired difference or unseen matches. There was no unedited-new-pooled public control, so the candidate-minus-adopted
public difference is a whole-pipeline comparison and cannot be attributed
specifically to the update-bias edit. Public labels were read only after the
one-use gate. This result was not used for training, tuning, reranking, or
promotion and does not support a full-match claim. See the
[frozen public result](results/v2_weight_edit_evaluation_v1_public_result.json).

The development results above come from the frozen aggregate and verified
control replay. The separate post-selection v1 readout is in
[`v2_weight_edit_development_v1_v1.json`](results/v2_weight_edit_development_v1_v1.json):
`update_bias_p025` reaches **.518850 spell / .823766 macro / .610325 auxiliary
70/30**, versus unedited **.518637 / .823302 / .610036** and adopted
**.531576 / .826853 / .620159**. The edit itself adds .000213 v1 spell versus the matched unedited
reconstruction. The larger .012726 gap to adopted is a whole-pipeline comparison
and cannot be attributed to the tensor edit; v1 is not a selection criterion. Two pooled fits are sealed. The one-use public readout is complete; GAR TEST
remains pending. Provider means, seed spread, spell and macro F1, and boundary
diagnostics stay together; the
per-provider table includes clean and v2 D precision/recall, short-spell recall,
and reference/predicted spell counts. Keep PR82 numbers labeled as historical
context.

## Execution record

Before candidate or post-selection v1 scoring, dispatch and control-list issues
were handled with additive CLI wrappers. The fixed arms, model tensors, metrics,
guards, cohort, and candidate ranking did not change. The scoring wrapper validates the frozen declaration and source hashes; the
v1 readout wrapper filters and deduplicates editable arms while retaining the
archived control inside the unchanged scorer. The CLI amendments are recorded in
[`execution_amendment_v2`](results/v2_weight_edit_development_v1_execution_amendment_v2.json),
[`execution_amendment_v3`](results/v2_weight_edit_development_v1_execution_amendment_v3.json),
[`execution_amendment_v5`](results/v2_weight_edit_development_v1_execution_amendment_v5.json), and
[`execution_amendment_v6`](results/v2_weight_edit_development_v1_execution_amendment_v6.json). The v1 adapter passed a synthetic ProcessPool test. Its first real invocation failed during archive model lookup before writing any cell or score; the preserved [failure receipt](results/v2_weight_edit_development_v1_readout_attempt1_failure.json) records the missing runtime-root initialization. Amendment v6 sets the restored archive path and initializes frozen roots in the parent before workers fork; neither frozen scorer nor adapter was edited.
The final commands are:

```bash
.venv/bin/python -m experiments.v2_weight_edit_development_execution control --workers 5
.venv/bin/python -m experiments.v2_weight_edit_development_execution score-primary --workers 4
.venv/bin/python -m experiments.v2_weight_edit_development_execution freeze-winner
MTAH_DATA_DIR=/workspace/mtah-data MTAH_SQD_DIR=/workspace/mtah-sqd MTAH_DT_DIR=/workspace/mtah-dt MTAH_RF_ARCHIVE_DIR=/workspace/mtah-rf/experiments/feed_robustness_v1/artifacts SOCCERNET_GAR_TRAIN_RECORDS=/workspace/more-than-a-hack-2027/data/experiments/v2_weight_edit_reconstruction_v1/gar_records_archived_pr8 \
.venv/bin/python -c "from experiments import v2_weight_edit_development as d; from experiments import v2_weight_edit_development_v1_readout_v2 as r; d.upstream.frozen._roots(); r.run_v1(workers=3)"
```

The current experiment suite passes **683 tests**. The root suite has 26 passes;
file-backed parity and bundled-CSV checks also passed.

The [primary aggregate](results/v2_weight_edit_development_v1.json), [winner
receipt](results/v2_weight_edit_development_v1_winner.json), and
[control replay](results/v2_weight_edit_development_v1_control_replay.json)
contain the machine-readable scorecard and replay evidence.

## Evidence limits and current state

The candidate's boundary direction is derived from original-provider TRAIN
H/A targets and GRU activations. It is an observational diagnostic, not evidence
that the direction is causal or that an edit caused any measured change. The
archived PR82 weights are unavailable,
so reconstruction parity is recipe-level rather than byte-level. GAR uses
generated silver labels and discontinuous segments. Synthetic corruption
views and the organizer's small public window cannot establish continuous
full-match possession accuracy. The adopted package and closed evaluation
policy remain unchanged.

The external references motivate a mechanism, not a model equivalence:
[Qwen weight editing / reduced thinking loops](https://dev.to/axrisi/qwen-weight-editing-how-astra-reduced-thinking-loops-2ccc)
and the primary [Weight-Space Programming article](https://rikiyat.github.io/blog/weight-space-programming/).
Code, declarations, results, and receipts will be linked here after their
respective stages complete. See the [development runner](v2_weight_edit_development.py),
[reconstruction runner](v2_weight_edit_reconstruction.py), and
[deployment runner](v2_weight_edit_deployment.py).
