# Reference methods and pretrained-model audit

Research checked 2026-10-24. No model was installed or run for this audit. The
recommendation is to train a small two-stage temporal model on the fixed Sportec
training matches, predicting dead/alive and then home/away. None of the inspected
pretrained assets provides a clean shortcut to a defensible 0.90 possession F1 on
our existing held-out match.

## Metric compatibility

| Method | Reported result | Meaning | Comparable to this challenge? |
| --- | --- | --- | --- |
| TCN, Borghesi et al. (2023) | 86.2% possession accuracy; 89.2% dead/alive accuracy | Classification accuracy on the authors' professional-match data | Relevant classes, but accuracy is neither frame macro-F1 nor possession-spell F1 |
| PathCRF, Kim et al. (2026), arXiv v1 Table 2 | 69.64% edge accuracy; 75.69% event F1 | Correct sender/receiver edge; event match requires same type, same acting player, and timing within one second | Different task, labels, evaluation population, and matching rule |

The challenge's spell F1 over temporal IoU thresholds must still be computed by
our own evaluator. A published accuracy, detector mAP, or on-ball event score must
not be reported as a possession F1 result.

## PathCRF: usable code, contaminated bundled weights for our split

[Official repository](https://github.com/hyunsungkim-ds/pathcrf), inspected at commit
`1b7c6f85a6b78c92e4543fde72ca2b86a933f997` (2026-06-08).
[License](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/LICENSE):
Mozilla Public License 2.0. The repository includes actual model checkpoints under
`saved/{100,110,120,130,140}/model/`, approximately 19–20 MB each, plus arguments,
logs, preprocessing, training, inference, and a tutorial.

The [paper](https://arxiv.org/html/2602.12080v1) predicts a possession path over
players and outside-pitch nodes using player trajectories. It evaluates only
in-play episodes. Labels combine synchronized human event annotations with
algorithmically detected ball touches. Its reported event F1 uses event identity
and a one-second tolerance, not possession intervals. These choices explain why
its result cannot be substituted for a whole-match H/A/D score.

### Exact checkpoint overlap

[Trial 140 arguments](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/saved/140/args.json)
record `sources=["sportec"]`, `lomocv=true`, `cv_fold_idx=6`.
The [split code](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/train.py#L471)
sorts match filenames, uses the chosen index as test, and the previous index as
validation. With the seven distributed Sportec match IDs, this means:

| Role in supplied trial 140 | Match IDs |
| --- | --- |
| Training | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY |
| Validation | J03WQQ |
| Test | J03WR9 |

Our fixed validation **J03WOY** and test **J03WPY** are therefore training matches
for this checkpoint according to its supplied configuration. Fine-tuning those
weights would retain the contamination. Trial 120 records `lomocv=false`, whose
current split code also selects the first five files for training. Trials 100,
110, and 130 have older argument schemas and do not record an exclusion of our
held-out matches; do not assume those checkpoints are independent either.

Keep our split fixed. A fresh model trained only on our training partition is a
valid experiment. Alternatively, a separate, explicitly declared external-model
experiment could use a new untouched match excluded from both pretraining and
selection; it would not replace our existing benchmark or its selection protocol.

### Input and preprocessing audit

The [dataset implementation](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/dataset.py)
builds eight node features: team indicator, goalkeeper indicator, x, y, vx, vy,
speed, and acceleration. It adds four outside nodes and orders teams by playing
direction. The standard configuration uses 105 × 68 m coordinates, 11 roster
slots per team, 5 Hz sampling, and 10-second windows. The standard model does not
include the ball as an input feature; a separate `ball_feature` option exists.

Its **inference wrapper is not independent of labels by default**:

- `datatools/utils.py::label_frames_and_episodes` builds episodes from rows where
  `ball_state == "alive"`.
- `inference.py::inference` processes positive episode IDs. Thus supplying true
  episode boundaries leaks the answer to the dead/alive part of our task even
  when no ball-state column enters a neural-network feature tensor.
- Training samples also filter missingness using possessing-player targets.
  This is reasonable training curation, but must not determine our test coverage.
- `inference.py` has evaluation and episode-end correction paths. A transfer
  adapter needs an explicit audit of those paths and must receive only tracking,
  period boundaries, and predicted state boundaries at deployment.

The [preprocessor](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/datatools/preprocess.py)
creates player-touch/path targets from synchronized events and ball geometry.
Our present NPZ files contain team/state targets, not those sender/receiver
targets. Direct retraining therefore needs additional event preprocessing on
**training matches only**, or a new simpler H/A/D output head. Its dependencies
also need a deliberate setup: source imports PyTorch and `torch_geometric`, which
are absent from the repository's minimal `requirements.txt`. No installation or
CPU/GPU performance estimate has been validated here.

Practical value: reuse the idea of temporally consistent decoding and
permutation-invariant player features. The full edge model is a larger adaptation
than this hackathon's three-class task presently needs.

## TCN possession classifier (2023)

[Borghesi, Costa, Morra, and Lamberti,
*Using Temporal Convolutional Networks to estimate ball possession in soccer games*](https://www.sciencedirect.com/science/article/pii/S0957417423002816),
Expert Systems with Applications 223, 119780, DOI
[10.1016/j.eswa.2023.119780](https://doi.org/10.1016/j.eswa.2023.119780).
An [institutional copy](https://iris.polito.it/handle/11583/2976470) is available.

This is the closest problem match: player/ball trajectories feed temporal
embeddings, followed by permutation-invariant aggregation; the strongest variant
uses self-attention and separate dead/alive and home/away classifiers. The
published abstract reports **86.2% possession accuracy** and **89.2% dead/alive
accuracy**. The publisher identifies the underlying data as confidential.
No official implementation or pretrained possession weights were verified in the
paper links or targeted repository searches. The article is CC BY-NC-ND; this is
an article license, not a software or model license.

Practical value: implement the architecture concept using our accessible Sportec
training set and tracking-only features. Neither its accuracy claim nor its
unavailable data establishes 0.90 F1 for our task.

## Other inspected public code

[Ball Radar (KDD 2023)](https://github.com/hyunsungkim-ds/ballradar) provides a
player-possession classifier and ball-trajectory regressor using set transformers
and bidirectional LSTMs. Its README describes 15 private Fitogether matches plus
three public Metrica matches and a reproducible Metrica-only route. The inspected
repository tree contains no saved checkpoints. No root project license was found;
the vendored `set_transformer/LICENSE` does not establish permission for the whole
project. Therefore it is a methodological reference, not a verified licensed
pretrained model to integrate. The open-source PathCRF repository
also contains a Ball Radar-style implementation under its own MPL-2.0 terms.

Public football video demos often distribute detector weights and attach a
nearest-player rule for possession. Detector weights improve ball/player
localization; they are not pretrained H/A/D classifiers and provide no verified
possession F1 merely by being pretrained. Since this benchmark already supplies
tracking, they do not address its immediate modeling bottleneck.

## Next experiment, if compute is allocated

Train a compact temporal classifier from scratch using the frozen Sportec split:
one dead/alive head over ball movement and collective player motion, and one
home/away head over ball-relative team geometry. Combine probabilities into H/A/D
and apply temporal decoding whose parameters are selected on validation only.
Use match/period boundaries for sequence resets, never true possession episodes.
Add player-set embeddings only if the simpler temporal model improves validation
spell F1 enough to justify the extra cost. Report both spell F1 and frame macro-F1
with per-class scores; open the held-out test only after selecting the model.

This is a proposed experiment, not an implemented or measured result. No source
review establishes that 0.90 F1 is attainable on the existing split.
