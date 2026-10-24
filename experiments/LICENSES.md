# Sources and attribution

- Sportec Open DFL data: Bassek et al., version 1,
  [DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Our conversion changes coordinate origins and packages tracking and provider
  labels into NPZ files; the reports publish derived aggregate metrics. Raw data
  and converted frame arrays are downloaded locally, not included in this branch.
- SkillCorner [opendata](https://github.com/SkillCorner/opendata), revision
  `4340d274572876239c154c90bc507a9b3250a656`, MIT. Raw and processed files are local.
- Hyunsung Kim and contributors,
  [PathCRF](https://github.com/hyunsungkim-ds/pathcrf), revision
  `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, MPL 2.0.
  `pathcrf_eval.py` follows and adapts its feature and decoding conventions and is
  marked MPL 2.0. The [license text](licenses/pathcrf-MPL-2.0.txt) is included.
  Upstream source and weights are obtained separately through the documented
  pinned checkout. Our adapter replaces label-derived episodes with fixed
  windows and maps predicted player-edge sources to teams.

Other experiment code follows the repository's root license. Organizer match
assets retain their original access and redistribution restrictions and are
excluded from this branch.
