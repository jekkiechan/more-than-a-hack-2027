#!/usr/bin/env bash
# Regenerate the submission CSV from the organizer tactical ATD and the four
# published model weights, then validate it and print its SHA-256.
#
# Usage: submission/make_submission.sh <team_name> [models_dir]
#   Organizer files expected in data/tactical/:
#     BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt
#   models_dir defaults to data/models/joint_gru_sn_dense_ensemble_v1 and must
#   hold dense_2026.joblib, teacher_2026.joblib, dense_2027.joblib,
#   teacher_2027.joblib (hashes are checked against experiments/baseline_manifest.json).
set -euo pipefail

team="${1:?usage: submission/make_submission.sh <team_name> [models_dir]}"
models="${2:-data/models/joint_gru_sn_dense_ensemble_v1}"
cd "$(dirname "$0")/.."

out_dir=data/submission
inputs=data/challenge/inputs.npz
mkdir -p "$out_dir" data/challenge
output="$out_dir/$team.csv"
if [ -e "$output" ]; then
  echo "$output already exists; move it away first" >&2
  exit 1
fi

python -m experiments.challenge --out "$inputs"
python -m experiments.current_baseline predict --inputs "$inputs" --models-dir "$models" --output "$output"
python evaluation/validate_submission.py "$output"
sha256sum "$output"
