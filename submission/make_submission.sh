#!/usr/bin/env bash
# Regenerate the frozen submission from tactical ATD and public release weights.
#
# Usage: submission/make_submission.sh <team_name> [models_dir]
#   Organizer files expected in data/tactical/:
#     BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt
#   Missing weights download automatically; existing weights are hash-checked.
#   PYTHON may select a virtual-environment interpreter. Run from the repo root.
set -euo pipefail

team="${1:?usage: submission/make_submission.sh <team_name> [models_dir]}"
cd "$(dirname "$0")/.."
python="${PYTHON:-python}"
manifest=submission/model_manifest.json
models="${2:-$("$python" -c 'import json; print(json.load(open("submission/model_manifest.json"))["model_directory"])')}"
"$python" - "$team" <<'PY'
import sys
name = sys.argv[1]
if name in ('.', '..') or any(c in name for c in '/\\\n\r\0') or not name.strip():
    raise SystemExit('Team name must be a nonempty filename without path separators or newlines')
PY

out_dir=data/submission
inputs="$out_dir/$team.inputs.npz"
mkdir -p "$out_dir"
output="$out_dir/$team.csv"
if [ -e "$output" ]; then
  echo "$output already exists; move it away first" >&2
  exit 1
fi

"$python" submission/download_weights.py --models-dir "$models"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -m experiments.challenge --out "$inputs"
"$python" -m experiments.current_baseline predict --manifest "$manifest" --inputs "$inputs" --models-dir "$models" --output "$output"
"$python" evaluation/validate_submission.py "$output"
"$python" - "$output" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = json.loads(pathlib.Path('submission/weights.json').read_text())['csv_sha256']
if actual != expected:
    raise SystemExit(f'Submission hash mismatch: {actual}; expected {expected}')
print(f'Exact sealed CSV reproduced: {actual}  {path}')
PY
cp submission/TECHNICAL_DOCUMENT.md "$out_dir/$team.md"
echo "Upload files: $output and $out_dir/$team.md"
