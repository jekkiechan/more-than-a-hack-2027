"""Generate a full-length submission from the externally selected pipeline.

Reads organizer tracking, never public or private possession labels. Outputs and
model files stay under ignored data/. This command does not submit to organizers.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from games import require
from experiments import challenge, possession_pipeline, robust
from experiments.challenge_pathcrf import load_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/experiments/pipeline_v1")
    parser.add_argument("--inputs", type=Path, help="Optional existing organizer tracking NPZ; targets are not deserialized")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/pipeline_v1/competition.csv")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite a submission; use a new output path")
    model_path = args.model_dir / "model.joblib"
    manifest = json.loads((args.model_dir / "model_manifest.json").read_text())
    protocol_path = args.model_dir / "protocol.json"
    if robust.sha256(model_path) != manifest["model_sha256"] or robust.sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Frozen model/protocol checksum mismatch")
    spec = json.loads(protocol_path.read_text())["protocol"]
    for name, digest in spec["source_sha256"].items():
        if robust.sha256(ROOT / "experiments" / name) != digest:
            raise ValueError(f"Frozen experiment source changed: {name}")
    model = joblib.load(model_path)
    if model["candidate"] != manifest["selected"]:
        raise ValueError("Selected candidate differs from model manifest")
    if args.inputs:
        inputs = load_inputs(args.inputs)
        input_paths = [args.inputs]
    else:
        input_paths = [Path(p) for p in require(["atd_meta", "atd_raw"])]
        inputs = robust.tracking_inputs(challenge.load_challenge())
    predictions = possession_pipeline.predict(inputs, model)
    predictions[np.asarray(inputs["period"]) <= 0] = "X"
    challenge.write_submission(predictions, args.out)
    provenance = {"model_sha256": manifest["model_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                  "candidate": model["candidate"], "submission_sha256": robust.sha256(args.out),
                  "frames": len(predictions), "competition_labels_accessed": False,
                  "input_sha256": {p.name: robust.sha256(p) for p in input_paths},
                  "prediction_source_sha256": {p: robust.sha256(ROOT / p) for p in (
                      "experiments/pipeline_predict.py", "experiments/challenge.py",
                      "experiments/challenge_pathcrf.py", "games.py", "starter/baseline.py")}}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote and validated {len(predictions)} rows to {args.out}; no competition labels read")


if __name__ == "__main__":
    main()
