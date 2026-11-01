"""Generate the experimental decoder winner's full CSV without reference access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import challenge, current_baseline as baseline, decoder_search, supervised
from experiments.eval_set.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=baseline.ROOT / "data/challenge/inputs.npz")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite candidate output")
    result_path = baseline.ROOT / "experiments/results/decoder_search_v2_aggregate.json"
    declaration_path = baseline.ROOT / "experiments/results/decoder_search_v2_declaration.json"
    result = json.loads(result_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    if sha256(declaration_path) != result["declaration_sha256"]:
        raise ValueError("Result is not bound to declaration")
    for rel, digest in declaration["source_sha256"].items():
        if sha256(baseline.ROOT / rel) != digest:
            raise ValueError(f"Frozen source mismatch: {rel}")
    selected = result["eligible_ranked_arm_ids"][0]
    config = next(c for c in declaration["arms"] if c["id"] == selected)
    manifest = json.loads(baseline.MANIFEST.read_text())
    inputs = baseline.load_inputs(args.inputs)
    models = baseline.load_models(args.models_dir)
    probability = baseline.predict_probabilities(inputs, models, manifest)
    periods = supervised.model_periods(inputs)
    prediction = decoder_search.decode(probability, periods, float(inputs["fps"]), config)
    prediction[np.asarray(inputs["period"]) <= 0] = "X"
    challenge.write_submission(prediction, args.output)
    receipt = {
        "study": "decoder_search_v2", "status": "experimental; adopted manifest unchanged",
        "config": config, "input_sha256": sha256(args.inputs),
        "output_sha256": sha256(args.output), "baseline_manifest_sha256": sha256(baseline.MANIFEST),
        "model_sha256": manifest["models_sha256"], "declaration_sha256": sha256(declaration_path),
        "result_sha256": sha256(result_path), "target_labels_read": False, "organizer_scored": False,
    }
    with args.output.with_suffix(".json").open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Generated and format-validated experimental candidate", selected, args.output)


if __name__ == "__main__":
    main()
