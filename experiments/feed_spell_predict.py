"""Generate an opt-in feed-first candidate CSV without reference-label access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import challenge, current_baseline as baseline, sn_dense_baseline
from experiments import feed_spell_push, feed_spell_refine, supervised
from experiments.eval_set.protocol import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=baseline.ROOT/"data/challenge/inputs.npz")
    parser.add_argument("--models-dir",type=Path)
    parser.add_argument("--round",choices=("broad","refine"),default="refine")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite a candidate output")
    study="feed_spell_refine_v2" if args.round=="refine" else "feed_spell_push_v1"
    results=baseline.ROOT/"experiments/results"
    result_path=results/(study+"_aggregate.json");declaration_path=results/(study+"_declaration.json")
    result=json.loads(result_path.read_text());d=json.loads(declaration_path.read_text())
    if sha256(declaration_path)!=result["declaration_sha256"]:
        raise ValueError("Result does not bind the declaration")
    for rel,digest in d["source_sha256"].items():
        if sha256(baseline.ROOT/rel)!=digest:raise ValueError("Frozen source changed: "+rel)
    config=result["selected_config"]
    if config["id"]!=result["eligible_ranked"][0] or config not in d["arms"]:
        raise ValueError("Selected configuration differs from frozen result")
    inputs=baseline.load_inputs(args.inputs);models=baseline.load_models(args.models_dir)
    manifest=json.loads(baseline.MANIFEST.read_text())
    members=sn_dense_baseline.member_probabilities(inputs,models,manifest)
    x=np.stack([members[k] for k in sn_dense_baseline.MEMBERS])
    module=feed_spell_refine if args.round=="refine" else feed_spell_push
    prediction=module.decode(x,supervised.model_periods(inputs),float(inputs["fps"]),config)
    prediction[np.asarray(inputs["period"])<=0]="X"
    challenge.write_submission(prediction,args.output)
    receipt={"study":study,"status":"experimental; adopted manifest unchanged","config":config,
        "input_sha256":sha256(args.inputs),"output_sha256":sha256(args.output),
        "baseline_manifest_sha256":sha256(baseline.MANIFEST),"model_sha256":manifest["models_sha256"],
        "declaration_sha256":sha256(declaration_path),"aggregate_sha256":sha256(result_path),
        "prediction_entrypoint_sha256":sha256(Path(__file__)),
        "target_labels_read":False,"organizer_scored":False}
    with args.output.with_suffix(".json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print("Generated and format-validated experimental candidate",config["id"],args.output)


if __name__=="__main__":main()
