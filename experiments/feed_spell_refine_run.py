"""One bounded adaptive round after the broad feed-first search."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

import numpy as np

from experiments import feed_spell_refine as inference, feed_spell_push_run as first
from experiments import decoder_search_run as scoring, spell_mbr_run as shared

ROOT = shared.ROOT
RESULTS = shared.RESULTS
OUT = ROOT / "data/experiments/feed_spell_refine_v2"
DECLARATION = RESULTS / "feed_spell_refine_v2_declaration.json"
AGGREGATE = RESULTS / "feed_spell_refine_v2_aggregate.json"
CELLS = []


def freeze():
    d = json.loads(first.DECLARATION.read_text())
    broad = json.loads(first.AGGREGATE.read_text())
    if broad["selected_config"] != dict(family="calibration", dt=4, ht=1, ha=3, dead=3, id="f062"):
        raise ValueError("This refinement was designed around f062; declare a different study for another winner")
    sources = set(first.SOURCES) | {"experiments/feed_spell_refine.py", "experiments/feed_spell_refine_run.py"}
    scoring.write_once(DECLARATION, {
        "study": "feed_spell_refine_v2", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does local D calibration refinement or calibrating each member before pooling improve the broad-search winner?",
        "adaptive_after": "Completed 194-arm feed_spell_push_v1; this is additional tuning on the same exposed cohort",
        "arms": inference.configurations(), "ensemble_cells": len(inference.configurations())*15,
        "component_seed_budget": 90, "new_fits": 0, "organizer_reference_access": False,
        "primary": d["primary"], "aggregation": d["aggregation"], "guards": d["guards"],
        "stop": "Complete this finite round and report the best supported development candidate; no post-result extensions under this study ID",
        "historical_clean_screen": d["historical_clean_screen"],
        "cohort": d["cohort"], "reconstruction_status": d["reconstruction_status"],
        "prediction_files": d["prediction_files"], "evaluation_input_hashes": d["evaluation_input_hashes"],
        "model_checkpoint_hashes": d["model_checkpoint_hashes"],
        "broad_aggregate_sha256": shared.sha256_file(first.AGGREGATE),
        "source_sha256": {s: shared.sha256_file(ROOT / s) for s in sorted(sources)},
        "control_tolerance": 1e-12, "promotion_authorized": False,
    })


def evaluate(config):
    rows = []
    for c in CELLS:
        if c["member"] == "ensemble":
            y = inference.decode(c["members"],c["model_period"],c["fps"],config)
            rows.append(first.identity(c) | scoring.quick_score(c,y))
    return config, rows, scoring.summarize(rows)


def run():
    global CELLS
    d = json.loads(DECLARATION.read_text())
    if d["arms"] != inference.configurations() or shared.sha256_file(first.AGGREGATE) != d["broad_aggregate_sha256"]:
        raise ValueError("Frozen arms or broad results changed")
    for path, sha in d["source_sha256"].items():
        if shared.sha256_file(ROOT/path) != sha:
            raise ValueError("Frozen source changed: "+path)
    OUT.mkdir(parents=True,exist_ok=False)
    scoring.write_once(OUT/"lock.json",{"declaration_sha256":shared.sha256_file(DECLARATION)})
    CELLS = first.paired_cells(d)
    broad = json.loads(first.AGGREGATE.read_text())
    baseline = broad["baseline"]
    _, control_rows, replay = evaluate(d["arms"][1])
    for view in ("clean","feed_like"):
        for provider in shared.PROVIDERS:
            for key in ("spell_f1","macro_f1","d_f1","graded_frames","reference_spell_count"):
                if abs(replay[view]["by_provider"][provider][key]-
                       broad["arms"]["f062"][view]["by_provider"][provider][key]) > 1e-12:
                    raise ValueError("Broad winner scalar replay failed")
    for c in CELLS:
        if c["member"] == "ensemble":
            y = inference.decode(c["members"],c["model_period"],c["fps"],d["arms"][1])
            with np.load(first.OUT/"f062"/Path(c["relative_path"]).name) as saved:
                if not np.array_equal(y,saved["prediction"]):
                    raise ValueError("Broad winner exact-path replay failed")
    scoring.write_once(RESULTS/"feed_spell_refine_v2_control_replay.json",{"passed":True,"exact_paths":15,"scalar_cells":15})
    summaries, arm_rows = {},{}
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("fork")) as pool:
        for config,rows,summary in pool.map(evaluate,d["arms"]):
            for row in rows:
                ref = next(r for r in control_rows if all(row[k] == r[k] for k in first.identity(row)))
                if any(row[k] != ref[k] for k in ("graded_frames","reference_spell_count")):
                    raise ValueError("Grading population changed")
            arm=config["id"];summaries[arm]=summary;arm_rows[arm]=rows
            scoring.write_once(OUT/(arm+".json"),{"config":config,"rows":rows})
            print(arm,config["family"],summary["clean"]["equal_provider_mean"]["spell_f1"],
                  summary["feed_like"]["equal_provider_mean"]["spell_f1"],flush=True)
    ranked=sorted(summaries,key=lambda a:first.rank(summaries[a]),reverse=True)
    allowed=[a for a in ranked if first.eligible(summaries[a],baseline)]
    finalists={}
    for arm in allowed[:3]:
        config=next(c for c in d["arms"] if c["id"]==arm);rows=[]
        for c in CELLS:
            y=inference.decode(c["members"],c["model_period"],c["fps"],config)
            row=first.identity(c)|shared._score_prediction(c["labels"],y,c["period"],c["timestamps"],c["fps"])
            if c["member"]=="ensemble":
                old=next(r for r in arm_rows[arm] if all(r[k]==row[k] for k in first.identity(row)))
                if abs(old["spell_f1"]-row["spell_f1"])>1e-12:raise ValueError("Full official replay failed")
            rows.append(row);path=OUT/arm/Path(c["relative_path"]).name;path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,prediction=y)
        scoring.write_once(OUT/(arm+"_final.json"),{"rows":rows})
        finalists[arm]={m:scoring.summarize(rows,m) for m in shared.MEMBERS}
    cross={}
    for held in shared.PROVIDERS:
        train=[p for p in shared.PROVIDERS if p!=held]
        choices=[a for a in summaries if first.eligible(summaries[a],baseline,train)]
        chosen=max(choices,key=lambda a:first.rank(summaries[a],train))
        cross[held]={"selected":chosen,"selection_providers":train,
            "scores":{v:summaries[chosen][v]["by_provider"][held] for v in ("clean","feed_like")}}
    scoring.write_once(AGGREGATE,{
        "study":"feed_spell_refine_v2","declaration_sha256":shared.sha256_file(DECLARATION),
        "arms":summaries,"ranked":ranked,"eligible_ranked":allowed,
        "selected_config":next(c for c in d["arms"] if c["id"]==allowed[0]),
        "finalists":finalists,"baseline":baseline,"baseline_seeds":broad["baseline_seeds"],
        "broad_winner":broad["arms"]["f062"],"previous_feed_leader":broad["previous_feed_leader"],
        "cross_provider_selection":cross,"completed_ensemble_cells":len(summaries)*15,
        "component_seed_cells":len(finalists)*30,"organizer_labels_accessed":False,
        "private_prediction_hashes":{str(p.relative_to(ROOT)):shared.sha256_file(p) for p in OUT.rglob("*.npz")}})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("freeze","run"))
    args=parser.parse_args();freeze() if args.stage=="freeze" else run()
