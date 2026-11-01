"""Separate scalar replay of hierarchy, feed ranking, guards and provider selection."""
import json
from pathlib import Path
from statistics import mean

from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
OUT = ROOT / "data/experiments/feed_spell_push_v1"


def main():
    result = json.loads((RESULTS / "feed_spell_push_v1_aggregate.json").read_text())
    declaration = json.loads((RESULTS / "feed_spell_push_v1_declaration.json").read_text())
    providers = ("Metrica", "Sportec", "SkillCorner")
    scores = {}
    keys = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
            "reference_spell_count", "predicted_spell_count", "weighted_70_30")
    population = None
    for config in declaration["arms"]:
        arm = config["id"]
        rows = json.loads((OUT / (arm+".json")).read_text())["rows"]
        if len(rows) != 15 or len({(r["provider"],r["match"],r["view"]) for r in rows}) != 15:
            raise ValueError("Incomplete or duplicate paired cohort")
        current = {(r["provider"],r["match"],r["view"]):
                   (r["graded_frames"],r["reference_spell_count"]) for r in rows}
        if population is not None and current != population:
            raise ValueError("Raw grading population differs across arms")
        population = current
        scores[arm] = {}
        for view in ("clean", "feed_like"):
            scores[arm][view] = {}
            for provider in providers:
                subset = [r for r in rows if r["provider"] == provider and r["view"].startswith(view)]
                matches = {r["match"] for r in subset}
                values = {k: mean([mean([r[k] for r in subset if r["match"] == m])
                                  for m in matches]) for k in keys}
                scores[arm][view][provider] = values
                for k, value in values.items():
                    if abs(value-result["arms"][arm][view]["by_provider"][provider][k]) > 1e-12:
                        raise ValueError("Provider hierarchy replay failed")
            for k in keys:
                if abs(mean([scores[arm][view][p][k] for p in providers])-
                       result["arms"][arm][view]["equal_provider_mean"][k]) > 1e-12:
                    raise ValueError("Equal provider mean replay failed")
    def rank(arm, ps):
        return tuple(mean([scores[arm]["feed_like"][p][k] for p in ps])
                     for k in ("spell_f1", "macro_f1"))
    def allowed(arm, ps):
        b = result["baseline"]
        return (mean([scores[arm]["clean"][p]["spell_f1"] for p in ps]) >=
                mean([b["clean"]["by_provider"][p]["spell_f1"] for p in ps])-.005
                and all(scores[arm]["clean"][p][k] >= b["clean"]["by_provider"][p][k]-.02
                        for p in ps for k in ("d_f1", "macro_f1"))
                and all(scores[arm]["feed_like"][p]["spell_f1"] >=
                        b["feed_like"]["by_provider"][p]["spell_f1"]-.01 for p in ps))
    ranked = sorted(scores, key=lambda a: rank(a, providers), reverse=True)
    if ranked != result["ranked"] or [a for a in ranked if allowed(a,providers)] != result["eligible_ranked"]:
        raise ValueError("Feed-first ranking/guard replay failed")
    for held in providers:
        ps = [p for p in providers if p != held]
        chosen = max([a for a in scores if allowed(a,ps)], key=lambda a: rank(a,ps))
        if chosen != result["cross_provider_selection"][held]["selected"]:
            raise ValueError("Two-provider-only selection replay failed")
    receipt = {"study": "feed_spell_push_v1", "passed": True, "scalar_arms": len(scores),
        "ranking_view": "feed_like", "cohort_cells": len(scores)*15,
        "separate_scalar_implementation": True, "no_new_inference_or_scoring": True,
        "aggregate_sha256": sha256(RESULTS / "feed_spell_push_v1_aggregate.json"),
        "source_sha256": sha256(Path(__file__)),
        "cross_provider_scores": {v: {k: mean([result["cross_provider_selection"][p]["scores"][v][k]
             for p in providers]) for k in keys} for v in ("clean", "feed_like")}}
    with (RESULTS / "feed_spell_push_v1_readout.json").open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
