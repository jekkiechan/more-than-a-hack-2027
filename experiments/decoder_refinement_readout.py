"""Scalar-only independent hierarchy replay and refinement readout."""
import json

from experiments import decoder_refinement_run as study


def hierarchy(rows, view, key):
    providers={}
    for provider in ("Metrica","Sportec","SkillCorner"):
        matches={r["match"] for r in rows if r["provider"]==provider}
        means=[]
        for match in matches:
            values=[r[key] for r in rows if r["provider"]==provider and r["match"]==match
                    and (r["view"]=="clean" if view=="clean" else r["view"].startswith("feed_like_"))]
            if len(values)!=(1 if view=="clean" else 2):
                raise ValueError("Incomplete scalar cohort")
            means.append(sum(values)/len(values))
        providers[provider]=sum(means)/len(means)
    return sum(providers.values())/3


def main():
    result=json.loads(study.AGGREGATE.read_text())
    declaration=json.loads(study.DECLARATION.read_text())
    if result["declaration_sha256"]!=study.shared.sha256_file(study.DECLARATION):
        raise ValueError("Declaration binding failed")
    for config in declaration["arms"]:
        arm=config["id"]
        doc=json.loads((study.OUT/(arm+".json")).read_text())
        if doc["config"]!=config or len(doc["rows"])!=15:
            raise ValueError("Scalar receipt configuration/count differs")
        for view in ("clean","feed_like"):
            for key in ("spell_f1","macro_f1","d_f1","d_precision","d_recall","reference_spell_count","predicted_spell_count"):
                if abs(hierarchy(doc["rows"],view,key)-result["arms"][arm][view]["equal_provider_mean"][key])>1e-12:
                    raise ValueError("Independent hierarchy mismatch")
    ranked=sorted(result["arms"],key=lambda a:tuple(result["arms"][a]["clean"]["equal_provider_mean"][k]
                  for k in ("spell_f1","macro_f1")),reverse=True)
    if ranked!=result["ranked"]:
        raise ValueError("Independent ranking differs")
    selected=result["selected_config"]["id"]
    summary=result["arms"][selected]
    families={}
    for arm in ranked:
        config=next(c for c in declaration["arms"] if c["id"]==arm)
        families.setdefault(config["family"],{"config":config,"scores":result["arms"][arm],
                             "passes_guards":arm in result["eligible_ranked"]})
    cross={v:{k:sum(result["cross_provider_selection"][p]["scores"][v][k]
                  for p in study.shared.PROVIDERS)/3 for k in ("spell_f1","macro_f1","d_f1")}
           for v in ("clean","feed_like")}
    output={"study":"decoder_refinement_v3","aggregate_sha256":study.shared.sha256_file(study.AGGREGATE),
        "independent_hierarchy_and_rank_passed":True,"all_118_scalar_receipts_verified":True,
        "selected_config":result["selected_config"],"family_best":families,"cross_provider_mean":cross,
        "cross_provider_minus_adopted":{v:{k:x-result["baseline"][v]["equal_provider_mean"][k]
            for k,x in cross[v].items()} for v in cross},
        "selected_minus_adopted":{v:{k:summary[v]["equal_provider_mean"][k]-result["baseline"][v]["equal_provider_mean"][k]
            for k in ("spell_f1","macro_f1","d_f1")} for v in ("clean","feed_like")},
        "selected_minus_previous":{v:{k:summary[v]["equal_provider_mean"][k]-result["previous_winner"][v]["equal_provider_mean"][k]
            for k in ("spell_f1","macro_f1","d_f1")} for v in ("clean","feed_like")},
        "plus_0_01_clean_screen":summary["clean"]["equal_provider_mean"]["spell_f1"]-result["baseline"]["clean"]["equal_provider_mean"]["spell_f1"]>=.01,
        "public_labels_read":False}
    study.previous.write_once(study.RESULTS/'decoder_refinement_v3_readout.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='family_best'},indent=2))


if __name__=='__main__':
    main()
