"""Recompute selection and guards from saved scalars; never rescore labels."""
from __future__ import annotations

import json
from collections import defaultdict

from experiments import decoder_search_run as study, spell_mbr_run as shared


def readout():
    aggregate_path = study.RESULTS / "decoder_search_v2_aggregate.json"
    result = json.loads(aggregate_path.read_text())
    declaration = json.loads(study.DECLARATION.read_text())
    if shared.sha256_file(study.DECLARATION) != result["declaration_sha256"]:
        raise ValueError("Declaration binding failed")
    baseline = result["baseline"]
    config = {c["id"]: c for c in declaration["arms"]}
    summaries = {}
    for arm in config:
        receipt = json.loads((study.OUT / (arm+".json")).read_text())
        if receipt["config"] != config[arm] or len(receipt["rows"]) != 15:
            raise ValueError("Incomplete arm receipt")
        for row in receipt["rows"]:
            control = baseline["clean" if row["view"] == "clean" else "feed_like"]["by_provider"][row["provider"]]
            # Aggregate counts below complement per-cell equality in the runner.
            if row["reference_spell_count"] <= 0 or row["graded_frames"] <= 0 or control["reference_spell_count"] <= 0:
                raise ValueError("Invalid grading population")
        summaries[arm] = study.summarize(receipt["rows"])
        if summaries[arm] != result["arms"][arm]:
            raise ValueError("Saved aggregate differs from per-arm receipts")
    ranked = sorted(summaries, key=lambda a: study.rank(summaries[a]), reverse=True)
    eligible = [a for a in ranked if study.guards(summaries[a], baseline)]
    if ranked != result["ranked_arm_ids"] or eligible != result["eligible_ranked_arm_ids"]:
        raise ValueError("Selection replay failed")
    best = eligible[0]
    families = defaultdict(list)
    for arm in ranked:
        families[config[arm]["family"]].append(arm)
    family_best = {family: {"config": config[arms[0]], "passes_guards": arms[0] in eligible,
                          **{v: summaries[arms[0]][v]["equal_provider_mean"] for v in ("clean", "feed_like")}}
                   for family, arms in families.items()}
    cross_means = {v: {key: sum(result["cross_provider_selection"][p]["held_provider"][v][key]
                               for p in shared.PROVIDERS)/3
                     for key in ("spell_f1", "macro_f1", "d_f1")}
                   for v in ("clean", "feed_like")}
    payload = {
        "study": "decoder_search_v2", "aggregate_sha256": shared.sha256_file(aggregate_path),
        "selection_replay_passed": True, "all_134_arm_scalar_aggregates_match": len(summaries) == 134,
        "best_unconstrained_config": config[ranked[0]], "best_guard_passing_config": config[best],
        "family_best": family_best, "cross_provider_selected_mean": cross_means,
        "cross_provider_selected_minus_baseline": {
            v: {key: value-baseline[v]["equal_provider_mean"][key] for key, value in values.items()}
            for v, values in cross_means.items()},
        "best_minus_baseline": {v: {key: summaries[best][v]["equal_provider_mean"][key]-baseline[v]["equal_provider_mean"][key]
                                   for key in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "predicted_spell_count")}
                                for v in ("clean", "feed_like")},
        "clean_gain_plus_0_01_screen_passes": study.rank(summaries[best])[0]-study.rank(baseline)[0] >= .01,
        "best_guards_pass": study.guards(summaries[best], baseline),
        "source_arrays": "Same reconstructed control_D2 predictions, with full provider exclusions; original fold weights were unavailable",
        "evidence_limit": "134-way search on five repeatedly exposed development matches; synthetic stress; no independent or organizer confirmation",
        "promotion_performed": False,
    }
    study.write_once(study.RESULTS / "decoder_search_v2_readout.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "family_best"}, indent=2))


if __name__ == "__main__":
    readout()
