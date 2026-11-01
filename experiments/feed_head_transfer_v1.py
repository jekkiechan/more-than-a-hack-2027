"""Bounded component transfer if raw-augmentation ensembles miss 0.55.

Two fixed H/A-versus-D crosses, using only already sealed provider-held-out
models. No fit, calibration search, or organizer reference access.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse

import numpy as np
from experiments import feed_robustness_v1 as parent
from experiments import spell_mbr_run as shared, decoder_search_run as scoring
from experiments import feed_spell_push_run as selection

STUDY = "feed_head_transfer_v1"
OUT = parent.ROOT / "data/experiments" / STUDY
DECLARATION = parent.RESULTS / f"{STUDY}_declaration.json"
ARMS = ("robust_ha", "robust_d")


def cross(ha, dead):
    """Preserve conditional H/A from one model and D from the other."""
    ha, dead = np.asarray(ha), np.asarray(dead)
    if ha.shape != dead.shape or ha.ndim != 2 or ha.shape[1] != 3:
        raise ValueError("Aligned H/A/D probability matrices required")
    if not np.isfinite(ha).all() or not np.isfinite(dead).all() or (ha < 0).any() or (dead < 0).any():
        raise ValueError("Finite nonnegative probabilities required")
    if not np.allclose(ha.sum(axis=1), 1, atol=1e-5) or not np.allclose(dead.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Normalized probabilities required")
    mass = ha[:, :2].sum(axis=1, keepdims=True)
    q = np.divide(ha[:, :2], mass, out=np.full_like(ha[:, :2], .5, dtype=float), where=mass > 0)
    d = dead[:, 2:3]
    return np.column_stack((q*(1-d), d))


def declare():
    parent.write(DECLARATION, {"study": STUDY, "utc": datetime.now(timezone.utc).isoformat(),
        "condition": "Run only if parent study does not reach 0.55; two arms fixed before parent scores are available.",
        "hypothesis": "Raw defects may help team attribution and dead-ball inference differently; retain a useful component without the other component's regression.",
        "arms": {"robust_ha": "Augmented conditional H/A, baseline D", "robust_d": "Baseline conditional H/A, augmented D"},
        "ensemble": "Cross within each matched seed first, then average native probabilities. Apply unchanged f062 decoder after averaging.",
        "fits": 0, "scored_cells": 90, "guards": "Same feed-first guards as parent against promoted baseline",
        "selection": "Rank these two arms and the three frozen parent arms on original development feed spell, macro tie-break. No fresh-corruption results used.",
        "parent_declaration_sha256": parent.sha(parent.DECLARATION),
        "source_sha256": {str(Path(__file__).relative_to(parent.ROOT)): parent.sha(Path(__file__))},
        "organizer_reference_access": False, "promotion": False,
        "evidence": "Reused development matches and synthetic defects. This component comparison is not independent confirmation."})


def run():
    declared = parent.read(DECLARATION)
    for rel, digest in declared["source_sha256"].items():
        if parent.sha(parent.ROOT / rel) != digest:
            raise ValueError("Frozen head-transfer source changed")
    if declared["parent_declaration_sha256"] != parent.sha(parent.DECLARATION):
        raise ValueError("Parent declaration changed")
    d = parent.check()
    previous = parent.read(parent.RESULTS / "feed_robustness_v1_aggregate.json")
    if previous["stretch_reached"]:
        parent.write(parent.RESULTS / f"{STUDY}_skipped.json", {"reason": "Parent reached the stretch target", "scored": False})
        return
    verification = parent.read(parent.RESULTS / "feed_robustness_v1_verification.json")
    if not verification["passed"] or verification["aggregate_sha256"] != parent.sha(parent.RESULTS / "feed_robustness_v1_aggregate.json"):
        raise ValueError("Verified parent evidence required")
    parent.write(OUT / "score_lock.json", {"declaration_sha256": parent.sha(DECLARATION),
        "parent_aggregate_sha256": parent.sha(parent.RESULTS / "feed_robustness_v1_aggregate.json"),
        "candidate_predictions": previous["prediction_sha256"]})
    cells = shared._load_cells(d)
    predictions = {a: {} for a in ARMS}
    for c in cells:
        if c["member"] == "ensemble":
            continue
        key = tuple(c[k] for k in ("provider", "match", "view", "member"))
        path = parent.OUT / "predictions/feed_augmented" / ("__".join(key)+".npz")
        if parent.sha(path) != previous["prediction_sha256"][str(path.relative_to(parent.ROOT))]:
            raise ValueError("Candidate probabilities changed")
        with np.load(path) as z:
            p = z["probabilities"]
        predictions["robust_ha"][key] = cross(p, c["probabilities"])
        predictions["robust_d"][key] = cross(c["probabilities"], p)
    rows, files = {a: [] for a in ARMS}, {}
    for arm in ARMS:
        for c in cells:
            key = tuple(c[k] for k in ("provider", "match", "view", "member"))
            p = (np.mean([predictions[arm][key[:-1]+(str(s),)] for s in parent.SEEDS], axis=0)
                 if c["member"] == "ensemble" else predictions[arm][key])
            row, prediction = parent.score(c, p)
            rows[arm].append(row)
            path = OUT / "predictions" / arm / ("__".join(key)+".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probabilities=p, prediction=prediction,
                labels=c["labels"], period=c["period"], timestamps=c["timestamps"],
                model_period=c["model_period"], fps=c["fps"])
            files[str(path.relative_to(parent.ROOT))] = parent.sha(path)
    summaries = {**previous["summaries"], **{a: {m: scoring.summarize(r, m) for m in shared.MEMBERS} for a, r in rows.items()}}
    baseline = summaries["baseline"]["ensemble"]
    guards = {a: selection.eligible(s["ensemble"], baseline) for a, s in summaries.items()}
    winner = max((a for a in summaries if guards[a]), key=lambda a: selection.rank(summaries[a]["ensemble"]))
    parent.write(parent.RESULTS / f"{STUDY}_aggregate.json", {"study": STUDY, "rows": rows,
        "summaries": summaries, "guards_pass": guards, "winner": winner,
        "stretch_reached": summaries[winner]["ensemble"]["feed_like"]["equal_provider_mean"]["spell_f1"] >= .55,
        "prediction_sha256": files, "organizer_reference_access": False})
    for arm in ARMS:
        print(arm, {v: summaries[arm]["ensemble"][v]["equal_provider_mean"]["spell_f1"] for v in ("clean", "feed_like")}, guards[arm])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("declare", "run"))
    globals()[p.parse_args().action]()
