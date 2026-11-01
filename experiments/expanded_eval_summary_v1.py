"""Pool every held-out match scored on the challenge-calibrated feed (`feed_like_v2`).

Sources (aggregate JSON plus per-view benchmark scores; no new inference, no organizer labels):
* 5 development matches: data/experiments/feed_like_v2_benchmark/scores (4 realizations each);
* 2 extra Sportec matches: experiments/results/sportec_extra_eval_v1.json (4 realizations each);
* 19 SoccerNet-GAR VALID/TEST matches: experiments/results/gar_heldout_eval_v1.json
  (2 realizations each, silver event-derived labels).
Each match counts once (mean over its realizations). The paired bootstrap resamples matches
within each group, and resamples groups proportionally for the pooled estimate.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results/expanded_eval_summary_v1.json"
ARMS = ("adop", "cand", "raw", "v2train")
MODES = ("phase0", "phase_tta")
PROVIDER = {"J03WOY": "Sportec", "metrica_3": "Metrica"}


def dev_rows(metric="spell_f1"):
    rows = {}
    for f in glob.glob(str(ROOT / "data/experiments/feed_like_v2_benchmark/scores/*.json")):
        match, _ = Path(f).name[:-5].split("__")
        d = json.loads(Path(f).read_text())
        for arm in ARMS:
            for mode in MODES:
                key = arm if mode == "phase0" else f"{arm}_phase_tta"
                rows.setdefault(match, {}).setdefault((arm, mode), []).append(d[key][metric])
    return {m: {k: float(np.mean(v)) for k, v in r.items()} for m, r in rows.items()}


def sportec_rows(metric="spell_f1"):
    d = json.loads((ROOT / "experiments/results/sportec_extra_eval_v1.json").read_text())["scores"]
    out = {}
    for arm in ARMS:
        for mode in MODES:
            for match, v in d[f"{arm}__{mode}__feed_like_v2"]["per_new_match"].items():
                out.setdefault(match, {})[(arm, mode)] = v[metric]
    return out


def gar_rows(metric="spell_f1"):
    d = json.loads((ROOT / "experiments/results/gar_heldout_eval_v1.json").read_text())["per_game"]
    return {g: {(arm, mode): v[f"feed_v2/{arm}/{mode}"][metric] for arm in ARMS for mode in MODES}
            for g, v in d.items()}


def bootstrap(groups, a, b, draws=4000, seed=20261102):
    rng = np.random.default_rng(seed)
    diffs = {g: np.array([r[a]-r[b] for r in rows.values()]) for g, rows in groups.items()}
    pooled = np.concatenate(list(diffs.values()))
    sample = [np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs.values()]).mean()
              for _ in range(draws)]
    return dict(delta=float(pooled.mean()), ci90=[float(np.percentile(sample, 5)), float(np.percentile(sample, 95))],
                p_positive=float(np.mean(np.array(sample) > 0)), matches=int(len(pooled)),
                matches_positive=int((pooled > 0).sum()))


def run():
    out = {}
    for metric in ("spell_f1", "macro_f1"):
        groups = dict(dev5=dev_rows(metric), sportec_extra2=sportec_rows(metric), gar19=gar_rows(metric))
        groups["full_pitch_dev"] = {m: r for m, r in groups["dev5"].items() if m in PROVIDER}
        means = {g: {f"{a}/{m}": float(np.mean([r[(a, m)] for r in rows.values()])) for a in ARMS for m in MODES}
                 for g, rows in groups.items()}
        pool = {k: v for g in ("dev5", "sportec_extra2", "gar19") for k, v in groups[g].items()}
        means["all26"] = {f"{a}/{m}": float(np.mean([r[(a, m)] for r in pool.values()])) for a in ARMS for m in MODES}
        core = {g: groups[g] for g in ("dev5", "sportec_extra2", "gar19")}
        comparisons = {
            "cand-adop/phase0": (("cand", "phase0"), ("adop", "phase0")),
            "cand-adop/phase_tta": (("cand", "phase_tta"), ("adop", "phase_tta")),
            "cand_tta-adop/phase0": (("cand", "phase_tta"), ("adop", "phase0")),
            "cand_tta-cand/phase0": (("cand", "phase_tta"), ("cand", "phase0")),
            "adop_tta-adop/phase0": (("adop", "phase_tta"), ("adop", "phase0")),
            "v2train-raw/phase0": (("v2train", "phase0"), ("raw", "phase0")),
        }
        boot = {name: dict(all26=bootstrap(core, a, b),
                           **{g: bootstrap({g: rows}, a, b) for g, rows in core.items()})
                for name, (a, b) in comparisons.items()}
        out[metric] = dict(means=means, bootstrap=boot)
    payload = dict(study="expanded_eval_summary_v1", view="feed_like_v2 (challenge-calibrated)",
                   matches=dict(dev5=5, sportec_extra2=2, gar19=19), weighting="each match once",
                   organizer_reference_access=False, public_scoring=False, promotion=False, **out)
    RESULT.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


if __name__ == "__main__":
    r = run()
    for g, m in r["spell_f1"]["means"].items():
        print(g, {k: round(v, 4) for k, v in m.items()})
    for k, v in r["spell_f1"]["bootstrap"].items():
        print(k, {g: (round(x["delta"], 4), [round(c, 4) for c in x["ci90"]], round(x["p_positive"], 2)) for g, x in v.items()})
