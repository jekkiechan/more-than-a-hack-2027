"""Bounded, leakage-safe protocol helpers for future touch-rule calibration.

This module deliberately does not translate provider event streams into touch
runs. The sources currently expose different target contracts (see the report),
so a common event adapter must be established before scores can be populated.
"""
from __future__ import annotations

from itertools import product
import math
from numbers import Real


def candidate_grid() -> list[dict]:
    """Return the predeclared 24-arm sensitivity grid for GAR touch runs.

    CH/RE end any run in v1; CL ends a run only when it is a single-frame run.
    The second candidate expands CL to any run; the third treats no end code
    as evidence of an uncontrolled run.
    """
    end_rules = {
        "v1_ch_re_any_cl_single": {"CH": "any", "RE": "any", "CL": "single_frame"},
        "cl_any_run_sensitivity": {"CH": "any", "RE": "any", "CL": "any"},
        "no_uncontrolled_end_codes": {},
    }
    # Keep the calibration bounded: two realistic boundary policies per axis.
    return [
        {
            "uncontrolled_end_rules": dict(end_rules[end_name]),
            "uncontrolled_rule": end_name,
            "uncontrolled_opponent": opponent,
            "censored_run_controlled": censored,
            "out_tail_grace_seconds": grace,
        }
        for end_name, opponent, censored, grace in product(
            end_rules, ("hold", "literal"), (True, False), (0.0, 3.0)
        )
    ]


def select_by_provider_mean(scores_by_provider: dict[str, dict[str, float]],
                            excluded_provider: str) -> dict:
    """Select a candidate using only whole-provider training scores.

    Input maps provider -> candidate-id -> scalar loss (lower is better).
    The excluded provider must be absent, and every included provider must
    provide the same candidate set. One eligible training provider is enough
    (for example, when only Metrica has usable event mapping in a fold). This
    does not itself compute a score or prove match-level separation: the caller
    must ensure each score uses training matches only, and must document the
    match IDs and any inner whole-match split used to produce it.
    """
    if excluded_provider in scores_by_provider:
        raise ValueError("excluded provider appears in calibration data")
    if not scores_by_provider:
        raise ValueError("need at least one eligible training provider")
    providers = sorted(scores_by_provider)
    candidate_ids = set(scores_by_provider[providers[0]])
    if not candidate_ids:
        raise ValueError("empty candidate score set")
    for provider in providers:
        scores = scores_by_provider[provider]
        if set(scores) != candidate_ids:
            raise ValueError("candidate sets differ across providers")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v))
               for v in scores.values()):
            raise ValueError("candidate losses must be finite real numbers")
    means = {
        candidate: sum(float(scores_by_provider[p][candidate]) for p in providers) / len(providers)
        for candidate in sorted(candidate_ids)
    }
    chosen = min(means, key=lambda key: (means[key], key))
    return {"candidate": chosen, "provider_mean_loss": means[chosen],
            "provider_losses": {p: float(scores_by_provider[p][chosen]) for p in providers},
            "training_providers": providers, "excluded_provider": excluded_provider}
