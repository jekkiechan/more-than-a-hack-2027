"""Feature-only inference for the adopted SoccerNet conditional H/A ensemble.

The two capacity checkpoints require their matching original checkpoints for
frozen normalization and P(D). No study gates, labels or training data are read.
"""
from __future__ import annotations

import numpy as np

from experiments import learned_temporal as lt, live_gap_experiment, supervised
from experiments import soccernet_capacity_model as capacity

MEMBERS = {"capacity_2026": 0.5, "capacity_2027": 0.5}
TEACHERS = {f"capacity_{seed}": f"teacher_{seed}" for seed in (2026, 2027)}


def validate_models(models, architecture):
    if (architecture.get("members") != MEMBERS
            or architecture.get("teachers") != TEACHERS
            or set(models) != set(MEMBERS) | set(TEACHERS.values())
            or architecture.get("decoder_switch_cost") != 3
            or lt.CORE != 50 or lt.CONTEXT != 25):
        raise ValueError("Expected two fixed capacity/teacher pairs and cost-3 original-context decoding")
    for seed in (2026, 2027):
        tuned = models[f"capacity_{seed}"]
        original = models[f"teacher_{seed}"]
        training = tuned.get("training", {})
        if (tuned.get("kind") != "soccernet_capacity"
                or tuned.get("hidden") != 96 or tuned.get("step") != 2048
                or training.get("seed") != seed or training.get("lr") != 3e-4
                or training.get("fraction") != 0.5
                or original.get("seed") != seed
                or original.get("player_only") is not True
                or original.get("context_seconds", 5) != 5):
            raise ValueError("Capacity/teacher seed, recipe or context mismatch")
        if (tuned.get("source", {}).get("checkpoint_sha256") != capacity._checkpoint_sha(original)
                or tuned.get("source", {}).get("seed") != seed
                or list(tuned.get("feature_names", [])) != list(original.get("feature_names", []))
                or not np.array_equal(tuned.get("median"), original.get("median"))
                or not np.array_equal(tuned.get("scale"), original.get("scale"))):
            raise ValueError("Capacity checkpoint has the wrong frozen teacher or normalization")


def member_probabilities(inputs, models, manifest):
    validate_models(models, manifest["architecture"])
    placeholder = np.full(len(inputs["ball"]), "X", dtype="U1")
    joint = live_gap_experiment.feature_view(inputs, placeholder, "competition", "BAR-ATH")["joint"]
    # Match the sealed capacity predictor's feature-only record exactly. In
    # particular, it does not pass targets or add timestamp-based window splits.
    record = {key: joint[key] for key in ("x", "names", "period", "indices", "native_n", "fps")}
    periods = supervised.model_periods(inputs)
    result = {}
    for member, teacher in TEACHERS.items():
        sampled = capacity.predict(record, models[member], models[teacher])
        probability = np.column_stack([
            lt.upsample(sampled[:, k], record, periods) for k in range(3)
        ])
        if (probability.shape != (len(inputs["ball"]), 3)
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid capacity member H/A/D probabilities")
        result[member] = probability
    return result


def predict_probabilities(inputs, models, manifest):
    members = list(member_probabilities(inputs, models, manifest).values())
    return (members[0] + members[1]) * 0.5
