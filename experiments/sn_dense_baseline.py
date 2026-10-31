"""Feature-only inference for the adopted sn_dense_v1 D2 two-seed ensemble.

Each member is a width-96 conditional H/A head plus a fine-tuned D path, paired
with its pooled combined-objective teacher for frozen normalization. Members
average native H/A/D probabilities before the cost-3 decoder. No study gates,
labels or training data are read.
"""
from __future__ import annotations

import numpy as np

from experiments import live_gap_experiment, soccernet_capacity as sc, sn_dense_train as sn
from experiments import soccernet_capacity_model as capacity

MEMBERS = {"dense_2026": 0.5, "dense_2027": 0.5}
TEACHERS = {f"dense_{seed}": f"teacher_{seed}" for seed in (2026, 2027)}


def validate_models(models, architecture):
    if (architecture.get("members") != MEMBERS
            or architecture.get("teachers") != TEACHERS
            or set(models) != set(MEMBERS) | set(TEACHERS.values())
            or architecture.get("decoder_switch_cost") != 3
            or sc.lt.CORE != 50 or sc.lt.CONTEXT != 25):
        raise ValueError("Expected two fixed dense/teacher pairs and cost-3 original-context decoding")
    for seed in (2026, 2027):
        tuned, original = models[f"dense_{seed}"], models[f"teacher_{seed}"]
        training = tuned.get("training", {})
        if (tuned.get("kind") != "soccernet_capacity" or "d_state" not in tuned
                or tuned.get("hidden") != 96 or tuned.get("step") != 2048
                or training.get("seed") != seed or training.get("lr") != 3e-4
                or training.get("fraction") != 0.5 or training.get("ha_target") != "dense"
                or training.get("train_d") is not True or original.get("seed") != seed):
            raise ValueError("Dense member seed, recipe or teacher mismatch")
        if (tuned.get("source", {}).get("checkpoint_sha256") != capacity._checkpoint_sha(original)
                or list(tuned.get("feature_names", [])) != list(original.get("feature_names", []))
                or not np.array_equal(tuned.get("median"), original.get("median"))
                or not np.array_equal(tuned.get("scale"), original.get("scale"))):
            raise ValueError("Dense checkpoint has the wrong teacher or normalization")


def member_probabilities(inputs, models, manifest):
    """Mirror the sealed sn_dense_public predictor exactly."""
    validate_models(models, manifest["architecture"])
    placeholder = np.full(len(inputs["ball"]), "X", dtype="U1")
    record = sc.old.inference_record(live_gap_experiment.feature_view(
        inputs, placeholder, "competition", "BAR-ATH")["joint"])
    result = {}
    for member, teacher in TEACHERS.items():
        probability = sc.old.upsample(sn.predict(record, models[member], models[teacher]), record, inputs)
        if (probability.shape != (len(inputs["ball"]), 3) or not np.isfinite(probability).all()
                or not np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid dense member H/A/D probabilities")
        result[member] = probability
    return result


def predict_probabilities(inputs, models, manifest):
    members = list(member_probabilities(inputs, models, manifest).values())
    return (members[0] + members[1]) * 0.5
