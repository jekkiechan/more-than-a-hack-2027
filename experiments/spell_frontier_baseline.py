"""Feature-only inference for the frozen three-expert spell-frontier package."""
from __future__ import annotations

import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import live_gap_experiment, soccernet_extension as old
from experiments import sn_dense_train as sn
from experiments import supervised, tracking_corruption, v2_iterative_runtime
from experiments import v2_spell_balanced_eval as blend
from experiments.v2_spell_balanced_public import _pool_seed_phase_probabilities

KIND = "sn_dense_spell_frontier_ensemble"
RECIPE = dict(ce_weight=.5, calibrated_weight=.375, adopted_weight=.125,
              pooling="arithmetic", ha_cost=3.)
ROLES = ("ce", "adopted", "calibrated", "teacher_adopted", "teacher_calibrated")
SEEDS = (2026, 2027)


def validate(models, manifest):
    arch = manifest["architecture"]
    if arch.get("kind") != KIND or arch.get("recipe") != RECIPE:
        raise ValueError("Expected the frozen spell-frontier recipe")
    if arch.get("phase_order") != "seeds_then_phases":
        raise ValueError("Expected seed means within phase, then phase mean")
    if set(models) != {f"{role}_{seed}" for role in ROLES for seed in SEEDS}:
        raise ValueError("The ten frozen model roles are required")


def components_from_records(inputs, records, models, manifest):
    """Use shared native phase records, with no labels entering prediction."""
    validate(models, manifest)
    clean = tracking_corruption.inputs_only(inputs)
    by_expert = {role: [] for role in ("ce", "adopted", "calibrated")}
    for phase in records:
        record = old.inference_record(phase)
        record["names"] = list(record["names"])
        for role in by_expert:
            teacher_role = "teacher_calibrated" if role == "calibrated" else "teacher_adopted"
            paired = []
            for seed in SEEDS:
                sampled = sn.predict(record, models[f"{role}_{seed}"], models[f"{teacher_role}_{seed}"])
                paired.append(old.upsample(sampled, record, clean))
            by_expert[role].append(paired)
    if not records:
        raise ValueError("Native phases are required")
    components = {}
    for role, rows in by_expert.items():
        array = np.transpose(np.asarray(rows, dtype=np.float64), (1, 0, 2, 3))
        components[role] = _pool_seed_phase_probabilities(array.tolist())
    return components


def native_components(inputs, models, manifest):
    clean = tracking_corruption.inputs_only(inputs)
    if float(clean["fps"]) not in (10., 25.):
        raise ValueError("Use explicit native phase records for other frame rates")
    placeholder = np.full(len(clean["ball"]), "X", dtype="U1")
    with v2_iterative_runtime.installed():
        feature = live_gap_experiment.feature_view(clean, placeholder, "competition", "BAR-ATH")["joint"]
        records = audit.phase_records(clean, old.inference_record(feature))
    if len(records) != round(float(clean["fps"]) / 5):
        raise ValueError("Incomplete native phase inventory")
    return components_from_records(clean, records, models, manifest)


def decode_components(inputs, components):
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    adopted = audit.decode(components["adopted"], period, fps, audit.CONFIG)
    probability = blend.recipe_probability(components, RECIPE["ce_weight"],
                                            RECIPE["calibrated_weight"], RECIPE["pooling"])
    labels = blend.decode_fixed_d(probability, adopted, period, fps, RECIPE["ha_cost"])
    if not np.array_equal(np.asarray(labels) == "D", np.asarray(adopted) == "D"):
        raise ValueError("The frozen adopted D mask changed")
    return labels


def predict_labels(inputs, models, manifest):
    return decode_components(inputs, native_components(inputs, models, manifest))


def predict_probabilities(inputs, models, manifest):
    raise ValueError("This package requires predict_labels to preserve its decoded D mask")
