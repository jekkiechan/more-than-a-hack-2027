"""Feature-only deployment of the sealed original_full phase-average recipe."""
from __future__ import annotations

import numpy as np

from experiments import calibrated_dense_baseline as calibration
from experiments import feed_like_v2_public, feed_spell_more, feed_spell_push
from experiments import original_full_baseline, supervised, tracking_corruption

KIND = "sn_dense_original_diversity_phase_ensemble"
PHASE_AVERAGING = dict(rate_hz=5, weighting="uniform", order="seeds_then_phases", native_fps=25, phases=5)


def native_probability(inputs, models, manifest):
    architecture = manifest["architecture"]
    if (architecture.get("kind") != KIND or architecture.get("phase_tta") is not True
            or architecture.get("phase_averaging") != PHASE_AVERAGING
            or architecture.get("postprocessing") != original_full_baseline.POSTPROCESSING):
        raise ValueError("Expected the sealed original_full phase-average package")
    clean = tracking_corruption.inputs_only(inputs)
    if float(clean["fps"]) != 25:
        raise ValueError("This sealed deployment package requires 25 Hz tracking")
    phases = feed_like_v2_public.phase_native(clean, models, manifest)
    if len(phases) != 5:
        raise ValueError("Expected all five sampling phases")
    native = np.mean(phases, axis=0)
    if (native.shape != (len(clean["ball"]), 3) or not np.isfinite(native).all()
            or (native < 0).any() or not np.allclose(native.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid phase-averaged native probabilities")
    return native


def predict_probabilities(inputs, models, manifest):
    native = native_probability(inputs, models, manifest)
    return feed_spell_more.emissions(native[None], supervised.model_periods(inputs),
                                    float(inputs["fps"]), feed_spell_more.BASE)


def predict_labels(inputs, models, manifest):
    native = native_probability(inputs, models, manifest)
    return feed_spell_push.decode(native[None], supervised.model_periods(inputs),
                                 float(inputs["fps"]), calibration.CONFIG)
