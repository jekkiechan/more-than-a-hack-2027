"""Feature-only deployment of the sealed original_full two-seed recipe."""
from __future__ import annotations

import numpy as np

from experiments import calibrated_dense_baseline as calibration
from experiments import feed_spell_more, feed_spell_push, sn_dense_baseline
from experiments import supervised, tracking_corruption

KIND = "sn_dense_original_diversity_ensemble"
POSTPROCESSING = dict(calibration.POSTPROCESSING,
    member_normalization="after_equal_native_probability_pooling")


def native_probability(inputs, models, manifest):
    architecture = manifest["architecture"]
    if (architecture.get("kind") != KIND
            or architecture.get("postprocessing") != POSTPROCESSING):
        raise ValueError("Expected the sealed original_full temperature-4 package")
    # Mirror short_quality_public_v1.prepare: strip target fields, build shared
    # features, predict the two paired models, then pool BEFORE normalization.
    members = sn_dense_baseline.member_probabilities(
        tracking_corruption.inputs_only(inputs), models, manifest)
    return np.mean([members[name] for name in sn_dense_baseline.MEMBERS], axis=0)


def predict_probabilities(inputs, models, manifest):
    native = native_probability(inputs, models, manifest)
    return feed_spell_more.emissions(native[None], supervised.model_periods(inputs),
                                    float(inputs["fps"]), feed_spell_more.BASE)


def predict_labels(inputs, models, manifest):
    native = native_probability(inputs, models, manifest)
    return feed_spell_push.decode(native[None], supervised.model_periods(inputs),
                                  float(inputs["fps"]), calibration.CONFIG)
