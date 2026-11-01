"""Inference for the adopted dense ensemble with frozen f062 postprocessing."""
from __future__ import annotations

import numpy as np

from experiments import feed_spell_more, feed_spell_push, sn_dense_baseline, supervised

KIND = "sn_dense_calibrated_ensemble"
POSTPROCESSING = {"kind": "dead_logit_temperature", "temperature": 4.0,
                  "member_normalization": "before_equal_native_probability_pooling",
                  "ha_switch_cost": 3, "dead_switch_cost": 3,
                  "frozen_study": "feed_spell_push_v1", "frozen_arm": "f062"}
CONFIG = dict(family="calibration", dt=4, ht=1, ha=3, dead=3, id="f062")


def _members(inputs, models, manifest):
    architecture = manifest["architecture"]
    if architecture.get("kind") != KIND or architecture.get("postprocessing") != POSTPROCESSING:
        raise ValueError("Expected the frozen f062 temperature-4 package")
    pieces = sn_dense_baseline.member_probabilities(inputs, models, manifest)
    return np.stack([pieces[member] for member in sn_dense_baseline.MEMBERS])


def probabilities_from_members(members, periods, fps):
    """Calibrated emissions for callers that need native H/A/D probabilities."""
    return feed_spell_more.emissions(members, periods, fps, feed_spell_more.BASE)


def predict_probabilities(inputs, models, manifest):
    return probabilities_from_members(_members(inputs, models, manifest),
        supervised.model_periods(inputs), float(inputs["fps"]))


def predict_labels(inputs, models, manifest):
    # Use the exact selected execution path, including normalization and ties.
    return feed_spell_push.decode(_members(inputs, models, manifest),
        supervised.model_periods(inputs), float(inputs["fps"]), CONFIG)
