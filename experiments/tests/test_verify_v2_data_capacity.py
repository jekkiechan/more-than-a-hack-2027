"""Focused contracts for the independent v2 capacity readout verifier."""
import unittest

from scripts.verify_v2_data_capacity import (
    METRICS, VIEWS, evaluate_gates, historical_diagnostics, recompute_aggregates,
    verify_dead_mask, _verify_model_artifact, _verify_training_cross_arm_contract,
)


class CapacityVerifierTests(unittest.TestCase):
    def test_recompute_respects_rep_match_provider_hierarchy(self):
        matches = {}
        for provider, match, values in (
            ("p1", "m1", (.2, .4)),
            ("p1", "m2", (.8,)),
            ("p2", "m3", (.6,)),
        ):
            for view in VIEWS:
                for rep, value in enumerate(values, 1):
                    metrics = {name: value for name in METRICS}
                    matches[f"{provider}|{match}|{view}|{rep}"] = {
                        "metrics": {"candidate": metrics},
                    }
        # Restrict the helper to a single candidate arm for this contract.
        aggregate = recompute_aggregates(matches, ["candidate"])
        # p1: mean((.2+.4)/2, .8) = .55; p2: .6; providers equal.
        self.assertAlmostEqual(aggregate["clean"]["candidate"]["spell_f1"]["providers"]["p1"], .55)
        self.assertAlmostEqual(aggregate["clean"]["candidate"]["spell_f1"]["equal_provider"], .575)

    def test_training_cross_arm_contract_rejects_sampler_or_mixed_draw_drift(self):
        arms = ("width96_control", "width192_capacity", "mixed_v1_v2_width96")
        rows = []
        for arm in arms:
            draws = {"m1/v2r0": 3, "m1/v2r1": 3, "m1/v2r2": 3, "m1/v2r3": 3}
            if arm == "mixed_v1_v2_width96":
                draws = {"m1/v1": 12, "m1/v2r0": 3, "m1/v2r1": 3,
                         "m1/v2r2": 3, "m1/v2r3": 3}
            rows.append({"arm": arm, "fold": "leave_out_p", "seed": 2026,
                         "sampler": {"sampler": "same"}, "exposures": {"old/m1": 10},
                         "training_matches": ["m1"], "input_view_draws": draws})
        # The helper requires six fold/seed groups; duplicate this valid group under new IDs.
        rows = [dict(row, fold=f"leave_out_p{i//2}", seed=2026+i%2)
                for i in range(6) for row in rows]
        _verify_training_cross_arm_contract(rows)
        sampler_drift = [dict(row) for row in rows]
        sampler_drift[2]["sampler"] = {"sampler": "changed"}
        with self.assertRaisesRegex(ValueError, "sampler differs"):
            _verify_training_cross_arm_contract(sampler_drift)
        draw_drift = [dict(row, input_view_draws=dict(row["input_view_draws"])) for row in rows]
        draw_drift[2]["input_view_draws"]["m1/v1"] = 20
        with self.assertRaisesRegex(ValueError, "family mass is not balanced"):
            _verify_training_cross_arm_contract(draw_drift)

    def test_serialized_model_checks_hidden_shapes_count_and_fold_normalization(self):
        import numpy as np
        width, features = 96, 408
        state = {
            "sequence.weight_ih_l0": np.zeros((3*width, features)),
            "sequence.weight_hh_l0": np.zeros((3*width, width)),
            "sequence.bias_ih_l0": np.zeros(3*width),
            "sequence.bias_hh_l0": np.zeros(3*width),
            "sequence.weight_ih_l0_reverse": np.zeros((3*width, features)),
            "sequence.weight_hh_l0_reverse": np.zeros((3*width, width)),
            "sequence.bias_ih_l0_reverse": np.zeros(3*width),
            "sequence.bias_hh_l0_reverse": np.zeros(3*width),
            "output.weight": np.zeros((2, 2*width)),
            "output.bias": np.zeros(2),
        }
        teacher = {"state": {"weight": np.asarray([1.])}, "median": np.zeros(features),
                   "scale": np.ones(features), "feature_names": [f"f{i}" for i in range(features)],
                   "input_features": features}
        model = {"kind": "soccernet_capacity", "hidden": width, "state": state,
                 "d_state": {"weight": np.asarray([2.])}, "median": teacher["median"].copy(),
                 "scale": teacher["scale"].copy(), "feature_names": list(teacher["feature_names"]),
                 "input_features": features, "step": 2048,
                 "training": {"steps": 2048}}
        _verify_model_artifact(model, teacher, width, "width96_control", "leave_out_x", 2026)
        model["hidden"] = 192
        with self.assertRaisesRegex(ValueError, "Serialized hidden width"):
            _verify_model_artifact(model, teacher, width, "width96_control", "leave_out_x", 2026)

    def test_protected_decoder_requires_identical_dead_mask(self):
        import numpy as np
        adopted = np.array([False, True, False, True])
        verify_dead_mask(adopted, adopted.copy())
        with self.assertRaisesRegex(ValueError, "changed the frozen adopted D mask"):
            verify_dead_mask(np.array([False, False, False, True]), adopted)

    def test_historical_minus_point_zero_two_provider_guard_is_separate(self):
        aggregate = {
            "clean": {
                "candidate": {metric: {"providers": {"p1": .79}} for metric in ("macro_f1", "d_f1")},
                "adopted": {metric: {"providers": {"p1": .8}} for metric in ("macro_f1", "d_f1")},
            },
            "feed_like_v2": {
                "candidate": {"spell_f1": {"equal_provider": .59}},
                "adopted": {"spell_f1": {"equal_provider": .60}},
            },
        }
        outcome = historical_diagnostics(aggregate, "candidate", "adopted")
        self.assertTrue(outcome["all_passed"])
        self.assertFalse(outcome["primary_gate"])

    def test_gate_is_strict_at_adopted_and_pr82_floors(self):
        strict_metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                          "short_spell_recall_iou_0_5_lt3s")
        adopted = {view: {metric: 0.6 for metric in strict_metrics}
                   for view in VIEWS}
        candidate = {view: dict(metrics) for view, metrics in adopted.items()}
        candidate["feed_like_v2"]["spell_f1"] = .5927491913
        self.assertFalse(evaluate_gates(candidate, adopted)["all_passed"])
        candidate["feed_like_v2"]["spell_f1"] = .6
        self.assertTrue(evaluate_gates(candidate, adopted)["all_passed"])
        candidate["clean"]["d_f1"] -= 1e-12
        self.assertFalse(evaluate_gates(candidate, adopted)["all_passed"])


if __name__ == "__main__":
    unittest.main()
