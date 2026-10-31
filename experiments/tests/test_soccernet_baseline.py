"""Promotion integrity and feature-only two-pair baseline inference contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import current_baseline, soccernet_baseline as baseline
from experiments import promote_soccernet_baseline as promotion


class SoccerNetBaselineContracts(unittest.TestCase):
    def setUp(self):
        self.models = {}
        for seed in (2026, 2027):
            teacher = dict(seed=seed, player_only=True, kind="factorized_had",
                           feature_names=["one"], median=np.zeros(1), scale=np.ones(1),
                           state={"weight": np.array([seed], dtype=np.float32)})
            tuned = dict(kind="soccernet_capacity", hidden=96, step=2048,
                         training=dict(seed=seed, lr=3e-4, fraction=.5),
                         feature_names=["one"], median=np.zeros(1), scale=np.ones(1),
                         source=dict(seed=seed, checkpoint_sha256=baseline.capacity._checkpoint_sha(teacher)))
            self.models[f"capacity_{seed}"] = tuned
            self.models[f"teacher_{seed}"] = teacher
        self.manifest = dict(architecture=dict(kind="soccernet_capacity_ensemble",
                            members=baseline.MEMBERS.copy(), teachers=baseline.TEACHERS.copy(),
                            decoder_switch_cost=3))
        self.inputs = dict(ball=np.zeros((4, 2)), period=np.ones(4), fps=np.asarray(5.))
        self.record = dict(x=np.zeros((4, 1)), names=["one"], period=np.zeros(4),
                           indices=np.arange(4), native_n=4, fps=5., y=object(),
                           timestamps=np.arange(4) / 5)
        self.a = np.tile([.8, .1, .1], (4, 1))
        self.b = np.tile([.2, .6, .2], (4, 1))

    def test_matching_pairs_average_native_probabilities_with_feature_only_shared_record(self):
        with patch.object(baseline.live_gap_experiment, "feature_view", return_value={"joint": self.record}) as features, \
             patch.object(baseline.capacity, "predict", side_effect=[self.a, self.b]) as predict:
            result = current_baseline.predict_probabilities(self.inputs, self.models, self.manifest)
        np.testing.assert_array_equal(result, (self.a + self.b) * .5)
        self.assertEqual(features.call_count, 1)
        self.assertTrue(np.all(features.call_args.args[1] == "X"))
        for index, seed in enumerate((2026, 2027)):
            args = predict.call_args_list[index].args
            self.assertIs(args[1], self.models[f"capacity_{seed}"])
            self.assertIs(args[2], self.models[f"teacher_{seed}"])
            self.assertNotIn("y", args[0])
            self.assertNotIn("timestamps", args[0])

    def test_mixed_teachers_and_changed_normalization_are_rejected(self):
        for kind in ("seed", "normalizer", "state"):
            models = copy.deepcopy(self.models)
            if kind == "seed":
                models["teacher_2026"]["seed"] = 2027
            elif kind == "normalizer":
                models["capacity_2026"]["median"][0] = 1
            else:
                models["teacher_2026"]["state"]["weight"][0] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                baseline.validate_models(models, self.manifest["architecture"])

    def test_missing_teacher_and_non_equal_weights_are_rejected(self):
        missing = dict(self.models)
        missing.pop("teacher_2027")
        with self.assertRaises(ValueError):
            baseline.validate_models(missing, self.manifest["architecture"])
        architecture = copy.deepcopy(self.manifest["architecture"])
        architecture["members"]["capacity_2026"] = .4
        with self.assertRaises(ValueError):
            baseline.validate_models(self.models, architecture)

    def test_invalid_categorical_member_is_rejected(self):
        with patch.object(baseline.live_gap_experiment, "feature_view", return_value={"joint": self.record}), \
             patch.object(baseline.capacity, "predict", return_value=np.full((4, 3), np.nan)), \
             self.assertRaisesRegex(ValueError, "probabilities"):
            baseline.member_probabilities(self.inputs, self.models, self.manifest)


class SoccerNetPromotionEvidenceContracts(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "results"
        self.report = json.loads((root / "soccernet_capacity_final_report.json").read_text())
        self.control = json.loads((root / "soccernet_capacity_control_report.json").read_text())
        self.decision = json.loads((root / "soccernet_capacity_final_decision.json").read_text())

    def test_authorized_exception_preserves_failed_historical_screen(self):
        evidence = promotion.selection_evidence(self.report, self.control, self.decision)
        self.assertFalse(evidence["historical_improvement_screen_passes"])
        self.assertFalse(evidence["historical_adopt_model"])
        self.assertTrue(evidence["exception_is_post_hoc"])
        self.assertGreater(evidence["clean_and_feed_like_spell_deltas"]["clean"], 0)
        self.assertLess(evidence["clean_and_feed_like_spell_deltas"]["clean"], .01)
        self.assertTrue(all(evidence["provider_clean_guards"].values()))

    def test_failed_provider_guard_cannot_be_hidden_by_positive_mean(self):
        self.report["summaries"]["w96"]["clean"]["per_provider"]["sportec"]["d_f1"] = 0
        with self.assertRaisesRegex(ValueError, "guards"):
            promotion.selection_evidence(self.report, self.control, self.decision)

    def test_failed_old_only_attribution_control_blocks_promotion(self):
        self.control["summaries"]["old_only"]["clean"]["equal_provider_mean"]["spell_f1"] = 1
        with self.assertRaisesRegex(ValueError, "old-only"):
            promotion.selection_evidence(self.report, self.control, self.decision)

    def test_rewriting_historical_screen_is_rejected(self):
        self.decision["improvement_screen_passes"] = True
        with self.assertRaisesRegex(ValueError, "Historical"):
            promotion.selection_evidence(self.report, self.control, self.decision)


class SoccerNetPromotionRecoveryContracts(unittest.TestCase):
    def test_interrupted_manifest_policy_transaction_resumes_without_replacing_history(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(promotion, "ROOT", root))
            names = ["STAGED", "ACTIVE", "POLICY", "PREVIOUS", "PREVIOUS_POLICY", "RECEIPT", "CSV", "TRANSACTION"]
            for name in names:
                stack.enter_context(patch.object(promotion, name, root / name.lower()))
            promotion.ACTIVE.write_bytes(b'{"old": true}\n')
            prior_policy = dict(stress_guardrail=dict(reference_recipe="previous"))
            promotion.POLICY.write_bytes(promotion.encode(prior_policy))
            promotion.PREVIOUS.write_bytes(promotion.ACTIVE.read_bytes())
            promotion.PREVIOUS_POLICY.write_bytes(promotion.POLICY.read_bytes())
            candidate = dict(promoted_utc="2026-10-30T00:00:00+00:00", status="development",
                             previous_manifest_sha256=promotion.sha256(promotion.PREVIOUS),
                             previous_policy_sha256=promotion.sha256(promotion.PREVIOUS_POLICY),
                             models_sha256={}, source_sha256={})
            promotion.STAGED.write_bytes(promotion.encode(candidate))
            policy = promotion.updated_policy(prior_policy, candidate)
            promotion.CSV.write_bytes(b'candidate')
            promotion.CSV.with_suffix('.json').write_bytes(b'validation')
            replay = root / 'replay.json'
            replay.write_bytes(b'replay')
            receipt = dict(baseline_manifest_sha256=promotion.sha256(promotion.STAGED),
                           policy_sha256=hashlib.sha256(promotion.encode(policy)).hexdigest(),
                           probability_replay_receipt_sha256=promotion.sha256(replay),
                           candidate_csv_sha256=promotion.sha256(promotion.CSV),
                           validation_receipt_sha256=promotion.sha256(promotion.CSV.with_suffix('.json')))
            promotion.TRANSACTION.write_bytes(promotion.encode(dict(manifest=candidate, policy=policy, receipt=receipt)))
            stack.enter_context(patch.object(promotion, "verified_inputs"))
            stack.enter_context(patch.object(promotion, "verify_replay", return_value=replay))
            replace = promotion.atomic_replace
            def fail_after_manifest(path, content):
                if path == promotion.POLICY:
                    raise OSError('simulated interruption')
                replace(path, content)
            with patch.object(promotion, "atomic_replace", side_effect=fail_after_manifest), \
                 self.assertRaises(OSError):
                promotion.finish_transaction()
            self.assertEqual(promotion.ACTIVE.read_bytes(), promotion.STAGED.read_bytes())
            self.assertEqual(promotion.PREVIOUS.read_bytes(), b'{"old": true}\n')
            self.assertFalse(promotion.RECEIPT.exists())
            self.assertEqual(promotion.promote(), receipt)
            self.assertEqual(promotion.read(promotion.POLICY), policy)
            self.assertEqual(promotion.read(promotion.RECEIPT), receipt)

    def test_replay_cannot_claim_success_for_another_input_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, staged = root / 'inputs.npz', root / 'manifest.json'
            inputs.write_bytes(b'inputs')
            staged.write_bytes(b'manifest')
            probability = np.tile([.4, .4, .2], (4, 1))
            np.savez(root / 'competition_probability.npz', probability=probability)
            candidate = dict(models_sha256={'one.joblib': 'sealed'})
            receipt = dict(package_version=promotion.PACKAGE,
                           baseline_manifest_sha256=promotion.sha256(staged),
                           input_sha256=promotion.sha256(inputs), models_sha256=candidate['models_sha256'],
                           probability_exact=True, decoded_prediction_exact=True, csv_exact=True,
                           candidate_csv_sha256=promotion.EXPECTED_CSV, rows=199319,
                           probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                           verifier_source_sha256=promotion.EXPECTED_REPLAY_VERIFIER,
                           organizer_reference_accessed=False, organizer_scored=False, model_training_performed=False)
            with patch.multiple(promotion, DIRECTORY=root, OUT=root, INPUTS=inputs, STAGED=staged,
                                EXPECTED_PROBABILITY=receipt['probability_sha256']):
                (root / 'probability_replay.json').write_bytes(promotion.encode(receipt))
                promotion.verify_replay(candidate)
                for key, bad in [('input_sha256', 'stale'), ('models_sha256', {}),
                                 ('probability_sha256', 'stale'), ('rows', 1)]:
                    wrong = dict(receipt, **{key: bad})
                    (root / 'probability_replay.json').write_bytes(promotion.encode(wrong))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'bound'):
                        promotion.verify_replay(candidate)


if __name__ == "__main__":
    unittest.main()
