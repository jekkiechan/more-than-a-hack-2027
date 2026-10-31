"""End-to-end synthetic optimization and frozen-teacher contracts."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments import soccernet_capacity_model as cap, soccernet_training_ideas as ideas
from experiments.tests.test_soccernet_capacity_model import teacher, full_native_record, record


def fixture():
    import torch
    original = teacher(seed=2026)
    torch.manual_seed(2026)
    net = cap.network(original["input_features"], 96)
    cap._copy_teacher_function(original, net)
    with torch.no_grad():
        net.output.bias.copy_(torch.tensor([2.0, -2.0]))
    initial = dict(kind="soccernet_capacity", hidden=96, step=2048,
        source=dict(checkpoint_sha256=cap._checkpoint_sha(original), seed=2026),
        state={k: v.detach().numpy().copy() for k, v in net.state_dict().items()},
        feature_names=original["feature_names"], median=original["median"].copy(),
        scale=original["scale"].copy(), input_features=original["input_features"],
        training=dict(seed=2026, input_dropout=.1), continuation={"stale": True})
    old = [full_native_record(source="sportec", match="old-a")]
    new = record(source="soccernet", match="soccernet_3812", n=70, anchors=(5, 8, 12, 55))
    new.update(segment_id="3812_p1_0-419", anchor_frame_mapping=[
        dict(status="mapped", sampled_index=i, team="home" if new["y"][i] == 0 else "away",
             sample_id=f"event-{i}", action=action)
        for i, action in zip((5, 8, 12, 55), ("PASS", "PASS", "SHOT", "CROSS"))])
    new["gap_x"] = new["x"].copy()
    new["gap_x"][15:30] = np.nan
    new["consistency_mask"] = np.zeros(len(new["x"]), bool)
    new["consistency_mask"][15:30] = True
    new["gap_provenance"] = dict(recipe=ideas.gap.RECIPE_VERSION)
    plan = dict(fold="leave_out_metrica", train=[dict(source="sportec", match="old-a")],
                selection=[dict(source="metrica", match="heldout")])
    return initial, original, old, [new], plan


class TrainingIdeasContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_all_arms_optimize_and_preserve_teacher_inference_and_sampling(self):
        initial, original, old, new, plan = fixture()
        original_hash = cap._checkpoint_sha(original)
        initial_hash = cap._checkpoint_sha(initial)
        draws = []
        for arm in ideas.ARMS:
            with self.subTest(arm=arm):
                got = ideas.train(initial, original, old, new, plan=plan, fixtures=["3812"],
                                  arm=arm, seed=2026, updates=2, batch_size=4)
                self.assertNotEqual(cap._checkpoint_sha(got), initial_hash)
                self.assertNotIn("continuation", got)
                np.testing.assert_array_equal(got["median"], initial["median"])
                np.testing.assert_array_equal(got["scale"], initial["scale"])
                self.assertEqual(set(got["state"]), set(initial["state"]))
                followup = got["training"]["followup"]
                draws.append(followup["supervised_draws_sha256"])
                self.assertTrue(all(np.isfinite(h["total"]) for h in followup["history"]))
                # Teacher D remains exact and inference ignores all target fields.
                feature = {k: new[0][k] for k in ("x", "names", "period", "timestamps", "fps")}
                reference = np.tile(np.array([.3, .5, .2], np.float32), (len(feature["x"]), 1))
                probability = cap.predict(feature, got, original, reference)
                poisoned = dict(feature, y=object(), full_y=object(), anchor_frame_mapping=object())
                np.testing.assert_array_equal(cap.predict(poisoned, got, original, reference), probability)
                np.testing.assert_array_equal(probability[:, 2], reference[:, 2])
                np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
                if arm in ("context_reconstruction", "gap_consistency"):
                    self.assertGreater(followup["history"][-1]["auxiliary"], 0)
        self.assertEqual(len(set(draws)), 1)
        self.assertEqual(cap._checkpoint_sha(original), original_hash)
        self.assertEqual(cap._checkpoint_sha(initial), initial_hash)

    def test_deterministic_optimizer_and_file_round_trip(self):
        initial, original, old, new, plan = fixture()
        kwargs = dict(plan=plan, fixtures=["3812"], arm="context_reconstruction",
                      seed=2026, updates=2, batch_size=4)
        a = ideas.train(initial, original, old, new, **kwargs)
        b = ideas.train(initial, original, old, new, **kwargs)
        self.assertEqual(cap._checkpoint_sha(a), cap._checkpoint_sha(b))
        self.assertEqual(a["training"]["followup"], b["training"]["followup"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(a, path)
            self.assertEqual(cap._checkpoint_sha(a), cap._checkpoint_sha(joblib.load(path)))

    def test_holdout_unknown_teacher_and_gap_inputs_fail_closed(self):
        initial, original, old, new, plan = fixture()
        kwargs = dict(plan=plan, fixtures=["3812"], arm="control", seed=2026, updates=1, batch_size=2)
        bad = copy.deepcopy(old)
        bad[0]["source"] = "metrica"
        with self.assertRaisesRegex(ValueError, "whole-match TRAIN"):
            ideas.train(initial, original, bad, new, **kwargs)
        with self.assertRaisesRegex(ValueError, "outside TRAIN"):
            ideas.train(initial, original, old, new, **dict(kwargs, fixtures=["unknown"]))
        bad = copy.deepcopy(initial)
        bad["source"]["checkpoint_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "exact frozen teacher"):
            ideas.train(bad, original, old, new, **kwargs)
        bad = copy.deepcopy(new)
        bad[0].pop("gap_x")
        with self.assertRaisesRegex(ValueError, "attach_consistency_view"):
            ideas.train(initial, original, old, bad, **dict(kwargs, arm="gap_consistency"))
        bad = copy.deepcopy(new)
        bad[0]["y"][0] = 2
        with self.assertRaisesRegex(ValueError, "never D"):
            ideas.train(initial, original, old, bad, **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ideas.train(initial, original, old, new + new, **kwargs)
        bad = copy.deepcopy(new)
        bad[0]["y"][1] = 0
        with self.assertRaisesRegex(ValueError, "unmatched known targets"):
            ideas.train(initial, original, old, bad, **kwargs)

    def test_cli_seals_reject_tampering_and_preserve_completed_fit(self):
        initial, original, old, new, plan = fixture()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            bundle_path, declaration, output = base / "bundle.joblib", base / "decl.json", base / "model.joblib"
            bundle = dict(initial=initial, teacher=original, old_records=old, gar_records=new,
                          fold=plan["fold"], seed=2026, fixtures=["3812"], gar_split="train")
            joblib.dump(bundle, bundle_path)
            # Synthetic files cannot match real historical checkpoint hashes;
            # exercise the actual seal and fit path after isolating that check.
            with patch.object(ideas, "TRAIN_FIXTURES", ("3812",)), \
                 patch.object(ideas.prep, "split_plan", return_value=plan), \
                 patch.object(ideas, "verify_checkpoint_provenance", return_value={}), \
                 patch.dict(ideas.CONFIG, updates=1, batch_size=2):
                ideas.freeze(bundle_path, declaration)
                original_declaration = declaration.read_bytes()
                altered = json.loads(original_declaration)
                altered["plan"]["train"] = []
                declaration.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "declaration bytes"):
                    ideas.fit(declaration, "control", output)
                declaration.write_bytes(original_declaration)
                ideas.fit(declaration, "control", output)
                completed = ideas.prep.sha256(output)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    ideas.fit(declaration, "control", output)
                self.assertEqual(completed, ideas.prep.sha256(output))
                with bundle_path.open("ab") as stream:
                    stream.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    ideas.fit(declaration, "control", base / "another.joblib")

    def test_historical_teacher_provenance_rejects_wrong_fold_checkpoint(self):
        initial, original, old, new, plan = fixture()
        with tempfile.TemporaryDirectory() as folder:
            teacher_path = Path(folder) / "teacher.joblib"
            joblib.dump(original, teacher_path)
            bundle = dict(fold="leave_out_metrica", seed=2026,
                          teacher_path=str(teacher_path), initial_path=str(teacher_path))
            with self.assertRaisesRegex(ValueError, "full-provider provenance"):
                ideas.verify_checkpoint_provenance(bundle, plan)


if __name__ == "__main__":
    unittest.main()
