"""Reference-free contracts for the final v2 spell frontier prediction."""
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

from experiments import v2_spell_frontier_public as public


class FrontierPublicContractTests(unittest.TestCase):
    def setUp(self):
        # Exercise file/hash contracts with synthetic bytes, independent of the
        # historical local-only organizer prediction and its frozen seal.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        recipe = {"recipe_id": "synthetic-contract-recipe", "ce_weight": .5,
                  "calibrated_weight": .25, "adopted_weight": .25,
                  "pooling": "arithmetic", "ha_cost": 3.0}
        csv, seal, plan = root / "old.csv", root / "seal.json", root / "plan.json"
        csv.write_bytes(b"frame,label\n0,H\n1,D\n2,A\n")
        seal.write_text(json.dumps({"selected_recipe": recipe}))
        csv_sha, seal_sha = public.sha(csv), public.sha(seal)
        plan.write_text(json.dumps({"previous_study": {
            "old_public_gate_claimed": False, "old_public_score_exists": False,
            "old_reference_free_csv_sha256": csv_sha,
            "old_prediction_seal_sha256": seal_sha}}))
        patcher = patch.multiple(public, BALANCED_CSV=csv, BALANCED_SEAL=seal,
            BALANCED_GATE=root / "gate.json", BALANCED_RESULT=root / "result.json",
            PLAN=plan, OLD_CSV_SHA256=csv_sha, OLD_SEAL_SHA256=seal_sha)
        patcher.start()
        self.addCleanup(patcher.stop)

    def selection(self, arm="base", stage=1024, recipe=None):
        if recipe is None:
            recipe = {**public._json(public.BALANCED_SEAL)["selected_recipe"]}
        return {"study_id": public.STUDY, "final_selection": True,
                "training_arm": arm, "training_stage": stage,
                "public_input_sha256": "input-hash", "recipe": recipe}

    def test_stage_and_arm_are_a_fixed_pair(self):
        public._selection_contract(self.selection("base", 1024))
        public._selection_contract(self.selection("repeat", 1536))
        public._selection_contract(self.selection("fresh", 1536))
        for arm, stage in (("base", 1536), ("repeat", 1024), ("fresh", 1024), ("other", 1536)):
            with self.subTest(arm=arm, stage=stage), self.assertRaises(ValueError):
                public._selection_contract(self.selection(arm, stage))

    def test_invalid_weights_and_missing_input_binding_fail_closed(self):
        bad = self.selection()
        bad["recipe"] = {**bad["recipe"], "ce_weight": float("nan")}
        with self.assertRaises(ValueError):
            public._selection_contract(bad)
        missing = self.selection()
        del missing["public_input_sha256"]
        with self.assertRaises(ValueError):
            public._selection_contract(missing)

    def test_reuse_requires_exact_balanced_recipe_values(self):
        recipe = public._json(public.BALANCED_SEAL)["selected_recipe"]
        self.assertTrue(public._recipe_matches_balanced(recipe))
        changed = {**recipe, "ce_weight": recipe["ce_weight"] + .001}
        self.assertFalse(public._recipe_matches_balanced(changed))

    def test_copy_reuse_keeps_old_prediction_seal_and_csv_immutable(self):
        old_csv_sha = public.sha(public.BALANCED_CSV)
        old_seal_sha = public.sha(public.BALANCED_SEAL)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "frontier_candidate.csv"
            with patch.object(public, "CSV", target):
                public._copy_balanced_prediction()
            self.assertEqual(public.sha(target), old_csv_sha)
        self.assertEqual(public.sha(public.BALANCED_CSV), old_csv_sha)
        self.assertEqual(public.sha(public.BALANCED_SEAL), old_seal_sha)

    def test_copy_reuse_refuses_to_overwrite_existing_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "existing.csv"
            target.write_bytes(b"keep-existing-prediction")
            with patch.object(public, "CSV", target), self.assertRaises(FileExistsError):
                public._copy_balanced_prediction()
            self.assertEqual(target.read_bytes(), b"keep-existing-prediction")

    def test_prior_facts_reject_changed_csv_or_seal(self):
        facts = public._prior_public_facts()
        self.assertEqual(facts["csv_sha256"], public.sha(public.BALANCED_CSV))
        for path in (public.BALANCED_CSV, public.BALANCED_SEAL):
            original = path.read_bytes()
            with self.subTest(path=path.name):
                path.write_bytes(original + b"tampered")
                with self.assertRaises(ValueError):
                    public._prior_public_facts()
                path.write_bytes(original)

    def test_prior_facts_reject_a_claimed_gate_or_existing_result(self):
        for path in (public.BALANCED_GATE, public.BALANCED_RESULT):
            with self.subTest(path=path.name):
                path.write_text('{}')
                with self.assertRaises(FileExistsError):
                    public._prior_public_facts()
                path.unlink()

    def test_frontier_pooled_seal_uses_trainer_model_and_receipt_schema(self):
        with tempfile.TemporaryDirectory(dir=public.ROOT) as temp:
            root = Path(temp)
            selected = root / "selection.json"
            selection = self.selection("fresh", 1536)
            selected.write_text(json.dumps(selection))
            models = {}
            fits = []
            paths = {}
            for seed in (2026, 2027):
                path = root / f"stage1536_{seed}.joblib"
                path.write_bytes(f"fresh-model-{seed}".encode())
                paths[seed] = path
                fact = {"path": str(path.relative_to(public.ROOT)), "sha256": public.sha(path)}
                models[str(seed)] = fact
                fits.append({"study_id": public.STUDY, "arm": "fresh", "stage": 1536,
                    "seed": seed, **fact, "declaration_sha256": "pooled-decl-hash",
                    "d_state_unchanged": True})
            pool_decl = {"study_id": public.STUDY, "training_arm": "fresh",
                "training_stage": 1536, "selection_sha256": public.sha(selected)}
            pool_seal = {"study_id": public.STUDY,
                "pooled_declaration_sha256": "pooled-decl-hash",
                "training_arm": "fresh", "training_stage": 1536,
                "models": models, "fits": fits,
                "all_selected_pooled_models_ready": True}
            with patch.object(public, "SELECTION", selected):
                public._validate_frontier_pool_seal(selection, pool_seal, pool_decl,
                                                     "pooled-decl-hash", paths)
                bad = {**pool_seal, "fits": []}
                with self.assertRaises(ValueError):
                    public._validate_frontier_pool_seal(selection, bad, pool_decl,
                                                        "pooled-decl-hash", paths)


if __name__ == "__main__":
    unittest.main()
