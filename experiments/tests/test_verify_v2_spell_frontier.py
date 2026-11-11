"""Focused, label-safe tests for the frontier independent auditor."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np

from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_frontier_eval as frontier
from scripts import verify_v2_spell_frontier as verifier


class FrontierAuditHelperTests(unittest.TestCase):
    def test_typed_nested_values_compare_exactly(self):
        verifier._close({"comparison": "strictly_greater", "passed": True},
                        {"comparison": "strictly_greater", "passed": True}, "test")
        with self.assertRaises(AssertionError):
            verifier._close({"comparison": "strictly_greater"},
                            {"comparison": "greater_or_equal"}, "test")

    def test_mix_is_independent_and_does_not_mutate_arrays(self):
        rng = np.random.default_rng(11)
        values = {name: rng.random((8, 3)) for name in ("ce", "calibrated", "adopted")}
        values = {name: p / p.sum(axis=1, keepdims=True) for name, p in values.items()}
        before = {name: p.copy() for name, p in values.items()}
        recipe = {"ce_weight": .5, "calibrated_weight": .25, "adopted_weight": .25,
                  "pooling": "arithmetic"}
        got = verifier._mix(values, recipe)
        expected = balanced.recipe_probability(values, .5, .25, "arithmetic")
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)
        for name in values:
            np.testing.assert_array_equal(values[name], before[name])

    def test_stage1024_loader_checks_real_n_by_three_arrays_and_2_5_phase_means(self):
        with tempfile.TemporaryDirectory(dir=frontier.ROOT) as temp:
            root = Path(temp)
            cell = "Metrica|fixture|feed_like_v2|1"
            for fps, phase_count in ((10.0, 2), (25.0, 5)):
                rng = np.random.default_rng(int(fps))
                shape = (7, 3)
                phases = {seed: [] for seed in frontier.SEEDS}
                for seed in frontier.SEEDS:
                    for _phase in range(phase_count):
                        probs = rng.random(shape)
                        phases[seed].append(probs / probs.sum(axis=1, keepdims=True))
                seed_ce = {seed: np.mean(np.stack(phases[seed]), axis=0) for seed in frontier.SEEDS}
                paired = np.stack([np.mean(np.stack([phases[seed][phase] for seed in frontier.SEEDS]), axis=0)
                                   for phase in range(phase_count)])
                pooled = np.mean(paired, axis=0)
                other = {name: rng.random(shape) for name in ("adopted", "calibrated")}
                other = {name: p / p.sum(axis=1, keepdims=True) for name, p in other.items()}
                identity = {"study_id": "control-study", "cell": cell, "cache": str(fps)}
                control_path = root / f"control_{int(fps)}.npz"
                control_arrays = {
                    "prob__adopted": other["adopted"], "prob__calibrated": other["calibrated"],
                    "labels": np.zeros(7, dtype=np.int64), "period": np.ones(7, dtype=np.int64),
                    "timestamps": np.arange(7, dtype=np.float64),
                    "model_period": np.ones(7, dtype=np.int64), "fps": np.asarray(fps)}
                for seed in frontier.SEEDS:
                    control_arrays[f"prob__cand_seed{seed}"] = other["adopted"]
                    control_arrays[f"prob__adop_seed{seed}"] = other["calibrated"]
                control_arrays.update({f"identity__{name}": np.asarray(str(value))
                                       for name, value in identity.items()})
                np.savez_compressed(control_path, **control_arrays)

                r2_decl = root / f"balanced_r2_decl_{int(fps)}.json"
                r2_decl.write_text("{}\n")
                pred_path = root / f"stage_{int(fps)}.npz"
                pred_identity = {"study_id": "v2_spell_balanced_v1",
                    "declaration_sha256": verifier.sha(r2_decl), "cell": cell,
                    "phase_count": phase_count}
                pred_arrays = {"pooled_ce": pooled}
                pred_arrays.update({f"seed_ce_{seed}": seed_ce[seed] for seed in frontier.SEEDS})
                pred_arrays.update({f"phase_ce_seed{seed}_{phase}": phases[seed][phase]
                                    for seed in frontier.SEEDS for phase in range(phase_count)})
                pred_arrays.update({f"identity__{name}": np.asarray(str(value))
                                    for name, value in pred_identity.items()})
                np.savez_compressed(pred_path, **pred_arrays)
                c_fact = {"path": str(control_path.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(control_path), "identity": identity}
                p_fact = {"path": str(pred_path.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(pred_path), "identity": pred_identity}
                declaration = {"inputs": {"control": {cell: c_fact},
                    "stage1024_probability_cache": {cell: p_fact}}}
                with patch.object(frontier.balanced_r2, "DECLARATION", r2_decl):
                    loaded = verifier._load_components(cell, declaration)
                self.assertEqual(loaded["ce"].shape, shape)
                np.testing.assert_allclose(loaded["ce"], pooled, rtol=0, atol=1e-12)

    def test_equal_provider_reducer_is_independent_of_recipe_order(self):
        rows = {}
        for provider, matches in (("Metrica", ("m1", "m2")),
                                  ("SkillCorner", ("s1", "s2")),
                                  ("Sportec", ("x1", "x2"))):
            for match in matches:
                for view, repetitions in balanced.VIEWS:
                    for rep in repetitions:
                        key = f"{provider}|{match}|{view}|{rep}"
                        rows[key] = {"r2": {metric: .5 for metric in balanced.METRICS},
                                     "r1": {metric: .25 for metric in balanced.METRICS}}
        result = verifier._aggregate(rows, ["r2", "r1"])
        self.assertEqual(list(result), ["r2", "r1"])
        self.assertEqual(result["r2"]["feed_like_v2"]["spell_f1"]["equal_provider"], .5)


class FrontierAuditFullFixtureTests(unittest.TestCase):
    def test_both_audit_entrypoints_cover_35_cells_30_recipes_and_nested_pipelines(self):
        from experiments import v2_spell_frontier_round2 as round2
        cells = sorted(frontier.expected_cells())
        self.assertEqual(len(cells), 35)
        n = 8
        labels = np.zeros(n, dtype=np.int64)
        period = np.ones(n, dtype=np.int64)
        timestamps = np.arange(n, dtype=np.float64)
        base_prob = {
            "adopted": np.tile(np.array([[.4, .5, .1]], dtype=np.float64), (n, 1)),
            "calibrated": np.tile(np.array([[.2, .7, .1]], dtype=np.float64), (n, 1)),
            "ce": np.tile(np.array([[.9, .05, .05]], dtype=np.float64), (n, 1)),
        }

        def decode(prob, adopted_decoded, *_args):
            label = "H" if float(np.mean(prob[:, 0])) > .62 else "A"
            return np.full(n, label, dtype="U1")

        def fake_metrics(_cell, prediction):
            label = str(np.asarray(prediction)[0])
            return {"spell_f1": .8 if label == "H" else .6,
                "macro_f1": .75, "d_f1": .5, "d_precision": .5, "d_recall": .5,
                "predicted_spell_count": 2.0, "reference_spell_count": 2,
                "short_spell_recall_iou_0_5_lt3s": .6,
                "spell_by_iou": {"0.3": .7, "0.5": .6},
                "comparison": "strictly_greater", "verified": True, "optional": None}

        with tempfile.TemporaryDirectory(dir=frontier.ROOT) as temp:
            root = Path(temp)
            r1_decl_path, r1_result_path = root / "r1_decl.json", root / "r1_result.json"
            replay_path, control_replay_path = root / "leader_replay.json", root / "control_replay.json"
            balanced_r2_decl = root / "balanced_r2_decl.json"
            balanced_r2_decl.write_text("fixture balanced R2 declaration\n")
            controls, stage_cache, control_facts = {}, {}, {}
            stage_facts = {}
            loaded_controls = {}
            native_phase_count = {}
            for idx, cell in enumerate(cells):
                provider = cell.split("|", 1)[0]
                fps = 10. if provider == "SkillCorner" else 25.
                phases_n = 2 if fps == 10. else 5
                native_phase_count[cell] = phases_n
                identity = {"study_id": "frontier_control_fixture", "cell": cell, "slot": str(idx)}
                cpath = root / f"control_{idx}.npz"
                c_arrays = {"prob__adopted": base_prob["adopted"],
                    "prob__calibrated": base_prob["calibrated"],
                    "labels": labels, "period": period, "timestamps": timestamps,
                    "model_period": period, "fps": np.asarray(fps)}
                for seed in frontier.SEEDS:
                    c_arrays[f"prob__cand_seed{seed}"] = base_prob["adopted"]
                    c_arrays[f"prob__adop_seed{seed}"] = base_prob["calibrated"]
                c_arrays.update({f"identity__{name}": np.asarray(str(value))
                                 for name, value in identity.items()})
                np.savez_compressed(cpath, **c_arrays)
                c_fact = {"path": str(cpath.relative_to(frontier.ROOT)),
                          "sha256": verifier.sha(cpath), "identity": identity}
                controls[cell], control_facts[cell] = c_fact, c_fact

                phase_probs = {seed: [base_prob["ce"].copy() for _ in range(phases_n)]
                               for seed in frontier.SEEDS}
                pooled = np.mean([np.mean([phase_probs[seed][phase] for seed in frontier.SEEDS], axis=0)
                                  for phase in range(phases_n)], axis=0)
                seed_ce = {seed: np.mean(phase_probs[seed], axis=0) for seed in frontier.SEEDS}
                ppath = root / f"stage1024_{idx}.npz"
                pident = {"study_id": "v2_spell_balanced_v1",
                    "declaration_sha256": verifier.sha(balanced_r2_decl), "cell": cell,
                    "phase_count": phases_n}
                p_arrays = {"pooled_ce": pooled}
                p_arrays.update({f"seed_ce_{seed}": seed_ce[seed] for seed in frontier.SEEDS})
                p_arrays.update({f"phase_ce_seed{seed}_{phase}": phase_probs[seed][phase]
                                 for seed in frontier.SEEDS for phase in range(phases_n)})
                p_arrays.update({f"identity__{name}": np.asarray(str(value))
                                 for name, value in pident.items()})
                np.savez_compressed(ppath, **p_arrays)
                stage_cache[cell] = {"path": str(ppath.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(ppath), "identity": pident}
                control_facts[cell] = c_fact

                loaded_controls[cell] = {"adopted": base_prob["adopted"],
                    "calibrated": base_prob["calibrated"],
                    "seed_adopted": {seed: base_prob["adopted"] for seed in frontier.SEEDS},
                    "seed_calibrated": {seed: base_prob["calibrated"] for seed in frontier.SEEDS},
                    "seed_ce": seed_ce, "ce": pooled,
                    "labels": labels, "period": period, "timestamps": timestamps,
                    "model_period": period, "fps": fps,
                    **{f"cand_seed{seed}": base_prob["adopted"] for seed in frontier.SEEDS},
                    **{f"adop_seed{seed}": base_prob["calibrated"] for seed in frontier.SEEDS}}
            r1_decl = {"source_sha256": {"fixture": "bound"},
                "recipe_grid": frontier.RECIPES, "recipe_count": len(frontier.RECIPES),
                "inputs": {"control": controls, "stage1024_probability_cache": stage_cache}}
            r1_decl_path.write_text(json.dumps(r1_decl))
            r1_decl_sha = verifier.sha(r1_decl_path)

            baseline_rows = {cell: {**fake_metrics(None, np.full(n, "A")),
                                    "spell_f1": .5, "macro_f1": .75,
                                    "short_spell_recall_iou_0_5_lt3s": .6,
                                    "d_f1": .5, "d_precision": .5, "d_recall": .5}
                             for cell in cells}
            adopted_aggregate = balanced.aggregate_cell_metrics(baseline_rows)
            references = {cell: 2 for cell in cells}
            control_replay_path.write_text(json.dumps({"adopted": adopted_aggregate,
                "reference_spell_counts": references,
                "cell_counts": balanced._cell_counts(baseline_rows)}))
            recipe_rows = {r["recipe_id"]: r for r in frontier.RECIPES}
            cell_rows, saved_predictions, leader_rows = {}, {}, {rid: {} for rid in frontier.LEADER_IDS}
            seed_shortlist = None
            for cell in cells:
                control = loaded_controls[cell]
                adopted_decoded = np.full(n, "A", dtype="U1")
                cell_rows[cell], saved_predictions[cell] = {}, {}
                for recipe in frontier.RECIPES:
                    rid = recipe["recipe_id"]
                    pred = decode(verifier._mix(control, recipe), adopted_decoded)
                    row = fake_metrics(None, pred)
                    cell_rows[cell][rid] = row
                    saved_predictions[cell][rid] = np.asarray([{"H": 0, "A": 1, "D": 2, "X": 3}[x]
                                                                 for x in pred], dtype=np.int8)
                    if rid in leader_rows:
                        leader_rows[rid][cell] = row
            aggregate = verifier._aggregate(cell_rows, list(frontier.ARM_IDS))
            gates = {}
            for rid in frontier.ARM_IDS:
                candidate = {v: {m: aggregate[rid][v][m]["equal_provider"] for m in balanced.METRICS}
                             for v, _ in balanced.VIEWS}
                target = {v: {m: adopted_aggregate[v][m]["equal_provider"] for m in balanced.METRICS}
                          for v, _ in balanced.VIEWS}
                gate = balanced.eligibility(candidate, target)
                gate["d_metrics_unchanged"] = True
                gate["reference_spell_counts_invariant"] = True
                gate["beats_current_best"] = {"actual": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                    "minimum_strictly_greater": frontier.CURRENT_BEST_V2,
                    "passed": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"] > frontier.CURRENT_BEST_V2}
                gates[rid] = gate
            seed_shortlist = frontier._seed_shortlist(gates, aggregate)
            ranking = sorted(frontier.ARM_IDS, key=lambda rid: (
                -aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                -aggregate[rid]["feed_like_v2"]["macro_f1"]["equal_provider"], rid))
            self.assertNotEqual([rid for rid in ranking if gates[rid]["eligible"]][0], frontier.ARM_IDS[0])
            seed_cells = {cell: {} for cell in cells}
            for cell in cells:
                for rid in seed_shortlist:
                    for seed in frontier.SEEDS:
                        pred = decode(verifier._mix(loaded_controls[cell], recipe_rows[rid], seed=seed),
                                      np.full(n, "A", dtype="U1"))
                        seed_cells[cell][f"{rid}|{seed}"] = fake_metrics(None, pred)
            seed_aggregate = verifier._seed_aggregate(seed_cells, seed_shortlist)

            pred_inventory, seed_inventory = {}, {}
            for idx, cell in enumerate(cells):
                pooled_path = root / f"r1pred_{idx}.npz"
                identity = {"study_id": frontier.STUDY, "declaration_sha256": r1_decl_sha,
                    "cell": cell, "recipe_order": list(frontier.ARM_IDS)}
                arrays = {f"pred__{rid}": saved_predictions[cell][rid] for rid in frontier.ARM_IDS}
                arrays.update({f"identity__{name}": np.asarray(json.dumps(value) if isinstance(value, list) else str(value))
                               for name, value in identity.items()})
                np.savez_compressed(pooled_path, **arrays)
                pred_inventory[cell] = {"path": str(pooled_path.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(pooled_path), "identity": identity}
                seed_path = root / f"r1seed_{idx}.npz"
                seed_identity = {"study_id": frontier.STUDY, "declaration_sha256": r1_decl_sha,
                    "cell": cell, "recipe_order": seed_shortlist, "seed_order": list(frontier.SEEDS)}
                arrays = {f"pred__{rid}|{seed}": np.full(n, 1, dtype=np.int8)
                          for rid in seed_shortlist for seed in frontier.SEEDS}
                for rid in seed_shortlist:
                    for seed in frontier.SEEDS:
                        control = loaded_controls[cell]
                        pred = decode(verifier._mix(control, recipe_rows[rid], seed=seed),
                                      np.full(n, "A", dtype="U1"))
                        arrays[f"pred__{rid}|{seed}"] = np.asarray([{"H": 0, "A": 1}[x] for x in pred], dtype=np.int8)
                arrays.update({f"identity__{name}": np.asarray(json.dumps(value) if isinstance(value, list) else str(value))
                               for name, value in seed_identity.items()})
                np.savez_compressed(seed_path, **arrays)
                seed_inventory[cell] = {"path": str(seed_path.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(seed_path), "identity": seed_identity}

            replay_aggregate, replay_checks = {}, {}
            balanced_r2_result_path = root / "balanced_r2_result.json"
            for rid in frontier.LEADER_IDS:
                replay_aggregate[rid] = balanced.aggregate_cell_metrics(leader_rows[rid])
                for view, _ in balanced.VIEWS:
                    for metric in balanced.METRICS:
                        actual = replay_aggregate[rid][view][metric]["equal_provider"]
                        replay_checks[f"{rid}/{view}/{metric}"] = {"actual": actual,
                            "expected": actual, "absolute_error": 0., "passed": True}
            balanced_r2_result_path.write_text(json.dumps({"aggregate": {
                rid: replay_aggregate[rid] for rid in frontier.LEADER_IDS}}))
            replay_path.write_text(json.dumps({"passed": True, "candidate_scoring": False,
                "aggregate": replay_aggregate, "checks": replay_checks}))
            r1_result = {"study_id": frontier.STUDY, "round": 1,
                "declaration_sha256": r1_decl_sha, "replay_sha256": verifier.sha(replay_path),
                "recipes": frontier.RECIPES, "recipe_ids": list(frontier.ARM_IDS),
                "aggregate": aggregate, "eligibility": gates,
                "eligible_ranking": [rid for rid in ranking if gates[rid]["eligible"]],
                "seed_shortlist": seed_shortlist, "seed_diagnostics": seed_aggregate,
                "seed_cells": seed_cells, "cell_metrics": cell_rows,
                "prediction_cache": pred_inventory, "seed_prediction_cache": seed_inventory,
                "cohort": {"reference_spell_counts": references},
                "organizer_reference_access": False, "public_scoring": False}
            r1_result_path.write_text(json.dumps(r1_result))

            # R2 keeps the actual nested pipeline schema and mixes each saved native phase pool.
            r2_decl_path = root / "r2_decl.json"
            r2_result_path = root / "r2_result.json"
            r2_pipelines = [round2._pipeline(recipe_rows[rid], arm)
                            for rid in seed_shortlist for arm in round2.ARMS]
            r2_ids = [row["pipeline_id"] for row in r2_pipelines]
            r2_decl = {"study_id": frontier.STUDY, "source_sha256": {"fixture": "bound"},
                "pipelines": r2_pipelines, "pipeline_ids": r2_ids,
                "controls": {cell: {"prepared_path": "fixture", "prepared_sha256": "fixture"}
                             for cell in cells}}
            r2_decl_path.write_text(json.dumps(r2_decl))
            r2_decl_sha = verifier.sha(r2_decl_path)
            r2_rows, r2_seed_cells, r2_cache, r2_phase_counts = {}, {}, {}, {}
            for idx, cell in enumerate(cells):
                control = loaded_controls[cell]
                phases_n = native_phase_count[cell]
                r2_phase_counts[cell] = phases_n
                probs, preds, seeds = {}, {}, {}
                for arm in round2.ARMS:
                    seed_phase = {}
                    for seed in round2.SEEDS:
                        arrays_phase = []
                        for phase in range(phases_n):
                            val = base_prob["ce"].copy()
                            val[:, 0] -= phase * .001
                            val[:, 1] += phase * .001
                            val[:, 2] += 1e-8
                            arrays_phase.append(val)
                            probs[f"phase__{arm}__{seed}__{phase}"] = val
                        seed_phase[seed] = arrays_phase
                    pooled = np.mean([np.mean([seed_phase[seed][phase] for seed in round2.SEEDS], axis=0)
                                      for phase in range(phases_n)], axis=0)
                    per_seed = {seed: np.mean(seed_phase[seed], axis=0) for seed in round2.SEEDS}
                    for pipeline in r2_pipelines:
                        pid = pipeline["pipeline_id"]
                        if pipeline["training_arm"] != arm:
                            continue
                        recipe = pipeline["recipe"]
                        probs[f"pooled__{pid}"] = pooled
                        pred = decode(verifier._mix({"ce": pooled, "calibrated": control["calibrated"],
                            "adopted": control["adopted"]}, recipe), np.full(n, "A", dtype="U1"))
                        preds[f"pred__{pid}"] = np.asarray([{"H": 0, "A": 1}[x] for x in pred], dtype=np.int8)
                        r2_rows.setdefault(cell, {})[pid] = fake_metrics(None, pred)
                        for seed in round2.SEEDS:
                            seed_pred = decode(verifier._mix({"ce": per_seed[seed],
                                "calibrated": control[f"adop_seed{seed}"],
                                "adopted": control[f"cand_seed{seed}"]}, recipe),
                                np.full(n, "A", dtype="U1"))
                            seeds[f"seedpred__{pid}|{seed}"] = np.asarray(
                                [{"H": 0, "A": 1}[x] for x in seed_pred], dtype=np.int8)
                            r2_seed_cells.setdefault(cell, {})[f"{pid}|{seed}"] = fake_metrics(None, seed_pred)
                ident = {"study_id": frontier.STUDY, "declaration_sha256": r2_decl_sha,
                    "cell": cell, "pipeline_order": r2_ids, "seed_order": list(round2.SEEDS)}
                arrays = {**probs, **preds, **seeds}
                arrays.update({f"identity__{name}": np.asarray(json.dumps(value) if isinstance(value, list) else str(value))
                               for name, value in ident.items()})
                cache_path = root / f"r2pred_{idx}.npz"
                np.savez_compressed(cache_path, **arrays)
                r2_cache[cell] = {"path": str(cache_path.relative_to(frontier.ROOT)),
                    "sha256": verifier.sha(cache_path), "identity": ident}
            r2_aggregate = verifier._aggregate(r2_rows, r2_ids)
            r2_seed_aggregate = verifier._seed_aggregate(r2_seed_cells, r2_ids)
            r2_baseline = control_replay_path
            gates2 = {}
            for pid in r2_ids:
                candidate = {v: {m: r2_aggregate[pid][v][m]["equal_provider"] for m in balanced.METRICS}
                             for v, _ in balanced.VIEWS}
                target = {v: {m: adopted_aggregate[v][m]["equal_provider"] for m in balanced.METRICS}
                          for v, _ in balanced.VIEWS}
                gate = balanced.eligibility(candidate, target)
                gate["d_metrics_unchanged"] = True
                gate["reference_spell_counts_invariant"] = True
                gate["beats_current_best"] = {"actual": r2_aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                    "minimum_strictly_greater": frontier.CURRENT_BEST_V2,
                    "passed": True}
                gate["eligible"] = gate["eligible"] and True
                gates2[pid] = gate
            ranking2 = sorted(r2_ids, key=lambda pid: (
                -r2_aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                -r2_aggregate[pid]["feed_like_v2"]["macro_f1"]["equal_provider"], pid))
            r2_result = {"study_id": frontier.STUDY, "round": 2, "training_stage": 1536,
                "declaration_sha256": r2_decl_sha, "pipelines": r2_pipelines,
                "pipeline_ids": r2_ids, "aggregate": r2_aggregate,
                "eligibility": gates2, "cell_metrics": r2_rows,
                "eligible_ranking": [pid for pid in ranking2 if gates2[pid]["eligible"]],
                "seed_cells": r2_seed_cells, "seed_diagnostics": r2_seed_aggregate,
                "prediction_cache": r2_cache,
                "cohort": {"cell_count": 35, "reference_spell_counts": references}}
            r2_result_path.write_text(json.dumps(r2_result))

            with ExitStack() as stack:
                stack.enter_context(patch.object(frontier, "DECLARATION", r1_decl_path))
                stack.enter_context(patch.object(frontier, "RESULT", r1_result_path))
                stack.enter_context(patch.object(frontier, "REPLAY", replay_path))
                stack.enter_context(patch.object(frontier, "CONTROL_REPLAY", control_replay_path))
                stack.enter_context(patch.object(frontier, "CONTROL_ROUND2", balanced_r2_result_path))
                stack.enter_context(patch.object(frontier, "_source_hashes", lambda: {"fixture": "bound"}))
                stack.enter_context(patch.object(frontier, "balanced_r2", frontier.balanced_r2))
                stack.enter_context(patch.object(frontier.balanced_r2, "DECLARATION", balanced_r2_decl))
                stack.enter_context(patch.object(round2, "DECLARATION", r2_decl_path))
                stack.enter_context(patch.object(round2, "RESULT", r2_result_path))
                stack.enter_context(patch.object(round2, "_source_hashes", lambda: {"fixture": "bound"}))
                stack.enter_context(patch.object(round2, "_cell_inputs", lambda key, fact:
                    ({"phases": [object()] * native_phase_count[key], "view": {}}, loaded_controls[key])))
                stack.enter_context(patch.object(verifier, "_check_audit_plan", lambda _module: {}))
                stack.enter_context(patch.object(balanced.runtime.frozen.audit, "decode",
                    lambda *args: np.full(n, "A", dtype="U1")))
                stack.enter_context(patch.object(balanced, "decode_fixed_d", decode))
                stack.enter_context(patch.object(balanced.controls.audit, "score", fake_metrics))
                audit1 = verifier.verify_round1()
                audit2 = verifier.verify_round2()
            self.assertTrue(audit1["passed"])
            self.assertEqual((audit1["cell_count"], audit1["recipe_count"]), (35, 30))
            self.assertNotEqual(r1_result["eligible_ranking"][0], frontier.ARM_IDS[0])
            self.assertTrue(audit2["passed"])
            self.assertEqual((audit2["cell_count"], audit2["recipe_count"]), (35, len(r2_ids)))
            self.assertEqual(set(audit2["phase_counts_by_cell"].values()), {2, 5})


if __name__ == "__main__":
    unittest.main()
