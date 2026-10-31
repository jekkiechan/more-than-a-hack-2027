"""Synthetic blend and exclusive-check contracts; no organizer assets are read."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from experiments import temporal_competition_check as check


class CompetitionBlendTests(unittest.TestCase):
    def test_full_vector_blend_and_team_swap(self):
        tree = np.array([[.7, .2, .1], [.1, .3, .6]])
        gru = np.array([[.1, .6, .3], [.5, .4, .1]])
        recipe = dict(tree=.5, gru=.5)
        result = check.combine(dict(tree=tree, gru=gru), recipe)
        np.testing.assert_allclose(result, [[.4, .4, .2], [.3, .35, .35]])
        np.testing.assert_array_equal(check.combine(dict(tree=tree), dict(tree=1)), tree)
        np.testing.assert_allclose(
            check.combine(dict(tree=tree[:, [1, 0, 2]], gru=gru[:, [1, 0, 2]]), recipe),
            result[:, [1, 0, 2]])

    def test_rejects_invalid_weights_vectors_and_broadcast_frame_clocks(self):
        p = np.array([[.2, .3, .5], [.1, .8, .1]])
        for recipe in ({}, dict(a=-.5, b=1.5), dict(a=float('nan')),
                       dict(a=float('inf')), dict(a=.8)):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                check.combine(dict(a=p, b=p), recipe)
        for bad in (p[:1], p[:, :2], np.full_like(p, np.nan),
                    np.array([[-.1, .5, .6], [.1, .8, .1]]), p*.5):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                check.combine(dict(a=p, b=bad), dict(a=.5, b=.5))


class CompetitionGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root/'diagnostic'
        self.out.mkdir()
        self.spec = dict(recipes=deepcopy(check.RECIPES), bootstrap_draws=1000,
                         scope='Synthetic test only')
        (self.out/'predeclared.json').write_text(json.dumps(self.spec))
        for name in check.MODELS:
            (self.out/(name+'.joblib')).write_bytes(b'fake model; never deserialized')
            (self.out/(name+'.json')).write_text('{}')
        for suffix in ('.joblib', '.json'):
            (self.out/('competition_features'+suffix)).write_bytes(b'fake cache')
        model_paths = [self.out/(name+suffix) for name in check.MODELS
                       for suffix in ('.joblib', '.json')]
        model_files = {str(p.relative_to(self.root)): check.sha256(p) for p in model_paths}
        (self.out/'checkpoints_frozen.json').write_text(json.dumps(dict(files_sha256=model_files)))
        candidates = {}
        for name, recipe in check.RECIPES.items():
            csv, npz = self.out/(name+'.csv'), self.out/(name+'.npz')
            # Gate tests inspect sealed metadata/hashes only. No real prediction
            # or reference is loaded, and score IO is replaced with synthetic data.
            csv.write_text('synthetic sealed candidate')
            npz.write_bytes(b'synthetic sealed probabilities')
            candidates[name] = dict(csv=csv.name, csv_sha256=check.sha256(csv),
                                    probability_file=npz.name, probability_file_sha256=check.sha256(npz),
                                    frames=199319, recipe=deepcopy(recipe))
        paths = model_paths + [self.out/name for name in (
            'predeclared.json', 'checkpoints_frozen.json',
            'competition_features.joblib', 'competition_features.json')]
        paths += [self.out/(name+suffix) for name in check.RECIPES for suffix in ('.csv', '.npz')]
        self.seal = dict(candidates=candidates, reference_labels_read=False,
                         full_length_csvs_validated=True,
                         files_sha256={str(p.relative_to(self.root)): check.sha256(p) for p in paths})
        self.write_seal(self.seal)
        original_verify = check.verify_files
        for mocker in (patch.object(check, 'ROOT', self.root),
                       patch.object(check, 'PUBLIC', self.root/'public_report.json'),
                       patch.object(check, 'verify_files',
                                    side_effect=lambda files: original_verify(files, root=self.root))):
            mocker.start()
            self.addCleanup(mocker.stop)

    def write_seal(self, seal):
        (self.out/'prediction_seal.json').write_text(json.dumps(seal))

    def test_valid_gate_is_exclusive(self):
        check.start_gate(self.out, self.spec, self.seal)
        marker = json.loads((self.out/'public_check_started.json').read_text())
        self.assertEqual(marker['prediction_seal_sha256'], check.sha256(self.out/'prediction_seal.json'))
        with self.assertRaises(ValueError):
            check.start_gate(self.out, self.spec, self.seal)

    def test_candidate_set_and_metadata_are_bound_to_fixed_sealed_files(self):
        changes = []
        missing = deepcopy(self.seal); missing['candidates'].pop('robust_blend'); changes.append(missing)
        extra = deepcopy(self.seal); extra['candidates']['fifth_arm'] = {}; changes.append(extra)
        for field, value in (('csv', '../unsealed.csv'), ('probability_file', 'other.npz'),
                             ('csv_sha256', 'wrong'), ('frames', 7500),
                             ('recipe', {'tree_full_127': .6, 'gru_slot_dropout_2026': .4})):
            changed = deepcopy(self.seal)
            changed['candidates']['robust_blend'][field] = value
            changes.append(changed)
        changed = deepcopy(self.seal); changed['files_sha256'].pop(next(iter(changed['files_sha256'])))
        changes.append(changed)
        changed = deepcopy(self.seal); changed['full_length_csvs_validated'] = False; changes.append(changed)
        for i, changed in enumerate(changes):
            with self.subTest(case=i), self.assertRaises(ValueError):
                check.start_gate(self.out, self.spec, changed)
            self.assertFalse((self.out/'public_check_started.json').exists())

    def test_changed_artifact_is_rejected_before_reference_io_or_gate(self):
        (self.out/'robust_blend.csv').write_text('changed after sealing')
        with patch.object(check, 'check_spec', return_value=self.spec), \
             patch.object(pd, 'read_csv') as reader, patch.object(pm, 'evaluate') as evaluate:
            with self.assertRaises(ValueError):
                check.score(self.out)
            reader.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse((self.out/'public_check_started.json').exists())

    def test_scoring_failure_consumes_gate_and_retry_never_reads_reference(self):
        results = self.root/'experiments/results'
        results.mkdir(parents=True)
        gt = self.root/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
        gt.parent.mkdir(parents=True)
        gt.write_text('synthetic bytes; pd.read_csv is mocked')
        (results/'current_baseline_public_check.json').write_text(json.dumps(
            dict(organizer_gt_sha256=check.sha256(gt))))
        for name in ('light_corruption_public_check.json', 'joint_gru_public_check.json'):
            (results/name).write_text(json.dumps(dict(score=dict(spell_f1_mean=.5))))
        frame = pd.DataFrame(dict(frame=np.arange(199319), label=np.full(199319, 'X')))
        reads = []

        def synthetic_read(path):
            self.assertTrue((self.out/'public_check_started.json').exists())
            reads.append(Path(path))
            return frame

        with patch.object(check, 'check_spec', return_value=self.spec), \
             patch.object(pd, 'read_csv', side_effect=synthetic_read), \
             patch.object(pm, 'evaluate', side_effect=RuntimeError('synthetic scoring failure')) as evaluate:
            with self.assertRaisesRegex(RuntimeError, 'synthetic scoring failure'):
                check.score(self.out)
            self.assertEqual(reads.count(gt), 1)
            self.assertEqual(evaluate.call_count, 1)
            before = list(reads)
            with self.assertRaisesRegex(ValueError, 'gate is closed'):
                check.score(self.out)
            self.assertEqual(reads, before)
            self.assertEqual(evaluate.call_count, 1)
        self.assertFalse((self.out/'report.json').exists())
        self.assertTrue((self.out/'public_check_started.json').exists())

    def test_all_mutating_actions_reject_closed_gate_before_loading_inputs(self):
        (self.out/'public_check_started.json').write_text('{}')
        with patch.object(check, 'check_spec', side_effect=AssertionError('read after closed gate')), \
             patch.object(check.current_baseline, 'load_inputs') as inputs:
            for action in (check.freeze, check.prepare, check.seal_models, check.predict, check.score):
                with self.subTest(action=action.__name__), self.assertRaises(ValueError):
                    action(self.out)
            inputs.assert_not_called()

    def test_prediction_keeps_native_out_of_play_frames_unknown(self):
        (self.out/'prediction_seal.json').unlink()
        for name in check.RECIPES:
            for suffix in ('.csv', '.npz'):
                (self.out/(name+suffix)).unlink()
        record = dict(indices=np.arange(4), native_n=4)
        cache = dict(tree=record, gru=record, period=np.arange(4), fps=25.,
                     native_in_play=np.array([False, True, True, False]))
        probability = np.tile([.8, .15, .05], (4, 1))
        observed = []

        def load_model(out, path):
            return cache if path.name == 'competition_features.joblib' else {}

        def write_submission(labels, path):
            observed.append(np.asarray(labels).copy())
            path.write_text('synthetic short-clock prediction')

        with patch.object(check, 'check_spec', return_value=self.spec), \
             patch.object(check, 'seal_models'), \
             patch.object(check.ce, 'load_model', side_effect=load_model), \
             patch.object(check.joint_lgbm, 'predict', return_value=probability), \
             patch.object(check.factorized_had, 'predict', return_value=probability), \
             patch.object(check.lt, 'upsample', side_effect=lambda values, record, period: values), \
             patch.object(check.challenge, 'write_submission', side_effect=write_submission), \
             patch.object(pd, 'read_csv') as reader, patch.object(pm, 'evaluate') as evaluate:
            check.predict(self.out)
            reader.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(len(observed), 4)
        for labels in observed:
            np.testing.assert_array_equal(labels, ['X', 'H', 'H', 'X'])


if __name__ == '__main__':
    unittest.main()
