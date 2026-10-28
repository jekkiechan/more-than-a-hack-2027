"""Reusable preparation contracts, synthetic arrays only; no model fitting."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from experiments import learned_temporal as lt, preparation as prep, supervised


def synthetic(n=200):
    t = np.arange(n)/10
    players = np.stack([np.column_stack((25+i+t*.01, 20+np.sin(t*.1+i)))
                        for i in range(12)], axis=1)
    ball = players[:, 0].copy()
    ball[50:80] = np.nan
    return dict(players=players, ball=ball, teams=np.repeat([0, 1], 6),
                period=np.ones(n, dtype=int), timestamps=t, fps=np.array(10.),
                pitch=np.array([105., 68.]), labels=np.resize(['H', 'A', 'D', 'X'], n))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = json.loads(prep.MANIFEST.read_text())
        self.warnings = warnings.catch_warnings()
        self.warnings.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)
        self.addCleanup(self.warnings.__exit__, None, None, None)

    def plan(self, manifest, fold='pooled'):
        path = self.root/'manifest.json'
        path.write_text(json.dumps(manifest))
        return prep.split_plan(path, fold)

    def item(self, data=None):
        data = synthetic() if data is None else data
        path = self.root/'external/skillcorner_had_synthetic.npz'
        path.parent.mkdir(exist_ok=True)
        data['player_observed'] = np.isfinite(data['players']).all(axis=2)
        data['ball_observed'] = np.isfinite(data['ball']).all(axis=1)
        # Finite but provider-unobserved coordinates must not enter features.
        data['player_observed'][10:20, :2] = False
        np.savez(path, **data)
        return dict(source='skillcorner', match=path.stem, group_id=path.stem,
                    path='data/external/'+path.name, sha256=prep.sha256(path), role='train')

    def test_existing_cohort_and_full_provider_folds(self):
        pooled = self.plan(self.manifest)
        self.assertEqual((len(pooled['train']), len(pooled['selection'])), (13, 5))
        for source in self.manifest['sources']:
            plan = self.plan(self.manifest, 'leave_out_'+source)
            self.assertNotIn(source, {r['source'] for r in plan['train']})
            self.assertEqual({r['source'] for r in plan['selection']}, {source})
            self.assertFalse({r['group_id'] for r in plan['train']} &
                             {r['group_id'] for r in plan['selection']})

    def test_rejects_group_file_aliases_unsafe_paths_and_holdout_leakage(self):
        for field in ('group_id', 'match', 'path'):
            manifest = copy.deepcopy(self.manifest)
            source = manifest['sources']['sportec']
            source['selection'][0][field] = source['train'][0][field]
            with self.assertRaises(ValueError):
                self.plan(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['sources']['sportec']['train'][0]['path'] = 'data/../private.npz'
        with self.assertRaises(ValueError):
            self.plan(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['folds'][1]['train_sources'].append('sportec')
        with self.assertRaises(ValueError):
            self.plan(manifest, 'leave_out_sportec')

    def test_unknown_fold_and_historical_tests_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(self.manifest, 'fresh_test')
        manifest = copy.deepcopy(self.manifest)
        item = manifest['sources']['sportec']['train'][0]
        item.update(match='J03WPY', path='data/external/J03WPY.npz')
        with self.assertRaises(ValueError):
            self.plan(manifest)

    def test_features_ignore_target_fields_and_preserve_blocks_unknowns_swaps(self):
        from experiments.live_gap_experiment import feature_view
        data = synthetic(203)
        data['period'][71:] = 2
        data['timestamps'][153:] += 10
        labels = data['labels'].copy()
        a = prep.build_record(data, labels, 's', 'm')
        baseline = feature_view(data, labels, 's', 'm')['joint']
        for key in ('x', 'y', 'indices', 'period', 'names'):
            np.testing.assert_array_equal(a[key], baseline[key])
        b = prep.build_record(dict(data, labels=np.full(203, 'D'), owner=np.ones(203)), labels, 's', 'm')
        np.testing.assert_array_equal(a['x'], b['x'])
        expected = lt.sampled_indices(data)
        np.testing.assert_array_equal(a['indices'], expected)
        self.assertIn(71, expected)
        self.assertIn(153, expected)
        np.testing.assert_array_equal(a['y'][labels[expected] == 'X'], -1)
        for _, _, full in lt.chunks(a):
            self.assertEqual(len(np.unique(a['period'][full])), 1)
        swapped = prep.build_record(dict(data, teams=1-data['teams']), labels, 's', 'm')
        np.testing.assert_allclose(swapped['x'], supervised.swap_team_features(a['x'], a['names']),
                                   equal_nan=True, atol=1e-5)

    def test_cache_reuse_integrity_and_native_grading(self):
        item = self.item()
        config = prep.recipe()
        directory = prep.prepare_match(item, self.root, self.root/'cache', config)
        record = prep.load_record(directory)
        self.assertTrue((record['native_labels'][10:20] == 'X').all())
        self.assertFalse(record['native_player_observed'][10:20, :2].any())
        with patch.object(prep, 'build_record', side_effect=AssertionError('cache was rebuilt')):
            self.assertEqual(prep.prepare_match(dict(item, role='selection'), self.root,
                                               self.root/'cache', config), directory)
        with (directory/'record.npz').open('ab') as handle:
            handle.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prep.prepare_match(item, self.root, self.root/'cache', config)

    def test_input_seal_and_recipe_changes_invalidate(self):
        item = self.item()
        config = prep.recipe()
        self.assertEqual(config, prep.recipe(seed=999))  # Clean features are seed-independent.
        directory = prep.prepare_match(item, self.root, self.root/'cache', config)
        changed = copy.deepcopy(config)
        changed['source_sha256']['preparation.py'] = 'changed'
        other = prep.prepare_match(item, self.root, self.root/'cache', changed)
        self.assertNotEqual(directory, other)
        path = prep.data_path(item, self.root)
        with path.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Input checksum'):
            prep.prepare_match(item, self.root, self.root/'cache', config)

    def test_stress_is_deterministic_and_keeps_native_population(self):
        item = self.item(synthetic(500))
        calibration = dict(ball_missing_fraction=.41, gap_edges_seconds=[.2, 1., 3.],
                           gap_bin_probability=[.5, .5])
        config = prep.recipe('feed_like', 17, calibration)
        a = prep.load_record(prep.prepare_match(item, self.root, self.root/'a', config))
        b = prep.load_record(prep.prepare_match(item, self.root, self.root/'b', config))
        clean = prep.load_record(prep.prepare_match(item, self.root, self.root/'a', prep.recipe()))
        np.testing.assert_array_equal(a['x'], b['x'])
        for key in ('y', 'native_labels', 'native_player_observed', 'indices', 'period'):
            np.testing.assert_array_equal(a[key], clean[key])
        self.assertFalse(np.array_equal(a['x'], clean['x'], equal_nan=True))
        with self.assertRaises(ValueError):
            prep.recipe('feed_like')

    def test_prepare_cli_writes_reusable_sealed_plan(self):
        manifest = copy.deepcopy(self.manifest)
        for values in manifest['sources'].values():
            for role in ('train', 'selection'):
                values[role] = values[role][:1]
                for item in values[role]:
                    path = prep.data_path(item, self.root)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = synthetic()
                    data['player_observed'] = np.isfinite(data['players']).all(axis=2)
                    data['ball_observed'] = np.isfinite(data['ball']).all(axis=1)
                    np.savez(path, **data)
                    item['sha256'] = prep.sha256(path)
        manifest_path = self.root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        argv = ['preparation', 'prepare', '--manifest', str(manifest_path),
                '--fold', 'leave_out_skillcorner', '--data-dir', str(self.root)]
        output = io.StringIO()
        with patch('sys.argv', argv), redirect_stdout(output):
            prep.main()
        plan = json.loads(output.getvalue())
        cache = self.root/'experiments/prepared_v1'
        sealed = cache/'plans'/(prep.fingerprint(plan)+'.json')
        self.assertEqual(json.loads(sealed.read_text()), plan)
        self.assertEqual((len(plan['train']), len(plan['selection'])), (2, 1))
        for item in plan['train']+plan['selection']:
            self.assertEqual(prep.load_record(cache/item['cache_key'])['match'], item['match'])
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()), patch.object(
                prep, 'build_record', side_effect=AssertionError('cache rebuilt')):
            prep.main()
        sealed.write_text('{}')
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                ValueError, 'plan checksum'):
            prep.main()


if __name__ == '__main__':
    unittest.main()
