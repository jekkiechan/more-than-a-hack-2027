import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import soccernet_segments as segments


def _entity(jersey, visibility='VISIBLE', x=0.0, y=0.0, player_id=None):
    result = {'jerseyNum': jersey, 'visibility': visibility, 'x': x, 'y': y}
    if player_id is not None:
        result['playerId'] = player_id
    return result


def _row(frame, *, period=1, valid=True, timestamp=None, home=None, away=None,
         balls=None, home_team=None):
    if timestamp is None and valid:
        timestamp = frame / 29.970001
    return {
        'frameNum': frame,
        'period': period if valid else -1,
        'videoTimeMs': None if not valid else timestamp * 1000,
        'tracking_valid': valid,
        'homePlayers': json.dumps(home if home is not None else [_entity('7', x=frame)]),
        'awayPlayers': json.dumps(away if away is not None else [_entity('2', x=-frame)]),
        'balls': json.dumps(balls if balls is not None else [_entity(None, x=frame / 2)]),
        'home_team': home_team,
        # Must never be read as a feature or restore an invisible raw point.
        'homePlayersSmoothed': json.dumps([_entity('7', x=9999)]),
    }


def _sample(sample_id, start, center, team='home', action='PASS', period=1):
    return {
        'sample_id': sample_id, 'start_frame': start, 'center_frame': center,
        'team': team, 'action': action, 'fps': 29.970001,
        'period_hint': period,
    }


def _patched_features():
    def team_features(inputs):
        allowed = {'ball', 'players', 'teams', 'period', 'fps', 'timestamps',
                   'pitch', 'ball_observed', 'player_observed'}
        if set(inputs) != allowed:
            raise AssertionError(f'Feature API received unexpected inputs: {set(inputs)-allowed}')
        ball_x = np.nan_to_num(inputs['ball'][:, 0], nan=-123.0)
        return np.column_stack((ball_x, inputs['ball_observed'].astype(float))), ['ball_x', 'ball_observed']

    def player_features(inputs):
        allowed = {'ball', 'players', 'teams', 'period', 'fps', 'timestamps',
                   'pitch', 'ball_observed', 'player_observed'}
        if not set(inputs).issubset(allowed):
            raise AssertionError('Player feature allowlist widened')
        count = np.isfinite(inputs['players']).all(axis=2).sum(axis=1)
        return count[:, None].astype(float), ['visible_player_count']

    return patch('experiments.refine_features.features', side_effect=team_features), \
        patch('experiments.learned_temporal.player_features', side_effect=player_features)


class SoccerNetSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.team_patch, self.player_patch = _patched_features()
        self.team_patch.start()
        self.player_patch.start()

    def tearDown(self):
        self.team_patch.stop()
        self.player_patch.stop()

    def test_exact_overlap_deduplicates_and_keeps_one_sparse_anchor(self):
        first = [_row(i) for i in range(8)]
        second = [_row(i) for i in range(4, 10)]
        samples = [_sample('clip_a', 0, 4), _sample('clip_b', 4, 8, action='OUT')]
        records, stats, anchors, _ = segments._assemble_game(
            '3812', samples, {'clip_a': first, 'clip_b': second})

        self.assertEqual(stats['duplicate_rows_exact'], 4)
        self.assertEqual(stats['duplicate_rows_conflicting'], 0)
        self.assertEqual(len(records), 1)
        record = records[0]
        np.testing.assert_array_equal(record['full_frame_numbers'], np.arange(10))
        self.assertEqual(record['full_y'][4], 0)
        self.assertEqual(np.count_nonzero(record['full_y'] >= 0), 1)
        self.assertEqual(np.count_nonzero(record['y'] >= 0), 1)
        self.assertEqual(record['y'][1], 0)  # frame 4 maps to sampled frame 6.
        self.assertEqual(record['fps'], 29.970001 / 6)
        self.assertEqual(record['native_fps'], 29.970001)
        self.assertEqual(anchors[0]['status'], 'mapped')

    def test_conflicting_duplicate_is_removed_and_splits_segments(self):
        rows_a = [_row(i) for i in range(4)]
        rows_b = [_row(i) for i in range(4)]
        changed = json.loads(rows_b[2]['balls'])
        changed[0]['x'] = 50.0
        rows_b[2]['balls'] = json.dumps(changed)
        samples = [_sample('left', 0, 1, 'home'), _sample('right', 0, 3, 'away')]
        records, stats, anchors, _ = segments._assemble_game(
            '10502', samples, {'left': rows_a, 'right': rows_b})

        self.assertEqual(stats['duplicate_rows_conflicting'], 1)
        self.assertEqual(len(records), 2)
        all_frames = np.concatenate([record['full_frame_numbers'] for record in records])
        self.assertNotIn(2, all_frames)
        self.assertEqual(sorted(record['y'][0] for record in records), [0, 1])

    def test_invalid_rows_gaps_period_changes_and_time_jumps_are_breaks(self):
        rows = [
            _row(0), _row(1), _row(2),
            _row(3, valid=False),
            _row(6), _row(7),
            _row(8, period=2), _row(9, period=2),
            _row(10, period=2, timestamp=10.0),
            _row(11, period=2, timestamp=10.0 + 1 / 29.970001),
        ]
        samples = [_sample('a', 0, 2), _sample('b', 0, 7, 'away'),
                   _sample('c', 0, 9), _sample('d', 0, 11, 'away')]
        records, stats, _, _ = segments._assemble_game(
            '3813', samples, {name: rows for name in ('a', 'b', 'c', 'd')})

        self.assertEqual(stats['invalid_structural_rows'], 4)
        self.assertGreaterEqual(stats['frame_gap_breaks'], 1)
        self.assertGreaterEqual(stats['period_breaks'], 1)
        self.assertGreaterEqual(stats['timestamp_breaks'], 1)
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(np.count_nonzero(record['y'] >= 0), 1)
            self.assertFalse(np.isin(record['y'], (2, 3)).any())
        self.assertNotIn(3, np.concatenate([r['full_frame_numbers'] for r in records]))

    def test_only_visible_raw_entities_become_observations(self):
        rows = []
        for frame in range(13):
            rows.append(_row(
                frame,
                home=[_entity('7', 'VISIBLE', x=frame, y=0)],
                away=[_entity('2', 'ESTIMATED', x=40 + frame, y=10)],
                balls=[_entity(None, 'ESTIMATED', x=20 + frame, y=5)],
            ))
        sample = _sample('masked', 0, 6)
        records, stats, _, _ = segments._assemble_game('3814', [sample], {'masked': rows})
        record = records[0]

        self.assertEqual(record['full_player_observed'].shape[1], 2)
        self.assertTrue(record['full_player_observed'][:, 0].all())
        self.assertFalse(record['full_player_observed'][:, 1].any())
        self.assertFalse(record['full_ball_observed'].any())
        self.assertEqual(stats['estimated_player_entities_ignored'], 13)
        self.assertEqual(stats['estimated_ball_entities_ignored'], 13)
        self.assertTrue(np.all(record['x'][:, 1] == 0.0))
        self.assertTrue(np.all(record['x'][:, 2] == 1.0))

    def test_anchor_three_native_frames_maps_and_colliding_teams_stay_unknown(self):
        rows = [_row(i, home_team=None) for i in range(24)]
        samples = [_sample('anchor0', 0, 0, 'home'),
                   _sample('anchor7', 0, 7, 'home'),
                   _sample('anchor8', 0, 8, 'away'),
                   _sample('anchor15', 0, 15, 'home')]
        records, stats, anchors, _ = segments._assemble_game(
            '3815', samples, {sample['sample_id']: rows for sample in samples})
        record = records[0]

        self.assertEqual(record['indices'].tolist(), [0, 6, 12, 18])
        self.assertEqual(record['full_y'][7], 0)
        self.assertEqual(record['full_y'][8], 1)
        self.assertEqual(record['full_y'][15], 0)
        self.assertEqual(record['y'][0], 0)
        self.assertEqual(record['y'][1], -1)
        self.assertEqual(record['y'][2], 0)
        self.assertEqual(stats['anchor_grid_conflict'], 2)
        mapping = {entry['sample_id']: entry for entry in record['anchor_frame_mapping']}
        self.assertEqual(mapping['anchor7']['delta_native_frames'], 1)
        self.assertEqual(mapping['anchor7']['status'], 'grid_conflict')
        self.assertEqual(mapping['anchor8']['status'], 'grid_conflict')
        self.assertEqual(mapping['anchor15']['delta_native_frames'], 3)
        self.assertEqual(mapping['anchor15']['status'], 'mapped')
        self.assertNotIn(2, record['y'])

    def test_metadata_team_changes_targets_but_never_features(self):
        rows = [_row(i, home_team=None) for i in range(13)]
        home = _sample('event', 0, 6, 'home')
        away = _sample('event', 0, 6, 'away')
        home_records, *_ = segments._assemble_game('3816', [home], {'event': rows})
        away_records, *_ = segments._assemble_game('3816', [away], {'event': rows})

        np.testing.assert_array_equal(home_records[0]['x'], away_records[0]['x'])
        self.assertEqual(home_records[0]['y'][1], 0)
        self.assertEqual(away_records[0]['y'][1], 1)
        self.assertEqual(home_records[0]['names'], away_records[0]['names'])

    def test_conflicting_center_home_team_remains_unknown(self):
        rows = [_row(i, home_team='0' if i == 6 else None) for i in range(13)]
        sample = _sample('conflict', 0, 6, 'home')
        records, stats, anchors, _ = segments._assemble_game(
            '3817', [sample], {'conflict': rows})
        self.assertEqual(stats['anchor_unknown_team_conflict'], 1)
        self.assertEqual(anchors[0]['status'], 'center_team_conflict')
        self.assertEqual(records, [])

    def test_home_team_flag_conflict_changes_anchor_only_not_geometry(self):
        geometry = [_row(i, home_team='1' if i == 6 else None) for i in range(13)]
        flag_conflict = [_row(i, home_team='0' if i == 6 else None) for i in range(13)]
        samples = [_sample('center_home', 0, 6, 'home'),
                   _sample('center_away', 0, 6, 'away'),
                   _sample('edge_home', 0, 0, 'home')]
        base, _, _, _ = segments._assemble_game(
            '3817', [samples[0], samples[2]],
            {'center_home': geometry, 'edge_home': geometry})
        changed, stats, anchors, _ = segments._assemble_game(
            '3817', samples,
            {'center_home': geometry, 'center_away': flag_conflict, 'edge_home': geometry})

        self.assertEqual(stats['duplicate_rows_conflicting'], 0)
        self.assertEqual(stats['duplicate_home_team_metadata_conflicts'], 1)
        self.assertEqual(stats['anchor_center_metadata_flag_conflict'], 2)
        self.assertEqual(len(changed), 1)
        np.testing.assert_array_equal(changed[0]['full_frame_numbers'], base[0]['full_frame_numbers'])
        np.testing.assert_array_equal(changed[0]['full_player_observed'], base[0]['full_player_observed'])
        np.testing.assert_array_equal(changed[0]['x'], base[0]['x'])
        self.assertEqual(changed[0]['y'][0], 0)
        self.assertEqual(changed[0]['y'][1], -1)
        self.assertTrue(all(a['status'] == 'center_home_team_metadata_conflict'
                            for a in anchors if a['frame'] == 6))

    def test_game_assembly_is_deterministic_under_input_order(self):
        first = [_row(i) for i in range(18)]
        samples = [_sample('late', 6, 12, 'away'), _sample('early', 0, 3, 'home')]
        clips = {'late': first[6:], 'early': first[:12]}
        left, _, _, _ = segments._assemble_game('3818', samples, clips)
        right, _, _, _ = segments._assemble_game('3818', samples[::-1], clips)
        self.assertEqual([r['segment_id'] for r in left], [r['segment_id'] for r in right])
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a['x'], b['x'])
            np.testing.assert_array_equal(a['y'], b['y'])

    def test_train_reader_uses_only_train_metadata_and_shard_manifest(self):
        import pyarrow.parquet as pq

        metadata = [{'sample_id': 'train_000000', 'sample_index': 0,
                     'shard_name': 'shard-000000.tar',
                     'sample_payload': json.dumps({
                         'id': 'train_000000', 'labels': {'action': {'label': 'PASS'}},
                         'metadata': {'game_id': '3812', 'team': 'home',
                                      'game_time': '1 - 00:00',
                                      'conversion': {'start_frame': 0, 'center_frame': 72,
                                                     'measured_source_fps': 29.97}}})}]
        manifest = [{'sample_id': 'train_000000', 'shard_name': 'shard-000000.tar',
                     'wds_member': '000000000.0.parquet',
                     'input_type': 'tracking_parquet', 'status': 'ok'}]
        calls = []

        class Table:
            def __init__(self, values):
                self.values = values
            def to_pylist(self):
                return self.values

        def fake_read_table(path, columns):
            calls.append(str(path))
            return Table(metadata if str(path).endswith('train/metadata.parquet') else manifest)

        with patch.object(pq, 'read_table', side_effect=fake_read_table):
            samples, members = segments._read_train_samples(Path('/private/tracking-full'))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['period_hint'], 1)
        self.assertEqual(members['train_000000']['wds_member'], '000000000.0.parquet')
        self.assertTrue(all('/train/' in call for call in calls))
        self.assertEqual(len(calls), 2)

    def test_load_records_is_manifest_sealed_and_numerically_ordered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'games').mkdir()
            outputs = {}
            for game in ('10503', '3812'):
                relative = f'games/{game}.joblib'
                payload = {'records': [{
                    'match': f'soccernet_{game}', 'segment_id': f'{game}_p1_0-5',
                    'full_period': np.asarray([1]), 'full_frame_numbers': np.asarray([0])}]}
                joblib = segments.joblib
                joblib.dump(payload, root / relative)
                outputs[relative] = segments._sha256(root / relative)
            manifest = {'schema': 'soccernet_observed_segments_v1', 'split_used': 'train',
                        'organizer_reference_access': False,
                        'source_sha256': segments._source_hashes(), 'game_files': outputs}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            loaded = segments.load_records(root)
            self.assertEqual([record['match'] for record in loaded],
                             ['soccernet_3812', 'soccernet_10503'])
            outputs['games/10503.joblib'] = '0' * 64
            manifest['game_files'] = outputs
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'cache seal mismatch'):
                segments.load_records(root)


if __name__ == '__main__':
    unittest.main()
