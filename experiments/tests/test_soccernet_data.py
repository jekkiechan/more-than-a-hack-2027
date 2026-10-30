"""Synthetic contracts for the bounded SoccerNet-GAR preparation adapter."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import soccernet_data as gar


def sample_metadata(sample_id='train_000000', game_id='3812', team='home', action='PASS', index=0):
    payload = {
        'id': sample_id,
        'labels': {'action': {'label': action}},
        'metadata': {
            'game_id': str(game_id), 'team': team, 'position_ms': 5000,
            'conversion': {'start_frame': 1000, 'center_frame': 1072,
                           'end_frame_exclusive': 1144,
                           'measured_source_fps': 29.97},
        },
    }
    return {
        'sample_id': sample_id, 'sample_index': index,
        'shard_name': 'shard-000000.tar', 'game_id': str(game_id),
        'action': action, 'team': team, 'metadata': payload['metadata'],
        'payload': payload,
    }


def synthetic_rows():
    rows = []
    for i in range(144):
        home = [
            {'jerseyNum': jersey, 'visibility': 'VISIBLE', 'confidence': 'HIGH',
             'x': -15 + jersey + i * .025, 'y': -10 + jersey}
            for jersey in (1, 2)
        ]
        away = [
            {'jerseyNum': jersey, 'visibility': 'VISIBLE', 'confidence': 'MEDIUM',
             'x': 15 + jersey - i * .02, 'y': 12 - jersey}
            for jersey in (3, 4)
        ]
        rows.append({
            'frameNum': 1000 + i, 'videoTimeMs': (1000 + i) / 29.97 * 1000,
            'period': 1 if i < 80 else 2, 'tracking_valid': True,
            'home_team': '1', 'balls': json.dumps([{
                'visibility': 'VISIBLE', 'x': i * .05, 'y': 2.0, 'z': 0.0,
            }]),
            'homePlayers': json.dumps(home), 'awayPlayers': json.dumps(away),
            # Deliberately label-like fields must not affect model inputs.
            'game_event_id': 7000 + i, 'possession_event_id': 8000 + i,
            'game_event_type': 'OTB', 'possession_event_type': 'PA',
            'player_name': 'annotation name', 'player_id': 'annotation id',
            'team_id': 'annotation team', 'homePlayersSmoothed': 'corrupt smoothed data',
            'awayPlayersSmoothed': 'corrupt smoothed data', 'ballsSmoothed': 'corrupt smoothed data',
        })
    return rows


class SoccerNetPreparationTests(unittest.TestCase):
    def test_selection_is_stable_limited_per_game_and_train_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'tracking-full'
            (root / 'train').mkdir(parents=True)
            import pyarrow as pa
            import pyarrow.parquet as pq

            rows = []
            for game in ('3812', '3813'):
                for i in range(5):
                    sample_id = f'train_{game}_{i:03d}'
                    metadata = sample_metadata(sample_id, game, 'home' if i % 2 else 'away',
                                               'PASS', i)
                    rows.append({'sample_id': sample_id,
                                 'sample_payload': json.dumps(metadata['payload']),
                                 'shard_name': metadata['shard_name'], 'sample_index': i})
            # Ineligible rows are rejected before any tracking TAR is required.
            for i, (action, team) in enumerate((('HEADER', 'home'), ('PASS', 'unknown'))):
                sample_id = f'train_bad_{i}'
                metadata = sample_metadata(sample_id, '3812', team, action, 10+i)
                rows.append({'sample_id': sample_id,
                             'sample_payload': json.dumps(metadata['payload']),
                             'shard_name': metadata['shard_name'], 'sample_index': 10+i})
            pq.write_table(pa.Table.from_pylist(rows), root / 'train/metadata.parquet')
            (root / 'valid').mkdir()
            (root / 'test').mkdir()
            for split in ('valid', 'test'):
                sample_id = f'{split}_should_not_be_read'
                excluded = sample_metadata(sample_id, '9999', 'home', 'PASS', 0)
                pq.write_table(pa.Table.from_pylist([{
                    'sample_id': sample_id,
                    'sample_payload': json.dumps(excluded['payload']),
                    'shard_name': excluded['shard_name'], 'sample_index': 0,
                }]), root / split / 'metadata.parquet')
            first = gar.select_samples(root, max_per_game=2)
            second = gar.select_samples(root, max_per_game=2)
            self.assertEqual([r['sample_id'] for r in first], [r['sample_id'] for r in second])
            self.assertEqual({r['game_id'] for r in first}, {'3812', '3813'})
            self.assertEqual({game: sum(r['game_id'] == game for r in first)
                              for game in ('3812', '3813')}, {'3812': 2, '3813': 2})
            for game in ('3812', '3813'):
                expected = sorted((r['sample_id'] for r in rows
                                   if json.loads(r['sample_payload'])['metadata']['game_id'] == game
                                   and json.loads(r['sample_payload'])['labels']['action']['label'] == 'PASS'
                                   and json.loads(r['sample_payload'])['metadata']['team'] in ('home', 'away')),
                                  key=lambda sample_id: (hashlib.sha256(sample_id.encode()).hexdigest(), sample_id))[:2]
                self.assertEqual([r['sample_id'] for r in first if r['game_id'] == game], expected)

    def test_event_labels_are_not_features_and_only_center_is_known(self):
        rows = synthetic_rows()
        sample = sample_metadata()
        allowlist = {'ball', 'players', 'teams', 'period', 'fps', 'timestamps',
                     'pitch', 'ball_observed'}
        self.assertEqual(set(gar.clip_inputs(rows, sample)), allowlist)
        record = gar.build_record(rows, sample)
        changed = copy.deepcopy(sample)
        changed['action'] = 'SHOT'
        changed['payload']['labels']['action']['label'] = 'SHOT'
        second = gar.build_record(rows, changed)
        np.testing.assert_allclose(record['x'], second['x'], equal_nan=True)
        self.assertEqual(record['full_y'].shape, (144,))
        self.assertEqual(np.flatnonzero(record['full_y'] >= 0).tolist(), [72])
        self.assertEqual(record['full_y'][72], 0)
        self.assertEqual(np.flatnonzero(record['y'] >= 0).tolist(), [12])
        self.assertEqual(record['y'][12], 0)
        np.testing.assert_array_equal(record['indices'], np.arange(0, 144, 6))
        self.assertIn(72, record['indices'])
        self.assertEqual(record['clip_id'].item(), sample['sample_id'])
        self.assertEqual(record['source'].item(), 'soccernet')
        self.assertEqual(record['match'].item(), 'soccernet_3812')

    def test_visibility_and_invalid_rows_remain_masks_and_break_context(self):
        rows = synthetic_rows()
        rows[20]['tracking_valid'] = False
        rows[20]['balls'] = '[]'
        rows[21]['homePlayers'] = json.dumps([
            {'jerseyNum': j, 'visibility': 'ESTIMATED', 'x': -5, 'y': 0}
            for j in (1, 2)])
        rows[21]['balls'] = json.dumps([{'visibility': 'ESTIMATED', 'x': 1, 'y': 2}])
        record = gar.build_record(rows, sample_metadata())
        self.assertFalse(record['native_tracking_valid'][20])
        self.assertFalse(record['native_player_observed'][20].any())
        self.assertFalse(record['native_ball_observed'][20])
        self.assertFalse(record['native_player_observed'][21, :2].any())
        self.assertFalse(record['native_ball_observed'][21])
        self.assertTrue(np.isnan(gar.clip_inputs(rows, sample_metadata())['players'][21, :2]).all())
        period = record['full_period']
        # An invalid row and a source-period change create separate feature blocks.
        model_period = record['period']
        self.assertLess(model_period[3], model_period[4])  # rows 18 then 24, around invalid 20
        self.assertLess(model_period[12], model_period[14])  # sampled 72 then 84 across half break at 80
        self.assertEqual(period[72], 1)
        self.assertEqual(period[84], 2)

    def test_invalid_or_conflicting_center_is_rejected(self):
        rows = synthetic_rows()
        rows[72]['tracking_valid'] = False
        with self.assertRaisesRegex(ValueError, 'center tracking row invalid'):
            gar.build_record(rows, sample_metadata())
        rows = synthetic_rows()
        rows[72]['home_team'] = '0'
        with self.assertRaisesRegex(ValueError, 'conflicts with sidecar team'):
            gar.build_record(rows, sample_metadata(team='home'))

    def test_home_away_swap_changes_only_team_attribution(self):
        rows = synthetic_rows()
        original = gar.build_record(rows, sample_metadata(team='home'))
        swapped_rows = copy.deepcopy(rows)
        for row in swapped_rows:
            row['homePlayers'], row['awayPlayers'] = row['awayPlayers'], row['homePlayers']
            row['home_team'] = '0'
        swapped = gar.build_record(swapped_rows, sample_metadata(team='away'))
        from experiments.supervised import swap_team_features
        expected = swap_team_features(original['x'], original['names'].tolist())
        np.testing.assert_allclose(swapped['x'], expected, equal_nan=True, atol=1e-4)
        self.assertEqual(original['full_y'][72], 0)
        self.assertEqual(swapped['full_y'][72], 1)

    def test_clips_are_built_independently(self):
        first_rows = synthetic_rows()
        second_rows = synthetic_rows()
        for row in second_rows:
            row['frameNum'] += 10_000
            row['videoTimeMs'] += 10_000_000
            row['balls'] = json.dumps([{'visibility': 'VISIBLE', 'x': 30, 'y': 20}])
        sample_a = sample_metadata('train_000000', '3812', 'home', 'PASS', 0)
        sample_b = sample_metadata('train_000001', '3812', 'away', 'CROSS', 1)
        sample_b['metadata']['conversion'].update(start_frame=11_000,
                                                   center_frame=11_072,
                                                   end_frame_exclusive=11_144)
        for row in second_rows:
            row['home_team'] = '0'
        a = gar.build_record(first_rows, sample_a)
        b_alone = gar.build_record(second_rows, sample_b)
        # Building another clip cannot change the first clip's boundary features.
        gar.build_record(first_rows, sample_a)
        b_again = gar.build_record(second_rows, sample_b)
        np.testing.assert_allclose(b_alone['x'], b_again['x'], equal_nan=True)
        self.assertEqual(a['native_n'].item(), 144)
        self.assertEqual(b_alone['native_n'].item(), 144)
        self.assertEqual(a['indices'][0], 0)
        self.assertEqual(b_alone['indices'][0], 0)
        self.assertNotEqual(a['clip_id'].item(), b_alone['clip_id'].item())

    def test_acquisition_receipt_sizes_are_checked_without_rehashing_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'tracking-full/train/shards').mkdir(parents=True)
            paths = ['tracking-full/train/metadata.parquet'] + [
                f'tracking-full/train/shards/shard-{i:06d}.tar' for i in range(5)]
            entries = []
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'acquired fixture')
                entries.append({'path': relative, 'bytes': path.stat().st_size,
                                'sha256': 'a' * 64, 'verified_lfs_sha256': True})
            (root / 'acquisition_manifest.json').write_text(json.dumps({
                'dataset': 'OpenSportsLab/SoccerNet-GAR', 'branch': 'tracking-full',
                'files': entries,
            }))
            with patch.object(gar, '_sha256', side_effect=AssertionError('rehash')):
                receipt, verified = gar._verify_receipts(root)
            self.assertEqual(receipt['files'][0]['sha256'], 'a' * 64)
            self.assertEqual(verified[1]['sha256_from_acquisition_receipt'], 'a' * 64)
            self.assertEqual(len(verified), 6)
            (root / paths[-1]).write_bytes(b'wrong size')
            with self.assertRaisesRegex(ValueError, 'differs from receipt'):
                gar._verify_receipts(root)

    def test_prepare_reads_only_selected_train_members_and_writes_private_records(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            track = root / 'tracking-full'
            (track / 'train/shards').mkdir(parents=True)
            sample = sample_metadata('train_000000', '3812', 'home', 'PASS', 0)
            payload_bytes = json.dumps(sample['payload']).encode()
            parquet_sink = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(synthetic_rows()), parquet_sink)
            parquet_bytes = parquet_sink.getvalue()
            shard_paths = [track / f'train/shards/shard-{i:06d}.tar' for i in range(5)]
            with tarfile.open(shard_paths[0], 'w') as archive:
                for member_name, content in (('000000000.json', payload_bytes),
                                             ('000000000.0.parquet', parquet_bytes)):
                    info = tarfile.TarInfo(member_name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            for path in shard_paths[1:]:
                with tarfile.open(path, 'w'):
                    pass
            metadata_path = track / 'train/metadata.parquet'
            pq.write_table(pa.Table.from_pylist([{
                'sample_id': sample['sample_id'],
                'sample_payload': json.dumps(sample['payload']),
                'shard_name': sample['shard_name'], 'sample_index': 0,
            }]), metadata_path)
            receipt_files = []
            for path in [metadata_path, *shard_paths]:
                relative = path.relative_to(root).as_posix()
                receipt_files.append({'path': relative, 'bytes': path.stat().st_size,
                                      'sha256': 'c' * 64, 'verified_lfs_sha256': True})
            (root / 'acquisition_manifest.json').write_text(json.dumps({
                'dataset': 'OpenSportsLab/SoccerNet-GAR', 'revision': 'fixture',
                'branch': 'tracking-full', 'files': receipt_files,
            }))
            result = gar.prepare(root, root / 'prepared', max_per_game=1)
            self.assertEqual(result['counts']['selected_after_metadata_cap'], 1)
            self.assertEqual(result['counts']['prepared_records'], 1)
            self.assertEqual(result['counts']['rejections'], {})
            self.assertEqual(result['source_receipts'][1]['sha256_from_acquisition_receipt'], 'c' * 64)
            records = gar.joblib.load(root / 'prepared/records.joblib')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['clip_id'].item(), sample['sample_id'])
            self.assertEqual(np.flatnonzero(records[0]['full_y'] >= 0).tolist(), [72])
            self.assertTrue((root / 'prepared/manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
