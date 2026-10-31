"""Prepare bounded, event-centered SoccerNet-GAR training features.

This module only reads the official train split. It keeps action-window clips
independent and provides a center-row event-team anchor, not dense possession
labels. No model fitting or evaluation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'data/independent/soccernet_gar'
MAX_PER_GAME = 64
ELIGIBLE_ACTIONS = frozenset(('PASS', 'CROSS', 'SHOT'))
NATIVE_ROWS = 144
SAMPLE_STRIDE = 6
SAMPLED_INDICES = np.arange(0, NATIVE_ROWS, SAMPLE_STRIDE, dtype=np.int64)
FEATURE_SOURCES = ('soccernet_data.py', 'refine_features.py', 'learned_temporal.py',
                   'refine_domain.py', 'supervised.py', 'robust.py')


def _tracking_root(root):
    root = Path(root)
    candidate = root / 'tracking-full'
    return candidate if (candidate / 'train/metadata.parquet').is_file() else root


def _dataset_root(root):
    root = Path(root)
    return root.parent if root.name == 'tracking-full' else root


def _read_split_metadata(root):
    import pyarrow.parquet as pq

    path = _tracking_root(root) / 'train/metadata.parquet'
    if not path.is_file():
        raise FileNotFoundError(f'Missing train metadata parquet: {path}')
    rows = pq.read_table(path, columns=['sample_id', 'sample_payload', 'shard_name', 'sample_index']).to_pylist()
    result = []
    for row in rows:
        payload = json.loads(row['sample_payload'])
        metadata = payload.get('metadata', {})
        action = payload.get('labels', {}).get('action', {}).get('label')
        sample_id = payload.get('id', row['sample_id'])
        if sample_id != row['sample_id']:
            raise ValueError(f'Metadata sample id mismatch: {row["sample_id"]}')
        result.append({
            'sample_id': sample_id,
            'sample_index': int(row['sample_index']),
            'shard_name': row['shard_name'],
            'game_id': str(metadata.get('game_id', '')),
            'action': action,
            'team': metadata.get('team'),
            'metadata': metadata,
            'payload': payload,
        })
    return result


def select_samples(root, max_per_game=MAX_PER_GAME):
    """Select up to ``max_per_game`` eligible TRAIN clips per match.

    Eligibility and stable SHA-256 ranking use only the split metadata parquet;
    no TAR or tracking parquet is opened until after this function returns.
    """
    if max_per_game < 1:
        raise ValueError('max_per_game must be positive')
    eligible = [row for row in _read_split_metadata(root)
                if row['action'] in ELIGIBLE_ACTIONS and row['team'] in ('home', 'away')]
    by_game = {}
    for row in eligible:
        by_game.setdefault(row['game_id'], []).append(row)
    chosen = []
    for game_id in sorted(by_game, key=lambda value: (int(value) if value.isdigit() else value)):
        group = sorted(by_game[game_id], key=lambda row: (
            hashlib.sha256(row['sample_id'].encode()).hexdigest(), row['sample_id']))
        chosen.extend(group[:max_per_game])
    return chosen


def _parse_entities(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _coordinate(entity, visible=True):
    if visible and entity.get('visibility') != 'VISIBLE':
        return None
    try:
        x, y = float(entity['x']), float(entity['y'])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite((x, y)).all():
        return None
    # The local dataset card describes pitch-centered tracking coordinates.
    # Shift to the existing project convention [0, 105] x [0, 68], preserving
    # legitimate outside-pitch positions rather than clipping them.
    return x + 52.5, y + 34.0


def _split_periods(period, timestamps, valid, fps):
    """Give each continuous valid run its own period id; invalid rows break both sides."""
    segmented = np.empty(len(period), dtype=np.int32)
    segment = -1
    previous_valid = False
    previous_period = None
    previous_time = None
    for i in range(len(period)):
        if not valid[i]:
            segment += 1
            segmented[i] = segment
            previous_valid = False
            previous_period = previous_time = None
            continue
        discontinuity = (not previous_valid or period[i] != previous_period or
                         timestamps[i] <= previous_time or
                         timestamps[i] - previous_time > 1.5 / fps)
        if discontinuity:
            segment += 1
        segmented[i] = segment
        previous_valid = True
        previous_period = period[i]
        previous_time = timestamps[i]
    return segmented


def _extract_clip(rows, sample):
    if len(rows) != NATIVE_ROWS:
        raise ValueError(f'Expected {NATIVE_ROWS} native row slots, got {len(rows)}')
    metadata = sample.get('metadata', {})
    conversion = metadata.get('conversion', {})
    fps = conversion.get('measured_source_fps')
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError('Missing positive metadata.conversion.measured_source_fps')
    fps = float(fps)
    center = rows[NATIVE_ROWS // 2]
    if not bool(center.get('tracking_valid')):
        raise ValueError('center tracking row invalid')
    team = sample.get('team')
    if team not in ('home', 'away'):
        raise ValueError('center event team is not home/away')
    center_home_flag = center.get('home_team')
    if center_home_flag not in (None, ''):
        if str(center_home_flag) not in ('0', '1'):
            raise ValueError('center home_team value is not 0/1')
        if (team == 'home') != (str(center_home_flag) == '1'):
            raise ValueError('center home_team conflicts with sidecar team')
    if conversion.get('center_frame') is not None and int(center['frameNum']) != int(conversion['center_frame']):
        raise ValueError('center frame does not match metadata conversion')

    valid = np.asarray([bool(row.get('tracking_valid')) for row in rows], dtype=bool)
    original_period = np.asarray([int(row.get('period') if row.get('period') is not None else -1)
                                  for row in rows], dtype=np.int32)
    start_frame = int(conversion.get('start_frame', rows[0].get('frameNum', 0)))
    frame_numbers = np.asarray([int(row['frameNum']) if row.get('frameNum') is not None
                                else start_frame + i for i, row in enumerate(rows)], dtype=np.int64)
    timestamps = np.empty(NATIVE_ROWS, dtype=np.float64)
    for i, row in enumerate(rows):
        value = row.get('videoTimeMs')
        timestamps[i] = float(value) / 1000 if value is not None else frame_numbers[i] / fps

    # Stable clip-local slots are keyed by team and jersey number. Slot ordering
    # is deterministic, but IDs/names/jerseys never enter the feature matrix.
    jersey_sets = [set(), set()]
    parsed = []
    for i, row in enumerate(rows):
        home = _parse_entities(row.get('homePlayers'))
        away = _parse_entities(row.get('awayPlayers'))
        ball_entities = _parse_entities(row.get('balls'))
        parsed.append((home, away, ball_entities))
        if valid[i]:
            for code, entities in enumerate((home, away)):
                jersey_sets[code].update(str(e.get('jerseyNum')) for e in entities
                                         if e.get('jerseyNum') not in (None, ''))
    jerseys = [sorted(values) for values in jersey_sets]
    if not jerseys[0] or not jerseys[1]:
        raise ValueError('tracking does not provide at least one jersey slot for each team')
    slot_maps = [{jersey: i for i, jersey in enumerate(team_jerseys)}
                 for team_jerseys in jerseys]
    team_offsets = (0, len(jerseys[0]))
    players = np.full((NATIVE_ROWS, len(jerseys[0]) + len(jerseys[1]), 2), np.nan,
                       dtype=np.float64)
    native_player_observed = np.zeros(players.shape[:2], dtype=bool)
    ball = np.full((NATIVE_ROWS, 2), np.nan, dtype=np.float64)
    ball_observed = np.zeros(NATIVE_ROWS, dtype=bool)
    for i, (home, away, balls) in enumerate(parsed):
        if not valid[i]:
            continue
        for code, entities in enumerate((home, away)):
            for entity in entities:
                jersey = entity.get('jerseyNum')
                if jersey in (None, ''):
                    continue
                point = _coordinate(entity)
                if point is None:
                    continue
                slot = team_offsets[code] + slot_maps[code][str(jersey)]
                # Duplicate jersey rows are uncommon; the first visible row is stable.
                if not native_player_observed[i, slot]:
                    players[i, slot] = point
                    native_player_observed[i, slot] = True
        for entity in balls:
            point = _coordinate(entity)
            if point is not None:
                ball[i] = point
                ball_observed[i] = True
                break

    segmented_period = _split_periods(original_period, timestamps, valid, fps)
    inputs = {
        'ball': ball,
        'players': players,
        'teams': np.r_[np.zeros(len(jerseys[0]), dtype=np.int8),
                       np.ones(len(jerseys[1]), dtype=np.int8)],
        'period': segmented_period,
        'fps': np.asarray(fps),
        'timestamps': timestamps,
        'pitch': np.asarray([105.0, 68.0]),
        'ball_observed': ball_observed,
    }
    masks = {
        'tracking_valid': valid,
        'native_player_observed': native_player_observed,
        'native_ball_observed': ball_observed.copy(),
    }
    return inputs, masks, original_period


def clip_inputs(rows, metadata):
    """Return the strict tracking-only feature input allowlist for one clip."""
    inputs, _, _ = _extract_clip(rows, metadata)
    return inputs


def build_record(rows, sample):
    """Build one independent native-rate-feature / 5 Hz record for one action clip."""
    from experiments import learned_temporal as lt, refine_features, supervised

    inputs, masks, original_period = _extract_clip(rows, sample)
    team_x, team_names = refine_features.features(inputs)
    player_x, player_names = lt.player_features(inputs)
    if team_x.shape[0] != NATIVE_ROWS or player_x.shape[0] != NATIVE_ROWS:
        raise ValueError('Feature APIs did not preserve native frame rate')
    indices = SAMPLED_INDICES.copy()
    full_y = np.full(NATIVE_ROWS, -1, dtype=np.int8)
    full_y[NATIVE_ROWS // 2] = 0 if sample['team'] == 'home' else 1
    periods = supervised.model_periods(inputs)
    record = {
        'x': np.column_stack((team_x[indices], player_x[indices])).astype(np.float32),
        'y': full_y[indices].copy(),
        'names': np.asarray(team_names + ['D_' + name for name in player_names]),
        'indices': indices,
        'period': periods[indices].astype(np.int32),
        'native_n': np.asarray(NATIVE_ROWS, dtype=np.int32),
        'fps': np.asarray(float(inputs['fps'])),
        'source': np.asarray('soccernet'),
        'match': np.asarray('soccernet_' + str(sample['game_id'])),
        'clip_id': np.asarray(sample['sample_id']),
        'full_y': full_y,
        'full_period': original_period,
        'native_timestamps': inputs['timestamps'],
        'native_tracking_valid': masks['tracking_valid'],
        'native_player_observed': masks['native_player_observed'],
        'native_ball_observed': masks['native_ball_observed'],
        'sampled_tracking_valid': masks['tracking_valid'][indices],
        'sampled_player_observed': masks['native_player_observed'][indices],
        'sampled_ball_observed': masks['native_ball_observed'][indices],
    }
    return record


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_receipts(root):
    """Verify file sizes against the already acquired LFS receipt, not multi-GB hashes."""
    dataset_root = _dataset_root(root)
    receipt_path = dataset_root / 'acquisition_manifest.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('dataset') != 'OpenSportsLab/SoccerNet-GAR' or
            receipt.get('branch') != 'tracking-full'):
        raise ValueError('Acquisition receipt is not for SoccerNet-GAR tracking-full')
    required = ['tracking-full/train/metadata.parquet'] + [
        f'tracking-full/train/shards/shard-{i:06d}.tar' for i in range(5)]
    by_path = {entry['path']: entry for entry in receipt['files']}
    verified = []
    for relative in required:
        entry = by_path.get(relative)
        path = dataset_root / relative
        if (entry is None or not entry.get('verified_lfs_sha256') or
                len(entry.get('sha256', '')) != 64):
            raise ValueError(f'Missing verified acquisition receipt for {relative}')
        if not path.is_file() or path.stat().st_size != int(entry['bytes']):
            raise ValueError(f'Acquired file size differs from receipt: {relative}')
        verified.append({'path': relative, 'bytes': int(entry['bytes']),
                         'sha256_from_acquisition_receipt': entry['sha256'],
                         'verified_lfs_sha256': True})
    return receipt, verified


def _selected_manifest_row(sample):
    return {key: sample[key] for key in ('sample_id', 'sample_index', 'shard_name',
                                         'game_id', 'action', 'team')}


def prepare(root=DEFAULT_ROOT, output=None, max_per_game=MAX_PER_GAME):
    """Prepare eligible TRAIN records and a private manifest; no fit or scoring."""
    from datetime import datetime, timezone

    root = Path(root)
    dataset_root = _dataset_root(root)
    tracking_root = _tracking_root(root)
    if output is None:
        output = dataset_root / 'prepared_train_v1'
    output = Path(output)
    acquisition, source_receipts = _verify_receipts(root)
    selected = select_samples(root, max_per_game=max_per_game)
    by_shard = {}
    for sample in selected:
        by_shard.setdefault(sample['shard_name'], []).append(sample)
    records, rejections = [], {}
    for shard_name in sorted(by_shard):
        shard_path = tracking_root / 'train/shards' / shard_name
        lookup = {}
        for sample in by_shard[shard_name]:
            basename = f"{sample['sample_index']:09d}"
            lookup[basename + '.json'] = sample
            lookup[basename + '.0.parquet'] = sample
        with tarfile.open(shard_path, mode='r:') as archive:
            members = {member.name: member for member in archive.getmembers()
                       if member.isfile() and member.name in lookup}
            for sample in by_shard[shard_name]:
                basename = f"{sample['sample_index']:09d}"
                json_name, parquet_name = basename + '.json', basename + '.0.parquet'
                if json_name not in members or parquet_name not in members:
                    rejections['selected_tar_member_missing'] = rejections.get('selected_tar_member_missing', 0) + 1
                    continue
                sidecar = json.load(archive.extractfile(members[json_name]))
                if sidecar.get('id') != sample['sample_id']:
                    rejections['selected_sidecar_id_mismatch'] = rejections.get('selected_sidecar_id_mismatch', 0) + 1
                    continue
                # Metadata parquet supplies ranking/eligibility; selected sidecars
                # are checked for parity before the tracking rows are prepared.
                payload = sidecar
                metadata = payload.get('metadata', {})
                if (payload.get('labels', {}).get('action', {}).get('label') != sample['action'] or
                        metadata.get('team') != sample['team'] or
                        str(metadata.get('game_id')) != sample['game_id']):
                    rejections['selected_sidecar_metadata_mismatch'] = rejections.get('selected_sidecar_metadata_mismatch', 0) + 1
                    continue
                sample_for_record = dict(sample, metadata=metadata, payload=payload)
                import pyarrow.parquet as pq
                raw = archive.extractfile(members[parquet_name]).read()
                rows = pq.read_table(io.BytesIO(raw)).to_pylist()
                try:
                    record = build_record(rows, sample_for_record)
                except ValueError as exc:
                    message = str(exc)
                    key = ('reject_center_invalid' if 'center tracking row invalid' in message else
                           'reject_center_team_conflict' if 'conflicts with sidecar team' in message else
                           'reject_unusable_clip')
                    rejections[key] = rejections.get(key, 0) + 1
                    continue
                records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'records.joblib'
    joblib.dump(records, records_path, compress=3)
    feature_hashes = {name: _sha256(ROOT / 'experiments' / name)
                      for name in FEATURE_SOURCES}
    manifest = {
        'schema': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': acquisition.get('dataset'),
        'revision': acquisition.get('revision'),
        'split_used': 'train',
        'eligibility': {'actions': sorted(ELIGIBLE_ACTIONS), 'sidecar_team': ['home', 'away'],
                        'max_per_game': int(max_per_game),
                        'ranking': 'ascending SHA-256(sample_id), then sample_id; selected before opening TAR members'},
        'label_recipe': 'full_y=-1 except row 72; center label 0=H or 1=A from sidecar event team; no dense possession claims',
        'feature_recipe': {
            'raw_input_allowlist': ['ball', 'players', 'teams', 'period', 'fps',
                                    'timestamps', 'pitch', 'ball_observed'],
            'entity_fields_read': ['homePlayers', 'awayPlayers', 'balls'],
            'ignored_fields': ['homePlayersSmoothed', 'awayPlayersSmoothed', 'ballsSmoothed',
                               'game_event_id', 'possession_event_id', 'game_event_type',
                               'possession_event_type', 'player_name', 'player_id', 'team_id',
                               'home_team', 'sidecar action/team/game identity as features'],
            'coordinate_transform': 'pitch-centered metres to 105x68 corner coordinates: x+52.5, y+34; no clipping',
            'visibility_mask': 'only visibility==VISIBLE and tracking_valid==true; all other coordinates NaN',
            'fps': 'metadata.conversion.measured_source_fps',
            'timestamps': 'videoTimeMs/1000 when present; otherwise frameNum/fps (frameNum fallback start_frame+row)',
            'feature_apis': ['experiments.refine_features.features', 'experiments.learned_temporal.player_features'],
            'feature_rate': 'native 144 rows; sample arange(0,144,6) after feature calculation; center index 72 included',
            'discontinuities': 'new feature period for invalid rows, source period changes, or timestamp jumps; each clip processed independently',
            'feature_source_sha256': feature_hashes,
            'dependency_versions': {package: importlib.metadata.version(package)
                                    for package in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        },
        'source_receipts': source_receipts,
        'counts': {'selected_after_metadata_cap': len(selected), 'prepared_records': len(records),
                   'rejections': rejections,
                   'selected_by_game': {game: sum(row['game_id'] == game for row in selected)
                                        for game in sorted({row['game_id'] for row in selected})}},
        'selected_samples': [_selected_manifest_row(sample) for sample in selected],
        'record_file': {'path': records_path.name, 'sha256': _sha256(records_path),
                        'bytes': records_path.stat().st_size},
        'semantics_limit': 'Each record is an independent action-centered window. Off-center labels remain unknown; this is not dense H/A/D/X possession training data.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-per-game', type=int, default=MAX_PER_GAME)
    args = parser.parse_args()
    result = prepare(args.root, args.output, args.max_per_game)
    print(json.dumps({'records': result['counts']['prepared_records'],
                      'selected': result['counts']['selected_after_metadata_cap'],
                      'rejections': result['counts']['rejections'],
                      'output': str(args.output or (_dataset_root(args.root) / 'prepared_train_v1'))},
                     indent=2))


if __name__ == '__main__':
    main()
