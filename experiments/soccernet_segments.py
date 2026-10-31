"""Reconstruct observed SoccerNet-GAR TRAIN subsegments from overlapping clips.

The action annotations remain sparse event-team anchors. This module merges
only exact duplicate tracking rows, preserves unknown labels and observation
masks, and breaks at missing or discontinuous source frames. It never reads
GAR valid/test splits and never fits or scores a model.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import tarfile
from concurrent.futures import ProcessPoolExecutor

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_WORKERS = 3
ELIGIBLE_ACTIONS = frozenset(('PASS', 'CROSS', 'SHOT'))
RAW_COLUMNS = ('frameNum', 'period', 'videoTimeMs', 'tracking_valid',
               'homePlayers', 'awayPlayers', 'balls', 'home_team')
DEDUP_COLUMNS = ('frameNum', 'period', 'videoTimeMs', 'tracking_valid',
                 'homePlayers', 'awayPlayers', 'balls')
FEATURE_SOURCES = ('soccernet_segments.py', 'soccernet_data.py',
                   'refine_features.py', 'learned_temporal.py',
                   'refine_domain.py', 'supervised.py', 'robust.py')
SAMPLE_STRIDE = 6

_ARCHIVES: dict[str, tarfile.TarFile] = {}


def _tracking_root(root: str | Path) -> Path:
    root = Path(root)
    candidate = root / 'tracking-full'
    return candidate if (candidate / 'train/metadata.parquet').is_file() else root


def _dataset_root(root: str | Path) -> Path:
    tracking = _tracking_root(root)
    return tracking.parent if tracking.name == 'tracking-full' else tracking


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _verify_acquisition(root: Path) -> dict:
    """Check the pinned TRAIN receipts and small-file hashes without rereading TARs."""
    receipt_path = _dataset_root(root) / 'acquisition_manifest.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'Missing acquisition receipt: {receipt_path}')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('dataset') != 'OpenSportsLab/SoccerNet-GAR' or receipt.get('branch') != 'tracking-full':
        raise ValueError('Acquisition receipt is not for SoccerNet-GAR tracking-full')
    files = {entry['path']: entry for entry in receipt.get('files', [])}
    required = ['tracking-full/train/metadata.parquet',
                'tracking-full/train/shard_manifest.parquet'] + [
                f'tracking-full/train/shards/shard-{index:06d}.tar' for index in range(5)]
    checked = []
    base = _dataset_root(root)
    for relative in required:
        entry = files.get(relative)
        path = base / relative
        if not entry or not path.is_file():
            raise FileNotFoundError(f'Missing receipt-listed TRAIN input: {relative}')
        if path.stat().st_size != int(entry['bytes']):
            raise ValueError(f'Acquired file size differs from receipt: {relative}')
        if relative.endswith('.parquet') and _sha256(path) != entry['sha256']:
            raise ValueError(f'Acquired TRAIN metadata hash differs from receipt: {relative}')
        if relative.endswith('.tar') and entry.get('verified_lfs_sha256') is not True:
            raise ValueError(f'TAR lacks a verified acquisition hash receipt: {relative}')
        checked.append({'path': relative, 'bytes': int(entry['bytes']),
                        'sha256': entry['sha256'],
                        'verification': 'content_hash' if relative.endswith('.parquet')
                        else 'acquisition_receipt_and_exact_size'})
    return {'path': str(receipt_path.relative_to(_dataset_root(root))),
            'sha256': _sha256(receipt_path), 'revision': receipt.get('revision'),
            'branch': receipt.get('branch'), 'train_files': checked}


def _parse_payload(row: dict) -> dict:
    payload = json.loads(row['sample_payload'])
    if payload.get('id') != row['sample_id']:
        raise ValueError(f'Metadata sample_id mismatch for {row["sample_id"]}')
    metadata = payload.get('metadata') or {}
    conversion = metadata.get('conversion') or {}
    game_time = str(metadata.get('game_time', ''))
    try:
        period_hint = int(game_time.split(' - ', 1)[0])
    except (TypeError, ValueError):
        period_hint = -1
    fps = conversion.get('measured_source_fps')
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f'Missing native FPS for {row["sample_id"]}')
    return {
        'sample_id': str(row['sample_id']),
        'sample_index': int(row['sample_index']),
        'shard_name': str(row['shard_name']),
        'game_id': str(metadata.get('game_id', '')),
        'action': payload.get('labels', {}).get('action', {}).get('label'),
        'team': metadata.get('team'),
        'start_frame': int(conversion.get('start_frame', 0)),
        'center_frame': (int(conversion['center_frame'])
                         if conversion.get('center_frame') is not None else None),
        'period_hint': period_hint,
        'fps': float(fps),
    }


def _read_train_samples(tracking: Path) -> tuple[list[dict], dict[str, dict]]:
    import pyarrow.parquet as pq

    metadata_path = tracking / 'train/metadata.parquet'
    manifest_path = tracking / 'train/shard_manifest.parquet'
    raw_rows = pq.read_table(metadata_path, columns=[
        'sample_id', 'sample_index', 'shard_name', 'sample_payload']).to_pylist()
    samples = [_parse_payload(row) for row in raw_rows]
    if any(not sample['game_id'] for sample in samples):
        raise ValueError('A TRAIN sample has no game_id')
    manifest_rows = pq.read_table(manifest_path, columns=[
        'sample_id', 'shard_name', 'wds_member', 'input_type', 'status']).to_pylist()
    members = {}
    for row in manifest_rows:
        if row['input_type'] != 'tracking_parquet' or row['status'] != 'ok':
            continue
        members[str(row['sample_id'])] = {
            'shard_name': str(row['shard_name']), 'wds_member': str(row['wds_member'])}
    if set(s['sample_id'] for s in samples) != set(members):
        raise ValueError('TRAIN metadata and shard manifest sample IDs differ')
    for sample in samples:
        member = members[sample['sample_id']]
        if member['shard_name'] != sample['shard_name']:
            raise ValueError(f'Shard mismatch for {sample["sample_id"]}')
        sample['wds_member'] = member['wds_member']
    return samples, members


def _json_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _canonical(value):
    """Type-preserving canonical form for exact duplicate-row comparison."""
    if isinstance(value, dict):
        return ('dict', tuple((str(key), _canonical(value[key]))
                             for key in sorted(value, key=str)))
    if isinstance(value, list):
        return ('list', tuple(_canonical(item) for item in value))
    if isinstance(value, tuple):
        return ('tuple', tuple(_canonical(item) for item in value))
    if isinstance(value, bool):
        return ('bool', value)
    if value is None:
        return ('none',)
    if isinstance(value, (int, np.integer)):
        return ('int', int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return ('float', value.hex() if math.isfinite(value) else str(value))
    return (type(value).__name__, str(value))


def _row_signature(row: dict) -> tuple:
    # home_team is sparse event metadata. Its conflicts are audited separately
    # and must never alter geometry deduplication or segment boundaries.
    return tuple((key, _canonical(_json_value(row.get(key)))) for key in DEDUP_COLUMNS)


def _entity_compact(value) -> tuple[tuple, ...]:
    entities = _json_value(value)
    if entities is None:
        return ()
    if isinstance(entities, dict):
        entities = [entities]
    if not isinstance(entities, list):
        raise ValueError('Entity column is not a JSON list/object')
    compact = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError('Entity array contains a non-object value')
        jersey = entity.get('jerseyNum')
        jersey_key = None if jersey in (None, '') else str(jersey)
        visibility = entity.get('visibility')
        try:
            x, y = float(entity['x']), float(entity['y'])
            if not np.isfinite((x, y)).all():
                x = y = float('nan')
        except (KeyError, TypeError, ValueError):
            x = y = float('nan')
        compact.append((jersey_key, visibility, x, y))
    return tuple(compact)


def _home_flag(value) -> str | None:
    if value in (None, ''):
        return None
    value = str(value)
    if value not in ('0', '1'):
        return 'INVALID:' + value
    return value


def _native_row(row: dict) -> dict:
    frame_value = row.get('frameNum')
    if frame_value is None:
        raise ValueError('TRAIN tracking row has no frameNum')
    try:
        frame = int(frame_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid source frameNum: {frame_value!r}') from exc
    try:
        period = int(row.get('period')) if row.get('period') is not None else -1
    except (TypeError, ValueError):
        period = -1
    try:
        timestamp = (float(row['videoTimeMs']) / 1000.0
                     if row.get('videoTimeMs') is not None else None)
    except (TypeError, ValueError):
        timestamp = None
    valid = bool(row.get('tracking_valid'))
    home_flag = _home_flag(row.get('home_team'))
    return {
        'frame': frame,
        'period': period,
        'timestamp': timestamp,
        'valid': valid,
        'home_team': home_flag,
        'home': _entity_compact(row.get('homePlayers')),
        'away': _entity_compact(row.get('awayPlayers')),
        'balls': _entity_compact(row.get('balls')),
        'signature': _row_signature(row),
    }


def _empty_stats() -> dict:
    return {
        'clips_read': 0, 'raw_rows_read': 0, 'unique_valid_frames': 0,
        'duplicate_rows_exact': 0, 'duplicate_rows_conflicting': 0,
        'invalid_structural_rows': 0, 'invalid_duplicate_rows_exact': 0,
        'invalid_valid_status_conflicts': 0,
        'missing_or_invalid_timestamps': 0, 'timestamp_breaks': 0,
        'frame_gap_breaks': 0, 'period_breaks': 0,
        'unknown_jersey_entities': 0, 'duplicate_jersey_entities': 0,
        'visible_player_observations': 0, 'estimated_player_entities_ignored': 0,
        'visible_ball_observations': 0, 'estimated_ball_entities_ignored': 0,
        'entity_coords_outside_nominal_pitch': 0,
        'native_player_slots': 0, 'native_ball_rows': 0,
        'anchor_candidates': 0, 'anchor_native_known': 0,
        'anchor_unknown_team_conflict': 0, 'anchor_center_unobserved': 0,
        'anchor_center_invalid_period': 0, 'anchor_center_removed_after_merge': 0,
        'anchor_center_metadata_flag_conflict': 0,
        'duplicate_home_team_metadata_conflicts': 0,
        'anchor_out_of_grid_tolerance': 0,
        'anchor_grid_conflict': 0, 'anchor_mapped_known': 0,
        'segments_emitted': 0, 'records_emitted': 0,
        'unique_valid_frames_by_period': {},
    }


def _feature_arrays(rows: list[dict], jerseys: list[list[str]], fps: float):
    """Build the strict raw-observation input allowlist for one native segment."""
    from experiments import learned_temporal as lt
    from experiments import refine_features
    n = len(rows)
    home_slots = {value: i for i, value in enumerate(jerseys[0])}
    away_slots = {value: i + len(jerseys[0]) for i, value in enumerate(jerseys[1])}
    players = np.full((n, len(home_slots) + len(away_slots), 2), np.nan, dtype=np.float64)
    player_observed = np.zeros(players.shape[:2], dtype=bool)
    ball = np.full((n, 2), np.nan, dtype=np.float64)
    ball_observed = np.zeros(n, dtype=bool)
    duplicate_jerseys = 0
    missing_jerseys = 0
    estimated_players = 0
    estimated_balls = 0
    outside_pitch = 0
    for i, row in enumerate(rows):
        for side, slot_map in (('home', home_slots), ('away', away_slots)):
            for jersey, visibility, x, y in row[side]:
                if jersey is None:
                    missing_jerseys += 1
                    continue
                if visibility != 'VISIBLE':
                    if visibility == 'ESTIMATED':
                        estimated_players += 1
                    continue
                if not np.isfinite((x, y)).all():
                    continue
                slot = slot_map.get(jersey)
                if slot is None:
                    missing_jerseys += 1
                    continue
                if player_observed[i, slot]:
                    duplicate_jerseys += 1
                    # Duplicate identity rows are never averaged or overwritten.
                    continue
                shifted = (x + 52.5, y + 34.0)
                if not (0 <= shifted[0] <= 105 and 0 <= shifted[1] <= 68):
                    outside_pitch += 1
                players[i, slot] = shifted
                player_observed[i, slot] = True
        for _, visibility, x, y in row['balls']:
            if visibility != 'VISIBLE':
                if visibility == 'ESTIMATED':
                    estimated_balls += 1
                continue
            if np.isfinite((x, y)).all():
                shifted = (x + 52.5, y + 34.0)
                if not (0 <= shifted[0] <= 105 and 0 <= shifted[1] <= 68):
                    outside_pitch += 1
                ball[i] = shifted
                ball_observed[i] = True
                break
    teams = np.r_[np.zeros(len(home_slots), dtype=np.int8),
                  np.ones(len(away_slots), dtype=np.int8)]
    period_id = int(rows[0]['period'])
    timestamps = np.asarray([row['timestamp'] for row in rows], dtype=np.float64)
    inputs = {
        'ball': ball, 'players': players, 'teams': teams,
        'period': np.full(n, period_id, dtype=np.int32),
        'fps': np.asarray(float(fps)), 'timestamps': timestamps,
        'pitch': np.asarray([105.0, 68.0]), 'ball_observed': ball_observed,
        'player_observed': player_observed,
    }
    team_x, team_names = refine_features.features(inputs)
    player_x, player_names = lt.player_features(inputs)
    if team_x.shape[0] != n or player_x.shape[0] != n:
        raise ValueError('Native feature APIs changed the frame count')
    return (np.column_stack((team_x, player_x)).astype(np.float32),
            list(team_names) + ['D_' + name for name in player_names],
            player_observed, ball_observed,
            {'missing_jersey_entities': missing_jerseys,
             'duplicate_jersey_entities': duplicate_jerseys,
             'visible_player_observations': int(player_observed.sum()),
             'estimated_player_entities_ignored': estimated_players,
             'visible_ball_observations': int(ball_observed.sum()),
             'estimated_ball_entities_ignored': estimated_balls,
             'entity_coords_outside_nominal_pitch': outside_pitch})


def _assemble_game(game_id: str, samples: list[dict], clips: dict[str, list[dict]]):
    """Merge one game's TRAIN clips and return independent contiguous records."""
    samples = sorted(samples, key=lambda item: (item['start_frame'], item['sample_id']))
    stats = _empty_stats()
    frames: dict[tuple[int, int], dict] = {}
    raw_signatures: dict[tuple, tuple] = {}
    status_by_frame: dict[tuple[int, int], tuple[bool, tuple]] = {}
    invalid_keys: set[tuple[int, int]] = set()
    conflict_keys: set[tuple[int, int]] = set()
    home_flags_by_key: dict[tuple[int, int], set[str]] = {}
    home_flag_conflicts: set[tuple[int, int]] = set()
    jersey_union = [set(), set()]
    anchors = []
    fps_values = [float(sample['fps']) for sample in samples]
    fps = float(np.median(fps_values)) if fps_values else 0.0
    if not fps or max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f'Inconsistent measured native FPS in game {game_id}')

    for sample_index, sample in enumerate(samples):
        sample_id = sample['sample_id']
        clip = clips(sample) if callable(clips) else clips.get(sample_id)
        if clip is None:
            raise ValueError(f'Missing TRAIN clip {sample_id}')
        if callable(clips) and len(clip) != 144:
            raise ValueError(f'{sample_id} has {len(clip)} rows; expected 144 native slots')
        if (callable(clips) and sample['center_frame'] is not None and
                sample['center_frame'] != sample['start_frame'] + 72):
            raise ValueError(f'Metadata center is not native row 72 for {sample_id}')
        stats['clips_read'] += 1
        stats['raw_rows_read'] += len(clip)
        center_row = None
        for row in clip:
            parsed = _native_row(row)
            if sample['center_frame'] is not None and parsed['frame'] == sample['center_frame']:
                center_row = parsed
            if not parsed['valid']:
                stats['invalid_structural_rows'] += 1
                invalid_period = sample['period_hint']
                invalid_key = (invalid_period, parsed['frame'])
                if invalid_period > 0:
                    invalid_keys.add(invalid_key)
                status = status_by_frame.get(invalid_key)
                if status is not None and status[0]:
                    stats['invalid_valid_status_conflicts'] += 1
                    conflict_keys.add(invalid_key)
                if status is None and invalid_period > 0:
                    status_by_frame[invalid_key] = (False, invalid_key)
                signature_key = ('invalid', invalid_period, parsed['frame'])
                prior = raw_signatures.get(signature_key)
                if prior is not None:
                    if prior != parsed['signature']:
                        stats['duplicate_rows_conflicting'] += 1
                    else:
                        stats['invalid_duplicate_rows_exact'] += 1
                else:
                    raw_signatures[signature_key] = parsed['signature']
                continue
            if parsed['period'] <= 0 or parsed['timestamp'] is None or not np.isfinite(parsed['timestamp']):
                stats['missing_or_invalid_timestamps'] += 1
                if parsed['period'] > 0:
                    invalid_key = (parsed['period'], parsed['frame'])
                    invalid_keys.add(invalid_key)
                    conflict_keys.add(invalid_key)
                continue
            key = (parsed['period'], parsed['frame'])
            if parsed['home_team'] is not None:
                flags = home_flags_by_key.setdefault(key, set())
                flags.add(parsed['home_team'])
                if len(flags) > 1 and key not in home_flag_conflicts:
                    home_flag_conflicts.add(key)
                    stats['duplicate_home_team_metadata_conflicts'] += 1
            status = status_by_frame.get(key)
            if status is not None and not status[0]:
                stats['invalid_valid_status_conflicts'] += 1
                conflict_keys.add(key)
            else:
                status_by_frame[key] = (True, key)
            for side_i, side in enumerate(('home', 'away')):
                jersey_union[side_i].update(entity[0] for entity in parsed[side]
                                            if entity[0] is not None)
            signature_key = ('valid', key)
            prior = raw_signatures.get(signature_key)
            if prior is not None:
                if prior == parsed['signature']:
                    stats['duplicate_rows_exact'] += 1
                else:
                    stats['duplicate_rows_conflicting'] += 1
                    conflict_keys.add(key)
                continue
            raw_signatures[signature_key] = parsed['signature']
            parsed.pop('signature')
            frames[key] = parsed

        if sample['action'] in ELIGIBLE_ACTIONS and sample['team'] in ('home', 'away'):
            stats['anchor_candidates'] += 1
            if (center_row is None or not center_row['valid'] or
                    center_row['timestamp'] is None or not np.isfinite(center_row['timestamp'])):
                anchors.append({'sample_id': sample_id, 'action': sample['action'],
                                'team': sample['team'], 'frame': sample['center_frame'],
                                'period': -1, 'status': 'center_unobserved'})
                stats['anchor_center_unobserved'] += 1
            elif center_row['period'] <= 0:
                anchors.append({'sample_id': sample_id, 'action': sample['action'],
                                'team': sample['team'], 'frame': sample['center_frame'],
                                'period': -1, 'status': 'center_invalid_period'})
                stats['anchor_center_invalid_period'] += 1
            else:
                expected_flag = '1' if sample['team'] == 'home' else '0'
                center_key = (center_row['period'], center_row['frame'])
                if center_key in home_flag_conflicts:
                    anchors.append({'sample_id': sample_id, 'action': sample['action'],
                                    'team': sample['team'], 'frame': sample['center_frame'],
                                    'period': center_row['period'],
                                    'status': 'center_home_team_metadata_conflict'})
                    stats['anchor_center_metadata_flag_conflict'] += 1
                elif center_row['home_team'] is not None and center_row['home_team'] != expected_flag:
                    anchors.append({'sample_id': sample_id, 'action': sample['action'],
                                    'team': sample['team'], 'frame': sample['center_frame'],
                                    'period': center_row['period'], 'status': 'center_team_conflict'})
                    stats['anchor_unknown_team_conflict'] += 1
                else:
                    anchors.append({'sample_id': sample_id, 'action': sample['action'],
                                    'team': sample['team'], 'frame': sample['center_frame'],
                                    'period': center_row['period'], 'status': 'pending'})

        # Starts are chronological within game. Retain only raw signatures whose
        # frames can still overlap a later clip; compact frame geometry persists.
        if sample_index + 1 < len(samples):
            next_start = samples[sample_index + 1]['start_frame']
            raw_signatures = {key: signature for key, signature in raw_signatures.items()
                              if (key[1][1] if key[0] == 'valid' else key[2]) >= next_start}
            status_by_frame = {key: status for key, status in status_by_frame.items()
                               if key[1] >= next_start}
            home_flags_by_key = {key: flags for key, flags in home_flags_by_key.items()
                                 if key[1] >= next_start}

    # A later overlapping clip can reveal a home_team metadata conflict for a
    # center anchor already read earlier. Resolve all affected anchors only
    # after the complete game's duplicate metadata has been inspected.
    for anchor in anchors:
        key = (anchor.get('period', -1), anchor.get('frame', -1))
        if anchor.get('status') == 'pending' and key in home_flag_conflicts:
            anchor['status'] = 'center_home_team_metadata_conflict'
            stats['anchor_center_metadata_flag_conflict'] += 1

    for key in conflict_keys:
        frames.pop(key, None)
    stats['unique_valid_frames'] = len(frames)
    stats['unique_valid_frames_by_period'] = {
        str(period): sum(1 for candidate_period, _ in frames if candidate_period == period)
        for period in sorted({key[0] for key in frames})}
    jerseys = [sorted(values) for values in jersey_union]
    if not jerseys[0] or not jerseys[1]:
        raise ValueError(f'Game {game_id} lacks jersey slots for one team')

    # Split before feature extraction. Invalid/missing frames are absent from
    # `frames`, so every nonconsecutive frame number becomes a structural break.
    rows_sorted = sorted(frames.values(), key=lambda row: (row['period'], row['frame']))
    segments = []
    current = []
    current_period = None
    previous = None
    for row in rows_sorted:
        if (row['period'], row['frame']) in invalid_keys or (row['period'], row['frame']) in conflict_keys:
            if current:
                segments.append(current)
                current = []
            previous = None
            current_period = None
            continue
        discontinuity = False
        if previous is not None:
            delta = row['timestamp'] - previous['timestamp']
            discontinuity = (row['period'] != current_period or row['frame'] != previous['frame'] + 1 or
                             delta <= 0 or delta > 1.5 / fps)
            if row['period'] != current_period:
                stats['period_breaks'] += 1
            elif row['frame'] != previous['frame'] + 1:
                stats['frame_gap_breaks'] += 1
            elif delta <= 0 or delta > 1.5 / fps:
                stats['timestamp_breaks'] += 1
        if discontinuity and current:
            segments.append(current)
            current = []
        current.append(row)
        current_period = row['period']
        previous = row
    if current:
        segments.append(current)

    records = []
    names_global = None
    game_prefix = f'soccernet_{game_id}'
    for segment_number, native_rows in enumerate(segments):
        if not native_rows:
            continue
        segment_id = (f'{game_id}_p{native_rows[0]["period"]}_'
                      f'{native_rows[0]["frame"]}-{native_rows[-1]["frame"]}')
        x_native, names, player_mask, ball_mask, feature_stats = _feature_arrays(
            native_rows, jerseys, fps)
        if names_global is None:
            names_global = names
        elif names != names_global:
            raise ValueError(f'Feature names differ within {game_id}')
        stats['unknown_jersey_entities'] += feature_stats['missing_jersey_entities']
        stats['duplicate_jersey_entities'] += feature_stats['duplicate_jersey_entities']
        stats['visible_player_observations'] += feature_stats['visible_player_observations']
        stats['estimated_player_entities_ignored'] += feature_stats['estimated_player_entities_ignored']
        stats['visible_ball_observations'] += feature_stats['visible_ball_observations']
        stats['estimated_ball_entities_ignored'] += feature_stats['estimated_ball_entities_ignored']
        stats['entity_coords_outside_nominal_pitch'] += feature_stats['entity_coords_outside_nominal_pitch']
        indices = np.arange(0, len(native_rows), SAMPLE_STRIDE, dtype=np.int64)
        sampled_frame_numbers = np.asarray([native_rows[i]['frame'] for i in indices], dtype=np.int64)
        sampled_timestamps = np.asarray([native_rows[i]['timestamp'] for i in indices], dtype=np.float64)
        y = np.full(len(indices), -1, dtype=np.int8)
        full_y = np.full(len(native_rows), -1, dtype=np.int8)
        full_period = np.full(len(native_rows), native_rows[0]['period'], dtype=np.int32)
        local_by_frame = {row['frame']: i for i, row in enumerate(native_rows)}
        mapped_pending: dict[int, list[dict]] = {}
        native_anchors: dict[int, set[int]] = {}
        for anchor in anchors:
            if anchor['status'] != 'pending' or anchor['period'] != native_rows[0]['period']:
                continue
            local = local_by_frame.get(anchor['frame'])
            if local is None:
                continue
            native_anchors.setdefault(local, set()).add(0 if anchor['team'] == 'home' else 1)
            nearest_j = min(range(len(indices)), key=lambda j: (abs(int(sampled_frame_numbers[j]) - anchor['frame']), j))
            nearest_frame = int(sampled_frame_numbers[nearest_j])
            delta_frames = abs(nearest_frame - int(anchor['frame']))
            mapping = anchor
            mapping.update({'nearest_frame': nearest_frame,
                            'sampled_index': int(nearest_j),
                            'delta_native_frames': delta_frames})
            if delta_frames <= 3:
                mapped_pending.setdefault(nearest_j, []).append(mapping)
                anchor['status'] = 'mapped_candidate'
                anchor['mapped_index'] = int(nearest_j)
                anchor['mapped_frame'] = nearest_frame
                anchor['delta_native_frames'] = delta_frames
            else:
                anchor['status'] = 'out_of_grid_tolerance'
                anchor['nearest_frame'] = nearest_frame
                anchor['delta_native_frames'] = delta_frames
                stats['anchor_out_of_grid_tolerance'] += 1
        for local, teams in native_anchors.items():
            full_y[local] = next(iter(teams)) if len(teams) == 1 else -1
        for sampled_index, mapped in mapped_pending.items():
            target_teams = {0 if anchor['team'] == 'home' else 1 for anchor in mapped}
            if len(target_teams) == 1:
                y[sampled_index] = next(iter(target_teams))
                for anchor in mapped:
                    anchor['status'] = 'mapped'
                    stats['anchor_mapped_known'] += 1
            else:
                for anchor in mapped:
                    anchor['status'] = 'grid_conflict'
                stats['anchor_grid_conflict'] += len(mapped)
        stats['anchor_native_known'] += sum(len(v) == 1 for v in native_anchors.values())
        stats['native_player_slots'] += int(player_mask.size)
        stats['native_ball_rows'] += int(ball_mask.size)
        if not np.isin(y, (0, 1)).any():
            continue
        stats['segments_emitted'] += 1
        record = {
            'x': x_native[indices],
            'y': y,
            'names': list(names),
            'period': np.full(len(indices), segment_number, dtype=np.int32),
            'source': 'soccernet',
            'match': game_prefix,
            'segment_id': str(segment_id),
            'fps': np.asarray(fps / SAMPLE_STRIDE),
            'native_fps': np.asarray(fps),
            'indices': indices,
            'native_n': np.asarray(len(native_rows), dtype=np.int32),
            'full_y': full_y,
            'full_period': full_period,
            'full_frame_numbers': np.asarray([r['frame'] for r in native_rows], dtype=np.int64),
            'full_timestamps': np.asarray([r['timestamp'] for r in native_rows], dtype=np.float64),
            'full_player_observed': player_mask,
            'full_ball_observed': ball_mask,
            'sampled_frame_numbers': sampled_frame_numbers,
            'sampled_timestamps': sampled_timestamps,
            'timestamps': sampled_timestamps.copy(),
            'anchor_frame_mapping': [dict(a) for a in anchors
                                     if a.get('period') == native_rows[0]['period'] and
                                     a.get('frame') in local_by_frame],
        }
        records.append(record)
    stats['records_emitted'] = len(records)
    for anchor in anchors:
        if anchor['status'] == 'pending':
            anchor['status'] = 'center_removed_after_merge'
            stats['anchor_center_removed_after_merge'] += 1
    # Keep samples whose anchor could not map in a private per-game audit list.
    anchor_audit = [dict(anchor) for anchor in anchors]
    return records, stats, anchor_audit, names_global


def _worker_init(tracking_path: str, shard_names: tuple[str, ...]):
    global _ARCHIVES
    _ARCHIVES = {}
    base = Path(tracking_path) / 'train/shards'
    for shard_name in shard_names:
        archive = tarfile.open(base / shard_name, mode='r:')
        archive.getmembers()  # Build each TAR's random-access index once per worker.
        _ARCHIVES[shard_name] = archive


def _process_game(args):
    game_id, samples, tracking_path, output_path = args
    import pyarrow.parquet as pq

    def read_clip(sample):
        archive = _ARCHIVES[sample['shard_name']]
        member = archive.getmember(sample['wds_member'])
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f'Cannot read TAR member {sample["wds_member"]}')
        payload = handle.read()
        table = pq.read_table(io.BytesIO(payload), columns=list(RAW_COLUMNS))
        rows = table.to_pylist()
        del payload, table
        if len(rows) != 144:
            raise ValueError(f'{sample["sample_id"]} has {len(rows)} rows; expected 144')
        for index, row in enumerate(rows):
            if row.get('frameNum') is None or int(row['frameNum']) != sample['start_frame'] + index:
                raise ValueError(f'Frame numbering mismatch in {sample["sample_id"]} at row {index}')
        return rows
    records, stats, anchors, feature_names = _assemble_game(game_id, samples, read_clip)
    output = Path(output_path) / 'games' / f'{game_id}.joblib'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.joblib.tmp')
    joblib.dump({'records': records, 'stats': stats, 'anchors': anchors,
                 'feature_names': feature_names}, temporary, compress=0)
    temporary.replace(output)
    anchor_status_counts = {}
    for anchor in anchors:
        status = anchor.get('status', 'unknown')
        anchor_status_counts[status] = anchor_status_counts.get(status, 0) + 1
    return {'game_id': game_id, 'path': str(output), 'records': len(records),
            'stats': stats, 'anchor_status_counts': anchor_status_counts,
            'feature_names': feature_names, 'sha256': _sha256(output)}


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / 'experiments' / relative)
            for relative in FEATURE_SOURCES}


def prepare(root: str | Path, output: str | Path, workers: int = MAX_WORKERS) -> dict:
    """Prepare all observed TRAIN game segments; never opens valid/test data."""
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be in 1..{MAX_WORKERS}')
    tracking = _tracking_root(root)
    output = Path(output)
    if (output / 'manifest.json').exists():
        cached = _load_manifest(output)
        if cached['acquisition_receipt'] != _verify_acquisition(tracking):
            raise ValueError('Cached segment preparation is bound to different TRAIN source receipts')
        return cached
    if (output / 'games').exists() and any((output / 'games').iterdir()):
        raise FileExistsError('Partial segment cache exists without a manifest')
    receipt = _verify_acquisition(tracking)
    samples, _ = _read_train_samples(tracking)
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample['game_id'], []).append(sample)
    shard_names = tuple(sorted({sample['shard_name'] for sample in samples}))
    if shard_names != tuple(f'shard-{index:06d}.tar' for index in range(5)):
        raise ValueError('Unexpected TRAIN shard cohort')
    compact_groups = {}
    for game_id, game_samples in grouped.items():
        compact_groups[game_id] = sorted(game_samples, key=lambda item: (
            item['start_frame'], item['sample_id']))
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(game_id, game_samples, str(tracking), str(output))
             for game_id, game_samples in sorted(compact_groups.items(),
                 key=lambda item: (int(item[0]) if item[0].isdigit() else item[0]))]
    if workers == 1:
        _worker_init(str(tracking), shard_names)
        results = [_process_game(task) for task in tasks]
        for archive in _ARCHIVES.values():
            archive.close()
    else:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_worker_init,
                                 initargs=(str(tracking), shard_names)) as pool:
            results = list(pool.map(_process_game, tasks, chunksize=1))
    results.sort(key=lambda item: (int(item['game_id']) if item['game_id'].isdigit()
                                   else item['game_id']))
    all_stats = _empty_stats()
    for result in results:
        for key, value in result['stats'].items():
            if isinstance(value, dict):
                destination = all_stats.setdefault(key, {})
                for subkey, count in value.items():
                    destination[subkey] = destination.get(subkey, 0) + count
            else:
                all_stats[key] += value
    # Raw-row and clip counts have distinct meanings; recompute cohort totals.
    all_stats['clips_read'] = len(samples)
    all_stats['raw_rows_read'] = len(samples) * 144
    files = {result['path'].replace(str(output) + os.sep, ''): result['sha256']
             for result in results}
    feature_names = next((result['feature_names'] for result in results
                          if result['feature_names'] is not None), None)
    if feature_names is None:
        raise ValueError('No game produced any feature records')
    if any(result['feature_names'] not in (None, feature_names) for result in results):
        raise ValueError('Feature schema differs across TRAIN games')
    versions = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'pyarrow', 'joblib'):
        versions[package] = importlib.metadata.version(package)
    versions['python'] = platform.python_version()
    manifest = {
        'schema': 'soccernet_observed_segments_v1',
        'dataset': 'OpenSportsLab/SoccerNet-GAR',
        'revision': receipt['revision'], 'branch': 'tracking-full',
        'split_used': 'train', 'organizer_reference_access': False,
        'created_utc': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        'acquisition_receipt': receipt,
        'train_metadata_rows': len(samples),
        'games': [result['game_id'] for result in results],
        'counts': all_stats,
        'game_audits': {result['game_id']: {
            'records': result['records'], 'sha256': result['sha256'],
            'stats': result['stats'],
            'anchor_status_counts': result['anchor_status_counts']}
            for result in results},
        'feature_names': feature_names,
        'feature_schema_sha256': hashlib.sha256(json.dumps(
            feature_names, separators=(',', ':')).encode()).hexdigest(),
        'eligible_anchors': {
            'actions': sorted(ELIGIBLE_ACTIONS),
            'event_team_rule': 'sidecar home/away at center_frame only',
            'unlabeled_rows': -1, 'D_targets': 0,
        },
        'feature_recipe': {
            'raw_columns': list(RAW_COLUMNS),
            'entity_sources': ['homePlayers', 'awayPlayers', 'balls'],
            'estimated_entities': 'ignored; only visibility == VISIBLE contributes coordinates',
            'smoothed_entities': 'not read',
            'coordinate_transform': 'nominal pitch-centered metres to [0,105]x[0,68] by +52.5,+34; no clipping; per-fixture size not documented',
            'feature_apis': ['experiments.refine_features.features', 'experiments.learned_temporal.player_features'],
            'native_feature_rate': 'metadata.conversion.measured_source_fps, one consistent rate per game',
            'sample_stride': SAMPLE_STRIDE,
            'sample_rate_hz': 'fps/6; approximately 4.995 Hz at 29.970 FPS',
            'record_fps': 'native_fps/6, approximately 4.995 Hz; timestamps are sampled source videoTimeMs values',
            'features_before_sampling': True,
            'raw_input_allowlist': ['ball','players','teams','period','fps','timestamps','pitch','ball_observed','player_observed'],
            'excluded_as_features': ['game_event_id','possession_event_id','game_event_type','possession_event_type','player_id','player_name','team_id','home_team','jerseyNum','visibility','confidence','sidecar action/team/game identity'],
        },
        'segment_recipe': {
            'dedup_key': '(game_id, period, frameNum)',
            'duplicate_policy': 'Exact row signature equality across period/time/tracking validity/raw entity arrays and visibility; exact copies retained once; geometry conflicts exclude the key and split the segment; never average. Sparse home_team metadata is compared separately, never changes geometry, and only marks a center anchor unknown.',
            'invalid_tracking_rows': 'structural breaks; not imputed',
            'breaks': ['uncovered frameNum gap','period change','nonmonotonic timestamps','timestamp delta > 1.5/fps','invalid/conflicting row'],
            'anchors': 'nearest sampled source frame within 3 native frames; out-of-tolerance anchors remain in anchor_frame_mapping/full_y and do not receive grid loss; conflicting teams at one grid sample remain unknown',
            'labels': 'sparse H/A actor anchors only; no D, dense labels, forward hold or off-center negatives',
        },
        'source_sha256': _source_hashes(),
        'dependency_versions': versions,
        'game_files': files,
    }
    manifest_path = output / 'manifest.json'
    temporary_manifest = output / 'manifest.json.tmp'
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary_manifest.replace(manifest_path)
    return manifest


def _load_manifest(output: str | Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('schema') != 'soccernet_observed_segments_v1' or manifest.get('split_used') != 'train':
        raise ValueError('Unsupported or non-TRAIN segment manifest')
    if manifest.get('organizer_reference_access') is not False:
        raise ValueError('Manifest organizer-reference assertion changed')
    for relative, expected in manifest['game_files'].items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'Segment game-cache seal mismatch: {relative}')
    if manifest['source_sha256'] != _source_hashes():
        raise ValueError('Feature/preparation source hashes changed')
    return manifest


def load_records(output: str | Path) -> list[dict]:
    """Load sealed per-game files in deterministic official TRAIN game order."""
    output = Path(output)
    manifest = _load_manifest(output)
    records = []
    for relative in manifest['game_files']:
        payload = joblib.load(output / relative)
        records.extend(payload['records'])
    records.sort(key=lambda record: (
        int(str(record['match']).removeprefix('soccernet_')),
        int(record['full_period'][0]) if len(record['full_period']) else -1,
        int(record['full_frame_numbers'][0]) if len(record['full_frame_numbers']) else -1))
    return records


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/independent/soccernet_gar')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data/experiments/soccernet_segments_v1')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    result = prepare(args.root, args.output, args.workers)
    print(json.dumps({'output': str(args.output), 'games': len(result['games']),
                      'train_metadata_rows': result['train_metadata_rows'],
                      'counts': result['counts']}, indent=2))


if __name__ == '__main__':
    main()
