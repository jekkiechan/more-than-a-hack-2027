"""Dense silver H/A/D labels for SoccerNet-GAR ``tracking-full`` (recipe v1).

Reuses the frozen ``soccernet_segments`` merge / dedupe / segment / 204-feature
path (without editing it) and adds per-frame labels derived from the native
event stream columns (game_event_type, game_event_id, possession_event_type,
home_team). Works on train, valid and test; it never fits or scores a model and
never reads organizer public ground truth. The recipe is documented in
``experiments/SOCCERNET_DENSE.md``.

Label encoding (identical to every other record in the pipeline):
``0`` = home (H), ``1`` = away (A), ``2`` = dead ball (D), ``-1`` = unknown.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import multiprocessing
import os
import tarfile
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import joblib
import numpy as np

from experiments import soccernet_segments as seg

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'valid', 'test')
DEFAULT_RAW = ROOT / 'data/independent/soccernet_gar'
DEFAULT_OUT = ROOT / 'data/experiments/sn_dense_v1/records'
SAMPLE_STRIDE = seg.SAMPLE_STRIDE
MAX_WORKERS = 3
SCHEMA = 'soccernet_dense_v1'

# Label codes.
H, A, D, UNK = 0, 1, 2, -1
# Per-frame provenance codes.
P_UNKNOWN, P_TOUCH, P_PROPAGATED, P_DEAD, P_HELD, P_ANCHOR = 0, 1, 2, 3, 4, 5
PROVENANCE_NAMES = {P_UNKNOWN: 'unknown', P_TOUCH: 'touch', P_PROPAGATED: 'propagated',
                    P_DEAD: 'dead', P_HELD: 'held_uncontrolled', P_ANCHOR: 'anchor_only'}

TOUCH_TYPES = ('OTB',)          # plus any type containing KICKOFF
UNCONTROLLED_END = frozenset(('CH', 'RE'))
UNCONTROLLED_SINGLE_END = frozenset(('CH', 'RE', 'CL'))
OUT_TYPE = 'OUT'
TAIL_GRACE_SECONDS = 3.0        # owner survives to a following OUT only within this

RECORD_RUN_COLUMNS = ('start', 'end', 'team', 'end_code', 'controlled', 'censored')
END_CODES = {'': 0, 'PA': 1, 'CH': 2, 'RE': 3, 'SH': 4, 'CR': 5, 'BC': 6, 'CL': 7}


# --------------------------------------------------------------------------- #
# Frozen-path reuse: patch the frozen game assembler's source in memory.
# --------------------------------------------------------------------------- #
def _patched_assemble():
    """Return the frozen ``_assemble_game`` with four asserted textual hooks.

    The frozen module is not modified. Hooks: (1) carry event columns on each
    parsed row, (2) count event-column conflicts among duplicate frames,
    (3) emit every contiguous segment (not only those with sparse anchors),
    (4) expose per-frame event arrays on each record for labelling.
    """
    source = textwrap.dedent(inspect.getsource(seg._assemble_game))
    patches = [
        ("            parsed = _native_row(row)\n",
         "            parsed = _native_row(row)\n"
         "            parsed['ev'] = _event_tuple(row)\n"),
        ("            key = (parsed['period'], parsed['frame'])\n"
         "            if parsed['home_team'] is not None:\n",
         "            key = (parsed['period'], parsed['frame'])\n"
         "            _seen = stats.setdefault('_ev_seen', {})\n"
         "            _first = _seen.setdefault(key, (parsed['ev'], parsed['home_team']))\n"
         "            if _first != (parsed['ev'], parsed['home_team']):\n"
         "                stats['event_column_conflicts'] = stats.get('event_column_conflicts', 0) + 1\n"
         "            if parsed['home_team'] is not None:\n"),
        ("        if not np.isin(y, (0, 1)).any():\n            continue\n", ""),
        ("        records.append(record)\n",
         "        record['_ev'] = [r['ev'] for r in native_rows]\n"
         "        record['_home_flag'] = [r['home_team'] for r in native_rows]\n"
         "        records.append(record)\n"),
    ]
    for old, new in patches:
        if source.count(old) != 1:
            raise RuntimeError('Frozen soccernet_segments._assemble_game changed; '
                               'dense-label hooks no longer apply: ' + old.strip()[:60])
        source = source.replace(old, new)
    namespace = dict(vars(seg))
    namespace['_event_tuple'] = _event_tuple
    exec(compile(source, 'soccernet_segments._assemble_game(dense-patched)', 'exec'), namespace)
    return namespace['_assemble_game']


def _clean(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and value != value:
        return ''
    return str(value)


def _event_tuple(row: dict) -> tuple:
    """(game_event_type, game_event_id, possession_event_type) with NaN -> ''/-1."""
    try:
        event_id = int(row.get('game_event_id'))
    except (TypeError, ValueError):
        event_id = -1
    return (_clean(row.get('game_event_type')), event_id, _clean(row.get('possession_event_type')))


# --------------------------------------------------------------------------- #
# Dense labelling (pure function of the native event stream of one record).
# --------------------------------------------------------------------------- #
def _is_touch(event_type: str) -> bool:
    return event_type in TOUCH_TYPES or 'KICKOFF' in event_type


def touch_runs(events: list[tuple], home_flags: list, treat_censored_as_controlled=True):
    """Group consecutive same-game_event_id touch frames into runs.

    Returns (runs, out_frames). Each run is a dict with start/end (inclusive
    native record index), team (0/1/None), end_type, controlled, censored.
    """
    n = len(events)
    runs, out_frames = [], []
    i = 0
    while i < n:
        event_type, event_id, _ = events[i]
        if event_type == OUT_TYPE:
            out_frames.append(i)
            i += 1
            continue
        if not _is_touch(event_type):
            i += 1
            continue
        j = i
        while (j + 1 < n and _is_touch(events[j + 1][0]) and events[j + 1][1] == event_id
               and (event_id != -1 or events[j + 1][0] == event_type)):
            j += 1
        flags = {home_flags[k] for k in range(i, j + 1) if home_flags[k] in ('0', '1')}
        team = None
        if len(flags) == 1:
            team = 0 if next(iter(flags)) == '1' else 1
        end_type = ''
        for k in range(j, i - 1, -1):
            if events[k][2]:
                end_type = events[k][2]
                break
        length = j - i + 1
        censored = (j == n - 1) and not end_type
        uncontrolled = (end_type in UNCONTROLLED_END or
                        (length == 1 and end_type in UNCONTROLLED_SINGLE_END))
        if censored and not treat_censored_as_controlled:
            uncontrolled = True
        runs.append({'start': i, 'end': j, 'team': team, 'end_type': end_type,
                     'controlled': not uncontrolled, 'censored': bool(censored)})
        i = j + 1
    return runs, out_frames


def dense_labels(events: list[tuple], home_flags: list, fps: float,
                 uncontrolled_opponent: str = 'hold'):
    """Native-frame dense labels and provenance for one contiguous record.

    ``uncontrolled_opponent``: 'hold' (default) labels frames inside an
    uncontrolled opponent run (CH/RE end, or 1-frame CH/RE/CL) with the
    current owner (provenance P_HELD); 'literal' labels them with the touching
    team (provenance P_TOUCH), which is the raw "inside a run: that team" rule.
    """
    if uncontrolled_opponent not in ('hold', 'literal'):
        raise ValueError('uncontrolled_opponent must be hold or literal')
    n = len(events)
    y = np.full(n, UNK, dtype=np.int8)
    prov = np.zeros(n, dtype=np.int8)
    runs, out_frames = touch_runs(events, home_flags)
    items = sorted([(r['start'], 0, r) for r in runs] + [(o, 1, o) for o in out_frames],
                   key=lambda t: (t[0], t[1]))
    owner = None
    dead = False
    cursor = 0
    grace = int(round(TAIL_GRACE_SECONDS * fps))

    def fill(lo, hi, value, code):
        if hi > lo:
            y[lo:hi] = value
            prov[lo:hi] = code

    for position, kind, item in items:
        if kind == 1:                                   # OUT frame
            if dead:
                continue
            if owner is not None and position - cursor <= grace:
                fill(cursor, position, owner, P_PROPAGATED)
            cursor = position
            dead = True
            owner = None
            continue
        run = item
        start, end, team = run['start'], run['end'], run['team']
        if dead:
            fill(cursor, start, D, P_DEAD)
            dead = False
            owner = team                                # restart: next toucher owns
            if team is not None:
                fill(start, end + 1, team, P_TOUCH)
        elif owner is None:
            # Head of record, or after an unknown-team run: gap stays unknown.
            if team is not None and run['controlled']:
                owner = team
                fill(start, end + 1, team, P_TOUCH)
        elif team is None:
            fill(cursor, start, owner, P_PROPAGATED)    # ball was still with owner
            owner = None
        elif team == owner:
            fill(cursor, start, owner, P_PROPAGATED)
            fill(start, end + 1, owner, P_TOUCH)
        elif run['controlled']:
            fill(cursor, start, owner, P_PROPAGATED)
            owner = team
            fill(start, end + 1, team, P_TOUCH)
        else:                                           # uncontrolled opponent touch
            fill(cursor, start, owner, P_PROPAGATED)
            if uncontrolled_opponent == 'hold':
                fill(start, end + 1, owner, P_HELD)
            else:
                fill(start, end + 1, team, P_TOUCH)
        cursor = end + 1
    if dead:
        fill(cursor, n, D, P_DEAD)
    return y, prov, runs, out_frames


def run_table(runs: list[dict]) -> np.ndarray:
    table = np.full((len(runs), len(RECORD_RUN_COLUMNS)), -1, dtype=np.int32)
    for k, r in enumerate(runs):
        table[k] = (r['start'], r['end'], -1 if r['team'] is None else r['team'],
                    END_CODES.get(r['end_type'], 9), int(r['controlled']), int(r['censored']))
    return table


def finalize_record(record: dict, fps_native: float) -> dict | None:
    """Replace sparse labels by dense ones; keep sparse anchors as ``anchor_*``."""
    events = record.pop('_ev')
    home_flags = record.pop('_home_flag')
    n = len(events)
    full_y, full_prov, runs, out_frames = dense_labels(events, home_flags, fps_native, 'hold')
    literal_y, _, _, _ = dense_labels(events, home_flags, fps_native, 'literal')
    anchor_full = record['full_y'].copy()
    # Sparse anchors only fill frames the event recipe left unknown.
    fill_anchor = (full_y < 0) & (anchor_full >= 0)
    full_y[fill_anchor] = anchor_full[fill_anchor]
    full_prov[fill_anchor] = P_ANCHOR
    literal_y[(literal_y < 0) & (anchor_full >= 0)] = anchor_full[(literal_y < 0) & (anchor_full >= 0)]
    if not (full_y >= 0).any():
        return None
    indices = record['indices']
    anchor_sampled = record['y'].copy()
    n_samples = len(indices)
    record['anchor_y'] = anchor_sampled
    record['anchor_full_y'] = anchor_full
    record['y'] = full_y[indices].astype(np.int8)
    record['prov'] = full_prov[indices].astype(np.int8)
    record['full_y'] = full_y.astype(np.int8)
    record['full_prov'] = full_prov.astype(np.int8)
    record['full_y_literal'] = literal_y.astype(np.int8)
    record['game_period'] = int(record['full_period'][0])
    # transition_targets / spell_duration_factors require period == full_period[indices].
    block = np.full(n, int(record['period'][0]), dtype=np.int32)
    record['full_period'] = block
    record['period'] = block[indices].astype(np.int32)
    record['touch_runs'] = run_table(runs)
    record['touch_run_columns'] = list(RECORD_RUN_COLUMNS)
    record['out_frames'] = np.asarray(out_frames, dtype=np.int64)
    record['full_game_event_type'] = np.asarray(
        [0 if e[0] == '' else (1 if e[0] == 'OTB' else 2 if e[0] == OUT_TYPE else 3) for e in events],
        dtype=np.int8)
    record['label_schema'] = SCHEMA
    record['label_encoding'] = {'H': 0, 'A': 1, 'D': 2, 'unknown': -1}
    record['provenance_encoding'] = {v: k for k, v in PROVENANCE_NAMES.items()}
    assert len(record['y']) == n_samples
    return record


# --------------------------------------------------------------------------- #
# Split-aware data path (the frozen reader is hard-wired to train).
# --------------------------------------------------------------------------- #
def tracking_root(root) -> Path:
    root = Path(root)
    return root / 'tracking-full' if (root / 'tracking-full').is_dir() else root


def read_split_samples(tracking: Path, split: str):
    import pyarrow.parquet as pq
    raw_rows = pq.read_table(tracking / split / 'metadata.parquet', columns=[
        'sample_id', 'sample_index', 'shard_name', 'sample_payload']).to_pylist()
    samples = [seg._parse_payload(row) for row in raw_rows]
    if any(not s['game_id'] for s in samples):
        raise ValueError('A sample has no game_id')
    manifest = pq.read_table(tracking / split / 'shard_manifest.parquet', columns=[
        'sample_id', 'shard_name', 'wds_member', 'input_type', 'status']).to_pylist()
    members = {str(r['sample_id']): r for r in manifest
               if r['input_type'] == 'tracking_parquet' and r['status'] == 'ok'}
    if {s['sample_id'] for s in samples} != set(members):
        raise ValueError('metadata and shard manifest sample IDs differ')
    for s in samples:
        if members[s['sample_id']]['shard_name'] != s['shard_name']:
            raise ValueError(f'Shard mismatch for {s["sample_id"]}')
        s['wds_member'] = str(members[s['sample_id']]['wds_member'])
    return samples


EXTRA_COLUMNS = ('game_event_type', 'game_event_id', 'possession_event_type')
_ARCHIVES: dict[str, tarfile.TarFile] = {}


def _worker_init(tracking_path: str, split: str, shard_names):
    global _ARCHIVES
    _ARCHIVES = {}
    for name in shard_names:
        archive = tarfile.open(Path(tracking_path) / split / 'shards' / name, mode='r:')
        archive.getmembers()
        _ARCHIVES[name] = archive


def _process_game(args):
    game_id, samples, split, output = args
    import pyarrow.parquet as pq
    columns = list(seg.RAW_COLUMNS) + list(EXTRA_COLUMNS)

    def read_clip(sample):
        archive = _ARCHIVES[sample['shard_name']]
        handle = archive.extractfile(archive.getmember(sample['wds_member']))
        table = pq.read_table(io.BytesIO(handle.read()), columns=columns)
        rows = table.to_pylist()
        del table
        if len(rows) != 144:
            raise ValueError(f'{sample["sample_id"]} has {len(rows)} rows')
        for index, row in enumerate(rows):
            if row.get('frameNum') is None or int(row['frameNum']) != sample['start_frame'] + index:
                raise ValueError(f'Frame numbering mismatch in {sample["sample_id"]}')
        return rows

    assemble = _patched_assemble()
    records, stats, anchors, feature_names = assemble(game_id, samples, read_clip)
    fps = float(np.median([s['fps'] for s in samples]))
    stats.pop('_ev_seen', None)
    dense = []
    for record in records:
        finished = finalize_record(record, fps)
        if finished is not None:
            dense.append(finished)
    sample_meta = [{k: s[k] for k in ('sample_id', 'action', 'team', 'start_frame',
                                      'center_frame', 'period_hint')} for s in samples]
    path = Path(output) / split / 'games' / f'{game_id}.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.joblib.tmp')
    joblib.dump({'records': dense, 'stats': stats, 'anchors': anchors,
                 'feature_names': feature_names, 'samples': sample_meta, 'fps': fps},
                tmp, compress=0)
    tmp.replace(path)
    return {'game_id': game_id, 'path': str(path), 'records': len(dense),
            'segments_total': len(records), 'stats': stats,
            'feature_names': feature_names}


def prepare_split(split: str, root=DEFAULT_RAW, output=DEFAULT_OUT, workers=MAX_WORKERS,
                  games: list[str] | None = None) -> dict:
    if split not in SPLITS:
        raise ValueError(split)
    tracking = tracking_root(root)
    samples = read_split_samples(tracking, split)
    grouped: dict[str, list[dict]] = {}
    for s in samples:
        grouped.setdefault(s['game_id'], []).append(s)
    shard_names = tuple(sorted({s['shard_name'] for s in samples}))
    ids = sorted(grouped, key=lambda g: int(g) if g.isdigit() else g)
    if games is not None:
        ids = [g for g in ids if g in set(games)]
    tasks = [(g, sorted(grouped[g], key=lambda s: (s['start_frame'], s['sample_id'])),
              split, str(output)) for g in ids]
    if workers == 1:
        _worker_init(str(tracking), split, shard_names)
        results = [_process_game(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(str(tracking), split, shard_names)) as pool:
            results = list(pool.map(_process_game, tasks, chunksize=1))
    names = results[0]['feature_names']
    if any(r['feature_names'] != names for r in results):
        raise ValueError('feature schema differs across games')
    manifest = {
        'schema': SCHEMA, 'split': split, 'games': [r['game_id'] for r in results],
        'n_samples': len(samples), 'feature_names': names,
        'feature_schema_sha256': hashlib.sha256(json.dumps(
            names, separators=(',', ':')).encode()).hexdigest(),
        'game_files': {Path(r['path']).name: seg._sha256(Path(r['path'])) for r in results},
        'game_records': {r['game_id']: r['records'] for r in results},
        'source_sha256': seg._source_hashes(),
        'dense_module_sha256': seg._sha256(Path(__file__)),
    }
    out = Path(output) / split
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def load_dense_records(split: str, output=DEFAULT_OUT, with_payload: bool = False):
    """Load prepared dense records for ``split`` in deterministic order.

    Each record is a dict; see experiments/SOCCERNET_DENSE.md for keys. Labels:
    0=H, 1=A, 2=D, -1=unknown (``y`` at ~5 Hz, ``full_y`` at native rate,
    ``indices`` maps y -> full_y).
    """
    out = Path(output) / split
    manifest = json.loads((out / 'manifest.json').read_text())
    records, payloads = [], []
    for name, digest in manifest['game_files'].items():
        path = out / 'games' / name
        if seg._sha256(path) != digest:
            raise ValueError(f'seal mismatch: {path}')
        payload = joblib.load(path)
        records.extend(payload['records'])
        if with_payload:
            payloads.append(payload)
    records.sort(key=lambda r: (int(str(r['match']).removeprefix('soccernet_')),
                                int(r['game_period']), int(r['full_frame_numbers'][0])))
    return (records, payloads) if with_payload else records


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--splits', nargs='+', default=list(SPLITS))
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--games', nargs='*', default=None)
    args = parser.parse_args()
    for split in args.splits:
        m = prepare_split(split, args.root, args.output, args.workers, args.games)
        print(split, len(m['games']), 'games', sum(m['game_records'].values()), 'records')


if __name__ == '__main__':
    main()
