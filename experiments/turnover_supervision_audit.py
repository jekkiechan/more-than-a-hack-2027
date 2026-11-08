"""TRAIN-only inventory of native H/A turnover supervision and sampler exposure.

This is a descriptive audit. It reads the 13 original TRAIN full-native caches
and the 45 GAR TRAIN records, never fits or scores a model, and emits aggregates
only. A turnover requires adjacent H/A labels to change. Context requires stable
same-team H/A labels for the requested duration immediately before and after.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H, A, D, UNK = 0, 1, 2, -1
FPS_NATIVE = 25.0
CONTEXT_SECONDS = (1, 2)
SHORT_SECONDS = 3
CORE_SECONDS = 10


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _chunks(y: np.ndarray, period: np.ndarray, fps: float):
    """Return maximal contiguous valid native blocks, honoring explicit breaks."""
    y, period = np.asarray(y), np.asarray(period)
    if y.ndim != 1 or period.shape != y.shape:
        raise ValueError('labels and period must be aligned 1D arrays')
    if not np.isin(y, (-1, 0, 1, 2)).all():
        raise ValueError('unexpected H/A/D/unknown encoding')
    # Full native arrays in these sources are frame-contiguous within a period.
    cuts = np.flatnonzero(np.r_[True, period[1:] != period[:-1], True])
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def inventory(records: list[dict], *, kind: str, fps_key: str = 'native_fps') -> dict:
    if kind not in ('original', 'gar'):
        raise ValueError('kind must be original or gar')
    spans = {str(SHORT_SECONDS): 0}
    span_rows = {str(SHORT_SECONDS): 0}
    total_edges = 0
    edges = {str(s): 0 for s in CONTEXT_SECONDS}
    central_core = 0
    central_core_by_position = {'first_quarter': 0, 'middle_half': 0, 'last_quarter': 0}
    runs = 0
    valid_ha_rows = dead_rows = unknown_rows = 0
    for record in records:
        y = np.asarray(record.get('full_y', record.get('y')), dtype=np.int8)
        period = np.asarray(record.get('full_period', record.get('period')))
        fps = float(record.get(fps_key, record.get('fps', FPS_NATIVE)))
        sample_indices = np.asarray(record.get('indices', np.arange(len(y))), dtype=np.int64)
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError('invalid native fps')
        valid_ha_rows += int(np.isin(y, (H, A)).sum())
        dead_rows += int((y == D).sum())
        unknown_rows += int((y == UNK).sum())
        for lo, hi in _chunks(y, period, fps):
            run_starts = np.r_[lo, np.flatnonzero(y[lo+1:hi] != y[lo:hi-1]) + lo + 1]
            run_stops = np.r_[run_starts[1:], hi]
            # Complete short possessions have known boundaries on both sides;
            # unknown ends and period/record ends are censored.
            for i, (a, b) in enumerate(zip(run_starts, run_stops)):
                val = int(y[a])
                if val not in (H, A):
                    continue
                runs += 1
                duration = (b-a)/fps
                complete = a > lo and b < hi and y[a-1] != UNK and y[b] != UNK
                if complete and duration <= SHORT_SECONDS:
                    spans[str(SHORT_SECONDS)] += 1
                    span_rows[str(SHORT_SECONDS)] += b-a
            for edge in range(lo, hi-1):
                left, right = int(y[edge]), int(y[edge+1])
                if left not in (H, A) or right not in (H, A) or left == right:
                    continue
                total_edges += 1
                qualifies = {}
                for seconds in CONTEXT_SECONDS:
                    n = int(round(seconds*fps))
                    qualifies[seconds] = (edge+1-lo >= n and hi-(edge+1) >= n and
                        bool(np.all(y[edge+1-n:edge+1] == left)) and
                        bool(np.all(y[edge+1:edge+1+n] == right)))
                    if qualifies[seconds]:
                        edges[str(seconds)] += 1
                # Shared trainer uses 50 sampled rows per core. Apply that
                # geometry on each source's actual native-to-sampled indices.
                sample_lo = int(np.searchsorted(sample_indices, lo, side='left'))
                sample_hi = int(np.searchsorted(sample_indices, hi, side='left'))
                sample_edge = int(np.searchsorted(sample_indices, edge+1, side='left'))
                core_start = sample_lo + max(0, (sample_edge-sample_lo-1)//50)*50
                core_stop = min(core_start+50, sample_hi)
                left_boundary = int(sample_indices[core_start]) if core_start < sample_hi else lo
                right_boundary = int(sample_indices[core_stop]) if core_stop < sample_hi else hi
                central = edge+1-left_boundary >= fps and right_boundary-(edge+1) >= fps
                if central and qualifies[1]:
                    central_core += 1
                    rel = (sample_edge-core_start)/max(1, core_stop-core_start)
                    bucket = ('first_quarter' if rel < .25 else
                              'last_quarter' if rel >= .75 else 'middle_half')
                    central_core_by_position[bucket] += 1
    return {
        'records': len(records), 'valid_ha_native_rows': valid_ha_rows,
        'dead_native_rows': dead_rows, 'unknown_native_rows': unknown_rows,
        'constant_ha_runs': runs, 'complete_constant_ha_spans_le_3s': spans['3'],
        'native_rows_in_complete_spans_le_3s': span_rows['3'],
        'adjacent_ha_turnovers': total_edges,
        'turnovers_with_stable_same_team_context_both_sides': edges,
        'turnovers_with_1s_stable_context_inside_50_sample_core': central_core,
        'central_core_edge_position_counts': central_core_by_position,
    }


def load_train() -> tuple[list[dict], list[dict], dict]:
    from experiments import preparation
    data = Path(os.environ.get('MTAH_DATA_DIR', '/workspace/mtah-data')).resolve()
    sqd = Path(os.environ.get('MTAH_SQD_DIR', '/workspace/mtah-sqd')).resolve()
    dt = Path(os.environ.get('MTAH_DT_DIR', '/workspace/mtah-dt')).resolve()
    original_manifest = data/'experiments/objective_alignment_v1/predeclared.json'
    spec = json.loads(original_manifest.read_text())
    plan = preparation.split_plan(fold='pooled')['train']
    if len(plan) != 13:
        raise ValueError('expected exactly 13 original TRAIN matches')
    old_records, old_hashes = [], {}
    for item in plan:
        entry = spec['prepared_training_caches'][item['match']]
        path = data/Path(entry['cache']).relative_to('data')
        digest = source_hash(path)
        if digest != entry['sha256']:
            raise ValueError(f'original TRAIN cache hash mismatch: {item["match"]}')
        cache = joblib.load(path)
        if cache['match'] != item['match'] or cache['source'] != item['source']:
            raise ValueError('original TRAIN cache identity mismatch')
        old_records.append({'source': cache['source'], 'match': cache['match'],
                            'full_y': cache['full_y'], 'full_period': cache['full_period'],
                            'fps': cache['fps'], 'indices': cache['records']['clean']['indices']})
        old_hashes[item['match']] = digest
    gar_root = dt/'experiments/dense_targeted_followups_v1/artifacts/train_records'
    manifest = json.loads((gar_root/'manifest.json').read_text())
    if len(manifest['game_files']) != 45:
        raise ValueError('expected exactly 45 GAR TRAIN games')
    gar_records = []
    gar_hashes = {}
    for name, expected in sorted(manifest['game_files'].items()):
        path = gar_root/'games'/name
        digest = source_hash(path)
        if digest != expected:
            raise ValueError(f'GAR TRAIN cache hash mismatch: {name}')
        payload = joblib.load(path)
        for record in payload['records']:
            gar_records.append(record)
        gar_hashes[name] = digest
    provenance = {
        'original_cache_manifest_sha256': source_hash(original_manifest),
        'original_train_cache_sha256': old_hashes,
        'gar_train_manifest_sha256': source_hash(gar_root/'manifest.json'),
        'gar_train_game_sha256': gar_hashes,
        'warmstart_paths_present': all((sqd/'experiments/short_quality_diversity_delivery_v1/artifacts/short_original_diversity_v1/models'/f'leave_out_{p}_{seed}.joblib').is_file()
            for p in ('metrica','skillcorner','sportec') for seed in (2026,2027)),
    }
    return old_records, gar_records, provenance


def main() -> None:
    old, gar, provenance = load_train()
    short_audit_path = ROOT/'experiments/results/short_original_diversity_v1_training_audit.json'
    short_audit = json.loads(short_audit_path.read_text())
    old_sampler = []
    for row in short_audit['rows']:
        sample = row['sampling']
        old_sampler.append({
            'fold': row['fold'], 'seed': row['seed'],
            'eligible_windows_with_short_ha': sample['eligible_windows']['old'],
            'paired_record_possession_candidates': sample['original_eligible_possessions_including_paired_records'],
            'paired_record_nominated_possessions': sample['original_nominated_possessions_including_paired_records'],
            'replacement_draws': sample['counts']['old']['replacement_draws'],
            'short_containing_draws': sample['counts']['old']['short_containing_draws'],
            'total_draws': sample['counts']['old']['draws'],
        })
    report = {
        'study': 'turnover_supervision_audit',
        'scope': 'TRAIN only: 13 original-provider matches and 45 GAR games',
        'label_encoding': {'H': 0, 'A': 1, 'D': 2, 'unknown': -1},
        'limitations': [
            'Stable edge context is based on label continuity and does not prove the whole possession was observed or independently annotated.',
            'Original prepared full-native caches expose labels, period IDs and FPS but no native timestamp array; adjacency follows the preserved cache row order and period blocks. GAR records were already segmented at native frame and timestamp discontinuities.',
            'Core-center counts are geometry diagnostics for the existing 50-sample core sampler; they do not imply a turnover-specific sampler was previously used.',
        ],
        'definition': {
            'edge': 'Adjacent native H/A rows with different teams in same record and period. Stable context does not establish a fully observed possession spell.',
            'context': 'Immediately preceding and following 1s/2s must be uninterrupted H/A rows, all equal to their respective edge-side team; D, unknown, another turnover, period or record break invalidates context. This is a label-continuity inventory, not independent annotation.',
            'short_span': 'Maximal constant-team H/A run <=3s with known boundaries; period/record ends and unknown-boundary spans are censored.',
            'sampling': 'The shared trainer uses nonoverlapping 50-sample cores and 25-sample context; these correspond to about 10s on original features and 12s on GAR sampled features. Counts separately show stable edges with >=1s native context inside the actual 50-sample core. Short sampler uses these existing core windows and changes access frequency. For future within-record sampling, deduplicate by (source, match, native start, native stop, team); map native span endpoints through each paired record indices array to existing windows. Keep all paired records in the unchanged sampler, or define identity on native coordinates so they cannot masquerade as new events.',
        },
        'original': inventory(old, kind='original', fps_key='fps'),
        'original_by_provider': {source: inventory([r for r in old if str(r['source']).lower() == source], kind='original', fps_key='fps')
                                 for source in sorted({str(r['source']).lower() for r in old})},
        'original_by_leave_out_fold': {
            f'leave_out_{provider}': inventory([r for r in old if str(r['source']).lower() != provider], kind='original', fps_key='fps')
            for provider in ('metrica', 'skillcorner', 'sportec')},
        'gar': inventory(gar, kind='gar'),
        'existing_original_short_sampler': {
            'receipt_sha256': source_hash(short_audit_path),
            'schedule_rows': old_sampler,
            'interpretation': 'Historical distinct <=3s possession sampler: approximately 25% of old draws are replacement requests; distinct possession/window bag counts repeat each event across paired views. Use native event identity for any turnover sampler.',
        },
        'input_hashes': provenance,
        'audit_code_sha256': source_hash(Path(__file__)),
        'accessed_validation_test_or_organizer_labels': False,
        'fit_or_candidate_score_run': False,
    }
    out = ROOT/'experiments/results/turnover_supervision_audit.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              default=lambda value: int(value) if isinstance(value, np.integer) else value)+'\n')
    print(json.dumps({k: report[k] for k in ('original','gar')}, indent=2,
                     default=lambda value: int(value) if isinstance(value, np.integer) else value))


if __name__ == '__main__':
    main()
