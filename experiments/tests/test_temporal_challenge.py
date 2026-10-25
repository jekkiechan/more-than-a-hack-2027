"""Check that challenge guards fail before any public target access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import learned_temporal as lt, temporal_challenge as tc


class ChallengeGuards(unittest.TestCase):
    def test_missing_assets_do_not_open_public_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'pooled_hgb.joblib').write_bytes(b'not-loaded-without-inputs')
            (out/'attribution.joblib').write_bytes(b'not-loaded-without-inputs')
            (out/'predeclared.json').write_text('{}')
            selection = dict(selected_model='hgb', selected_model_sha256=lt.digest(out/'pooled_hgb.joblib'),
                             attribution_model_sha256=lt.digest(out/'attribution.joblib'),
                             predeclared_spec_sha256=lt.digest(out/'predeclared.json'))
            (out/'selection.json').write_text(json.dumps(selection))
            (out/'pooled_hgb_frozen.json').write_text(json.dumps(dict(
                model_sha256=selection['selected_model_sha256'],
                spec_sha256=selection['predeclared_spec_sha256'])))
            with patch.object(tc.challenge, 'load_challenge', side_effect=FileNotFoundError('no assets')):
                with patch.object(tc.pd, 'read_csv') as read_gt:
                    with self.assertRaises(FileNotFoundError):
                        tc.generate(out, out/'missing.npz')
                    read_gt.assert_not_called()
            self.assertFalse((out/'challenge/public_check_started.json').exists())

    def test_changed_prediction_fails_before_public_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            target = out/'challenge'
            target.mkdir()
            candidate = target/'selected.csv'
            candidate.write_text('frame,label\n0,H\n')
            original_hash = lt.digest(candidate)
            (target/'frozen.json').write_text(json.dumps({'csv_sha256': {'selected': original_hash}}))
            candidate.write_text('frame,label\n0,D\n')
            with patch.object(tc.pd, 'read_csv') as read_gt:
                with self.assertRaisesRegex(ValueError, 'Frozen predictions changed'):
                    tc.check(out, None)
                read_gt.assert_not_called()
            self.assertFalse((target/'public_check_started.json').exists())

    def test_all_declared_candidates_are_checked_once(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            target = out/'challenge'
            target.mkdir()
            hashes = {}
            for name in ('selected', 'tcn'):
                path = target/f'{name}.csv'
                path.write_text('frame,label\n0,H\n1,A\n2,D\n')
                hashes[name] = lt.digest(path)
            (target/'frozen.json').write_text(json.dumps({'csv_sha256': hashes}))
            truth = tc.pd.DataFrame({'frame': [0, 1, 2], 'label': ['H', 'A', 'D']})
            original_read = tc.pd.read_csv
            def read(path):
                return truth if Path(path).name == 'BAR-ATH_possession_gt_public.csv' else original_read(path)
            with patch.object(tc.pd, 'read_csv', side_effect=read):
                tc.check(out, None)
                with self.assertRaises(FileExistsError):
                    tc.check(out, None)
            result = json.loads((target/'public_check.json').read_text())
            self.assertEqual(set(result['scores']), {'selected', 'tcn'})
            self.assertEqual(result['scores']['tcn']['macro_f1'], 1.)


if __name__ == '__main__':
    unittest.main()
