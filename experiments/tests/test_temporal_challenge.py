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


if __name__ == '__main__':
    unittest.main()
