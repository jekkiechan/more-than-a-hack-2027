"""Fail closed before organizer label access when check artifacts are incomplete."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import live_attribution_challenge as lc


class LiveAttributionChallengeTests(unittest.TestCase):
    def test_missing_inputs_fail_before_model_or_reference_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(lc, 'frozen_models') as models, patch.object(lc.temporal_challenge, 'check') as check:
                with self.assertRaises(FileNotFoundError):
                    lc.generate(root/'out', root/'missing.npz')
                models.assert_not_called()
                check.assert_not_called()

    def test_partial_comparison_fails_before_public_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'challenge'
            target.mkdir()
            (target/'frozen.json').write_text(json.dumps(dict(csv_sha256={'live_only': 'unused'})))
            with patch.object(lc, 'REPORT', root/'result.json'), patch.object(lc.temporal_challenge, 'check') as check:
                with self.assertRaisesRegex(ValueError, 'full declared pair'):
                    lc.check(root)
                check.assert_not_called()

    def test_changed_source_fails_before_public_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'challenge'
            target.mkdir()
            frozen = dict(csv_sha256={name: 'unused' for name in lc.NAMES},
                          source_sha256={'evaluation/possession_metrics.py': 'wrong-hash'})
            (target/'frozen.json').write_text(json.dumps(frozen))
            with patch.object(lc, 'REPORT', root/'result.json'), patch.object(lc.temporal_challenge, 'check') as check:
                with self.assertRaisesRegex(ValueError, 'source/report changed'):
                    lc.check(root)
                check.assert_not_called()

    def test_completed_report_cannot_be_rescored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root/'result.json'
            report.write_text('{}')
            with patch.object(lc, 'REPORT', report), patch.object(lc.temporal_challenge, 'check') as check:
                with self.assertRaisesRegex(ValueError, 'do not repeat'):
                    lc.check(root)
                check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
