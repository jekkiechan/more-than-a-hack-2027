"""Public-gate contracts using synthetic paths only; never open real labels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import short_quality_public_v1 as study


class FixedPublicGateContracts(unittest.TestCase):
    def test_exclusive_writer_preserves_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json';study.write(path,{'closed':True})
            with self.assertRaises(FileExistsError):study.write(path,{'closed':False})
            self.assertEqual(study.p.read(path),{'closed':True})

    def test_failed_reference_read_consumes_gate_and_blocks_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            d=dict(baseline_csv_sha256='old');s=dict(candidate_sha256='new')
            seal=root/'seal.json';seal.write_text('{}')
            with patch.object(study,'STARTED',root/'started.json'),patch.object(study,'RESULT',root/'result.json'),patch.object(study,'SEAL',seal),patch.object(study,'verify_seal',return_value=(d,s)),patch.object(study,'read_reference',side_effect=RuntimeError('synthetic failed read')):
                with self.assertRaisesRegex(RuntimeError,'synthetic'):study.score()
                self.assertTrue(study.STARTED.exists())
                with self.assertRaises(ValueError):study.score()
                with self.assertRaises(ValueError):study.fit(2026)
                with self.assertRaises(ValueError):study.prepare()

    def test_existing_result_reuses_saved_scores_without_reference_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'result.json';study.write(result,{'candidate':{'spell_f1_mean':.6}})
            with patch.object(study,'RESULT',result),patch.object(study,'read_reference',side_effect=AssertionError('must not read')):
                self.assertEqual(study.score()['candidate']['spell_f1_mean'],.6)
                with self.assertRaises(ValueError):study.require_open()

    def test_source_or_artifact_tamper_rejected_before_gate_or_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seal=root/'seal.json';study.write(seal,{})
            with patch.object(study,'STARTED',root/'started.json'),patch.object(study,'RESULT',root/'result.json'),patch.object(study,'SEAL',seal),patch.object(study,'verify_seal',side_effect=ValueError('sealed artifact changed')),patch.object(study,'read_reference',side_effect=AssertionError('must not read')):
                with self.assertRaises(ValueError):study.score()
                self.assertFalse(study.STARTED.exists())

    def test_reference_reader_requires_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'STARTED',Path(tmp)/'not-started'):
            with self.assertRaises(AssertionError):study.read_reference()


if __name__=='__main__':unittest.main()
