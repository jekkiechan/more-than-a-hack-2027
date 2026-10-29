"""Synthetic guards for the single-use public diagnostic; no reference reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import smart_tagging_check as study


class SmartTaggingCheckContracts(unittest.TestCase):
    def test_cli_resolves_relative_output_before_artifact_sealing(self):
        with patch('sys.argv',['check','predict','--out','data/synthetic']), \
                patch.object(study,'predict') as predict:
            study.main()
            self.assertEqual(predict.call_args.args[0], Path('data/synthetic').resolve())

    def test_exclusive_write_preserves_existing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json'
            study.write(path, {'closed':True})
            with self.assertRaises(FileExistsError):
                study.write(path, {'closed':False})
            self.assertEqual(json.loads(path.read_text()), {'closed':True})

    def test_closed_global_report_or_local_gate_blocks_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(study,'REPORT',out/'public.json'):
                study.require_open(out)
                study.write(out/'public_check_started.json',{})
                with self.assertRaises(ValueError): study.require_open(out)
                (out/'public_check_started.json').unlink()
                study.write(out/'public.json',{})
                with self.assertRaises(ValueError): study.require_open(out)

    def test_frozen_file_change_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study,'ROOT',Path(tmp)):
            path=Path(tmp)/'input'
            path.write_text('before')
            mapping={'input':study.sha(path)}
            study.verify(mapping)
            path.write_text('changed')
            with self.assertRaises(ValueError): study.verify(mapping)
            with self.assertRaises(ValueError): study.verify({'../outside':'0'*64})

    def test_source_tamper_rejected_before_any_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            study.write(out/'declaration.json', {'source_sha256':{'code':'old'},'files_sha256':{}})
            with patch.object(study,'source_files',return_value={'code':'changed'}):
                with self.assertRaises(ValueError): study.read_spec(out)

    def test_gate_is_consumed_before_even_failed_reference_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'run'; out.mkdir()
            declaration=out/'declaration.json'; study.write(declaration,{})
            artifacts={}
            for name in ('baseline','smart_tagging_missing_ball'):
                for suffix in ('.csv','.npz'):
                    path=out/(name+suffix); path.write_bytes(b'synthetic')
                    artifacts[str(path.relative_to(root))]=study.sha(path)
            study.write(out/'prediction_seal.json',dict(
                files_sha256=artifacts,rows=199319,declaration_sha256=study.sha(declaration),
                reference_labels_read=False,baseline_reproduces_saved_scored_csv=True,
                full_length_csvs_validated=True,native_p_d_bitwise_equal=True))
            with patch.object(study,'ROOT',root), patch.object(study,'REPORT',root/'report.json'), \
                    patch.object(study,'read_spec',return_value={'candidates':['smart_tagging_missing_ball']}), \
                    patch('pandas.read_csv',side_effect=RuntimeError('synthetic failed read')):
                with self.assertRaises(RuntimeError): study.check(out)
                self.assertTrue((out/'public_check_started.json').is_file())
                with self.assertRaises(ValueError): study.check(out)
                other=root/'alternate'; other.mkdir()
                with self.assertRaises(ValueError): study.require_open(other)


if __name__=='__main__':
    unittest.main()
