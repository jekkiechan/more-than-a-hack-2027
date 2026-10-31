import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('capacity_reader_runtime_contract',ROOT/'scripts/soccernet_capacity_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

class ConfirmationReader(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for name in ('a.joblib','b.joblib','predeclared.json','coarse_checkpoints_frozen.json'):(self.root/name).write_bytes(name.encode())
        s=runtime.study
        self.value={'declaration_sha256':s.sha256(self.root/'predeclared.json'),'coarse_seal_sha256':s.sha256(self.root/'coarse_checkpoints_frozen.json'),'coarse_models':{'a':s.sha256(self.root/'a.joblib')},'selected_models':{'b':s.sha256(self.root/'b.joblib')}}
        self.path=self.root/'checkpoints_confirm_frozen.json';self.path.write_text(json.dumps(self.value))
        self.a=patch.object(s,'OUT',self.root);self.a.start()
        self.b=patch.object(s,'_resolve_key',side_effect=lambda name:self.root/(name+'.joblib'));self.b.start()
    def tearDown(self):self.b.stop();self.a.stop();self.tmp.cleanup()
    def test_checks_both_tables_without_rewriting_seal(self):
        before=self.path.read_bytes();result=runtime.check_confirmation_seal(self.path.name)
        self.assertEqual(set(result['models']),{'a','b'});self.assertEqual(before,self.path.read_bytes())
    def test_rejects_tampered_selected_checkpoint(self):
        (self.root/'b.joblib').write_bytes(b'different')
        with self.assertRaises(ValueError):runtime.check_confirmation_seal(self.path.name)
    def test_rejects_tampered_coarse_receipt(self):
        (self.root/'coarse_checkpoints_frozen.json').write_bytes(b'different')
        with self.assertRaises(ValueError):runtime.check_confirmation_seal(self.path.name)
    def test_rejects_missing_or_conflicting_tables(self):
        self.value['selected_models']={'a':'0'*64};self.path.write_text(json.dumps(self.value))
        with self.assertRaises(ValueError):runtime.check_confirmation_seal(self.path.name)

if __name__=='__main__':unittest.main()
