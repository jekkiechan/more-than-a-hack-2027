import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import soccernet_capacity_public as public

class PublicCapacityGate(unittest.TestCase):
    def test_gate_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'started.json';public.exclusive(p,{'sealed':True})
            with self.assertRaises(FileExistsError):public.exclusive(p,{'sealed':True})

    def test_closed_gate_prevents_reference_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'started.json';marker.write_text('{}')
            seal=root/'seal.json';seal.write_text('{}')
            with patch.object(public,'STARTED',marker),patch.object(public,'SEAL',seal),patch.object(public,'RESULT',root/'missing.json'),patch.object(public,'verify_seal'),patch.object(public.study,'read_json',return_value={}),patch.object(public.subprocess,'run') as access:
                with self.assertRaises(FileExistsError):public.score()
                access.assert_not_called()

    def test_predicate_requires_complete_external_evidence_first(self):
        with patch.object(public,'assert_external_complete',side_effect=ValueError('not selected')),patch.object(public,'assert_open') as gate,patch.object(public.joblib,'load') as models:
            with self.assertRaisesRegex(ValueError,'not selected'):public.predict()
            gate.assert_not_called();models.assert_not_called()

if __name__=='__main__':unittest.main()
