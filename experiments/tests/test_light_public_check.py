"""A changed candidate or repeated check must fail before any GT access."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments import light_public_check as public
from experiments.eval_set.protocol import sha256


class PublicCheckContracts(unittest.TestCase):
    def test_frozen_candidate_tampering_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root/'candidate.csv'
            candidate.write_text('frame,label\n0,H\n')
            frozen = dict(files_sha256={'candidate.csv': sha256(candidate)})
            public.verify(frozen, root)
            candidate.write_text('frame,label\n0,A\n')
            with self.assertRaisesRegex(ValueError, 'changed'): public.verify(frozen, root)

    def test_exclusive_marker_refuses_repeated_check(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'public_check_predeclared.json').write_text('{}')
            public.start(out, {'selected_model': 'paired25'})
            with self.assertRaises(FileExistsError): public.start(out, {'selected_model': 'paired25'})
            marker = json.loads((out/'public_check_started.json').read_text())
            self.assertEqual(marker['selected_model'], 'paired25')


if __name__ == '__main__': unittest.main()
