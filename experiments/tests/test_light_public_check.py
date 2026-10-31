"""A changed candidate or repeated check must fail before any GT access."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments import light_public_check as public
from experiments.light_candidate import diagnostic_selection
from experiments.eval_set.protocol import sha256


class PublicCheckContracts(unittest.TestCase):
    def test_diagnostic_uses_external_best_new_without_replacing_working_selection(self):
        working = {'selected': {'model': 'baseline'}, 'pooled_checkpoints_sha256': {}}
        def row(model, spell, macro, view='clean'):
            return dict(evaluation='provider_holdout_D', view=view, model=model,
                        spell_f1=spell, macro_f1=macro)
        report = {'summaries': [row('baseline', .6, .8), row('ball_team25', .59, .8),
                  row('player_D25', .58, .95), row('paired25', .9, .9, 'corrupted')]}
        selected = diagnostic_selection(report, working)
        self.assertEqual(selected['selected']['model'], 'ball_team25')
        self.assertEqual(working['selected']['model'], 'baseline')
        self.assertTrue(selected['diagnostic_only'])

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
