"""The public diagnostic must be label-free until an exclusive frozen check."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments import joint_competition_check as public, live_gap_experiment as old
from experiments.tests.test_tracking_corruption import synthetic


class JointCompetitionContracts(unittest.TestCase):
    def test_inference_features_match_development_without_targets(self):
        data=synthetic(150)
        labels=np.full(150,'H')
        expected=old.feature_view(data,labels,'s','m')['joint']
        actual,period=public.feature_record(dict(data,labels=np.full(150,'D'),owner=np.ones(150)))
        self.assertNotIn('y',actual)
        self.assertEqual(actual['names'],expected['names'])
        for key in ('x','indices','period'):
            np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(actual['native_n'],150)
        self.assertEqual(len(period),150)

    def test_pooled_fit_excludes_all_development_matches(self):
        training=public.training_paths()
        self.assertEqual(len(training),13)
        self.assertEqual(len({p.stem for _,p in training}),13)
        evaluation={p.stem for split in public.dev.dt.SPLITS.values() for p in split['eval']}
        self.assertFalse(evaluation & {p.stem for _,p in training})
        self.assertEqual(public.SEED,2026)
        self.assertEqual(public.ARM,'factorized_player_D')

    def test_tampered_candidate_and_post_seal_fitting_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'candidate.csv'
            path.write_text('frame,label\n0,H\n')
            frozen=dict(files_sha256={'candidate.csv':public.sha256(path)})
            public.verify_files(frozen,root)
            path.write_text('frame,label\n0,A\n')
            with self.assertRaisesRegex(ValueError,'changed'): public.verify_files(frozen,root)
            (root/'public_check_predeclared.json').write_text('{}')
            with patch.object(public,'REPORT',root/'report.json'):
                with self.assertRaisesRegex(ValueError,'sealed'): public.require_open_training(root)

    def test_repeated_check_fails_before_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            frozen=dict(arm=public.ARM,seed=public.SEED,files_sha256={})
            (out/'public_check_predeclared.json').write_text(json.dumps(frozen))
            public.start(out,frozen)
            with patch.object(public,'REPORT',out/'report.json'),patch.object(public,'verify_plan',return_value={}),patch('pandas.read_csv') as read:
                with self.assertRaises(FileExistsError): public.check(out)
                read.assert_not_called()


if __name__=='__main__': unittest.main()
