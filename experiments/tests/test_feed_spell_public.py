import json,tempfile,unittest
from pathlib import Path
from unittest import mock
from experiments import feed_spell_public as public


class FeedPublicGateContracts(unittest.TestCase):
    def test_failed_reference_hash_keeps_gate_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=root/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
            ref.parent.mkdir(parents=True);ref.write_text('deliberately invalid fixture')
            seal=root/'seal.json';seal.write_text(json.dumps({'expected_reference_sha256':'0'*64}))
            started=root/'started.json'
            with mock.patch.multiple(public,ROOT=root,SEAL=seal,STARTED=started,RESULT=root/'result.json'),mock.patch.object(public,'validate_seal'):
                with self.assertRaisesRegex(ValueError,'Reference SHA mismatch'):public.score()
                self.assertTrue(started.exists())
                with self.assertRaisesRegex(ValueError,'permanently closed'):public.score()

    def test_legacy_requirement_follows_nested_config(self):
        self.assertFalse(public.uses_legacy({'family':'base'}))
        self.assertTrue(public.uses_legacy({'family':'duration','parent':{'family':'legacy'}}))
        self.assertTrue(public.uses_legacy({'family':'consensus','left':{'family':'base'},'right':{'family':'legacy'}}))

    def test_completed_result_is_read_without_reference_or_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'result.json';path.write_text('{"saved":true}')
            with mock.patch.object(public,'RESULT',path),mock.patch.object(public,'validate_seal',side_effect=AssertionError('unexpected scoring')):
                self.assertEqual(public.score(),{'saved':True})
