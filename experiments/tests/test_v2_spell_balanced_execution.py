import unittest
from unittest.mock import patch
from experiments import v2_spell_balanced_execution as execution


class StageMetadataCorrectionTests(unittest.TestCase):
    def test_only_stage_metadata_changes_and_original_is_restored_on_failure(self):
        recipe = dict(recipe_id='anchor', ce_weight=.5, calibrated_weight=0.,
                      adopted_weight=.5, pooling='arithmetic', ha_cost=3., training_stage=512)
        def original(): return [recipe]
        with patch.object(execution.r2, '_shortlist', original):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with execution.corrected_stage_labels():
                    changed=execution.r2._shortlist()[0]
                    self.assertEqual(changed, {**recipe, 'training_stage':1024})
                    self.assertEqual(recipe['training_stage'],512)
                    raise RuntimeError('fixture')
            self.assertIs(execution.r2._shortlist,original)

    def test_original_recipe_validation_cannot_be_bypassed(self):
        with patch.object(execution.r2, '_shortlist', side_effect=ValueError('invalid tuple')):
            with execution.corrected_stage_labels():
                with self.assertRaisesRegex(ValueError, 'invalid tuple'):
                    execution.r2._shortlist()
