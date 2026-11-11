"""Versioned baseline loading and honest corruption-promotion evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import joblib
from unittest.mock import patch
import numpy as np
from experiments import current_baseline as baseline
from experiments.promote_ballgap_baseline import evidence
from experiments.eval_set.protocol import sha256


class BaselinePromotionContracts(unittest.TestCase):
    def test_robustness_override_does_not_claim_a_clean_win(self):
        def row(model, view, spell, macro):
            return dict(evaluation='provider_holdout_D', model=model, view=view,
                        spell_f1=spell, macro_f1=macro)
        report = dict(summaries=[row('baseline', 'clean', .6, .8),
            row('ball_team25', 'clean', .599, .799), row('baseline', 'corrupted', .42, .77),
            row('ball_team25', 'corrupted', .43, .78)],
            rows=[dict(model=m, source='source', match='one', view='clean', fold='without_source')
                  for m in ('baseline', 'ball_team25')])
        result = evidence(report)
        self.assertFalse(result['clean_rule_candidate_wins'])
        self.assertLess(result['clean_spell_delta'], 0)
        self.assertGreater(result['corrupted_spell_delta'], 0)
        report['rows'][1]['match'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical'): evidence(report)

    def test_manifest_selects_exact_version_and_rejects_tampered_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'attribution.joblib'
            joblib.dump({'version': 'ballgap25'}, model)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(model_directory=str(root),
                models_sha256={'attribution.joblib': sha256(model)})))
            loaded = baseline.load_models(manifest_path=manifest)
            self.assertEqual(loaded['attribution']['version'], 'ballgap25')
            joblib.dump({'version': 'wrong'}, model)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                baseline.load_models(manifest_path=manifest)

    def test_versioned_default_cannot_overwrite_old_candidate(self):
        old = baseline.default_output({'model': 'gru_hgb'})
        new = baseline.default_output({'model': 'gru_hgb_ballgap25'})
        self.assertNotEqual(old, new)
        self.assertEqual(new.name, 'gru_hgb_ballgap25.csv')


    def test_manifest_kind_selects_joint_gru_inference(self):
        inputs = dict(ball=np.zeros((4, 2)))
        with patch.object(baseline, 'joint_gru_probabilities', return_value='joint') as joint, \
             patch.object(baseline, 'dead_probability', return_value='dead') as dead:
            result = baseline.predict_probabilities(inputs, {'joint_gru': object()},
                                                    {'architecture': {'kind': 'joint_gru'}})
        self.assertEqual(result, 'joint')
        joint.assert_called_once()
        dead.assert_not_called()

    def test_active_manifest_and_policy_agree_on_promoted_model(self):
        root = baseline.ROOT/'experiments'
        manifest = json.loads((root/'baseline_manifest.json').read_text())
        policy = json.loads((root/'model_decision_policy.json').read_text())
        self.assertEqual(policy['working_candidate']['model'], manifest['model'])
        self.assertEqual(policy['working_candidate']['baseline_manifest_sha256'],
                         sha256(root/'baseline_manifest.json'))
        if manifest['architecture']['kind']=='sn_dense_spell_frontier_ball_exit_ensemble':
            from experiments import ball_exit_runtime, spell_frontier_baseline
            self.assertEqual(manifest['architecture']['recipe'], spell_frontier_baseline.RECIPE)
            self.assertEqual(manifest['architecture']['ball_exit'], ball_exit_runtime.CORRECTION)
            previous_manifest = json.loads((root.parent/manifest['previous_manifest']).read_text())
            if manifest['promotion'].get('source_status') == 'reconstructed_not_original_checkpoint_restore':
                self.assertEqual(set(manifest['models_sha256']), set(previous_manifest['models_sha256']))
                changed = {name for name, digest in manifest['models_sha256'].items()
                           if digest != previous_manifest['models_sha256'][name]}
                self.assertEqual(changed, {'ce_2026.joblib', 'ce_2027.joblib'})
                reconstruction = manifest['reconstruction']
                seal_path = root.parent/reconstruction['training_seal']
                self.assertEqual(sha256(seal_path), reconstruction['training_seal_sha256'])
                seal = json.loads(seal_path.read_text())
                for name in changed:
                    output = reconstruction['training_outputs'][name]
                    self.assertEqual(output['sha256'], manifest['models_sha256'][name])
                    self.assertEqual(seal['models_sha256'][output['path']], output['sha256'])
                verification_path = root.parent/reconstruction['verification']
                self.assertEqual(sha256(verification_path), reconstruction['verification_sha256'])
                self.assertTrue(json.loads(verification_path.read_text())['all_passed'])
                self.assertTrue(manifest['promotion']['deployment_verified'])
                self.assertTrue(manifest['promotion']['reconstruction_prediction_parity'])
                self.assertTrue(manifest['promotion']['regression_reports_verified'])
                self.assertTrue(manifest['promotion']['organizer_rescoring_performed'])
                self.assertFalse(manifest['promotion']['public_used_for_selection'])
            else:
                self.assertEqual(manifest['models_sha256'], previous_manifest['models_sha256'])
                self.assertFalse(manifest['promotion']['organizer_rescoring_performed'])
            self.assertFalse(manifest['promotion']['fitting_during_promotion'])
            self.assertNotIn('prediction_csv_sha256', manifest['evidence'])
            self.assertNotIn('organizer_public_diagnostic', manifest)
            self.assertTrue(policy['clean_improvement_screen']['current_screen_passes'])
        elif manifest['architecture']['kind']=='sn_dense_spell_frontier_ensemble':
            from experiments import spell_frontier_baseline
            self.assertEqual(manifest['architecture']['recipe'], spell_frontier_baseline.RECIPE)
            self.assertEqual(manifest['architecture']['phase_order'], 'seeds_then_phases')
            self.assertEqual(len(manifest['models_sha256']), 10)
            self.assertFalse(manifest['evidence']['historical_clean_screen_passes'])
            self.assertFalse(policy['clean_improvement_screen']['current_screen_passes'])
            self.assertTrue(policy['clean_improvement_screen']['current_adoption_exception'])
            self.assertFalse(manifest['promotion']['fitting_during_promotion'])
            self.assertFalse(manifest['promotion']['organizer_rescoring_performed'])
            previous_manifest = json.loads((root.parent/manifest['previous_manifest']).read_text())
            for seed in (2026, 2027):
                self.assertEqual(manifest['models_sha256'][f'adopted_{seed}.joblib'],
                                 previous_manifest['models_sha256'][f'dense_{seed}.joblib'])
                self.assertEqual(manifest['models_sha256'][f'teacher_adopted_{seed}.joblib'],
                                 previous_manifest['models_sha256'][f'teacher_{seed}.joblib'])
        elif manifest['architecture']['kind']=='sn_dense_original_diversity_phase_ensemble':
            self.assertTrue(manifest['architecture']['phase_tta'])
            self.assertEqual(manifest['architecture']['phase_averaging']['phases'], 5)
            self.assertEqual(manifest['architecture']['postprocessing']['temperature'], 4)
            self.assertTrue(manifest['evidence']['historical_active_guards_pass'])
            self.assertFalse(manifest['evidence']['historical_clean_screen_passes'])
            self.assertFalse(policy['clean_improvement_screen']['current_screen_passes'])
            self.assertTrue(policy['clean_improvement_screen']['current_adoption_exception'])
            self.assertEqual(policy['primary_evaluation_view'], 'feed_like_v2')
            self.assertGreaterEqual(policy['stress_guardrail']['minimum_realizations'], 4)
            self.assertFalse(manifest['promotion']['fitting_during_promotion'])
            self.assertFalse(manifest['promotion']['organizer_rescoring_performed'])
            previous_manifest = json.loads((root.parent/manifest['previous_manifest']).read_text())
            self.assertEqual(manifest['models_sha256'], previous_manifest['models_sha256'])
            self.assertFalse(previous_manifest['architecture']['phase_tta'])
        elif manifest['architecture']['kind']=='sn_dense_original_diversity_ensemble':
            development = manifest['evidence']['development_full_provider_holdout']
            self.assertTrue(development['provider_and_feed_like_guards_pass'])
            self.assertFalse(development['clean_improvement_screen_plus_01_passes'])
            self.assertFalse(policy['clean_improvement_screen']['current_screen_passes'])
            self.assertTrue(policy['clean_improvement_screen']['current_adoption_exception'])
            self.assertEqual(manifest['architecture']['postprocessing']['temperature'], 4)
            self.assertFalse(manifest['architecture']['phase_tta'])
            self.assertFalse(manifest['promotion']['fitting_during_promotion'])
            self.assertFalse(manifest['organizer_public_diagnostic']['used_for_model_or_hyperparameter_selection'])
            self.assertEqual(sorted(manifest['architecture']['members'].values()), [.5, .5])
            previous_manifest = json.loads((root.parent/manifest['previous_manifest']).read_text())
            for seed in (2026, 2027):
                name = f'teacher_{seed}.joblib'
                self.assertEqual(manifest['models_sha256'][name], previous_manifest['models_sha256'][name])
                name = f'dense_{seed}.joblib'
                self.assertNotEqual(manifest['models_sha256'][name], previous_manifest['models_sha256'][name])
        elif manifest['architecture']['kind']=='sn_dense_calibrated_ensemble':
            development = manifest['evidence']['development_full_provider_holdout']
            self.assertTrue(development['provider_and_feed_like_guards_pass'])
            self.assertTrue(development['clean_improvement_screen_plus_01_passes'])
            self.assertEqual(manifest['architecture']['postprocessing']['temperature'], 4)
            self.assertTrue(manifest['promotion']['model_checkpoints_unchanged'])
            self.assertTrue(manifest['promotion']['public_diagnostic_known_before_authorization'])
            self.assertFalse(manifest['organizer_public_diagnostic']['used_for_model_or_hyperparameter_selection'])
            previous_manifest = json.loads((root.parent/manifest['previous_manifest']).read_text())
            self.assertEqual(manifest['models_sha256'], previous_manifest['models_sha256'])
        elif manifest['architecture']['kind']=='sn_dense_ensemble':
            development = manifest['evidence']['development_full_provider_holdout']
            self.assertTrue(development['provider_and_feed_like_guards_pass'])
            self.assertFalse(development['clean_improvement_screen_plus_01_passes'])
            self.assertTrue(manifest['promotion']['public_diagnostic_known_before_authorization'])
            self.assertFalse(manifest['organizer_public_diagnostic']['used_for_model_or_hyperparameter_selection'])
            self.assertEqual(sorted(manifest['architecture']['members'].values()), [.5, .5])
        elif manifest['architecture']['kind']=='soccernet_capacity_ensemble':
            comparison = manifest['evidence']['comparison']
            self.assertTrue(all(comparison['provider_clean_guards'].values()))
            self.assertTrue(comparison['feed_like_guard_passes'])
            self.assertFalse(comparison['historical_improvement_screen_passes'])
            self.assertTrue(manifest['promotion']['public_diagnostic_known_before_authorization'])
            self.assertEqual(sorted(manifest['architecture']['members'].values()), [.5, .5])
        elif manifest['architecture']['kind']=='joint_gru_ensemble':
            if manifest['version'] >= 5:
                guards = manifest['evidence']['comparison']['selection_gate']
                self.assertTrue(guards['passes'])
                self.assertTrue(guards['checks'])
                self.assertTrue(all(guards['checks'].values()))
            else:
                self.assertTrue(manifest['evidence']['guardrails']['passes'])
            self.assertEqual(sorted(manifest['architecture']['members'].values()),[.5,.5])
        else:
            self.assertTrue(all(seed['passes_feed_like_guardrails']
                                for seed in manifest['evidence']['seeds'].values()))
        previous = root.parent/manifest['previous_manifest']
        self.assertEqual(sha256(previous), manifest['previous_manifest_sha256'])


if __name__ == '__main__': unittest.main()
