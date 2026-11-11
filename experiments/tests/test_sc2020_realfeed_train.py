import unittest

import numpy as np

from experiments import sc2020_realfeed_train as study
from experiments import sc2020_realfeed_score as score
from experiments.tests.test_sn_dense_train import old_record, sn_record, teacher as factorized_teacher


def _initial():
    import torch
    torch.set_num_threads(1)
    teacher = factorized_teacher(seed=17)
    ha = study.cap.network(teacher["input_features"], 96)
    prior = dict(kind="soccernet_capacity", hidden=96, feature_names=list(teacher["feature_names"]),
                 median=teacher["median"].copy(), scale=teacher["scale"].copy(),
                 state={k: v.detach().cpu().numpy().copy() for k, v in ha.state_dict().items()},
                 d_state={k: v.copy() for k, v in teacher["state"].items()})
    return study.iterative.warmstart_state(teacher, prior), prior


class FitLoopContracts(unittest.TestCase):
    def setUp(self):
        self.old = [old_record("old-a", 3), old_record("old-b", 4)]
        self.gar = [sn_record("game-a", 5), sn_record("game-b", 6)]
        self.extra = [dict(sn_record("sc-a", 8), source=study.SOURCE), dict(sn_record("sc-b", 9), source=study.SOURCE)]

    def test_without_extra_windows_matches_shared_iterative_objective(self):
        initial, _ = _initial()
        expected = study.iterative._fit_loop(initial, self.old, self.gar, seed=2026, arm="ce_control", steps=3)
        old_lr = study.iterative.LR
        actual = study.fit_loop(initial, self.old, self.gar, [], seed=2026, extra_count=0,
                                lr=old_lr, snapshots=(3,))[3]
        for key, value in expected["model"]["state"].items():
            np.testing.assert_array_equal(actual["state"][key], value)

    def test_extra_windows_replace_gar_windows_and_keep_d_state(self):
        initial, prior = _initial()
        models = study.fit_loop(initial, self.old, self.gar, self.extra, seed=2026, extra_count=4, snapshots=(1, 2))
        self.assertEqual(sorted(models), [1, 2])
        exposures = models[2]["metadata"]["exposures"]
        total = lambda prefix: sum(v for k, v in exposures.items() if k.startswith(prefix))
        self.assertEqual((total("old/"), total("new/"), total("extra/")), (32, 24, 8))
        self.assertAlmostEqual(sum(models[2]["training"]["loss_weights"].values()), 1.0)
        for key, value in prior["d_state"].items():
            np.testing.assert_array_equal(models[2]["d_state"][key], value)
        self.assertFalse(all(np.array_equal(models[1]["state"][k], models[2]["state"][k]) for k in models[1]["state"]))

    def test_rejects_extra_stream_without_records_or_without_gar(self):
        initial, _ = _initial()
        with self.assertRaises(ValueError):
            study.fit_loop(initial, self.old, self.gar, [], seed=2026, extra_count=4, snapshots=(1,))
        with self.assertRaises(ValueError):
            study.fit_loop(initial, self.old, self.gar, self.extra, seed=2026, extra_count=16, snapshots=(1,))

    def test_skillcorner_fold_never_trains_on_skillcorner_2020(self):
        with self.assertRaisesRegex(ValueError, "excluded"):
            study.fit_one("sc4", "leave_out_skillcorner", 2026)


class GuardContracts(unittest.TestCase):
    def _agg(self, spell, short=0.3, macro=0.85):
        row = {"spell_f1": spell, score.SHORT: short, "macro_f1": macro,
               "macro_f1/providers": {"Sportec": macro}}
        return {view: dict(row) for view in score.VIEW_NAMES}

    def test_v2_must_strictly_improve_and_nothing_may_regress(self):
        control = self._agg(0.59)
        self.assertEqual(score.guards(self._agg(0.59), control), ["feed_like_v2/spell_f1"])
        self.assertEqual(score.guards(self._agg(0.60), control), [])
        worse = self._agg(0.60); worse["clean"][score.SHORT] = 0.29
        self.assertEqual(score.guards(worse, control), [f"clean/{score.SHORT}"])
        macro = self._agg(0.60); macro["feed_like_v1"]["macro_f1"] = 0.8489
        self.assertEqual(score.guards(macro, control), ["feed_like_v1/macro_f1"])


if __name__ == "__main__":
    unittest.main()
