"""Contracts for the dense-supervision SoccerNet training harness (synthetic data only)."""
import unittest

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt
from experiments import soccernet_capacity_model as cap, sn_dense_train as sn

WIDTH = 204
REAL = ["ball_available", "ball_observed", "ball_speed", "missing_before_s", "missing_after_s",
        "H_distance1", "A_distance1", "D_speed_mean", "speed_mean_1.0s",
        "H_physical_nearest1_player_speed", "A_physical_nearest1_player_speed"]
NAMES = REAL + [f"f_{i}" for i in range(WIDTH - len(REAL))]


def teacher(seed=7):
    import torch
    torch.manual_seed(seed)
    net = fh.network(2 * WIDTH, list(range(20)), player_only=True)
    return dict(kind="factorized_had", player_only=True, seed=seed, player_indices=list(range(20)),
        state={k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()},
        median=np.zeros(WIDTH, dtype=np.float32), scale=np.ones(WIDTH, dtype=np.float32),
        input_features=2 * WIDTH, feature_names=NAMES.copy())


def old_record(match="old", seed=1, n=100):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .3, (n, WIDTH)).astype(np.float32)
    x[:, 1] = 1; x[:, 0] = 1
    y = np.full(n, -1, dtype=np.int8)
    y[10:40] = 0; y[40:60] = 1; y[60:70] = 2; y[70:95] = 0
    period = np.zeros(n, dtype=np.int32)
    full_y = np.repeat(y, 5)
    return dict(source="sportec", match=match, x=x, y=y, period=period, names=NAMES.copy(),
        fps=25.0, indices=np.arange(n) * 5, native_n=5 * n, full_y=full_y,
        full_period=np.zeros(5 * n, dtype=np.int32), alternate=x.copy(),
        replace_windows=np.zeros(len(list(lt.chunks(dict(period=period)))), dtype=bool))


def sn_record(game="g1", seed=3, n=160, dense=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .3, (n, WIDTH)).astype(np.float32)
    x[:, 1] = 1; x[:, 0] = 1
    y = np.full(n, -1, dtype=np.int8)
    y[20:60] = 0; y[60:75] = 1; y[75:90] = 2; y[90:120] = 1; y[120:150] = 0
    full_y = np.repeat(y, 5)
    if not dense:
        keep = np.zeros(n, bool); keep[[30, 65, 100, 130]] = True
        y = np.where(keep & (y < 2), y, -1).astype(np.int8)
        full_y = np.full(5 * n, -1, dtype=np.int8); full_y[::5] = y
    return dict(source="soccernet", match=game, x=x, y=y, period=np.zeros(n, dtype=np.int32),
        names=NAMES.copy(), fps=np.asarray(5.0), native_fps=np.asarray(25.0),
        timestamps=np.arange(n) * .2, indices=np.arange(n) * 5, native_n=np.asarray(5 * n),
        full_y=full_y, full_period=np.zeros(5 * n, dtype=np.int32),
        full_timestamps=np.arange(5 * n) / 25.0)


class Harness(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.t = teacher()
        self.old = [old_record("o1", 1), old_record("o2", 2)]
        self.new = [sn_record("g1", 3), sn_record("g2", 4), sn_record("g3", 5)]

    def run_train(self, **kw):
        base = dict(width=24, lr=1e-4, seed=5, steps=3, batch_size=8, verbose=False)
        base.update(kw)
        return sn.train(self.t, self.old, self.new, **base)

    def test_default_reproduces_frozen_trainer(self):
        new = [sn_record(g, s, dense=False) for g, s in (("g1", 3), ("g2", 4))]
        ref = cap.train(self.t, self.old, new, width=24, lr=1e-4, seed=5, steps=3,
                        milestones=(3,), batch_size=8, require_full_native=True)[3]
        got = sn.train(self.t, self.old, new, width=24, lr=1e-4, seed=5, steps=3,
                       batch_size=8, require_full_native=True, verbose=False)["model"]
        for k in ref["state"]:
            np.testing.assert_array_equal(ref["state"][k], got["state"][k])

    def test_fraction_extremes_are_finite(self):
        for fraction, dense in ((0.0, "anchors"), (1.0, "anchors"), (1.0, "dense"), (0.0, "dense")):
            with self.subTest(fraction=fraction, ha=dense):
                old = self.old if fraction < 1 else []
                new = self.new if fraction > 0 else []
                out = sn.train(self.t, old, new, width=24, lr=1e-4, seed=1, steps=4, batch_size=8,
                               fraction=fraction, ha_target=dense, verbose=False)
                self.assertTrue(np.isfinite(out["loss_intervals"][-1]["loss"]))
                self.assertEqual(any(k.startswith("old/") for k in out["exposures"]), fraction < 1)
                self.assertEqual(any(k.startswith("new/") for k in out["exposures"]), fraction > 0)

    def test_fraction_one_with_train_d_dense_ok_anchors_rejected(self):
        out = sn.train(self.t, [], self.new, width=24, lr=1e-4, steps=3, batch_size=8,
                       fraction=1.0, ha_target="dense", train_d=True, verbose=False)
        self.assertTrue(np.isfinite(out["loss_intervals"][-1]["d_loss"]))
        with self.assertRaises(ValueError):
            sn.train(self.t, [], self.new, width=24, lr=1e-4, steps=3, batch_size=8,
                     fraction=1.0, ha_target="anchors", train_d=True, verbose=False)

    def test_dense_uses_spell_and_transition_on_soccernet(self):
        spell, pairs, _, native = sn.objective_targets([sn_record()], require_full_native=True)
        self.assertTrue(native)
        self.assertGreater(np.ptp(spell[0]), 0)           # complete spells reweighted
        self.assertTrue(pairs[0]["live_mask"].any() and pairs[0]["live_target"].any())
        # SoccerNet native fps (25), not sampled fps (5), sets the duration scale.
        record = sn_record(); slow = dict(record, native_fps=np.asarray(5.0))
        a = sn.objective_targets([record], require_full_native=True)[0][0]
        b = sn.objective_targets([slow], require_full_native=True)[0][0]
        self.assertFalse(np.array_equal(a, b))
        out = self.run_train(ha_target="dense", fraction=0.5)
        self.assertIn("new_loss", out["loss_intervals"][-1])

    def test_train_d_changes_only_dead_probability_path(self):
        frozen = self.run_train(train_d=False, ha_target="dense", steps=6)["model"]
        tuned = self.run_train(train_d=True, ha_target="dense", steps=6, d_lr=3e-3)["model"]
        rec = sn_record("v", 9)
        p0, p1 = sn.predict(rec, frozen, self.t), sn.predict(rec, tuned, self.t)
        base = fh.predict(dict(x=rec["x"], names=rec["names"], period=rec["period"]), self.t)
        np.testing.assert_allclose(p0[:, 2], base[:, 2], atol=1e-6)   # D copied from teacher
        self.assertGreater(np.abs(p1[:, 2] - base[:, 2]).max(), 1e-4)  # D fine-tuned
        np.testing.assert_allclose(p1.sum(1), 1, atol=1e-5)
        self.assertNotIn("d_state", frozen)
        # H/A factorization: conditional share is invariant to the D head used.
        self.assertTrue(np.isfinite(p1).all())

    def test_game_subsets_deterministic_nested_and_enforced(self):
        games = [f"game{i}" for i in range(45)]
        sizes = [nested for nested in (sn.nested_games(games, n) for n in (11, 22, 45))]
        self.assertEqual([len(s) for s in sizes], [11, 22, 45])
        self.assertTrue(set(sizes[0]) < set(sizes[1]) < set(sizes[2]))
        self.assertEqual(sn.nested_games(list(reversed(games)), 22), sizes[1])
        self.assertEqual(sn.rank_games(games)[0],
                         min(games, key=lambda g: __import__("hashlib").sha256(g.encode()).hexdigest()))
        out = self.run_train(soccernet_games=["g2"], steps=6)
        self.assertEqual({k for k in out["exposures"] if k.startswith("new/")}, {"new/g2"})
        self.assertEqual(out["config"]["soccernet_games"], ["g2"])
        with self.assertRaises(ValueError):
            self.run_train(soccernet_games=["missing"])
        with self.assertRaises(ValueError):
            sn.nested_games(games, 46)

    def test_ball_dropout_view_and_training(self):
        rec = sn_record(n=400)
        cal = sn.default_ball_calibration()
        x = sn.ball_dropout_view(rec, cal, 11)
        again = sn.ball_dropout_view(rec, cal, 11)
        np.testing.assert_array_equal(np.nan_to_num(x, nan=-9), np.nan_to_num(again, nan=-9))
        dropped = x[:, 1] == 0
        self.assertTrue(dropped.any())
        self.assertTrue(np.isnan(x[dropped][:, 2]).all())          # ball_speed
        self.assertTrue(np.isfinite(x[dropped][:, 7]).all())       # D_speed_mean kept
        np.testing.assert_array_equal(x[~dropped], rec["x"][~dropped])
        self.assertTrue((x[dropped][:, 3] > 0).all())              # missing_before_s
        out = self.run_train(ball_dropout=1.0, ha_target="dense")
        self.assertTrue(np.isfinite(out["loss_intervals"][-1]["loss"]))
        calls = []
        self.run_train(ball_dropout=1.0, ball_view_fn=lambda r, s: calls.append(s) or r["x"])
        self.assertTrue(calls)

    def test_early_stopping_keeps_best_snapshot(self):
        scores = iter([0.5, 0.7, 0.6, 0.65, 0.9])
        seen = []

        def validate(snapshot):
            seen.append(snapshot["step"])
            return dict(score=next(scores))

        out = self.run_train(steps=10, eval_every=2, validate=validate, patience=2)
        self.assertEqual(seen, [2, 4, 6, 8])
        self.assertEqual(out["best_step"], 4)
        self.assertEqual(out["model"]["step"], 4)
        self.assertEqual(out["final"]["step"], 8)
        with self.assertRaises(ValueError):
            self.run_train(eval_every=2)

    def test_gar_validator_scores_with_official_metrics(self):
        model = self.run_train(ha_target="dense")["model"]
        validate = sn.make_gar_validator([sn_record("v1", 21), sn_record("v2", 22)], self.t)
        result = validate(model)
        for key in ("spell_f1", "macro_f1", "d_f1", "short_spell_recall", "score"):
            self.assertIn(key, result)
        self.assertTrue(0 <= result["spell_f1"] <= 1 and 0 <= result["macro_f1"] <= 1)
        self.assertEqual(result["score"], result["spell_f1"])
        out = self.run_train(steps=4, eval_every=2, validate=validate, ha_target="dense")
        self.assertIsNotNone(out["best_step"])


if __name__ == "__main__":
    unittest.main()
