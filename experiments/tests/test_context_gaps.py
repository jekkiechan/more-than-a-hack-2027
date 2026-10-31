import numpy as np
import pytest

from experiments import context_gaps as cg, supervised

FPS = 25.


class ConstModel:
    def predict_proba(self, X):
        h = 0.002 + 0.004 * (np.asarray(X)[:, 0] > 5)
        return np.column_stack([1 - h, h])


def _hz():
    return dict(model=ConstModel(), median=np.full(len(cg.FEATURES), 5.0),
                tercile_edges=np.array([0.0025, 0.005]),
                durations=[np.array([1., 2.]), np.array([2., 3.]), np.array([1., 3.])])


def _data(n=6000, slots=6, seed=0):
    rng = np.random.default_rng(seed)
    period = np.r_[np.full(n // 2, 1), np.full(n // 2, 2)].astype(np.int16)
    ts = np.r_[np.arange(n // 2), np.arange(n // 2)] / FPS
    walk = lambda *s: np.cumsum(rng.normal(0, .05, s), axis=0)
    ball = 52 + walk(n, 2)
    players = 40 + walk(n, slots, 2) + rng.uniform(0, 20, (1, slots, 2))
    return dict(ball=ball, players=players, teams=np.array([0, 1] * (slots // 2), np.int8), period=period,
                fps=np.float64(FPS), timestamps=ts, pitch=np.array([105., 68.]),
                ball_observed=np.ones(n, bool), player_observed=np.ones((n, slots), bool))


def _gaps(data, scalar, seed):
    X, aux = cg.context_features(data)
    h = cg.hazard_per_frame(_hz(), data, X, aux)
    return h, cg.simulate_context_gaps(data, h, _hz(), scalar, seed)


def test_features_finite_and_shapes():
    d = _data()
    X, aux = cg.context_features(d)
    assert X.shape == (len(d["ball"]), len(cg.FEATURES))
    assert np.isfinite(X[20:3000]).all()
    assert np.isnan(X[:10]).all()          # lag window not available


def test_determinism_and_seed_dependence():
    d = _data()
    _, a = _gaps(d, 3., 11)
    _, b = _gaps(d, 3., 11)
    _, c = _gaps(d, 3., 12)
    assert a == b and a != c and len(a) > 0


def test_periods_reset_and_no_overlap():
    d = _data()
    _, g = _gaps(d, 5., 3)
    blocks = supervised.blocks(supervised.model_periods(d))
    for s, e, _ in g:
        assert any(a <= s and e <= b for a, b in blocks)       # never spans the half-time break
    ends = [x[1] for x in g]
    starts = [x[0] for x in g]
    assert all(s2 >= e1 for e1, s2 in zip(ends, starts[1:]))


def test_missing_fraction_near_target():
    d = _data(12000)
    X, aux = cg.context_features(d)
    h = cg.hazard_per_frame(_hz(), d, X, aux)
    s, ach = cg.calibrate_scalar([(d, h)], _hz(), target=0.25)
    assert abs(ach - 0.25) < 0.04
    g = cg.simulate_context_gaps(d, h, _hz(), s, 5)
    cov = cg.gap_mask(len(h), g).mean()
    assert abs(cov - 0.25) < 0.1


def test_random_matched_keeps_counts_and_durations():
    d = _data()
    _, g = _gaps(d, 4., 9)
    r = cg.random_matched_gaps(d, g, 1)
    assert len(r) == len(g)
    assert sorted(x[1] - x[0] for x in r) == sorted(x[1] - x[0] for x in g)
    blocks = supervised.blocks(supervised.model_periods(d))
    for s, e, _ in r:
        assert any(a <= s and e <= b for a, b in blocks)
    m = cg.gap_mask(len(d["ball"]), r)
    assert m.sum() == sum(x[1] - x[0] for x in r)             # no overlaps


def test_build_view_only_changes_ball_and_keeps_metadata():
    d = _data(3000, slots=6)
    cal = dict(ball_missing_fraction=.4, gap_edges_seconds=[.08, .2, .5, 1., 2., 5., 10., 30.],
               gap_bin_probability=[.1, .1, .1, .1, .2, .2, .2])
    before = {k: np.array(v, copy=True) for k, v in d.items()}
    _, g = _gaps(d, 5., 2)
    v = cg.build_view(d, cal, 7, g)
    for k in ("period", "timestamps", "teams", "fps", "pitch"):
        assert np.array_equal(v[k], d[k])
    for k in before:                                          # input not mutated
        assert np.array_equal(before[k], d[k])
    assert "labels" not in v
    m = cg.gap_mask(len(d["ball"]), g)
    assert np.isnan(v["ball"][m]).all()
    assert np.array_equal(v["ball_observed"], np.isfinite(v["ball"]).all(1))
    # identical non-ball-gap components as the feed_like call with the same seed
    from experiments import stress_components as sc
    ref = sc.corrupt(d, cal, 7, cg.FEED_NO_BALL_GAPS)
    assert np.array_equal(np.nan_to_num(ref["players"], nan=-1), np.nan_to_num(v["players"], nan=-1))


def test_transition_points():
    lab = np.array(list("HHAADDHHXXAA"))
    d = dict(period=np.ones(len(lab), np.int16), timestamps=np.arange(len(lab)) / FPS, fps=FPS)
    t = cg.transition_points(lab, d)
    assert list(t["ha_change"]) == [2] and list(t["live_dead"]) == [4, 6]
