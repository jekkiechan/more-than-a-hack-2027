import numpy as np

from experiments import sn_dense_run as run, supervised


def test_viterbi_c_matches_frozen_decode():
    rng = np.random.default_rng(0)
    for n, fps in ((1, 25.0), (2, 25.0), (500, 29.97), (3000, 10.0)):
        p = rng.dirichlet([1, 1, 0.5], size=n)
        period = np.sort(rng.integers(0, 3, n))
        for cost in (0, 3):
            assert np.array_equal(run.viterbi_c(p, period, fps, cost),
                                  supervised.decode(p, period, fps, cost))


def test_viterbi_c_missing_class():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1], size=400); p = np.column_stack([p, np.zeros(400)])
    period = np.zeros(400, int)
    assert np.array_equal(run.viterbi_c(p, period, 25.0, 3), supervised.decode(p, period, 25.0, 3))


def test_tag():
    assert run.tag_of("D2", 0.5, 45, 2026) == "D2_f050_g45_s2026"
