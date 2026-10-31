import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments import turnover_decoder as v1
from experiments import turnover_decoder_v2 as v2


def blocks_fixture():
    rng = np.random.default_rng(517)
    blocks = []
    for source, match, n in (("sportec", "m1", 4), ("sportec", "m2", 8),
                             ("metrica", "m3", 6)):
        x = rng.normal(size=(n, len(v1.FEATURE_NAMES)))
        target = (x[:, 0] > 0).astype(np.int8)
        if target.min() == target.max():
            target[0], target[-1] = 0, 1
        blocks.append({"source": source, "match": match, "features": x, "target": target})
    return blocks


class TurnoverDecoderV2WeightTests(unittest.TestCase):
    def test_global_weight_factor_preserves_relative_source_match_masses_and_statistics(self):
        blocks = blocks_fixture()
        weights = v1._balanced_source_match_weights(blocks)
        scaled = weights * len(weights)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(scaled.mean()), 1.0)
        self.assertAlmostEqual(float(scaled.sum()), float(len(scaled)))
        np.testing.assert_allclose(scaled / scaled.sum(), weights / weights.sum(), rtol=0, atol=1e-15)

        # Explicitly accumulate by block so source and per-match masses remain clear.
        def masses(weight):
            result, offset = {}, 0
            for block in blocks:
                stop = offset + len(block["target"])
                result.setdefault(block["source"], []).append(float(weight[offset:stop].sum()))
                offset = stop
            return result
        norm_mass = masses(weights)
        scaled_mass = masses(scaled)
        self.assertEqual(norm_mass.keys(), scaled_mass.keys())
        for source in norm_mass:
            np.testing.assert_allclose(np.asarray(scaled_mass[source]) / len(scaled),
                                       norm_mass[source], rtol=0, atol=1e-14)

        features = np.concatenate([b["features"] for b in blocks])
        target = np.concatenate([b["target"] for b in blocks])
        tx, stats = v1._fit_transform(features, weights)
        # V2 deliberately uses normalized weights for preprocessing; the factor
        # applies only to the summed optimizer loss, preventing a weighted-median
        # tie from moving due solely to floating-point accumulation order.
        tx_scaled, stats_scaled = v1._fit_transform(features, scaled / len(scaled))
        np.testing.assert_allclose(tx, tx_scaled, rtol=1e-13, atol=1e-13)
        for key in ("median", "mean", "scale"):
            np.testing.assert_allclose(stats[key], stats_scaled[key], rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(float(np.average(target, weights=weights)),
                               float(np.average(target, weights=scaled)), places=14)

    def test_scaled_weights_match_effective_c_and_predictor_uses_features(self):
        blocks = blocks_fixture()
        weights = v1._balanced_source_match_weights(blocks)
        features = np.concatenate([b["features"] for b in blocks])
        target = np.concatenate([b["target"] for b in blocks])
        transformed, _ = v1._fit_transform(features, weights)
        n = len(weights)
        scaled_fit = LogisticRegression(C=v1.LOGISTIC_C, solver="lbfgs", max_iter=1000,
                                        random_state=0).fit(transformed, target, sample_weight=weights * n)
        effective_c_fit = LogisticRegression(C=v1.LOGISTIC_C * n, solver="lbfgs", max_iter=1000,
                                             random_state=0).fit(transformed, target, sample_weight=weights)
        np.testing.assert_allclose(scaled_fit.coef_, effective_c_fit.coef_, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(scaled_fit.intercept_, effective_c_fit.intercept_, rtol=1e-8, atol=1e-9)

        old_head = v1.fit_head(blocks, "sportec")
        head, diagnostic = v2.fit_scaled_head(blocks, "sportec", old_head)
        q = head["estimator"].predict_proba(v1._transform_features(head, features))[:, 1]
        self.assertGreater(float(np.ptp(q)), .05)
        self.assertAlmostEqual(diagnostic["v1"]["weight_sum"], 1.)
        self.assertAlmostEqual(diagnostic["v2"]["weight_mean"], 1.)
        self.assertEqual(diagnostic["transform_unchanged"], True)
        self.assertEqual(diagnostic["prevalence_unchanged"], True)


if __name__ == "__main__":
    unittest.main()
