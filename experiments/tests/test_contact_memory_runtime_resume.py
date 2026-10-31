"""Contracts for the numerical-only contact-memory score recovery."""
import ast
import inspect
import unittest

import numpy as np

from experiments import contact_memory_runtime_resume as resume


def correction_strings():
    source = inspect.getsource(resume._corrected_score_function)
    tree = ast.parse(source)
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                    "exact_guard", "corrected_guard"}:
                values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"exact_guard", "corrected_guard"}:
        raise AssertionError("Runtime wrapper guard contract could not be read")
    return values["exact_guard"], values["corrected_guard"]


class ContactMemoryRuntimeCorrectionTests(unittest.TestCase):
    def test_only_local_replay_guard_is_tolerant_and_archive_guard_stays_exact(self):
        exact_guard, corrected_guard = correction_strings()
        toy = """def score(candidate_probability, base_probability, saved_probability):
    if not np.array_equal(candidate_probability[:, 2], base_probability[:, 2]):
        raise ValueError('local replay')
    if not np.array_equal(candidate_probability[:, 2], saved_probability[:, 2]):
        raise ValueError('archived baseline')
    return True
"""
        self.assertEqual(toy.count(exact_guard), 1)
        corrected = toy.replace(exact_guard, corrected_guard)
        self.assertEqual(corrected.count(corrected_guard), 1)
        self.assertEqual(corrected.count(
            "if not np.array_equal(candidate_probability[:, 2], saved_probability[:, 2]):"), 1)
        namespace = {"np": np}
        exec(compile(corrected, "<synthetic-contact-memory-guard>", "exec"), namespace)
        score = namespace["score"]

        baseline = np.array([[.25, .25, .5]], dtype=np.float64)
        within_tolerance = np.array([[.25, .25, .5000005]], dtype=np.float64)
        self.assertTrue(score(within_tolerance, baseline, within_tolerance))

        beyond_tolerance = np.array([[.25, .25, .500002]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "local replay"):
            score(beyond_tolerance, baseline, beyond_tolerance)

        archived_mismatch = np.array([[.25, .25, .50000001]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "archived baseline"):
            score(baseline, baseline, archived_mismatch)


if __name__ == "__main__":
    unittest.main()
