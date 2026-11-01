"""Bounded correction of an unsupported aggregate-field replay assertion.

The frozen runner and failed declaration/output remain untouched. The shared
aggregator deliberately omits graded_frames, so that invariant belongs to raw
paired cells. Only that unsupported aggregate lookup is removed in memory;
all scalar, exact-path and per-cell population guards remain in force.
"""
import hashlib
import inspect
import json
from pathlib import Path

from experiments import feed_spell_refine_run as runner


def main():
    original = inspect.getsource(runner.run)
    needle = '("spell_f1","macro_f1","d_f1","graded_frames","reference_spell_count")'
    replacement = '("spell_f1","macro_f1","d_f1","reference_spell_count")'
    if original.count(needle) != 1:
        raise ValueError("Expected exactly one unsupported aggregate population lookup")
    effective = original.replace(needle,replacement)
    failed = runner.OUT
    if not (failed/"lock.json").exists() or list(failed.glob("g*.json")):
        raise ValueError("Expected the preserved pre-candidate failure and no scored arms")
    d=json.loads(runner.DECLARATION.read_text())
    for path, sha in d["source_sha256"].items():
        if runner.shared.sha256_file(runner.ROOT/path)!=sha:
            raise ValueError("Frozen source changed")
    # Explicitly verify population from paired native rows, where it is defined.
    reference=json.loads((runner.first.OUT/"f062_final.json").read_text())["rows"]
    cells=runner.first.paired_cells(d)
    checked=0
    for c in cells:
        if c["member"]!="ensemble":continue
        y=runner.inference.decode(c["members"],c["model_period"],c["fps"],d["arms"][1])
        row=runner.first.identity(c)|runner.scoring.quick_score(c,y)
        before=next(r for r in reference if all(r[k]==row[k] for k in runner.first.identity(row)))
        if any(row[k]!=before[k] for k in ("graded_frames","reference_spell_count")):
            raise ValueError("Raw control population replay failed")
        checked+=1
    receipt={"study":"feed_spell_refine_v2","status":"pre-candidate execution amendment",
        "failure":"KeyError: graded_frames; shared aggregate omits the field",
        "failed_output_preserved":str(failed.relative_to(runner.ROOT)),"candidate_cells_before_correction":0,
        "original_declaration_sha256":runner.shared.sha256_file(runner.DECLARATION),
        "frozen_runner_sha256":runner.shared.sha256_file(Path(inspect.getsourcefile(runner))),
        "runtime_source_sha256":runner.shared.sha256_file(Path(__file__)),
        "original_function_sha256":hashlib.sha256(original.encode()).hexdigest(),
        "effective_function_sha256":hashlib.sha256(effective.encode()).hexdigest(),
        "exact_replacement":{"from":needle,"to":replacement},"raw_population_control_cells":checked,
        "all_arms_parameters_metrics_guards_unchanged":True,
        "corrected_output":"data/experiments/feed_spell_refine_v2/corrected"}
    runner.scoring.write_once(runner.RESULTS/"feed_spell_refine_v2_execution_amendment.json",receipt)
    runner.OUT=failed/"corrected"
    exec(compile(effective,inspect.getsourcefile(runner.run),"exec"),runner.__dict__)
    runner.run()


if __name__=="__main__":main()
