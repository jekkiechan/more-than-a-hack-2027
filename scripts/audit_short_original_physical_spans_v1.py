"""Deduplicate paired TRAIN records when reporting original-span diversity."""
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import short_original_diversity_v1 as study
from experiments import dense_short_team_v1 as short
p=study.p


def main():
    rows=[]
    for fold in p.FOLDS:
        records=p.training(fold);spans=Counter()
        for r in records:
            y=np.asarray(r['full_y']);idx=np.asarray(r['indices'])
            mask=short.complete_short_spell_mask(y,r['full_period'],float(r.get('native_fps',r['fps'])),3.)
            starts=np.flatnonzero(mask & np.r_[True,~mask[:-1]|(y[:-1]!=y[1:])])
            stops=np.flatnonzero(mask & np.r_[~mask[1:]|(y[:-1]!=y[1:]),True])+1
            for a,b in zip(starts,stops):
                if np.any((idx>=a)&(idx<b)):spans[(str(r['match']),int(a),int(b))]+=1
        expected=next(r['sampling']['original_eligible_possessions_including_paired_records'] for r in p.read(study.AUDIT)['rows'] if r['fold']==fold)
        assert sum(spans.values())==expected
        rows.append(dict(fold=fold,unique_physical_spans=len(spans),paired_record_spans=sum(spans.values()),
            multiplicity_range=[min(spans.values()),max(spans.values())],train_matches=sorted({str(r['match']) for r in records})))
    result=dict(train_only=True,rows=rows,source_sha256=p.sha(Path(__file__)),
        interpretation='Match plus native endpoints deduplicate paired feature views and H/A swaps. No new physical examples, labels, evaluation or fitting.')
    p.write(p.RESULTS/f'{study.STUDY}_physical_span_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
