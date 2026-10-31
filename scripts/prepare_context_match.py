"""Parallel cache preparation for one match using the frozen study's recipe.

Scheduling helper only: preserve the original within-provider augmentation
index. The complete feature and model budget are unchanged. Do not fit here.
"""
from pathlib import Path
import sys
import argparse
import warnings
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import seed_gap_context as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('match')
    args=parser.parse_args()
    spec=s.context_check()
    source,i,path=next((source,i,path) for source,split in s.dt.SPLITS.items()
                       for i,path in enumerate(split['train']) if path.stem==args.match)
    dest=s.OUT/f'train_{path.stem}.joblib'
    if dest.exists():
        if s.sha256(dest)!=s.json.loads(dest.with_suffix('.json').read_text())['sha256']:
            raise ValueError('Changed training cache')
        return
    inputs,labels=s.ms.view(path)
    legacy=s.robust.masked_inputs(inputs,s.lt.SEED+i)
    gap=s.lc.ball_gap_view(inputs,spec['calibration'],s.ce.seed(path.stem,'train'))
    views=dict(clean=inputs,gap=gap,dropout_legacy=s.gr.slot_view(legacy,s.lt.SEED+i),
               dropout_gap=s.gr.slot_view(gap,s.lt.SEED+i))
    records={name:dict(joint=s.old.feature_view(view,labels,source,path.stem)['joint']) for name,view in views.items()}
    joblib.dump(records,dest,compress=3)
    s.write(dest.with_suffix('.json'),dict(sha256=s.sha256(dest),input_sha256=s.sha256(path)))
    print('PREPARED',source,path.stem,'original_augmentation_index',i,flush=True)


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        main()
