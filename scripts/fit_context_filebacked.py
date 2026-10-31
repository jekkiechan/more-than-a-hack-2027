"""Run the frozen context fit with file-backed copies of raw training arrays.

The mathematical fit is factorized_had.fit verbatim. Raw records are not read
after normalized windows are made, so file backing lets the OS evict those
pages during training. Every raw array is byte-verified before the fit.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import sys
import tempfile
import warnings
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import seed_gap_context as s


@contextmanager
def file_backed_fit():
    original=s.model.fit
    def mapped(records,**kwargs):
        s.OUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='raw_fit_',dir=s.OUT) as folder:
            path=Path(folder)/'records.joblib'
            joblib.dump(records,path,compress=0)
            restored=joblib.load(path,mmap_mode='r')
            assert len(restored)==len(records)
            arrays=0
            for a,b in zip(records,restored):
                assert a.keys()==b.keys()
                for key in a:
                    if isinstance(a[key],np.ndarray):
                        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
                        assert s.digest_array(a[key])==s.digest_array(b[key])
                        arrays+=1
                    else:
                        assert a[key]==b[key]
            # The study runner never accesses its raw records after model.fit.
            # Release its list in-place; preserve the verified file-backed list.
            records.clear()
            del a,b
            print('FILE_BACKED_RAW_ARRAYS_VERIFIED',arrays,flush=True)
            return original(restored,**kwargs)
    try:
        s.model.fit=mapped
        yield
    finally:
        s.model.fit=original


def contract():
    """Exact state parity for plain and file-backed calls to the same fit."""
    rng=np.random.default_rng(13)
    records=[]
    for i in range(2):
        x=rng.normal(size=(80,3)).astype(np.float32)
        x[8:12,0]=np.nan
        records.append(dict(x=x,alternate=x.copy(),physical_x=x.copy(),
            y=np.tile(np.array([0,1,2,-1],np.int8),20),
            period=np.zeros(80,int),source='synthetic',match=str(i),
            names=['H_test','A_test','D_test'],replace_rows=np.zeros(80,bool),
            replace_windows=np.zeros(2,bool)))
    with s.context(15):
        plain=s.model.fit(records,player_only=True,seed=2026)
        with file_backed_fit():
            backed=s.model.fit(records.copy(),player_only=True,seed=2026)
    for key in plain:
        if key=='state':
            assert all(np.array_equal(plain[key][k],backed[key][k]) for k in plain[key])
        elif isinstance(plain[key],np.ndarray):
            assert np.array_equal(plain[key],backed[key])
        else: assert plain[key]==backed[key]
    print('EXACT_FILE_BACKED_FIT_CONTRACT_PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=tuple(s.dt.SPLITS))
    parser.add_argument('--seed',type=int,choices=s.SEEDS)
    parser.add_argument('--contract',action='store_true')
    args=parser.parse_args()
    if args.contract: contract()
    else:
        if args.source is None or args.seed is None:parser.error('source and seed required')
        with file_backed_fit():s.fit(args.source,args.seed)


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        main()
