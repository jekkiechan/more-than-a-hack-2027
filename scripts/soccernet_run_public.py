"""Run the single public diagnostic with the audited probability-sum guard."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import soccernet_probability_runtime as probability
from experiments import soccernet_public_check as public
p=argparse.ArgumentParser();p.add_argument('action',choices=('predict','score'));args=p.parse_args()
probability.amend();getattr(public,args.action)()
