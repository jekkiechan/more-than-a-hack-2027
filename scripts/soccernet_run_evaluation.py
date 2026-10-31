"""Run frozen GAR evaluation with its separately audited numeric guards."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import soccernet_evaluate_runtime as replay
import soccernet_probability_runtime as probability
p=argparse.ArgumentParser();p.add_argument('--match');args=p.parse_args()
replay.amend();probability.amend();replay.study.evaluate(args.match)
