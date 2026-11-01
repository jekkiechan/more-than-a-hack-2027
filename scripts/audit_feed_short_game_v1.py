"""TRAIN-only audit of game-level access to eligible GAR windows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
from experiments import feed_short_sampling_v1 as study,feed_short_sampler as base
from experiments.feed_short_game_sampler import GameResampler
p=study.p
study.check();gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS);rows=[]
for fold in p.FOLDS:
 records=p.training(fold);names=list(joblib.load(p.sc.initial_path(fold,2026))['feature_names'])
 for seed in p.SEEDS:
  observed=base.replay(GameResampler(records,gar,names,seed),seed)
  control=next(r for r in p.read(study.AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)
  assert observed['exposures']==control['targeted']['exposures']
  rows.append({'fold':fold,'seed':seed,'game_sampling':observed})
  print(fold,seed,observed['eligible_gar_games'],observed['counts']['new']['short_ha_rows']/observed['counts']['new']['known_ha_rows'],flush=True)
p.write(p.RESULTS/'feed_short_game_v1_exposure.json',{'train_only':True,'rows':rows,'sampler_sha256':p.sha(p.ROOT/'experiments/feed_short_game_sampler.py'),'organizer_reference_access':False})
