import json,joblib
from experiments import feed_robustness_v1 as p,feed_short_sampler as s
p.check()
gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
rows=[]
for fold in p.FOLDS:
 records=p.training(fold)
 names=list(joblib.load(p.sc.initial_path(fold,2026))['feature_names'])
 for seed in p.SEEDS:
  a=s.replay(s.Resampler(records,gar,names,seed,0),seed)
  b=s.replay(s.Resampler(records,gar,names,seed),seed)
  receipt=p.read(p.OUT/'models'/f'{fold}_{seed}.json')
  assert a['exposures']==b['exposures']==receipt['exposures']
  rows.append(dict(fold=fold,seed=seed,baseline=a,targeted=b))
  print(fold,seed,{g:(a['counts'][g]['short_ha_rows']/a['counts'][g]['known_ha_rows'],b['counts'][g]['short_ha_rows']/b['counts'][g]['known_ha_rows']) for g in ('old','new')},flush=True)
p.write(p.RESULTS/'feed_short_sampling_v1_exposure.json',{'train_only':True,'rows':rows,'source_sha256':p.sha(p.ROOT/'experiments/feed_short_sampler.py'),'organizer_reference_access':False})
