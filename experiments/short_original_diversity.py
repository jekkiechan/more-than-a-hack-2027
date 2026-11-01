"""Unique native short-possession bags within the selected original record.

Keeps the quality/diversity GAR recipe fixed and changes only original-record
window access. All eligibility is TRAIN-only; labels and features are untouched.
"""
from collections import Counter, defaultdict
import random
import numpy as np

from experiments import short_label_quality as quality
from experiments import dense_short_team_v1 as short


class OriginalDiversitySampler(quality.QualitySampler):
    def __init__(self,old_records,new_records,names,seed,rate=.25):
        super().__init__(old_records,new_records,names,seed,diverse=True,rate=rate)
        self.original_rng=random.Random(seed+1106644)
        self.original_bags={}
        self.original_targets=Counter()
        self.original_catalog=defaultdict(dict)
        windows=defaultdict(list)
        for values in self.groups['old'].values():
            for w in values:windows[w[0]].append(w)
        for ri,r in enumerate(old_records):
            y=np.asarray(r['full_y']);idx=np.asarray(r['indices'])
            mask=short.complete_short_spell_mask(y,r['full_period'],float(r.get('native_fps',r['fps'])),3.)
            starts=np.flatnonzero(mask & np.r_[True,~mask[:-1]|(y[:-1]!=y[1:])])
            stops=np.flatnonzero(mask & np.r_[~mask[1:]|(y[:-1]!=y[1:]),True])+1
            for a,b in zip(starts,stops):
                rows=np.flatnonzero((idx>=a)&(idx<b))
                if not len(rows):continue
                lo,hi=int(rows[0]),int(rows[-1])
                choices=[w for w in windows[ri] if w[1]<=hi and w[1]+w[2]>lo]
                if choices:self.original_catalog[ri][(int(a),int(b),int(y[a]))]=choices

    def replace(self,group,window):
        if group=='new':return super().replace(group,window)
        # Consume the parent RNG identically to isolate the GAR schedule. The
        # parent counts refer to its proposed window; replace them with counts
        # for the window actually used. Source/match exposure increments once.
        before=self.counts['old'].copy()
        super().replace('old',window)
        self.counts['old']=before
        selected=window;ri=window[0];entries=self.original_catalog.get(ri,{})
        if self.rate and self.original_rng.random()<self.rate:
            self.counts['old']['requested_replacement']+=1
            if entries:
                if not self.original_bags.get(ri):
                    self.original_bags[ri]=sorted(entries)
                    self.original_rng.shuffle(self.original_bags[ri])
                    self.counts['old']['possession_cycles_started']+=1
                identity=self.original_bags[ri].pop()
                self.original_targets[(ri,identity)]+=1
                selected=self.original_rng.choice(entries[identity])
                self.counts['old']['replacement_draws']+=1
        assert selected[0]==ri
        _,start,size,_,_=selected;r=self.records['old'][ri]
        known=np.isin(np.asarray(r['y'])[start:start+size],(0,1))
        eligible=self.eligible['old'][ri][start:start+size]&known;c=self.counts['old']
        c['draws']+=1;c['known_ha_rows']+=int(known.sum())
        c['short_ha_rows']+=int(eligible.sum());c['short_containing_draws']+=int(eligible.any())
        return selected

    def receipt(self):
        r=super().receipt()
        r['original_sampling']='distinct complete <=3s native possession bags within selected record'
        r['original_eligible_possessions_including_paired_records']=sum(len(x) for x in self.original_catalog.values())
        r['original_nominated_possessions_including_paired_records']=len(self.original_targets)
        return r
