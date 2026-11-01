"""Extend short-window access across records within an already selected GAR game."""
import numpy as np
import random
from collections import defaultdict
from experiments.feed_short_sampler import Resampler as WithinRecord

class GameResampler(WithinRecord):
    def __init__(self,old_records,new_records,names,seed,rate=.25):
        super().__init__(old_records,new_records,names,seed,rate)
        self.game_rng=random.Random(seed+884422)
        self.game_pool=defaultdict(list)
        for ri,windows in self.pool['new'].items():
            self.game_pool[str(self.records['new'][ri]['match'])].extend(windows)

    def replace(self,group,window):
        if group=='old':return super().replace(group,window)
        ri=window[0];game=str(self.records['new'][ri]['match']);selected=window
        # Consume the within-record RNG exactly as the matched control, so old
        # sampling decisions remain identical. GAR replacement has its own RNG.
        if self.rate and self.rng.random()<self.rate:
            original_pool=self.pool['new'].get(ri,())
            if original_pool:self.rng.choice(original_pool)
        if self.rate and self.game_rng.random()<self.rate:
            self.counts[group]['requested_replacement']+=1
            candidates=self.game_pool.get(game,())
            if candidates:
                selected=self.game_rng.choice(candidates);self.counts[group]['replacement_draws']+=1
        ri,start,size,_,_=selected;record=self.records[group][ri]
        assert str(record['match'])==game
        known=np.isin(np.asarray(record['y'])[start:start+size],(0,1))
        eligible=self.eligible[group][ri][start:start+size]&known;c=self.counts[group]
        c['draws']+=1;c['known_ha_rows']+=int(known.sum());c['short_ha_rows']+=int(eligible.sum());c['short_containing_draws']+=int(eligible.any())
        self.exposures['new/'+game]+=1
        return selected

    def receipt(self):
        value=super().receipt();value['eligible_gar_games']=len(self.game_pool)
        value['scope']='old: within record; GAR: within selected game'
        return value
