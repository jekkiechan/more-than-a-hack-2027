"""TRAIN-only within-record short-possession window resampling.

The base RNG, source/match/record selection and trainer remain unchanged.
A separate RNG replaces a fixed fraction of draws by eligible windows from
that same record. Native complete-span eligibility never reaches inference.
"""
from contextlib import contextmanager
from collections import Counter, defaultdict
import random
import numpy as np
from experiments import soccernet_capacity_model as cap
from experiments import dense_short_team_v1 as short

RATE = .25
SECONDS = 3.0


def masks(records):
    return [short.short_spell_frame_weights(r, threshold_seconds=SECONDS) > 1 for r in records]


def pools(groups, eligible):
    result = defaultdict(list)
    for windows in groups.values():
        for w in windows:
            ri, start, size, _, _ = w
            if eligible[ri][start:start+size].any():
                result[ri].append(w)
    return dict(result)


class Resampler:
    def __init__(self, old_records, new_records, names, seed, rate=RATE):
        if not 0 <= rate <= 1:
            raise ValueError('Replacement rate outside [0,1]')
        self.rate = rate
        self.rng = random.Random(seed + 773311)
        self.groups = {}
        self.eligible = {}
        self.pool = {}
        self.records = {'old': old_records, 'new': new_records}
        self.counts = {g: Counter() for g in self.records}
        self.exposures = Counter()
        for g, records in self.records.items():
            groups, _ = cap._window_index(records, names, old=g=='old')
            self.groups[g] = groups
            self.eligible[g] = masks(records)
            self.pool[g] = pools(groups, self.eligible[g])

    def replace(self, group, window):
        ri = window[0]
        selected = window
        if self.rate and self.rng.random() < self.rate:
            self.counts[group]['requested_replacement'] += 1
            candidates = self.pool[group].get(ri, ())
            if candidates:
                selected = self.rng.choice(candidates)
                self.counts[group]['replacement_draws'] += 1
        assert selected[0] == ri
        _, start, size, _, _ = selected
        record = self.records[group][ri]
        labels = np.asarray(record['y'])[start:start+size]
        known = np.isin(labels, (0,1))
        eligible = self.eligible[group][ri][start:start+size] & known
        c = self.counts[group]
        c['draws'] += 1
        c['known_ha_rows'] += int(known.sum())
        c['short_ha_rows'] += int(eligible.sum())
        c['short_containing_draws'] += int(eligible.any())
        identity = ('old/'+str(record['source'])+'/'+str(record['match']) if group=='old'
                    else 'new/'+str(record['match']))
        self.exposures[identity] += 1
        return selected

    def receipt(self):
        return {'rate': self.rate, 'seconds': SECONDS,
                'counts': {g: dict(c) for g,c in self.counts.items()},
                'exposures': dict(sorted(self.exposures.items())),
                'eligible_windows': {g:sum(map(len,p.values())) for g,p in self.pool.items()},
                'eligible_records': {g:len(p) for g,p in self.pool.items()}}


@contextmanager
def installed(sampler):
    old_draw, new_draw = cap._draw_old, cap._draw_new
    cap._draw_old = lambda rng,groups,records: sampler.replace('old',old_draw(rng,groups,records))
    cap._draw_new = lambda rng,groups: sampler.replace('new',new_draw(rng,groups))
    try:
        yield sampler
    finally:
        cap._draw_old, cap._draw_new = old_draw,new_draw


def replay(sampler, seed, steps=2048):
    rng = random.Random(seed)
    with installed(sampler):
        for _ in range(steps):
            for _ in range(16):cap._draw_old(rng,sampler.groups['old'],sampler.records['old'])
            for _ in range(16):
                cap._draw_new(rng,sampler.groups['new']);rng.random()
            order=list(range(32));rng.shuffle(order)
    return sampler.receipt()
