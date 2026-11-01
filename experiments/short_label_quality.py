"""TRAIN-only short-span corroboration and possession-level diversity sampling.

Silver labels remain unchanged. Geometry is a conservative sampling proxy,
never a gold-label claim or an inference feature/target transform.
"""
from collections import Counter, defaultdict
import random

import numpy as np

from experiments import dense_short_team_v1 as short
from experiments import feed_short_sampler as previous


def spans(record):
    """Return complete <=3s native possessions with independent proxy checks."""
    y = np.asarray(record['full_y'])
    period = np.asarray(record['full_period'])
    indices = np.asarray(record['indices'])
    fps = float(record.get('native_fps', record['fps']))
    mask = short.complete_short_spell_mask(y, period, fps, 3.)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1] | (y[:-1] != y[1:])])
    stops = np.flatnonzero(mask & np.r_[~mask[1:] | (y[:-1] != y[1:]), True]) + 1
    names = list(record['names'])
    runs = np.asarray(record['touch_runs'])
    result = []
    for a, b in zip(starts, stops):
        team = int(y[a])
        rows = np.flatnonzero((indices >= a) & (indices < b))
        x = np.asarray(record['x'])[rows]
        own = x[:, names.index(('H' if team == 0 else 'A') + '_distance1')]
        opposing = x[:, names.index(('A' if team == 0 else 'H') + '_distance1')]
        observed = x[:, names.index('ball_observed')] > .5
        geometric = observed & np.isfinite(own) & np.isfinite(opposing)
        supported = geometric & (opposing - own >= .5)
        anchors = np.asarray(record['anchor_full_y'])[a:b]
        conflict = bool((np.isin(anchors, (0, 1)) & (anchors != team)).any())
        proof = runs[(runs[:, 0] < b) & (runs[:, 1] >= a) &
                     (runs[:, 2] == team) & (runs[:, 4] == 1) & (runs[:, 5] == 0)]
        frame = np.asarray(record['full_frame_numbers'])
        identity = (str(record['match']), int(record['game_period']),
                    int(frame[a]), int(frame[b-1]), team)
        result.append(dict(start=int(a), stop=int(b), team=team,
            identity=identity, seconds=float((b-a)/fps), sampled_rows=rows.tolist(),
            supported_rows=int(supported.sum()), no_observed_ball=not bool(geometric.any()),
            anchor_conflict=conflict, controlled_proof=bool(len(proof)),
            held=bool((np.asarray(record['full_prov'])[a:b] == 4).any()),
            quality=bool(len(proof) and not conflict and supported.sum() >= 2)))
    return result


def catalog(records, groups):
    """Map each distinct eligible possession to existing overlapping windows."""
    windows = defaultdict(list)
    for values in groups.values():
        for w in values:
            windows[w[0]].append(w)
    result = defaultdict(dict)
    counts = Counter()
    per_game = defaultdict(Counter)
    for ri, record in enumerate(records):
        for span in spans(record):
            game = str(record['match'])
            counts['complete_short_spans'] += 1
            for k in ('no_observed_ball', 'anchor_conflict', 'controlled_proof', 'held', 'quality'):
                counts[k] += int(span[k])
            counts['no_sampled_rows'] += int(not span['sampled_rows'])
            per_game[game]['all'] += 1
            if not span['quality']:
                continue
            per_game[game]['quality'] += 1
            lo, hi = min(span['sampled_rows']), max(span['sampled_rows'])
            # Same existing-window criterion as previous short sampling: any
            # eligible sampled row in the core. Context/window metadata retained.
            choices = [w for w in windows[ri] if w[1] <= hi and w[1]+w[2] > lo]
            if not choices:
                continue
            identity = span['identity']
            if identity in result[game]:
                raise ValueError('Duplicate native possession identity')
            result[game][identity] = choices
    return dict(result), dict(counts), {g: dict(c) for g, c in sorted(per_game.items())}


class QualitySampler(previous.Resampler):
    """Matched original sampling; quality-gated GAR windows or unique-span bags."""
    def __init__(self, old_records, new_records, names, seed, *, diverse=False, rate=.25):
        super().__init__(old_records, new_records, names, seed, rate)
        self.diverse = bool(diverse)
        self.quality_rng = random.Random(seed + 995533)
        self.catalog, self.quality_counts, self.per_game = catalog(new_records, self.groups['new'])
        self.bags = {}
        self.seen = Counter()
        self.targets = Counter()
        self.selected_windows = Counter()
        self.window_pool = {}
        self.window_spans = defaultdict(set)
        for game, entries in self.catalog.items():
            unique = {}
            for identity, choices in entries.items():
                for w in choices:
                    unique[(w[0], w[1])] = w
                    self.window_spans[(w[0], w[1])].add(identity)
            self.window_pool[game] = list(unique.values())

    def replace(self, group, window):
        if group == 'old':
            return super().replace(group, window)
        ri = window[0]
        game = str(self.records['new'][ri]['match'])
        selected = window
        # Preserve all original-provider sampling decisions from the numerical
        # leader by consuming its separate within-record RNG identically.
        if self.rate and self.rng.random() < self.rate:
            pool = self.pool['new'].get(ri, ())
            if pool:
                self.rng.choice(pool)
        if self.rate and self.quality_rng.random() < self.rate:
            self.counts['new']['requested_replacement'] += 1
            entries = self.catalog.get(game, {})
            if entries:
                if self.diverse:
                    if not self.bags.get(game):
                        self.bags[game] = sorted(entries)
                        self.quality_rng.shuffle(self.bags[game])
                        self.counts['new']['possession_cycles_started'] += 1
                    identity = self.bags[game].pop()
                    selected = self.quality_rng.choice(entries[identity])
                    self.targets[identity] += 1
                else:
                    selected = self.quality_rng.choice(self.window_pool[game])
                for identity in self.window_spans[(selected[0], selected[1])]:
                    self.seen[identity] += 1
                self.selected_windows[(selected[0], selected[1])] += 1
                self.counts['new']['replacement_draws'] += 1
        ri, start, size, _, _ = selected
        record = self.records['new'][ri]
        assert str(record['match']) == game
        known = np.isin(np.asarray(record['y'])[start:start+size], (0,1))
        eligible = self.eligible['new'][ri][start:start+size] & known
        c = self.counts['new']
        c['draws'] += 1
        c['known_ha_rows'] += int(known.sum())
        c['short_ha_rows'] += int(eligible.sum())
        c['short_containing_draws'] += int(eligible.any())
        self.exposures['new/'+game] += 1
        return selected

    def receipt(self):
        value = super().receipt()
        n = sum(self.seen.values())
        per_game = defaultdict(list)
        for identity, count in self.seen.items():
            per_game[identity[0]].append(count)
        value.update(diverse=self.diverse, quality_counts=self.quality_counts,
            quality_per_game=self.per_game, eligible_quality_games=len(self.catalog),
            replacement_distinct_possessions=len(self.seen),
            replacement_distinct_windows=len(self.selected_windows),
            nominated_distinct_possessions=len(self.targets),
            replacement_effective_possessions=float(n*n/sum(v*v for v in self.seen.values())) if n else 0.,
            replacement_repeat_range_per_game={g:[min(v),max(v)] for g,v in sorted(per_game.items())})
        return value
