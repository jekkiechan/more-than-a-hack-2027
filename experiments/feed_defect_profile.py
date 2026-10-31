"""Label-free defect profile: organizer feed vs external clean and stress views.

Reads tracking inputs only. Reports ball gaps, simultaneous player loss
(whole-frame and regional blanking signatures), per-track gap lengths and
track fragmentation, so stress recipes can be compared with the real feed.
"""
from __future__ import annotations
import json
import numpy as np
from experiments import (multisource_eval as ms, dead_transfer as dt, supervised,
                         tracking_corruption as tc, corruption_experiment as ce, live_gap_experiment as old)

ROOT = old.ROOT


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def profile(data):
    data = tc.inputs_only(data)
    fps = float(data['fps'])
    active = np.asarray(data['period']) > 0
    ball = np.isfinite(data['ball']).all(axis=1)
    seen = np.isfinite(data['players']).all(axis=2)
    count = seen.sum(axis=1)
    ball_gaps, slot_gaps, drops = [], [], []
    frames = 0
    blank = partial = big_drop = 0
    tracks = 0
    for a, b in supervised.blocks(supervised.model_periods(data)):
        if not active[a]:
            continue
        frames += b-a
        s, e = runs(~ball[a:b])
        ball_gaps.extend((e-s)/fps)
        c = count[a:b]
        blank += int((c == 0).sum())
        # Frames that lose >=25% of the players visible one frame earlier.
        prev = seen[a:b-1]; cur = seen[a+1:b]
        lost = (prev & ~cur).sum(axis=1)
        frac = np.divide(lost, prev.sum(axis=1), out=np.zeros(len(lost)), where=prev.sum(axis=1) > 0)
        big_drop += int((frac >= .25).sum())
        partial += int(((lost >= 3)).sum())
        for slot in range(seen.shape[1]):
            obs = np.flatnonzero(seen[a:b, slot])
            if not len(obs):
                continue
            tracks += 1
            inner = ~seen[a+obs[0]:a+obs[-1]+1, slot]
            s, e = runs(inner)
            slot_gaps.extend((e-s)/fps)
    minutes = frames/fps/60
    q = lambda x, p: np.quantile(x, p).round(3).tolist() if len(x) else []
    return dict(active_minutes=round(minutes, 2),
                ball_missing_fraction=round(float((~ball[active]).mean()), 4),
                ball_gap_seconds_q10_50_90_99=q(ball_gaps, [.1, .5, .9, .99]),
                ball_gaps_per_minute=round(len(ball_gaps)/minutes, 2),
                visible_players_q01_10_50=q(count[active], [.01, .1, .5]),
                zero_player_frame_fraction=round(blank/frames, 5),
                frames_losing_25pct_players_per_minute=round(big_drop/minutes, 3),
                frames_losing_3plus_players_per_minute=round(partial/minutes, 3),
                within_track_gaps_per_track_minute=round(len(slot_gaps)/max(tracks, 1)/minutes, 4),
                within_track_gap_seconds_q50_90=q(slot_gaps, [.5, .9]),
                tracks_with_observations=tracks)


def main():
    out = {}
    z = np.load(ROOT/'data/challenge/inputs.npz', allow_pickle=False)
    challenge = {k: z[k] for k in z.files if k != 'labels'}
    out['BAR-ATH organizer feed'] = profile(challenge)
    calibration = json.loads((ROOT/'experiments/results/realistic_corruption_development.json').read_text())['spec']['calibration']
    for source in dt.SPLITS:
        for path in dt.SPLITS[source]['eval']:
            original, _ = ms.view(path)
            out[f'{path.stem} clean'] = profile(original)
            out[f'{path.stem} composite corrupt_1'] = profile(tc.corrupt(original, calibration, ce.seed(path.stem, 'corrupt_1')))
            print(path.stem, flush=True)
    (ROOT/'experiments/results/feed_defect_profile.json').write_text(json.dumps(out, indent=1))
    keys = list(next(iter(out.values())))
    for k in keys:
        print(f'\n{k}')
        for name, v in out.items():
            print(f'  {name:45s} {v[k]}')


if __name__ == '__main__':
    main()
