"""Two fixed, tracking-only hypotheses for exits hidden between observations."""
from __future__ import annotations

import numpy as np

from experiments import ball_exit_dmask_v1 as frozen
from experiments import supervised


def refine_start(dead, ball, pitch, period, fps, method):
    """Move snap's outside-observation start to its bracketed crossing.

    The observation bracket must fit the inherited five-second delay window
    and one model block. Existing D run ends are preserved. Missing evidence
    leaves snap unchanged. No targets or provider identity are accepted.
    """
    if method not in ("linear", "midpoint"):
        raise ValueError(method)
    ball = np.asarray(ball, dtype=float)
    inside, outside = frozen.ball_state(ball, pitch, 0.0)
    out = np.array(dead, dtype=bool, copy=True)
    for start, stop in supervised.blocks(np.asarray(period)):
        for a, b in frozen._runs(np.asarray(dead)[start:stop]):
            a, b = a + start, b + start
            if not outside[a]:
                continue
            low = max(start, a - int(round(frozen.WINDOW_S * fps)))
            seen = np.flatnonzero(inside[low:a] | outside[low:a])
            if not len(seen):
                continue
            left = low + seen[-1]
            if not inside[left] or out[left:a].any():
                continue
            alpha = 0.5
            if method == "linear":
                # First intersection of the straight segment with the rectangle.
                hits = []
                for axis, bound in enumerate(pitch):
                    x0, x1 = ball[left, axis], ball[a, axis]
                    if x1 < 0:
                        hits.append(-x0 / (x1 - x0))
                    elif x1 > bound:
                        hits.append((bound - x0) / (x1 - x0))
                alpha = min(hits)
            boundary = left + int(np.ceil(alpha * (a - left)))
            boundary = min(a, max(left + 1, boundary))
            out[boundary:a] = True
    return out
