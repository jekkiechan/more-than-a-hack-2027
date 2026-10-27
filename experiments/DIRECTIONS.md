# Preprocessing review and research rationale

The active prioritized queue is [STATUS](../docs/research/STATUS.md).
Completed and draft work is indexed in [EXPERIMENTS](../docs/research/EXPERIMENTS.md).
This dated review explains the original preprocessing hypotheses; its
conclusions describe evidence available on 2026-10-26, before the later joint
GRU D-repair study. It does not establish today's bottleneck for every model.

The [completed tracking pilot](TOP_THREE_PILOT.md) tested state-conditioned
views, broadcast recovery and gap/de-duplication features. It found no matched
overall spell gain; calibration and ball confidence remain unresolved. Read its
results before proposing those arms again.

Superseded proposals and the previous priority list remain in
[Git history](https://github.com/jekkiechan/more-than-a-hack-2027/blob/1b3fc4a4ed1868bf8c7ca26c99ec443947cf3458/experiments/DIRECTIONS.md).
They are not an additional active queue.

## Why preprocessing has not moved the scores (2026-10-26 review)

This review reads only existing code and recorded aggregate results; no new
fitting or scoring was run. The current package applies **no** input
preprocessing at inference: `current_baseline.py` passes raw tracking to the
feature extractors. The observation repair in draft PR #8 (isolated spike
removal, player gaps up to 0.2 s) scored .5251 versus .5259 spell F1 for its
matched baseline, and was never merged. Corruption training traded clean for
stress performance. Five reasons explain why neither shows a gain:

1. **Development data lacks the defects being repaired.** Sportec has complete
   ball XY (frozen during D), Metrica blanks the ball almost only during D, and
   SkillCorner's gaps come from broadcast cuts. None has the challenge's
   live-play ball loss, 46 fragmenting tracklets, or up to 19 visible tracks
   per team. Repairing clean inputs gives a near-zero delta by construction.
2. **Degraded views do not exercise the repairs.** They come from the same
   synthetic families used in training augmentation (long gaps, slot swaps),
   while the repairs target isolated spikes and subsecond gaps.
3. **The repairs work at the wrong scale.** Organizer ball gaps have a median
   of 10.7 s and cover about 41% of graded frames; the repairs bridge at most
   0.5 s (ball) and 0.2 s (players). HGB already handles NaN natively and the
   feature code already rejects >13 m/s player and >65 m/s ball speeds. During
   long gaps every ball-relative feature is NaN and H/A rests on player motion
   and contact memory.
4. **Synthetic gaps are placed independently of game state.** Real occlusion
   likely concentrates on aerial balls, duels and crowded areas, i.e. around
   turnovers. A renewal-process mask teaches "gaps are random", which is why
   corruption training improves only its own stress family.
5. **The public window cannot resolve the effect.** The .4889 to .5257 public
   change is one fewer spurious possession (8 to 7) out of 17 reference
   possessions; the 90% spell interval is [.3657, .6743]. The wide interval does
   not establish a reliable model ordering, and the window must not be used for
   tuning.

The recorded public diagnostics also locate the remaining error: alive/dead F1
is .968 and reference-live team accuracy 86–87%, but only 2 of 20 H/A/D
transitions fall within ±0.5 s (5 of 20 within ±1 s). Externally, 98% of wrong
frames are H/A swaps and oracle-perfect H/A gives .996 spell F1 on J03WOY
([HEADROOM.md](HEADROOM.md)). The bottleneck is team attribution during ball
gaps and turnover timing, not dead-ball detection or subsecond repair.
