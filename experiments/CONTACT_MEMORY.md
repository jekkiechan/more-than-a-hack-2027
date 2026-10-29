# Contact-memory residual experiment

The contact-memory adapter did not improve the current combined-GRU ensemble's primary clean spell F1. The ensemble scored **0.628430 clean** and **0.499523 feed-like**, versus **0.633205** and **0.498830** for the frozen baseline. Clean spell F1 fell by **0.004775**; feed-like spell F1 rose by **0.000694**. The baseline remains the selected model.

| Measure | Clean delta | Feed-like delta |
| --- | ---: | ---: |
| Official spell F1 | -0.004775 | +0.000694 |
| Macro F1 | -0.001025 | -0.001184 |
| 70/30 auxiliary summary | -0.003650 | +0.000130 |
| Short-spell recall | +0.003436 | +0.010732 |

The original frozen declaration required positive clean spell F1, and the separately recorded high-value screen required at least **+0.01** clean spell F1. Both clean criteria fail. The feed-like floor of -0.01 and the per-provider clean spell, macro, and D-F1 floors of -0.02 pass. Short-spell recall improved on average while the official clean spell score declined.

| Held-out provider | Clean spell F1 delta | Clean macro F1 delta | Clean D-F1 delta | Feed-like spell F1 delta |
| --- | ---: | ---: | ---: | ---: |
| Metrica | -0.008181 | -0.000604 | -0.000173 | +0.004833 |
| SkillCorner | -0.003262 | -0.001287 | -0.000391 | -0.004201 |
| Sportec | -0.002882 | -0.001184 | -0.001211 | +0.001449 |

| Seed | Clean spell F1 delta | Feed-like spell F1 delta |
| --- | ---: | ---: |
| 2026 | -0.001522 | -0.002585 |
| 2027 | -0.003741 | +0.001475 |

The treatment froze each provider-held-out combined-objective GRU, its training-only normalization, and its player-only D path. Six small adapters were fitted, one per held-out provider and seed. Each adapter adds a zero-initialized, tanh-bounded H/A logit residual using frozen hidden/logit state and two sparse contact-event slots. A residual is exactly zero when neither event is available and is bounded to ±0.5 logits. At each 5 Hz query, memory uses only observed tracking data: the nearest completed prior event and starting future event, each stable for at least 0.2 seconds, with query-local support no older than 10 seconds. Native player and ball observation masks are applied, and event metadata and possession targets are ignored.

Training reused the four cached clean/gap/player-dropout views, their direct/team-swapped records, selected windows, spell-duration weighting, and live-transition loss. All four regenerated views for each of the 13 training matches passed feature, label-vector, and sample-index parity against the frozen baseline caches before fitting. The held-out provider was excluded from both backbone and adapter training. Frozen archived member P(D) was copied exactly into each candidate and only H/A conditional mass was rescaled; all 30 member outputs and 15 ensemble outputs passed bit-exact archived P(D) checks. Viterbi can still change decoded D labels when H/A probabilities change.

During the first scoring attempt, one same-checkpoint local P(D) replay comparison failed exact equality on SkillCorner `skillcorner_had_1996435` clean. A label-free replay probe measured maximum drift of **3.58e-7** against the archived baseline. The frozen runner and declaration remain unchanged. The correction receipt records a runtime wrapper that relaxes only this local-to-local replay guard to absolute tolerance 1e-6; archived member and ensemble P(D) checks remain exact. It preserved 12 completed cell artifacts and their reconstructed rows, then resumed only the three missing cells. The final outputs were audited again against the archived D arrays. See [the runtime correction receipt](results/contact_memory_runtime_correction.json) and [the wrapper](contact_memory_runtime_resume.py).

This is development evidence on previously exposed matches, not independent confirmation. Metrica and SkillCorner use event-derived silver labels; feed-like inputs are synthetic stress views. The adapter's training hidden states come from frozen backbones evaluated on their own training matches, so adapter rows are not cross-fitted. The adapter also receives frozen hidden/logit context and there was no capacity-matched hidden/logits-only adapter control; any difference measures the full adapter package and cannot be attributed specifically to the contact-event tokens.

The frozen experiment budget used six adapter fits and 15 ensemble scoring cells with 30 seed diagnostics. No extra arms or sweeps were run, no organizer labels were read, and neither the baseline nor decision policy was changed.

Artifacts: [frozen declaration](results/contact_memory_declaration.json), [aggregate readout](results/contact_memory_readout.json), [high-value screen](results/spell_f1_followup_high_value_screen.json), [contact-memory features](contact_memory.py), [experiment runner](contact_memory_experiment.py), [memory contracts](tests/test_contact_memory.py), [adapter contracts](tests/test_contact_memory_experiment.py), and [runtime correction contract](tests/test_contact_memory_runtime_resume.py). Private checkpoints and probability arrays are in `data/experiments/contact_memory_v1`.

Independent replay of all 15 ensemble and 30 seed outputs verifies official decodes, scoring, equal-provider aggregation, exact archived P(D), source/model seals and the one-guard runtime correction. See the [independent verification receipt](results/contact_memory_independent_verification.json).

The [shared artifact receipt](results/spell_f1_followup_artifacts.json) records the locally verified archive. Remote upload was blocked by automatic approval review pending explicit authorization; no fresh remote restore is claimed.
