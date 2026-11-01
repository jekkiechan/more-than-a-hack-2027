"""Assemble the completed saved-metric readout; never performs inference/scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments/results'


def read(name):
    return json.loads((RESULTS/name).read_text())


def digest(name):
    return hashlib.sha256((RESULTS/name).read_bytes()).hexdigest()


def write_new(name,data):
    with (RESULTS/name).open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True)
        f.write('\n')


def fmt(value):
    return f'{value:.9f}'


def build():
    original=read('original_full_phase_average_scorecard_v1.json')
    guards=read('phase_average_guard_v1.json')
    gar=read('gar_deployment_eval_v2.json')
    verified=read('gar_deployment_eval_v2_verification.json')
    advisory=read('original_full_phase_average_astra_review_v2.json')
    guard_verification=read('phase_average_guard_v1_verification.json')
    combined=dict(original,
        original_five_match_extended_metrics=guards['summaries'],
        active_policy_guard_completion=guards,
        original_five_class_and_iou_metrics=guard_verification,
        gar_exact_pooled_packages=gar,
        gar_metric_verification=verified,
        promotion_review=advisory,
        missing_metrics='Original-five v2 D/per-seed metrics were not published. Clean/v1 provider and seed gaps are now filled by the fixed guard audit.',
        additional_source_sha256={n:digest(n) for n in ['phase_average_guard_v1.json','gar_deployment_eval_v2.json',
            'gar_deployment_eval_v2_verification.json','original_full_phase_average_astra_review_v2.json',
            'phase_average_guard_v1_verification.json']})
    write_new('original_full_phase_average_scorecard_v2.json',combined)
    chunks=['\n## Completed original-provider guards and component seeds\n',
        '\nThe fixed audit replays both published phase0/phase-average spell and macro scorecards '
        'exactly on the same exposed matches. It adds metrics, not independent evidence. '
        f"All active v1 guard checks pass: **{guards['all_active_policy_guards_pass']}**. "
        'The historical +.01 clean screen still fails.\n',
        '\n| Provider | Phase0 clean D-F1 | Phase-average clean D-F1 | Delta | Phase-average v1 D-F1 |\n',
        '| --- | ---: | ---: | ---: | ---: |\n']
    summaries=guards['summaries']
    initial,phase=summaries['ensemble']['phase0'],summaries['ensemble']['phase_avg']
    for provider in ('Metrica','SkillCorner','Sportec'):
        a=initial['clean']['d_f1']['providers'][provider]
        b=phase['clean']['d_f1']['providers'][provider]
        feed=phase['feed_like']['d_f1']['providers'][provider]
        chunks.append(f'| {provider} | {fmt(a)} | {fmt(b)} | {b-a:+.9f} | {fmt(feed)} |\n')
    chunks.extend(['\n| Component seed | Mode | Clean spell / macro / D-F1 | V1 spell / macro / D-F1 |\n',
        '| --- | --- | --- | --- |\n'])
    for seed in ('2026','2027'):
        for mode in ('phase0','phase_avg'):
            row=summaries[seed][mode]
            def triple(view):return ' / '.join(fmt(row[view][m]['equal_provider_mean']) for m in ('spell_f1','macro_f1','d_f1'))
            chunks.append(f'| {seed} | {mode} | {triple("clean")} | {triple("feed_like")} |\n')
    chunks.extend(['\nThe [guard audit](PHASE_AVERAGE_GUARD_V1.md) and '
        '[full expanded scorecard](results/original_full_phase_average_scorecard_v2.json) '
        'include every provider/seed metric and per-IoU metric available in the saved evidence.\n',
        '\n## Completed exact-package GAR evaluation\n',
        '\nThese scores use exact pooled deployment checkpoints. Ten TEST games are the primary '
        'matched comparison; the nine previously selected-on VALID games are reported separately. '
        'GAR uses six native sampling phases, preserving 29.97 Hz tracking; the target feed uses five at 25 Hz. '
        'The sources are event-derived silver labels inside discontinuous clips.\n',
        '\n| Split / view | Configuration | Spell F1 | Macro F1 | D-F1 | D precision | D recall | Short-spell recall |\n',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'])
    for split in ('test','valid'):
        for view in ('clean','v2'):
            for arm,mode in [('original_full','phase0'),('original_full','phase_avg'),('calibrated','phase0'),('calibrated','phase_avg')]:
                m=gar['splits'][split]['scores'][view][arm][mode]['mean']
                values=' | '.join(fmt(m[k]) for k in ('spell_f1','macro_f1','d_f1','d_precision','d_recall','short_spell_recall_iou_0_5_lt3s'))
                chunks.append(f'| {split.upper()} / {view} | {arm} {mode} | {values} |\n')
    chunks.extend(['\n| Split | Games | Records | Graded frames | Graded hours |\n',
        '| --- | ---: | ---: | ---: | ---: |\n'])
    for split in ('test','valid'):
        s=gar['splits'][split]
        chunks.append(f'| {split.upper()} | {s["games"]} | {s["records"]:,} | {s["graded_frames"]:,} | {s["graded_hours"]:.6f} |\n')
    chunks.extend(['\n| TEST comparison | View | Spell delta | Paired 95% interval | Games with positive delta |\n',
        '| --- | --- | ---: | --- | ---: |\n'])
    for view in ('clean','v2'):
        for name,label in [('phase_gain','Original_full phases − same package phase0'),('package_with_phase','Original_full phases − calibrated phases')]:
            c=gar['comparisons']['test'][view][name]['spell_f1']
            chunks.append(f'| {label} | {view} | {c["delta"]:+.9f} | [{c["ci95"][0]:+.9f}, {c["ci95"][1]:+.9f}] | {c["games_positive"]}/{c["games"]} |\n')
    chunks.append(f'\nAll **{verified["cells_verified"]} saved cells** pass official scalar replay; '
        f'maximum scalar difference is {verified["max_scalar_error"]:.3g}. '
        'Whole-game aggregation is independently checked. Reference/graded populations stay fixed '
        'across arms, phases and stress draws. Native predictions remain ignored; new native artifacts '
        'are locally verified, with no new remote-native archive claim.\n')
    chunks.append('\nThe [GAR report](GAR_DEPLOYMENT_EVAL_V2.md) owns protocol and artifact details; '
        '[verification](results/gar_deployment_eval_v2_verification.json) supplies all per-IoU means. '
        'The expanded scorecard includes paired intervals, per-game values and possession counts.\n')
    chunks.append('\n## Final promotion recommendation\n\n'+advisory['recommendation_text']+'\n\n'
        'The adopted manifest and policy remain unchanged. A promotion would require its own explicit '
        'working-baseline decision and a preserved exception to the failed historical clean screen. '
        'No new local tuning or organizer scoring follows from this readout.\n')
    path=ROOT/'experiments/ORIGINAL_FULL_PHASE_AVERAGE.md'
    text=path.read_text()
    text=text.replace('Original five-match phase-average D metrics, per-seed scores and per-IoU\n'
        'breakdowns were not published in the saved aggregate phase scorecards; they\n'
        'are unavailable here rather than inferred from other models or views.',
        'The completed guard audit below fills original clean/v1 D and component-seed metrics. '
        'Original-five v2 D/per-seed metrics were not published and remain unavailable; '
        'they are not inferred from other views.')
    if '## Completed original-provider guards' in text:raise ValueError('Readout already extended')
    path.write_text(text+''.join(chunks))
    print('Expanded saved-metric readout written')


if __name__=='__main__':
    build()
