"""Independent recomputation of the held-out planner results, written separately from scripts/evaluation-metrics.ts
(standard library only, its own bootstrap and random generator). Reads the per-case rows of each held-out report and
the stored labels in models/data/cases-v2, then checks the reported numbers. Runs no simulation or inference.

  python3 scripts/recompute_heldout.py        -> evidence/heldout-recompute.json
"""
import glob, json, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAWS, SEED, EPS = 2000, 20261018, 1e-6

labels = {}
for f in glob.glob(str(ROOT / 'models/data/cases-v2/*.json')):
    s = json.loads(Path(f).read_text())
    if s['split'] == 'heldout' and s['solver'] == 'li-1.0.0' and s['promptVersion'] == 'runofflab-planner-prompt-5':
        labels[(s['family'], s['variation'])] = (s['noInterventionM3'], s['objectiveM3'])


def captured(none, ref, value):
    if none - ref <= EPS: return None   # no room to improve: undefined, not 0
    return (none - (none if value is None else value)) / (none - ref)


def bootstrap(per_family, rng):
    """Case-weighted pooled mean over families drawn with replacement; returns the 2.5% and 97.5% percentiles."""
    fams = sorted(per_family); means = []
    for _ in range(DRAWS):
        pick = [per_family[rng.choice(fams)] for _ in fams]
        vals = [v for group in pick for v in group]
        means.append(sum(vals) / len(vals))
    means.sort()
    return [means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS) - 1]]


def summarize(rows_by_case, method, rng, comparator=None):
    per_family, diffs_m3 = {}, []
    for key, rows in rows_by_case.items():
        none, ref = labels[key]
        m = rows[method]
        value = m['objective'] if m['valid'] else None   # invalid answer = no gardens
        c = captured(none, ref, value)
        if c is None: continue
        if comparator:
            o = rows[comparator]
            c -= captured(none, ref, o['objective'] if o['valid'] else None)
            diffs_m3.append((none if value is None else value) - o['objective'])
        per_family.setdefault(key[0], []).append(c)
    vals = [v for g in per_family.values() for v in g]
    out = {'cases': len(vals), 'families': len(per_family), 'mean': sum(vals) / len(vals), 'ci95': bootstrap(per_family, rng)}
    if comparator: out['meanObjectiveDifferenceM3'] = statistics.fmean(diffs_m3)
    return out


report = {'method': 'independent Python recomputation; bootstrap seed differs from the report, so interval ends agree only approximately',
          'draws': DRAWS, 'seed': SEED, 'heldoutLabels': len(labels), 'runs': {}}
for path in sorted(glob.glob(str(ROOT / 'evidence/planner-eval/v2-heldout-heldout-*.json'))):
    if path.endswith('.summary.json'): continue
    d = json.loads(Path(path).read_text()); tag = d['tag']
    rows_by_case = {}
    for r in d['results']:
        rows_by_case.setdefault((r['family'], r['variation']), {})[r['method']] = r
    assert set(rows_by_case) == set(labels), f'{tag}: case set differs from the stored held-out labels'
    model = next(m for m in rows_by_case[next(iter(rows_by_case))] if m.startswith('llm:'))
    best = next(m for m in rows_by_case[next(iter(rows_by_case))] if m.startswith('best-of-2:'))
    problems = []
    for key, rows in rows_by_case.items():
        none, _ = labels[key]
        if abs(rows['none']['objective'] - none) > 1e-9: problems.append(f'{key}: none row differs from stored label')
        if any(rows[m]['status'] != 'completed' for m in rows): problems.append(f'{key}: not completed')
        mv = rows[model]['objective'] if rows[model]['valid'] else None
        expect = rows['heuristic']['objective'] if mv is None else min(mv, rows['heuristic']['objective'])
        if abs(rows[best]['objective'] - expect) > 1e-9: problems.append(f'{key}: best-of-2 {rows[best]["objective"]} != min(model, heuristic) {expect}')
    rng = random.Random(SEED)
    report['runs'][tag] = {
        'model': summarize(rows_by_case, model, rng), 'bestOfTwo': summarize(rows_by_case, best, rng),
        'heuristic': summarize(rows_by_case, 'heuristic', rng), 'greedy': summarize(rows_by_case, 'greedy', rng),
        'random': summarize(rows_by_case, 'random', rng),
        'modelMinusHeuristic': summarize(rows_by_case, model, rng, 'heuristic'),
        'bestOfTwoMinusHeuristic': summarize(rows_by_case, best, rng, 'heuristic'),
        'modelMinusGreedy': summarize(rows_by_case, model, rng, 'greedy'),
        'validity': sum(rows[model]['valid'] for rows in rows_by_case.values()) / len(rows_by_case),
        'problems': problems,
    }

out = ROOT / 'evidence/heldout-recompute.json'
out.write_text(json.dumps(report, indent=2) + '\n')
r3 = lambda x: round(x, 3)
for tag, r in report['runs'].items():
    print(tag, '| model', r3(r['model']['mean']), [r3(x) for x in r['model']['ci95']], '| best2', r3(r['bestOfTwo']['mean']),
          '| m-h', r3(r['modelMinusHeuristic']['mean']), [r3(x) for x in r['modelMinusHeuristic']['ci95']], 'm3', r3(r['modelMinusHeuristic']['meanObjectiveDifferenceM3']),
          '| b2-h', r3(r['bestOfTwoMinusHeuristic']['mean']), [r3(x) for x in r['bestOfTwoMinusHeuristic']['ci95']],
          '| heur', r3(r['heuristic']['mean']), 'greedy', r3(r['greedy']['mean']), 'random', r3(r['random']['mean']),
          '| valid', r3(r['validity']), 'cases', r['model']['cases'], 'problems', len(r['problems']))
