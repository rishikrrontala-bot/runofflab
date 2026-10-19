/**
 * Collects the frozen held-out runs into one table (evidence/heldout-results.json and .md). Reads only finished
 * reports; runs no inference and no simulations. Baselines are recomputed inside every run, so the script also checks
 * that each run reports identical none / random / heuristic / greedy numbers.
 *   node --import tsx scripts/collect_heldout.ts
 */
import {readFileSync, writeFileSync} from 'node:fs';

const DIR = 'evidence/planner-eval';
const runs = [
  {tag: 'heldout-adaption-qwen35-4b-e6', label: 'Adaption Qwen3.5-4B, 6 epochs (selected)', role: 'primary', trained: true},
  {tag: 'heldout-adaption-qwen35-4b-b8', label: 'Adaption Qwen3.5-4B, 3 epochs', role: 'secondary', trained: true},
  {tag: 'heldout-adaption-qwen35-9b-q8', label: 'Adaption Qwen3.5-9B, MLX 8-bit', role: 'secondary', trained: true},
  {tag: 'heldout-local-qwen35-08b', label: 'Local Qwen3.5-0.8B (MLX LoRA)', role: 'secondary', trained: true},
  {tag: 'heldout-deepseek-v32', label: 'Untuned DeepSeek-V3.2', role: 'comparator', trained: false},
  {tag: 'heldout-thinkoff-Qwen_Qwen3.5-397B-A17B', label: 'Untuned Qwen3.5-397B-A17B, thinking off', role: 'comparator', trained: false},
  {tag: 'heldout-thinkoff-Qwen_Qwen3.5-9B', label: 'Untuned Qwen3.5-9B, thinking off', role: 'comparator', trained: false},
  {tag: 'heldout-thinkoff-Qwen_Qwen3.5-4B', label: 'Untuned Qwen3.5-4B, thinking off', role: 'comparator', trained: false},
];
const BASELINES = ['none', 'random', 'heuristic', 'greedy'] as const;
const read = (p: string) => JSON.parse(readFileSync(p, 'utf8'));
const r3 = (x: number | null | undefined) => x === null || x === undefined ? null : Math.round(x * 1000) / 1000;
const ci = (c: [number, number] | null | undefined) => c ? [r3(c[0]), r3(c[1])] : null;

type Summary = Record<string, any>;
const marginal = (full: Summary, recomputed: Summary) => ({
  validity: r3(full.validity), firstPassValidity: r3(full.firstPassValidity), coverage: r3(recomputed.coverage), infraFailures: recomputed.infraFailures,
  casesWithRoomToImprove: recomputed.casesWithRoomToImprove, meanCapturedImprovement: r3(recomputed.meanCapturedImprovement),
  capturedImprovementCI95: ci(recomputed.capturedImprovementCI95), real: r3(recomputed.bySource?.real?.meanCapturedImprovement),
  synthetic: r3(recomputed.bySource?.synthetic?.meanCapturedImprovement), exactMatch: r3(full.exactMatch), worseThanNoGardens: recomputed.worseThanNoGardens,
  meanRelativeReduction: r3(recomputed.meanRelativeReduction), solverCallsPerCase: r3(full.meanLogicalSimulationQueries),
  meanEndToEndSeconds: full.meanEndToEndMs === undefined ? null : r3(full.meanEndToEndMs / 1000), repairRequests: full.repairRequests,
  truncatedAttempts: full.truncatedAttempts, inputTokens: full.inputTokens, outputTokens: full.outputTokens, costUsd: r3(full.costUsd),
  invalidReasons: full.invalidReasons,
});
const pair = (p: Record<string, any> | undefined) => p && {
  pairedCases: p.pairedCases, families: p.families, comparisonStatus: p.comparisonStatus, modelBetter: p.modelBetter, modelWorse: p.modelWorse,
  meanCapturedDifference: r3(p.meanCapturedDifference), capturedDifferenceCI95: ci(p.capturedDifferenceCI95),
  meanObjectiveDifferenceM3: r3(p.meanObjectiveDifferenceM3), meanRelativeReductionDifference: r3(p.meanRelativeReductionDifference),
};

let baselines: Record<string, ReturnType<typeof marginal>> | null = null, baselineSource = '';
const rows = runs.map(run => {
  const full = read(`${DIR}/v2-heldout-${run.tag}.json`), rec = read(`${DIR}/v2-heldout-${run.tag}.summary.json`);
  if (full.split !== 'heldout' || full.cases !== 420) throw Error(`${run.tag}: expected the 420-case held-out split`);
  const keys = Object.keys(rec.summary), modelKey = keys.find(k => k.startsWith('llm:')), bestKey = keys.find(k => k.startsWith('best-of-2:'));
  if (!modelKey || !bestKey) throw Error(`${run.tag}: missing model rows`);
  for (const k of [modelKey, ...BASELINES])   // the recomputed summary (current helpers) must agree with the report
    if (Math.abs(full.summary[k].meanCapturedImprovement - rec.summary[k].meanCapturedImprovement) > 1e-9) throw Error(`${run.tag}: ${k} disagrees with its summary`);
  const b = Object.fromEntries(BASELINES.map(k => [k, marginal(full.summary[k], rec.summary[k])]));
  if (!baselines) { baselines = b; baselineSource = run.tag; }
  else for (const k of BASELINES)
    if (b[k].meanCapturedImprovement !== baselines[k].meanCapturedImprovement || b[k].validity !== baselines[k].validity)
      throw Error(`${run.tag}: baseline ${k} differs from ${baselineSource}`);
  return {...run, report: `${DIR}/v2-heldout-${run.tag}.json`, generation: full.generation, spentUsd: r3(full.spentUsd), infraRetried: full.infraRetried ?? 0,
    model: marginal(full.summary[modelKey], rec.summary[modelKey]), bestOfTwo: marginal(full.summary[bestKey], rec.summary[bestKey]),
    paired: {model: {heuristic: pair(rec.paired.model?.heuristic ?? rec.paired[modelKey]?.heuristic), greedy: pair(rec.paired.model?.greedy ?? rec.paired[modelKey]?.greedy)},
      bestOfTwo: {heuristic: pair(rec.paired.bestOfTwo?.heuristic ?? rec.paired[bestKey]?.heuristic), greedy: pair(rec.paired.bestOfTwo?.greedy ?? rec.paired[bestKey]?.greedy)}}};
});

const out = {generatedAt: new Date().toISOString(), split: 'heldout', cases: 420, baselineSource, baselines, runs: rows,
  notes: ['Captured improvement = (none - planner) / (none - search label) over the cases where the label improves on no gardens; invalid answers count as 0.',
    'Intervals: 2,000-draw family-cluster bootstrap (7 held-out families); paired comparisons use the same draws.',
    '"Better" or "worse" is claimed only when the paired interval excludes 0 with complete coverage (comparisonStatus "complete").',
    'Objective difference = planner minus comparator peak ponded volume (m3); negative means the planner leaves less water.',
    'Best-of-2 runs the simulator on the model design and on the D8 heuristic design and keeps the better one (2 solver calls).',
    'Greedy solver calls are logical queries answered from the labelling cache; its wall time here is not a live timing (see site evaluation).']};
writeFileSync('evidence/heldout-results.json', JSON.stringify(out, null, 2) + '\n');

const f = (x: number | null) => x === null ? '—' : x.toFixed(3), sgn = (x: number | null) => x === null ? '—' : (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3);
const iv = (c: (number | null)[] | null) => c ? `${c[0]!.toFixed(3)}–${c[1]!.toFixed(3)}` : '—';
const verdict = (p: ReturnType<typeof pair>) => !p ? '—' : `${sgn(p.meanCapturedDifference)} (${p.capturedDifferenceCI95 ? `${sgn(p.capturedDifferenceCI95[0])} to ${sgn(p.capturedDifferenceCI95[1])}` : '—'}), ${p.comparisonStatus !== 'complete' ? 'incomplete' : p.modelBetter ? 'better' : p.modelWorse ? 'worse' : 'not significant'}`;
const lines = ['# Held-out results (420 cases, 7 families, run once)', '', `Generated by \`scripts/collect_heldout.ts\` from \`${DIR}/v2-heldout-*.json\`. Baselines are identical in every run (checked).`, '',
  '| Planner | Valid (first pass) | Captured (95% CI) | Real / synthetic | vs D8 heuristic, paired | Solver calls | Time per case | Cost |', '|---|---|---|---|---|---|---|---|'];
for (const k of BASELINES) { const b = baselines![k];
  lines.push(`| ${k === 'heuristic' ? 'D8 heuristic (no model)' : k === 'greedy' ? 'Greedy simulator search' : k === 'none' ? 'No gardens' : 'Random (seeded)'} | — | ${f(b.meanCapturedImprovement)} (${iv(b.capturedImprovementCI95)}) | ${f(b.real)} / ${f(b.synthetic)} | — | ${f(b.solverCallsPerCase)} | — | $0 |`); }
for (const r of rows) for (const [name, m, p] of [[r.label, r.model, r.paired.model.heuristic], [`${r.label} + heuristic, best of 2`, r.bestOfTwo, r.paired.bestOfTwo.heuristic]] as const)
  lines.push(`| ${r.role === 'primary' ? `**${name}**` : name} | ${(100 * m.validity!).toFixed(1)}% (${(100 * m.firstPassValidity!).toFixed(1)}%) | ${f(m.meanCapturedImprovement)} (${iv(m.capturedImprovementCI95)}) | ${f(m.real)} / ${f(m.synthetic)} | ${verdict(p)} | ${f(m.solverCallsPerCase)} | ${m.meanEndToEndSeconds === null ? '—' : `${m.meanEndToEndSeconds.toFixed(1)} s`} | ${name.includes('best of 2') ? '—' : `$${(r.spentUsd ?? 0).toFixed(2)}`} |`);
lines.push('', '## Paired against greedy and on raw volume', '', '| Planner | vs greedy, captured (paired) | vs heuristic, mean m³ difference | vs heuristic, relative-reduction difference |', '|---|---|---|---|');
for (const r of rows) for (const [name, pr] of [[r.label, r.paired.model], [`${r.label}, best of 2`, r.paired.bestOfTwo]] as const)
  lines.push(`| ${name} | ${verdict(pr.greedy)} | ${sgn(pr.heuristic?.meanObjectiveDifferenceM3 ?? null)} | ${sgn(pr.heuristic?.meanRelativeReductionDifference ?? null)} |`);
lines.push('', ...out.notes.map(n => `- ${n}`), '');
writeFileSync('evidence/heldout-results.md', lines.join('\n'));
console.log(lines.join('\n'));
