/**
 * Recomputes the summary and paired comparisons of an evaluate_planners.ts report from its per-case rows with the
 * current helpers (scripts/evaluation-metrics.ts), without rerunning any model. Case outcomes come from the stored
 * labelling records. Writes <report>.summary.json next to the report.
 *   node --import tsx scripts/summarize_eval.ts evidence/planner-eval/v2-validation-<tag>.json
 */
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {familyDraws, marginalSummary, pairedComparison, type CaseOutcome, type MethodRow} from './evaluation-metrics';

const path = process.argv[2];
if (!path) throw Error('Usage: summarize_eval.ts <report.json>');
const text = readFileSync(path, 'utf8'), report = JSON.parse(text);
const keys = [...new Set(report.results.map((r: {family: string; variation: number}) => `${r.family}#${r.variation}`))] as string[];
// EVAL-CX-011: the cohort is the one the report declares, not whatever rows happen to be present. A case missing from
// every method, or a row from another split or version, stops the recomputation instead of shrinking the cohort.
if (typeof report.cases === 'number' && keys.length !== report.cases)
  throw Error(`${path} declares ${report.cases} cases but its rows cover ${keys.length}; refusing to recompute a different cohort`);
const declaredFamilies = Array.isArray(report.families) ? new Set(report.families as string[]) : null;
const outcomes: CaseOutcome[] = keys.map(k => {
  const [family, variation] = [k.slice(0, k.lastIndexOf('#')), Number(k.slice(k.lastIndexOf('#') + 1))];
  const stored = JSON.parse(readFileSync(`models/data/cases-v2/${family}-${variation}.json`, 'utf8'));
  for (const [field, want] of [['split', report.split], ['solver', report.solver], ['promptVersion', report.promptVersion]] as const)
    if (want !== undefined && stored[field] !== want) throw Error(`${k}: stored ${field} ${stored[field]} differs from the report's ${want}`);
  if (declaredFamilies && !declaredFamilies.has(family)) throw Error(`${k}: family not declared by the report`);
  return {family, variation, source: stored.source, noneM3: stored.noInterventionM3, referenceM3: stored.objectiveM3};
});
const families = [...new Set(outcomes.map(c => c.family))], draws = familyDraws(families.length);
const rowsOf = (method: string): MethodRow[] => report.results.filter((r: {method: string}) => r.method === method).map((r: any) => ({
  family: r.family, variation: r.variation, source: r.source, status: r.status ?? 'completed', valid: r.valid ?? null,
  firstPassValid: r.firstPassValid ?? r.valid ?? null, objective: r.objective ?? null}));
const methods = [...new Set(report.results.map((r: {method: string}) => r.method))] as string[];
const model = methods.find(m => m.startsWith('llm:'));
// Older reports predate the best-of-2 method; rebuild it from the model and heuristic rows (same rule as the evaluator).
if (model && !methods.includes(`best-of-2:${model.slice(4)}`) && methods.includes('heuristic')) {
  const name = `best-of-2:${model.slice(4)}`;
  for (const m of report.results.filter((r: {method: string}) => r.method === model)) {
    const h = report.results.find((r: any) => r.method === 'heuristic' && r.family === m.family && r.variation === m.variation);
    const useModel = (m.status ?? 'completed') === 'completed' && m.objective !== null && m.objective <= h.objective;
    // EVAL-CX-012: first-pass validity always describes the model's first answer, as in the online evaluator.
    report.results.push({...(useModel ? m : h), method: name, status: m.status ?? 'completed', valid: (m.status ?? 'completed') === 'completed' ? true : null,
      firstPassValid: m.firstPassValid ?? m.valid ?? null});
  }
  methods.push(name);
}
const out = {source: path, sourceSha256: createHash('sha256').update(text).digest('hex'), recomputedAt: new Date().toISOString(), cases: outcomes.length, families,
  declared: {split: report.split, solver: report.solver, promptVersion: report.promptVersion, cases: report.cases},
  summary: Object.fromEntries(methods.map(m => [m, marginalSummary(outcomes, rowsOf(m), families, draws)])),
  paired: model ? Object.fromEntries([model, `best-of-2:${model.slice(4)}`].filter(m => methods.includes(m)).map(m => [m,
    Object.fromEntries(['heuristic', 'greedy', 'random', 'none'].filter(k => methods.includes(k)).map(k => [k, pairedComparison(outcomes, rowsOf(m), rowsOf(k), families, draws)]))])) : null};
writeFileSync(path.replace(/\.json$/, '.summary.json'), JSON.stringify(out, null, 2) + '\n');
console.log(JSON.stringify({summary: Object.fromEntries(Object.entries(out.summary).map(([k, v]: [string, any]) => [k, {coverage: v.coverage, validity: v.validity, captured: v.meanCapturedImprovement, ci: v.capturedImprovementCI95}])), paired: out.paired}, null, 1));
