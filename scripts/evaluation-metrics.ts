/**
 * Shared, testable pieces of the planner evaluators (scripts/evaluate_planners.ts, scripts/evaluate_site.ts):
 * scoring, coverage, family-cluster bootstrap (marginal and paired), timing, the real-site scenario fingerprint
 * and local model-server access. Nothing here runs the solver or reads evaluation data.
 *
 * Scoring convention (EVAL-CX-001): a method that returns no valid design is scored as no gardens, so it captures
 * 0 of the reference improvement when the reference improves on no gardens, and null only when there is no room to
 * improve. Infrastructure failures (HTTP error, timeout, missing output) are not model answers: they are reported
 * as incomplete coverage and left out of validity and capture.
 */
import {createHash} from 'node:crypto';
import {seededRandom} from '../src/core/planner';
import type {Config, Grid, Patch} from '../src/core/types';

/** A reference must improve on no gardens by more than this (m3) for captured improvement to be defined. */
export const ROOM_EPS_M3 = 1e-6;
export const BOOTSTRAP_DRAWS = 2000;
export const BOOTSTRAP_SEED = 20261017;

// ---- Scoring -----------------------------------------------------------------------------------

/** The objective a method is scored with: its design's objective, or no gardens when it gave no valid design. */
export const effectiveObjective = (noneM3: number, objective: number | null | undefined) => objective ?? noneM3;

/** (none - method) / (none - reference). Null when the reference has no room to improve; can be < 0 or > 1. */
export function capturedImprovement(noneM3: number, referenceM3: number, objective: number | null | undefined): number | null {
  const room = noneM3 - referenceM3;
  return room > ROOM_EPS_M3 ? (noneM3 - effectiveObjective(noneM3, objective)) / room : null;
}

/** 1 - method / none (0 when no gardens leaves no water in the target). */
export const relativeReduction = (noneM3: number, objective: number | null | undefined) =>
  noneM3 > ROOM_EPS_M3 ? 1 - effectiveObjective(noneM3, objective) / noneM3 : 0;

/** Mean, or null for an empty group (an empty group has no mean; it is not 0). */
export const meanOrNull = (values: number[]) => values.length ? values.reduce((s, x) => s + x, 0) / values.length : null;
export const ratioOrNull = (count: number, total: number) => total ? count / total : null;

// ---- Family-cluster bootstrap -------------------------------------------------------------------

/** Bootstrap draws of whole families (indices into a family list), made once so every method and every
 *  comparison is resampled with the same families (common random numbers). */
export function familyDraws(familyCount: number, draws = BOOTSTRAP_DRAWS, seed = BOOTSTRAP_SEED): number[][] {
  const rng = seededRandom(seed);
  return Array.from({length: draws}, () => Array.from({length: familyCount}, () => Math.floor(rng() * familyCount)));
}

/** Pooled case mean for each draw: every value of every sampled family (with repeats). Draws with no values are skipped. */
export function clusterBootstrapMeans(valuesByFamily: number[][], draws: number[][]): number[] {
  const means: number[] = [];
  for (const draw of draws) {
    let sum = 0, n = 0;
    for (const f of draw) for (const v of valuesByFamily[f]) { sum += v; n++; }
    if (n) means.push(sum / n);
  }
  return means;
}

/** 2.5th and 97.5th percentiles (same index rule as the original evaluator), or null without draws. */
export function percentileCI(values: number[]): [number, number] | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return [sorted[Math.floor(sorted.length * .025)], sorted[Math.floor(sorted.length * .975)]];
}

// ---- Per-method and paired summaries ------------------------------------------------------------

export type RowStatus = 'completed' | 'infra-failure';
/** One case: its no-gardens objective and the bounded-search reference objective. */
export type CaseOutcome = {family: string; variation: number; source: string; noneM3: number; referenceM3: number};
/** One method's answer on one case. objective is null when the answer was invalid (or the request failed). */
export type MethodRow = {family: string; variation: number; source: string; status: RowStatus; valid: boolean | null; firstPassValid: boolean | null; objective: number | null};

export const caseKey = (r: {family: string; variation: number}) => `${r.family}#${r.variation}`;

/** EVAL-CX-006: rows must cover the expected cohort exactly once, with matching source metadata. Missing cases are
 *  materialized as explicit infrastructure failures so coverage is always computed against the expected cases. */
export function alignRows(cases: CaseOutcome[], rows: MethodRow[]): MethodRow[] {
  const expected = new Map(cases.map(c => [caseKey(c), c]));
  if (expected.size !== cases.length) throw Error('Duplicate case keys in the expected cohort');
  const seen = new Map<string, MethodRow>();
  for (const r of rows) {
    const k = caseKey(r), c = expected.get(k);
    if (!c) throw Error(`Row for unknown case ${k}`);
    if (seen.has(k)) throw Error(`Duplicate row for case ${k}`);
    if (r.source !== c.source) throw Error(`Source mismatch for case ${k}: ${r.source} vs ${c.source}`);
    seen.set(k, r);
  }
  return cases.map(c => seen.get(caseKey(c)) ?? {family: c.family, variation: c.variation, source: c.source, status: 'infra-failure', valid: null, firstPassValid: null, objective: null});
}

function groupByFamily(families: string[], items: {family: string; value: number}[]): number[][] {
  const index = new Map(families.map((f, i) => [f, i])), out = families.map(() => [] as number[]);
  for (const item of items) out[index.get(item.family)!].push(item.value);
  return out;
}

/**
 * Marginal summary of one method. Estimand: the case-weighted pooled mean of captured improvement over completed
 * cases whose reference improves on no gardens; families are resampled whole for the interval.
 */
export function marginalSummary(cases: CaseOutcome[], supplied: MethodRow[], families: string[], draws: number[][], sources = ['real', 'synthetic']) {
  const rows = alignRows(cases, supplied), byKey = new Map(cases.map(c => [caseKey(c), c]));
  const completed = rows.filter(r => r.status === 'completed').map(r => ({r, c: byKey.get(caseKey(r))!}));
  const qualifying: {family: string; source: string; captured: number}[] = [];
  for (const {r, c} of completed) {
    const captured = capturedImprovement(c.noneM3, c.referenceM3, r.objective);
    if (captured !== null) qualifying.push({family: r.family, source: r.source, captured});
  }
  const valuesByFamily = groupByFamily(families, qualifying.map(x => ({family: x.family, value: x.captured})));
  return {
    cases: rows.length, completed: completed.length, infraFailures: rows.length - completed.length, coverage: ratioOrNull(completed.length, rows.length),
    validity: ratioOrNull(completed.filter(x => x.r.valid).length, completed.length),
    firstPassValidity: ratioOrNull(completed.filter(x => x.r.firstPassValid).length, completed.length),
    casesWithRoomToImprove: qualifying.length, familiesWithRoomToImprove: valuesByFamily.filter(v => v.length).length,
    meanCapturedImprovement: meanOrNull(qualifying.map(x => x.captured)),
    capturedImprovementCI95: percentileCI(clusterBootstrapMeans(valuesByFamily, draws)),
    bySource: Object.fromEntries([...new Set([...sources, ...rows.map(r => r.source)])].map(source => {
      const values = qualifying.filter(x => x.source === source).map(x => x.captured);
      return [source, {casesWithRoomToImprove: values.length, meanCapturedImprovement: meanOrNull(values)}];
    })),
    meanRelativeReduction: meanOrNull(completed.map(x => relativeReduction(x.c.noneM3, x.r.objective))),
    worseThanNoGardens: completed.filter(x => x.r.objective !== null && x.r.objective > x.c.noneM3 + 1e-8).length,
  };
}

/**
 * Paired comparison, model minus comparator, on the same cases (EVAL-CX-005). A pair needs both methods completed
 * on the case; captured differences use cases whose reference improves on no gardens, with invalid answers scored
 * as no gardens. For each bootstrap draw the sampled families' paired differences are pooled (case-weighted).
 */
export function pairedComparison(cases: CaseOutcome[], modelRows: MethodRow[], comparatorRows: MethodRow[], families: string[], draws: number[][]) {
  const model = new Map(alignRows(cases, modelRows).map(r => [caseKey(r), r])), comparator = new Map(alignRows(cases, comparatorRows).map(r => [caseKey(r), r]));
  const pairs: {family: string; captured: number; modelCaptured: number; comparatorCaptured: number; objectiveM3: number; relativeReduction: number}[] = [];
  let incompletePairs = 0, pairsWithoutRoom = 0;
  for (const c of cases) {
    const m = model.get(caseKey(c)), k = comparator.get(caseKey(c));
    if (m?.status !== 'completed' || k?.status !== 'completed') { incompletePairs++; continue; }
    const mc = capturedImprovement(c.noneM3, c.referenceM3, m.objective), kc = capturedImprovement(c.noneM3, c.referenceM3, k.objective);
    if (mc === null || kc === null) { pairsWithoutRoom++; continue; }
    pairs.push({family: c.family, captured: mc - kc, modelCaptured: mc, comparatorCaptured: kc,
      objectiveM3: effectiveObjective(c.noneM3, m.objective) - effectiveObjective(c.noneM3, k.objective),
      relativeReduction: relativeReduction(c.noneM3, m.objective) - relativeReduction(c.noneM3, k.objective)});
  }
  const valuesByFamily = groupByFamily(families, pairs.map(p => ({family: p.family, value: p.captured})));
  const ci = percentileCI(clusterBootstrapMeans(valuesByFamily, draws));
  return {
    pairedCases: pairs.length, families: valuesByFamily.filter(v => v.length).length, pairsWithoutRoom, incompletePairs,
    // EVAL-CX-007: a primary directional claim needs every expected pair completed; partial runs stay descriptive.
    comparisonStatus: incompletePairs ? 'incomplete' as const : 'complete' as const,
    modelBetter: !incompletePairs && ci ? ci[0] > 0 : null, modelWorse: !incompletePairs && ci ? ci[1] < 0 : null,
    meanCapturedDifference: meanOrNull(pairs.map(p => p.captured)), capturedDifferenceCI95: ci,
    modelMeanCaptured: meanOrNull(pairs.map(p => p.modelCaptured)), comparatorMeanCaptured: meanOrNull(pairs.map(p => p.comparatorCaptured)),
    meanObjectiveDifferenceM3: meanOrNull(pairs.map(p => p.objectiveM3)), meanRelativeReductionDifference: meanOrNull(pairs.map(p => p.relativeReduction)),
  };
}

// ---- Timing (EVAL-CX-003) -----------------------------------------------------------------------

/** Times selection and the scoring of the selected design separately. Each clock is read after the work it
 *  closes has finished, so scoring is never left out of the end-to-end time. */
export async function timedProposal<P, S>(select: () => P | Promise<P>, score: (proposal: P) => S) {
  const start = performance.now();
  const proposal = await select();
  const selected = performance.now();
  const scored = score(proposal);
  const end = performance.now();
  return {proposal, scored, selectionMs: selected - start, scoringMs: end - selected, endToEndMs: end - start};
}

// ---- Real-site scenario fingerprint (EVAL-CX-004) -----------------------------------------------

/** JSON with object keys sorted at every level (arrays keep their order). */
export const canonicalJson = (value: unknown) => JSON.stringify(value, (_key, v) =>
  v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.keys(v).sort().map(k => [k, v[k]])) : v);

/** SHA-256 over everything a cached scenario depends on: terrain and masks, the full configuration, the ordered
 *  candidates (all fields), the prompt, solver and prompt versions and the reference-search policy. */
export function siteScenarioFingerprint(input: {grid: Grid; config: Config; candidates: Patch[]; prompt: string; solverVersion: string; promptVersion: string; searchPolicy: string}): string {
  const {grid: g} = input;
  return createHash('sha256').update(canonicalJson({
    terrain: {nx: g.nx, ny: g.ny, dx: g.dx, dy: g.dy, z: g.z, mask: g.mask, sink: g.sink ?? null, eligible: g.eligible ?? null},
    config: input.config, candidates: input.candidates, prompt: input.prompt,
    solverVersion: input.solverVersion, promptVersion: input.promptVersion, searchPolicy: input.searchPolicy,
  })).digest('hex');
}

// ---- Model access shared by both evaluators -----------------------------------------------------

export type InfraKind = 'http' | 'timeout' | 'network' | 'missing-output' | 'spend-guard' | 'checkpoint-mismatch';
/** A failure of the serving infrastructure, not an answer from the model. */
export class InfraError extends Error {
  kind: InfraKind;
  constructor(kind: InfraKind, message: string) { super(message); this.kind = kind; }
}
export function asInfraError(e: unknown): InfraError {
  if (e instanceof InfraError) return e;
  const name = e instanceof Error ? e.name : '', message = e instanceof Error ? e.message : String(e);
  return new InfraError(name === 'TimeoutError' || name === 'AbortError' ? 'timeout' : 'network', message.slice(0, 300));
}

/** Empty output that used the whole budget, or a provider-reported length stop. */
export const isTruncated = (a: {raw: string; outputTokens: number; finishReason?: string}, maxTokens: number | null | undefined) =>
  a.finishReason === 'length' || (!a.raw.trim() && typeof maxTokens === 'number' && a.outputTokens >= maxTokens);

/** Identity and effective generation settings reported by scripts/model_server.py. */
export type LocalStatus = {model: string; method: string; checkpoint: string; trainingRows: number; generation?: Record<string, unknown>} & Record<string, unknown>;
export async function localStatus(url: string): Promise<LocalStatus> {
  let response: Response;
  try { response = await fetch(`${url}/status`, {signal: AbortSignal.timeout(10_000)}); }
  catch (e) { throw Error(`Local model server unreachable at ${url}/status: ${asInfraError(e).message}`); }
  if (!response.ok) throw Error(`Local model server ${url}/status returned ${response.status}`);
  return await response.json() as LocalStatus;
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/** One local inference. HTTP 429 (server busy) is retried with bounded exponential backoff; latencyMs covers the
 *  answered request only. Every other failure throws an InfraError. */
export async function localInfer(url: string, prompt: string, options: {expectedCheckpoint?: string; timeoutMs?: number; busyRetries?: number} = {}) {
  const {timeoutMs = 120_000, busyRetries = 6} = options;
  for (let retry = 0; ; retry++) {
    const start = performance.now();
    let response: Response;
    try {
      response = await fetch(`${url}/infer`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt}), signal: AbortSignal.timeout(timeoutMs)});
    } catch (e) { throw asInfraError(e); }
    if (response.status === 429 && retry < busyRetries) { await sleep(Math.min(8000, 250 * 2 ** retry)); continue; }
    const body: any = await response.json().catch(() => null);
    const latencyMs = performance.now() - start;
    if (!response.ok) throw new InfraError('http', `Local model ${response.status}: ${String(body?.error ?? 'request failed').slice(0, 200)}`);
    if (typeof body?.output !== 'string') throw new InfraError('missing-output', 'Local model response had no output text');
    if (options.expectedCheckpoint !== undefined && body.checkpoint !== options.expectedCheckpoint)
      throw new InfraError('checkpoint-mismatch', `Server now reports checkpoint ${body.checkpoint}, expected ${options.expectedCheckpoint}`);
    return {output: body.output as string, outputTokens: Number(body.outputTokens ?? 0), checkpoint: body.checkpoint as string | undefined, latencyMs, busyRetries: retry,
      finishReason: typeof body.finishReason === 'string' ? body.finishReason : undefined};
  }
}
