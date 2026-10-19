/**
 * Paired planner evaluation on simulator-labelled cases (dataset v2).
 *
 * Each case is rebuilt exactly with buildCase(family, variation); its stored prompt must match plannerPrompt()
 * byte for byte. Proposals are scored by the solver objective of the proposed design: a stored outcome when the
 * labelling search already evaluated it, otherwise a fresh simulation with the same solver and configuration.
 * Model proposals pass through acceptModelText(), with at most one repair request. Invalid proposals stay invalid.
 *
 *   node --import tsx scripts/evaluate_planners.ts --split heldout --tag baselines
 *   node --import tsx scripts/evaluate_planners.ts --split heldout --llm featherless:deepseek-ai/DeepSeek-V3.2 --max-usd 1
 *   node --import tsx scripts/evaluate_planners.ts --split heldout --llm local:http://127.0.0.1:8787 --tag trained
 * Options: --limit N (evenly spaced sample)  --concurrency N  --max-tokens N  --methods none,random,heuristic,greedy,oracle
 *          --thinking off  (asks the server to skip hidden reasoning: chat_template_kwargs.enable_thinking=false; EVAL-001)
 */
import {readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync, appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {acceptModelText, designKey, greedy, heuristic, plannerPrompt, randomDesign, repairInstruction, seededRandom, PROMPT_VERSION} from '../src/core/planner';
import {runSync} from '../src/core/solver';
import {buildCase, CASE_VERSION, type TrainingCase} from '../src/core/trainingCases';
import {SOLVER_VERSION, patchArea, type Patch, type Run} from '../src/core/types';
import {loadFamilies} from './families';
import {InfraError, asInfraError, caseKey, familyDraws, isTruncated, localInfer, localStatus, marginalSummary, pairedComparison, type CaseOutcome, type MethodRow} from './evaluation-metrics';

type Stored = {family: string; variation: number; split: string; source: string; prompt: string; completion: string; promptVersion: string; solver: string;
  objectiveM3: number; noInterventionM3: number; outcomes: {ids: string[]; objectiveM3: number; treatedM2: number}[]};
type Case = TrainingCase & {stored: Stored; objectives: Map<string, number>; simulated: number};
type Attempt = {raw: string; latencyMs: number; inputTokens: number; outputTokens: number; costUsd: number; requestId?: string; cached: boolean; reasoningChars?: number; finishReason?: string};
/** status 'infra-failure': the serving infrastructure failed (not a model answer); excluded from validity and capture, reported as coverage. */
type Result = {family: string; variation: number; source: string; method: string; status: 'completed' | 'infra-failure'; ids: string[] | null; valid: boolean | null;
  firstPassValid: boolean | null; objective: number | null; lenientIds?: string[] | null; lenientObjective?: number | null; error?: string;
  logicalSimulationQueries: number; executedSimulations: number; selectionMs: number; scoringMs: number; endToEndMs: number; treatedM2: number | null;
  truncatedAttempts?: number; attempts?: Attempt[];
  /** EVAL-CX-010: the failed try that preceded an end-of-run infrastructure retry, kept so its answers, usage and cost are not lost. */
  infraRetryOf?: {error?: string; attempts?: Attempt[]; firstPassValid: boolean | null}};

const arg = (name: string, fallback?: string) => {
  const inline = process.argv.find(a => a.startsWith(`--${name}=`));
  if (inline) return inline.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
};
const split = arg('split', 'heldout')!, casesDir = arg('cases-dir', 'models/data/cases-v2')!, limit = Number(arg('limit', '0'));
const methods = arg('methods', 'none,random,heuristic,greedy,oracle')!.split(',').filter(Boolean);
const llm = arg('llm'), maxUsd = Number(arg('max-usd', '1')), maxTokens = Number(arg('max-tokens', '256'));
const local = llm?.startsWith('local:') ?? false, localUrl = local ? llm!.slice('local:'.length) : '';
// EVAL-CX-002: the local server serves one request at a time, so local runs are sequential.
const concurrency = Number(arg('concurrency', local ? '1' : '4'));
if (local && concurrency !== 1) throw Error('Local model evaluation must use --concurrency 1 (the server accepts one request at a time)');
const thinking = arg('thinking', 'default')!;
const tag = arg('tag', llm ? llm.replace(/[^A-Za-z0-9.-]+/g, '_') : 'baselines')!;

function loadCases(): Case[] {
  const families = new Map(loadFamilies().map(f => [f.id, f]));
  const stored = readdirSync(casesDir).filter(f => f.endsWith('.json')).map(f => JSON.parse(readFileSync(`${casesDir}/${f}`, 'utf8')) as Stored)
    .filter(s => s.split === split && s.solver === SOLVER_VERSION && ((s as {caseVersion?: string}).caseVersion ?? 'runofflab-cases-1') === CASE_VERSION && s.promptVersion === PROMPT_VERSION)
    .sort((a, b) => a.family.localeCompare(b.family) || a.variation - b.variation);
  const picked = limit > 0 && limit < stored.length ? stored.filter((_, i) => i % Math.ceil(stored.length / limit) === 0).slice(0, limit) : stored;
  return picked.map(s => {
    const c = buildCase(families.get(s.family)!, s.variation);
    if (plannerPrompt(c.grid, c.config, c.candidates) !== s.prompt) throw Error(`Rebuilt prompt differs for ${s.family}-${s.variation}`);
    return {...c, stored: s, objectives: new Map(s.outcomes.map(o => [designKey(o.ids), o.objectiveM3])), simulated: 0};
  });
}

/** Exact objective of a feasible design: stored outcome, or a new simulation with the same solver and config. */
function objectiveOf(c: Case, patches: Patch[]): number {
  const k = designKey(patches);
  let v = c.objectives.get(k);
  if (v === undefined) { v = runSync(c.grid, c.config, patches, {snapshots: false}).objective; c.objectives.set(k, v); c.simulated++; }
  return v;
}
const byIds = (c: Case, ids: string[]) => ids.map(id => c.candidates.find(p => p.id === id)!);
const scored = (c: Case, ids: string[] | null) => ids ? {objective: objectiveOf(c, byIds(c, ids)), treatedM2: patchArea(c.grid, byIds(c, ids))} : {objective: null, treatedM2: null};

async function baseline(c: Case, method: string): Promise<Result> {
  const start = performance.now(), executedBefore = c.simulated;
  let ids: string[], calls = 1;
  if (method === 'none') ids = [];
  else if (method === 'random') ids = randomDesign(c.grid, c.config, c.candidates, c.seed).map(p => p.id);
  else if (method === 'heuristic') ids = heuristic(c.grid, c.config, c.candidates).map(p => p.id);
  else if (method === 'greedy') {
    const plan = await greedy(c.grid, c.config, c.candidates, async p => ({objective: objectiveOf(c, p), patches: p}) as unknown as Run);
    ids = plan.proposal.selected_patch_ids; calls = plan.simulationCalls;
  } else if (method === 'oracle') { ids = JSON.parse(c.stored.completion).selected_patch_ids; calls = c.stored.outcomes.length; }
  else throw Error(`Unknown method ${method}`);
  const selected = performance.now(), result = scored(c, ids), end = performance.now();   // EVAL-CX-003: clocks read after the work
  return {family: c.family, variation: c.variation, source: c.source, method, status: 'completed', ids, valid: true, firstPassValid: true,
    logicalSimulationQueries: calls, executedSimulations: c.simulated - executedBefore, selectionMs: selected - start, scoringMs: end - selected, endToEndMs: end - start, ...result};
}

// ---- Model access ----------------------------------------------------------------------------
function env(name: string): string | undefined {
  if (process.env[name]) return process.env[name];
  if (!existsSync('.env.local')) return;
  for (const line of readFileSync('.env.local', 'utf8').split('\n')) {
    const [k, ...v] = line.split('=');
    if (k?.trim() === name) return v.join('=').trim().replace(/^["']|["']$/g, '') || undefined;
  }
}
const catalog = existsSync('models/featherless-catalog.json') ? (JSON.parse(readFileSync('models/featherless-catalog.json', 'utf8')).data as any[]) : [];
let spentUsd = 0;
const requestIds: string[] = [];

async function complete(prompt: string): Promise<Attempt> {
  const [provider, ...rest] = llm!.split(':'), target = rest.join(':');
  if (provider === 'local') {
    const r = await localInfer(localUrl, prompt, {expectedCheckpoint: effective?.checkpoint as string | undefined});
    return {raw: r.output, latencyMs: r.latencyMs, inputTokens: 0, outputTokens: r.outputTokens, costUsd: 0, requestId: r.checkpoint, cached: false, finishReason: r.finishReason};
  }
  if (provider !== 'featherless') throw Error(`Unknown provider ${provider}`);
  const cacheDir = `evidence/llm-cache/${provider}/${target.replace(/[^A-Za-z0-9.-]+/g, '_')}`;
  const cachePath = `${cacheDir}/${createHash('sha256').update(`${maxTokens}\n${thinking}\n${prompt}`).digest('hex')}.json`;
  if (existsSync(cachePath)) return {...JSON.parse(readFileSync(cachePath, 'utf8')), cached: true};
  mkdirSync(cacheDir, {recursive: true});
  const price = catalog.find(m => m.id === target)?.pricing;
  if (!price) throw Error(`No Featherless catalog price for ${target}; refusing an unpriced request`);
  const estimate = prompt.length / 3 * price.input / 1e6 + maxTokens * price.output / 1e6;
  if (spentUsd + estimate > maxUsd) throw new InfraError('spend-guard', `Spend guard: $${spentUsd.toFixed(4)} spent; next request could exceed --max-usd ${maxUsd}`);
  const token = env('FEATHERLESS_API_KEY');
  if (!token) throw Error('FEATHERLESS_API_KEY is not configured');
  const start = performance.now();
  let response: Response | undefined;
  try {
    for (let retry = 0; retry < 4; retry++) {
      response = await fetch('https://api.featherless.ai/v1/chat/completions', {method: 'POST', signal: AbortSignal.timeout(180_000),
        headers: {Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'},
        body: JSON.stringify({model: target, messages: [{role: 'user', content: prompt}], temperature: 0, max_tokens: maxTokens,
          ...(thinking === 'off' ? {chat_template_kwargs: {enable_thinking: false}} : {})})});
      if (response.status !== 429 && response.status < 500) break;
      await new Promise(r => setTimeout(r, 2000 * 2 ** retry));
    }
  } catch (e) { throw asInfraError(e); }
  const body: any = await response!.json().catch(() => ({}));
  if (!response!.ok) throw new InfraError('http', `Featherless ${response!.status}: ${String(body?.error?.message ?? body?.error ?? 'request failed').slice(0, 200)}`);
  const inputTokens = body.usage?.prompt_tokens ?? 0, outputTokens = body.usage?.completion_tokens ?? 0;
  const costUsd = inputTokens * price.input / 1e6 + outputTokens * price.output / 1e6;
  spentUsd += costUsd; if (body.id) requestIds.push(body.id);
  const message = body.choices?.[0]?.message ?? {};
  const attempt = {raw: message.content ?? '', latencyMs: performance.now() - start, inputTokens, outputTokens, costUsd, requestId: body.id, cached: false,
    reasoningChars: String(message.reasoning_content ?? message.reasoning ?? '').length, finishReason: body.choices?.[0]?.finish_reason};
  writeFileSync(cachePath, JSON.stringify(attempt) + '\n');
  return attempt;
}

/** Lenient reading for analysis only: strips a think block / code fence and reads the first JSON object. */
function lenient(raw: string): string {
  const text = raw.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/```(?:json)?/g, '');
  const start = text.indexOf('{'), end = text.lastIndexOf('}');
  return start >= 0 && end > start ? text.slice(start, end + 1) : text;
}

async function modelResult(c: Case): Promise<Result> {
  const attempts: Attempt[] = [], executedBefore = c.simulated, start = performance.now();
  let ids: string[] | null = null, error = '', request = c.stored.prompt, firstPassValid = false;
  try {
    for (let i = 0; i < 2 && !ids; i++) {
      const a = await complete(request); attempts.push(a);
      try { ids = acceptModelText(a.raw.trim(), c.grid, c.config, c.candidates).selected_patch_ids; if (i === 0) firstPassValid = true; }
      catch (e) { error = e instanceof Error ? e.message : 'Invalid model response'; request = `${c.stored.prompt}\n${repairInstruction(e)}`; }
    }
  } catch (e) {
    const infra = asInfraError(e);
    return {family: c.family, variation: c.variation, source: c.source, method: `llm:${llm}`, status: 'infra-failure', ids: null, valid: null, firstPassValid: null,
      objective: null, error: `${infra.kind}: ${infra.message}`, logicalSimulationQueries: 0, executedSimulations: 0, selectionMs: performance.now() - start,
      scoringMs: 0, endToEndMs: performance.now() - start, treatedM2: null, attempts};
  }
  const selected = performance.now(), result = scored(c, ids), end = performance.now();
  let lenientIds: string[] | null = null;
  for (const a of attempts) { try { lenientIds = acceptModelText(lenient(a.raw), c.grid, c.config, c.candidates).selected_patch_ids; break; } catch { /* still invalid */ } }
  return {family: c.family, variation: c.variation, source: c.source, method: `llm:${llm}`, status: 'completed', ids, valid: !!ids, firstPassValid,
    lenientIds, lenientObjective: scored(c, lenientIds).objective, error: ids ? undefined : error,
    logicalSimulationQueries: ids ? 1 : 0, executedSimulations: c.simulated - executedBefore, selectionMs: selected - start, scoringMs: end - selected, endToEndMs: end - start,
    truncatedAttempts: attempts.filter(a => isTruncated(a, local ? localCap : maxTokens)).length, attempts, ...result};
}

// ---- Run --------------------------------------------------------------------------------------
const effective = local ? await localStatus(localUrl) : null;
// EVAL-CX-009: local truncation is judged against the server's effective cap (and its finishReason when reported).
const localCap = typeof (effective?.generation as {maxTokens?: unknown} | undefined)?.maxTokens === 'number' ? (effective!.generation as {maxTokens: number}).maxTokens : null;
const cases = loadCases();
if (!cases.length) throw Error(`No ${split} cases in ${casesDir} for ${SOLVER_VERSION} / ${PROMPT_VERSION}`);
const results: Result[] = [];
// Baselines run on the same cases every time, so model runs carry their own paired comparators (EVAL-CX-005).
const baselineMethods = llm ? ['none', 'random', 'heuristic', 'greedy'] : methods;
for (const c of cases) for (const m of baselineMethods) results.push(await baseline(c, m));
if (llm) {
  let next = 0, done = 0;
  await Promise.all(Array.from({length: concurrency}, async () => {
    while (next < cases.length) {
      const c = cases[next++];
      results.push(await modelResult(c));
      if (++done % 20 === 0) console.log(`${done}/${cases.length} · $${spentUsd.toFixed(4)}`);
    }
  }));
}

// Infrastructure failures (timeouts, HTTP errors) are not model answers: retry each once at the end so a transient
// failure does not leave the cohort incomplete. Invalid answers are never retried here.
let infraRetried = 0;
if (llm) for (const [i, r] of results.entries()) {
  if (r.method !== `llm:${llm}` || r.status !== 'infra-failure') continue;
  // Policy: the retry is a fresh request from the original prompt (temperature 0), so its first pass is a real first
  // answer. The failed try's history is kept on the row and its usage still counts (EVAL-CX-010).
  const retry = await modelResult(cases.find(c => c.family === r.family && c.variation === r.variation)!);
  infraRetried++;
  results[i] = {...retry, infraRetryOf: {error: r.error, attempts: r.attempts, firstPassValid: r.firstPassValid}};
}

// Predeclared (validation, before any held-out look): verify both the model's design and the D8 heuristic's design
// and keep the better one. Two solver runs; never worse than either alone. Invalid model answers fall back to the heuristic.
if (llm) for (const c of cases) {
  const m = results.find(r => r.method === `llm:${llm}` && r.family === c.family && r.variation === c.variation)!;
  const h = results.find(r => r.method === 'heuristic' && r.family === c.family && r.variation === c.variation)!;
  const useModel = m.status === 'completed' && m.objective !== null && m.objective <= h.objective!;
  results.push({...(useModel ? m : h), method: `best-of-2:${llm}`, status: m.status, valid: m.status === 'completed' ? true : null, firstPassValid: m.firstPassValid,
    logicalSimulationQueries: 2, executedSimulations: 0, selectionMs: m.selectionMs + h.selectionMs, scoringMs: m.scoringMs + h.scoringMs, endToEndMs: m.endToEndMs + h.endToEndMs,
    attempts: undefined, truncatedAttempts: undefined});
}

// ---- Metrics ----------------------------------------------------------------------------------
const outcomes: CaseOutcome[] = cases.map(c => ({family: c.family, variation: c.variation, source: c.source, noneM3: c.stored.noInterventionM3, referenceM3: c.stored.objectiveM3}));
const familyList = [...new Set(cases.map(c => c.family))], draws = familyDraws(familyList.length);
const rowsOf = (method: string): MethodRow[] => results.filter(r => r.method === method)
  .map(r => ({family: r.family, variation: r.variation, source: r.source, status: r.status, valid: r.valid, firstPassValid: r.firstPassValid, objective: r.objective}));
const mean = (v: number[]) => v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
const byCase = new Map(cases.map(c => [caseKey(c), c]));
const allAttempts = (r: Result) => [...(r.infraRetryOf?.attempts ?? []), ...(r.attempts ?? [])];   // usage includes a failed try before a retry
const summary: Record<string, unknown> = {};
for (const method of [...new Set(results.map(r => r.method))]) {
  const rs = results.filter(r => r.method === method), completed = rs.filter(r => r.status === 'completed');
  summary[method] = {
    ...marginalSummary(outcomes, rowsOf(method), familyList, draws),
    exactMatch: mean(completed.map(r => r.ids && designKey(r.ids) === designKey(JSON.parse(byCase.get(caseKey(r))!.stored.completion).selected_patch_ids) ? 1 : 0)),
    lenientValidity: rs.some(r => r.lenientIds !== undefined) ? mean(completed.map(r => r.lenientIds ? 1 : 0)) : undefined,
    meanTreatedM2: mean(completed.map(r => r.treatedM2 ?? 0)),
    meanLogicalSimulationQueries: mean(completed.map(r => r.logicalSimulationQueries)), executedSimulations: rs.reduce((s, r) => s + r.executedSimulations, 0),
    meanSelectionMs: mean(completed.map(r => r.selectionMs)), meanScoringMs: mean(completed.map(r => r.scoringMs)), meanEndToEndMs: mean(completed.map(r => r.endToEndMs)),
    repairRequests: rs.reduce((s, r) => s + Math.max(0, (r.attempts?.length ?? 1) - 1), 0),
    truncatedAttempts: rs.reduce((s, r) => s + (r.truncatedAttempts ?? 0), 0),
    inputTokens: rs.reduce((s, r) => s + allAttempts(r).reduce((t, a) => t + a.inputTokens, 0), 0),
    outputTokens: rs.reduce((s, r) => s + allAttempts(r).reduce((t, a) => t + a.outputTokens, 0), 0),
    costUsd: rs.reduce((s, r) => s + allAttempts(r).reduce((t, a) => t + (a.cached ? 0 : a.costUsd), 0), 0),
    invalidReasons: Object.entries(completed.filter(r => !r.valid).reduce((m: Record<string, number>, r) => ({...m, [r.error ?? '?']: (m[r.error ?? '?'] ?? 0) + 1}), {})).slice(0, 8),
    infraFailureReasons: Object.entries(rs.filter(r => r.status === 'infra-failure').reduce((m: Record<string, number>, r) => ({...m, [r.error ?? '?']: (m[r.error ?? '?'] ?? 0) + 1}), {})).slice(0, 8),
  };
}
const comparators = ['heuristic', 'greedy', 'random', 'none'];
const paired = llm ? {
  model: Object.fromEntries(comparators.map(k => [k, pairedComparison(outcomes, rowsOf(`llm:${llm}`), rowsOf(k), familyList, draws)])),
  bestOfTwo: Object.fromEntries(comparators.map(k => [k, pairedComparison(outcomes, rowsOf(`best-of-2:${llm}`), rowsOf(k), familyList, draws)])),
} : null;
const report = {timestamp: new Date().toISOString(), split, tag, solver: SOLVER_VERSION, promptVersion: PROMPT_VERSION, caseVersion: CASE_VERSION, cases: cases.length,
  families: familyList, llm: llm ?? null,
  generation: llm ? {requested: {maxTokens, thinking, concurrency, temperature: 0}, effective: local ? effective : {maxTokens, thinking: thinking === 'off' ? 'requested off (backend may ignore; see truncatedAttempts)' : 'provider default', temperature: 0}} : null,
  spentUsd, requestIds, infraRetried, extraSimulations: cases.reduce((s, c) => s + c.simulated, 0), summary, paired,
  notes: ['Objective: solver peak ponded volume in the target for the proposed design (stored labelling outcome or a fresh simulation).',
    'Captured improvement = (none - method) / (none - reference) over completed cases whose reference improves on no gardens by > 1e-6 m3. The reference is the bounded-search label, not a proven optimum, so values can exceed 1. Invalid answers count as no gardens (0). Infrastructure failures are excluded and reported as coverage.',
    'Estimand: case-weighted pooled mean over qualifying cases. Intervals: 2,000-draw family-cluster bootstrap with one shared set of family draws for every method and every paired comparison.',
    'Baseline timings use cached objective lookups for designs the labelling search already evaluated; they are cache-assisted and are not live speed measurements. executedSimulations counts fresh solver runs in this process.'],
  results: results.map(({attempts, ...r}) => ({...r, attempts: attempts?.map(({raw, ...a}) => ({...a, raw: raw.slice(0, 400)}))}))};
mkdirSync('evidence/planner-eval', {recursive: true});
const out = `evidence/planner-eval/v2-${split}-${tag}.json`;
writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
if (llm?.startsWith('featherless:')) appendFileSync('models/spend.jsonl', JSON.stringify({service: 'featherless', operation: `planner-eval-${split}`, timestamp: new Date().toISOString(),
  model: llm.slice('featherless:'.length), dataset_hash: createHash('sha256').update(cases.map(c => c.stored.prompt).join('\n')).digest('hex'), job_id: out,
  quoted_cost: null, actual_cost: Number(spentUsd.toFixed(6)), unit: 'USD', status: 'complete', requests: requestIds.length, note: 'cost = catalog per-token prices x reported usage'}) + '\n');
console.log(JSON.stringify({out, summary: Object.fromEntries(Object.entries(summary).map(([k, v]: [string, any]) => [k, {validity: v.validity, coverage: v.coverage, meanCapturedImprovement: v.meanCapturedImprovement, ci: v.capturedImprovementCI95}])), paired}, null, 1));
