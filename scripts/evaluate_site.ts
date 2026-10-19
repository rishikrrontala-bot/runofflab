/**
 * Out-of-distribution check on the real showcase site (4 m, real lawn, buildings, lake), which is never in training.
 * Scenarios: the three rule-selected targets x four storms. For each, the bounded search used for training labels
 * (greedy + swap passes) finds the reference design. Scenarios are cached in evidence/site-scenarios.json under a
 * SHA-256 fingerprint of everything they depend on and rebuilt when it changes (EVAL-CX-004).
 *
 *   node --import tsx scripts/evaluate_site.ts                                   # scenarios + baselines + live greedy timing
 *   node --import tsx scripts/evaluate_site.ts --llm local:http://127.0.0.1:8787 --tag trained-4b
 *
 * Scoring: invalid model answers count as no gardens (captured 0 when the reference has room); infrastructure failures
 * are reported separately (EVAL-CX-001). Speed (EVAL-CX-003): live greedy on the default storm for each target versus
 * model inference plus one verification simulation, both measured end to end in this process.
 */
import {existsSync, readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {preparedSite, candidates, targets} from '../src/core/site';
import {runSync} from '../src/core/solver';
import {acceptModelText, designKey, greedy, heuristic, plannerPrompt, randomDesign, repairInstruction, searchBest, PROMPT_VERSION} from '../src/core/planner';
import {defaultConfig, SOLVER_VERSION, type Config, type Patch} from '../src/core/types';
import {asInfraError, capturedImprovement, localInfer, localStatus, siteScenarioFingerprint} from './evaluation-metrics';

const arg = (name: string) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : process.argv.find(a => a.startsWith(`--${name}=`))?.split('=')[1]; };
const llm = arg('llm'), tag = arg('tag') ?? 'baselines', cachePath = 'evidence/site-scenarios.json';
const SEARCH_POLICY = 'searchBest: greedy then best-improvement swap/add/remove passes, maxPasses 4';
const storms = [
  {name: 'default 60 mm/h, 20 min', rainMmH: 60, rainMinutes: 20, drainMinutes: 10},
  {name: 'short burst 100 mm/h, 10 min', rainMmH: 100, rainMinutes: 10, drainMinutes: 10},
  {name: 'long 35 mm/h, 45 min', rainMmH: 35, rainMinutes: 45, drainMinutes: 15},
  {name: 'wet soil 60 mm/h, 20 min', rainMmH: 60, rainMinutes: 20, drainMinutes: 10, initialSoilM: .03},
];
const g = preparedSite();
type Scenario = {key: string; fingerprint: string; target: string; storm: string; config: Config; prompt: string; noneM3: number; bestIds: string[]; bestM3: number;
  greedyIds: string[]; greedyM3: number; greedyCalls: number; searchCalls: number; outcomes: Record<string, number>};
const cache: {scenarios: Scenario[]} = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, 'utf8')) : {scenarios: []};
const save = () => { mkdirSync('evidence', {recursive: true}); writeFileSync(cachePath, JSON.stringify(cache, null, 1) + '\n'); };

let rebuilt = 0;
const scenarios: Scenario[] = [];
for (const roi of targets) for (const s of storms) {
  const key = `${roi.name} · ${s.name}`, {name: _name, ...storm} = s;
  const config: Config = {...defaultConfig(), ...storm, roi}, pool = candidates(g, roi), prompt = plannerPrompt(g, config, pool);
  const fingerprint = siteScenarioFingerprint({grid: g, config, candidates: pool, prompt, solverVersion: SOLVER_VERSION, promptVersion: PROMPT_VERSION, searchPolicy: SEARCH_POLICY});
  let sc = cache.scenarios.find(x => x.key === key && x.fingerprint === fingerprint);
  if (!sc) {
    console.log('Searching', key); rebuilt++;
    const search = await searchBest(g, config, pool, async p => runSync(g, config, p, {snapshots: false}));
    sc = {key, fingerprint, target: roi.name, storm: s.name, config, prompt, noneM3: search.evaluated.find(e => !e.ids.length)!.run.objective,
      bestIds: search.best.patches.map(p => p.id), bestM3: search.best.objective, greedyIds: search.greedy.proposal.selected_patch_ids, greedyM3: search.greedy.run.objective,
      greedyCalls: search.greedy.simulationCalls, searchCalls: search.simulationCalls, outcomes: Object.fromEntries(search.evaluated.map(e => [designKey(e.ids), e.run.objective]))};
    cache.scenarios = [...cache.scenarios.filter(x => x.key !== key), sc]; save();
  }
  scenarios.push(sc);
}

const poolOf = (sc: Scenario) => candidates(g, sc.config.roi);
const objective = (sc: Scenario, ids: string[]) => {
  const k = designKey(ids);
  if (sc.outcomes[k] === undefined) { sc.outcomes[k] = runSync(g, sc.config, ids.map(id => poolOf(sc).find(p => p.id === id)!), {snapshots: false}).objective; save(); }
  return sc.outcomes[k];
};
const status = llm ? await localStatus(llm.slice('local:'.length)) : null;
if (llm && !llm.startsWith('local:')) throw Error('Site evaluation supports local models only');
const results: Record<string, unknown>[] = [];
for (const sc of scenarios) {
  const pool = poolOf(sc), captured = (v: number | null) => capturedImprovement(sc.noneM3, sc.bestM3, v);
  const row: Record<string, unknown> = {key: sc.key, storm: sc.storm, fingerprint: sc.fingerprint, noneM3: sc.noneM3, bestM3: sc.bestM3, bestIds: sc.bestIds, bestReductionPct: 100 * (1 - sc.bestM3 / sc.noneM3)};
  if (!llm) {
    const h = heuristic(g, sc.config, pool).map(p => p.id), r = randomDesign(g, sc.config, pool, 42).map(p => p.id);
    Object.assign(row, {heuristicIds: h, heuristicCaptured: captured(objective(sc, h)), randomIds: r, randomCaptured: captured(objective(sc, r)),
      greedyIds: sc.greedyIds, greedyCaptured: captured(sc.greedyM3), greedyCalls: sc.greedyCalls, searchCalls: sc.searchCalls});
    if (sc.storm.startsWith('default')) {   // live greedy timing: every simulation executed now, none from cache
      const t = performance.now();
      const plan = await greedy(g, sc.config, pool, async p => runSync(g, sc.config, p, {snapshots: false}));
      Object.assign(row, {liveGreedySeconds: (performance.now() - t) / 1000, liveGreedyExecutedSimulations: plan.simulationCalls, liveGreedyIds: plan.proposal.selected_patch_ids});
    }
  } else {
    const url = llm.slice('local:'.length);
    let request = sc.prompt, ids: string[] | null = null, error = '', firstPassValid = false, inferenceSeconds = 0, infra: string | null = null;
    const raw: string[] = [], start = performance.now();
    try {
      for (let attempt = 0; attempt < 2 && !ids; attempt++) {
        const r = await localInfer(url, request, {expectedCheckpoint: status?.checkpoint as string | undefined});
        inferenceSeconds += r.latencyMs / 1000; raw.push(r.output);
        try { ids = acceptModelText(r.output.trim(), g, sc.config, pool).selected_patch_ids; if (attempt === 0) firstPassValid = true; }
        catch (e) { error = e instanceof Error ? e.message : 'invalid'; request = `${sc.prompt}\n${repairInstruction(e)}`; }
      }
    } catch (e) { const x = asInfraError(e); infra = `${x.kind}: ${x.message}`; }
    if (infra) Object.assign(row, {status: 'infra-failure', error: infra});
    else {
      // Verification: one live simulation of the proposed design (or of no gardens when the answer was invalid).
      const t = performance.now();
      const verified = runSync(g, sc.config, (ids ?? []).map(id => pool.find(p => p.id === id)!) as Patch[], {snapshots: false}).objective;
      const verifySeconds = (performance.now() - t) / 1000;
      Object.assign(row, {status: 'completed', modelIds: ids, valid: !!ids, firstPassValid, error: ids ? undefined : error, effectiveObjectiveM3: verified,
        modelCaptured: captured(verified), inferenceSeconds, verifySeconds, endToEndSeconds: (performance.now() - start) / 1000, raw});
    }
  }
  results.push(row); console.log(JSON.stringify({key: row.key, ...(llm ? {ids: row.modelIds, captured: row.modelCaptured, s: row.endToEndSeconds} : {h: row.heuristicCaptured, g: row.greedyCaptured, live: row.liveGreedySeconds})}));
}
const done = results.filter(r => !llm || r.status === 'completed'), mean = (k: string) => {
  const v = done.map(r => r[k]).filter((x): x is number => typeof x === 'number'); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };
mkdirSync('evidence/planner-eval', {recursive: true});
writeFileSync(`evidence/planner-eval/site-${tag}.json`, JSON.stringify({timestamp: new Date().toISOString(), solver: SOLVER_VERSION, promptVersion: PROMPT_VERSION,
  searchPolicy: SEARCH_POLICY, scenariosRebuilt: rebuilt, llm: llm ?? null, effective: status,
  summary: llm ? {scenarios: results.length, completed: done.length, infraFailures: results.length - done.length,
      validity: done.length ? done.filter(r => r.valid).length / done.length : null, meanCaptured: mean('modelCaptured'),
      // EVAL-CX-008: speed is compared on the same workload as live greedy (default storm, three targets).
      defaultStormMeanEndToEndSeconds: (() => { const v = done.filter(r => String(r.storm).startsWith('default')).map(r => r.endToEndSeconds as number); return v.length === targets.length ? v.reduce((s, x) => s + x, 0) / v.length : null; })(),
      meanInferenceSeconds: mean('inferenceSeconds'), meanVerifySeconds: mean('verifySeconds')}
    : {meanHeuristicCaptured: mean('heuristicCaptured'), meanGreedyCaptured: mean('greedyCaptured'), meanRandomCaptured: mean('randomCaptured'), meanLiveGreedySeconds: mean('liveGreedySeconds')},
  notes: ['Invalid answers count as no gardens; infrastructure failures are excluded and listed.', 'liveGreedySeconds: greedy re-run with fresh simulations (default storm only).',
    'endToEndSeconds: model inference (with at most one repair) plus one verification simulation, measured live.'], results}, null, 2) + '\n');
