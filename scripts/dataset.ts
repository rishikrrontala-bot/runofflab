/**
 * Simulator-labelled planner dataset (v2).
 *
 *   node --import tsx scripts/dataset.ts --pilot                 # throughput pilot, train families only
 *   node --import tsx scripts/dataset.ts --full --workers=10     # full generation, sharded across processes
 *
 * For every case: rebuild it deterministically (buildCase), run the bounded search (greedy, then swap/add/remove
 * passes) with the same solver the app uses, and keep the best evaluated design as the label. Every evaluated
 * design and its objective is stored so any later proposal on the same case can be scored exactly. Cases are
 * cached in models/data/cases-v2 and reused when the solver and prompt versions match.
 */
import {mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync, appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {buildCase, CASE_VERSION} from '../src/core/trainingCases';
import {loadFamilies} from './families';
import {runSync} from '../src/core/solver';
import {PROMPT_VERSION, designKey, plannerPrompt, searchBest} from '../src/core/planner';
import {SOLVER_VERSION, patchArea} from '../src/core/types';

const CASES = 'models/data/cases-v2', OUT = 'models/data/v2';
const PER_FAMILY = {real: {train: 180, validation: 60, heldout: 60}, synthetic: {train: 130, validation: 60, heldout: 60}};
const arg = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.split('=')[1];
const allowPartial = process.argv.includes('--allow-partial');
const pilot = process.argv.includes('--pilot'), assembleOnly = process.argv.includes('--assemble'), full = process.argv.includes('--full') || assembleOnly, workers = Number(arg('workers') ?? 1);
if (!pilot && !full) throw Error('Choose --pilot, --full or --assemble');
const cachedRow = (row: {solver?: string; caseVersion?: string}) => row.solver === SOLVER_VERSION && (row.caseVersion ?? 'runofflab-cases-1') === CASE_VERSION;

const families = loadFamilies();
const jobs = families.flatMap(f => Array.from({length: pilot ? (f.split === 'train' && families.filter(x => x.split === 'train').indexOf(f) % 5 === 0 ? 6 : 0) : PER_FAMILY[f.source][f.split]}, (_, v) => ({f, v})));

if (assembleOnly) { assemble(0); process.exit(0); }
if (full && workers > 1 && !arg('shard')) {
  const started = performance.now();
  await Promise.all(Array.from({length: workers}, (_, i) => new Promise<void>((resolve, reject) => {
    const child = spawn(process.execPath, ['--import', 'tsx', fileURLToPath(import.meta.url), '--full', `--shard=${i}/${workers}`], {stdio: 'inherit'});
    child.on('error', reject); child.on('exit', code => code === 0 ? resolve() : reject(Error(`Shard ${i} exited ${code}`)));
  })));
  assemble((performance.now() - started) / 1000);
  process.exit(0);
}

const [shard, shardCount] = (arg('shard') ?? '0/1').split('/').map(Number);
mkdirSync(CASES, {recursive: true}); mkdirSync('evidence', {recursive: true});
const mine = jobs.filter((_, i) => i % shardCount === shard);
const started = performance.now();
let done = 0, cached = 0, rejected = 0, calls = 0, maxResidual = 0;
for (const {f, v} of mine) {
  const path = `${CASES}/${f.id}-${v}.json`;
  if (existsSync(path)) {
    const row = JSON.parse(readFileSync(path, 'utf8'));
    if (cachedRow(row)) { cached++; done++; continue; }
  }
  try {
    const c = buildCase(f, v), t0 = performance.now();
    const search = await searchBest(c.grid, c.config, c.candidates, async p => {
      const r = runSync(c.grid, c.config, p, {snapshots: false});
      maxResidual = Math.max(maxResidual, Math.abs(r.balance.relativeResidual ?? 0));
      if (Math.abs(r.balance.relativeResidual ?? 0) > 1e-6 || !Number.isFinite(r.objective)) throw Error('Unqualified numerical label');
      return r;
    });
    calls += search.simulationCalls;
    const none = search.evaluated.find(e => e.ids.length === 0)!.run.objective;
    const row = {
      family: f.id, source: f.source, split: f.split, variation: v, seed: c.seed, window: c.window, solver: SOLVER_VERSION, caseVersion: CASE_VERSION, promptVersion: PROMPT_VERSION,
      resolutionM: c.grid.dx, config: c.config, candidates: c.candidates,
      prompt: plannerPrompt(c.grid, c.config, c.candidates),
      completion: JSON.stringify({selected_patch_ids: search.best.patches.map(p => p.id).sort((a, b) => a.localeCompare(b, 'en', {numeric: true}))}),
      objectiveM3: search.best.objective, noInterventionM3: none, greedyObjectiveM3: search.greedy.run.objective, greedyIds: search.greedy.proposal.selected_patch_ids,
      search: {method: 'greedy + best-improvement swap/add/remove passes', passes: search.passes, simulationCalls: search.simulationCalls, wallMs: performance.now() - t0},
      outcomes: search.evaluated.map(e => ({ids: e.ids, objectiveM3: e.run.objective, treatedM2: patchArea(c.grid, e.run.patches)})),
    };
    writeFileSync(path, JSON.stringify(row) + '\n');
    done++;
  } catch (e) {
    rejected++; appendFileSync('models/data/rejected-v2.jsonl', JSON.stringify({family: f.id, variation: v, error: String(e)}) + '\n');
  }
  if ((done + rejected) % 10 === 0) console.log(JSON.stringify({shard, done, of: mine.length, cached, rejected, elapsedS: Math.round((performance.now() - started) / 1000)}));
}
const report = {shard, shardCount, pilot, cases: done, cached, rejected, simulationCalls: calls, maxRelativeResidual: maxResidual, seconds: (performance.now() - started) / 1000};
writeFileSync(`evidence/dataset-v2-${pilot ? 'pilot' : `shard-${shard}`}.json`, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
if (pilot || shardCount === 1) assemble(report.seconds);

/** Writes split JSONL files (prompt/completion only) and a manifest with checksums and dataset statistics. */
function assemble(seconds: number) {
  mkdirSync(OUT, {recursive: true});
  const byId = new Map(families.map(f => [f.id, f]));
  const rows = readdirSync(CASES).filter(f => f.endsWith('.json')).map(f => JSON.parse(readFileSync(`${CASES}/${f}`, 'utf8')))
    .filter(r => cachedRow(r) && jobs.some(j => j.f.id === r.family && j.v === r.variation))
    .sort((a, b) => a.family.localeCompare(b.family) || a.variation - b.variation);
  // Prompts are a pure function of the (deterministic) case, so they are rebuilt here with the current
  // PROMPT_VERSION; the expensive simulator labels are reused unchanged.
  let refreshed = 0;
  for (const r of rows) {
    if (r.promptVersion === PROMPT_VERSION) continue;
    const c = buildCase(byId.get(r.family)!, r.variation);
    r.prompt = plannerPrompt(c.grid, c.config, c.candidates); r.promptVersion = PROMPT_VERSION; r.caseVersion = CASE_VERSION; refreshed++;
    writeFileSync(`${CASES}/${r.family}-${r.variation}.json`, JSON.stringify(r) + '\n');
  }
  if (refreshed) console.log(`Rebuilt prompts for ${refreshed} cached cases (${PROMPT_VERSION})`);
  // DATA-CX-001: the standard export paths only ever hold the complete planned experiment.
  const have = new Set(rows.map(r => `${r.family}#${r.variation}`)), missing = jobs.filter(j => !have.has(`${j.f.id}#${j.v}`));
  if (missing.length && !pilot && !allowPartial) throw Error(`${missing.length} planned cases are missing (e.g. ${missing.slice(0, 3).map(j => `${j.f.id}-${j.v}`).join(', ')}); refusing to write full exports. Use --allow-partial for diagnostics only.`);
  const promptSplits = new Map<string, Set<string>>();
  for (const r of rows) {
    const h = createHash('sha256').update(r.prompt).digest('hex');
    promptSplits.set(h, (promptSplits.get(h) ?? new Set()).add(r.split));
  }
  const crossSplit = [...promptSplits.values()].filter(s => s.size > 1).length;
  if (crossSplit) throw Error(`${crossSplit} prompts appear in more than one split; refusing to export`);
  const files: Record<string, unknown> = {};
  for (const split of ['train', 'validation', 'heldout'] as const) {
    const path = `${OUT}/${pilot ? 'pilot-' : allowPartial ? 'partial-' : ''}${split}.jsonl`, selected = rows.filter(r => r.split === split);
    writeFileSync(path, selected.map(r => JSON.stringify({prompt: r.prompt, completion: r.completion})).join('\n') + (selected.length ? '\n' : ''));
    files[path] = {rows: selected.length, bytes: statSync(path).size, sha256: createHash('sha256').update(readFileSync(path)).digest('hex'),
      families: [...new Set(selected.map(r => r.family))], realRows: selected.filter(r => r.source === 'real').length};
  }
  const gain = rows.map(r => r.noInterventionM3 > 1e-6 ? 1 - r.objectiveM3 / r.noInterventionM3 : 0);
  const greedyGap = rows.filter(r => r.greedyObjectiveM3 - r.objectiveM3 > 1e-8).length;
  const report = {timestamp: new Date().toISOString(), solver: SOLVER_VERSION, caseVersion: CASE_VERSION, promptVersion: PROMPT_VERSION, pilot, partial: allowPartial, cases: rows.length, seconds,
    plannedCases: jobs.length, missingCases: missing.length, crossSplitPromptDuplicates: crossSplit,
    familiesManifestSha256: createHash('sha256').update(readFileSync('data/families-manifest.json')).digest('hex'),
    secondsPerCase: seconds / Math.max(1, rows.length), meanSimulationCalls: rows.reduce((s, r) => s + r.search.simulationCalls, 0) / Math.max(1, rows.length),
    emptyLabels: rows.filter(r => JSON.parse(r.completion).selected_patch_ids.length === 0).length,
    labelsBetterThanGreedy: greedyGap, meanRelativeReduction: gain.reduce((s, v) => s + v, 0) / Math.max(1, gain.length),
    meanCandidates: rows.reduce((s, r) => s + r.candidates.length, 0) / Math.max(1, rows.length),
    designKeyExample: rows[0] ? designKey(JSON.parse(rows[0].completion).selected_patch_ids) : null, files,
    limitations: ['Labels are the best design found by a bounded search, not a proven optimum; see labelsBetterThanGreedy and the exhaustive check.',
      'Real families are bare-earth LiDAR without buildings; eligible areas are random ellipses, not mapped lawns.',
      'Simulated objective on 8 m grids; the showcase is 4 m and is never in training.']};
  writeFileSync(`evidence/dataset-v2-${pilot ? 'pilot' : allowPartial ? 'partial' : 'full'}.json`, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 1));
}
