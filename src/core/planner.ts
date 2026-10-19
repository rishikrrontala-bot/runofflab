import { cells, patchArea, validatePatches, type Config, type Grid, type Patch, type Plan, type Proposal, type Run } from './types';
import { terrainFeatures } from './terrainFeatures';

export const PROMPT_VERSION = 'runofflab-planner-prompt-5';

/** The learned planner was trained only on li-1.0.0, threshold 0 and a dry initial surface; its prompt does not
 *  encode those settings, so it must not be asked about other ones (PROMPT-CX-001). Returns a reason, or null. */
export function plannerUnsupported(c: Config): string | null {
  if (c.solver !== 'li-1.0.0' || c.thresholdM !== 0 || c.initialDepthM !== 0)
    return 'The trained planner only supports its training settings: the li-1.0.0 solver, all ponded water counted (threshold 0) and a dry starting surface. Greedy planning works with any setting.';
  return null;
}

/** Strict model-output parsing: exactly {"selected_patch_ids": [...]}, known unique IDs, every constraint met. */
export function parseProposal(value: unknown, g: Grid, c: Config, available: Patch[]): Patch[] {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== 1 || !('selected_patch_ids' in value)) throw Error('Expected exactly selected_patch_ids');
  const ids = (value as Proposal).selected_patch_ids;
  if (!Array.isArray(ids) || ids.some(i => typeof i !== 'string') || new Set(ids).size !== ids.length) throw Error('Invalid or duplicate candidate IDs');
  const patches = ids.map(id => { const p = available.find(p => p.id === id); if (!p) throw Error(`Unknown candidate: ${id}`); return p; });
  validatePatches(g, c, patches); return patches;
}
/** Formatting errors (not JSON) and constraint errors get different repair messages. */
export class ModelFormatError extends Error {}
export function acceptModelText(text: string, g: Grid, c: Config, available: Patch[]): Proposal {
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new ModelFormatError('Response was not a bare JSON object'); }
  return { selected_patch_ids: parseProposal(parsed, g, c, available).map(p => p.id) };
}
export function repairInstruction(error: unknown): string {
  return error instanceof ModelFormatError
    ? 'Formatting only: reply with the bare JSON object, with no code fences, labels or explanation.'
    : `The selection broke a rule (${error instanceof Error ? error.message : 'invalid'}). Choose again within the limits.`;
}

/** Production-available features: terrain, scenario and candidate geometry only. No simulation result enters. */
export function features(g: Grid, c: Config, available: Patch[]) {
  const routing = terrainFeatures(g, c.roi), area = g.dx * g.dy, target = cells(g, c.roi), targetSet = new Set(target);
  /** Unique upstream area whose D8 path crosses the garden and afterwards reaches the target. Each source cell is
   *  counted once even if its path leaves and re-enters the garden (FEAT-CX-001); target cells end a path. */
  const routedThrough = (garden: Set<number>) => {
    const through = new Uint8Array(g.z.length); let count = 0;
    for (const i of routing.order) {   // rising elevation: a cell's downhill neighbour is resolved first
      through[i] = targetSet.has(i) ? 0 : garden.has(i) ? routing.reachesTarget[i] : routing.next[i] >= 0 ? through[routing.next[i]] : 0;
      count += through[i];
    }
    return count * area;
  };
  const targetZ = target.reduce((s, i) => s + g.z[i], 0) / target.length;
  const tx = c.roi.x + c.roi.width / 2, ty = c.roi.y + c.roi.height / 2;
  const slopeAt = (i: number) => {
    const x = i % g.nx, y = Math.floor(i / g.nx);
    const zx = (g.z[y * g.nx + Math.min(g.nx - 1, x + 1)] - g.z[y * g.nx + Math.max(0, x - 1)]) / ((Math.min(g.nx - 1, x + 1) - Math.max(0, x - 1)) * g.dx);
    const zy = (g.z[Math.min(g.ny - 1, y + 1) * g.nx + x] - g.z[Math.max(0, y - 1) * g.nx + x]) / ((Math.min(g.ny - 1, y + 1) - Math.max(0, y - 1)) * g.dy);
    return Math.hypot(zx, zy);
  };
  return {
    gridM: g.dx,
    storm: { intensityMmH: c.rainMmH, durationMin: c.rainMinutes, drainMin: c.drainMinutes },
    ground: { infiltrationMmH: c.infiltrationMmH, soilM: c.soilCapacityM, initialSoilM: c.initialSoilM, roughness: c.roughness, edges: c.boundary },
    target: { areaM2: target.length * area, elevationM: targetZ, upstreamM2: routing.targetUpstreamM2 },
    limits: { maxGardens: c.maxPatches, budgetM2: c.areaBudgetM2 },
    candidates: available.map(p => {
      const ii = cells(g, p), routed = routedThrough(new Set(ii));
      return {
        id: p.id, areaM2: ii.length * area,
        eastM: (p.x + p.width / 2 - tx) * g.dx, southM: (p.y + p.height / 2 - ty) * g.dy,
        distanceM: Math.hypot((p.x + p.width / 2 - tx) * g.dx, (p.y + p.height / 2 - ty) * g.dy),
        relElevationM: ii.reduce((s, i) => s + g.z[i], 0) / ii.length - targetZ,
        slopePct: 100 * ii.reduce((s, i) => s + slopeAt(i), 0) / ii.length,
        routedToTargetM2: routed,
        reachFraction: ii.filter(i => routing.reachesTarget[i]).length / ii.length,
        maxUpstreamM2: Math.max(...ii.map(i => routing.contributingAreaM2[i])),
        storageM3: p.pondingDepthM * ii.length * area, pondingDepthM: p.pondingDepthM, infiltrationMmH: p.infiltrationMmH, soilM: p.soilCapacityM,
      };
    }),
  };
}

const num = (v: number, decimals: number) => { const r = Number(v.toFixed(decimals)); return Object.is(r, -0) ? '0' : String(r); };

/** The single prompt used for training data, local serving and hosted evaluation. Numbers are rounded the
 *  same way everywhere, so training and serving inputs are identical for identical scenarios. */
export function plannerPrompt(g: Grid, c: Config, available: Patch[]): string {
  const f = features(g, c, available), first = f.candidates[0];
  const uniform = f.candidates.every(k => k.pondingDepthM === first?.pondingDepthM && k.infiltrationMmH === first?.infiltrationMmH && k.soilM === first?.soilM && k.areaM2 === first?.areaM2);
  const gardenLine = first && uniform
    ? `Each garden: ${num(first.areaM2, 0)} m2, stores up to ${num(first.pondingDepthM, 3)} m of runoff, infiltrates ${num(first.infiltrationMmH, 1)} mm/h into ${num(first.soilM, 3)} m of soil.`
    : 'Gardens differ; see storage_m3 per candidate.';
  return [
    `Choose rain gardens that most reduce the peak volume of water ponded in the target during this storm. Use at most ${f.limits.maxGardens} garden${f.limits.maxGardens === 1 ? '' : 's'} and ${num(f.limits.budgetM2, 0)} m2 in total. Return an empty list only if no garden would lower the target's peak.`,
    'A garden helps by capturing runoff that would otherwise flow over the ground into the target; water moves downhill across the surface in two dimensions.',
    'Reply with only the JSON object {"selected_patch_ids": [...]}, with no code fences or explanation.',
    `Storm: ${num(f.storm.intensityMmH, 1)} mm/h for ${num(f.storm.durationMin, 1)} min, then ${num(f.storm.drainMin, 1)} min without rain.`,
    `Ground: infiltration ${num(f.ground.infiltrationMmH, 1)} mm/h, soil storage ${num(f.ground.soilM, 3)} m (initially ${num(f.ground.initialSoilM, 3)} m), Manning n ${num(f.ground.roughness, 3)}, ${f.ground.edges} edges, ${num(f.gridM, 1)} m cells.`,
    gardenLine,
    `Target: ${num(f.target.areaM2, 0)} m2, mean elevation ${num(f.target.elevationM, 2)} m, downhill-routed upstream area ${num(f.target.upstreamM2, 0)} m2.`,
    'Columns: east_m and south_m = garden centre minus target centre; distance_m = straight-line distance; rel_elev_m = garden mean elevation minus target mean elevation (positive means uphill of the target); slope_pct = mean ground slope; routed_to_target_m2 = upstream area whose steepest-downhill (D8) path passes through the garden and then reaches the target; reach_frac = share of garden cells whose D8 path reaches the target; max_upstream_m2 = largest D8 upstream area inside the garden; storage_m3 = runoff the garden can hold. D8 values are simple estimates that miss some two-dimensional flow.',
    'Candidates (id,east_m,south_m,distance_m,rel_elev_m,slope_pct,routed_to_target_m2,reach_frac,max_upstream_m2,storage_m3):',
    ...f.candidates.map(k => [k.id, num(k.eastM, 0), num(k.southM, 0), num(k.distanceM, 0), num(k.relElevationM, 2), num(k.slopePct, 1),
      num(k.routedToTargetM2, 0), num(k.reachFraction, 2), num(k.maxUpstreamM2, 0), num(k.storageM3, 1)].join(',')),
  ].join('\n');
}

/** All feasible designs (used for small exhaustive checks only; grows combinatorially). */
export function feasibleDesigns(g: Grid, c: Config, available: Patch[]): Patch[][] {
  const out: Patch[][] = [[]];
  function visit(index: number, chosen: Patch[]) {
    if (chosen.length >= c.maxPatches) return;
    for (let i = index; i < available.length; i++) {
      const p = [...chosen, available[i]];
      try { validatePatches(g, c, p); } catch { continue; }
      out.push(p); visit(i + 1, p);
    }
  }
  visit(0, []); return out;
}

export const designKey = (patches: Patch[] | string[]) => patches.map(p => typeof p === 'string' ? p : p.id).sort().join(',');

/** Lower objective wins; ties go to less treated area, then the lexically first ID set. */
export function better(g: Grid, a: Run, b: Run): boolean {
  if (Math.abs(a.objective - b.objective) > 1e-8) return a.objective < b.objective;
  const aa = patchArea(g, a.patches), bb = patchArea(g, b.patches);
  return aa !== bb ? aa < bb : designKey(a.patches) < designKey(b.patches);
}

type Evaluate = (p: Patch[]) => Promise<Run>;
/** Memoized evaluation so a search never pays twice for the same design. */
function memo(evaluate: Evaluate) {
  const cache = new Map<string, Promise<Run>>(); let calls = 0;
  return { calls: () => calls, cache, evaluate: (p: Patch[]) => {
    const k = designKey(p); let r = cache.get(k);
    if (!r) { calls++; r = evaluate(p); cache.set(k, r); }
    return r;
  } };
}

/** Greedy forward selection using simulator-evaluated marginal benefit. Stops when no addition improves. */
export async function greedy(g: Grid, c: Config, available: Patch[], evaluate: Evaluate, progress: (done: number, total: number) => void = () => {}): Promise<Plan> {
  const start = performance.now(), m = memo(evaluate), upper = 1 + available.length * c.maxPatches;
  let best = await m.evaluate([]), selected: Patch[] = [];
  for (let round = 0; round < c.maxPatches; round++) {
    let chosen = best;
    for (const p of available) {
      if (selected.some(q => q.id === p.id)) continue;
      const trial = [...selected, p];
      try { validatePatches(g, c, trial); } catch { continue; }
      const run = await m.evaluate(trial); progress(m.calls(), upper);
      if (better(g, run, chosen)) chosen = run;
    }
    if (chosen === best) break;
    best = chosen; selected = best.patches;
  }
  return { proposal: { selected_patch_ids: selected.map(p => p.id) }, run: best, method: 'Greedy · solver-evaluated marginal benefit', simulationCalls: m.calls(), wallMs: performance.now() - start };
}

/**
 * Bounded search used to label training data: greedy, then repeated best-improvement passes over single
 * swaps, additions and removals until a full pass finds nothing better. The label is the best design the
 * search evaluated; every evaluated design and its objective is returned for exact scoring later.
 */
export async function searchBest(g: Grid, c: Config, available: Patch[], evaluate: Evaluate, maxPasses = 4) {
  const start = performance.now(), m = memo(evaluate);
  const greedyPlan = await greedy(g, c, available, m.evaluate);
  let best = greedyPlan.run, passes = 0;
  for (; passes < maxPasses; passes++) {
    const selected = best.patches, ids = new Set(selected.map(p => p.id)), neighbours: Patch[][] = [];
    for (const p of selected) neighbours.push(selected.filter(q => q.id !== p.id));
    for (const u of available) if (!ids.has(u.id)) {
      neighbours.push([...selected, u]);
      for (const p of selected) neighbours.push([...selected.filter(q => q.id !== p.id), u]);
    }
    let improved = best;
    for (const trial of neighbours) {
      try { validatePatches(g, c, trial); } catch { continue; }
      const run = await m.evaluate(trial);
      if (better(g, run, improved)) improved = run;
    }
    if (improved === best) break;
    best = improved;
  }
  const evaluated = await Promise.all([...m.cache.entries()].map(async ([k, r]) => ({ ids: k ? k.split(',') : [], run: await r })));
  return { best, greedy: greedyPlan, passes, simulationCalls: m.calls(), evaluated, wallMs: performance.now() - start };
}

export function seededRandom(seed: number): () => number {
  let a = seed >>> 0; return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function randomDesign(g: Grid, c: Config, available: Patch[], seed: number): Patch[] {
  const rng = seededRandom(seed), pool = [...available];
  for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const selected: Patch[] = [];
  for (const p of pool) try { validatePatches(g, c, [...selected, p]); selected.push(p); } catch { /* infeasible candidate */ }
  return selected;
}
/** Simple production-available heuristic: fill the budget with the gardens that route the most upstream area
 *  to the target (D8), nearest first on ties. No simulation. */
export function heuristic(g: Grid, c: Config, available: Patch[]): Patch[] {
  const data = new Map(features(g, c, available).candidates.map(k => [k.id, k]));
  const ordered = [...available].sort((a, b) => data.get(b.id)!.routedToTargetM2 - data.get(a.id)!.routedToTargetM2 || data.get(a.id)!.distanceM - data.get(b.id)!.distanceM || a.id.localeCompare(b.id));
  const selected: Patch[] = [];
  for (const p of ordered) {
    if (data.get(p.id)!.routedToTargetM2 <= 0) break;
    try { validatePatches(g, c, [...selected, p]); selected.push(p); } catch { /* budget */ }
  }
  return selected;
}
