/**
 * Numerical evidence for the shipped configuration (npm run evidence). Writes evidence/numerical-results.json
 * progressively, so a long run keeps everything finished so far.
 *   1. Garden effect per target: greedy (up to 3 gardens) with the default storm.
 *   2. Conservation: water-balance residuals for every run in this file.
 *   3. Cross-check: li-1.0.0 vs the independent dw-1.0.0 formulation, baseline and design.
 *   4. Temporal refinement: stability coefficient halved (li 0.7 -> 0.35).
 *   5. Domain size: 320 / 384 / 448 / 512 m domains cropped from the 512 m LiDAR export.
 *   6. Spatial refinement: 8 m (block mean of 4 m), 4 m, 2 m (separate 2 m export) on identical geometry.
 */
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {buildings, candidates, preparedSite, targets} from '../src/core/site';
import {runSync} from '../src/core/solver';
import {greedy} from '../src/core/planner';
import {cells, defaultConfig, type Config, type Grid, type Patch, type Region, type Run} from '../src/core/types';
import {summary} from '../src/core/scenario';

mkdirSync('evidence', {recursive: true});
const out: Record<string, any> = {timestamp: new Date().toISOString(), environment: {node: process.version, platform: process.platform, arch: process.arch}, runs: {}};
const save = () => writeFileSync('evidence/numerical-results.json', JSON.stringify(out, null, 2) + '\n');
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
function measured(name: string, g: Grid, c: Config, design: Patch[] = [], courant?: number): Run {
  console.log('Computing', name);
  const r = runSync(g, c, design, {snapshots: false, method: name, courant});
  out.runs[name] = {...summary(r), gridId: g.id, cells: g.nx * g.ny, dx: g.dx, courant: courant ?? 'default'};
  save(); return r;
}

const site = preparedSite(), base = defaultConfig();

// 1. Garden effect per target.
out.gardenEffects = {};
const designs: Record<string, Patch[]> = {};
for (const roi of targets) {
  const c = {...base, roi}, pool = candidates(site, roi);
  const plan = await greedy(site, c, pool, async p => runSync(site, c, p, {snapshots: false}));
  const none = measured(`baseline · ${roi.name}`, site, c), best = measured(`greedy design · ${roi.name}`, site, c, plan.run.patches);
  designs[roi.name] = plan.run.patches;
  out.gardenEffects[roi.name] = {candidates: pool.length, gardens: plan.proposal.selected_patch_ids, simulationCalls: plan.simulationCalls,
    baselineM3: none.objective, designM3: best.objective, reductionM3: none.objective - best.objective, reductionPct: 100 * (1 - best.objective / none.objective),
    heldInGardensAtEndM3: best.balance.garden, intoLakeM3: best.balance.waterBody};
  save();
}
const roi0 = targets[0], c0 = {...base, roi: roi0}, design0 = designs[roi0.name];

// 3. Cross-check against the independent diffusive-wave formulation.
const liBase = runSync(site, c0, [], {snapshots: false}), liDesign = runSync(site, c0, design0, {snapshots: false});
const dwBase = measured('dw-1.0.0 baseline', site, {...c0, solver: 'dw-1.0.0'}), dwDesign = measured('dw-1.0.0 design', site, {...c0, solver: 'dw-1.0.0'}, design0);
out.crossCheck = {target: roi0.name, liBaselineM3: liBase.objective, dwBaselineM3: dwBase.objective, baselineRelativeDifference: rel(dwBase.objective, liBase.objective),
  liEffectM3: liBase.objective - liDesign.objective, dwEffectM3: dwBase.objective - dwDesign.objective,
  wallSeconds: {li: liBase.wallMs / 1000, dw: dwBase.wallMs / 1000}, steps: {li: liBase.steps, dw: dwBase.steps}};
save();

// 4. Temporal refinement.
const halfBase = measured('li courant 0.35 baseline', site, c0, [], .35), halfDesign = measured('li courant 0.35 design', site, c0, design0, .35);
out.temporal = {baselineRelativeChange: rel(halfBase.objective, liBase.objective), designRelativeChange: rel(halfDesign.objective, liDesign.objective),
  effectRelativeChange: rel(halfBase.objective - halfDesign.objective, liBase.objective - liDesign.objective), target: 0.05};
save();

// 5. Domain size, from the 512 m export (domain origin is 16 cells inside it).
const full = JSON.parse(readFileSync('data/derived/park-512m-4m.json', 'utf8'));
const shiftRegion = (r: Region, d: number): Region => ({...r, x: r.x + d, y: r.y + d});
function cropFull(margin: number): Grid {
  const lo = 16 - margin, size = 96 + 2 * margin;
  const z: number[] = [], mask: number[] = [], sink: number[] = [], eligible: number[] = [];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = (lo + y) * 128 + lo + x;
    z.push(full.z[i]); mask.push(full.mask[i]); sink.push(full.sink[i]);
    const dx = x - margin, dy = y - margin, inside = dx >= 0 && dy >= 0 && dx < 96 && dy < 96;
    eligible.push(inside ? site.eligible![dy * 96 + dx] : 0);
  }
  const g: Grid = {id: `domain-${size * 4}m`, name: '', nx: size, ny: size, dx: 4, dy: 4, z, mask, sink, eligible};
  for (const b of buildings) for (const i of cells(g, shiftRegion(b, margin))) { g.mask[i] = 0; g.eligible![i] = 0; g.sink![i] = 0; }
  return g;
}
out.domain = {target: roi0.name, rows: []};
for (const margin of [-8, 0, 8, 16]) {
  const g = cropFull(margin), c = {...c0, roi: shiftRegion(roi0, margin)}, design = design0.map(p => ({...p, ...shiftRegion(p, margin)}));
  const b = measured(`domain ${g.nx * 4} m baseline`, g, c), d = measured(`domain ${g.nx * 4} m design`, g, c, design);
  out.domain.rows.push({extentM: g.nx * 4, baselineM3: b.objective, designM3: d.objective, effectM3: b.objective - d.objective});
  save();
}
const largest = out.domain.rows.at(-1);
out.domain.rows.forEach((r: any) => { r.baselineRelativeToLargest = rel(r.baselineM3, largest.baselineM3); r.effectRelativeToLargest = rel(r.effectM3, largest.effectM3); });
save();

// 6. Spatial refinement on identical physical geometry: target and gardens snapped to even 4 m indices.
const even = (r: Region): Region => ({...r, x: r.x - (r.x % 2), y: r.y - (r.y % 2), width: r.width + (r.width % 2), height: r.height + (r.height % 2)});
const roiE = even(roi0), designE = design0.map(p => ({...p, ...even(p)}));
function coarsen(g: Grid, id: string): Grid {
  // 4 m -> 8 m block mean; masks need every fine cell valid, sink if any fine cell is water.
  const n = g.nx / 2, z: number[] = [], mask: number[] = [], sink: number[] = [], eligible: number[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const ii = [0, 1].flatMap(a => [0, 1].map(b => (2 * y + a) * g.nx + 2 * x + b));
    z.push(ii.reduce((s, i) => s + g.z[i], 0) / 4); mask.push(ii.every(i => g.mask[i]) ? 1 : 0);
    sink.push(ii.some(i => g.sink?.[i]) ? 1 : 0); eligible.push(ii.every(i => g.eligible?.[i]) ? 1 : 0);
  }
  return {id, name: '', nx: n, ny: n, dx: g.dx * 2, dy: g.dy * 2, z, mask, sink, eligible};
}
const fine = JSON.parse(readFileSync('data/derived/park-domain-2m.json', 'utf8'));
const g2: Grid = {id: 'park-domain-2m', name: '', nx: 192, ny: 192, dx: 2, dy: 2, z: fine.z, mask: fine.mask, sink: fine.sink,
  eligible: Array.from({length: 192 * 192}, (_, i) => site.eligible![Math.floor(Math.floor(i / 192) / 2) * 96 + Math.floor(i % 192 / 2)])};
for (const b of buildings) for (const i of cells(g2, {...b, x: b.x * 2, y: b.y * 2, width: b.width * 2, height: b.height * 2})) { g2.mask[i] = 0; g2.eligible![i] = 0; g2.sink![i] = 0; }
const g8 = coarsen(site, 'park-domain-8m');
const at = (r: Region, k: number): Region => ({...r, x: r.x * k, y: r.y * k, width: r.width * k, height: r.height * k});
out.spatial = {target: roiE, note: 'Target and gardens snapped to even 4 m indices so all three grids represent identical areas. 2 m is a separate bilinear resampling of the same nominal 10 ft DEM product; this is combined grid/terrain-sampling sensitivity.', rows: []};
for (const [g, k] of [[g8, .5], [site, 1], [g2, 2]] as const) {
  const c = {...c0, roi: at(roiE, k)}, design = designE.map(p => ({...p, ...at(p, k)}));
  const b = measured(`spatial ${g.dx} m baseline`, g, c), d = measured(`spatial ${g.dx} m design`, g, c, design);
  out.spatial.rows.push({dx: g.dx, cells: g.nx * g.ny, baselineM3: b.objective, designM3: d.objective, effectM3: b.objective - d.objective, peakDepthM: b.peakRoiDepth, wallSeconds: b.wallMs / 1000});
  save();
}
const [r8, r4, r2] = out.spatial.rows;
out.spatial.change8to4 = rel(r4.baselineM3, r8.baselineM3); out.spatial.change4to2 = rel(r2.baselineM3, r4.baselineM3);
out.spatial.effectChange4to2 = rel(r2.effectM3, r4.effectM3);

// 2. Conservation over every run above.
const residuals = Object.values(out.runs).map((r: any) => Math.abs(r.balanceM3.relativeResidual ?? 0));
out.conservation = {runs: residuals.length, maxRelativeResidual: Math.max(...residuals), tolerance: 1e-3};
out.acceptance = {conservation: out.conservation.maxRelativeResidual < 1e-6, temporalUnder5Percent: out.temporal.baselineRelativeChange < .05 && out.temporal.designRelativeChange < .05,
  crossCheckUnder5Percent: out.crossCheck.baselineRelativeDifference < .05, fieldValidation: false};
save();
console.log(JSON.stringify({gardenEffects: out.gardenEffects, crossCheck: out.crossCheck, temporal: out.temporal, domain: out.domain, spatial: out.spatial, acceptance: out.acceptance}, null, 1));
