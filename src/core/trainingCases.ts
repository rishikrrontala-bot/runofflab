/**
 * Planner training cases. A case is one 256 m window of a terrain family with its own storm, soil, target,
 * lawn-like eligible area, garden assumptions and limits. Labels come from the simulator (scripts/dataset.ts);
 * nothing here runs it. Pure functions: real-family grids are loaded by the caller.
 *
 * Families are the unit of independence. Real families are public NJ LiDAR crops (models/data/families,
 * split fixed in scripts/prepare_families.py); synthetic families are explicitly artificial terrain from the
 * analytic archetypes below. The Mercer County Park showcase is never a family.
 */
import { cells, defaultConfig, validatePatches, type Config, type Grid, type Patch, type Region } from './types';
import { candidates } from './site';
import { seededRandom } from './planner';
import { terrainFeatures } from './terrainFeatures';

export type Split = 'train' | 'validation' | 'heldout';
export interface Family { id: string; source: 'real' | 'synthetic'; split: Split; seed: number; grid: Grid }
export interface TrainingCase {
  family: string; source: Family['source']; split: Split; variation: number; seed: number;
  window: { x: number; y: number; size: number }; grid: Grid; config: Config; candidates: Patch[];
}

/** Bump when buildCase() would produce different cases; cached simulator labels are keyed on it. */
export const CASE_VERSION = 'runofflab-cases-1';
export const CASE_WINDOW_CELLS = 32;   // 256 m at 8 m
const hashString = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };

/** Sixteen synthetic families (64 x 64 at 8 m), splits fixed by index before generation: 10 / 3 / 3. */
export function syntheticFamilies(): Family[] {
  return Array.from({ length: 16 }, (_, k) => {
    const id = `synthetic-${String(k + 1).padStart(2, '0')}`, seed = 7301 + k * 971, rng = seededRandom(seed), n = 64, dx = 8;
    const archetype = k % 4, relief = 1 + rng() * 7, tiltX = (rng() - .5) * .02, tiltY = (rng() - .5) * .02;
    const cx = .3 + rng() * .4, cy = .3 + rng() * .4, phases = Array.from({ length: 6 }, () => rng() * Math.PI * 2), micro = .05 + rng() * .25;
    const z: number[] = [];
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const X = (x + .5) / n, Y = (y + .5) / n;
      const bowl = (X - cx) ** 2 + 1.3 * (Y - cy) ** 2;
      const shape = archetype === 0 ? bowl                                               // closed basin
        : archetype === 1 ? Math.abs(X - cx - .3 * (Y - .5)) * .7 + (1 - Y) * .2           // valley draining south
        : archetype === 2 ? bowl + .15 * Math.exp(-(((Y - cy) / .08) ** 2)) * Math.min(1, Math.abs(X - cx) * 6)  // ridge with a saddle
        : .5 * bowl + .06 * Math.sin(X * Math.PI * 5 + phases[0]) * Math.sin(Y * Math.PI * 4 + phases[1]);  // hummocky plain
      const roughness = micro * (Math.sin(X * 23 + phases[2]) * Math.sin(Y * 19 + phases[3]) + .5 * Math.sin((X + Y) * 41 + phases[4]));
      z.push(20 + relief * shape + tiltX * x * dx + tiltY * y * dx + roughness);
    }
    const grid: Grid = { id, name: 'Explicitly synthetic terrain (analytic archetype)', nx: n, ny: n, dx, dy: dx, z, mask: Array(n * n).fill(1), sink: Array(n * n).fill(0) };
    return { id, source: 'synthetic' as const, split: (k < 10 ? 'train' : k < 13 ? 'validation' : 'heldout') as Split, seed, grid };
  });
}

function crop(g: Grid, x0: number, y0: number, size: number, id: string): Grid {
  const pick = <T>(a: T[]) => { const out: T[] = []; for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) out.push(a[(y0 + y) * g.nx + x0 + x]); return out; };
  return { id, name: g.name, nx: size, ny: size, dx: g.dx, dy: g.dy, z: pick(g.z), mask: pick(g.mask), sink: pick(g.sink ?? Array(g.z.length).fill(0)) };
}

/**
 * Deterministic case for (family, variation). Target: a 40 x 24 m window (5 x 3 or 3 x 5 cells) chosen at random
 * among the eight with the largest downhill-routed upstream area, i.e. where runoff gathers; no simulation is used.
 * Eligible area: two to four random ellipses (a stand-in for lawns). Candidates: the site's lattice rule, at most 28,
 * with IDs and presentation order shuffled together so position in the list carries no information.
 */
export function buildCase(family: Family, variation: number): TrainingCase {
  if (!Number.isInteger(variation) || variation < 0) throw Error('Invalid variation');
  const seed = (hashString(family.id) ^ Math.imul(variation + 1, 0x9E3779B1)) >>> 0, rng = seededRandom(seed);
  const size = CASE_WINDOW_CELLS, room = family.grid.nx - size;
  for (let attempt = 0; attempt < 40; attempt++) {
    const x0 = Math.floor(rng() * (room + 1)), y0 = Math.floor(rng() * (room + 1));
    const g = crop(family.grid, x0, y0, size, `${family.id}@${x0},${y0}`);
    const usable = (i: number) => g.mask[i] && !g.sink![i];
    if (g.z.filter((_, i) => usable(i)).length < .95 * g.z.length) continue;

    // Target where runoff gathers.
    const windows: { r: Region; up: number }[] = [];
    for (const [w, h] of [[5, 3], [3, 5]]) for (let y = 1; y + h < size; y += 2) for (let x = 1; x + w < size; x += 2) {
      const r = { x, y, width: w, height: h, name: 'Training target' };
      if (cells(g, r).every(usable)) windows.push({ r, up: terrainFeatures(g, r).targetUpstreamM2 });
    }
    windows.sort((a, b) => b.up - a.up || a.r.y - b.r.y || a.r.x - b.r.x);
    if (!windows.length) continue;
    const roi = windows[Math.floor(rng() * Math.min(8, windows.length))].r;

    // Lawn-like eligible area.
    const ellipses = Array.from({ length: 2 + Math.floor(rng() * 3) }, () => ({ cx: rng() * size, cy: rng() * size, rx: 3 + rng() * 8, ry: 3 + rng() * 8 }));
    g.eligible = g.z.map((_, i) => {
      const x = i % size + .5, y = Math.floor(i / size) + .5;
      return usable(i) && ellipses.some(e => ((x - e.cx) / e.rx) ** 2 + ((y - e.cy) / e.ry) ** 2 <= 1) ? 1 : 0;
    });
    const garden = { pondingDepthM: Math.round((.1 + rng() * .2) * 100) / 100, infiltrationMmH: Math.round(10 + rng() * 50), soilCapacityM: Math.round((.1 + rng() * .2) * 100) / 100 };
    let pool = candidates(g, roi, garden);
    if (pool.length < 6) continue;
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    pool = pool.slice(0, 28);
    const ids = pool.map((_, k) => `G${k + 1}`);
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    pool = pool.map((p, k) => ({ ...p, id: ids[k], name: `Candidate ${ids[k]}` }));

    const maxPatches = rng() < .2 ? 1 : rng() < .375 ? 2 : 3, gardenArea = pool[0].width * pool[0].height * g.dx * g.dy;
    const soilCapacityM = Math.round((.01 + rng() * .07) * 1000) / 1000;
    const config: Config = {
      ...defaultConfig(), roi, maxPatches,
      areaBudgetM2: rng() < .25 && maxPatches > 1 ? (maxPatches - 1) * gardenArea : maxPatches * gardenArea,
      rainMmH: Math.round(30 + rng() * 90), rainMinutes: Math.round(10 + rng() * 35), drainMinutes: Math.round(5 + rng() * 15),
      infiltrationMmH: Math.round(rng() * 15), soilCapacityM, initialSoilM: Math.round(rng() * .7 * soilCapacityM * 1000) / 1000,
      roughness: Math.round((.02 + rng() * .06) * 1000) / 1000, boundary: rng() < .85 ? 'open' : 'closed', outputSeconds: 300,
    };
    pool.forEach(p => validatePatches(g, config, [p]));
    return { family: family.id, source: family.source, split: family.split, variation, seed, window: { x: x0, y: y0, size }, grid: g, config, candidates: pool };
  }
  throw Error(`No usable case window for ${family.id} variation ${variation}`);
}
