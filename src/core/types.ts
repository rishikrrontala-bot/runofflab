/** Solver formulations. li-1.0.0 (local inertial, Bates et al. 2010) is the default; dw-1.0.0
 *  (regularized diffusive wave) is kept as an independent reference implementation. */
export const SOLVERS = ['li-1.0.0', 'dw-1.0.0'] as const;
export type SolverId = (typeof SOLVERS)[number];
export const SOLVER_VERSION: SolverId = 'li-1.0.0';

export interface Region { x: number; y: number; width: number; height: number; name: string }
export interface Grid {
  id: string; name: string; nx: number; ny: number; dx: number; dy: number;
  /** Ground elevation (m), row-major, row 0 = north. */
  z: number[];
  /** 1 = computed cell, 0 = nodata or building footprint (closed faces, no rain). */
  mask: number[];
  /** 1 = a rain garden may be placed here. */
  eligible?: number[];
  /** 1 = hydro-flattened water body, treated as a fixed-level open sink. */
  sink?: number[];
  /** The study area shown to the user inside a larger computational domain. */
  site?: Region;
  crs?: string; bounds?: number[]; sourceYears?: string; catalogEdition?: string;
}
/** A rain garden: a bermed basin that stores up to pondingDepthM of runoff over its footprint,
 *  which then infiltrates into a finite soil reservoir. Ground elevation is not altered. */
export interface Patch extends Region { id: string; pondingDepthM: number; infiltrationMmH: number; soilCapacityM: number }
export interface Config {
  solver: SolverId; rainMmH: number; rainMinutes: number; drainMinutes: number;
  infiltrationMmH: number; soilCapacityM: number; initialSoilM: number;
  initialDepthM: number; roughness: number; boundary: 'closed' | 'open';
  /** dw-1.0.0 only: head-difference regularization (m). */
  headEpsilonM: number;
  maxDt: number; outputSeconds: number;
  /** Depth (m) subtracted before counting target water. 0 = all ponded water in the target. */
  thresholdM: number;
  roi: Region; areaBudgetM2: number; maxPatches: number;
}
/** Water balance in m³: initial + rain = surface + soil + garden + outflow + waterBody + residual. */
export interface Accounting {
  initial: number; rain: number; surface: number; soil: number; garden: number;
  outflow: number; waterBody: number; residual: number; relativeResidual: number | null;
}
export interface Frame { t: number; h: Float32Array; balance: Accounting; roiDepth: number; roiVolume: number; roiExcess: number }
export interface Run {
  config: Config; gridId: string; patches: Patch[]; frames: Frame[];
  /** Planner objective: peak over time of sum(max(h - thresholdM, 0) * cellArea) in the target (m³). */
  objective: number; peakRoiDepth: number; peakRoiVolume: number; peakDomainVolume: number;
  balance: Accounting; wallMs: number; steps: number;
  /** Smallest stability-limited timestep (s); output and rain-end clipping excluded. */
  minStableDt: number; method: string;
}
export interface Proposal { selected_patch_ids: string[] }
export interface Plan { proposal: Proposal; run: Run; method: string; simulationCalls: number; wallMs: number }

export const defaultConfig = (): Config => ({
  solver: SOLVER_VERSION, rainMmH: 60, rainMinutes: 20, drainMinutes: 10,
  infiltrationMmH: 3, soilCapacityM: 0.04, initialSoilM: 0, initialDepthM: 0,
  roughness: 0.045, boundary: 'open', headEpsilonM: 0.01, maxDt: 2, outputSeconds: 30, thresholdM: 0,
  roi: { x: 47, y: 68, width: 10, height: 6, name: 'South loop target' },
  areaBudgetM2: 768, maxPatches: 3,
});

export const cells = (g: Grid, r: Region): number[] => {
  const out: number[] = [];
  for (let y = r.y; y < r.y + r.height; y++) for (let x = r.x; x < r.x + r.width; x++) out.push(y * g.nx + x);
  return out;
};
export const patchArea = (g: Grid, patches: Patch[]) => new Set(patches.flatMap(p => cells(g, p))).size * g.dx * g.dy;

const binary = (values: number[] | undefined, length: number) => !values || (values.length === length && values.every(v => v === 0 || v === 1));

export function validateGrid(g: Grid): void {
  if (!g || !Number.isInteger(g.nx) || !Number.isInteger(g.ny) || g.nx < 2 || g.ny < 2 || g.nx * g.ny > 65536 || !(g.dx > 0) || !(g.dy > 0) || !Number.isFinite(g.dx * g.dy)) throw Error('Invalid grid dimensions or cell size');
  if (g.z.length !== g.nx * g.ny || g.mask.length !== g.z.length || g.z.some(z => !Number.isFinite(z)) || !binary(g.mask, g.z.length) || !g.mask.some(Boolean)) throw Error('Invalid elevations or nodata mask');
  if (!binary(g.eligible, g.z.length)) throw Error('Invalid eligibility mask');
  if (!binary(g.sink, g.z.length) || g.sink?.some((s, i) => s && !g.mask[i])) throw Error('Invalid water-body mask');
}

export function validateRegion(g: Grid, r: Region): void {
  if (!r || ![r.x, r.y, r.width, r.height].every(Number.isInteger) || r.x < 0 || r.y < 0 || r.width < 1 || r.height < 1 || r.x + r.width > g.nx || r.y + r.height > g.ny) throw Error('Region is outside the grid');
  if (cells(g, r).some(i => !g.mask[i])) throw Error('Region includes nodata or building cells');
  if (cells(g, r).some(i => g.sink?.[i])) throw Error('Region includes water-body cells');
}

export function validateConfig(g: Grid, c: Config): void {
  validateGrid(g); validateRegion(g, c.roi);
  if (!SOLVERS.includes(c.solver)) throw Error('Unsupported solver version');
  for (const k of ['rainMmH', 'rainMinutes', 'drainMinutes', 'infiltrationMmH', 'soilCapacityM', 'initialSoilM', 'initialDepthM', 'thresholdM', 'areaBudgetM2'] as const) if (!Number.isFinite(c[k]) || c[k] < 0) throw Error(`Invalid ${k}`);
  if (c.rainMmH > 300 || c.rainMinutes > 180 || c.drainMinutes > 180 || c.rainMinutes + c.drainMinutes <= 0
    || !(c.roughness >= 0.015 && c.roughness <= 0.3) || !(c.headEpsilonM >= 1e-5 && c.headEpsilonM <= 0.02)
    || !(c.maxDt > 0 && c.maxDt <= 10) || !(c.outputSeconds >= 1 && c.outputSeconds <= 300)
    || !Number.isInteger(c.maxPatches) || c.maxPatches < 0 || c.maxPatches > 8 || c.initialSoilM > c.soilCapacityM
    || c.initialDepthM > 2 || c.soilCapacityM > 1 || c.thresholdM > 1 || !['closed', 'open'].includes(c.boundary)) throw Error('Scenario parameter outside supported range');
}

export function validatePatches(g: Grid, c: Config, patches: Patch[]): void {
  if (!Array.isArray(patches) || patches.length > c.maxPatches) throw Error('Maximum patch count exceeded');
  if (new Set(patches.map(p => p.id)).size !== patches.length) throw Error('Duplicate patch IDs');
  const target = new Set(cells(g, c.roi)), used = new Set<number>();
  for (const p of patches) {
    validateRegion(g, p);
    if (!p.id || p.id.length > 40) throw Error('Garden IDs must be 1-40 characters');
    if (p.x === 0 || p.y === 0 || p.x + p.width === g.nx || p.y + p.height === g.ny) throw Error('Patches must stay inside the boundary');
    const footprint = cells(g, p);
    if (footprint.some(i => g.eligible && !g.eligible[i])) throw Error('Patch includes an ineligible cell');
    if (footprint.some(i => target.has(i))) throw Error('Gardens must sit outside the inspected target');
    if (footprint.some(i => used.has(i))) throw Error('Gardens may not overlap');
    footprint.forEach(i => used.add(i));
    if (![p.pondingDepthM, p.infiltrationMmH, p.soilCapacityM].every(Number.isFinite) || p.pondingDepthM < 0 || p.pondingDepthM > 0.5 || p.infiltrationMmH < 0 || p.infiltrationMmH > 100 || p.soilCapacityM < 0 || p.soilCapacityM > 0.5) throw Error('Invalid rain-garden assumptions');
  }
  if (patchArea(g, patches) > c.areaBudgetM2 + 1e-9) throw Error('Treated area exceeds budget');
}
