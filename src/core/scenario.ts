import { SOLVERS, validateConfig, validatePatches, type Config, type Grid, type Patch, type Run } from './types';

/** Version 2: storage-based rain gardens (pondingDepthM), selectable solver, 384 m computational domain. */
export interface Scenario { version: 2; solver: string; gridId: string; terrainSignature: string; config: Config; patches: Patch[] }

// Non-security identifier for accidental dataset changes; SHA-256 source integrity lives in data/manifest.json.
export function terrainSignature(g: Grid): string {
  let h = 2166136261; const s = JSON.stringify([g.nx, g.ny, g.dx, g.dy, g.z, g.mask, g.eligible, g.sink ?? null]);
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(16);
}
export function scenario(g: Grid, c: Config, patches: Patch[]): Scenario {
  validateConfig(g, c); validatePatches(g, c, patches);
  return { version: 2, solver: c.solver, gridId: g.id, terrainSignature: terrainSignature(g), config: structuredClone(c), patches: structuredClone(patches) };
}
export function importScenario(text: string, g: Grid): Scenario {
  if (text.length > 100_000) throw Error('Scenario exceeds 100 kB limit');
  let s: Scenario;
  try { s = JSON.parse(text) as Scenario; } catch { throw Error('Scenario file is not valid JSON'); }
  if (!s || s.version !== 2) throw Error('Incompatible scenario version; this build reads version 2');
  if (!SOLVERS.includes(s.solver as never) || s.config?.solver !== s.solver) throw Error('Unknown or inconsistent solver version');
  if (s.gridId !== g.id || s.terrainSignature !== terrainSignature(g)) throw Error('Scenario was made for different terrain');
  validateConfig(g, s.config); validatePatches(g, s.config, s.patches); return s;
}
export function summary(run: Run) {
  return { solver: run.config.solver, method: run.method, observationSeconds: (run.config.rainMinutes + run.config.drainMinutes) * 60, objectiveM3: run.objective,
    thresholdM: run.config.thresholdM, peakRoiDepthM: run.peakRoiDepth, peakRoiVolumeM3: run.peakRoiVolume, peakDomainVolumeM3: run.peakDomainVolume,
    balanceM3: run.balance, wallMs: run.wallMs, steps: run.steps, minStableDt: Number.isFinite(run.minStableDt) ? run.minStableDt : null, gardens: run.patches.map(p => p.id) };
}
