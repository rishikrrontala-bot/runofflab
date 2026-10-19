/**
 * Core regression suite for RunoffLab: units, both surface-water solvers (li-1.0.0 local inertial and the
 * dw-1.0.0 diffusive-wave reference), storage rain gardens, water-body sinks, validation, the prepared real
 * site, the planner, scenario files and the D8 terrain features.
 *
 * Physics tests use small synthetic grids from tests/fixtures/grids.ts. Only two runs touch the 96 x 96 real
 * domain, both with li-1.0.0 and snapshots off (dw-1.0.0 takes about 100 times longer there, so it is never
 * run on it).
 * Frame depths are Float32Array; checks that need full precision use the float64 frame metrics
 * (roiDepth, roiVolume, roiExcess) and the float64 water balance instead.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  SOLVERS, SOLVER_VERSION, cells, defaultConfig, patchArea, validateConfig, validateGrid, validatePatches, validateRegion,
  type Config, type Frame, type Grid, type Patch, type Region, type Run,
} from '../src/core/types';
import { DIFFUSIVE, GRAVITY, INERTIAL, donorFactors, faceFlux, inertialDischarge, mmHToMS, runSync, simulate, type SimOptions } from '../src/core/solver';
import {
  ModelFormatError, acceptModelText, better, designKey, feasibleDesigns, features, greedy, heuristic, parseProposal, plannerPrompt,
  plannerUnsupported, randomDesign, repairInstruction, searchBest, seededRandom,
} from '../src/core/planner';
import { importScenario, scenario, summary, terrainSignature } from '../src/core/scenario';
import { GARDEN, SURFACE, buildings, candidates, preparedSite, targets } from '../src/core/site';
import { terrainFeatures } from '../src/core/terrainFeatures';
import * as F from './fixtures/grids';

// ---------------------------------------------------------------------------------------------- helpers

const sim = (g: Grid, overrides: Partial<Config> = {}, patches: Patch[] = [], options?: SimOptions) => runSync(g, F.baseConfig(g, overrides), patches, options);
const BALANCE_KEYS = ['initial', 'rain', 'surface', 'soil', 'garden', 'outflow', 'waterBody'] as const;

/** The documented identity initial + rain = surface + soil + garden + outflow + waterBody + residual, checked
 *  independently of the solver's own residual, with every term nonnegative. */
function expectConserved(run: Run, tolerance = 1e-9) {
  for (const frame of [...run.frames, { balance: run.balance } as Frame]) {
    const b = frame.balance;
    for (const k of BALANCE_KEYS) expect(b[k], k).toBeGreaterThanOrEqual(0);
    const recomputed = b.initial + b.rain - b.surface - b.soil - b.garden - b.outflow - b.waterBody;
    expect(Math.abs(recomputed - b.residual)).toBeLessThanOrEqual(1e-12 * Math.max(1, b.initial + b.rain));
    if (b.relativeResidual === null) expect(Math.abs(b.residual)).toBeLessThan(1e-12);
    else expect(Math.abs(b.relativeResidual)).toBeLessThan(tolerance);
  }
}
const allDepths = (run: Run) => run.frames.flatMap(f => Array.from(f.h));
const sumAt = (h: ArrayLike<number>, ids: number[]) => ids.reduce((s, i) => s + h[i], 0);
const withoutClock = (run: Run) => ({ ...run, wallMs: 0 });
const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
const rowsOf = (g: Grid, from: number, to: number) => cells(g, F.region(0, from, g.nx, to - from));
const markRows = (g: Grid, rows: number[]) => g.z.map((_, i) => rows.includes(Math.floor(i / g.nx)) ? 1 : 0);
/** A stand-in Run for planner logic tests: only objective and patches are read by better()/greedy()/searchBest(). */
const fakeRun = (patches: Patch[], objective: number) => ({ objective, patches }) as unknown as Run;

// ---------------------------------------------------------------------------------------------- units

describe('units and cell geometry', () => {
  it('converts mm/h to m/s exactly', () => {
    expect(mmHToMS(3600)).toBe(0.001);
    expect(mmHToMS(0)).toBe(0);
    expect(mmHToMS(60) * 3600).toBeCloseTo(0.06, 15);
  });

  it('indexes regions row-major and measures area with rectangular cells', () => {
    const g = F.flat(5, 4, { dx: 2, dy: 3 });
    expect(cells(g, F.region(1, 2, 2, 2))).toEqual([11, 12, 16, 17]);
    expect(patchArea(g, [F.garden('A', 1, 1, 2, 2), F.garden('B', 3, 1, 1, 2)])).toBe(6 * 6);
  });

  it.each(F.SOLVER_IDS)('%s: rain volume uses dx * dy, not dx squared', solver => {
    const g = F.flat(6, 5, { dx: 2, dy: 3 }), run = sim(g, { solver });
    expect(run.balance.rain).toBeCloseTo(30 * 6 * 0.001, 12);
    expect(run.balance.surface).toBeCloseTo(run.balance.rain, 12);
    expect(run.frames.at(-1)!.roiDepth).toBeCloseTo(0.001, 12);
  });
});

// ---------------------------------------------------------------------------------------------- both solvers

describe.each(F.SOLVER_IDS)('solver %s: shared invariants', solver => {
  it('a closed flat basin holds exactly the rain that fell, uniformly', () => {
    const g = F.flat(6, 5, { dx: 2, dy: 3 }), run = sim(g, { solver, rainMinutes: 3, outputSeconds: 30 });
    const depth = 60 / 3.6e6 * 180;
    expect(run.balance.rain).toBeCloseTo(depth * 30 * 6, 12);
    expect(run.balance.surface).toBeCloseTo(run.balance.rain, 12);
    for (const k of ['outflow', 'waterBody', 'garden', 'soil'] as const) expect(run.balance[k]).toBe(0);
    for (const f of run.frames) {
      expect(f.roiDepth).toBeCloseTo(f.roiVolume / (30 * 6), 13);   // max depth equals mean depth: no flow
      expect(Array.from(f.h).every(h => h === f.h[0])).toBe(true);
    }
    expectConserved(run, 1e-12);
  });

  it('with no rain and no initial water nothing moves', () => {
    const g = F.rough(8, 7), run = sim(g, { solver, rainMmH: 0, boundary: 'open', rainMinutes: 5, drainMinutes: 5 });
    expect(allDepths(run).every(h => h === 0)).toBe(true);
    for (const k of BALANCE_KEYS) expect(run.balance[k]).toBe(0);
    expect(run.balance.relativeResidual).toBeNull();
    expect(run.objective).toBe(0);
    expect(run.minStableDt).toBe(Infinity);
  });

  it('lake at rest on a flat bed: depths unchanged to 1e-12', () => {
    const g = F.flat(6, 5), h0 = 0.3, run = sim(g, { solver, rainMmH: 0, initialDepthM: h0, rainMinutes: 0, drainMinutes: 5, outputSeconds: 30 });
    const v0 = h0 * 30 * 16;
    for (const f of run.frames) {
      // Max depth never rises above h0 and total volume never changes, so no cell can fall below h0 either.
      expect(Math.abs(f.roiDepth - h0)).toBeLessThan(1e-12);
      expect(Math.abs(f.roiVolume - v0) / v0).toBeLessThan(1e-12);
      expect(Array.from(f.h).every(h => h === Math.fround(h0))).toBe(true);
    }
    expect(run.balance.outflow).toBe(0);
  });

  it('water on a slope drains downhill and is conserved in a closed domain', () => {
    const g = F.southPlane(6, 10, 0.02), run = sim(g, { solver, rainMmH: 0, initialDepthM: 0.01, rainMinutes: 0, drainMinutes: 10, outputSeconds: 60 });
    const h = run.frames.at(-1)!.h, north = rowsOf(g, 0, 5), south = rowsOf(g, 5, 10);
    expect(sumAt(h, south)).toBeGreaterThan(2 * sumAt(h, north));
    expect(sumAt(h, rowsOf(g, 9, 10))).toBeGreaterThan(6 * 0.01);
    expect(sumAt(h, rowsOf(g, 0, 1))).toBeLessThan(6 * 0.01);
    expect(rel(run.balance.surface, run.balance.initial)).toBeLessThan(1e-12);
    expect(run.balance.outflow).toBe(0);
  });

  it('a closed depression retains all rain; with open edges only rim water can leave', () => {
    const g = F.bowl(9, 0.3), closed = sim(g, { solver, rainMinutes: 15, drainMinutes: 10, outputSeconds: 60 });
    expect(closed.balance.outflow).toBe(0);
    expect(rel(closed.balance.surface, closed.balance.rain)).toBeLessThan(1e-12);
    const open = sim(g, { solver, boundary: 'open', rainMinutes: 15, drainMinutes: 10, outputSeconds: 60 });
    const rimRain = 32 * 16 * mmHToMS(60) * 900;   // the 32 edge cells of a 9 x 9 grid
    expect(open.balance.outflow).toBeGreaterThan(0);
    expect(open.balance.outflow).toBeLessThan(rimRain);
    expect(open.balance.surface / open.balance.rain).toBeGreaterThan(0.9);
    expectConserved(open);
  });

  it('depths stay finite and nonnegative under intense rain on rough terrain', () => {
    const g = F.rough(10, 10), run = sim(g, { solver, boundary: 'open', rainMmH: 300, rainMinutes: 10, drainMinutes: 5, outputSeconds: 60 });
    expect(allDepths(run).every(h => Number.isFinite(h) && h >= 0)).toBe(true);
    expect(run.peakDomainVolume).toBeGreaterThan(0);
    expectConserved(run);
  });

  it('closes the water balance with initial soil, initial water, gardens, sinks, masks and open edges', () => {
    const g = F.rough(12, 10);
    g.mask[5 * 12 + 3] = g.mask[5 * 12 + 4] = 0;
    g.sink = g.z.map((_, i) => i % 12 === 10 && Math.floor(i / 12) > 0 && Math.floor(i / 12) < 9 ? 1 : 0);
    const patches = [F.garden('A', 2, 1), F.garden('B', 6, 6, 3, 2, { pondingDepthM: 0.05, soilCapacityM: 0.02 })];
    const run = sim(g, { solver, boundary: 'open', roi: F.region(6, 2, 2, 2), initialDepthM: 0.005, initialSoilM: 0.01, soilCapacityM: 0.04,
      infiltrationMmH: 10, rainMmH: 90, rainMinutes: 10, drainMinutes: 10, outputSeconds: 60 }, patches);
    const valid = g.mask.filter((m, i) => m && !g.sink![i]).length;
    expect(run.balance.initial).toBeCloseTo((0.005 + 0.01) * valid * 16, 12);
    for (const k of BALANCE_KEYS) expect(run.balance[k], k).toBeGreaterThan(0);
    expectConserved(run, 1e-9);
  });

  it('counts water leaving open edges as outflow, and none through closed or masked edges', () => {
    const g = F.flat(6, 6), open = sim(g, { solver, boundary: 'open', rainMinutes: 5, drainMinutes: 5, outputSeconds: 60 });
    expect(open.balance.outflow).toBeGreaterThan(0);
    expect(rel(open.balance.surface + open.balance.outflow, open.balance.rain)).toBeLessThan(1e-12);
    expect(sim(g, { solver, rainMinutes: 5 }).balance.outflow).toBe(0);
    const ringed = F.rough(8, 8);
    ringed.mask = ringed.mask.map((_, i) => { const x = i % 8, y = Math.floor(i / 8); return x === 0 || y === 0 || x === 7 || y === 7 ? 0 : 1; });
    const r = sim(ringed, { solver, boundary: 'open', roi: F.region(2, 2, 2, 2), rainMinutes: 10, outputSeconds: 60 });
    expect(r.balance.outflow).toBe(0);
    expect(rel(r.balance.surface, r.balance.rain)).toBeLessThan(1e-12);
  });

  it('masked (building or nodata) cells stay dry, get no rain, and have closed faces', () => {
    // A masked wall across row 4 separates a steep north slope from a flat, initially wet south strip.
    const g = F.southPlane(6, 9, 0.05);
    for (let x = 0; x < 6; x++) g.mask[4 * 6 + x] = 0;
    for (let i = 5 * 6; i < 54; i++) g.z[i] = 0;
    const south = F.region(0, 5, 6, 4, 'South of the wall');
    const run = sim(g, { solver, rainMmH: 0, initialDepthM: 0.02, rainMinutes: 0, drainMinutes: 10, roi: south, outputSeconds: 60 });
    for (const f of run.frames) {
      expect(Math.abs(f.roiVolume - 0.02 * 24 * 16)).toBeLessThan(1e-13);
      expect(Math.abs(f.roiDepth - 0.02)).toBeLessThan(1e-15);
      expect(Array.from(f.h.slice(24, 30))).toEqual([0, 0, 0, 0, 0, 0]);
    }
    expect(run.balance.initial).toBeCloseTo(0.02 * 48 * 16, 12);
    expect(Math.min(...Array.from(run.frames.at(-1)!.h.slice(18, 24)))).toBeGreaterThan(0.05);   // ponded against the wall
    const block = F.flat(6, 6); block.mask[14] = block.mask[15] = 0;
    const rained = sim(block, { solver, roi: F.region(0, 0, 2, 2) });
    expect(rained.balance.rain).toBeCloseTo(34 * 16 * 0.001, 12);
    expect(rained.frames.every(f => f.h[14] === 0 && f.h[15] === 0)).toBe(true);
  });

  it('is deterministic: generator frames equal runSync frames, and repeated runs are identical', () => {
    const g = F.rough(8, 8), c = F.baseConfig(g, { solver, boundary: 'open', rainMinutes: 3, drainMinutes: 2, outputSeconds: 20 }), p = [F.garden('A', 2, 2)];
    c.roi = F.region(5, 5, 2, 2);
    const gen = simulate(g, c, p), yielded: Frame[] = [];
    let step = gen.next();
    while (!step.done) { yielded.push(step.value); step = gen.next(); }
    const ref = runSync(g, c, p);
    expect(yielded).toEqual(ref.frames);
    expect(withoutClock(step.value)).toEqual(withoutClock(ref));
    expect(withoutClock(runSync(g, c, p))).toEqual(withoutClock(ref));
  });

  it('objective is the peak target volume above thresholdM, tracked every step', () => {
    const g = F.bowl(9, 0.3), o = { solver, roi: F.region(3, 3, 3, 3), rainMinutes: 5, drainMinutes: 5, outputSeconds: 60 };
    const all = sim(g, o);
    expect(all.objective).toBeGreaterThan(0);
    expect(all.objective).toBe(all.peakRoiVolume);   // thresholdM 0: every ponded cubic metre counts
    for (const f of all.frames) {
      expect(f.roiExcess).toBe(f.roiVolume);
      expect(f.roiExcess).toBeLessThanOrEqual(all.objective);
      expect(f.roiDepth).toBeLessThanOrEqual(all.peakRoiDepth);
      expect(f.roiVolume).toBeLessThanOrEqual(all.peakDomainVolume);
    }
    const half = sim(g, { ...o, thresholdM: all.peakRoiDepth / 2 });
    expect(half.objective).toBeGreaterThan(0);
    expect(half.objective).toBeLessThan(all.objective);
    expect(half.frames.every(f => f.roiExcess <= f.roiVolume)).toBe(true);
    expect(sim(g, { ...o, thresholdM: all.peakRoiDepth }).objective).toBe(0);
  });

  it('saves a frame at t = 0, at every output interval and at the end', () => {
    const g = F.flat(4, 4), run = sim(g, { solver, rainMinutes: 2, drainMinutes: 1.5, outputSeconds: 40 });
    expect(run.frames.map(f => f.t)).toEqual([0, 40, 80, 120, 160, 200, 210].map(t => expect.closeTo(t, 9)));
    expect(sim(g, { solver, rainMinutes: 2 }, [], { snapshots: false }).frames).toEqual([]);
  });

  it('halving maxDt leaves the flat-basin rain solution unchanged', () => {
    const g = F.flat(6, 5, { dx: 2, dy: 3 }), a = sim(g, { solver, rainMinutes: 3 }), b = sim(g, { solver, rainMinutes: 3, maxDt: 1 });
    expect(rel(b.balance.surface, a.balance.surface)).toBeLessThan(1e-12);
    expect(Math.abs(b.frames.at(-1)!.roiDepth - a.frames.at(-1)!.roiDepth)).toBeLessThan(1e-15);
    expect(b.steps).toBeGreaterThan(a.steps);
  });

  it('rejects invalid input before taking a step', () => {
    const g = F.flat(4, 4), gen = simulate(g, { ...F.baseConfig(g), solver } as Config, [F.garden('A', 0, 0)]);
    expect(() => gen.next()).toThrow('Patches must stay inside the boundary');
  });

  it('returns copies of its inputs and leaves the grid untouched', () => {
    const g = F.rough(8, 8), before = JSON.stringify(g), c = F.baseConfig(g, { solver, roi: F.region(5, 5, 2, 2) }), p = [F.garden('A', 2, 2)];
    const run = runSync(g, c, p);
    expect(JSON.stringify(g)).toBe(before);
    c.rainMmH = 1; p[0].pondingDepthM = 0;
    expect(run.config.rainMmH).toBe(60);
    expect(run.patches[0].pondingDepthM).toBe(0.15);
    expect(run.gridId).toBe(g.id);
    expect(run.method).toBe('manual');
    expect(runSync(g, run.config, run.patches, { method: 'Greedy' }).method).toBe('Greedy');
  });
});

describe('stability-limited timestep reporting', () => {
  it('li-1.0.0 reports the local-inertial CFL limit alpha * dx / sqrt(g * h)', () => {
    const g = F.flat(6, 5, { dx: 2, dy: 3 }), run = sim(g, { solver: 'li-1.0.0' });
    const bound = INERTIAL.alpha * Math.min(g.dx, g.dy) / Math.sqrt(GRAVITY * 0.001);   // at the final 1 mm depth
    expect(run.minStableDt).toBeGreaterThanOrEqual(bound);
    expect(run.minStableDt).toBeLessThan(1.03 * bound);
  });

  it('dw-1.0.0 reports stability * cell area / total face conductance', () => {
    const g = F.flat(6, 5, { dx: 2, dy: 3 }), c = F.baseConfig(g, { solver: 'dw-1.0.0' }), run = runSync(g, c);
    // Interior cell on a flat surface: dEta = 0, so every face uses the headEpsilonM regularization.
    const hf = Math.pow(0.001, 5 / 3) / c.roughness;
    const total = 2 * g.dy * hf / Math.sqrt(g.dx * c.headEpsilonM) + 2 * g.dx * hf / Math.sqrt(g.dy * c.headEpsilonM);
    const bound = DIFFUSIVE.stability * g.dx * g.dy / total;
    expect(run.minStableDt).toBeGreaterThanOrEqual(bound);
    expect(run.minStableDt).toBeLessThan(1.07 * bound);
  });
});

// ---------------------------------------------------------------------------------------------- face fluxes

describe('li-1.0.0 face discharge (Bates et al. 2010, implicit friction)', () => {
  const n = 0.045, d = 4, dt = 1;
  it('flows from the higher to the lower water surface and is antisymmetric', () => {
    const cases = [[1.0, 0.9, 0.1], [0.123, 0.456, 0.02], [5, 5.0001, 0.3], [2.5, -1, 1.2]];
    for (const [a, b, hf] of cases) {
      const q = inertialDischarge(0, a, b, hf, d, dt, n);
      expect(Math.sign(q)).toBe(Math.sign(a - b));
      expect(inertialDischarge(0, b, a, hf, d, dt, n)).toBe(-q);
    }
  });

  it('is exactly zero at equal heads from rest, for any flow depth (well balanced)', () => {
    for (const hf of [1e-4, 0.01, 0.5, 2]) expect(Object.is(inertialDischarge(0, 7.25, 7.25, hf, d, dt, n), 0)).toBe(true);
    // Lake at rest over an uneven bed: equal surfaces over beds 0.0 and 0.2 give zero flux in both schemes.
    expect(inertialDischarge(0, 0.5, 0.5, 0.5 - 0.2, d, dt, n)).toBe(0);
    expect(faceFlux(0, 0.2, 0.5, 0.3, d, d, n)).toBe(0);
  });

  it('decays existing momentum by friction without reversing it', () => {
    const q = inertialDischarge(0.05, 1, 1, 0.1, d, dt, n);
    expect(q).toBeGreaterThan(0);
    expect(q).toBeLessThan(0.05);
    expect(inertialDischarge(-0.05, 1, 1, 0.1, d, dt, n)).toBe(-q);
  });

  it('solves q(1 + g dt n^2 |q| / hf^(7/3)) = q_old - g hf dt dEta / d exactly', () => {
    for (const [qOld, a, b, hf, step] of [[0, 1.01, 1, 0.05, 1], [0.02, 1, 1.03, 0.2, 0.5], [-0.1, 2, 1.9, 0.4, 2]]) {
      const q = inertialDischarge(qOld, a, b, hf, d, step, n);
      const qStar = qOld - GRAVITY * hf * step * (b - a) / d, k = GRAVITY * step * n * n / Math.pow(hf, 7 / 3);
      expect(Math.abs(q * (1 + k * Math.abs(q)) - qStar)).toBeLessThan(1e-14 * Math.max(1, Math.abs(qStar)));
    }
    expect(inertialDischarge(0.01, 1.01, 1, 0.05, d, dt, 0)).toBeCloseTo(0.01 + GRAVITY * 0.05 * dt * 0.01 / d, 16);   // frictionless limit
  });

  it('tends to Manning normal flow hf^(5/3) S^(1/2) / n as dt grows', () => {
    const hf = 0.05, S = 0.01, manning = Math.pow(hf, 5 / 3) * Math.sqrt(S) / n;
    const ratios = [1, 10, 100, 1e4].map(step => inertialDischarge(0, S * d, 0, hf, d, step, n) / manning);
    for (let k = 1; k < ratios.length; k++) expect(ratios[k]).toBeGreaterThan(ratios[k - 1]);
    expect(ratios.every(r => r <= 1)).toBe(true);
    expect(1 - ratios.at(-1)!).toBeLessThan(1e-3);
  });
});

describe('dw-1.0.0 face flux and donor limiting', () => {
  it('is antisymmetric, zero at equal heads, and zero across a dry high face', () => {
    const q = faceFlux(1, 0, 0.03, 0.01, 3, 2, 0.04);
    expect(q).toBeGreaterThan(0);
    expect(faceFlux(0, 1, 0.01, 0.03, 3, 2, 0.04)).toBe(-q);
    expect(faceFlux(0, 0, 0.2, 0.2, 3, 2, 0.04)).toBe(0);
    expect(faceFlux(0, 1, 0.1, 0, 2, 2, 0.04)).toBe(0);
  });

  it('equals Manning flow when the head difference exceeds the regularization', () => {
    const n = 0.045, w = 1, dist = 4, h = 0.05;
    expect(rel(faceFlux(0.04, 0, h, h, w, dist, n), Math.pow(h, 5 / 3) * Math.sqrt(0.01) / n)).toBeLessThan(1e-14);
  });

  it('scales each donor to the water it holds and leaves others alone', () => {
    const f = donorFactors(new Float64Array([0.01, 0, 0.5]), new Float64Array([100, 0, 1]), 2, 6);
    expect(100 * 2 * f[0]).toBeCloseTo(0.01 * 6, 14);
    expect(f[1]).toBe(1);
    expect(f[2]).toBe(1);
  });
});

// ---------------------------------------------------------------------------------------------- scientific checks

describe.each(F.SOLVER_IDS)('solver %s: analytical and steady-state checks', solver => {
  it('a closed bowl settles to a lake at rest: flat water surface over an uneven bed', () => {
    const g = F.bowl(9, 0.5), run = sim(g, { solver, rainMmH: 120, rainMinutes: 10, drainMinutes: 60, outputSeconds: 300 });
    const h = run.frames.at(-1)!.h, wet = [...h.keys()].filter(i => h[i] > 0.01), eta = wet.map(i => g.z[i] + h[i]);
    expect(wet.length).toBeGreaterThan(9);
    expect(Math.max(...eta) - Math.min(...eta)).toBeLessThan(1e-4);
    expect(rel(run.balance.surface, run.balance.rain)).toBeLessThan(1e-12);
  });

  it('steady rain on an open sloping plane: outflow rate converges to the rain input', () => {
    const g = F.southPlane(12, 12, 0.01), run = sim(g, { solver, boundary: 'open', rainMinutes: 60, outputSeconds: 60 });
    const f = run.frames, rate = (f.at(-1)!.balance.outflow - f.at(-2)!.balance.outflow) / 60, input = mmHToMS(60) * 144 * 16;
    expect(rel(rate, input)).toBeLessThan(1e-3);
    expect((f[5].balance.outflow - f[4].balance.outflow) / 60).toBeLessThan(0.9 * input);   // still rising early on
  });

  it('steady 1D plane draining into a water body matches Manning and the kinematic-wave depth', () => {
    // Two identical closed columns, 1 % slope, bottom row a water-body sink. At steady state the unit discharge
    // through the face below row j is q = i (j + 1) dy, and the upwind depth must satisfy Manning exactly with
    // the simulated surface slope, and approximately with the bed slope (kinematic wave, h = (n q / S^1/2)^(3/5)).
    const ny = 24, S = 0.01, g = F.southPlane(2, ny, S);
    g.sink = markRows(g, [ny - 1]);
    const c = F.baseConfig(g, { solver, rainMinutes: 60, outputSeconds: 60, roi: F.region(0, 0, 2, 4) }), run = runSync(g, c);
    const f = run.frames, i = mmHToMS(c.rainMmH), h = f.at(-1)!.h;
    expect(rel((f.at(-1)!.balance.waterBody - f.at(-2)!.balance.waterBody) / 60, i * 2 * ny * 16)).toBeLessThan(1e-3);
    for (let j = 0; j < ny - 2; j++) {
      const a = 2 * j, b = 2 * (j + 1), q = i * (j + 1) * g.dy, Sf = (g.z[a] + h[a] - g.z[b] - h[b]) / g.dy;
      expect(h[a]).toBe(h[a + 1]);
      expect(rel(h[a], Math.pow(c.roughness * q / Math.sqrt(Sf), 0.6))).toBeLessThan(1e-5);
      expect(rel(h[a], Math.pow(c.roughness * q / Math.sqrt(S), 0.6))).toBeLessThan(0.015);
    }
  });

  // Hunter et al. (2005) / Bates et al. (2010) test the advance of a wetting front over a flat frictional plane
  // driven by a prescribed, time-varying boundary depth h(0, t). The core API has no stage or inflow boundary
  // condition and only a uniform initial depth, so that analytical case cannot be posed. The steady kinematic
  // test above checks the same momentum and friction balance against an analytical depth instead.
  it.todo('1D moving wetting front of Hunter et al. 2005 (needs a prescribed-depth boundary the API does not expose)');
});

describe('solver cross-checks and convergence', () => {
  it('li-1.0.0 and dw-1.0.0 agree within 5 % on a closed bowl storm objective', () => {
    const g = F.bowl(11, 0.3), o = { roi: F.region(4, 4, 3, 3), rainMinutes: 15, drainMinutes: 10, infiltrationMmH: 3, outputSeconds: 60 };
    const li = sim(g, { ...o, solver: 'li-1.0.0' }), dw = sim(g, { ...o, solver: 'dw-1.0.0' });
    expect(li.objective).toBeGreaterThan(1);
    expect(rel(li.objective, dw.objective)).toBeLessThan(0.05);
    expect(rel(li.peakRoiDepth, dw.peakRoiDepth)).toBeLessThan(0.05);
  });

  it('li-1.0.0 and dw-1.0.0 agree within 3 % on moderate rough terrain with open edges', () => {
    const g = F.rough(12, 12), o = { boundary: 'open' as const, roi: F.region(4, 4, 3, 3), rainMinutes: 15, drainMinutes: 10, infiltrationMmH: 3, outputSeconds: 60 };
    const li = sim(g, { ...o, solver: 'li-1.0.0' }), dw = sim(g, { ...o, solver: 'dw-1.0.0' });
    expect(rel(li.objective, dw.objective)).toBeLessThan(0.03);
    expect(rel(li.balance.outflow, dw.balance.outflow)).toBeLessThan(0.05);
  });

  it('li-1.0.0: halving maxDt changes a bowl storm objective by under 0.5 %', () => {
    // On this bowl li-1.0.0 is limited by maxDt (its CFL step is ~3 s); dw-1.0.0 is stability-limited far
    // below 1 s, so halving maxDt would not change its steps at all.
    const g = F.bowl(11, 0.3), o = { solver: 'li-1.0.0' as const, roi: F.region(4, 4, 3, 3), rainMinutes: 15, drainMinutes: 10, infiltrationMmH: 3, outputSeconds: 60 };
    const a = sim(g, { ...o, maxDt: 2 }), b = sim(g, { ...o, maxDt: 1 });
    expect(a.minStableDt).toBeGreaterThan(2);
    expect(b.steps).toBeGreaterThan(1.8 * a.steps);
    expect(rel(b.objective, a.objective)).toBeLessThan(0.005);
  });

  it('li-1.0.0: halving the Courant coefficient alpha halves the step and barely moves the objective', () => {
    // INERTIAL is an exported mutable object; restore it whatever happens.
    const g = F.rough(12, 12), o = { solver: 'li-1.0.0' as const, boundary: 'open' as const, roi: F.region(4, 4, 3, 3), rainMinutes: 15, drainMinutes: 10, infiltrationMmH: 3, outputSeconds: 60, maxDt: 10 };
    const full = sim(g, o), saved = INERTIAL.alpha;
    let half: Run;
    try { INERTIAL.alpha = saved / 2; half = sim(g, o); } finally { INERTIAL.alpha = saved; }
    expect(INERTIAL.alpha).toBe(0.7);
    expect(half.minStableDt / full.minStableDt).toBeCloseTo(0.5, 2);
    expect(half.steps).toBeGreaterThan(1.8 * full.steps);
    expect(rel(half.objective, full.objective)).toBeLessThan(2e-3);
  });
});

// ---------------------------------------------------------------------------------------------- gardens

describe.each(F.SOLVER_IDS)('solver %s: storage rain gardens', solver => {
  const corner = F.region(0, 0, 2, 2, 'Corner target');

  it('captures surface water into garden storage, counted in balance.garden', () => {
    const g = F.flat(8, 8), p = F.garden('A', 3, 3, 2, 2, { infiltrationMmH: 0, soilCapacityM: 0 });
    const run = sim(g, { solver, roi: corner, rainMinutes: 10, drainMinutes: 10, outputSeconds: 60 }, [p]);
    const ids = cells(g, p), ownRain = 4 * 16 * mmHToMS(60) * 600;
    expect(run.balance.garden).toBeGreaterThan(ownRain);   // its own rain plus run-on from the lawn around it
    expect(run.balance.soil).toBe(0);
    expect(rel(run.balance.surface + run.balance.garden, run.balance.rain)).toBeLessThan(1e-12);
    expect(run.frames.every(f => ids.every(i => f.h[i] === 0))).toBe(true);   // basin not yet full
    expectConserved(run, 1e-12);
  });

  it('never holds more than pondingDepthM over its footprint; overflow stays on the surface', () => {
    const g = F.bowl(9, 0.3), p = F.garden('A', 3, 3, 3, 3, { pondingDepthM: 0.05, infiltrationMmH: 0, soilCapacityM: 0 }), capacity = 0.05 * 9 * 16;
    const run = sim(g, { solver, roi: corner, rainMmH: 200, rainMinutes: 10, drainMinutes: 5, outputSeconds: 60 }, [p]);
    expect(run.frames.every(f => f.balance.garden <= capacity * (1 + 1e-12))).toBe(true);
    expect(rel(run.balance.garden, capacity)).toBeLessThan(1e-12);
    expect(cells(g, p).every(i => run.frames.at(-1)!.h[i] > 0.01)).toBe(true);   // full basin, water ponds above it
    expectConserved(run);
  });

  it('infiltrates stored water into the garden soil at the garden rate', () => {
    const g = F.bowl(9, 0.3), p = F.garden('A', 3, 3, 3, 3, { pondingDepthM: 0.1, infiltrationMmH: 50, soilCapacityM: 0.5 });
    const run = sim(g, { solver, roi: corner, infiltrationMmH: 0, rainMinutes: 10, drainMinutes: 60, outputSeconds: 300 }, [p]);
    // Ground infiltration is off, so all soil water is garden soil. While it rains (60 mm/h, above the 50 mm/h
    // garden rate) no basin cell is ever empty, so soil fills at exactly the garden rate over 9 cells; after the
    // storm it can only fall behind that rate as the shallower basin cells run dry.
    for (const f of run.frames) {
      const atRate = mmHToMS(50) * f.t * 9 * 16;
      if (f.t <= 600) expect(f.balance.soil).toBeCloseTo(atRate, 9);
      else expect(f.balance.soil).toBeLessThanOrEqual(atRate * (1 + 1e-12));
    }
    expect(run.balance.soil).toBeGreaterThan(0.5 * mmHToMS(50) * 4200 * 9 * 16);
    const stored = run.frames.map(f => f.balance.garden), peak = Math.max(...stored);
    expect(stored.at(-1)!).toBeLessThan(0.7 * peak);   // drains into the soil after the storm
    expectConserved(run);
    const small = sim(g, { solver, roi: corner, infiltrationMmH: 0, rainMinutes: 10, drainMinutes: 20, outputSeconds: 300 }, [{ ...p, soilCapacityM: 0.01 }]);
    expect(small.balance.soil).toBeCloseTo(0.01 * 9 * 16, 12);   // finite garden soil fills exactly once
  });

  it('a garden upstream of a closed target lowers the target peak volume; a larger one lowers it more', () => {
    const g = F.southPlane(6, 12, 0.01), roi = F.region(1, 10, 4, 2, 'Bottom target');
    const o = { solver, roi, rainMinutes: 15, drainMinutes: 10, outputSeconds: 60, infiltrationMmH: 3 };
    const none = sim(g, o), small = sim(g, o, [F.garden('A', 2, 4)]), large = sim(g, o, [F.garden('A', 1, 6, 4, 3)]);
    expect(small.objective).toBeLessThan(0.9 * none.objective);
    expect(large.objective).toBeLessThan(0.6 * none.objective);
    expect(large.objective).toBeLessThan(small.objective);
  });

  it('does not alter ground elevation', () => {
    const g = F.southPlane(6, 12, 0.01), z = [...g.z];
    sim(g, { solver, roi: F.region(1, 10, 4, 2), rainMinutes: 5 }, [F.garden('A', 1, 6, 4, 3)]);
    expect(g.z).toEqual(z);
  });
});

// ---------------------------------------------------------------------------------------------- sinks

describe.each(F.SOLVER_IDS)('solver %s: water-body sinks', solver => {
  it('removes water that reaches the lake into balance.waterBody; lake cells stay dry', () => {
    const g = F.southPlane(6, 10, 0.01);
    g.sink = markRows(g, [9]);
    const run = sim(g, { solver, roi: F.region(1, 1, 2, 2), initialDepthM: 0.01, rainMinutes: 10, drainMinutes: 30, outputSeconds: 60 });
    const lake = rowsOf(g, 9, 10), lakeRain = 6 * 16 * mmHToMS(60) * 600;
    expect(run.frames.every(f => lake.every(i => f.h[i] === 0))).toBe(true);
    expect(run.balance.initial).toBeCloseTo(0.01 * 54 * 16, 12);   // lake cells start empty
    expect(run.balance.waterBody).toBeGreaterThan(lakeRain + 0.5 * run.balance.initial);
    expect(run.balance.outflow).toBe(0);
    expectConserved(run, 1e-12);
  });

  it('lake cells do not infiltrate', () => {
    const g = F.flat(6, 6);
    g.sink = g.z.map((_, i) => i % 6 === 5 ? 1 : 0);
    const run = sim(g, { solver, roi: F.region(0, 0, 2, 2), rainMinutes: 10, infiltrationMmH: 10 });
    expect(run.balance.soil).toBeCloseTo(mmHToMS(10) * 600 * 30 * 16, 12);   // 30 land cells, not 36
    expectConserved(run);
  });
});

describe('water-body validation', () => {
  it('rejects sink cells on masked cells and regions or gardens that include sink cells', () => {
    const g = F.flat(8, 8);
    g.sink = g.z.map((_, i) => i === 27 ? 1 : 0);
    validateGrid(g);
    expect(() => validateGrid({ ...g, mask: g.mask.map((m, i) => i === 27 ? 0 : m) })).toThrow('Invalid water-body mask');
    expect(() => validateRegion(g, F.region(2, 2, 2, 2))).toThrow('Region includes water-body cells');
    expect(() => validateConfig(g, F.baseConfig(g))).toThrow('Region includes water-body cells');
    const c = F.baseConfig(g, { roi: F.region(5, 5, 2, 2) });
    expect(() => validatePatches(g, c, [F.garden('A', 3, 3)])).toThrow('Region includes water-body cells');
  });
});

// ---------------------------------------------------------------------------------------------- validation

describe('validation', () => {
  it('rejects malformed grids, masks and spacing', () => {
    const good = F.flat(4, 4), z = good.z, ones = good.mask;
    validateGrid(good);
    validateGrid({ ...good, eligible: undefined });
    const dims: Partial<Grid>[] = [{ nx: 4.5 }, { nx: 1 }, { ny: 0 }, { dx: 0 }, { dy: -1 }, { dx: NaN }, { dx: Infinity },
      { nx: 300, ny: 300, z: new Array(90000).fill(0), mask: new Array(90000).fill(1) }];
    for (const v of dims) expect(() => validateGrid({ ...good, ...v }), JSON.stringify(v)).toThrow('Invalid grid dimensions or cell size');
    const values: Partial<Grid>[] = [{ z: z.slice(1) }, { mask: ones.slice(1) }, { z: z.map((v, i) => i === 3 ? NaN : v) },
      { z: z.map((v, i) => i === 3 ? Infinity : v) }, { mask: ones.map((v, i) => i === 3 ? 2 : v) }, { mask: ones.map(() => 0) }];
    for (const v of values) expect(() => validateGrid({ ...good, ...v })).toThrow('Invalid elevations or nodata mask');
    for (const eligible of [ones.slice(1), ones.map(() => 0.5)]) expect(() => validateGrid({ ...good, eligible })).toThrow('Invalid eligibility mask');
    for (const sink of [ones.slice(1), ones.map((v, i) => i === 0 ? 2 : 0)]) expect(() => validateGrid({ ...good, sink })).toThrow('Invalid water-body mask');
  });

  it('rejects regions outside the grid or on masked cells', () => {
    const g = F.flat(6, 6);
    g.mask[14] = 0;
    for (const r of [F.region(-1, 0, 2, 2), F.region(5, 0, 2, 2), F.region(0, 0, 0, 2), F.region(0.5, 0, 2, 2), F.region(0, 5, 1, 2)])
      expect(() => validateRegion(g, r)).toThrow('Region is outside the grid');
    expect(() => validateRegion(g, F.region(1, 1, 2, 2))).toThrow('Region includes nodata or building cells');
    validateRegion(g, F.region(3, 3, 3, 3));
  });

  it('rejects invalid configurations, including an unknown solver', () => {
    const g = F.flat(6, 6), base = F.baseConfig(g);
    validateConfig(g, base);
    for (const solver of SOLVERS) validateConfig(g, { ...base, solver });
    validateConfig(g, { ...base, maxPatches: 0 });
    validateConfig(g, { ...base, maxPatches: 8, boundary: 'open', thresholdM: 1, maxDt: 10 });
    expect(() => validateConfig(g, { ...base, solver: 'li-2.0.0' } as unknown as Config)).toThrow('Unsupported solver version');
    expect(() => validateConfig(g, { ...base, solver: undefined } as unknown as Config)).toThrow('Unsupported solver version');
    for (const [k, v] of [['rainMmH', NaN], ['rainMinutes', -1], ['drainMinutes', -0.1], ['infiltrationMmH', Infinity], ['soilCapacityM', -1],
      ['initialSoilM', -0.01], ['initialDepthM', -1], ['thresholdM', -0.1], ['areaBudgetM2', -1]] as const)
      expect(() => validateConfig(g, { ...base, [k]: v }), k).toThrow(`Invalid ${k}`);
    const ranges: Partial<Config>[] = [{ rainMmH: 301 }, { rainMinutes: 181 }, { drainMinutes: 181 }, { rainMinutes: 0, drainMinutes: 0 },
      { roughness: 0.01 }, { roughness: 0.31 }, { headEpsilonM: 0 }, { headEpsilonM: 0.03 }, { maxDt: 0 }, { maxDt: 10.5 },
      { outputSeconds: 0.5 }, { outputSeconds: 301 }, { maxPatches: 1.5 }, { maxPatches: -1 }, { maxPatches: 9 },
      { initialSoilM: 0.05 }, { initialDepthM: 2.1 }, { soilCapacityM: 1.1 }, { thresholdM: 1.1 }, { boundary: 'inflow' as 'open' }];
    for (const v of ranges) expect(() => validateConfig(g, { ...base, ...v }), JSON.stringify(v)).toThrow('Scenario parameter outside supported range');
    expect(() => validateConfig(g, { ...base, roi: F.region(5, 5, 2, 2) })).toThrow('Region is outside the grid');
  });

  describe('garden designs', () => {
    const g = F.flat(12, 12), c = F.baseConfig(g, { roi: F.region(5, 5, 2, 2, 'Middle target'), maxPatches: 3, areaBudgetM2: 3 * 64 });
    const ok = F.garden('A', 1, 1);

    it('accepts legal designs, including exactly the area budget and gardens that share an edge', () => {
      validatePatches(g, c, []);
      validatePatches(g, c, [ok, F.garden('B', 3, 1), F.garden('C', 8, 8)]);
      validatePatches(g, { ...c, areaBudgetM2: 64 }, [ok]);
      validatePatches({ ...g, eligible: undefined }, c, [ok]);
    });

    it('documents the target rule: overlap is rejected, sharing an edge with the target is allowed', () => {
      expect(() => validatePatches(g, c, [F.garden('T', 4, 4)])).toThrow('Gardens must sit outside the inspected target');
      expect(() => validatePatches(g, c, [F.garden('T', 6, 6)])).toThrow('Gardens must sit outside the inspected target');
      validatePatches(g, c, [F.garden('T', 3, 5)]);   // its east edge touches the target's west edge
    });

    it('rejects edge, outside, ineligible, masked, overlapping, duplicate and malformed gardens', () => {
      for (const p of [F.garden('E', 0, 3), F.garden('E', 3, 0), F.garden('E', 10, 3), F.garden('E', 3, 10)])
        expect(() => validatePatches(g, c, [p]), JSON.stringify(p)).toThrow('Patches must stay inside the boundary');
      for (const p of [{ ...ok, id: '' }, { ...ok, id: 'x'.repeat(41) }])
        expect(() => validatePatches(g, c, [p]), JSON.stringify(p)).toThrow('Garden IDs must be 1-40 characters');
      expect(() => validatePatches(g, c, [F.garden('O', 11, 3)])).toThrow('Region is outside the grid');
      const ineligible = structuredClone(g); ineligible.eligible![2 * 12 + 2] = 0;
      expect(() => validatePatches(ineligible, c, [ok])).toThrow('Patch includes an ineligible cell');
      const masked = structuredClone(g); masked.mask[2 * 12 + 2] = 0;
      expect(() => validatePatches(masked, c, [ok])).toThrow('Region includes nodata or building cells');
      expect(() => validatePatches(g, c, [ok, F.garden('B', 2, 2)])).toThrow('Gardens may not overlap');
      expect(() => validatePatches(g, c, [ok, { ...ok, x: 8 }])).toThrow('Duplicate patch IDs');
      expect(() => validatePatches(g, c, 'A' as unknown as Patch[])).toThrow();
    });

    it('enforces the garden count and the area budget', () => {
      const four = [ok, F.garden('B', 3, 1), F.garden('C', 8, 8), F.garden('D', 8, 1)];
      expect(() => validatePatches(g, c, four)).toThrow('Maximum patch count exceeded');
      expect(() => validatePatches(g, { ...c, maxPatches: 0 }, [ok])).toThrow('Maximum patch count exceeded');
      expect(() => validatePatches(g, { ...c, areaBudgetM2: 63 }, [ok])).toThrow('Treated area exceeds budget');
      expect(() => validatePatches(g, c, [F.garden('Big', 1, 1, 4, 4), F.garden('B', 8, 8)])).toThrow('Treated area exceeds budget');
    });

    it('rejects invalid garden assumptions', () => {
      const bad: Partial<Patch>[] = [{ pondingDepthM: -0.01 }, { pondingDepthM: 0.51 }, { pondingDepthM: NaN }, { infiltrationMmH: -1 },
        { infiltrationMmH: 101 }, { infiltrationMmH: Infinity }, { soilCapacityM: -0.1 }, { soilCapacityM: 0.51 }];
      for (const v of bad) expect(() => validatePatches(g, c, [{ ...ok, ...v }]), JSON.stringify(v)).toThrow('Invalid rain-garden assumptions');
      validatePatches(g, c, [{ ...ok, pondingDepthM: 0, infiltrationMmH: 0, soilCapacityM: 0 }]);
      validatePatches(g, c, [{ ...ok, pondingDepthM: 0.5, infiltrationMmH: 100, soilCapacityM: 0.5 }]);
    });
  });
});

// ---------------------------------------------------------------------------------------------- real site

/** Width and height of a baseline or progressive JPEG, from its first SOFn marker. */
function jpegSize(bytes: Buffer) {
  let i = 2;
  while (i < bytes.length) {
    if (bytes[i] !== 0xff) { i++; continue; }
    const marker = bytes[i + 1];
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return { width: bytes.readUInt16BE(i + 7), height: bytes.readUInt16BE(i + 5) };
    i += 2 + bytes.readUInt16BE(i + 2);
  }
  throw Error('No JPEG frame header');
}

describe('prepared real site (Mercer County Park domain)', () => {
  const g = preparedSite(), surface = (g as Grid & { surface: number[] }).surface, site = g.site!;
  const inSite = (i: number) => { const x = i % g.nx, y = Math.floor(i / g.nx); return x >= site.x && x < site.x + site.width && y >= site.y && y < site.y + site.height; };
  const overlaps = (a: Region, b: Region) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

  it('is the 384 m domain: 96 x 96 cells at 4 m with the 256 m site at (16, 16)', () => {
    validateGrid(g);
    expect([g.nx, g.ny, g.dx, g.dy]).toEqual([96, 96, 4, 4]);
    expect([site.x, site.y, site.width, site.height]).toEqual([16, 16, 64, 64]);
    expect(surface).toHaveLength(96 * 96);
    expect(new Set(surface)).toEqual(new Set(Object.values(SURFACE)));
  });

  it('matches its aerial at 8 px per cell', () => {
    const size = jpegSize(readFileSync(new URL('../public/data/park-domain-aerial.jpg', import.meta.url)));
    expect(size).toEqual({ width: g.nx * 8, height: g.ny * 8 });
  });

  it('masks exactly the two building footprints, which carry no surface class', () => {
    expect(buildings).toHaveLength(2);
    const footprint = new Set(buildings.flatMap(b => cells(g, b)));
    expect(footprint.size).toBe(55);
    g.mask.forEach((m, i) => expect(m === 0).toBe(footprint.has(i)));
    for (const i of footprint) expect(surface[i]).toBe(SURFACE.other);
  });

  it('has lake sink cells in the north-east, on unmasked ground, never eligible', () => {
    const lake = g.sink!.flatMap((s, i) => s ? [i] : []);
    expect(lake.length).toBeGreaterThan(100);
    for (const i of lake) {
      expect(i % g.nx).toBeGreaterThanOrEqual(g.nx / 2);
      expect(Math.floor(i / g.nx)).toBeLessThan(2 * g.ny / 3);
      expect(g.mask[i]).toBe(1);
      expect(g.eligible![i]).toBe(0);
    }
  });

  it('marks exactly the unmasked, non-lake lawn inside the study site as eligible', () => {
    g.eligible!.forEach((e, i) => expect(e).toBe(g.mask[i] && !g.sink![i] && surface[i] === SURFACE.lawn && inSite(i) ? 1 : 0));
    expect(g.eligible!.filter(Boolean).length).toBeGreaterThan(400);
  });

  it('returns a fresh copy on each call', () => {
    const a = preparedSite(); a.z[0] += 100; a.mask[1] = 0;
    expect(preparedSite().z[0]).toBe(g.z[0]);
    expect(terrainSignature(preparedSite())).toBe(terrainSignature(g));
  });

  it('has three valid, disjoint targets inside the site; the default target is the first', () => {
    expect(targets.map(t => t.name)).toEqual(['South loop target', 'East parking target', 'North parking target']);
    for (const t of targets) {
      validateRegion(g, t);
      expect(cells(g, t).every(inSite)).toBe(true);
    }
    for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) expect(overlaps(targets[a], targets[b])).toBe(false);
    expect(defaultConfig().roi).toEqual(targets[0]);
    validateConfig(g, defaultConfig());
    expect(defaultConfig()).toMatchObject({ solver: 'li-1.0.0', thresholdM: 0, maxPatches: 3, areaBudgetM2: 768 });
    expect(SOLVER_VERSION).toBe('li-1.0.0');
  });

  it.each(targets.map((t, k) => [t.name, k] as const))('candidates for the %s: legal 16 m lattice gardens G1..Gn', (_, k) => {
    const target = targets[k], c = { ...defaultConfig(), roi: target }, pool = candidates(g, target);
    expect(pool.length).toBeGreaterThanOrEqual(20);
    expect(pool.map(p => p.id)).toEqual(pool.map((_, j) => `G${j + 1}`));
    const used = new Set<number>();
    for (const p of pool) {
      expect([p.width * g.dx, p.height * g.dy]).toEqual([16, 16]);
      expect(p).toMatchObject({ pondingDepthM: GARDEN.pondingDepthM, infiltrationMmH: GARDEN.infiltrationMmH, soilCapacityM: GARDEN.soilCapacityM });
      expect(overlaps(p, target)).toBe(false);
      for (const i of cells(g, p)) { expect(used.has(i)).toBe(false); used.add(i); }
      validatePatches(g, c, [p]);
    }
    const order = pool.map(p => p.y * g.nx + p.x);
    expect(order).toEqual([...order].sort((a, b) => a - b));   // north-to-south, west-to-east
  });

  it('candidates for the default target: 25 gardens; the pool depends on the target', () => {
    const where = (pool: Patch[]) => pool.map(p => `${p.x},${p.y}`);
    expect(candidates(g, targets[0])).toHaveLength(25);
    expect(candidates(g, targets[0])).toEqual(candidates(g, targets[0]));
    expect(where(candidates(g, targets[2]))).not.toEqual(where(candidates(g, targets[0])));
    // Without a target the lattice includes gardens on the default target, which validation then rejects:
    // a pool must be recomputed whenever the target changes.
    const unrestricted = candidates(g), conflicting = unrestricted.filter(p => overlaps(p, targets[0]));
    expect(unrestricted.length).toBeGreaterThan(25);
    expect(conflicting.length).toBeGreaterThan(0);
    for (const p of conflicting) expect(() => validatePatches(g, defaultConfig(), [p])).toThrow('outside the inspected target');
  });

  it('a default-storm baseline run conserves water and a heuristic design lowers the objective', () => {
    const c = defaultConfig(), pool = candidates(g, c.roi);
    const base = runSync(g, c, [], { snapshots: false });
    expect(base.frames).toEqual([]);
    expect(base.objective).toBeGreaterThan(1);
    expect(base.balance.outflow).toBeGreaterThan(0);
    expect(base.balance.waterBody).toBeGreaterThan(0);
    expect(base.balance.rain).toBeCloseTo((96 * 96 - 55) * 16 * 0.02, 6);
    expect(Math.abs(base.balance.relativeResidual!)).toBeLessThan(1e-9);
    expect(base.minStableDt).toBeGreaterThan(0.1);
    expect(base.minStableDt).toBeLessThan(c.maxDt);
    const design = heuristic(g, c, pool);
    expect(design.length).toBeGreaterThan(0);
    const treated = runSync(g, c, design, { snapshots: false });
    expect(treated.objective).toBeLessThan(base.objective);
    expect(treated.balance.garden).toBeGreaterThan(0);
    expect(Math.abs(treated.balance.relativeResidual!)).toBeLessThan(1e-9);
  });
});

// ---------------------------------------------------------------------------------------------- planner

describe('planner: model output parsing', () => {
  const g = F.flat(12, 12), c = F.baseConfig(g, { roi: F.region(5, 5, 2, 2), maxPatches: 2, areaBudgetM2: 128 });
  const pool = [F.garden('P1', 1, 1), F.garden('P2', 3, 1), F.garden('P3', 8, 8), F.garden('P4', 2, 1)];   // P4 overlaps P1 and P2

  it('accepts the exact schema and returns the patches in order', () => {
    expect(parseProposal({ selected_patch_ids: [] }, g, c, pool)).toEqual([]);
    expect(parseProposal({ selected_patch_ids: ['P3', 'P1'] }, g, c, pool)).toEqual([pool[2], pool[0]]);
    expect(acceptModelText('  {"selected_patch_ids": ["P2"]}\n', g, c, pool)).toEqual({ selected_patch_ids: ['P2'] });
  });

  it('rejects unknown, duplicate, non-string, extra-key, wrong-type and infeasible proposals', () => {
    const cases: [unknown, string][] = [
      [{ selected_patch_ids: ['P9'] }, 'Unknown candidate: P9'],
      [{ selected_patch_ids: ['P1', 'P1'] }, 'Invalid or duplicate candidate IDs'],
      [{ selected_patch_ids: [1] }, 'Invalid or duplicate candidate IDs'],
      [{ selected_patch_ids: 'P1' }, 'Invalid or duplicate candidate IDs'],
      [{ selected_patch_ids: [], reason: 'none help' }, 'Expected exactly selected_patch_ids'],
      [{ ids: [] }, 'Expected exactly selected_patch_ids'],
      [[], 'Expected exactly selected_patch_ids'],
      [null, 'Expected exactly selected_patch_ids'],
      [{ selected_patch_ids: ['P1', 'P4'] }, 'Gardens may not overlap'],
      [{ selected_patch_ids: ['P1', 'P2', 'P3'] }, 'Maximum patch count exceeded'],
    ];
    for (const [value, message] of cases) expect(() => parseProposal(value, g, c, pool), JSON.stringify(value)).toThrow(message);
    expect(() => parseProposal({ selected_patch_ids: ['P1', 'P3'] }, g, { ...c, areaBudgetM2: 100 }, pool)).toThrow('Treated area exceeds budget');
  });

  it('raises ModelFormatError only for text that is not JSON', () => {
    for (const text of ['```json\n{"selected_patch_ids": []}\n```', 'I pick P1 and P3.', '', '{"selected_patch_ids": [P1]}']) {
      expect(() => acceptModelText(text, g, c, pool), text).toThrow(ModelFormatError);
    }
    for (const text of ['{"selected_patch_ids": ["P9"]}', '["P1"]', '"P1"', '{"selected_patch_ids": [], "why": 1}']) {
      let error: unknown;
      try { acceptModelText(text, g, c, pool); } catch (e) { error = e; }
      expect(error, text).toBeInstanceOf(Error);
      expect(error, text).not.toBeInstanceOf(ModelFormatError);
    }
  });

  it('gives different repair instructions for format and constraint errors', () => {
    const format = repairInstruction(new ModelFormatError('x')), rule = repairInstruction(Error('Treated area exceeds budget'));
    expect(format).not.toBe(rule);
    expect(format).toMatch(/^Formatting only/);
    expect(rule).toContain('Treated area exceeds budget');
    expect(repairInstruction('not an error')).toContain('invalid');
  });
});

describe('planner: learned-planner scope (PROMPT-CX-001)', () => {
  it('accepts only the training settings: li-1.0.0, threshold 0, dry initial surface', () => {
    expect(plannerUnsupported(defaultConfig())).toBeNull();
    expect(plannerUnsupported({ ...defaultConfig(), solver: 'dw-1.0.0' })).toMatch(/training settings/);
    expect(plannerUnsupported({ ...defaultConfig(), thresholdM: .02 })).toMatch(/training settings/);
    expect(plannerUnsupported({ ...defaultConfig(), initialDepthM: .01 })).toMatch(/training settings/);
    expect(plannerUnsupported({ ...defaultConfig(), rainMmH: 100, infiltrationMmH: 10 })).toBeNull();   // storm and soil are in the prompt
  });
});

describe('planner: routed-area feature counts each source cell once (FEAT-CX-001)', () => {
  // Codex's counterexample (coordination/codex/check-d8-reentry.ts): a strictly downhill path leaves the garden
  // and re-enters it, so exit-summed upstream areas counted the same cells twice (2,688 vs 2,560 m2).
  it('matches an independent path-following union on a re-entering path', () => {
    const nx = 12, ny = 12;
    const g = { id: 'reentry', name: 'Re-entering D8 path', nx, ny, dx: 8, dy: 8,
      z: Array.from({ length: nx * ny }, (_, i) => 2 + (i % nx) * .01 + Math.floor(i / nx) * .01),
      mask: Array(nx * ny).fill(1), eligible: Array(nx * ny).fill(1) };
    [[3, 3], [3, 2], [4, 2], [5, 3], [4, 4], [4, 5], [4, 6]].forEach(([x, y], k) => { g.z[y * nx + x] = [1, .2, .15, .1, 0, -.1, -.2][k]; });
    const roi = { x: 3, y: 6, width: 3, height: 5, name: 'Target' };
    const garden = { id: 'A', name: 'Garden', x: 3, y: 3, width: 2, height: 2, pondingDepthM: .15, infiltrationMmH: 25, soilCapacityM: .15 };
    const routing = terrainFeatures(g, roi), inside = new Set(cells(g, garden)), target = new Set(cells(g, roi));
    let union = 0;
    for (let i = 0; i < nx * ny; i++) {
      let j = i, passed = false;
      for (let step = 0; j >= 0 && step < nx * ny; step++) { if (inside.has(j)) passed = true; if (passed && target.has(j)) { union++; break; } j = routing.next[j]; }
    }
    expect(union * 64).toBe(2560);
    expect(features(g, { ...defaultConfig(), roi }, [garden]).candidates[0].routedToTargetM2).toBe(union * 64);
  });
});

describe('planner: production features and prompt', () => {
  // A plane falling south at 1 %: D8 routes every cell straight south, so the routed areas are known exactly.
  const g = F.southPlane(10, 12, 0.01), roi = F.region(3, 5, 4, 2, 'Plane target'), c = F.baseConfig(g, { roi, maxPatches: 2, areaBudgetM2: 128 });
  const above = F.garden('A', 3, 1), below = F.garden('B', 3, 8), side = F.garden('C', 8, 1), pool = [above, below, side];

  it('routes upstream area to the target exactly on a tilted plane; a downhill garden gets zero', () => {
    const f = features(g, c, pool), [a, b, s] = f.candidates;
    expect(f.target).toEqual({ areaM2: 8 * 16, elevationM: expect.closeTo(0.04 * 5.5, 12), upstreamM2: 4 * 5 * 16 });
    expect(a).toMatchObject({ routedToTargetM2: 2 * 3 * 16, reachFraction: 1, maxUpstreamM2: 3 * 16, eastM: -4, southM: -16, areaM2: 64 });
    expect(a.relElevationM).toBeCloseTo(0.16, 12);
    expect(a.slopePct).toBeCloseTo(1, 10);
    expect(a.storageM3).toBeCloseTo(0.15 * 64, 12);
    expect(b).toMatchObject({ routedToTargetM2: 0, reachFraction: 0 });
    expect(b.relElevationM).toBeLessThan(0);
    expect(s).toMatchObject({ routedToTargetM2: 0, reachFraction: 0 });
  });

  it('builds a deterministic prompt with every candidate once and no solver output', () => {
    const real = preparedSite(), rc = defaultConfig(), rpool = candidates(real, rc.roi);
    for (const [grid, config, list] of [[g, c, pool], [real, rc, rpool]] as const) {
      const prompt = plannerPrompt(grid, config, list);
      expect(plannerPrompt(grid, config, list)).toBe(prompt);
      expect(plannerPrompt(grid, { ...config, solver: config.solver === 'li-1.0.0' ? 'dw-1.0.0' : 'li-1.0.0' }, list)).toBe(prompt);
      for (const p of list) expect(prompt.split('\n').filter(line => line.startsWith(`${p.id},`)), p.id).toHaveLength(1);
      expect(prompt).not.toMatch(/objective|label|residual|outflow|waterBody|best/i);
      expect(prompt.split('\n').slice(1).join('\n')).not.toMatch(/peak/i);   // only the task sentence names the peak
      const rows = prompt.split('\n').filter(line => /^[A-Z]\d*,/.test(line));
      expect(rows).toHaveLength(list.length);
      expect(rows.every(r => r.split(',').length === 10)).toBe(true);
    }
    const realPrompt = plannerPrompt(real, rc, rpool);
    expect(realPrompt).toContain('Use at most 3 gardens and 768 m2 in total.');
    expect(realPrompt).toContain('Each garden: 256 m2, stores up to 0.15 m of runoff');
    expect(plannerPrompt(g, c, pool)).toContain('downhill-routed upstream area 320 m2');
  });

  it('gives finite, nonnegative routing features for every real-site candidate', () => {
    const real = preparedSite(), rc = defaultConfig(), f = features(real, rc, candidates(real, rc.roi));
    for (const k of f.candidates) {
      for (const v of Object.values(k)) if (typeof v === 'number') expect(Number.isFinite(v), k.id).toBe(true);
      expect(k.routedToTargetM2).toBeGreaterThanOrEqual(0);
      expect(k.reachFraction).toBeGreaterThanOrEqual(0);
      expect(k.reachFraction).toBeLessThanOrEqual(1);
    }
    expect(f.candidates.some(k => k.routedToTargetM2 > 0)).toBe(true);
    expect(f.target.upstreamM2).toBeGreaterThan(0);
  });
});

describe('planner: search', () => {
  // A valley draining south past a mid-slope target; U* gardens sit upstream in the valley, S* on the flanks,
  // D* downstream of the target. Runs take a few milliseconds each.
  const g = F.gridFrom(12, 14, (x, y) => 0.01 * (13 - y) * 4 + 0.01 * Math.abs(x - 5.5) * 4, { id: 'valley' });
  const c = F.baseConfig(g, { roi: F.region(4, 7, 4, 2, 'Valley target'), boundary: 'open', rainMinutes: 10, drainMinutes: 5, infiltrationMmH: 3, outputSeconds: 60, maxPatches: 2, areaBudgetM2: 128 });
  const pool = [F.garden('U1', 4, 2), F.garden('U2', 6, 4), F.garden('S1', 1, 3), F.garden('S2', 9, 3), F.garden('D1', 4, 10), F.garden('D2', 6, 11)];
  const evaluate = async (p: Patch[]) => runSync(g, c, p, { snapshots: false });
  const flat = F.flat(10, 10), fc = F.baseConfig(flat, { roi: F.region(4, 4, 2, 2), maxPatches: 2, areaBudgetM2: 1000 });
  const [A, B, C] = [F.garden('A', 1, 1), F.garden('B', 1, 7), F.garden('C', 7, 7)];
  const tableEvaluator = (table: Record<string, number>) => async (p: Patch[]) => fakeRun(p, table[designKey(p)]);

  it('greedy adds gardens only while each addition improves the objective', async () => {
    const plan = await greedy(g, c, pool, evaluate), ids = plan.proposal.selected_patch_ids;
    expect(ids.length).toBeGreaterThan(0);
    expect(plan.run.patches.map(p => p.id)).toEqual(ids);
    const prefixes = await Promise.all(ids.map((_, k) => evaluate(pool.filter(p => ids.slice(0, k + 1).includes(p.id)))));
    const objectives = [(await evaluate([])).objective, ...prefixes.map(r => r.objective)];
    for (let k = 1; k < objectives.length; k++) expect(objectives[k]).toBeLessThan(objectives[k - 1] - 1e-8);
    expect(plan.run.objective).toBe(objectives.at(-1));
    expect(plan.simulationCalls).toBeLessThanOrEqual(1 + pool.length * c.maxPatches);
  });

  it('greedy returns the empty design when nothing improves (including improvements below 1e-8)', async () => {
    const plan = await greedy(flat, fc, [A, B, C], tableEvaluator({ '': 10, A: 10, B: 10 - 5e-9, C: 12 }));
    expect(plan.proposal.selected_patch_ids).toEqual([]);
    expect(plan.simulationCalls).toBe(4);
    const dry = await greedy(g, { ...c, rainMmH: 0 }, pool, async p => runSync(g, { ...c, rainMmH: 0 }, p, { snapshots: false }));
    expect(dry.proposal.selected_patch_ids).toEqual([]);
    expect(dry.simulationCalls).toBe(1 + pool.length);
    const stops = await greedy(flat, fc, [A, B, C], tableEvaluator({ '': 10, A: 8, B: 9, C: 9, 'A,B': 8, 'A,C': 9 }));
    expect(stops.proposal.selected_patch_ids).toEqual(['A']);   // A+B ties A but treats more area
  });

  it('searchBest escapes a greedy trap with a swap, and its evaluated list contains the greedy design', async () => {
    const result = await searchBest(flat, fc, [A, B, C], tableEvaluator({ '': 10, A: 5, B: 6, C: 6, 'A,B': 4.5, 'A,C': 4.5, 'B,C': 1 }));
    expect(result.greedy.proposal.selected_patch_ids).toEqual(['A', 'B']);
    expect(designKey(result.best.patches)).toBe('B,C');
    expect(result.best.objective).toBe(1);
    expect(result.passes).toBe(1);
    const keys = result.evaluated.map(e => designKey(e.ids));
    expect(keys).toContain('A,B');
    expect(new Set(keys).size).toBe(keys.length);
    expect(result.simulationCalls).toBe(keys.length);
  });

  it('searchBest with the solver is at least as good as greedy and matches exhaustive search here', async () => {
    const result = await searchBest(g, c, pool, evaluate);
    expect(result.best.objective).toBeLessThanOrEqual(result.greedy.run.objective);
    expect(better(g, result.greedy.run, result.best)).toBe(false);
    const keys = result.evaluated.map(e => designKey(e.ids));
    expect(keys).toContain(designKey(result.greedy.proposal.selected_patch_ids));
    for (const e of result.evaluated) expect(designKey(e.run.patches)).toBe(designKey(e.ids));
    const all = feasibleDesigns(g, c, pool);
    expect(all).toHaveLength(1 + 6 + 15);
    const optimum = Math.min(...all.map(p => runSync(g, c, p, { snapshots: false }).objective));
    expect(result.best.objective).toBeCloseTo(optimum, 12);
  });

  it('heuristic and randomDesign are deterministic and feasible', () => {
    const h = heuristic(g, c, pool);
    expect(h).toEqual(heuristic(g, c, pool));
    validatePatches(g, c, h);
    expect(h.length).toBeGreaterThan(0);
    const routed = new Map(features(g, c, pool).candidates.map(k => [k.id, k.routedToTargetM2]));
    expect(h.every(p => routed.get(p.id)! > 0)).toBe(true);
    for (const seed of [0, 1, 42, 2 ** 31]) {
      const r = randomDesign(g, c, pool, seed);
      expect(r).toEqual(randomDesign(g, c, pool, seed));
      validatePatches(g, c, r);
      expect(r).toHaveLength(c.maxPatches);   // fills the count when every pair is feasible
    }
    const rng = seededRandom(7), draws = Array.from({ length: 1000 }, rng);
    expect(draws.every(v => v >= 0 && v < 1)).toBe(true);
    expect(seededRandom(7)()).toBe(draws[0]);
  });

  it('designKey is order-independent; better() breaks ties by area, then by IDs', () => {
    expect(designKey([B, A, C])).toBe(designKey([C, A, B]));
    expect(designKey(['G10', 'G2'])).toBe(designKey(['G2', 'G10']));
    expect(designKey([A, B])).toBe(designKey(['B', 'A']));
    expect(designKey([])).toBe('');
    const big = F.garden('Z', 4, 1, 3, 3);
    expect(better(flat, fakeRun([A], 1), fakeRun([B], 1 + 2e-8))).toBe(true);
    expect(better(flat, fakeRun([big], 1), fakeRun([A], 1 + 5e-9))).toBe(false);   // tie: less area wins
    expect(better(flat, fakeRun([A], 1), fakeRun([B], 1))).toBe(true);             // tie and equal area: IDs
    expect(better(flat, fakeRun([B], 1), fakeRun([A], 1))).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------- scenarios

describe('scenario files (version 2)', () => {
  const g = F.bowl(8, 0.3, { id: 'bowl-8' }), patches = [F.garden('A', 2, 2)];
  const config = (solver: Config['solver']) => F.baseConfig(g, { solver, roi: F.region(5, 5, 2, 2), boundary: 'open', rainMinutes: 3, drainMinutes: 2, infiltrationMmH: 3, outputSeconds: 30 });

  it.each(F.SOLVER_IDS)('%s: an exported and re-imported scenario reproduces the run exactly', solver => {
    const s = scenario(g, config(solver), patches), text = JSON.stringify(s), back = importScenario(text, g);
    expect(back).toEqual(s);
    expect(s).toMatchObject({ version: 2, solver, gridId: 'bowl-8', terrainSignature: terrainSignature(g) });
    expect(withoutClock(runSync(g, back.config, back.patches))).toEqual(withoutClock(runSync(g, config(solver), patches)));
  });

  it('rejects other versions, terrain, solvers, invalid JSON, oversize files and invalid contents', () => {
    const s = scenario(g, config('li-1.0.0'), patches), json = (v: unknown) => JSON.stringify(v);
    expect(() => importScenario(json({ ...s, version: 1 }), g)).toThrow('Incompatible scenario version');
    expect(() => importScenario(json({ ...s, version: undefined }), g)).toThrow('Incompatible scenario version');
    expect(() => importScenario('null', g)).toThrow('Incompatible scenario version');
    expect(() => importScenario(json(s), { ...g, z: g.z.map(z => z + 0.001) })).toThrow('different terrain');
    expect(() => importScenario(json(s), { ...g, id: 'other' })).toThrow('different terrain');
    expect(() => importScenario(json(s), { ...g, sink: g.z.map((_, i) => i === 0 ? 1 : 0) })).toThrow('different terrain');
    expect(() => importScenario(json({ ...s, solver: 'li-2.0.0', config: { ...s.config, solver: 'li-2.0.0' } }), g)).toThrow('Unknown or inconsistent solver');
    expect(() => importScenario(json({ ...s, solver: 'dw-1.0.0' }), g)).toThrow('Unknown or inconsistent solver');
    expect(() => importScenario('{"version": 2,', g)).toThrow('not valid JSON');
    expect(() => importScenario(json(s) + ' '.repeat(100_001), g)).toThrow('100 kB');
    expect(() => importScenario(json({ ...s, config: { ...s.config, rainMmH: 1000 } }), g)).toThrow('outside supported range');
    expect(() => importScenario(json({ ...s, patches: [{ ...patches[0], x: 0 }] }), g)).toThrow('inside the boundary');
    expect(() => scenario(g, config('li-1.0.0'), [F.garden('A', 0, 0)])).toThrow();
  });

  it('signs the terrain, masks and sinks, but not names', () => {
    const sig = terrainSignature(g);
    expect(terrainSignature(structuredClone(g))).toBe(sig);
    expect(terrainSignature({ ...g, name: 'Renamed' })).toBe(sig);
    for (const v of [{ z: g.z.map((z, i) => i === 5 ? z + 1e-9 : z) }, { mask: g.mask.map((m, i) => i === 0 ? 0 : m) },
      { eligible: g.eligible!.map((e, i) => i === 9 ? 0 : e) }, { sink: g.z.map(() => 0) }, { dx: 2 }])
      expect(terrainSignature({ ...g, ...v }), JSON.stringify(Object.keys(v))).not.toBe(sig);
  });

  it('summarizes a run with its solver, observation time and balance', () => {
    const run = runSync(g, config('dw-1.0.0'), patches), s = summary(run);
    expect(s).toMatchObject({ solver: 'dw-1.0.0', method: 'manual', observationSeconds: 300, objectiveM3: run.objective, gardens: ['A'], steps: run.steps });
    expect(s.balanceM3).toEqual(run.balance);
  });
});

// ---------------------------------------------------------------------------------------------- terrain features

describe('terrain features (D8)', () => {
  /** Follows next[] from i; returns the terminal cell and the path. */
  const follow = (next: Int32Array, i: number) => { const path = [i]; while (next[path.at(-1)!] >= 0) path.push(next[path.at(-1)!]); return path; };

  it('routes strictly downhill to unmasked cells; sinks and masked cells are terminal', () => {
    const g = F.rough(14, 11); g.mask[40] = g.mask[41] = 0; g.sink = g.z.map((_, i) => i % 14 === 0 ? 1 : 0);
    const real = preparedSite();
    for (const [grid, roi] of [[g, F.region(6, 5, 2, 2)], [real, targets[0]]] as const) {
      const r = terrainFeatures(grid, roi);
      r.next.forEach((j, i) => {
        if (j < 0) return;
        expect(grid.z[j]).toBeLessThan(grid.z[i]);
        expect(grid.mask[j]).toBe(1);
        expect(Math.abs((j % grid.nx) - (i % grid.nx))).toBeLessThanOrEqual(1);
        expect(Math.abs(Math.floor(j / grid.nx) - Math.floor(i / grid.nx))).toBeLessThanOrEqual(1);
      });
      grid.sink!.forEach((s, i) => { if (s) expect(r.next[i]).toBe(-1); });
      grid.mask.forEach((m, i) => { if (!m) { expect(r.next[i]).toBe(-1); expect(r.contributingAreaM2[i]).toBe(0); } });
      // Every valid cell drains to exactly one terminal (outlet, pit or sink), so terminal areas sum to the valid area.
      const valid = grid.mask.filter(Boolean).length * grid.dx * grid.dy;
      const terminal = Array.from(r.next).reduce((s, j, i) => s + (j < 0 ? r.contributingAreaM2[i] : 0), 0);
      expect(terminal).toBeCloseTo(valid, 6);
    }
  });

  it('agrees with independent path following for reach and contributing area', () => {
    const g = F.rough(14, 11, { dx: 3, dy: 5 }), roi = F.region(6, 5, 2, 2), target = new Set(cells(g, roi)), r = terrainFeatures(g, roi);
    const paths = g.z.map((_, i) => follow(r.next, i)), area = g.dx * g.dy;
    let upstream = 0;
    g.z.forEach((_, i) => {
      const reaches = paths[i].some(j => target.has(j));
      expect(r.reachesTarget[i], `cell ${i}`).toBe(reaches ? 1 : 0);
      if (reaches && !target.has(i)) upstream += area;
      expect(r.contributingAreaM2[i]).toBeCloseTo(paths.filter(p => p.includes(i)).length * area, 9);
    });
    expect(r.targetUpstreamM2).toBeCloseTo(upstream, 9);
    expect(upstream).toBeGreaterThan(0);
  });

  it('on a south-falling plane, the upstream area is the target columns north of the target', () => {
    const g = F.southPlane(10, 12, 0.01), roi = F.region(3, 5, 4, 2), r = terrainFeatures(g, roi);
    expect(r.targetUpstreamM2).toBe(4 * 5 * 16);
    expect(r.next[2 * 10 + 4]).toBe(3 * 10 + 4);   // straight south, not diagonal
    expect(r.contributingAreaM2[11 * 10 + 4]).toBe(12 * 16);
    expect(r.reachesTarget[8 * 10 + 4]).toBe(0);   // below the target
  });

  it('breaks slope ties by scan order', () => {
    const g = F.gridFrom(3, 3, (x, y) => x === 1 && y === 1 ? 1 : 0), r = terrainFeatures(g, F.region(0, 0, 1, 1));
    expect(r.next[4]).toBe(1);   // N, W, E and S tie; north comes first in scan order
    expect(Array.from(r.next).filter(j => j >= 0)).toEqual([1]);
  });
});
