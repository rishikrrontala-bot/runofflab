import { cells, validateConfig, validatePatches, type Accounting, type Config, type Frame, type Grid, type Patch, type Run } from './types';

export const GRAVITY = 9.81;
/** Local inertial scheme constants: Courant-type coefficient (Bates et al. 2010 recommend 0.7) and the
 *  flow depth below which a face is treated as dry. */
export const INERTIAL = { alpha: 0.7, minFlowDepthM: 1e-4 };
/** Diffusive-wave reference scheme: explicit stability fraction of cell area / total face conductance. */
export const DIFFUSIVE = { stability: 0.2 };

export const mmHToMS = (value: number) => value / 3_600_000;

/** dw-1.0.0 shared-face Manning flux in m³/s, positive A -> B, with head regularization below headEpsilonM. */
export function faceFlux(za: number, zb: number, ha: number, hb: number, width: number, distance: number, n: number, headEpsilonM = .01): number {
  const etaA = za + ha, etaB = zb + hb, dEta = etaA - etaB;
  const hf = Math.max(0, Math.max(etaA, etaB) - Math.max(za, zb));
  if (hf === 0 || dEta === 0) return 0;
  return width / n * Math.pow(hf, 5 / 3) * dEta / Math.sqrt(Math.max(Math.abs(dEta), headEpsilonM) * distance);
}

/** li-1.0.0 unit-width discharge update (m²/s) for one face, positive A -> B (Bates, Horritt & Fewtrell 2010):
 *    q_new = (q_old - g·hf·dt·(etaB - etaA)/d) / (1 + g·dt·n²·|q_new| / hf^(7/3))
 *  Friction uses |q_new| (implicit), solved exactly from the quadratic in a cancellation-free form,
 *  which stays stable in shallow flow where the semi-implicit |q_old| form oscillates. */
export function inertialDischarge(qOld: number, etaA: number, etaB: number, hf: number, distance: number, dt: number, n: number): number {
  const qStar = qOld - GRAVITY * hf * dt * (etaB - etaA) / distance;
  const k = GRAVITY * dt * n * n / (hf * hf * Math.cbrt(hf)), magnitude = Math.abs(qStar);   // hf^(7/3)
  return magnitude === 0 ? 0 : Math.sign(qStar) * 2 * magnitude / (1 + Math.sqrt(1 + 4 * k * magnitude));
}

/** Fraction of each donor's planned outgoing volume it can actually supply this step (<= 1). */
export function donorFactors(depth: Float64Array, outgoing: Float64Array, dt: number, area: number, into = new Float64Array(depth.length)): Float64Array {
  for (let i = 0; i < depth.length; i++) into[i] = outgoing[i] > 0 ? Math.min(1, depth[i] * area / (outgoing[i] * dt)) : 1;
  return into;
}

/** Faces between computed cells, plus one-way outlet faces on open edges (b = -1: a zero-depth ghost cell at the
 *  edge cell's own elevation, half a cell away). Built once per run in a fixed order, so runs are deterministic. */
function buildFaces(g: Grid, c: Config) {
  const a: number[] = [], b: number[] = [], width: number[] = [], distance: number[] = [];
  const add = (from: number, to: number, w: number, d: number) => { a.push(from); b.push(to); width.push(w); distance.push(d); };
  for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
    const i = y * g.nx + x;
    if (!g.mask[i]) continue;
    if (x + 1 < g.nx && g.mask[i + 1]) add(i, i + 1, g.dy, g.dx);
    if (y + 1 < g.ny && g.mask[i + g.nx]) add(i, i + g.nx, g.dx, g.dy);
    if (c.boundary === 'open') {
      if (x === 0 || x === g.nx - 1) add(i, -1, g.dy, g.dx / 2);
      if (y === 0 || y === g.ny - 1) add(i, -1, g.dx, g.dy / 2);
    }
  }
  return { count: a.length, a: Int32Array.from(a), b: Int32Array.from(b), width: Float64Array.from(width), distance: Float64Array.from(distance) };
}

/** courant: overrides the stability coefficient (INERTIAL.alpha or DIFFUSIVE.stability) for timestep-refinement studies. */
export interface SimOptions { snapshots?: boolean; method?: string; courant?: number }

/**
 * Deterministic surface-water simulation. A generator, so a worker can yield to its event loop between saved
 * states; no rendering clock ever enters the physics.
 *
 * Each step: (1) route the current water state across shared faces with the configured scheme, limiting each
 * donor's combined outflow to the water it holds; (2) add rain; (3) let rain gardens capture water on their
 * cells into basin storage up to their ponding depth; (4) infiltrate from garden storage, then from the
 * surface, into finite soil storage; (5) remove water that reached a water-body sink. Every transfer has a
 * destination, and the balance below is checked at the end of the run.
 */
export function* simulate(g: Grid, c: Config, patches: Patch[] = [], options: SimOptions = {}): Generator<Frame, Run, void> {
  validateConfig(g, c); validatePatches(g, c, patches);
  const start = performance.now(), N = g.z.length, area = g.dx * g.dy, total = (c.rainMinutes + c.drainMinutes) * 60;
  const z = g.z, h = new Float64Array(N), soil = new Float64Array(N), pond = new Float64Array(N);
  const infiltration = new Float64Array(N).fill(mmHToMS(c.infiltrationMmH)), soilCapacity = new Float64Array(N).fill(c.soilCapacityM);
  const pondCapacity = new Float64Array(N), sink = Uint8Array.from(g.sink ?? new Array(N).fill(0));
  for (let i = 0; i < N; i++) if (g.mask[i] && !sink[i]) { h[i] = c.initialDepthM; soil[i] = c.initialSoilM; }
  for (let i = 0; i < N; i++) if (sink[i]) { infiltration[i] = 0; soilCapacity[i] = 0; }
  for (const p of patches) for (const i of cells(g, p)) {
    pondCapacity[i] = p.pondingDepthM; infiltration[i] = mmHToMS(p.infiltrationMmH); soilCapacity[i] = p.soilCapacityM;
  }
  // A garden's soil cannot start fuller than its own capacity; the initial store is counted from the actual state.
  for (let i = 0; i < N; i++) if (soil[i] > soilCapacity[i]) soil[i] = soilCapacity[i];
  let initial = 0;
  for (let i = 0; i < N; i++) initial += (h[i] + soil[i]) * area;
  const roi = cells(g, c.roi), frames: Frame[] = [], faces = buildFaces(g, c);
  const inertial = c.solver === 'li-1.0.0', dMin = Math.min(g.dx, g.dy), n = c.roughness;
  const courant = options.courant ?? (inertial ? INERTIAL.alpha : DIFFUSIVE.stability);
  if (!(courant > 0 && courant <= 1)) throw Error('Invalid stability coefficient');

  let rain = 0, outflow = 0, waterBody = 0, t = 0, steps = 0, minStableDt = Infinity, nextOutput = c.outputSeconds;
  let objective = 0, peakRoiDepth = 0, peakRoiVolume = 0, peakDomainVolume = 0;
  const sum = (values: Float64Array) => { let s = 0; for (let i = 0; i < N; i++) s += values[i]; return s * area; };
  const balance = (): Accounting => {
    const surface = sum(h), stored = sum(soil), garden = sum(pond);
    const residual = initial + rain - surface - stored - garden - outflow - waterBody;
    return { initial, rain, surface, soil: stored, garden, outflow, waterBody, residual, relativeResidual: initial + rain > 1e-9 ? residual / (initial + rain) : null };
  };
  const metrics = () => {
    let depth = 0, volume = 0, excess = 0;
    for (const i of roi) { depth = Math.max(depth, h[i]); volume += h[i] * area; excess += Math.max(0, h[i] - c.thresholdM) * area; }
    objective = Math.max(objective, excess); peakRoiDepth = Math.max(peakRoiDepth, depth);
    peakRoiVolume = Math.max(peakRoiVolume, volume); peakDomainVolume = Math.max(peakDomainVolume, sum(h));
    return { roiDepth: depth, roiVolume: volume, roiExcess: excess };
  };
  const snapshot = (): Frame => ({ t, h: options.snapshots === false ? new Float32Array() : Float32Array.from(h), balance: balance(), ...metrics() });
  const first = snapshot(); if (options.snapshots !== false) frames.push(first); yield first;

  const q = new Float64Array(faces.count), planned = new Float64Array(faces.count);
  const out = new Float64Array(N), conductance = new Float64Array(N), delta = new Float64Array(N), donor = new Float64Array(N);
  const { a: fa, b: fb, width: fw, distance: fd } = faces;

  /** Plans face discharges for this step and returns the stable timestep (Infinity when unconstrained). */
  const planInertial = (dtCap: number): [number, number] => {
    let hmax = 0;
    for (let i = 0; i < N; i++) if (h[i] > hmax) hmax = h[i];
    const stable = hmax > 0 ? courant * dMin / Math.sqrt(GRAVITY * hmax) : Infinity, dt = Math.min(dtCap, stable);
    for (let f = 0; f < faces.count; f++) {
      const ia = fa[f], ib = fb[f], za = z[ia], zb = ib < 0 ? za : z[ib], etaA = za + h[ia], etaB = zb + (ib < 0 ? 0 : h[ib]);
      const hf = Math.max(etaA, etaB) - Math.max(za, zb);
      let value = hf > INERTIAL.minFlowDepthM ? inertialDischarge(q[f], etaA, etaB, hf, fd[f], dt, n) * fw[f] : 0;
      if (ib < 0 && value < 0) value = 0;   // the outlet ghost never supplies water
      planned[f] = value;
    }
    return [dt, stable];
  };
  const planDiffusive = (dtCap: number): [number, number] => {
    conductance.fill(0);
    for (let f = 0; f < faces.count; f++) {
      const ia = fa[f], ib = fb[f], za = z[ia], zb = ib < 0 ? za : z[ib], hb = ib < 0 ? 0 : h[ib];
      const hf = Math.max(0, Math.max(za + h[ia], zb + hb) - Math.max(za, zb)), dEta = za + h[ia] - zb - hb;
      const k = fw[f] / n * Math.pow(hf, 5 / 3) / Math.sqrt(fd[f] * Math.max(Math.abs(dEta), c.headEpsilonM));
      conductance[ia] += k; if (ib >= 0) conductance[ib] += k;
      planned[f] = ib < 0 ? Math.max(0, k * dEta) : k * dEta;
    }
    let stable = Infinity;
    for (let i = 0; i < N; i++) if (conductance[i] > 0) stable = Math.min(stable, courant * area / conductance[i]);
    return [Math.min(dtCap, stable), stable];
  };

  while (t < total - 1e-8) {
    const raining = t < c.rainMinutes * 60 - 1e-8, rate = raining ? mmHToMS(c.rainMmH) : 0;
    const dtCap = Math.min(c.maxDt, total - t, nextOutput - t, raining ? c.rainMinutes * 60 - t : Infinity);
    const [dt, stable] = inertial ? planInertial(dtCap) : planDiffusive(dtCap);
    if (!Number.isFinite(dt) || dt < 1e-7 || steps > 2_000_000) throw Error('Numerical timestep failed: reduce rain or inspect terrain');
    if (stable < minStableDt) minStableDt = stable;

    // Shared-face transfers, applied once with equal and opposite volumes. Each donor's combined outgoing
    // volume is scaled to the water it holds, so no clipping ever creates or destroys water.
    out.fill(0);
    for (let f = 0; f < faces.count; f++) {
      if (planned[f] > 0) out[fa[f]] += planned[f]; else if (planned[f] < 0) out[fb[f]] -= planned[f];
    }
    const scale = donorFactors(h, out, dt, area, donor);
    delta.fill(0);
    for (let f = 0; f < faces.count; f++) {
      const ia = fa[f], ib = fb[f], Q = planned[f] * scale[planned[f] >= 0 ? ia : ib];
      const v = Q * dt;
      delta[ia] -= v; if (ib >= 0) delta[ib] += v; else outflow += v;
      if (inertial) q[f] = Q / fw[f];   // the limited discharge carries into the next step
    }

    for (let i = 0; i < N; i++) {
      if (!g.mask[i]) continue;
      h[i] += delta[i] / area;
      if (!Number.isFinite(h[i]) || h[i] < -1e-10) throw Error('Nonfinite or negative water depth');
      if (h[i] < 0) h[i] = 0;   // roundoff below 1e-10 m only
      const added = rate * dt; h[i] += added; rain += added * area;
      if (pondCapacity[i] > 0) { const captured = Math.min(h[i], pondCapacity[i] - pond[i]); h[i] -= captured; pond[i] += captured; }
      let room = Math.min(infiltration[i] * dt, Math.max(0, soilCapacity[i] - soil[i]));
      const fromPond = Math.min(pond[i], room); pond[i] -= fromPond; room -= fromPond;
      const fromSurface = Math.min(h[i], room); h[i] -= fromSurface;
      soil[i] += fromPond + fromSurface;
      if (sink[i]) { waterBody += h[i] * area; h[i] = 0; }
    }

    t += dt; steps++; metrics();
    if (t >= nextOutput - 1e-8 || t >= total - 1e-8) {
      const frame = snapshot(); if (options.snapshots !== false) frames.push(frame); yield frame;
      nextOutput = Math.min(total, nextOutput + c.outputSeconds);
    }
  }
  const final = balance();
  if (Math.abs(final.residual) > Math.max(1e-6, (final.rain + final.initial) * 0.001)) throw Error('Water-balance residual exceeds tolerance');
  return { config: structuredClone(c), gridId: g.id, patches: structuredClone(patches), frames, objective, peakRoiDepth, peakRoiVolume, peakDomainVolume,
    balance: final, wallMs: performance.now() - start, steps, minStableDt, method: options.method ?? 'manual' };
}

export function runSync(g: Grid, c: Config, patches: Patch[] = [], options: SimOptions = {}): Run {
  const it = simulate(g, c, patches, options); let v = it.next(); while (!v.done) v = it.next(); return v.value;
}
