/**
 * Small synthetic terrains for the core regression suite. Every builder is deterministic and returns a fresh
 * object, so a test may mutate what it receives. Conventions follow src/core/types.ts: row-major, row 0 = north,
 * elevations in metres, cell sizes dx (east-west) and dy (north-south) in metres.
 */
import { defaultConfig, type Config, type Grid, type Patch, type Region } from '../../src/core/types';

export interface GridOptions { dx?: number; dy?: number; id?: string; eligible?: boolean }

/** A grid whose elevation is given by f(x, y) for cell column x and row y. */
export function gridFrom(nx: number, ny: number, f: (x: number, y: number) => number, o: GridOptions = {}): Grid {
  const z: number[] = [];
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) z.push(f(x, y));
  return {
    id: o.id ?? `fixture-${nx}x${ny}`, name: 'Synthetic fixture', nx, ny, dx: o.dx ?? 4, dy: o.dy ?? 4, z,
    mask: new Array(nx * ny).fill(1), ...(o.eligible === false ? {} : { eligible: new Array(nx * ny).fill(1) }),
  };
}

export const flat = (nx: number, ny: number, o: GridOptions = {}) => gridFrom(nx, ny, () => 0, o);

/** A plane falling `slope` (m/m) towards the south (increasing row), optionally also towards the east. */
export const southPlane = (nx: number, ny: number, slope: number, o: GridOptions = {}, eastSlope = 0) =>
  gridFrom(nx, ny, (x, y) => slope * (ny - 1 - y) * (o.dy ?? 4) + eastSlope * (nx - 1 - x) * (o.dx ?? 4), o);

/** A closed paraboloid bowl, `depth` metres deep at the centre relative to its rim corners. */
export function bowl(n: number, depth: number, o: GridOptions = {}): Grid {
  const c = (n - 1) / 2, r2 = 2 * c * c;
  return gridFrom(n, n, (x, y) => depth * (((x - c) ** 2 + (y - c) ** 2) / r2), o);
}

/** Deterministic rough terrain (sum of incommensurate sines plus a gentle tilt), for stress tests. */
export const rough = (nx: number, ny: number, o: GridOptions = {}) =>
  gridFrom(nx, ny, (x, y) => 0.02 * (nx - x) + 0.15 * Math.sin(1.3 * x + 0.4) * Math.cos(0.9 * y - 0.2) + 0.05 * Math.sin(0.37 * x * y), o);

export const region = (x: number, y: number, width: number, height: number, name = 'Fixture region'): Region => ({ x, y, width, height, name });
export const whole = (g: Grid): Region => region(0, 0, g.nx, g.ny, 'Whole fixture');

/** A short, closed, non-infiltrating storm over the whole grid unless overridden. */
export const baseConfig = (g: Grid, overrides: Partial<Config> = {}): Config => ({
  ...defaultConfig(), rainMmH: 60, rainMinutes: 1, drainMinutes: 0, infiltrationMmH: 0, soilCapacityM: 0.04,
  boundary: 'closed', outputSeconds: 10, roi: whole(g), ...overrides,
});

export const garden = (id: string, x: number, y: number, width = 2, height = 2, overrides: Partial<Patch> = {}): Patch => ({
  id, name: `Fixture garden ${id}`, x, y, width, height, pondingDepthM: 0.15, infiltrationMmH: 25, soilCapacityM: 0.15, ...overrides,
});

export const SOLVER_IDS = ['li-1.0.0', 'dw-1.0.0'] as const;
