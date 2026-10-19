import raw from '../../public/data/park-domain.json';
import { cells, type Grid, type Patch, type Region } from './types';

/** Building footprints traced from the 2015 NJ aerial (domain cells). Masked cells get no rain and have closed
 *  faces: roof runoff and drainage are not modelled, and water cannot flow through a building. */
export const buildings: Region[] = [
  { x: 66, y: 40, width: 5, height: 5, name: 'Picnic-loop building' },
  { x: 84, y: 37, width: 5, height: 6, name: 'White-roofed building east of the site' },
];

/** Surface classes from the aerial colour classification in scripts/prepare_data.py. */
export const SURFACE = { other: 0, paved: 1, lawn: 2 } as const;

/** The 384 m computational domain: the 256 m study site plus a 64 m buffer measured to capture the
 *  runoff that reaches the site's targets from outside it. */
export function preparedSite(): Grid {
  const g = structuredClone(raw) as unknown as Grid & { surface: number[] };
  g.name = 'Mercer County Park · west picnic loop';
  for (const b of buildings) for (const i of cells(g, b)) { g.mask[i] = 0; g.surface[i] = SURFACE.other; }
  const site = g.site!;
  const inSite = (i: number) => { const x = i % g.nx, y = Math.floor(i / g.nx); return x >= site.x && x < site.x + site.width && y >= site.y && y < site.y + site.height; };
  // Gardens may go on aerial-classified lawn inside the study site. Eligible is not permission or soil suitability.
  g.eligible = g.mask.map((m, i) => m && !g.sink?.[i] && g.surface[i] === SURFACE.lawn && inSite(i) ? 1 : 0);
  return g;
}

/** Paved inspection targets (domain cells): the top three non-overlapping windows from the fixed rule in
 *  scripts/select_targets.ts (at least half pavement, ranked by baseline peak ponded volume), chosen before
 *  any garden was simulated. Result: evidence/target-selection.json. */
export const targets: Region[] = [
  { x: 47, y: 68, width: 10, height: 6, name: 'South loop target' },
  { x: 65, y: 53, width: 6, height: 10, name: 'East parking target' },
  { x: 46, y: 52, width: 10, height: 6, name: 'North parking target' },
];

export const GARDEN = { sizeM: 16, pondingDepthM: 0.15, infiltrationMmH: 25, soilCapacityM: 0.15 };

/**
 * Candidate rain gardens by a fixed rule, with no simulation involved: every 16 x 16 m window on a lattice
 * of that size whose cells are all eligible, using the lattice offset that fits the most windows. Windows that
 * overlap the inspected target are left out. IDs run G1, G2, ... in north-to-south, west-to-east order.
 * The same rule (with case-specific garden parameters) generates training-case candidates.
 */
export function candidates(g: Grid, roi?: Region, garden: Omit<typeof GARDEN, 'sizeM'> = GARDEN): Patch[] {
  const size = Math.max(1, Math.round(GARDEN.sizeM / g.dx)), target = new Set(roi ? cells(g, roi) : []);
  const fits = (x: number, y: number) => x > 0 && y > 0 && x + size < g.nx && y + size < g.ny
    && cells(g, { x, y, width: size, height: size, name: '' }).every(i => g.eligible?.[i] && !target.has(i));
  let best: { x: number; y: number }[] = [];
  for (let oy = 0; oy < size; oy++) for (let ox = 0; ox < size; ox++) {
    const windows: { x: number; y: number }[] = [];
    for (let y = oy; y < g.ny; y += size) for (let x = ox; x < g.nx; x += size) if (fits(x, y)) windows.push({ x, y });
    if (windows.length > best.length) best = windows;
  }
  return best.map(({ x, y }, k) => ({ id: `G${k + 1}`, name: `Rain garden ${k + 1}`, x, y, width: size, height: size,
    pondingDepthM: garden.pondingDepthM, infiltrationMmH: garden.infiltrationMmH, soilCapacityM: garden.soilCapacityM }));
}
