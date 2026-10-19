import { cells, type Grid, type Region } from './types';

/**
 * Strictly downhill D8 graph on the unmodified ground: each cell drains to its steepest lower neighbour
 * (ties: first in scan order). No depression filling and no solver output. Water-body cells are terminal.
 * Used only for production-available planner features and simple heuristics, never as a physical result.
 */
export function terrainFeatures(g: Grid, roi: Region) {
  const n = g.z.length, next = new Int32Array(n).fill(-1), area = new Float64Array(n), reaches = new Uint8Array(n);
  const target = new Set(cells(g, roi));
  const order = Array.from({ length: n }, (_, i) => i).filter(i => g.mask[i]).sort((a, b) => g.z[a] - g.z[b] || a - b);
  for (const i of order) {
    area[i] = g.dx * g.dy;
    if (!g.sink?.[i]) {
      const x = i % g.nx, y = Math.floor(i / g.nx);
      let best = 0;
      for (let yy = -1; yy <= 1; yy++) for (let xx = -1; xx <= 1; xx++) {
        if ((!xx && !yy) || x + xx < 0 || x + xx >= g.nx || y + yy < 0 || y + yy >= g.ny) continue;
        const j = (y + yy) * g.nx + x + xx;
        if (!g.mask[j]) continue;
        const slope = (g.z[i] - g.z[j]) / Math.hypot(xx * g.dx, yy * g.dy);
        if (slope > best) { best = slope; next[i] = j; }
      }
    }
    // Processing in rising elevation order means a cell's downhill neighbour is already resolved.
    reaches[i] = target.has(i) || (next[i] >= 0 && reaches[next[i]]) ? 1 : 0;
  }
  for (let k = order.length - 1; k >= 0; k--) { const i = order[k]; if (next[i] >= 0) area[next[i]] += area[i]; }
  let targetUpstreamM2 = 0;
  for (const i of order) if (reaches[i] && !target.has(i)) targetUpstreamM2 += g.dx * g.dy;
  return { next, contributingAreaM2: area, reachesTarget: reaches, targetUpstreamM2, order };
}
