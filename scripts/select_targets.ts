/**
 * Reproducible target selection, run before any garden is simulated on the current solver and domain.
 * Rule: among 10 x 6 and 6 x 10 windows inside the study site whose cells are at least half aerial-classified
 * pavement (and contain no masked or water cells), rank by baseline peak ponded volume under the default storm
 * (from the saved 30 s states). Keep the top windows that do not overlap a better-ranked one.
 * Output: evidence/target-selection.json.
 */
import {writeFileSync, mkdirSync} from 'node:fs';
import {preparedSite, SURFACE} from '../src/core/site';
import {runSync} from '../src/core/solver';
import {cells, defaultConfig, type Region} from '../src/core/types';

const g = preparedSite() as ReturnType<typeof preparedSite> & {surface: number[]};
const c = defaultConfig(), area = g.dx * g.dy, site = g.site!;
const run = runSync(g, c, [], {method: 'target selection baseline'});
const windows: (Region & {pavedFraction: number; peakVolumeM3: number; peakTimeS: number})[] = [];
for (const [w, h] of [[10, 6], [6, 10]]) for (let y = site.y; y + h <= site.y + site.height; y++) for (let x = site.x; x + w <= site.x + site.width; x++) {
  const r = {x, y, width: w, height: h, name: ''}, ii = cells(g, r);
  if (ii.some(i => !g.mask[i] || g.sink?.[i])) continue;
  const pavedFraction = ii.filter(i => g.surface[i] === SURFACE.paved).length / ii.length;
  if (pavedFraction < .5) continue;
  let peakVolumeM3 = 0, peakTimeS = 0;
  for (const f of run.frames) { const v = ii.reduce((s, i) => s + f.h[i] * area, 0); if (v > peakVolumeM3) { peakVolumeM3 = v; peakTimeS = f.t; } }
  windows.push({...r, pavedFraction, peakVolumeM3, peakTimeS});
}
windows.sort((a, b) => b.peakVolumeM3 - a.peakVolumeM3 || a.y - b.y || a.x - b.x);
const chosen: typeof windows = [];
for (const w of windows) {
  const taken = new Set(chosen.flatMap(r => cells(g, r)));
  if (cells(g, w).some(i => taken.has(i))) continue;
  chosen.push(w); if (chosen.length === 5) break;
}
mkdirSync('evidence', {recursive: true});
writeFileSync('evidence/target-selection.json', JSON.stringify({timestamp: new Date().toISOString(), solver: c.solver, rule: 'see scripts/select_targets.ts header',
  storm: {rainMmH: c.rainMmH, rainMinutes: c.rainMinutes, drainMinutes: c.drainMinutes}, candidateWindows: windows.length, chosen, top20: windows.slice(0, 20)}, null, 2) + '\n');
console.log(JSON.stringify(chosen, null, 1));
