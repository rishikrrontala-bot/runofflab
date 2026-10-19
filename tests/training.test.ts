import {describe, it, expect} from 'vitest';
import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {buildCase, syntheticFamilies, CASE_WINDOW_CELLS} from '../src/core/trainingCases';
import {acceptModelText, designKey, plannerPrompt, searchBest, greedy, PROMPT_VERSION} from '../src/core/planner';
import {runSync} from '../src/core/solver';
import {cells, validateConfig, validatePatches, SOLVER_VERSION} from '../src/core/types';
import {loadFamilies} from '../scripts/families';

const families = loadFamilies();
const manifest = JSON.parse(readFileSync('data/families-manifest.json', 'utf8'));

describe('training families', () => {
  it('fixes synthetic splits before generation (10 / 3 / 3) and is deterministic', () => {
    const a = syntheticFamilies(), b = syntheticFamilies();
    expect(a.map(f => f.split)).toEqual([...Array(10).fill('train'), ...Array(3).fill('validation'), ...Array(3).fill('heldout')]);
    expect(a[3].grid.z).toEqual(b[3].grid.z);
    expect(a.every(f => f.grid.name.includes('synthetic'))).toBe(true);
  });
  it('assigns each real family to exactly one split and excludes the showcase', () => {
    const ids = manifest.families.map((f: {id: string}) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.some((id: string) => id.includes('mercer'))).toBe(false);
    expect(Object.keys(manifest.excluded)).toContain('mercer-county-park');
    const counts = manifest.families.reduce((m: Record<string, number>, f: {split: string}) => ({...m, [f.split]: (m[f.split] ?? 0) + 1}), {});
    expect(counts).toEqual({train: 15, validation: 4, heldout: 4});
  });
  it('loads 23 real and 16 synthetic families with valid 8 m grids', () => {
    expect(families.filter(f => f.source === 'real')).toHaveLength(23);
    expect(families.filter(f => f.source === 'synthetic')).toHaveLength(16);
    for (const f of families) { expect(f.grid.nx).toBe(64); expect(f.grid.dx).toBe(8); expect(f.grid.z.every(Number.isFinite)).toBe(true); }
  });
});

describe('case construction', () => {
  const real = families.find(f => f.source === 'real' && f.split === 'train')!;
  it('rebuilds identical cases from (family, variation)', () => {
    const a = buildCase(real, 7), b = buildCase(real, 7);
    expect(a).toEqual(b);
    expect(plannerPrompt(a.grid, a.config, a.candidates)).toBe(plannerPrompt(b.grid, b.config, b.candidates));
  });
  it('produces valid, non-overlapping candidates outside the target with shuffled unique IDs', () => {
    for (const v of [0, 1, 2, 3]) {
      const c = buildCase(real, v);
      validateConfig(c.grid, c.config);
      expect(c.grid.nx).toBe(CASE_WINDOW_CELLS);
      expect(c.candidates.length).toBeGreaterThanOrEqual(6);
      expect(c.candidates.length).toBeLessThanOrEqual(28);
      expect(new Set(c.candidates.map(p => p.id)).size).toBe(c.candidates.length);
      const target = new Set(cells(c.grid, c.config.roi)), used = new Set<number>();
      for (const p of c.candidates) {
        validatePatches(c.grid, c.config, [p]);
        for (const i of cells(c.grid, p)) { expect(target.has(i)).toBe(false); expect(used.has(i)).toBe(false); used.add(i); }
      }
    }
  });
  it('keeps solver outputs and labels out of the prompt', () => {
    const c = buildCase(real, 3), prompt = plannerPrompt(c.grid, c.config, c.candidates);
    for (const p of c.candidates) expect(prompt.split('\n').filter(l => l.startsWith(`${p.id},`))).toHaveLength(1);
    expect(prompt).not.toMatch(/objective|label|completion|peak ponded volume was|greedy/i);
  });
  it('labels with a search that is never worse than greedy and records the greedy design', async () => {
    const f = families.find(f => f.source === 'synthetic' && f.split === 'train')!;
    const c = buildCase(f, 2), config = {...c.config, maxPatches: 2, areaBudgetM2: 512, rainMinutes: 10, drainMinutes: 5};
    const pool = c.candidates.slice(0, 8), evaluate = async (p: typeof pool) => runSync(c.grid, config, p, {snapshots: false});
    const [search, plan] = await Promise.all([searchBest(c.grid, config, pool, evaluate), greedy(c.grid, config, pool, evaluate)]);
    expect(search.best.objective).toBeLessThanOrEqual(plan.run.objective + 1e-9);
    expect(search.evaluated.map(e => designKey(e.ids))).toContain(designKey(plan.proposal.selected_patch_ids));
    expect(acceptModelText(JSON.stringify({selected_patch_ids: search.best.patches.map(p => p.id)}), c.grid, config, pool).selected_patch_ids.length).toBeLessThanOrEqual(2);
  });
});

describe('generated dataset (when present)', () => {
  const dir = 'models/data/cases-v2';
  const rows = existsSync(dir) ? readdirSync(dir).filter(f => f.endsWith('.json')).map(f => JSON.parse(readFileSync(`${dir}/${f}`, 'utf8')))
    .filter(r => r.solver === SOLVER_VERSION && r.promptVersion === PROMPT_VERSION) : [];
  it.skipIf(!rows.length)('stored prompts match a fresh rebuild and labels are feasible stored outcomes', () => {
    const byId = new Map(families.map(f => [f.id, f]));
    for (const r of rows.filter((_, i) => i % Math.max(1, Math.floor(rows.length / 6)) === 0).slice(0, 6)) {
      const c = buildCase(byId.get(r.family)!, r.variation);
      expect(plannerPrompt(c.grid, c.config, c.candidates)).toBe(r.prompt);
      const label = JSON.parse(r.completion).selected_patch_ids as string[];
      acceptModelText(r.completion, c.grid, c.config, c.candidates);
      const stored = r.outcomes.find((o: {ids: string[]}) => designKey(o.ids) === designKey(label));
      expect(stored.objectiveM3).toBeCloseTo(r.objectiveM3, 12);
      expect(Math.min(...r.outcomes.map((o: {objectiveM3: number}) => o.objectiveM3))).toBeCloseTo(r.objectiveM3, 12);
    }
  });
  it.skipIf(!rows.length)('never places a family in more than one split', () => {
    const splitOf = new Map<string, Set<string>>();
    for (const r of rows) splitOf.set(r.family, (splitOf.get(r.family) ?? new Set()).add(r.split));
    for (const s of splitOf.values()) expect(s.size).toBe(1);
  });
});
