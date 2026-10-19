import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { designKey, heuristic, parseProposal } from '../src/core/planner';
import { candidates, preparedSite } from '../src/core/site';
import { runSync } from '../src/core/solver';
import { checkSavedInput, hybridRows, type SavedModelProposal } from '../src/core/savedHybrid';
import type { Reply, Request } from '../src/workers/simulation.worker';

// No mocks: the real worker, solver, heuristic and shipped saved answers, driven the way App.tsx drives them.
const saved: SavedModelProposal[] = JSON.parse(readFileSync('public/data/saved-proposals.json', 'utf8')).entries;
const grid = preparedSite();

async function compare(entry: SavedModelProposal) {
  const pool = candidates(grid, entry.config.roi), replies: Reply[] = [];
  const scope: {onmessage?: (event: MessageEvent<Request>) => Promise<void>; postMessage: (m: Reply, t?: Transferable[]) => void} = {
    postMessage: (m, t = []) => { replies.push(structuredClone(m, {transfer: t})); },
  };
  vi.stubGlobal('self', scope);
  vi.resetModules();
  await import('../src/workers/simulation.worker');
  const patches = parseProposal(entry.proposal, grid, entry.config, pool);
  await scope.onmessage!({data: {id: 1, grid, config: entry.config, patches, candidates: pool, mode: 'saved-hybrid', method: 'Saved model answer + downhill rule',
    savedProposal: structuredClone(entry)}} as MessageEvent<Request>);
  const complete = replies.find((r): r is Extract<Reply, {type: 'complete'}> => r.type === 'complete');
  if (!complete?.savedHybrid) throw Error(JSON.stringify(replies.find(r => r.type === 'error')));
  return {pool, complete, evidence: complete.savedHybrid};
}
afterEach(() => { vi.unstubAllGlobals(); });

describe('saved model + downhill on the real site (no mocks)', () => {
  it('every shipped saved answer matches the exact prompt the model answered', () => {
    expect(saved).toHaveLength(9);
    for (const e of saved) expect(() => checkSavedInput(grid, e.config, candidates(grid, e.config.roi), e)).not.toThrow();
  });

  it('South loop, default storm: the downhill design wins and is labelled as the downhill rule', async () => {
    const entry = saved.find(e => e.config.roi.name.startsWith('South') && e.config.rainMmH === 60)!;
    const {pool, complete, evidence} = await compare(entry);
    const downhill = heuristic(grid, entry.config, pool);
    expect(evidence.components.downhill.gardens).toEqual(downhill.map(p => p.id));
    expect(evidence.components.savedModel.gardens).toEqual(entry.proposal.selected_patch_ids);
    const exact = (ids: string[]) => runSync(grid, entry.config, ids.map(id => pool.find(p => p.id === id)!), {snapshots: false}).objective;
    expect(evidence.components.downhill.objectiveM3).toBe(exact(downhill.map(p => p.id)));
    expect(evidence.components.savedModel.objectiveM3).toBe(exact(entry.proposal.selected_patch_ids));
    expect(evidence).toMatchObject({winner: 'downhill', tie: false, sameDesign: false, executedSimulations: 2});
    expect(complete.run.patches.map(p => p.id)).toEqual(downhill.map(p => p.id));   // the kept history is the downhill run
    expect(complete.run.objective).toBe(evidence.components.downhill.objectiveM3);
    expect(complete.run.method).toBe('Saved model answer + downhill rule · downhill rule design kept · 2 live simulations');
    expect(complete.run.frames.length).toBeGreaterThan(30);
    expect(hybridRows(evidence).map(r => r.gardens.join(','))).toEqual(['G7,G8,G12', 'G7,G11,G12']);
  });

  it('East parking, default storm: both choose the same gardens, and the result says so', async () => {
    const entry = saved.find(e => e.config.roi.name.startsWith('East') && e.config.rainMmH === 60)!;
    const {complete, evidence} = await compare(entry);
    expect(designKey(evidence.components.downhill.gardens)).toBe(designKey(evidence.components.savedModel.gardens));
    expect(evidence).toMatchObject({winner: 'saved-model', tie: true, sameDesign: true});
    expect(complete.run.method).toBe('Saved model answer + downhill rule · both chose the same gardens · 2 live simulations');
    const [a, b] = hybridRows(evidence);
    expect(a.gardens).toEqual(b.gardens);   // identical sets display identically
  });
});
