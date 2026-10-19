import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defaultConfig, type Config, type Frame, type Grid, type Patch, type Run } from '../src/core/types';
import { promptSignature, type SavedModelProposal } from '../src/core/savedHybrid';
import { plannerPrompt } from '../src/core/planner';
import { terrainSignature } from '../src/core/scenario';
import type { Reply, Request } from '../src/workers/simulation.worker';

const mocks = vi.hoisted(() => ({ simulate: vi.fn(), runSync: vi.fn(), heuristic: vi.fn() }));
vi.mock('../src/core/solver', () => ({ simulate: mocks.simulate, runSync: mocks.runSync }));
vi.mock('../src/core/planner', async importOriginal => ({
  ...await importOriginal<typeof import('../src/core/planner')>(), heuristic: mocks.heuristic,
}));

const grid: Grid = {id:'toy',name:'toy',nx:8,ny:8,dx:4,dy:4,z:Array(64).fill(0),mask:Array(64).fill(1),eligible:Array(64).fill(1)};
const model: Patch = {id:'M',name:'Model',x:1,y:1,width:1,height:1,pondingDepthM:.15,infiltrationMmH:25,soilCapacityM:.15};
const downhill: Patch = {...model,id:'D',name:'Downhill',x:3,y:2};
const config: Config = {...defaultConfig(),rainMinutes:1,drainMinutes:0,roi:{x:5,y:5,width:1,height:1,name:'Target'}};
const saved: SavedModelProposal = {
  config,proposal:{selected_patch_ids:['M']},model:'Qwen/Qwen3.5-4B',method:'trained-adaption',
  recordedAt:'2026-10-18T02:38:24.084Z',promptVersion:'runofflab-planner-prompt-5',
  checkpoint:'models/checkpoints/selected',trainingRows:4000,solver:'li-1.0.0',
  promptSignature:promptSignature(plannerPrompt(grid,config,[model,downhill])),
};
const request = (): Request => ({id:93,grid,config,patches:[model],candidates:[model,downhill],mode:'saved-hybrid',method:'Hybrid',savedProposal:structuredClone(saved)});
const balance = {initial:0,rain:0,surface:0,soil:0,garden:0,outflow:0,waterBody:0,residual:0,relativeResidual:0};

function fixtureRun(c: Config, patches: Patch[], objective: number, marker: number, method: string): Run {
  const frame = (t: number): Frame => ({t,h:new Float32Array([marker]),balance:{...balance},roiDepth:marker,roiVolume:objective,roiExcess:objective});
  return {config:structuredClone(c),gridId:grid.id,patches:structuredClone(patches),frames:[frame(0),frame(60)],objective,
    peakRoiDepth:marker,peakRoiVolume:objective,peakDomainVolume:objective,balance:{...balance},wallMs:1,steps:1,minStableDt:1,method};
}

async function runWorker(input: Request, objectives: [number, number] = [9,7], failSecond = false) {
  const replies: Reply[] = [], runs: Run[] = [];
  mocks.simulate.mockImplementation((_g: Grid, c: Config, patches: Patch[], options: {method:string}) => {
    const index = runs.length;
    // A history must remain usable while the other design runs; early transfer would detach its typed arrays.
    if (index === 1) expect(runs[0].frames[0].h.byteLength).toBeGreaterThan(0);
    const run = fixtureRun(c,patches,objectives[index],index ? .2 : .1,options.method); runs.push(run);
    return (function* () {
      yield run.frames[0];
      if (index === 1 && failSecond) throw Error('Second simulation failed');
      yield run.frames[1];
      return run;
    })();
  });
  const scope: {onmessage?: (event: MessageEvent<Request>) => Promise<void>; postMessage: (message: Reply, transfer?: Transferable[]) => void} = {
    // Emulate actual postMessage transfers so detached histories cannot accidentally pass this verification.
    postMessage: (message,transfer = []) => { replies.push(structuredClone(message,{transfer})); },
  };
  vi.stubGlobal('self',scope);
  await import('../src/workers/simulation.worker');
  await scope.onmessage!({data:input} as MessageEvent<Request>);
  return {replies,runs,complete:replies.find((r): r is Extract<Reply,{type:'complete'}> => r.type === 'complete')};
}

beforeEach(() => { vi.resetModules(); vi.clearAllMocks(); mocks.heuristic.mockReturnValue([downhill]); });
afterEach(() => { vi.unstubAllGlobals(); });

describe('saved model + downhill worker flow (mocked solver)', () => {
  it('runs exactly twice, retains the winning history and exports both component objectives/provenance', async () => {
    const {replies,runs,complete} = await runWorker(request());
    expect(mocks.simulate).toHaveBeenCalledTimes(2); expect(mocks.runSync).not.toHaveBeenCalled();
    expect(replies.filter(r => r.type === 'frame' || r.type === 'replay')).toHaveLength(0);
    expect(replies.every(r => r.id === 93)).toBe(true);
    expect(replies.filter(r => r.type === 'hybrid-progress').map(r => r.fraction)).toEqual([0,.5,.5,1]);
    expect(complete?.simulationCalls).toBe(2); expect(complete?.run.patches.map(p => p.id)).toEqual(['D']);
    expect(complete?.run.frames).toHaveLength(2); expect(complete?.run.frames[1].h[0]).toBeCloseTo(.2);
    expect(runs[0].frames[0].h.byteLength).toBeGreaterThan(0); // losing run was never transferred
    expect(runs[1].frames[0].h.byteLength).toBe(0); // winning history transferred once at completion
    expect(complete?.savedHybrid).toMatchObject({
      winner:'downhill',executedSimulations:2,tieRule:'saved-model-on-equal-objective',config,
      source:{kind:'saved-output-not-live-inference',checkpoint:saved.checkpoint,trainingRows:4000,solver:'li-1.0.0',recordedAt:saved.recordedAt},
      components:{savedModel:{gardens:['M'],objectiveM3:9},downhill:{gardens:['D'],objectiveM3:7}},
    });
    expect(complete?.savedHybrid?.terrainSignature).toBe(terrainSignature(grid));
    expect(complete?.savedHybrid).toMatchObject({tie:false,sameDesign:false,promptSignature:saved.promptSignature});
    expect(complete?.run.method).toBe('Saved model answer + downhill rule · downhill rule design kept · 2 live simulations');
    // The downhill rule must see the full candidate pool, and each design must be simulated as itself.
    expect(mocks.heuristic).toHaveBeenCalledWith(grid,config,[model,downhill]);
    expect(mocks.simulate.mock.calls[0][2].map((q: Patch) => q.id)).toEqual(['M']);
    expect(mocks.simulate.mock.calls[1][2].map((q: Patch) => q.id)).toEqual(['D']);
  });

  it.each([[6,7],[7,7]] as [number,number][])('selects the saved model for lower/equal objectives (%s, %s)', async (m,h) => {
    const {complete} = await runWorker(request(),[m,h]);
    expect(mocks.simulate).toHaveBeenCalledTimes(2);
    expect(complete?.savedHybrid?.winner).toBe('saved-model');
    expect(complete?.run.frames[1].h[0]).toBeCloseTo(.1);
    expect(complete?.savedHybrid?.tie).toBe(m === h);
    expect(complete?.run.method).toBe(`Saved model answer + downhill rule · ${m === h ? 'equal peaks; saved model design kept by the tie rule' : 'saved model design kept'} · 2 live simulations`);
  });

  it('still executes two runs when both components choose the same IDs', async () => {
    mocks.heuristic.mockReturnValue([model]);
    const {complete} = await runWorker(request(),[7,7]);
    expect(mocks.simulate).toHaveBeenCalledTimes(2); expect(complete?.savedHybrid?.winner).toBe('saved-model');
    expect(complete?.savedHybrid).toMatchObject({tie:true,sameDesign:true});
    expect(complete?.run.method).toBe('Saved model answer + downhill rule · both chose the same gardens · 2 live simulations');
  });

  it.each([
    ['a different solver', (q: SavedModelProposal) => { q.solver = 'li-0.9'; }, 'Saved proposal records a different solver'],
    ['missing provenance', (q: SavedModelProposal) => { q.recordedAt = '   '; }, 'Saved proposal is missing recordedAt'],
    ['no prompt signature', (q: SavedModelProposal) => { delete q.promptSignature; }, 'Saved proposal has no prompt signature, so the input it answered cannot be checked'],
    ['a different model input', (q: SavedModelProposal) => { q.promptSignature = 'abc'; }, 'Saved proposal was recorded for a different model input (terrain, gardens or prompt changed); it must be re-recorded'],
  ])('rejects a saved answer with %s before any simulation', async (_name, change, error) => {
    const input = request(); change(input.savedProposal!);
    const {replies,complete} = await runWorker(input);
    expect(mocks.simulate).not.toHaveBeenCalled(); expect(mocks.heuristic).not.toHaveBeenCalled(); expect(complete).toBeUndefined();
    expect(replies).toEqual([{id:93,type:'error',error}]);
  });

  it('rejects settings outside the trained scope before any simulation', async () => {
    const wet = request(); wet.config = {...config,initialDepthM:.01}; wet.savedProposal!.config = wet.config;
    expect((await runWorker(wet)).replies).toEqual([{id:93,type:'error',error:expect.stringContaining('dry starting surface')}]);
    expect(mocks.simulate).not.toHaveBeenCalled();
  });

  it('rejects a comparison request that carries no saved answer', async () => {
    const none = request(); delete none.savedProposal;
    expect((await runWorker(none)).replies).toEqual([{id:93,type:'error',error:'A saved model proposal is required for this comparison'}]);
    expect(mocks.simulate).not.toHaveBeenCalled();
  });

  it('refuses a non-finite objective instead of picking a winner', async () => {
    const {replies,complete} = await runWorker(request(),[Number.NaN,7]);
    expect(complete).toBeUndefined(); expect(replies.at(-1)).toEqual({id:93,type:'error',error:'A component returned a non-finite objective'});
  });

  it('rejects invalid saved IDs explicitly, before running either design', async () => {
    const input=request(); input.savedProposal!.proposal.selected_patch_ids=['UNKNOWN'];
    const {replies,complete} = await runWorker(input);
    expect(mocks.simulate).not.toHaveBeenCalled(); expect(mocks.heuristic).not.toHaveBeenCalled(); expect(complete).toBeUndefined();
    expect(replies).toEqual([{id:93,type:'error',error:'Unknown candidate: UNKNOWN'}]);
  });

  it('rejects a saved answer for different settings before any simulation', async () => {
    const input=request(); input.savedProposal!.config.rainMmH++;
    const {replies,complete} = await runWorker(input);
    expect(mocks.simulate).not.toHaveBeenCalled(); expect(complete).toBeUndefined();
    expect(replies).toEqual([{id:93,type:'error',error:'Saved proposal is for a different scenario'}]);
  });

  it('rejects a saved answer from a different prompt version before any simulation', async () => {
    const input=request(); input.savedProposal!.promptVersion='runofflab-planner-prompt-4';
    const {replies,complete} = await runWorker(input);
    expect(mocks.simulate).not.toHaveBeenCalled(); expect(mocks.heuristic).not.toHaveBeenCalled(); expect(complete).toBeUndefined();
    expect(replies).toEqual([{id:93,type:'error',error:'Saved proposal uses a different planner prompt version'}]);
  });

  it('does not promote a partially checked model design if the second run fails', async () => {
    const {replies,complete} = await runWorker(request(),[6,7],true);
    expect(mocks.simulate).toHaveBeenCalledTimes(2); expect(complete).toBeUndefined();
    expect(replies.at(-1)).toEqual({id:93,type:'error',error:'Second simulation failed'});
    expect(replies.some(r => r.type === 'frame' || r.type === 'replay')).toBe(false);
  });
});
