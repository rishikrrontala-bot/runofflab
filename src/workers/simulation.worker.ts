/// <reference lib="webworker" />
import { runSync, simulate } from '../core/solver';
import { greedy } from '../core/planner';
import { evaluateSavedHybrid, type HybridComponent, type SavedHybridEvidence, type SavedModelProposal } from '../core/savedHybrid';
import type { Config, Frame, Grid, Patch, Run } from '../core/types';

export type Request = {
  id: number;
  grid: Grid;
  config: Config;
  patches: Patch[];
  candidates: Patch[];
  mode: 'run' | 'greedy' | 'saved-hybrid';
  method: string;
  savedProposal?: SavedModelProposal;
};

export type Reply =
  /** A saved physical state from the run being shown. */
  | { id: number; type: 'frame'; fraction: number; frame: Frame }
  /** Greedy search progress: designs evaluated so far and the most the search can evaluate. */
  | { id: number; type: 'search'; evaluated: number; upperBound: number; elapsedMs: number }
  /** The greedy search finished; its best design is now being replayed with saved states. */
  | { id: number; type: 'replay'; patches: Patch[]; simulationCalls: number; searchMs: number }
  /** Both components retain their histories; progress never presents an unfinished component as the winner. */
  | { id: number; type: 'hybrid-progress'; component: HybridComponent; fraction: number }
  | { id: number; type: 'complete'; run: Run; simulationCalls: number; searchMs?: number; savedHybrid?: SavedHybridEvidence }
  | { id: number; type: 'error'; error: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;

const post = (message: Reply, transfer: Transferable[] = []) => scope.postMessage(message, transfer);

const nextTask = () => new Promise<void>(resolve => setTimeout(resolve, 0));

/** Runs one design, streaming each saved state to the page so the map can follow the computation. */
async function streamRun(id: number, grid: Grid, config: Config, patches: Patch[], method: string, component?: HybridComponent): Promise<Run> {
  const total = (config.rainMinutes + config.drainMinutes) * 60;
  const it = simulate(grid, config, patches, { method });
  let step = it.next();
  while (!step.done) {
    const fraction = total > 0 ? step.value.t / total : 1;
    if (component) {
      post({ id, type: 'hybrid-progress', component, fraction: ((component === 'saved-model' ? 0 : 1) + fraction) / 2 });
    } else {
      const frame: Frame = { ...step.value, h: step.value.h.slice() };
      post({ id, type: 'frame', fraction, frame }, [frame.h.buffer]);
    }
    await nextTask();
    step = it.next();
  }
  return step.value;
}

scope.onmessage = async (event: MessageEvent<Request>) => {
  const { id, grid, config, patches, candidates, mode, method, savedProposal } = event.data;
  try {
    if (mode === 'run') {
      const run = await streamRun(id, grid, config, patches, method);
      post({ id, type: 'complete', run, simulationCalls: 1 }, run.frames.map(f => f.h.buffer));
      return;
    }

    if (mode === 'saved-hybrid') {
      if (!savedProposal) throw Error('A saved model proposal is required for this comparison');
      const { run, evidence } = await evaluateSavedHybrid(grid, config, candidates, savedProposal, (design, component) =>
        streamRun(id, grid, config, design, component === 'saved-model' ? 'Saved model design · live simulation' : 'Downhill heuristic · live simulation', component));
      // Exactly two simulations. The selected run already has snapshots, so there is no replay or third run.
      post({ id, type: 'complete', run, simulationCalls: 2, savedHybrid: evidence }, run.frames.map(f => f.h.buffer));
      return;
    }

    // Greedy: every candidate run keeps only summary numbers (no saved states), so 70+ runs stay small in memory.
    const start = performance.now();
    const evaluate = async (design: Patch[]) => runSync(grid, config, design, { snapshots: false, method });
    const plan = await greedy(grid, config, candidates, evaluate, (evaluated, upperBound) =>
      post({ id, type: 'search', evaluated, upperBound, elapsedMs: performance.now() - start }));
    const searchMs = performance.now() - start;

    // Replay the chosen design once with saved states so its history can be played and scrubbed.
    const chosen = plan.run.patches;
    post({ id, type: 'replay', patches: chosen, simulationCalls: plan.simulationCalls, searchMs });
    const calls = `${plan.simulationCalls} simulation call${plan.simulationCalls === 1 ? '' : 's'}`;
    const run = await streamRun(id, grid, config, chosen, `${plan.method} · ${calls}`);
    post({ id, type: 'complete', run, simulationCalls: plan.simulationCalls, searchMs }, run.frames.map(f => f.h.buffer));
  } catch (error) {
    post({ id, type: 'error', error: error instanceof Error ? error.message : 'Simulation failed' });
  }
};
