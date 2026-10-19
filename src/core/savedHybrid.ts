import { heuristic, parseProposal, plannerPrompt, plannerUnsupported, PROMPT_VERSION } from './planner';
import { summary, terrainSignature } from './scenario';
import { validateConfig, validatePatches, type Config, type Grid, type Patch, type Proposal, type Run } from './types';

/** A recorded answer, not a request to run model inference. Optional fields retain older saved-file compatibility. */
export interface SavedModelProposal {
  config: Config;
  proposal: Proposal;
  model: string;
  method: string;
  recordedAt: string;
  promptVersion: string;
  /** Signature of the exact prompt text the model answered (terrain, target, candidates and garden assumptions). */
  promptSignature?: string;
  checkpoint?: string;
  trainingRows?: number;
  solver?: string;
}

export type HybridComponent = 'saved-model' | 'downhill';
export interface SavedHybridEvidence {
  version: 1;
  method: 'saved-model-plus-downhill';
  config: Config;
  gridId: string;
  terrainSignature: string;
  promptSignature: string;
  source: Omit<SavedModelProposal, 'config' | 'proposal'> & { kind: 'saved-output-not-live-inference' };
  components: { savedModel: ReturnType<typeof summary>; downhill: ReturnType<typeof summary> };
  winner: HybridComponent;
  /** Equal unrounded objectives; the saved model is kept by the predeclared tie rule. */
  tie: boolean;
  /** Both methods chose the same set of gardens. */
  sameDesign: boolean;
  tieRule: 'saved-model-on-equal-objective';
  executedSimulations: 2;
  wallMs: number;
}

const canonicalKey = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonicalKey).join(',')}]`
  : value && typeof value === 'object' ? `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonicalKey((value as Record<string, unknown>)[k])}`).join(',')}}`
  : JSON.stringify(value);

/** Saved answers are usable only for their exact recorded scenario, independent of object-key ordering. */
export const sameScenarioConfig = (a: Config, b: Config) => canonicalKey(a) === canonicalKey(b);

/** Non-security FNV-1a signature (like terrainSignature) that detects a changed model input, synchronously in the browser. */
export function promptSignature(prompt: string): string {
  let h = 2166136261;
  for (let i = 0; i < prompt.length; i++) { h ^= prompt.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(16);
}

/**
 * A saved answer may be shown only for the exact input the model answered. The config match alone is not enough:
 * garden size and hydraulics, the lawn lattice and the terrain live outside Config, and all of them are in the prompt.
 */
export function checkSavedInput(grid: Grid, config: Config, candidates: Patch[], saved: SavedModelProposal): string {
  if (!sameScenarioConfig(saved.config, config)) throw Error('Saved proposal is for a different scenario');
  if (saved.promptVersion !== PROMPT_VERSION) throw Error('Saved proposal uses a different planner prompt version');
  const signature = promptSignature(plannerPrompt(grid, config, candidates));
  if (!saved.promptSignature) throw Error('Saved proposal has no prompt signature, so the input it answered cannot be checked');
  if (saved.promptSignature !== signature) throw Error('Saved proposal was recorded for a different model input (terrain, gardens or prompt changed); it must be re-recorded');
  return signature;
}

const byId = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true });
const sameIds = (a: string[], b: string[]) => a.length === b.length && [...a].sort(byId).every((id, i) => id === [...b].sort(byId)[i]);

/** Plain-language result of the comparison: states ties and identical designs instead of naming a winner. */
export function hybridOutcome(e: Pick<SavedHybridEvidence, 'winner' | 'tie' | 'sameDesign'>): string {
  if (e.sameDesign) return 'both chose the same gardens';
  if (e.tie) return 'equal peaks; saved model design kept by the tie rule';
  return e.winner === 'saved-model' ? 'saved model design kept' : 'downhill rule design kept';
}

/** Table rows with garden lists in a stable order, so identical sets look identical. */
export function hybridRows(e: SavedHybridEvidence) {
  return [
    { component: 'saved-model' as const, label: 'Trained model (saved answer)', gardens: [...e.components.savedModel.gardens].sort(byId), objectiveM3: e.components.savedModel.objectiveM3 },
    { component: 'downhill' as const, label: 'Downhill rule (computed now)', gardens: [...e.components.downhill.gardens].sort(byId), objectiveM3: e.components.downhill.objectiveM3 },
  ];
}

/** Verify both components exactly once. Both returned runs must retain snapshots; the chosen history is reused. */
export async function evaluateSavedHybrid(
  grid: Grid,
  config: Config,
  candidates: Patch[],
  saved: SavedModelProposal,
  evaluate: (patches: Patch[], component: HybridComponent) => Promise<Run>,
): Promise<{ run: Run; evidence: SavedHybridEvidence }> {
  const start = performance.now();
  validateConfig(grid, config);
  const unsupported = plannerUnsupported(config);
  if (unsupported) throw Error(unsupported);
  if (saved.solver !== undefined && saved.solver !== config.solver) throw Error('Saved proposal records a different solver');
  for (const key of ['model', 'method', 'recordedAt', 'promptVersion'] as const) {
    if (typeof saved[key] !== 'string' || !saved[key].trim()) throw Error(`Saved proposal is missing ${key}`);
  }
  const signature = checkSavedInput(grid, config, candidates, saved);
  // A rejected saved answer stays rejected: this mode does not replace invalid input with the downhill design.
  const modelDesign = parseProposal(saved.proposal, grid, config, candidates);
  const downhillDesign = heuristic(grid, config, candidates);
  validatePatches(grid, config, downhillDesign);

  // Intentionally no memoization, including when the two designs have the same IDs.
  const modelRun = await evaluate(modelDesign, 'saved-model');
  const downhillRun = await evaluate(downhillDesign, 'downhill');
  if (!Number.isFinite(modelRun.objective) || !Number.isFinite(downhillRun.objective)) throw Error('A component returned a non-finite objective');
  const winner: HybridComponent = modelRun.objective <= downhillRun.objective ? 'saved-model' : 'downhill';
  const chosen = winner === 'saved-model' ? modelRun : downhillRun;
  if (!chosen.frames.length) throw Error('The selected simulation did not retain its history');
  const evidence: SavedHybridEvidence = {
    version: 1, method: 'saved-model-plus-downhill', config: structuredClone(config), gridId: grid.id,
    terrainSignature: terrainSignature(grid), promptSignature: signature,
    source: {
      kind: 'saved-output-not-live-inference', model: saved.model, method: saved.method, recordedAt: saved.recordedAt,
      promptVersion: saved.promptVersion, promptSignature: saved.promptSignature, checkpoint: saved.checkpoint, trainingRows: saved.trainingRows, solver: saved.solver,
    },
    components: { savedModel: summary(modelRun), downhill: summary(downhillRun) },
    winner, tie: modelRun.objective === downhillRun.objective,
    sameDesign: sameIds(modelDesign.map(p => p.id), downhillDesign.map(p => p.id)),
    tieRule: 'saved-model-on-equal-objective', executedSimulations: 2, wallMs: performance.now() - start,
  };
  // The label describes the kept design: a downhill design is computed now, so it is never called saved output.
  const method = `Saved model answer + downhill rule · ${hybridOutcome(evidence)} · 2 live simulations`;
  return { run: { ...chosen, method }, evidence };
}
