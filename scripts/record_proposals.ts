/**
 * Records the trained planner's answers for the app's standard scenarios (three targets x default storm, plus the
 * site-evaluation storms) into public/data/saved-proposals.json, so a static deployment can show them clearly
 * labelled as saved output. Requires the local model server (scripts/model_server.py) serving a trained model.
 *   node --import tsx scripts/record_proposals.ts [--url http://127.0.0.1:8787]
 */
import {writeFileSync} from 'node:fs';
import {preparedSite, candidates, targets} from '../src/core/site';
import {acceptModelText, plannerPrompt, repairInstruction, PROMPT_VERSION} from '../src/core/planner';
import {defaultConfig, SOLVER_VERSION, type Config} from '../src/core/types';
import {promptSignature} from '../src/core/savedHybrid';

const url = process.argv.find(a => a.startsWith('--url='))?.split('=')[1] ?? 'http://127.0.0.1:8787';
const status = await (await fetch(`${url}/status`)).json();
if (!String(status.method).startsWith('trained')) throw Error(`Refusing to record: the server reports method ${status.method}, not a trained model`);
const g = preparedSite();
const storms = [{}, {rainMmH: 100, rainMinutes: 10, drainMinutes: 10}, {rainMmH: 35, rainMinutes: 45, drainMinutes: 15}];
const entries = [];
for (const roi of targets) for (const storm of storms) {
  const config: Config = {...defaultConfig(), roi, ...storm}, pool = candidates(g, roi), prompt = plannerPrompt(g, config, pool);
  let request = prompt, proposal = null, raw = '';
  for (let attempt = 0; attempt < 2 && !proposal; attempt++) {
    const body: any = await (await fetch(`${url}/infer`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt: request})})).json();
    raw = String(body.output);
    try { proposal = acceptModelText(raw.trim(), g, config, pool); } catch (e) { request = `${prompt}\n${repairInstruction(e)}`; }
  }
  if (!proposal) { console.log('Invalid answer, not recorded:', roi.name, JSON.stringify(storm), raw.slice(0, 80)); continue; }
  entries.push({config, proposal, model: status.model, method: status.method, checkpoint: status.checkpoint, trainingRows: status.trainingRows,
    recordedAt: new Date().toISOString(), promptVersion: PROMPT_VERSION, promptSignature: promptSignature(prompt), solver: SOLVER_VERSION});
  console.log(roi.name, JSON.stringify(storm), proposal.selected_patch_ids.join('+') || 'no gardens');
}
writeFileSync('public/data/saved-proposals.json', JSON.stringify({version: 1, note: 'Recorded answers from a trained planner model; the app reruns each design live and labels it as saved output.', entries}, null, 2) + '\n');
console.log(`Recorded ${entries.length} proposals`);
