/**
 * Adds promptSignature to saved proposals recorded before the field existed. It signs an entry only if today's prompt
 * for that scenario is byte-identical to the prompt the selected model answered in the site evaluation
 * (evidence/site-scenarios.json) and the saved IDs equal that run's answer; otherwise it refuses. No inference.
 *   node --import tsx scripts/sign_saved_proposals.ts
 */
import {readFileSync, writeFileSync} from 'node:fs';
import {preparedSite, candidates} from '../src/core/site';
import {designKey, plannerPrompt} from '../src/core/planner';
import {promptSignature, sameScenarioConfig} from '../src/core/savedHybrid';

const path = 'public/data/saved-proposals.json';
const saved = JSON.parse(readFileSync(path, 'utf8'));
const scenarios = JSON.parse(readFileSync('evidence/site-scenarios.json', 'utf8')).scenarios;
const site = JSON.parse(readFileSync('evidence/planner-eval/site-adaption-qwen35-4b-e6.json', 'utf8'));
const g = preparedSite();
for (const e of saved.entries) {
  const sc = scenarios.find((s: {config: never}) => sameScenarioConfig(s.config, e.config));
  if (!sc) throw Error(`No site scenario for ${e.config.roi.name}`);
  const prompt = plannerPrompt(g, e.config, candidates(g, e.config.roi));
  if (prompt !== sc.prompt) throw Error(`${sc.key}: current prompt differs from the prompt the model answered; re-record instead`);
  const answered = site.results.find((r: {key: string}) => r.key === sc.key);
  if (designKey(answered.modelIds) !== designKey(e.proposal.selected_patch_ids)) throw Error(`${sc.key}: saved IDs differ from the site evaluation answer`);
  if (e.checkpoint !== site.effective.checkpoint) throw Error(`${sc.key}: checkpoint differs`);
  e.promptSignature = promptSignature(prompt);
  console.log(sc.key, e.proposal.selected_patch_ids.join('+'), e.promptSignature);
}
writeFileSync(path, JSON.stringify(saved, null, 2) + '\n');
