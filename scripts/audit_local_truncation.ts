/**
 * EVAL-CX-009, after the fact: local reports made before model_server.py reported finishReason recorded
 * truncatedAttempts = 0 without being able to detect truncation. This reads every stored local attempt and checks it
 * directly: generated length against the server's effective cap, and whether the text is a complete JSON object.
 * Reads reports only; no inference.
 *   node --import tsx scripts/audit_local_truncation.ts
 */
import {readdirSync, readFileSync, writeFileSync} from 'node:fs';

const DIR = 'evidence/planner-eval';
const reports = readdirSync(DIR).filter(f => /^v2-(validation|heldout)-.*\.json$/.test(f) && !f.endsWith('.summary.json')).sort();
const rows = [];
for (const file of reports) {
  const d = JSON.parse(readFileSync(`${DIR}/${file}`, 'utf8'));
  if (!String(d.llm ?? '').startsWith('local:')) continue;
  const cap: number | null = d.generation?.effective?.generation?.maxTokens ?? null;
  const attempts = d.results.filter((r: {method: string}) => r.method.startsWith('llm:')).flatMap((r: {attempts?: {raw: string; outputTokens: number}[]}) => r.attempts ?? []);
  const complete = (raw: string) => { try { return typeof JSON.parse(raw.trim()) === 'object'; } catch { return false; } };
  rows.push({report: file, split: d.split, checkpoint: d.generation?.effective?.checkpoint ?? null, cap, attempts: attempts.length,
    maxOutputTokens: Math.max(...attempts.map((a: {outputTokens: number}) => a.outputTokens)),
    atOrAboveCap: cap === null ? null : attempts.filter((a: {outputTokens: number}) => a.outputTokens >= cap).length,
    notCompleteJson: attempts.filter((a: {raw: string}) => !complete(a.raw)).length});
}
const out = {checkedAt: new Date().toISOString(), method: 'outputTokens re-encodes the returned text with the model tokenizer; a truncated answer would reach the cap and usually not parse as JSON.',
  allClear: rows.every(r => r.atOrAboveCap === 0), reports: rows};
writeFileSync('evidence/local-truncation-audit.json', JSON.stringify(out, null, 2) + '\n');
for (const r of rows) console.log(r.report, `attempts ${r.attempts}, max ${r.maxOutputTokens}/${r.cap} tokens, at cap ${r.atOrAboveCap}, not JSON ${r.notCompleteJson}`);
console.log('allClear', out.allClear);
