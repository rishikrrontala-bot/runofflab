/**
 * Local app + planner endpoint (loopback only). Serves dist/ and POST /api/planner, which builds the shared
 * planner prompt on the server, asks the local model server (scripts/model_server.py, port 8787) for a proposal,
 * validates it with the same rules as the app, and allows at most one repair request. Invalid proposals are
 * reported as invalid; nothing is substituted. Every model request is logged to models/spend.jsonl.
 *   npm run build && npm run serve
 */
import {createServer} from 'node:http';
import {readFileSync, existsSync, appendFileSync} from 'node:fs';
import {resolve, extname, sep} from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
import {preparedSite, candidates} from '../src/core/site';
import {validateConfig, type Config} from '../src/core/types';
import {acceptModelText, plannerPrompt, plannerUnsupported, repairInstruction, PROMPT_VERSION} from '../src/core/planner';

const port = Number(process.env.PORT ?? 4173), modelUrl = 'http://127.0.0.1:8787', root = resolve('dist'), grid = preparedSite();
const mime: Record<string, string> = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.woff2': 'font/woff2', '.woff': 'font/woff'};
const allowedOrigins = ['http://127.0.0.1:5173', 'http://localhost:5173', `http://127.0.0.1:${port}`, `http://localhost:${port}`];
let busy = false, lastRequest = 0;

createServer(async (req, res) => {
  const send = (status: number, value: unknown) => {
    res.writeHead(status, {'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'});
    res.end(JSON.stringify(value));
  };
  if (![`127.0.0.1:${port}`, `localhost:${port}`].includes(req.headers.host ?? '')) return send(403, {error: 'Invalid host'});
  const path = new URL(req.url ?? '/', `http://127.0.0.1:${port}`).pathname;

  if (path === '/api/status') {
    try { const r = await fetch(`${modelUrl}/status`, {signal: AbortSignal.timeout(2000)}); return send(r.status, await r.json()); }
    catch { return send(503, {error: 'Local model is not running'}); }
  }

  if (path === '/api/planner') {
    if (req.method !== 'POST') return send(405, {error: 'POST required'});
    if (req.headers.origin && !allowedOrigins.includes(req.headers.origin)) return send(403, {error: 'Origin not allowed'});
    if (!req.headers['content-type']?.startsWith('application/json')) return send(415, {error: 'JSON required'});
    // Claim the single model slot synchronously, before any await, so two requests cannot both pass the check.
    if (busy || Date.now() - lastRequest < 1500) return send(429, {error: 'Model busy; try again shortly'});
    busy = true; lastRequest = Date.now();
    try {
      let body = '';
      try { for await (const chunk of req) { body += chunk.toString(); if (Buffer.byteLength(body) > 20000) return send(413, {error: 'Request exceeds 20 kB'}); } }
      catch { return send(400, {error: 'Invalid request body'}); }
      let config: Config;
      try { config = JSON.parse(body).config; validateConfig(grid, config); } catch { return send(400, {error: 'Invalid scenario'}); }
      const unsupported = plannerUnsupported(config);
      if (unsupported) return send(422, {error: unsupported});
      const available = candidates(grid, config.roi), id = randomUUID();
      const prompt = plannerPrompt(grid, config, available), hash = createHash('sha256').update(prompt).digest('hex');
      let request = prompt, lastError = '';
      for (let attempt = 0; attempt < 2; attempt++) {
        let result: any;
        try {
          const r = await fetch(`${modelUrl}/infer`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt: request}), signal: AbortSignal.timeout(30000)});
          if (!r.ok) { await r.text(); return send(r.status === 429 ? 429 : 503, {error: 'Provider unavailable. The local model did not accept the request.'}); }
          result = await r.json();
        } catch { return send(503, {error: 'Provider unavailable. Start the local model server; greedy planning remains available.'}); }
        let proposal, error: unknown;
        try { proposal = acceptModelText(String(result.output).trim(), grid, config, available); } catch (e) { error = e; lastError = e instanceof Error ? e.message : 'Invalid model response'; }
        appendFileSync('models/spend.jsonl', JSON.stringify({service: 'local-mlx', operation: attempt ? 'repair' : 'planner', timestamp: new Date().toISOString(), model: result.model,
          dataset_hash: hash, job_id: id, quoted_cost: 0, actual_cost: 0, unit: 'USD', status: proposal ? 'accepted' : 'invalid', output_tokens: result.outputTokens, seconds: result.seconds, prompt_version: PROMPT_VERSION}) + '\n');
        if (proposal) return send(200, {proposal, model: result.model, method: result.method, checkpoint: result.checkpoint, requests: attempt + 1, promptVersion: PROMPT_VERSION});
        request = `${prompt}\n${repairInstruction(error)}`;
      }
      return send(422, {error: `Invalid model response after one repair attempt: ${lastError}`});
    } finally { busy = false; }
  }

  if (req.method !== 'GET') return send(405, {error: 'GET required'});
  let decoded: string;
  try { decoded = decodeURIComponent(path === '/' ? '/index.html' : path); } catch { return send(400, {error: 'Malformed path'}); }
  const file = resolve(root, '.' + decoded);
  if (!file.startsWith(root + sep) || !existsSync(file)) return send(404, {error: 'Not found'});
  try { const content = readFileSync(file); res.writeHead(200, {'Content-Type': mime[extname(file)] ?? 'application/octet-stream', 'X-Content-Type-Options': 'nosniff'}); res.end(content); }
  catch { send(404, {error: 'Not found'}); }
}).listen(port, '127.0.0.1', () => console.log(`RunoffLab local app/API: http://127.0.0.1:${port}`));
