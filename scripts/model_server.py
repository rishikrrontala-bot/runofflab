"""Loopback-only local planner inference (MLX). Keeps model loading out of browser code.

Serves one of:
  --model models/base/qwen35-08b                              untuned base model
  --model models/base/qwen35-08b --adapter <local LoRA dir>    local MLX LoRA (scripts/train_local.py)
  --model <merged dir>                                         Adaption LoRA merged into full weights (scripts/merge_lora.py)
The method label comes from recorded provenance, never from a flag: a model is called trained only when its run
record says training completed on >= 1000 qualified rows. Smoke or pilot checkpoints are refused.
"""
import argparse, json, sys, threading, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/base/qwen35-08b'); parser.add_argument('--adapter'); parser.add_argument('--port', type=int, default=8787)
args = parser.parse_args()
model_dir = ROOT / args.model


def identity():
    if args.adapter:   # local MLX LoRA
        run = json.loads((ROOT / args.adapter / 'run.json').read_text())
        if run.get('status') != 'complete' or run.get('train', {}).get('rows', 0) < 1000:
            raise RuntimeError('Refusing to serve an unfinished or small adapter as a trained model')
        return {'model': run['base']['repo'], 'method': 'trained-local', 'checkpoint': args.adapter, 'trainingRows': run['train']['rows'], 'trainingSha256': run['train']['sha256']}
    merge = model_dir / 'runofflab-merge.json'
    if merge.exists():   # Adaption adapter merged into full weights
        record = json.loads(merge.read_text())
        run = json.loads((ROOT / 'models/runs' / record['adapter'].split('adaption-', 1)[1] / 'run.json').read_text())
        if run.get('run', {}).get('status') != 'succeeded' or run.get('dataset', {}).get('rows', 0) < 1000:
            raise RuntimeError('Refusing to serve an unfinished or small Adaption run as a trained model')
        return {'model': run['run']['model'], 'method': 'trained-adaption', 'checkpoint': args.model, 'trainingRows': run['dataset']['rows'],
                'trainingSha256': run['dataset']['sha256'], 'adaptionRun': run['run']['id']}
    revision = json.loads((ROOT / 'models/base/revision.json').read_text()) if model_dir.name == 'qwen35-08b' else {'repo': model_dir.name}
    return {'model': revision['repo'], 'method': 'untuned-local', 'checkpoint': args.model, 'trainingRows': 0}


GENERATION = {'thinking': 'off (enable_thinking=False in the chat template)', 'maxTokens': 96, 'temperature': 0, 'sampler': 'greedy (temp 0)',
              'template': 'tokenizer chat template; one user message with the prompt; add_generation_prompt'}
ident = {**identity(), 'generation': GENERATION}
model, tokenizer = load(str(model_dir), adapter_path=str(ROOT / args.adapter) if args.adapter else None)
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path != '/status': return self.reply(404, {'error': 'Not found'})
        self.reply(200, ident)

    def do_POST(self):
        if self.path != '/infer': return self.reply(404, {'error': 'Not found'})
        if self.headers.get('Host') not in [f'127.0.0.1:{args.port}', f'localhost:{args.port}']: return self.reply(403, {'error': 'Invalid host'})
        if self.headers.get('Origin'): return self.reply(403, {'error': 'Browser requests go through the app server'})
        n = int(self.headers.get('Content-Length', '0'))
        if n < 1 or n > 24000: return self.reply(413, {'error': 'Payload limit exceeded'})
        try:
            prompt = json.loads(self.rfile.read(n))['prompt']
            if not isinstance(prompt, str) or len(prompt) > 20000: raise ValueError()
        except Exception: return self.reply(400, {'error': 'Invalid prompt'})
        if not lock.acquire(blocking=False): return self.reply(429, {'error': 'Local model busy'})
        start = time.time()
        try:
            message = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if len(tokenizer.encode(message)) > 4096: return self.reply(413, {'error': 'Context exceeds limit'})
            # stream_generate is what generate() wraps; iterating it ourselves also gives the stop reason (EVAL-CX-009).
            output, last = '', None
            for last in stream_generate(model, tokenizer, prompt=message, max_tokens=GENERATION['maxTokens'], sampler=make_sampler(temp=GENERATION['temperature'])):
                output += last.text
            self.reply(200, {**ident, 'output': output, 'seconds': time.time() - start, 'outputTokens': last.generation_tokens if last else 0,
                             'finishReason': last.finish_reason if last else None})
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception:
            traceback.print_exc(file=sys.stderr)   # details stay in the server log, not in the HTTP response
            self.reply(500, {'error': 'Local inference failed'})
        finally: lock.release(); mx.clear_cache()


print(json.dumps({'listening': f'http://127.0.0.1:{args.port}', **ident}), flush=True)
# Single-threaded on purpose: MLX streams belong to the thread that loaded the model ("There is no Stream(cpu, 0) in
# current thread" otherwise), and the GPU serves one request at a time anyway. Extra requests wait in the socket backlog.
HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
