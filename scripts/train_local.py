"""Local LoRA fine-tuning of the planner on Apple silicon (MLX), Qwen/Qwen3.5-0.8B.

Memory: the stock mlx_lm training path runs Qwen3.5's gated-delta (linear attention) recurrence as a per-timestep
loop in every block, which exhausted GPU memory beyond one adapted block. Two changes fix that without changing
the maths (verified: float32 chunked vs reference gradients agree to ~5e-6, evidence/investigation-2026-10-17/training):
  1. blocks below the first adapted block need no gradient, so they keep the fused Metal kernel;
  2. adapted blocks use a chunk-parallel (WY form) gated-delta implementation (scripts/chunked_gdn.py).
Loss: completion tokens only, computed on the final 96 positions (exact for batch 1; checked at start-up).

  .venv/bin/python scripts/train_local.py --train models/data/v2/train.jsonl --valid models/data/v2/validation.jsonl \
      --output models/checkpoints/local-v2 --layers 8 --epochs 2
"""
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.datasets import CompletionsDataset, CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train, default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from chunked_gdn import make_patched_update  # noqa: E402

BASE = ROOT / 'models/base/qwen35-08b'


def completion_loss(model, batch, lengths):
    inputs = batch[:, :-1]; targets = batch[:, 1:]
    skip = max(0, inputs.shape[1] - 96)
    hidden = model.language_model.model(inputs)[:, skip:]
    text = model.language_model
    logits = text.model.embed_tokens.as_linear(hidden) if text.args.tie_word_embeddings else text.lm_head(hidden)
    steps = mx.arange(skip + 1, targets.shape[1] + 1)
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets[:, skip:]) * mask
    count = mask.sum()
    return ce.astype(mx.float32).sum() / count, count


class Record(TrainingCallback):
    def __init__(self, path): self.path = path
    def record(self, event, value):
        with self.path.open('a') as f: f.write(json.dumps({'event': event, 'timestamp': time.time(), **value}) + '\n')
    def on_train_loss_report(self, info): self.record('train', info)
    def on_val_loss_report(self, info): self.record('validation', info)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--valid', required=True); p.add_argument('--output', required=True)
    p.add_argument('--layers', type=int, default=8); p.add_argument('--rank', type=int, default=8)
    p.add_argument('--epochs', type=float, default=2.0); p.add_argument('--iters', type=int, default=0, help='overrides --epochs when > 0')
    p.add_argument('--learning-rate', type=float, default=1e-4); p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--steps-per-eval', type=int, default=500); p.add_argument('--steps-per-save', type=int, default=500)
    p.add_argument('--val-batches', type=int, default=40); p.add_argument('--mem-limit-gb', type=float, default=12.0)
    a = p.parse_args()
    out = (ROOT / a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.jsonl').unlink(missing_ok=True)

    def watchdog():   # shared machine: stop cleanly rather than swap the system
        while True:
            if mx.get_active_memory() > a.mem_limit_gb * 1e9:
                (out / 'run.json').write_text(json.dumps({'status': 'stopped-memory-limit', 'activeGB': mx.get_active_memory() / 1e9}) + '\n')
                os._exit(3)
            time.sleep(0.1)
    threading.Thread(target=watchdog, daemon=True).start()

    import mlx_lm.models.qwen3_5 as q35
    q35.gated_delta_update = make_patched_update(q35.gated_delta_update, a.chunk)
    np.random.seed(17); mx.random.seed(17)
    model, tokenizer = load(str(BASE))
    model.freeze()
    params = {'rank': a.rank, 'dropout': 0.0, 'scale': 16.0}
    linear_to_lora_layers(model, a.layers, params)
    layers = model.language_model.model.layers
    first = len(layers) - a.layers
    for layer in layers[:first]:
        if getattr(layer, 'is_linear', False):
            m = layer.linear_attn; base = type(m)
            m.__class__ = type('FrozenKernelGDN', (base,), {'_set_training_mode': lambda self, mode, base=base: base._set_training_mode(self, False)})
            m._training = False

    def dataset(path):
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s]
        data = CompletionsDataset(rows, tokenizer, 'prompt', 'completion', True)
        lengths = []
        for i in range(len(data)):
            tokens, offset = data.process(data[i])
            if len(tokens) - offset + 33 > 96: raise ValueError('Completion exceeds exact suffix-loss coverage')
            if len(tokens) > 2048: raise ValueError('Context would be truncated; rejected')
            lengths.append(len(tokens))
        return CacheDataset(data), {'file': str(Path(path)), 'rows': len(data), 'tokensMin': min(lengths), 'tokensMax': max(lengths),
                                    'tokensMean': sum(lengths) / len(lengths), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    train_set, train_meta = dataset(a.train); valid_set, valid_meta = dataset(a.valid)

    short = tokenizer.encode('Return JSON only: {"selected_patch_ids": []}')
    batch = mx.array([short]); length = mx.array([[1, len(short) - 1]])
    standard, _ = default_loss(model, batch, length); compact, _ = completion_loss(model, batch, length); mx.eval(standard, compact)
    if abs(standard.item() - compact.item()) > 1e-5: raise RuntimeError('Completion loss equivalence failed')

    iters = a.iters or int(round(a.epochs * train_meta['rows']))
    revision = json.loads((ROOT / 'models/base/revision.json').read_text())
    config = {'base': revision, 'fine_tune_type': 'lora', 'num_layers': a.layers, 'lora_parameters': params, 'batch_size': 1, 'iters': iters,
              'epochs': iters / train_meta['rows'], 'learning_rate': a.learning_rate, 'optimizer': 'Adam', 'seed': 17, 'chunk': a.chunk,
              'train': train_meta, 'validation': valid_meta, 'loss': 'completion-only, final 96 positions (exact for batch 1)',
              'lossEquivalenceDifference': abs(standard.item() - compact.item())}
    (out / 'adapter_config.json').write_text(json.dumps({**config, 'model': str(BASE.relative_to(ROOT)), 'fine_tune_type': 'lora', 'num_layers': a.layers,
                                                        'lora_parameters': params}, indent=2) + '\n')
    (out / 'run.json').write_text(json.dumps({'status': 'running', **config}, indent=2) + '\n')
    print(json.dumps({'status': 'starting', 'iters': iters, 'train': train_meta, 'validation': valid_meta}), flush=True)
    mx.reset_peak_memory(); start = time.time()
    args = TrainingArgs(batch_size=1, iters=iters, val_batches=a.val_batches, steps_per_report=25, steps_per_eval=a.steps_per_eval,
                        steps_per_save=a.steps_per_save, adapter_file=out / 'adapters.safetensors', max_seq_length=2048, grad_checkpoint=True)
    train(model, optim.Adam(learning_rate=a.learning_rate), train_set, valid_set, args=args, loss=completion_loss, training_callback=Record(out / 'metrics.jsonl'))
    report = {'status': 'complete', 'seconds': time.time() - start, 'peakMemoryGB': mx.get_peak_memory() / 1e9, **config}
    (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
