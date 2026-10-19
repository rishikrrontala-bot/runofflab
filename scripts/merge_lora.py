"""Merge a PEFT LoRA adapter (as downloaded from Adaption) into full base weights for local serving.

For every adapted linear layer: W' = W + (lora_alpha / r) * (B @ A), computed in float32 and stored in the base
dtype (bf16). Non-adapted tensors are copied unchanged. The merged directory keeps the base config/tokenizer and
records provenance in runofflab-merge.json. MLX loads the result directly (mlx_lm.load(<dir>)).

  .venv/bin/python scripts/merge_lora.py --base models/base/qwen35-4b \
      --adapter models/checkpoints/adaption-<run> --out models/checkpoints/merged-<run>
"""
import argparse, hashlib, json, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True); p.add_argument('--adapter', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    base, adapter, out = (ROOT / a.base), (ROOT / a.adapter), (ROOT / a.out)
    cfg = json.loads((adapter / 'adapter_config.json').read_text())
    if cfg.get('use_rslora') or cfg.get('use_dora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern'):
        raise SystemExit('Only plain LoRA (no rsLoRA/DoRA/per-module ranks) is supported')
    scale = cfg['lora_alpha'] / cfg['r']
    # 'base_model.model.<path>.lora_A.weight' -> '<path>.weight'
    lora = {}
    with safe_open(adapter / 'adapter_model.safetensors', 'pt') as f:
        for key in f.keys():
            stem = key.removeprefix('base_model.model.')
            target, part = stem.rsplit('.lora_', 1)
            lora.setdefault(target + '.weight', {})[part.split('.')[0]] = f.get_tensor(key).float()
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    merged, start = set(), time.time()
    for shard in sorted(set(index['weight_map'].values())):
        tensors = {}
        with safe_open(base / shard, 'pt') as f:
            meta = f.metadata()
            for key in f.keys():
                w = f.get_tensor(key)
                if key in lora:
                    A, B = lora[key]['A'], lora[key]['B']
                    if B.shape[0] != w.shape[0] or A.shape[1] != w.shape[1]: raise SystemExit(f'Shape mismatch for {key}')
                    w = (w.float() + scale * (B @ A)).to(w.dtype); merged.add(key)
                tensors[key] = w.contiguous()
        save_file(tensors, out / shard, metadata=meta or {'format': 'pt'})
    missing = sorted(set(lora) - merged)
    if missing: raise SystemExit(f'{len(missing)} adapter targets not found in base, e.g. {missing[:3]}')
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith('.safetensors'): shutil.copy2(f, out / f.name)
    record = {'base': str(base.relative_to(ROOT)), 'adapter': str(adapter.relative_to(ROOT)), 'scale': scale, 'r': cfg['r'], 'alpha': cfg['lora_alpha'],
              'mergedTensors': len(merged), 'adapterSha256': hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
              'seconds': time.time() - start}
    (out / 'runofflab-merge.json').write_text(json.dumps(record, indent=2) + '\n'); print(json.dumps(record))


if __name__ == '__main__':
    main()
