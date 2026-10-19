"""Adaption AutoScientist orchestration for RunoffLab planner training.

Every network action appends a redacted record to models/spend.jsonl and a run
record under models/runs/<name>/. The API key is read from the ignored
.env.local and never printed. AutoScientist has no pre-launch quote endpoint, so
cost is measured as the change in available credits (read with an
estimate-only request that creates and charges nothing).

Usage (from the project root):
  .venv/bin/python scripts/adaption_runs.py balance
  .venv/bin/python scripts/adaption_runs.py upload models/data/<file>.jsonl --name <run-name>
  .venv/bin/python scripts/adaption_runs.py recommend --name <run-name> [--model Qwen/Qwen3.5-0.8B]
  .venv/bin/python scripts/adaption_runs.py launch --name <run-name> --model <id> [--max-iterations 1] [--budget 400]
  .venv/bin/python scripts/adaption_runs.py status --name <run-name>
  .venv/bin/python scripts/adaption_runs.py download --name <run-name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'models/runs'
LEDGER = ROOT / 'models/spend.jsonl'
CHECKPOINTS = ROOT / 'models/checkpoints'


def load_key() -> None:
    env = ROOT / '.env.local'
    if not os.environ.get('ADAPTION_API_KEY') and env.exists():
        for line in env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                if key.strip() == 'ADAPTION_API_KEY':
                    os.environ['ADAPTION_API_KEY'] = value.strip().strip('"').strip("'")
    if not os.environ.get('ADAPTION_API_KEY'):
        sys.exit('ADAPTION_API_KEY is not configured in .env.local')


def client():
    load_key()
    from adaption import Adaption
    return Adaption()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger(**entry) -> None:
    record = {'service': 'adaption', 'timestamp': now(), 'model': None, 'dataset_hash': None, 'job_id': None,
              'quoted_cost': None, 'actual_cost': None, 'unit': 'platform_credits', **entry}
    with LEDGER.open('a') as f:
        f.write(json.dumps(record) + '\n')


def run_dir(name: str) -> Path:
    if not name.replace('-', '').replace('_', '').isalnum():
        sys.exit('Run name may contain letters, digits, - and _ only')
    path = RUNS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_record(name: str) -> dict:
    path = run_dir(name) / 'run.json'
    return json.loads(path.read_text()) if path.exists() else {'name': name}


def write_record(name: str, record: dict) -> None:
    (run_dir(name) / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')


def available_credits(c) -> float:
    """Estimate-only invent request: returns the team's available credits, creates nothing."""
    response = c.datasets.invent(rows=10, estimate=True, dataset_prompt='Estimate only: balance check.')
    return float(response.available_credits)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_balance(args) -> None:
    c = client()
    credits = available_credits(c)
    ledger(operation='balance-check', status='ok', actual_cost=0, available_credits=credits)
    print(json.dumps({'available_credits': credits}))


def cmd_upload(args) -> None:
    path = (ROOT / args.file).resolve()
    if not path.is_file() or ROOT not in path.parents:
        sys.exit(f'Dataset must be a file inside the project: {args.file}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(set(r) != {'prompt', 'completion'} for r in rows):
        sys.exit('Every row must contain exactly prompt and completion')
    for r in rows:
        json.loads(r['completion'])  # completions are solver-derived JSON, never free text
    digest = sha256(path)
    c = client()
    from adaption.lib.upload import do_upload_file
    response = do_upload_file(c.datasets, c.datasets.upload, path, name=args.name, processing_mode='raw',
                              column_mapping={'prompt': 'prompt', 'completion': 'completion'})
    dataset_id = response.dataset_id if hasattr(response, 'dataset_id') else json.loads(response.model_dump_json()).get('dataset_id')
    status = None
    for _ in range(90):
        status = json.loads(c.datasets.get(dataset_id).model_dump_json())
        if status.get('status') in ('succeeded', 'failed', 'awaiting_input'):
            break
        time.sleep(5)
    record = read_record(args.name)
    record.update({'dataset': {'file': str(path.relative_to(ROOT)), 'sha256': digest, 'rows': len(rows), 'dataset_id': dataset_id,
                               'status': status, 'uploaded_at': now(), 'processing_mode': 'raw', 'visibility': 'private (not published)'}})
    write_record(args.name, record)
    ledger(operation='dataset-upload', status=(status or {}).get('status'), actual_cost=0, dataset_hash=digest, job_id=dataset_id, rows=len(rows))
    print(json.dumps({'dataset_id': dataset_id, 'status': (status or {}).get('status'), 'rows': len(rows)}))


def cmd_recommend(args) -> None:
    record = read_record(args.name)
    c = client()
    kwargs = {'dataset_id': record['dataset']['dataset_id']}
    if args.model:
        kwargs['model'] = args.model
    value = json.loads(c.autoscientist.recommend_hyperparams(**kwargs).model_dump_json())
    record.setdefault('recommendations', []).append({'at': now(), 'model': args.model, 'value': value})
    write_record(args.name, record)
    print(json.dumps(value, indent=1))


def cmd_launch(args) -> None:
    record = read_record(args.name)
    if record.get('run', {}).get('id'):
        sys.exit(f"Run already launched: {record['run']['id']}")
    if args.dataset_from:
        # Reuse a dataset already uploaded under another run name (same bytes, same checksum).
        record['dataset'] = read_record(args.dataset_from)['dataset']
    if 'dataset' not in record:
        sys.exit('No uploaded dataset for this run name; upload first or pass --dataset-from')
    c = client()
    before = available_credits(c)
    if before < args.budget:
        sys.exit(f'Available credits {before} are below the per-run budget guard {args.budget}')
    # Raw ingest stores uploaded columns as original_prompt / original_completion (verified by
    # downloading the processed rows and comparing them byte-for-byte with the local file).
    params = {'dataset_id': record['dataset']['dataset_id'], 'model': args.model, 'training_method': 'instruction',
              'data_format': args.data_format, 'column_mapping': {'prompt': 'original_prompt', 'completion': 'original_completion'},
              'max_iterations': args.max_iterations, 'idempotency_key': f'runofflab-{args.name}'}
    if args.hyperparams:
        # AutoScientist's defaults trained the diagnostics for only 22 optimizer steps (1 epoch, batch 'max');
        # a structured selection task needs more updates, so main runs override epochs, batch and learning rate.
        params['hyperparams'] = json.loads(args.hyperparams)
    run = json.loads(c.autoscientist.create(**params).model_dump_json())
    record.update({'run': run, 'launch': {'at': now(), 'params': params, 'credits_before': before, 'budget_guard': args.budget}})
    write_record(args.name, record)
    ledger(operation='autoscientist-launch', status=run.get('status'), model=run.get('model'), job_id=run.get('id'),
           dataset_hash=record['dataset']['sha256'], available_credits_before=before, max_iterations=args.max_iterations)
    print(json.dumps({'run_id': run.get('id'), 'model': run.get('model'), 'status': run.get('status'), 'credits_before': before}))


def cmd_status(args) -> None:
    record = read_record(args.name)
    c = client()
    run = json.loads(c.autoscientist.get(record['run']['id']).model_dump_json())
    record['run'] = run
    terminal = run.get('status') in ('succeeded', 'failed', 'cancelled')
    if terminal and 'credits_after' not in record.get('launch', {}):
        after = available_credits(c)
        record['launch']['credits_after'] = after
        record['launch']['measured_cost'] = record['launch']['credits_before'] - after
        ledger(operation='autoscientist-complete', status=run.get('status'), model=run.get('model'), job_id=run.get('id'),
               dataset_hash=record['dataset']['sha256'], actual_cost=record['launch']['measured_cost'], available_credits_after=after,
               note='actual_cost = credits before launch minus credits after completion; concurrent spending would distort it')
    write_record(args.name, record)
    keys = ('id', 'status', 'model', 'iterations_completed', 'max_iterations', 'best_win_rate', 'download_available', 'error')
    print(json.dumps({**{k: run.get(k) for k in keys}, 'measured_cost': record.get('launch', {}).get('measured_cost')}, default=str))


def cmd_download(args) -> None:
    record = read_record(args.name)
    c = client()
    run = json.loads(c.autoscientist.get(record['run']['id']).model_dump_json())
    if not run.get('download_available'):
        sys.exit('Artifact is not available for download yet')
    out = CHECKPOINTS / f'adaption-{args.name}'
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'artifact.tar.gz'
    if not archive.exists():
        c.autoscientist.download(run['id']).write_to_file(archive)
    # Observed: the service streams a Zstandard-compressed tar despite the gzip media type.
    if archive.read_bytes()[:4] == b'\x28\xb5\x2f\xfd':
        import io, zstandard
        with archive.open('rb') as f:
            stream = io.BytesIO(zstandard.ZstdDecompressor().stream_reader(f).read())
        tar = tarfile.open(fileobj=stream, mode='r:')
    else:
        tar = tarfile.open(archive)
    with tar:
        for member in tar.getmembers():
            target = (out / member.name).resolve()
            if out.resolve() not in target.parents and target != out.resolve():
                sys.exit(f'Unsafe path in artifact: {member.name}')
        tar.extractall(out, filter='data')
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    record['artifact'] = {'path': str(out.relative_to(ROOT)), 'archive_sha256': sha256(archive), 'files': files, 'downloaded_at': now(),
                          'best_hyperparams': run.get('best_hyperparams')}
    write_record(args.name, record)
    ledger(operation='artifact-download', status='ok', model=run.get('model'), job_id=run['id'], actual_cost=0)
    print(json.dumps(record['artifact'], indent=1, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('balance')
    p = sub.add_parser('upload'); p.add_argument('file'); p.add_argument('--name', required=True)
    p = sub.add_parser('recommend'); p.add_argument('--name', required=True); p.add_argument('--model')
    p = sub.add_parser('launch'); p.add_argument('--name', required=True); p.add_argument('--model', required=True)
    p.add_argument('--max-iterations', type=int, default=1); p.add_argument('--budget', type=float, default=400)
    p.add_argument('--dataset-from', help='run name whose uploaded dataset to reuse')
    p.add_argument('--data-format', choices=['chat', 'instruction'], default='chat')
    p.add_argument('--hyperparams', help='JSON object of AutoScientist hyperparameter overrides')
    p = sub.add_parser('status'); p.add_argument('--name', required=True)
    p = sub.add_parser('download'); p.add_argument('--name', required=True)
    args = parser.parse_args()
    globals()[f'cmd_{args.command}'](args)


if __name__ == '__main__':
    main()
