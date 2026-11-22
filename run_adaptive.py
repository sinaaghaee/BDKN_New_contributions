"""Run standalone adaptive methods and refresh the existing offline website.

Edit the settings below, or use --help. No source-repository code is imported.
Completed fits are reused only after checking settings and saved-file hashes.
"""
# --- EDIT SETTINGS HERE ---
DATASETS = ['steps', 'cubic', 'forrester', 'he', 'amplitude-sine', 'linear-sinc',
            'square-wave', 'shifted-sinc', 'sine', 'cosine']
SETTINGS = {'low': {'lr': .001, 'epochs': 2000}, 'high': {'lr': .1, 'epochs': 40}}
METHODS = ['frde', 'wrde', 'forde']
LOCATIONS = ['training', 'random', 'combined']
DEVICE = 'cuda:0'
WORKERS = 'all'  # Start as many selected fits as GPU memory permits; no fixed worker cap.
GPU_MEMORY_PER_FIT_MIB = 450  # Includes the measured ~400 MiB CUDA context per standalone process.
GPU_MEMORY_RESERVE_MIB = 2048
TARGET_RATIO = .8
DUAL_LR = .1
LAMBDA_INITIAL = 0.
OUTPUT = 'results/adaptive'
UPDATE_WEBSITE = True

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_config(method):
    script = ROOT / ('adaptive_' + method + '.py')
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing standalone CONFIG: ' + method)


def make_jobs(args):
    """Read-only job planning: dry-run never creates data or result directories."""
    jobs = []
    for name in args.datasets:
        data = ROOT / 'data' / ('toy_1d.npz' if name == 'steps' else name + '.npz')
        if not data.is_file():
            raise FileNotFoundError(f'Missing {data}; generate the dataset before running')
        for setting in args.settings:
            budget = SETTINGS[setting] | ({'epochs': args.epochs} if args.epochs else {})
            for method in args.methods:
                for location in (['training'] if method == 'wrde' else args.locations):
                    cfg = source_config(method) | dict(lr=budget['lr'], epochs=budget['epochs'],
                        repulsion_location=location, target_ratio=args.target_ratio,
                        dual_lr=args.dual_lr, lambda_initial=args.lambda_initial)
                    if args.stage2_steps is not None:
                        cfg['stage2_steps'] = args.stage2_steps
                    signature = {k: v for k, v in cfg.items() if k not in ('device', 'data_path', 'output_root')}
                    signature.update(dataset_sha256=digest(data),
                        source_sha256=digest(ROOT / ('adaptive_' + method + '.py')))
                    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
                    identifier = f'{name}-{setting}-{method}-{location}-{fingerprint[:16]}'
                    jobs.append(dict(id=identifier, dataset=name, setting=setting, method=method,
                        location=location, fingerprint=fingerprint, signature=signature,
                        data=str(data), folder=str(args.output / 'runs' / identifier), status='pending'))
    return jobs


def validate_fit(job):
    """A completion marker alone is insufficient: verify outputs and their hashes."""
    folder = Path(job['folder'])
    marker = folder / 'complete.json'
    if not marker.is_file():
        return False
    try:
        saved = read_json(marker)
        if saved['fingerprint'] != job['fingerprint']:
            return False
        cfg = read_json(folder / 'config.json')
        if any(cfg.get(k) != v for k, v in job['signature'].items()):
            return False
        if cfg['constraint']['reference_sha256'] != digest(folder / 'reference_inputs.npz'):
            return False
        if not saved['files'] or any(not (folder / k).is_file() or digest(folder / k) != v for k, v in saved['files'].items()):
            return False
        import numpy as np
        rows = read_json(folder / 'metrics.json')
        method = 'adaptive_' + job['method']
        if {r['method'] for r in rows} != {method, method + '_ensemble'}:
            return False
        for row in rows:
            if row['dataset_sha256'] != job['signature']['dataset_sha256']:
                return False
            if not np.isfinite([row['rmse'], row['nll'], row['training_seconds']]).all() or row['training_seconds'] <= 0:
                return False
            with np.load(folder / row['method'] / 'predictions.npz', allow_pickle=False) as p, np.load(job['data'], allow_pickle=False) as d:
                for key in ('x_test', 'y_test', 'x_grid', 'f_grid'):
                    if not np.array_equal(p[key], d[key]):
                        return False
                for split in ('grid', 'test'):
                    for key in ('mean', 'lower', 'upper', 'scale'):
                        if len(p[split + '_' + key]) != len(d['x_' + split]) or not np.isfinite(p[split + '_' + key]).all():
                            return False
                    if np.any(p[split + '_scale'] <= 0) or np.any(p[split + '_lower'] > p[split + '_upper']):
                        return False
        with (folder / 'history.csv').open() as handle:
            history = list(csv.DictReader(handle))
        if not all(np.isfinite(float(value)) for row in history for value in row.values() if value != ''):
            return False
        return (sum(r['stage'] == '1' for r in history) == cfg['epochs']
                and sum(r['stage'] == '2' for r in history) == cfg['stage2_steps'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def mark_complete(job):
    folder = Path(job['folder'])
    method = 'adaptive_' + job['method']
    files = ['config.json', 'history.csv', 'reference_inputs.npz', 'preprocessing.npz',
             'stage1.pt', 'posterior.pt', 'metrics.json']
    files += [f'{m}/predictions.npz' for m in (method, method + '_ensemble')]
    atomic_json(folder / 'complete.json', dict(fingerprint=job['fingerprint'], files={k: digest(folder / k) for k in files}))
    if not validate_fit(job):
        (folder / 'complete.json').unlink()
        raise ValueError('Completed fit failed validation: ' + job['id'])


def command(job, device):
    cfg = job['signature']
    return [sys.executable, str(ROOT / ('adaptive_' + job['method'] + '.py')),
        '--data', job['data'], '--output', job['folder'], '--device', device,
        '--epochs', str(cfg['epochs']), '--learning-rate', str(cfg['lr']),
        '--repulsion-location', job['location'], '--target-ratio', str(cfg['target_ratio']),
        '--dual-lr', str(cfg['dual_lr']), '--lambda-initial', str(cfg['lambda_initial']),
        '--stage2-steps', str(cfg['stage2_steps'])]


def refresh(output):
    subprocess.run([sys.executable, str(ROOT / 'adaptive_report.py'), '--runs', str(output)], check=True, cwd=ROOT)


def concurrency_limit(device, requested, active, remaining, memory_per_fit=None, memory_reserve=None):
    """Account for CUDA contexts that are still starting, not only visible allocations.

    Free memory plus our active allocations is the budget for the entire pool.
    Dividing this total budget reserves memory for every launched process before
    CUDA initialization finishes. Other users' jobs are never signaled or changed.
    """
    if requested != 'all':
        return min(int(requested), remaining)
    if not device.startswith('cuda'):
        return remaining
    index = int(device.split(':')[-1]) if ':' in device else 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = visible.split(',')[index].strip() if visible else str(index)
    result = subprocess.run(['nvidia-smi', '--id=' + physical,
        '--query-gpu=memory.free,uuid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    free, uuid = result.stdout.strip().split(',')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    owned = 0.
    for line in apps.stdout.splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) == 3 and values[0].isdigit() and int(values[0]) in active and values[2] == uuid.strip():
            owned += float(values[1])
    per_fit = GPU_MEMORY_PER_FIT_MIB if memory_per_fit is None else memory_per_fit
    reserve = GPU_MEMORY_RESERVE_MIB if memory_reserve is None else memory_reserve
    capacity = max(0, int((float(free) + owned - reserve) // per_fit))
    return min(capacity, remaining)


def execute(args, jobs, *, validator=None, marker=None, command_builder=None, reporter=None):
    validate = validator or validate_fit
    mark = marker or mark_complete
    build_command = command_builder or command
    refresh_report = reporter or refresh
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An experiment runner already owns this output directory')
        manifest = dict(version=1, pid=os.getpid(), device=args.device, workers=args.workers,
            updated=datetime.now(timezone.utc).isoformat(), status='running', jobs=jobs)
        active = {}
        # Finish the short high-LR jobs first, without changing the requested coverage.
        waiting = sorted(jobs, key=lambda j: (j['signature']['epochs'], j['dataset'], j['method'], j['location']))
        for job in jobs:
            if validate(job):
                job['status'] = 'complete'
        waiting = [j for j in waiting if j['status'] != 'complete']
        def publish():
            manifest['updated'] = datetime.now(timezone.utc).isoformat()
            atomic_json(args.output / 'manifest.json', manifest)
        publish()
        last_report = 0.
        last_capacity = 0.
        limit = 0
        try:
            while waiting or active:
                if time.monotonic() - last_capacity >= 5:
                    limit = concurrency_limit(args.device, args.workers, active, len(waiting) + len(active),
                        getattr(args, 'memory_per_fit', None), getattr(args, 'memory_reserve', None))
                    if manifest.get('concurrency_limit') != limit:
                        print(f'CONCURRENCY {limit} fits on {args.device} (requested: {args.workers})', flush=True)
                    manifest['concurrency_limit'] = limit
                    last_capacity = time.monotonic()
                    publish()
                while waiting and len(active) < limit:
                    ready = next((j for j in waiting if time.time() >= j.get('retry_after', 0)), None)
                    if ready is None:
                        break
                    job = ready
                    waiting.remove(job)
                    folder = Path(job['folder'])
                    if folder.exists():
                        attempts = args.output / 'attempts'
                        attempts.mkdir(exist_ok=True)
                        shutil.move(str(folder), str(attempts / (job['id'] + '-' + str(time.time_ns()))))
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    log = (folder.parent / (job['id'] + '.log')).open('w')
                    env = os.environ | {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                    child = subprocess.Popen(build_command(job, args.device), stdout=log, stderr=subprocess.STDOUT,
                        cwd=ROOT, env=env)
                    active[child.pid] = (child, log, job)
                    job.update(status='running', pid=child.pid, started=datetime.now(timezone.utc).isoformat())
                    print('START ' + job['id'], flush=True)
                    publish()
                memory_retry_in_pass = False
                for pid, (child, log, job) in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[pid]
                    try:
                        if code:
                            raise RuntimeError(f'Trainer exited {code}; see runs/{job["id"]}.log')
                        mark(job)
                        job['status'] = 'complete'
                    except Exception as exc:
                        job.update(status='failed', error=str(exc))
                        if getattr(args, 'retry_memory_errors', False):
                            log_path = Path(job['folder']).parent / (job['id'] + '.log')
                            error_text = log_path.read_text(errors='replace').lower() if log_path.exists() else ''
                            if any(term in error_text for term in ('cuda error: out of memory', 'cuda out of memory', 'cublas_status_alloc_failed', 'cudaerrormemoryallocation')):
                                job.update(status='pending', retry_after=time.time()+60,
                                    oom_retries=job.get('oom_retries',0)+1)
                                memory_retry_in_pass = True
                                waiting.append(job)
                                last_capacity = 0.
                    print(job['status'].upper() + ' ' + job['id'], flush=True)
                    publish()
                    last_report = 0.
                if memory_retry_in_pass:
                    # One shared allocation event can fail several children.
                    # Increase the pool estimate once per pass, not exponentially
                    # once per failed child.
                    args.memory_per_fit = int(getattr(args, 'memory_per_fit', GPU_MEMORY_PER_FIT_MIB) * 1.5)
                if args.update_website and time.monotonic() - last_report >= 60:
                    try:
                        refresh_report(args.output)
                        manifest.pop('report_error', None)
                    except subprocess.CalledProcessError as exc:
                        manifest['report_error'] = str(exc)
                        print('REPORT ERROR: ' + str(exc), flush=True)
                    last_report = time.monotonic()
                    publish()
                if active or waiting:
                    time.sleep(1)
            manifest['status'] = 'failed' if any(j['status'] == 'failed' for j in jobs) else 'complete'
        except BaseException:
            manifest['status'] = 'interrupted'
            raise
        finally:
            for child, log, job in active.values():
                child.terminate()
            for child, log, job in active.values():
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                log.close()
                job['status'] = 'pending'
            publish()
            if args.update_website:
                refresh_report(args.output)
        if manifest['status'] == 'failed':
            raise SystemExit('Some fits failed; inspect manifest.json and run logs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--locations', nargs='+', choices=LOCATIONS, default=LOCATIONS)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--workers', default=WORKERS, help="all (GPU memory admission), or a positive concurrency limit")
    parser.add_argument('--target-ratio', type=float, default=TARGET_RATIO)
    parser.add_argument('--dual-lr', type=float, default=DUAL_LR)
    parser.add_argument('--lambda-initial', type=float, default=LAMBDA_INITIAL)
    parser.add_argument('--epochs', type=int, help='Override both budgets for smoke checks')
    parser.add_argument('--stage2-steps', type=int, help='Override Bayesian budget for smoke checks')
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--no-website', dest='update_website', action='store_false', default=UPDATE_WEBSITE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    import math
    if args.workers != 'all':
        try:
            args.workers = int(args.workers)
        except (ValueError, TypeError):
            parser.error('workers must be all or a positive integer')
    if (args.workers != 'all' and args.workers < 1) or (args.epochs is not None and args.epochs < 1) or (args.stage2_steps is not None and args.stage2_steps < 1):
        parser.error('Worker and epoch counts must be positive')
    if not all(math.isfinite(v) for v in (args.target_ratio, args.dual_lr, args.lambda_initial)) or not 0 < args.target_ratio < 1 or args.dual_lr <= 0 or args.lambda_initial < 0:
        parser.error('Invalid controller settings')
    for values in (args.datasets, args.settings, args.methods, args.locations):
        if len(values) != len(set(values)):
            parser.error('Selections must not contain duplicates')
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.report_only:
        refresh(args.output)
        return
    jobs = make_jobs(args)
    print(f'{len(jobs)} adaptive fits; {args.workers} workers on {args.device}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(job['id'])
        return
    if args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available() or int(args.device.split(':')[-1]) >= torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device unavailable; no silent CPU fallback')
    # A background SIGTERM must also stop this runner's own GPU children.
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Adaptive runner interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    execute(args, jobs)


if __name__ == '__main__':
    main()
