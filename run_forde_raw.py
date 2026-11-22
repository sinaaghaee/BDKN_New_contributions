"""Matched FoRDE ablation: raw versus normalized input gradients on the 1D data.

Edit this block and execute the file. Each fit runs the self-contained forde_raw.py.
The saved normalized FoRDE configuration supplies every other scientific setting.
"""
# --- EDIT SETTINGS HERE ---
DATASETS = ['steps', 'cubic', 'forrester', 'he', 'amplitude-sine', 'linear-sinc',
            'square-wave', 'shifted-sinc', 'sine', 'cosine']
SETTINGS = ['low', 'high']               # Existing low .001/2000 and high .1/40 settings
WEIGHTS = [.001, .01, .1, 1., 10.]
LOCATIONS = ['training', 'random', 'combined']
DEVICE = 'cuda:0'
WORKERS = 'all'                          # GPU memory admission; positive integer = a cap
MEMORY_PER_FIT_MIB = 512
MEMORY_RESERVE_MIB = 3072
OUTPUT = 'results/forde_raw'
UPDATE_WEBSITE = True

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from run_adaptive import ROOT, read_json, atomic_json, digest, execute


def defaults():
    for node in ast.parse((ROOT / 'forde_raw.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing raw model CONFIG')


def make_jobs(args):
    registry = read_json(ROOT / 'website/experiments.json')
    source = digest(ROOT / 'forde_raw.py')
    jobs = []
    for dataset in args.datasets:
        data = ROOT / 'data' / ('toy_1d.npz' if dataset == 'steps' else dataset + '.npz')
        if not data.is_file():
            raise FileNotFoundError('Missing shared dataset: ' + str(data))
        data_sha = digest(data)
        for setting in args.settings:
            rate = .001 if setting == 'low' else .1
            for weight in args.weights:
                for location in args.locations:
                    entry = next((e for e in registry['experiments'] if e['dataset'] == dataset
                        and float(e['learning_rate']) == rate and float(e['repulsion']) == weight
                        and e.get('repulsion_location', 'training') == location), None)
                    if entry is None:
                        raise ValueError(f'Missing matched normalized profile: {dataset}/{setting}/{weight}/{location}')
                    baseline = ROOT / entry['run'] / 'forde'
                    saved = read_json(baseline / 'config.json')
                    if saved['dataset_sha256'] != data_sha:
                        raise ValueError('Dataset differs from normalized reference: ' + str(baseline))
                    cfg = defaults()
                    cfg.update({k: saved[k] for k in cfg if k in saved and k not in ('device', 'output_root', 'data_path')})
                    cfg.update(lr=rate, repulsion_strength=weight, repulsion_location=location)
                    if args.epochs is not None:
                        cfg['epochs'] = args.epochs
                    if args.stage2_steps is not None:
                        cfg['stage2_steps'] = args.stage2_steps
                    signature = {k: v for k, v in cfg.items() if k not in ('device', 'output_root', 'data_path')}
                    signature.update(source_sha256=source, dataset_sha256=data_sha, gradient_representation='raw')
                    fp = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
                    identifier = f'{dataset}-{setting}-r{weight:g}-{location}-{fp[:16]}'
                    jobs.append(dict(id=identifier, dataset=dataset, setting=setting, weight=weight, location=location,
                        method='forde_raw', signature=signature, fingerprint=fp, data=str(data),
                        folder=str(args.output / 'runs' / identifier), status='pending', config=cfg,
                        normalized_reference=str(baseline), normalized_config_sha256=digest(baseline / 'config.json')))
    return jobs


def validate_fit(job):
    folder = Path(job['folder'])
    try:
        saved = read_json(folder / 'complete.json')
        if saved['fingerprint'] != job['fingerprint'] or not saved['files']:
            return False
        if any(not (folder / k).is_file() or digest(folder / k) != sha for k, sha in saved['files'].items()):
            return False
        cfg = read_json(folder / 'config.json')
        if any(cfg.get(k) != value for k, value in job['signature'].items()):
            return False
        rows = read_json(folder / 'metrics.json')
        if {r['method'] for r in rows} != {'forde_raw', 'forde_raw_ensemble'}:
            return False
        with np.load(job['data'], allow_pickle=False) as data:
            for row in rows:
                if row['dataset_sha256'] != cfg['dataset_sha256'] or not np.isfinite([row['rmse'], row['nll'], row['training_seconds']]).all():
                    return False
                with np.load(folder / row['method'] / 'predictions.npz', allow_pickle=False) as p:
                    for key in ('x_grid', 'f_grid', 'x_test', 'y_test'):
                        if not np.array_equal(p[key], data[key]):
                            return False
                    for part in ('grid', 'test'):
                        for key in ('mean', 'lower', 'upper', 'scale'):
                            if not np.isfinite(p[part + '_' + key]).all() or len(p[part + '_' + key]) != len(data['x_' + part]):
                                return False
                        if np.any(p[part + '_scale'] <= 0) or np.any(p[part + '_lower'] > p[part + '_upper']):
                            return False
        with (folder / 'history.csv').open() as f:
            history = list(csv.DictReader(f))
        if not all(np.isfinite(float(v)) for r in history for v in r.values() if v != ''):
            return False
        return sum(r['stage'] == '1' for r in history) == cfg['epochs'] and sum(r['stage'] == '2' for r in history) == cfg['stage2_steps']
    except (OSError, ValueError, KeyError, TypeError):
        return False


def mark_complete(job):
    folder = Path(job['folder'])
    files = ['config.json', 'history.csv', 'metrics.json', 'stage1.pt', 'posterior.pt', 'preprocessing.npz',
             'forde_raw/predictions.npz', 'forde_raw_ensemble/predictions.npz']
    atomic_json(folder / 'complete.json', dict(fingerprint=job['fingerprint'], files={k: digest(folder / k) for k in files}))
    if not validate_fit(job):
        (folder / 'complete.json').unlink()
        raise ValueError('Raw-gradient fit failed saved-artifact validation')


def command(job, device):
    if digest(ROOT / 'forde_raw.py') != job['signature']['source_sha256']:
        raise ValueError('Raw model source changed during this selection')
    config_path = Path(job['folder']).parent / (job['id'] + '.settings.json')
    atomic_json(config_path, job['config'])
    cfg = job['signature']
    return [sys.executable, str(ROOT / 'forde_raw.py'), '--config', str(config_path), '--data', job['data'],
        '--output', job['folder'], '--device', device, '--learning-rate', str(cfg['lr']),
        '--epochs', str(cfg['epochs']), '--stage2-steps', str(cfg['stage2_steps']),
        '--repulsion-strength', str(job['weight']), '--repulsion-location', job['location']]


def refresh(output):
    subprocess.run([sys.executable, str(ROOT / 'forde_raw_report.py'), '--runs', str(output)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--settings', nargs='+', choices=['low', 'high'], default=SETTINGS)
    parser.add_argument('--weights', nargs='+', type=float, default=WEIGHTS)
    parser.add_argument('--locations', nargs='+', choices=LOCATIONS, default=LOCATIONS)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--workers', default=WORKERS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--stage2-steps', type=int)
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--no-website', dest='update_website', action='store_false', default=UPDATE_WEBSITE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.workers != 'all':
        try:
            args.workers = int(args.workers)
        except ValueError:
            parser.error('workers must be all or a positive integer')
        if args.workers < 1:
            parser.error('workers must be positive')
    if (args.epochs is not None and args.epochs < 1) or (args.stage2_steps is not None and args.stage2_steps < 1):
        parser.error('Epoch budgets must be positive')
    if not args.weights or any(not np.isfinite(w) or w < 0 for w in args.weights):
        parser.error('Weights must be finite and nonnegative')
    for values in (args.datasets, args.settings, args.weights, args.locations):
        if len(values) != len(set(values)):
            parser.error('Selections must be distinct')
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.report_only:
        refresh(args.output)
        return
    jobs = make_jobs(args)
    print(f'{len(jobs)} matched raw-gradient fits on {args.device}; normalized results are saved references only', flush=True)
    if args.dry_run:
        return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device unavailable; no CPU fallback')
    args.memory_per_fit = MEMORY_PER_FIT_MIB
    args.memory_reserve = MEMORY_RESERVE_MIB
    args.retry_memory_errors = True
    import signal
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    execute(args, jobs, validator=validate_fit, marker=mark_complete, command_builder=command, reporter=refresh)


if __name__ == '__main__':
    main()
