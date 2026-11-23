"""D'Angelo paper recipe on every saved 1D dataset; GPU 0 only.

The existing README, low/high and adaptive results are kept. This adds a
separate, fingerprinted result bank, included in the existing author page.
"""
# ---- EDIT SELECTIONS HERE ----
DATASETS = ['author-xsine', 'steps', 'cubic', 'forrester', 'he', 'amplitude-sine',
            'linear-sinc', 'square-wave', 'shifted-sinc', 'sine', 'cosine']
METHODS = ['SGD', 'kde_WGD', 'kde_f_WGD', 'forde', 'control']
GPU = 'cuda:0'
OUTPUT = 'results/paper'
MEMORY_PER_FIT_MIB = 650
MEMORY_RESERVE_MIB = 2048
# Published training recipe; BDKN is our separately fitted extension.
PAPER = dict(steps=15000, lr=.01, n_particles=50, size_hidden=50,
             seed=42, normalize_x=False, batch_size=64)

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
CODE = ROOT/'paper_code'
from author_regression import digest
import run_author as original
import run_official_forde as forde
import run_adaptive as scheduler


def plan(output, datasets, methods):
    jobs = []
    files = {p.name:digest(p) for p in CODE.iterdir() if p.is_file() and not p.name.startswith('test_')
             and (p.suffix in ('.py','.json') or p.name=='requirements-forde.lock.txt')}
    for dataset in datasets:
        data = ROOT/'data'/('toy_1d.npz' if dataset=='steps' else dataset+'.npz')
        if not data.is_file():
            raise FileNotFoundError(data)
        for method in methods:
            if method in ('forde','control'):
                selected = dict(epochs=PAPER['steps'], steps=PAPER['steps'], lr=PAPER['lr'],
                    n_particles=50, hidden_dim=50, seed=42, lr_ratio=1.,
                    normalize_x=False, batch_size=64)
                cfg = {k:v for k,v in (forde.CONFIG|selected|dict(preset='paper',
                    method=method, weight_decay=1.)).items() if k not in ('data','output','device')}
                cfg.update(source_revision=forde.LOCK['revision'], dataset_sha256=digest(data), code=files)
                kind = 'official_forde'
            else:
                cfg = PAPER|dict(epochs=15000, bdkn=True, stage2_steps=100, stage2_lr=.1,
                    dataset_sha256=digest(data), code=files, source_revision=original.LOCK['revision'])
                kind = 'author'
            fingerprint = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            identifier = f'{dataset}-paper-{method.lower()}-{fingerprint[:12]}'
            jobs.append(dict(id=identifier, kind=kind, dataset=dataset, method=method,
                preset='paper', location='training', weight=None, signature=cfg,
                fingerprint=fingerprint, data=str(data), folder=str(output/'runs'/identifier),
                initialization=str(output/'initializations'/(dataset+'.npz')), status='pending'))
    return jobs


def validate(job):
    try:
        check_recipe(job)
        return (original if job['kind']=='author' else forde).validate(job)
    except (OSError,ValueError,KeyError):
        return False


def check_recipe(job):
    cfg=json.loads((Path(job['folder'])/'config.json').read_text())
    batch=cfg['upstream_config']['batch_size'] if job['kind']=='author' else cfg['batch_size']
    if batch!=64 or cfg['seed']!=42 or cfg['n_particles']!=50 or cfg['normalize_x']:
        raise ValueError('Saved fit does not match the paper recipe')


def mark(job):
    check_recipe(job)
    (original if job['kind']=='author' else forde).mark(job)


def command(job, device):
    code = Path(job['code_root'])
    args = ['--data',job['data'],'--output',job['folder'],'--preset','paper',
            '--method',job['method'],'--device',device]
    if job['kind']=='author':
        return [sys.executable,str(code/'author_regression.py')]+args
    return [str(ROOT/'.venv-forde/bin/python'),str(code/'official_forde.py')]+args+[
        '--initialization',job['initialization'],'--torch-python',sys.executable]


def refresh(output):
    subprocess.run([sys.executable,str(ROOT/'author_report.py')],check=True,cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--output',type=Path,default=ROOT/OUTPUT)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    jobs=plan(a.output,a.datasets,a.methods)
    print(f'{len(jobs)} paper-recipe fits selected; GPU 0 only',flush=True)
    if a.dry_run:
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0'):
        raise ValueError('Only physical GPU 0 is authorized')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 0 unavailable')
    # Freeze the adapters; all released source files remain pinned and unchanged.
    version=hashlib.sha256(json.dumps(jobs[0]['signature']['code'],sort_keys=True).encode()).hexdigest()[:16]
    runtime=a.output/'code'/version;runtime.mkdir(parents=True,exist_ok=True)
    for name,sha in jobs[0]['signature']['code'].items():
        target=runtime/name
        if not target.exists():shutil.copy2(CODE/name,target)
        if digest(target)!=sha:raise ValueError('Frozen source mismatch: '+name)
    for cache,lock in [('repulsive_ensembles','author_source.lock.json'),('official_forde','official_forde_source.lock.json')]:
        for name in json.loads((CODE/lock).read_text())['files']:
            target=runtime/'.cache'/cache/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(CODE/'.cache'/cache/name,target)
    compiled=runtime/'.cache/jax_forde'
    if not compiled.exists():compiled.symlink_to(ROOT/'.cache/jax_forde',target_is_directory=True)
    for job in jobs:job['code_root']=str(runtime)
    for dataset in a.datasets:
        job=next(j for j in jobs if j['dataset']==dataset)
        subprocess.run([sys.executable,str(runtime/'prepare_dangelo.py'),'--data',job['data'],
            '--preset','paper','--output',job['initialization']],check=True,cwd=ROOT)
    # Count the actual GPU trainer descendants, including JAX's BDKN subprocess.
    base_limit=scheduler.concurrency_limit
    def limit(device,requested,active,remaining,memory_per_fit=None,memory_reserve=None):
        parents={int(pid):int(ppid) for pid,ppid in (line.split() for line in
            subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True).stdout.splitlines())}
        owned=set(active)
        while True:
            new=owned|{pid for pid,ppid in parents.items() if ppid in owned}
            if new==owned:break
            owned=new
        return base_limit(device,requested,{pid:None for pid in owned},remaining,memory_per_fit,memory_reserve)
    scheduler.concurrency_limit=limit
    args=SimpleNamespace(output=a.output,device=GPU,workers='all',memory_per_fit=MEMORY_PER_FIT_MIB,
        memory_reserve=MEMORY_RESERVE_MIB,update_website=True,retry_memory_errors=True)
    import signal
    def interrupt(signum, frame):raise KeyboardInterrupt('Paper runner stopped')
    signal.signal(signal.SIGTERM,interrupt)
    scheduler.execute(args,jobs,validator=validate,marker=mark,command_builder=command,reporter=refresh)


if __name__=='__main__':main()
