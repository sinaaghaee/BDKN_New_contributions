"""Run the pinned author comparison on GPU 0; edit selections below, then run.

All selected fits are submitted together. The existing memory-aware scheduler
starts concurrent fits, retries CUDA allocation failures and updates the site.
"""
# --- EDIT SETTINGS HERE ---
DATASETS = ['author-xsine', 'steps', 'cubic', 'forrester', 'he', 'amplitude-sine',
            'linear-sinc', 'square-wave', 'shifted-sinc', 'sine', 'cosine']
PRESETS = ['upstream', 'low', 'high']
AUTHOR_METHODS = ['SGD', 'kde_WGD', 'kde_f_WGD']
ORIGINAL_METHODS = ['bdkn', 'deep_ensemble', 'frde', 'wrde', 'forde', 'gp', 'dkl', 'ldbll', 'vbll']
WEIGHTS = [.001, .01, .1, 1., 10.]
LOCATIONS = ['training', 'random', 'combined']
DEVICE = 'cuda:0'
WORKERS = 'all'  # Concurrent admission based on GPU memory, not a two-worker queue.
MEMORY_PER_FIT_MIB = 600
MEMORY_RESERVE_MIB = 2048
OUTPUT = 'results/author'

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
from author_regression import digest, settings, LABELS
from author_code import LOCK, prepare_source
from run_adaptive import execute, atomic_json, read_json
from run_suite import source_config

# One audited metadata-only reader repair. Keep completed, unaffected results
# with their ORIGINAL code fingerprint, rather than relabeling or retraining.
METADATA_REPAIR = (
    'e93256d8ce9efe7f3df63226f55dbe9e34026c0728f2e1717960d26f404c601d',
    'a0ef6901d68b38725a77934ba9db8931d4c4e91f62de9c93ad2700c7a85206c6',
)


def compatible_metadata_repair(old, new):
    old=json.loads(json.dumps(old));new=json.loads(json.dumps(new))
    previous=old.get('code',{}).get('author_regression.py')
    current=new.get('code',{}).get('author_regression.py')
    if (previous,current)!=METADATA_REPAIR: return False
    old['code']['author_regression.py']=current
    return old==new


def plan_jobs(args):
    """Read-only planning; do not generate data or create output directories."""
    jobs = []
    previous=read_json(args.output/'manifest.json')['jobs'] if (args.output/'manifest.json').exists() else []
    for dataset in args.datasets:
        path = ROOT/'data'/('toy_1d.npz' if dataset=='steps' else dataset+'.npz')
        if not path.is_file(): raise FileNotFoundError(path)
        import numpy as np
        with np.load(path, allow_pickle=False) as data: n = len(data['x_train'])
        def add(kind, method, preset, cfg, weight=None, location='training'):
            signature = dict(cfg, dataset_sha256=digest(path))
            scripts = ['author_regression.py', 'author_code.py', 'author_source.lock.json', 'bdkn.py'] if kind=='author' else [('vbll_regression' if method=='vbll' else method)+'.py']
            signature['code'] = {script:digest(ROOT/script) for script in scripts}
            if kind=='author': signature['source_revision'] = LOCK['revision']
            fingerprint = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            suffix = '' if weight is None else f'-r{weight:g}-{location}'
            identifier = f'{dataset}-{kind}-{method.lower()}-{preset}{suffix}-{fingerprint[:12]}'
            job=dict(id=identifier,kind=kind,dataset=dataset,method=method,preset=preset,
                location=location,weight=weight,signature=signature,fingerprint=fingerprint,
                data=str(path),folder=str(args.output/'runs'/identifier),status='pending')
            if kind=='author':
                for old in previous:
                    if (old['kind'],old['dataset'],old['method'],old['preset'])==(kind,dataset,method,preset) and compatible_metadata_repair(old['signature'],signature) and validate(old):
                        job=old.copy()
                        job['compatibility_note']='Completed before metadata-only steps-reader fix; original implementation fingerprint retained'
                        break
            jobs.append(job)
        for preset in args.presets:
            selected = settings(preset,n)
            for method in args.methods:
                add('author',method,preset,selected|dict(epochs=selected['steps'],bdkn=True,stage2_steps=100,stage2_lr=.1))
        if dataset=='author-xsine' and not args.no_originals:
            for preset in [p for p in args.presets if p!='upstream']:
                for method in ORIGINAL_METHODS:
                    cfg = source_config(method)
                    epochs = (10000 if method in ('dkl','ldbll') else 2000) if preset=='low' else 40
                    cfg.update(epochs=epochs)
                    if preset=='high' or method!='gp': cfg['lr'] = .001 if preset=='low' else .1
                    if method in ('frde','wrde','forde'):
                        for weight in WEIGHTS:
                            for location in (['training'] if method=='wrde' else LOCATIONS):
                                add('original',method,preset,cfg|dict(repulsion_strength=weight,repulsion_location=location),weight,location)
                    else:
                        add('original',method,preset,cfg)
    return jobs


def artifacts_valid(job):
    import numpy as np
    folder = Path(job['folder'])
    cfg = read_json(folder/'config.json')
    if cfg['dataset_sha256'] != job['signature']['dataset_sha256']: return False
    if job['kind']=='author':
        if cfg['smoke_test'] or cfg['source_revision']!=LOCK['revision']: return False
        if cfg['method']!=job['method'] or cfg['preset']!=job['preset']: return False
        for key in ('steps','lr','n_particles','size_hidden','seed','normalize_x','bdkn','stage2_steps','stage2_lr'):
            if cfg[key]!=job['signature'][key]: return False
        expected = {LABELS[job['method']],LABELS[job['method']]+'_bdkn'}
        for path in ('stage1.pt','posterior.pt','members.npz','native_uncertainty.npz','source.json'):
            if not (folder/path).is_file(): return False
        expected_epochs = cfg['steps']
    else:
        if job['method'] in ('frde','forde') and cfg.get('repulsion_location','training')!=job['location']: return False
        for key,value in job['signature'].items():
            if key not in ('code','device','data_path','output_root','repulsion_location') and cfg.get(key)!=value: return False
        expected = {job['method'],job['method']+'_ensemble'} if job['method'] in ('frde','wrde','forde') else {job['method']}
        expected_epochs = cfg['epochs']
        checkpoint = 'stage1.pt' if job['method'] in ('bdkn','deep_ensemble','frde','wrde','forde') else 'model.pt'
        if not (folder/checkpoint).is_file(): return False
        if job['method'] in ('bdkn','frde','wrde','forde') and not (folder/'posterior.pt').is_file(): return False
    rows = read_json(folder/'metrics.json')
    if {r['method'] for r in rows}!=expected: return False
    with np.load(job['data'],allow_pickle=False) as data:
        for row in rows:
            if row['dataset_sha256']!=cfg['dataset_sha256'] or not np.isfinite([row['rmse'],row['nll'],row['training_seconds']]).all(): return False
            with np.load(folder/row['method']/'predictions.npz',allow_pickle=False) as pred:
                for key in ('x_test','y_test','x_grid','f_grid'):
                    if not np.array_equal(pred[key],data[key]): return False
                for split in ('test','grid'):
                    for key in ('mean','scale','lower','upper'):
                        a=pred[split+'_'+key]
                        if len(a)!=len(data['x_'+split]) or not np.isfinite(a).all(): return False
                    if np.any(pred[split+'_scale']<=0): return False
    with (folder/'history.csv').open() as handle: history=list(csv.DictReader(handle))
    if sum(r.get('stage','1')=='1' for r in history)!=expected_epochs: return False
    if job['kind']=='author' or job['method'] in ('bdkn','frde','wrde','forde'):
        if sum(r.get('stage')=='2' for r in history)!=cfg['stage2_steps']: return False
    return True


def validate(job):
    folder = Path(job['folder'])
    try:
        marker=read_json(folder/'complete.json')
        return (marker['fingerprint']==job['fingerprint'] and
                all((folder/path).is_file() and digest(folder/path)==sha for path,sha in marker['files'].items()) and artifacts_valid(job))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def mark(job):
    if not artifacts_valid(job): raise ValueError('Result validation failed: '+job['id'])
    folder=Path(job['folder'])
    files={str(path.relative_to(folder)):digest(path) for path in folder.rglob('*')
           if path.is_file() and path.suffix in ('.json','.npz','.pt','.csv') and path.name!='complete.json'}
    atomic_json(folder/'complete.json',dict(fingerprint=job['fingerprint'],files=files))


def command(job, device):
    common=['--data',job['data'],'--output',job['folder'],'--device',device]
    if job['kind']=='author':
        return [sys.executable,str(ROOT/'author_regression.py'),'--method',job['method'],'--preset',job['preset']]+common
    script='vbll_regression.py' if job['method']=='vbll' else job['method']+'.py'
    cfg=job['signature']
    cmd=[sys.executable,str(ROOT/script)]+common+['--epochs',str(cfg['epochs']),'--learning-rate',str(cfg['lr'])]
    if job['method'] in ('frde','wrde','forde'): cmd+=['--repulsion-strength',str(job['weight'])]
    if job['method'] in ('frde','forde'): cmd+=['--repulsion-location',job['location']]
    return cmd


def refresh(output):
    subprocess.run([sys.executable,str(ROOT/'author_report.py'),'--runs',str(output)],check=True,cwd=ROOT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--presets',nargs='+',choices=PRESETS,default=PRESETS)
    parser.add_argument('--methods',nargs='+',choices=AUTHOR_METHODS,default=AUTHOR_METHODS)
    parser.add_argument('--output',type=Path,default=Path(OUTPUT))
    parser.add_argument('--device',default=DEVICE,choices=['cuda:0'])
    parser.add_argument('--workers',default=WORKERS)
    parser.add_argument('--no-originals',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    args.output=args.output.resolve()
    if args.report_only: refresh(args.output); return
    if args.workers!='all':
        args.workers=int(args.workers)
        if args.workers<1: parser.error('workers must be positive')
    if any(len(v)!=len(set(v)) for v in (args.datasets,args.presets,args.methods)):
        parser.error('Duplicate selections')
    if not args.dry_run:
        prepare_source()
        from generate_data import save_named_dataset
        if 'author-xsine' in args.datasets: save_named_dataset('author-xsine',ROOT/'data/author-xsine.npz')
    jobs=plan_jobs(args)
    print(f'{len(jobs)} fits submitted to GPU 0; concurrency {args.workers}',flush=True)
    if args.dry_run:
        for job in jobs: print(job['id'])
        return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU 0 unavailable; no fits started')
    # Restrict physical GPU access in every child; logical cuda:0 remains GPU 0.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in (None,'0'): raise RuntimeError('GPU mapping must be unset or physical GPU 0')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    args.memory_per_fit=MEMORY_PER_FIT_MIB
    args.memory_reserve=MEMORY_RESERVE_MIB
    args.update_website=True
    args.retry_memory_errors=True
    import signal
    def interrupt(signum,frame): raise KeyboardInterrupt('Author runner stopped')
    signal.signal(signal.SIGTERM,interrupt)
    execute(args,jobs,validator=validate,marker=mark,command_builder=command,reporter=refresh)


if __name__=='__main__': main()
