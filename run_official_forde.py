"""Run author FoRDE regression and matched controls; edit selections below.

Uses a separate JAX environment and result manifest. Existing experiments keep
running. Every fit also exports the unchanged PyTorch BDKN second stage.
"""
# --- EDIT SETTINGS HERE ---
DATASETS=['author-xsine','steps','cubic','forrester','he','amplitude-sine',
          'linear-sinc','square-wave','shifted-sinc','sine','cosine']
PRESETS=['low','high','upstream']
METHODS=['forde','control']
DEVICE='cuda:0'
WORKERS='all'
MEMORY_PER_FIT_MIB=1100  # JAX context/JIT plus the sequential BDKN subprocess.
MEMORY_RESERVE_MIB=2048
OUTPUT='results/official_forde'
JAX_PYTHON='.venv-forde/bin/python'

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
from official_forde import CONFIG,dangelo_settings,digest
from official_forde_core import LOCK,prepare_source
from run_adaptive import execute,atomic_json,read_json


def plan_jobs(args):
    jobs=[]
    for dataset in args.datasets:
        path=ROOT/'data'/('toy_1d.npz' if dataset=='steps' else dataset+'.npz')
        if not path.is_file():raise FileNotFoundError(path)
        import numpy as np
        with np.load(path,allow_pickle=False) as data:n=len(data['x_train'])
        for preset in args.presets:
            initialization=ROOT/'data/dangelo_initializations'/f'{dataset}-{preset}.npz'
            for method in args.methods:
                cfg={key:value for key,value in (CONFIG|dangelo_settings(preset,n)|dict(preset=preset,method=method,weight_decay=1.)).items()
                     if key not in ('data','output','device')}
                cfg.update(source_revision=LOCK['revision'],dataset_sha256=digest(path),
                    code={name:digest(ROOT/name) for name in ('official_forde.py','official_forde_core.py','prepare_dangelo.py','author_code.py','author_regression.py','author_source.lock.json',
                        'official_forde_bayes.py','official_forde_source.lock.json','requirements-forde.lock.txt','bdkn.py')})
                fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
                identifier=f'{dataset}-{preset}-{method}-{fingerprint[:12]}'
                jobs.append(dict(id=identifier,kind='official_forde',dataset=dataset,preset=preset,method=method,
                    location='training',signature=cfg,fingerprint=fingerprint,data=str(path),
                    folder=str(args.output/'runs'/identifier),initialization=str(initialization),status='pending'))
    return jobs


def artifacts_valid(job):
    import numpy as np
    folder=Path(job['folder']);cfg=read_json(folder/'config.json')
    if cfg['smoke_test'] or not cfg['stage2_enabled']:return False
    for key,value in job['signature'].items():
        if key!='code' and cfg.get(key)!=value:return False
    for name,sha in cfg['source_sha256'].items():
        if job['signature']['code'].get(name)!=sha:return False
    required=['stage1_params.npz','initial_params.npz','members.npz','posterior.pt','source.json','preprocessing.npz','native_uncertainty.npz']
    if not all((folder/name).is_file() for name in required):return False
    with (folder/'history.csv').open() as handle:history=list(csv.DictReader(handle))
    if sum(r['stage']=='1' for r in history)!=cfg['epochs']:return False
    if sum(r['stage']=='2' for r in history)!=cfg['stage2_steps']:return False
    base='author_forde' if job['method']=='forde' else 'author_forde_control'
    rows=read_json(folder/'metrics.json')
    if {r['method'] for r in rows}!={base,base+'_bdkn'}:return False
    with np.load(job['data'],allow_pickle=False) as data:
        for row in rows:
            if row['dataset_sha256']!=cfg['dataset_sha256'] or not np.isfinite([row['rmse'],row['nll'],row['training_seconds']]).all():return False
            with np.load(folder/row['method']/'predictions.npz',allow_pickle=False) as p:
                for key in ('x_test','y_test','x_grid','f_grid'):
                    if not np.array_equal(p[key],data[key]):return False
                for split in ('test','grid'):
                    for key in ('mean','scale','lower','upper'):
                        values=p[split+'_'+key]
                        if len(values)!=len(data['x_'+split]) or not np.isfinite(values).all():return False
                    if np.any(p[split+'_scale']<=0):return False
    return True


def validate(job):
    folder=Path(job['folder'])
    try:
        marker=read_json(folder/'complete.json')
        return (marker['fingerprint']==job['fingerprint'] and all((folder/name).is_file() and digest(folder/name)==sha
            for name,sha in marker['files'].items()) and artifacts_valid(job))
    except (OSError,ValueError,KeyError,TypeError):return False


def mark(job):
    if not artifacts_valid(job):raise ValueError('Author FoRDE result failed validation: '+job['id'])
    folder=Path(job['folder'])
    files={str(path.relative_to(folder)):digest(path) for path in folder.rglob('*') if path.is_file()
           and path.suffix in ('.npz','.json','.pt','.csv') and path.name!='complete.json'}
    atomic_json(folder/'complete.json',dict(fingerprint=job['fingerprint'],files=files))


def command(job,device):
    code=Path(job.get('code_root',str(ROOT)))
    return [os.environ.get('FORDE_JAX_PYTHON',str(ROOT/JAX_PYTHON)),str(code/'official_forde.py'),
        '--data',job['data'],'--output',job['folder'],'--device',device,'--preset',job['preset'],
        '--method',job['method'],'--torch-python',sys.executable,'--initialization',job['initialization']]


def refresh(output):
    subprocess.run([sys.executable,str(ROOT/'author_report.py'),'--forde-runs',str(output)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--presets',nargs='+',choices=PRESETS,default=PRESETS)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--output',type=Path,default=Path(OUTPUT))
    parser.add_argument('--jax-python',default=str(ROOT/JAX_PYTHON))
    parser.add_argument('--workers',default=WORKERS)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.device=DEVICE
    if args.report_only:refresh(args.output);return
    if args.workers!='all':
        args.workers=int(args.workers)
        if args.workers<1:parser.error('workers must be positive')
    if any(len(values)!=len(set(values)) for values in (args.datasets,args.presets,args.methods)):parser.error('Duplicate selections')
    jobs=plan_jobs(args);print(f'{len(jobs)} author FoRDE/control fits; GPU 0; concurrency {args.workers}',flush=True)
    if args.dry_run:
        for job in jobs:print(job['id'])
        return
    prepare_source()
    from prepare_dangelo import prepare
    for dataset in args.datasets:
        path=ROOT/'data'/('toy_1d.npz' if dataset=='steps' else dataset+'.npz')
        for preset in args.presets:prepare(path,preset,ROOT/'data/dangelo_initializations'/f'{dataset}-{preset}.npz')
    # Freeze the training implementation so subsequent website/UCI work cannot
    # change source underneath already submitted fits.
    import shutil
    source_id=hashlib.sha256(json.dumps(jobs[0]['signature']['code'],sort_keys=True).encode()).hexdigest()[:16]
    code_root=args.output/'code'/source_id;code_root.mkdir(parents=True,exist_ok=True)
    for name,sha in jobs[0]['signature']['code'].items():
        destination=code_root/name
        if not destination.exists():shutil.copy2(ROOT/name,destination)
        if digest(destination)!=sha:raise ValueError('Frozen implementation mismatch')
    for cache_name,lock_name in (('official_forde','official_forde_source.lock.json'),('repulsive_ensembles','author_source.lock.json')):
        for name in read_json(ROOT/lock_name)['files']:
            destination=code_root/'.cache'/cache_name/name;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copy2(ROOT/'.cache'/cache_name/name,destination)
    compiled=code_root/'.cache/jax_forde'
    if not compiled.exists():
        (ROOT/'.cache/jax_forde').mkdir(parents=True,exist_ok=True);compiled.symlink_to(ROOT/'.cache/jax_forde',target_is_directory=True)
    for job in jobs:job['code_root']=str(code_root)
    if not Path(args.jax_python).is_file():raise FileNotFoundError('Missing isolated JAX Python: '+args.jax_python)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU 0 unavailable; no silent CPU fallback')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0'):raise RuntimeError('Physical GPU must be 0')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',FORDE_JAX_PYTHON=args.jax_python)
    args.memory_per_fit=MEMORY_PER_FIT_MIB;args.memory_reserve=MEMORY_RESERVE_MIB
    args.update_website=True;args.retry_memory_errors=True
    import signal
    def interrupt(signum,frame):raise KeyboardInterrupt('Author FoRDE runner stopped')
    signal.signal(signal.SIGTERM,interrupt)
    execute(args,jobs,validator=validate,marker=mark,command_builder=command,reporter=refresh)


if __name__=='__main__':main()
