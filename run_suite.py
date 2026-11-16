"""Resumable two-GPU suite; edit the settings here, then run this file.

Every fit executes its standalone method script. Resume skips only validated
completed fits. Incomplete attempts are retained under an `attempts` directory.
"""
DATASETS = ['steps','cubic','forrester','he','amplitude-sine','linear-sinc',
            'square-wave','shifted-sinc','sine','cosine']
WEIGHTS = [.001,.01,.1,1.,10.]
GPUS = ['cuda:0','cuda:1']
LEARNING_RATE = .001
OUTPUT = 'results/suite'

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
from generate_data import DATASETS as DEFINITIONS, save_named_dataset

ROOT=Path(__file__).resolve().parent
METHODS=['bdkn','deep_ensemble','frde','wrde','forde','gp','dkl','ldbll','vbll']
REPULSIVE=['frde','wrde','forde']


def method_script(method): return ROOT/('vbll_regression.py' if method=='vbll' else method+'.py')
def weight_slug(weight): return format(float(weight),'.8g').replace('.','p')
def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path)
    # The trainer and report scheduler may finalize the same completed profile.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
        temporary=Path(handle.name);handle.write(json.dumps(value,indent=2)+'\n')
    try:temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def source_config(method):
    tree=ast.parse(method_script(method).read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing CONFIG in '+method)


def expected_config(method,epochs,lr,weight,gp_lr=None,repulsion_location="training"):
    cfg=source_config(method)
    if epochs is not None: cfg['epochs']=epochs
    if method!='gp': cfg['lr']=lr
    elif gp_lr is not None: cfg['lr']=gp_lr
    if method in REPULSIVE: cfg['repulsion_strength']=weight
    if method in ('frde','forde'):cfg['repulsion_location']=repulsion_location
    return cfg


def validate_fit(folder,method,data,epochs=None,lr=.001,weight=.001,gp_lr=None,repulsion_location="training"):
    """No metrics-only skips: check settings, predictions, history and checkpoint."""
    folder=Path(folder)
    if not (folder/'metrics.json').exists(): return False
    cfg=read_json(folder/'config.json');expected=expected_config(method,epochs,lr,weight,gp_lr,repulsion_location)
    for key,value in expected.items():
        if key not in ('data_path','output_root','device') and not key.startswith('_'):
            if cfg.get(key,'training' if key=='repulsion_location' else None)!=value: raise ValueError(f'Saved configuration mismatch: {folder}: {key}')
    digest=file_hash(data)
    if cfg['dataset_sha256']!=digest: raise ValueError('Saved dataset mismatch: '+str(folder))
    rows=read_json(folder/'metrics.json')
    keys=[method+'_ensemble',method] if method in REPULSIVE else [method]
    if {r['method'] for r in rows}!=set(keys): return False
    import numpy as np
    for row in rows:
        if row['dataset_sha256']!=digest or not np.isfinite([row['rmse'],row['nll'],row.get('training_seconds',float('nan'))]).all():return False
        if row['training_seconds']<=0:return False
        prediction=folder/row['method']/'predictions.npz'
        if not prediction.exists():return False
        with np.load(prediction,allow_pickle=False) as p, np.load(data,allow_pickle=False) as d:
            for coord in ('x_test','x_grid','f_grid'):
                if coord not in p or not np.array_equal(p[coord],d[coord]):return False
            for split in ('test','grid'):
                for key in ('mean','scale','lower','upper'):
                    a=p[split+'_'+key]
                    if len(a)!=len(d['x_'+split]) or not np.isfinite(a).all():return False
            if np.any(p['grid_scale']<=0) or np.any(p['test_scale']<=0):return False
    checkpoint='stage1.pt' if method in ['bdkn','deep_ensemble']+REPULSIVE else 'model.pt'
    if not (folder/checkpoint).is_file():return False
    if method in ['bdkn']+REPULSIVE and not (folder/'posterior.pt').is_file():return False
    if not (folder/'history.csv').exists():return False
    with (folder/'history.csv').open() as f:history=list(csv.DictReader(f))
    first=[r for r in history if r.get('stage','1')=='1']
    if len(first)!=cfg['epochs']:return False
    if method in ['bdkn']+REPULSIVE:
        second=[r for r in history if r.get('stage')=='2']
        if len(second)!=cfg['stage2_steps']:return False
        if not (folder/'stage1.pt').exists():return False
    return True


def build_jobs(datasets,weights,output,epochs,lr,devices=GPUS,gp_lr=None,repulsion_locations=("training",)):
    if tuple(repulsion_locations)!=("training",):
        return build_location_jobs(datasets,weights,output,epochs,lr,devices,gp_lr,repulsion_locations)
    jobs=[];profiles=[]
    for name in datasets:
        data=ROOT/'data'/('toy_1d.npz' if name=='steps' else name+'.npz')
        save_named_dataset(name,data)
        base=output/name/('r'+weight_slug(weights[0]))
        for weight in weights:
            run=output/name/('r'+weight_slug(weight));run.mkdir(parents=True,exist_ok=True)
            manifest=dict(data=str(data),dataset_sha256=file_hash(data),device=' / '.join(devices),
                          methods=METHODS,nn_learning_rate=lr,repulsion_strength=weight,epochs=epochs,gp_learning_rate=gp_lr,source_baseline=str(base))
            if (run/'run.json').exists():
                previous=read_json(run/'run.json')
                for key in ('dataset_sha256','nn_learning_rate','repulsion_strength','epochs','gp_learning_rate'):
                    if previous.get(key)!=manifest[key]:raise ValueError('Run settings changed; choose a new output directory: '+str(run))
            write_json(run/'run.json',manifest)
            selected=METHODS if weight==weights[0] else REPULSIVE
            for method in selected:
                folder=run/method
                if validate_fit(folder,method,data,epochs,lr,weight,gp_lr):continue
                jobs.append(dict(dataset=name,method=method,weight=weight,data=data,folder=folder,
                                 baseline=base,run=run,epochs=epochs,lr=lr,gp_lr=gp_lr))
            profiles.append((name,weight,run,base,data))
    return jobs,profiles


def build_location_jobs(datasets,weights,output,epochs,lr,devices,gp_lr,locations):
    """Existing registered training runs supply all location-independent fits."""
    jobs=[];profiles=[];registry=read_json(ROOT/'website/experiments.json')
    for location in locations:
        if location=='training':
            for name in datasets:
                references=[next((e for e in registry['experiments'] if e['dataset']==name and float(e['learning_rate'])==lr and float(e['repulsion'])==weight and e.get('repulsion_location','training')=='training'),None) for weight in weights]
                if all(references):
                    for weight,entry in zip(weights,references):
                        run=ROOT/entry['run'];data=Path(read_json(run/'run.json')['data'])
                        profiles.append((name,weight,run,run,data))
                else:
                    old_jobs,old_profiles=build_jobs([name],weights,output,epochs,lr,devices,gp_lr)
                    jobs+=old_jobs;profiles+=old_profiles
            continue
        for name in datasets:
            for weight in weights:
                old=next((e for e in registry['experiments'] if e['dataset']==name and float(e['learning_rate'])==lr and float(e['repulsion'])==weight and e.get('repulsion_location','training')=='training'),None)
                if old is None:raise ValueError(f'Missing training reference: {name}, LR={lr}, weight={weight}')
                base=ROOT/old['run'];data=Path(read_json(base/'run.json')['data'])
                run=output/'locations'/location/name/('r'+weight_slug(weight));run.mkdir(parents=True,exist_ok=True)
                manifest=dict(data=str(data),dataset_sha256=file_hash(data),device=' / '.join(devices),methods=METHODS,
                    nn_learning_rate=lr,repulsion_strength=weight,epochs=epochs,gp_learning_rate=gp_lr,
                    source_baseline=str(base),repulsion_location=location)
                if (run/'run.json').exists():
                    previous=read_json(run/'run.json')
                    for key in ('dataset_sha256','nn_learning_rate','repulsion_strength','epochs','gp_learning_rate','repulsion_location'):
                        if previous.get(key)!=manifest[key]:raise ValueError('Run settings changed: '+str(run))
                write_json(run/'run.json',manifest)
                for method in ('frde','forde'):
                    folder=run/method
                    if not validate_fit(folder,method,data,epochs,lr,weight,gp_lr,location):
                        jobs.append(dict(dataset=name,method=method,weight=weight,data=data,folder=folder,
                            baseline=base,run=run,epochs=epochs,lr=lr,gp_lr=gp_lr,repulsion_location=location))
                profiles.append((name,weight,run,base,data))
    return jobs,profiles


def run_jobs(jobs,devices,output):
    waiting=list(jobs);active={};completed=0;failures=[]
    try:
        while waiting or active:
            for device in devices:
                if device in active:continue
                ready=next((j for j in waiting if j['method']!='deep_ensemble' or (j['baseline']/'bdkn'/'stage1.pt').exists()),None)
                if ready is None:continue
                waiting.remove(ready);j=ready;folder=j['folder']
                if folder.exists():
                    attempts=j['run']/'attempts';attempts.mkdir(exist_ok=True)
                    saved=attempts/(folder.name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                    folder.rename(saved)
                    previous_log=j['run']/(j['method']+'.log')
                    if previous_log.exists():shutil.move(str(previous_log),str(saved/'run.log'))
                command=[sys.executable,str(method_script(j['method'])),'--data',str(j['data']),'--output',str(folder),'--device',device]
                if j['method']!='gp':command+=['--learning-rate',str(j['lr'])]
                elif j['gp_lr'] is not None:command+=['--learning-rate',str(j['gp_lr'])]
                if j['method'] in REPULSIVE:command+=['--repulsion-strength',str(j['weight'])]
                if j['method'] in ('frde','forde'):command+=['--repulsion-location',j.get('repulsion_location','training')]
                if j['epochs'] is not None:command+=['--epochs',str(j['epochs'])]
                if j['method']=='deep_ensemble':command+=['--reuse-stage1',str(j['baseline']/'bdkn'/'stage1.pt')]
                log=(j['run']/(j['method']+'.log')).open('w')
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                active[device]=(child,log,j)
                print(f"START {device}: {j['dataset']} {j['method']} location={j.get('repulsion_location','training')} λ={j['weight']}",flush=True)
            for device,(child,log,j) in list(active.items()):
                code=child.poll()
                if code is None:continue
                log.close();del active[device];completed+=1
                okay=code==0 and validate_fit(j['folder'],j['method'],j['data'],j['epochs'],j['lr'],j['weight'],j['gp_lr'],j.get('repulsion_location','training'))
                if not okay:
                    failures.append(dict(dataset=j['dataset'],method=j['method'],weight=j['weight'],repulsion_location=j.get('repulsion_location','training'),exit_code=code))
                    # A failed BDKN cannot supply the checkpoint required by its DE.
                    blocked=[w for w in waiting if w['method']=='deep_ensemble' and w['baseline']==j['baseline'] and j['method']=='bdkn']
                    for w in blocked:waiting.remove(w);failures.append(dict(dataset=w['dataset'],method='deep_ensemble',reason='BDKN dependency failed'))
                print(f"{'DONE' if okay else 'FAILED'} {device}: {j['dataset']} {j['method']} location={j.get('repulsion_location','training')} λ={j['weight']} ({completed}/{len(jobs)})",flush=True)
            write_json(output/'suite_status.json',dict(total=len(jobs),completed=completed,waiting=len(waiting),
                       active=[dict(device=d,dataset=j['dataset'],method=j['method'],weight=j['weight'],repulsion_location=j.get('repulsion_location','training')) for d,(_,_,j) in active.items()],failures=failures))
            if waiting and not active and not any(j['method']!='deep_ensemble' for j in waiting):raise RuntimeError('Unresolved DE dependencies')
            if active:time.sleep(1)
    finally:
        for child,log,_ in active.values():
            if child.poll() is None:child.terminate();child.wait()
            log.close()
    return failures


def finalize(profiles,epochs,lr,gp_lr=None):
    for name,weight,run,base,data in profiles:
        location=read_json(run/'run.json').get('repulsion_location','training')
        rows=[]
        for method in METHODS:
            folder=run/method
            if (method not in REPULSIVE or (location!='training' and method=='wrde')) and run!=base:
                source=base/method
                if not validate_fit(source,method,data,epochs,lr,weight,gp_lr):raise RuntimeError('Incomplete baseline: '+str(source))
                if not folder.exists():shutil.copytree(source,folder,copy_function=os.link)
            if not validate_fit(folder,method,data,epochs,lr,weight,gp_lr,location):raise RuntimeError('Incomplete profile: '+str(folder))
            rows+=read_json(folder/'metrics.json')
        with (run/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(run/'failures.json',[])
        print(f'PROFILE COMPLETE: {name} λ={weight}',flush=True)


def update_registry(profiles,lr,epochs=None,gp_lr=None):
    path=ROOT/'website/experiments.json';registry=read_json(path)
    for name in DATASETS:
        definition=DEFINITIONS[name]
        if name not in registry['datasets']:registry['datasets'][name]=definition
        else:registry['datasets'][name].update(formula_latex=definition['formula'],**{k:definition[k] for k in ('domain','sampling','noise_std','source')})
    for name,weight,run,_,_ in profiles:
        location=read_json(run/'run.json').get('repulsion_location','training')
        entry=dict(id=name+'-lr'+format(lr,'.8g').replace('.','p')+'-r'+weight_slug(weight),dataset=name,
                   learning_rate=lr,repulsion=weight,run=str(run.relative_to(ROOT)))
        entry['repulsion_location']=location
        if location!='training':entry['id']+='-'+location
        if epochs is not None:entry.update(epochs=epochs,gp_learning_rate=gp_lr,setting_label='High LR · 40 epochs')
        old=next((e for e in registry['experiments'] if e['dataset']==name and float(e['learning_rate'])==lr and float(e['repulsion'])==weight and e.get('repulsion_location','training')==location),None)
        if old:
            entry['id']=old['id']
            old.update(entry)
        else:registry['experiments'].append(entry)
    write_json(path,registry)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=['standard','high'],default='standard',help='high: all ten datasets, all methods LR 0.1, 40 epochs; Bayesian stage unchanged')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS)
    parser.add_argument('--weights',nargs='+',type=float,default=WEIGHTS)
    parser.add_argument('--repulsion-locations',nargs='+',choices=['training','random','combined'],default=['training'])
    parser.add_argument('--devices',nargs='+',default=GPUS)
    parser.add_argument('--learning-rate',type=float,default=LEARNING_RATE)
    parser.add_argument('--output')
    parser.add_argument('--epochs',type=int,help='Smoke-test override; omit for full configured budgets')
    parser.add_argument('--register',action='store_true',help='Register only after every selected job succeeds')
    args=parser.parse_args()
    if args.preset=='high':
        if args.epochs not in (None,40) or args.learning_rate not in (LEARNING_RATE,.1):parser.error('The high preset fixes LR 0.1 and 40 epochs; do not override them')
        args.epochs=40;args.learning_rate=.1
    elif args.register and args.epochs is not None:parser.error('Smoke-test budgets cannot be registered as full results')
    if len(set(args.repulsion_locations))!=len(args.repulsion_locations):parser.error('Locations must be unique')
    if args.datasets is None:args.datasets=DATASETS if args.preset=='high' or args.repulsion_locations!=['training'] else [name for name in DATASETS if name!='steps']
    gp_lr=.1 if args.preset=='high' else None
    if args.output is None:args.output='results/suite_high' if args.preset=='high' else OUTPUT
    import numpy as np
    if not args.weights or any(not np.isfinite(w) or w<0 for w in args.weights) or len(set(args.weights))!=len(args.weights):parser.error('Weights must be unique, finite, and nonnegative')
    if not np.isfinite(args.learning_rate) or args.learning_rate<=0:parser.error('Learning rate must be positive')
    if args.epochs is not None and args.epochs<1:parser.error('Epoch count must be positive')
    if not args.devices or len(set(args.devices))!=len(args.devices):parser.error('Choose at least one unique device')
    output=Path(args.output)
    if not output.is_absolute():output=ROOT/output
    output.mkdir(parents=True,exist_ok=True)
    jobs,profiles=build_jobs(args.datasets,args.weights,output,args.epochs,args.learning_rate,args.devices,gp_lr,args.repulsion_locations)
    print(f'{len(jobs)} pending jobs across {len(args.devices)} devices',flush=True)
    failures=run_jobs(jobs,args.devices,output)
    if failures:raise SystemExit('Suite contains failures; inspect suite_status.json and method logs, then resume')
    finalize(profiles,args.epochs,args.learning_rate,gp_lr)
    if args.register:
        update_registry(profiles,args.learning_rate,args.epochs,gp_lr)


if __name__=='__main__':main()
