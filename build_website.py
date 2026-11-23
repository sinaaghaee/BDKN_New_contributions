"""Build an offline results website from a completed run; no models are retrained."""
# --- EDIT SETTINGS HERE ---
RUN_PATH = 'results/timed_baseline'
REGISTRY_PATH = 'website/experiments.json'
OUTPUT_PATH = 'website/dist'

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report_ui import CSS, JS, load_registry, render_report, deduplicate_assets

NAMES = {
    'bdkn': 'BDKN', 'deep_ensemble': 'Deep ensemble',
    'frde_ensemble': 'Functional repulsion · DE', 'frde': 'Functional repulsion · BDKN',
    'wrde_ensemble': 'Weight repulsion · DE', 'wrde': 'Weight repulsion · BDKN',
    'forde_ensemble': 'First-order repulsion · DE', 'forde': 'First-order repulsion · BDKN',
    'gp': 'GP', 'dkl': 'DKL', 'ldbll': 'LDBLL', 'vbll': 'VBLL',
}
_PLOT_CACHE = {}

PAIRS = [('Original', 'deep_ensemble', 'bdkn'), ('Functional repulsion', 'frde_ensemble', 'frde'),
         ('Weight repulsion', 'wrde_ensemble', 'wrde'), ('First-order repulsion', 'forde_ensemble', 'forde')]
# Visual tokens and report structure follow BDKN_Experiments/website.


def relative(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path












def make_history_plots(records, method, save, reuse=None):
    """Keep both training stages and preserve negative losses on signed-log axes."""
    groups=[]
    if 'stage' in records[0]:
        first=[row for row in records if row['stage']=='1']
        if first:
            curves=[('Total training loss (NLL + repulsion)' if method in ('frde','wrde','forde') else 'Training NLL', [float(row['objective']) for row in first])]
            if method in ('frde','wrde','forde'):
                curves.append(('Data-fit NLL', [float(row['likelihood']) for row in first]))
            groups.append((method+'_history', first, curves, 'Stage 1 · ensemble training', 'Epoch'))
            if method in ('frde','wrde','forde'):
                groups.append((method+'_repulsion_history',first,[('Repulsion loss R (unweighted)',[float(row['repulsion']) for row in first])],'Repulsion loss R (unweighted)','Epoch'))
        second=[row for row in records if row['stage']=='2']
        if second:
            curves=[('Stage-two marginal NLL', [float(row['objective']) for row in second])]
            groups.append((method+'_stage2_history', second, curves, 'Stage 2 · Bayesian marginal NLL', 'Optimization step'))
    else:
        field='train_loss' if 'train_loss' in records[0] else 'objective'
        curves=[('Marginal NLL (before derivative KL)' if method=='ldbll' else field.replace('_',' '), [float(row[field]) for row in records])]
        groups.append((method+'_history', records, curves, 'Training history', 'Epoch'))
    charts=[]
    for name,group,curves,stage,xlabel in groups:
        x=np.asarray([int(row['epoch']) for row in group])
        values=np.concatenate([np.asarray(y) for _,y in curves])
        if not np.isfinite(values).all() or np.any(x<=0):
            raise ValueError(f'Invalid history for {method} / {stage}')
        # Ordinary log is undefined for zero/negative values; never drop or shift them.
        log_y_scale='log' if np.all(values>0) else 'symlog'
        caption=NAMES.get(method,method)+' · '+stage
        for xlog,ylog,suffix in ((False,False,''),(True,False,'_xlog'),(False,True,'_ylog'),(True,True,'_xylog')):
            if reuse and reuse(name+suffix+'.png'):continue
            fig,ax=plt.subplots(figsize=(8,3.5))
            for label,y in curves:
                ax.plot(x,y,label=label)
            if xlog: ax.set_xscale('log')
            if ylog:
                if log_y_scale=='symlog': ax.set_yscale('symlog',linthresh=1e-8 if name.endswith('_repulsion_history') else 1.)
                else: ax.set_yscale('log')
            y_label='Source loss'
            if ylog: y_label += ' · signed log (symlog)' if log_y_scale=='symlog' else ' · log'
            ax.set(xlabel=xlabel+(' · log' if xlog else ''),ylabel=y_label,title=caption)
            ax.legend(fontsize=8); ax.grid(alpha=.15,which='both')
            save(fig,name+suffix+'.png')
        charts.append(dict(method=method,name=name,caption=caption,log_y_scale=log_y_scale,points=len(group),stage=stage))
    return charts


def plot_key(expected,name,prediction_hashes,history_hashes,rows):
    if name.startswith('data.'):signature=expected
    elif 'history' in name:
        method=next(key for key in sorted(history_hashes,key=len,reverse=True) if name.startswith(key+'_'));signature=history_hashes[method]
    elif name.startswith('all_models'):signature=json.dumps(prediction_hashes,sort_keys=True)
    elif '_pair' in name:
        key=name.split('_pair')[0];signature=prediction_hashes[key]+prediction_hashes['deep_ensemble' if key=='bdkn' else key+'_ensemble']
    elif name.startswith('metrics.'):signature=json.dumps(rows,sort_keys=True)
    else:
        key=next(key for key in sorted(prediction_hashes,key=len,reverse=True) if name.startswith(key+'.') or name.startswith(key+'_'))
        signature=prediction_hashes[key]
    cache_key=(expected,name,signature)
    return cache_key


def build_run(run, output, registry, entry, ui_only=False):
    failure_path=run/'failures.json'
    if not failure_path.exists() or json.loads(failure_path.read_text()):
        raise ValueError('Build the report only from a completed run with no failed methods')
    run_config=json.loads((run/'run.json').read_text())
    dataset=Path(run_config['data'])
    expected=hashlib.sha256(dataset.read_bytes()).hexdigest()
    with np.load(dataset,allow_pickle=False) as archive: data={k:archive[k] for k in archive.files}
    rows=[]; predictions={}; configs={}; locations={}
    for folder in sorted(run.iterdir()):
        if not folder.is_dir() or not (folder/'metrics.json').exists(): continue
        cfg=json.loads((folder/'config.json').read_text()); configs[folder.name]=cfg
        for row in json.loads((folder/'metrics.json').read_text()):
            if row['dataset_sha256']!=expected: raise ValueError('Mixed dataset hashes')
            name=row['method']; rows.append(row); locations[name]=folder/name
            with np.load(folder/name/'predictions.npz',allow_pickle=False) as pred:
                predictions[name]={k:pred[k] for k in pred.files}
    if set(predictions)!=set(NAMES): raise ValueError('Expected all 12 prediction results')
    rows.sort(key=lambda row:list(NAMES).index(row['method']))
    lookup={r['method']:r for r in rows}
    if not np.isclose(configs['bdkn']['lr'],float(entry['learning_rate'])):
        raise ValueError('Learning-rate label does not match saved BDKN configuration')
    for method in ('frde','forde'):
        if configs[method].get('repulsion_location','training')!=entry['repulsion_location']:raise ValueError('Repulsion location does not match saved configuration')
    for method in ('frde','wrde','forde'):
        if not np.isclose(configs[method]['repulsion_strength'],float(entry['repulsion'])):
            raise ValueError('Repulsion label does not match saved configuration')
    ui_only=ui_only and (output/'report_manifest.json').exists()
    if ui_only:
        manifest=json.loads((output/'report_manifest.json').read_text())
        if manifest['dataset_sha256']!=expected or manifest['experiment_id']!=entry['id']:
            raise ValueError('Existing plot manifest does not match profile; rebuild figures')
        for key,digest in manifest.get('source_prediction_sha256',{}).items():
            if hashlib.sha256((locations[key]/'predictions.npz').read_bytes()).hexdigest()!=digest:
                raise ValueError('Saved fit changed; rebuild figures: '+key)
        for key,digest in manifest.get('source_history_sha256',{}).items():
            if hashlib.sha256((run/key/'history.csv').read_bytes()).hexdigest()!=digest:
                raise ValueError('Saved history changed; rebuild figures: '+key)
        context=dict(output=output,registry=registry,entry=entry,names=NAMES,pairs=PAIRS,rows=rows,
                     configs=configs,data=data,predictions=predictions,histories=manifest['history_charts'],dataset_hash=expected,ui_only=True)
        render_report(context)
        manifest['repulsion_location']=entry['repulsion_location']
        (output/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (output/'style.css').write_text(CSS);(output/'report.js').write_text(JS.strip()+'\n')
        shutil.copy2(ROOT/'provenance.json',output/'downloads/provenance.json')
        print('Refreshed '+entry['id'],flush=True)
        return expected
    assets=output/'assets'; downloads=output/'downloads'
    assets.mkdir(parents=True,exist_ok=True); downloads.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/'metrics.csv',downloads/'metrics.csv')
    shutil.copy2(ROOT/'provenance.json',downloads/'provenance.json')
    shutil.copy2(dataset,downloads/'toy_1d.npz')
    # Sanitize machine-specific paths from public report metadata.
    clean_configs={key:{k:v for k,v in cfg.items() if k not in ('data_path','output_root','reuse_stage1')} for key,cfg in configs.items()}
    (downloads/'settings.json').write_text(json.dumps(clean_configs,indent=2)+'\n')
    (downloads/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'style.css').write_text(CSS); (output/'report.js').write_text(JS.strip()+'\n')
    x=data['x_grid'].reshape(-1)
    lo=min(float(p['grid_lower'].min()) for p in predictions.values())
    hi=max(float(p['grid_upper'].max()) for p in predictions.values())
    lo=min(lo,float(data['y_train'].min()),float(data['y_test'].min()));hi=max(hi,float(data['y_train'].max()),float(data['y_test'].max()))
    pad=.04*(hi-lo); limits=(lo-pad,hi+pad)
    def base(ax, observations="train"):
        ax.plot(x,data['f_grid'].reshape(-1),'k--',lw=1.2,label='True function')
        if observations in ('test','both'):
            ax.scatter(data['x_test'],data['y_test'],s=7,color='#b36a30',alpha=.32,label='Test observations',zorder=3)
        if observations in ('train','both'):
            ax.scatter(data['x_train'],data['y_train'],s=13,color='#172033',alpha=.55,label='Training observations',zorder=4)
        ax.set(xlim=(float(x.min()),float(x.max())),ylim=limits,xlabel='x',ylabel='y')
        ax.grid(alpha=.12)
    def draw(ax,name,color='#396ca9',label=None):
        pred=predictions[name]
        ax.fill_between(x,pred['grid_lower'],pred['grid_upper'],color=color,alpha=.17,label='95% predictive interval' if label is None else label+' · 95% interval')
        ax.plot(x,pred['grid_mean'],color=color,lw=1.7,label=label or 'Predictive mean')
    prediction_hashes={key:hashlib.sha256((locations[key]/'predictions.npz').read_bytes()).hexdigest() for key in predictions}
    history_hashes={key:hashlib.sha256((run/key/'history.csv').read_bytes()).hexdigest() for key in configs if (run/key/'history.csv').exists()}
    def save(fig,name):
        # Repeated weight profiles share baseline plots and histories.
        cache_key=plot_key(expected,name,prediction_hashes,history_hashes,rows)
        previous=_PLOT_CACHE.get(cache_key)
        if previous and previous.is_file():shutil.copy2(previous,assets/name) if previous.resolve()!=(assets/name).resolve() else None
        else:
            fig.tight_layout();fig.savefig(assets/name,dpi=140,bbox_inches='tight')
            _PLOT_CACHE[cache_key]=assets/name
        plt.close(fig)
    def reuse(name):
        previous=_PLOT_CACHE.get(plot_key(expected,name,prediction_hashes,history_hashes,rows))
        if previous and previous.is_file():
            shutil.copy2(previous,assets/name) if previous.resolve()!=(assets/name).resolve() else None
            return True
        return False
    # Pre-render observation layers so the switches also work without a server.
    for observations,suffix in (('train',''),('test','_test'),('both','_both'),('none','_none')):
        for name in NAMES:
            if reuse(name+suffix+'.png'):continue
            fig,ax=plt.subplots(figsize=(8,4));base(ax,observations);draw(ax,name);ax.legend(fontsize=7)
            ax.set_title(f"{NAMES[name]} | RMSE {lookup[name]['rmse']:.4f} | NLL {lookup[name]['nll']:.4f}")
            own_lo=min(float(predictions[name]['grid_lower'].min()),float(data['y_train'].min()),float(data['y_test'].min()))
            own_hi=max(float(predictions[name]['grid_upper'].max()),float(data['y_train'].max()),float(data['y_test'].max()))
            own_pad=.04*max(own_hi-own_lo,1.)
            ax.set_ylim(own_lo-own_pad,own_hi+own_pad)
            save(fig,name+suffix+'.png')
        if not reuse('all_models'+suffix+'.png'):
            fig,axes=plt.subplots(4,3,figsize=(17,14),sharex=True,sharey=True)
            for ax,name in zip(axes.flat,NAMES):
                base(ax,observations);draw(ax,name);ax.set_title(NAMES[name],fontsize=10)
            fig.suptitle('All methods · '+observations+' observations · 95% predictive intervals',fontsize=15)
            save(fig,'all_models'+suffix+'.png')
        for label,de,bayesian in PAIRS:
            if reuse(bayesian+'_pair'+suffix+'.png'):continue
            fig,ax=plt.subplots(figsize=(9,4.5));base(ax,observations);draw(ax,de,'#4776b2','DE');draw(ax,bayesian,'#cc732f','BDKN');ax.legend(fontsize=8,ncol=2);ax.set_title(label+' · ensemble versus Bayesian aggregation')
            pair_lo=min(float(predictions[k]['grid_lower'].min()) for k in (de,bayesian))
            pair_hi=max(float(predictions[k]['grid_upper'].max()) for k in (de,bayesian))
            pair_lo=min(pair_lo,float(data['y_train'].min()),float(data['y_test'].min()))
            pair_hi=max(pair_hi,float(data['y_train'].max()),float(data['y_test'].max()))
            pair_pad=.04*max(pair_hi-pair_lo,1.)
            ax.set_ylim(pair_lo-pair_pad,pair_hi+pair_pad)
            save(fig,bayesian+'_pair'+suffix+'.png')
    for name in NAMES:
        shutil.copy2(locations[name]/'predictions.npz',downloads/(name+'_predictions.npz'))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(x,data['f_grid'],'k',lw=1.3,label='True function')
    axes[0].scatter(data['x_train'],data['y_train'],s=16,alpha=.65,label=f'{len(data["x_train"])} training observations')
    axes[0].legend(fontsize=8);axes[0].set_title('Shared training dataset')
    axes[1].plot(x,data['f_grid'],'k',lw=1.3)
    axes[1].scatter(data['x_test'],data['y_test'],s=3,alpha=.25)
    axes[1].set_title('Independent test dataset')
    for ax in axes: ax.set(xlabel='x',ylabel='y',xlim=(float(data['x_grid'].min()),float(data['x_grid'].max())));ax.grid(alpha=.15)
    save(fig,'data.png')
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,metric in zip(axes,('rmse','nll')):
        order=sorted(rows,key=lambda r:r[metric]);labels=[NAMES[r['method']] for r in order]
        ax.barh(labels,[r[metric] for r in order],color=['#35805b']+['#5477b8']*(len(order)-1))
        ax.invert_yaxis();ax.set_xlabel(metric.upper()+' · lower is better');ax.grid(axis='x',alpha=.15)
    save(fig,'metrics.png')
    history_figures=[]
    for method,cfg in configs.items():
        history=run/method/'history.csv'
        if not history.exists(): continue
        with history.open() as handle: records=list(csv.DictReader(handle))
        if not records: continue
        shutil.copy2(history, downloads/(method+'_history.csv'))
        history_figures.extend(make_history_plots(records, method, save, reuse))
    if not np.isclose(configs['bdkn']['lr'],float(entry['learning_rate'])):
        raise ValueError('Learning-rate label does not match saved BDKN configuration')
    for method in ('frde','wrde','forde'):
        if not np.isclose(configs[method]['repulsion_strength'],float(entry['repulsion'])):
            raise ValueError('Repulsion label does not match saved configuration')
    context=dict(output=output,registry=registry,entry=entry,names=NAMES,pairs=PAIRS,
                 rows=rows,configs=configs,data=data,predictions=predictions,histories=history_figures,dataset_hash=expected)
    render_report(context)
    (output/'report_manifest.json').write_text(json.dumps(dict(dataset_sha256=expected,methods=list(configs),predictions=list(NAMES),history_charts=history_figures,observation_modes=['train','test','both','none'],device=run_config['device'],experiment_id=entry['id'],repulsion_location=entry['repulsion_location'],source_prediction_sha256=prediction_hashes,source_history_sha256=history_hashes,generated_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    print(f'Built {entry["id"]}: {output}/index.html',flush=True)
    return expected


def write_overview_download(output,registry):
    combined=[]
    for entry in registry['experiments']:
        for folder in Path(entry['run']).iterdir():
            if folder.is_dir() and (folder/'metrics.json').exists():
                for row in json.loads((folder/'metrics.json').read_text()):
                    cfg=json.loads((folder/'config.json').read_text())
                    combined.append(dict(dataset=entry['dataset'],nn_learning_rate=entry['learning_rate'],method_learning_rate=cfg['lr'],epochs=cfg['epochs'],repulsion=entry['repulsion'],repulsion_location=entry['repulsion_location'],**row))
    (output/'downloads').mkdir(exist_ok=True)
    with (output/'downloads/all_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)


def build_dataset(args):
    registry,output,entries,ui_only=args;output=Path(output)
    for entry in entries:
        folder=output/entry['directory'];manifest_path=folder/'report_manifest.json';map_path=folder/'asset-map.js'
        if not manifest_path.exists() or not map_path.exists():continue
        manifest=json.loads(manifest_path.read_text());mapping=json.loads(map_path.read_text().split('=',1)[1].strip().removesuffix(';'))
        for logical,target in mapping.items():
            name=Path(logical).name
            if not logical.startswith('assets/') or name.startswith(('weight_','aggregation_','location_','metrics')):continue
            try:key=plot_key(manifest['dataset_sha256'],name,manifest['source_prediction_sha256'],manifest['source_history_sha256'],[])
            except (KeyError,StopIteration):continue
            candidate=(folder/target).resolve()
            if candidate.is_file():_PLOT_CACHE[key]=candidate
    hashes=set()
    for entry in entries:
        hashes.add(build_run(Path(entry['run']),output/entry['directory'],registry,entry,ui_only))
    if len(hashes)!=1:raise ValueError('Runs under one dataset label must use identical data')
    return len(entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',default=REGISTRY_PATH,help='Dataset, LR, strength, and completed-run registry')
    parser.add_argument('--aggregation-workers',type=int,default=2,help='CPU workers for saved-posterior diagnostics')
    parser.add_argument('--workers',type=int,default=1,help='Parallel dataset report builders')
    parser.add_argument('--ui-only',action='store_true',help='Refresh tables, controls and equations using existing validated report figures')
    parser.add_argument('--learning-rate',type=float,help='Render only this registered learning-rate setting')
    parser.add_argument('--datasets',nargs='+',help='Render only these dataset IDs while retaining the full overview registry')
    parser.add_argument('--run',help='Build just the default dataset/profile using this completed run')
    parser.add_argument('--output',default=OUTPUT_PATH)
    args=parser.parse_args()
    registry=load_registry(relative(args.registry),ROOT)
    if args.run:
        entry=next(e for e in registry['experiments'] if e['id']==registry['default']).copy()
        entry['run']=str(relative(args.run));registry['experiments']=[entry]
    output=relative(args.output);output.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'website/vendor',output/'vendor',dirs_exist_ok=True)
    from aggregation_report import generate as generate_aggregation
    generate_aggregation(registry,output,args.aggregation_workers)
    groups={}
    for entry in registry['experiments']:
        if args.datasets and entry['dataset'] not in args.datasets:continue
        if args.learning_rate is not None and float(entry['learning_rate'])!=args.learning_rate:continue
        groups.setdefault(entry['dataset'],[]).append(entry)
    jobs=[(registry,str(output),entries,args.ui_only) for entries in groups.values()]
    if args.workers==1:
        for job in jobs:build_dataset(job)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(build_dataset,jobs))
    deduplicate_assets(output,registry)
    from location_report import generate as generate_locations, render_all as render_location_pages, prune_unused as prune_locations
    location_index=generate_locations(registry,output)
    render_location_pages(registry,output,location_index)
    prune_locations(output,location_index)
    write_overview_download(output,registry)
    if args.ui_only or not args.datasets:deduplicate_assets(output,registry)
    if (ROOT/'results/adaptive/manifest.json').exists():
        from adaptive_report import build as build_adaptive
        build_adaptive(ROOT/'results/adaptive',output)
    if (ROOT/'results/forde_raw/manifest.json').exists():
        from forde_raw_report import build as build_raw
        build_raw(ROOT/'results/forde_raw',output)
    if (ROOT/'results/author/manifest.json').exists():
        from author_report import build as build_author
        build_author(ROOT/'results/author',output)
    print(f'Website: {output}/index.html · {len(registry["experiments"])} experiments')


if __name__=='__main__':main()
