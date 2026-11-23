"""Incrementally build the author-code comparison in the existing offline site."""
import argparse
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from author_code import ROOT, LOCK, COMPATIBILITY
from author_regression import LABELS, digest
from adaptive_report import write_text, pack_prediction
from report_ui import CSS, choices, latex
from generate_data import DATASETS

NAMES = dict(author_sgd='Author baseline', author_kde_wgd='Author KDE-WGD',
             author_forde='D’Angelo + author FoRDE', author_forde_control='FoRDE matched control',
             author_kde_f_wgd='Author functional KDE-WGD', deep_ensemble='Original DE',
             bdkn='Original BDKN', frde='Functional repulsion', wrde='Weight repulsion',
             forde='FoRDE', gp='GP', dkl='DKL', ldbll='LDBLL', vbll='VBLL')


def history_data(path):
    """Older scalar methods have no stage column; those are stage one."""
    with path.open() as handle: rows=list(csv.DictReader(handle))
    result={}
    for stage in ('1','2'):
        group=[r for r in rows if r.get('stage','1')==stage]
        if not group: continue
        indices=np.unique(np.r_[0,np.linspace(0,len(group)-1,min(500,len(group)),dtype=int),len(group)-1])
        result[stage]={key:[float(group[i][key]) for i in indices] for key in group[0]
            if key!='stage' and all(r.get(key,'')!='' for r in group)}
        result[stage]['count']=len(group)
    return result


def original_members(folder, data_path, indices):
    """Read frozen checkpoints on CPU; never refit a model for reporting."""
    import torch
    import bdkn as b
    torch.set_num_threads(1)
    saved=torch.load(folder/'stage1.pt',map_location='cpu',weights_only=True)['ensemble']
    model=b.BatchedSingleHeadEnsemble(saved['n_nets'],b.SingleHeadSpec(**saved['spec']))
    model.load_state_dict(saved['state_dict']); model.eval()
    with np.load(data_path,allow_pickle=False) as source: data={key:source[key] for key in source.files}
    cfg=json.loads((folder/'config.json').read_text())
    x,_,ym,ys=b.preprocess(data,cfg,torch.device('cpu'))
    with torch.inference_mode():
        means=model(x['x_grid'][indices])[0].cpu().numpy()*float(ys)+float(ym)
    return means.tolist()


def pack_entry(folder, data_path, output):
    """Cache each frozen fit by its actual source files; export local downloads."""
    folder=Path(folder); data_path=Path(data_path)
    cfg=json.loads((folder/'config.json').read_text())
    expected=digest(data_path)
    if cfg['dataset_sha256']!=expected: raise ValueError('Dataset hash mismatch: '+str(folder))
    rows=json.loads((folder/'metrics.json').read_text())
    files=['config.json','metrics.json','history.csv']
    files += [r['method']+'/predictions.npz' for r in rows]
    files += [p for p in ('members.npz','native_uncertainty.npz','source.json','stage1.pt') if (folder/p).exists()]
    if cfg.get('source_revision')=='e8f9d7418be43064ffdc247ba7e62a61149da542':
        files += ['stage1_params.npz','initial_params.npz','preprocessing.npz']
    signature=[(name,(folder/name).stat().st_mtime_ns,(folder/name).stat().st_size) for name in files]
    key=hashlib.sha256(json.dumps([str(folder),str(output),signature,expected,3]).encode()).hexdigest()
    cache=ROOT/'.cache/author_report'/f'{key}.json'
    identifier=hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    downloads=output/'author_assets/downloads'/identifier
    if cache.exists() and downloads.exists(): return json.loads(cache.read_text())
    with np.load(data_path,allow_pickle=False) as saved: data={key:saved[key] for key in saved.files}
    index=np.linspace(0,len(data['x_grid'])-1,min(400,len(data['x_grid'])),dtype=int)
    packed=[]
    histories=history_data(folder/'history.csv')
    members=None
    if (folder/'members.npz').exists():
        with np.load(folder/'members.npz',allow_pickle=False) as values:
            members=values['x_grid'][:,index].tolist()
    elif (folder/'stage1.pt').exists():
        members=original_members(folder,data_path,index)
    native=None
    if (folder/'native_uncertainty.npz').exists():
        with np.load(folder/'native_uncertainty.npz',allow_pickle=False) as values:
            native=dict(mean=values['mean'][index].tolist(),std=values['std'][index].tolist())
    downloads.mkdir(parents=True,exist_ok=True)
    links=[]
    for name in files:
        if name.endswith('.pt'): continue
        target=downloads/name.replace('/','_')
        shutil.copy2(folder/name,target)
        links.append(dict(label=name,path=target.relative_to(output).as_posix()))
    for row in rows:
        if row['dataset_sha256']!=expected or not np.isfinite([row['rmse'],row['nll'],row['training_seconds']]).all():
            raise ValueError('Nonfinite or mismatched metrics: '+str(folder))
        prediction=pack_prediction(folder/row['method']/'predictions.npz',data)
        if 'native_test_indices' in data and 'native' not in row:
            from scipy.stats import norm,t
            with np.load(folder/row['method']/'predictions.npz',allow_pickle=False) as p:
                indices=data['native_test_indices']; mean=p['test_mean'][indices]; scale=p['test_scale'][indices]
                df=float(p['df']); kwargs={} if np.isnan(df) else dict(df=df)
                distribution=norm if np.isnan(df) else t
                target=data['y_native'].reshape(-1)
                row['native']=dict(rmse=float(np.sqrt(np.mean((target-mean)**2))),
                    nll=float(-distribution.logpdf(target,loc=mean,scale=scale,**kwargs).mean()))
        base=row['method'].removesuffix('_bdkn').removesuffix('_ensemble')
        stage='bdkn' if row['method'].endswith('_bdkn') or row['method'] in ('bdkn','frde','wrde','forde') else 'de'
        title=NAMES.get(base,base)
        if base.startswith('author_'): title+=' · '+('BDKN extension' if stage=='bdkn' else 'DE')
        elif base in ('frde','wrde','forde'): title+=' · '+stage.upper()
        packed.append(dict(id=identifier+'-'+row['method'],label=row['method'],title=title,
            stage=stage,metrics=row,prediction=prediction,history=histories,config=cfg,
            members=members,native=native if stage=='de' else None,downloads=links))
    cache.parent.mkdir(parents=True,exist_ok=True)
    write_text(cache,json.dumps(packed,allow_nan=False))
    return packed


def install_navigation(output):
    for path in output.rglob('*.html'):
        if path.name=='author.html': continue
        text=path.read_text()
        if 'data-author-nav' in text: continue
        href=os.path.relpath(output/'author.html',path.parent)
        text=re.sub(r'(<nav aria-label="Report pages">.*?)</nav>',
            lambda match:match[1]+f'<a data-author-nav href="{href}">Author code</a></nav>',text,count=1,flags=re.S)
        write_text(path,text)


def _build(runs, output, forde_runs):
    manifest=json.loads((runs/'manifest.json').read_text())
    registry=json.loads((ROOT/'website/experiments.json').read_text())
    jobs=list(manifest['jobs'])
    for extra_runs in (forde_runs, ROOT/'results/paper'):
        if (extra_runs/'manifest.json').exists():
            extra=json.loads((extra_runs/'manifest.json').read_text())
            jobs+=extra['jobs']
            manifest['updated']=max(manifest['updated'],extra['updated'])
    manifest['status']='running' if any(j['status'] in ('pending','running') for j in jobs) else 'failed' if any(j['status']=='failed' for j in jobs) else 'complete'
    datasets=list(dict.fromkeys(j['dataset'] for j in jobs))
    cases={}; records={}; errors=[]
    output.mkdir(parents=True,exist_ok=True)
    for dataset in datasets:
        path=ROOT/'data'/('toy_1d.npz' if dataset=='steps' else dataset+'.npz')
        if not path.exists(): continue
        with np.load(path,allow_pickle=False) as data:
            index=np.linspace(0,len(data['x_grid'])-1,min(400,len(data['x_grid'])),dtype=int)
            case=dict(dataset=dataset,metadata=DATASETS[dataset] | json.loads(str(data['metadata'])),
                data=dict(x=data['x_grid'][index].reshape(-1).tolist(),truth=data['f_grid'][index].reshape(-1).tolist(),
                    train_x=data['x_train'].reshape(-1).tolist(),train_y=data['y_train'].reshape(-1).tolist(),
                    test_x=data['x_test'].reshape(-1).tolist(),test_y=data['y_test'].reshape(-1).tolist()),
                author={},original={},records={})
            if 'x_native' in data:
                case['data'].update(native_x=data['x_native'].reshape(-1).tolist(),native_y=data['y_native'].reshape(-1).tolist())
        cases[dataset]=case
    def add(dataset,folder,data_path,key,kind):
        try:
            packed=pack_entry(folder,data_path,output)
            ids=[]
            for entry in packed:
                entry['comparison_preset']=key.split('/')[0]
                if entry['label'] in ('forde','forde_ensemble'):entry['title']='Our '+entry['title']
                cases[dataset]['records'][entry['id']]=entry
                records[entry['id']]=entry
                ids.append(entry['id'])
            cases[dataset][kind][key]=dict(status='complete',ids=ids)
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(folder=str(folder),error=str(error)))
            cases[dataset][kind][key]=dict(status='unavailable',error=str(error))
    # Reuse saved original fixed methods across all existing weights/locations.
    seen=set()
    for profile in registry['experiments']:
        dataset=profile['dataset']
        if dataset not in cases: continue
        preset='low' if float(profile['learning_rate'])==.001 else 'high'
        weight=format(float(profile['repulsion']),'.8g'); location=profile.get('repulsion_location','training')
        for method in ('bdkn','deep_ensemble','frde','wrde','forde','gp','dkl','ldbll','vbll'):
            if method not in ('frde','forde') and location!='training': continue
            key=f'{preset}/{weight if method in ("frde","wrde","forde") else "any"}/{location if method in ("frde","forde") else "training"}/{method}'
            if (dataset,key) in seen: continue
            folder=ROOT/profile['run']/method
            if not (folder/'metrics.json').exists(): continue
            cfg=json.loads((folder/'config.json').read_text())
            add(dataset,folder,Path(cfg['data_path']) if Path(cfg['data_path']).is_absolute() else ROOT/cfg['data_path'],key,'original')
            seen.add((dataset,key))
    for job in jobs:
        if job['dataset'] not in cases: continue
        if job['kind']=='official_forde':
            preset='upstream' if job['preset']=='author_style' else job['preset']
            method='official_forde' if job['method']=='forde' else 'official_forde_control'
            key=f'{preset}/{method}';kind='author'
        elif job['kind']=='author': key=f'{job["preset"]}/{job["method"]}';kind='author'
        else:
            method=job['method'];weight=format(job['weight'],'.8g') if job['weight'] is not None else 'any'
            key=f'{job["preset"]}/{weight}/{job["location"]}/{method}';kind='original'
        entry=dict(status=job['status'],error=job.get('error',''))
        cases[job['dataset']][kind][key]=entry
        if job['status']=='complete': add(job['dataset'],Path(job['folder']),Path(job['data']),key,kind)
    assets=output/'author_assets';assets.mkdir(exist_ok=True)
    for dataset,case in cases.items():
        write_text(assets/(dataset+'.js'),'window.AUTHOR_CASES=window.AUTHOR_CASES||{};window.AUTHOR_CASES['+json.dumps(dataset)+']='+json.dumps(case,allow_nan=False)+';')
    catalog=dict(datasets={key:value['metadata']['label'] for key,value in cases.items()},
        overview={key:dict(author=value['author'],original=value['original'],
            records={identifier:{k:record[k] for k in ('metrics','stage','title','id')} for identifier,record in value['records'].items()})
            for key,value in cases.items()},
        source_revision=LOCK['revision'],status=manifest['status'],updated=manifest['updated'],
        jobs=[{key:j.get(key) for key in ('id','kind','dataset','method','preset','status','error')} for j in jobs],
        errors=errors)
    write_text(output/'author_catalog.js','window.AUTHOR_CATALOG='+json.dumps(catalog)+';')
    # All tables can load without fetch(), including when opened via file://.
    for vendor in ('katex','plotly'):
        shutil.copytree(ROOT/'website/vendor'/vendor,output/'vendor'/vendor,dirs_exist_ok=True)
    controls=choices('dataset','Dataset',list(catalog['datasets'].items()),datasets[0])
    controls+=choices('preset','Author setup',[('paper','D’Angelo paper · 50 networks · 15,000 steps'),('upstream','Author-specific recipes'),('low','Our low LR · 0.001'),('high','Our high LR · 0.1')],'low')
    controls+=choices('original-setting','Original methods',[('low','Low LR'),('high','High LR')],'low')
    controls+=choices('weight','Original repulsion weight',[(str(w),str(w)) for w in (.001,.01,.1,1,10)],'0.001')
    controls+=choices('location','Original repulsion inputs',[('training','Training'),('random','Random'),('combined','Training + random')],'training')
    controls+=choices('evaluation','Test targets',[('noisy','Noisy observations'),('native','Native noiseless · author data')],'noisy')
    controls+=choices('observations','Plot observations',[('train','Training'),('test','Test'),('both','Both'),('none','None')],'train')
    controls+=choices('stage','Prediction plots',[('both','DE + BDKN'),('de','DE'),('bdkn','BDKN')],'both')
    loss_controls=choices('xscale','History X',[('log','Log'),('linear','Linear')],'log')+choices('yscale','History Y',[('log','Log / signed log'),('linear','Linear')],'log')
    equations='<h3>Author baseline</h3>'+latex(r'\ell(\theta)=\log p(\theta)-\frac{N}{2B\sigma^2}\sum_{b=1}^{B}(y_b-f_\theta(x_b))^2,\qquad g_i=-H\nabla_{\theta_i}\ell(\theta_i)')
    equations+='<h3>Author weight-space KDE-WGD</h3>'+latex(r'g_i=-H\nabla_{\theta_i}\ell(\theta_i)+\frac{\sum_j\nabla_{\theta_i}k(\theta_i,\operatorname{sg}(\theta_j))}{\sum_jk(\theta_i,\operatorname{sg}(\theta_j))}')
    equations+='<h3>Author functional KDE-WGD</h3>'+latex(r'g_i=-J_i^\top\left[H\nabla_{f_i}\log p(y\mid f_i)+\widehat{\nabla_{f_i}\log p(f_i)}_{\mathrm{SSGE}}-\frac{\sum_j\nabla_{f_i}k(f_i,\operatorname{sg}(f_j))}{\sum_jk(f_i,\operatorname{sg}(f_j))}\right]')
    equations+='<h3>Shared kernel</h3>'+latex(r'k(z_i,z_j)=\exp\!\left(-\frac{\lVert z_i-z_j\rVert^2}{2s^2}\right),\qquad s^2=\operatorname{sg}\!\left(\frac{\operatorname{median}_{i,j}\lVert z_i-z_j\rVert^2}{2\log(H+1)}\right)')
    body='<section><h1>Author-code comparison</h1><p id="progress" role="status"></p><div class="toolbar">'+controls+'</div><p id="setup-note" class="note"></p></section>'
    body+='<div id="tables"></div>'+(ROOT/'author_settings.html').read_text()+'<section><h2>Predictions</h2><p id="data-formula"></p><div id="predictions" class="author-grid"></div></section>'
    body+='<section><h2>Individual networks</h2><div id="members" class="author-grid"></div></section>'
    body+='<section><h2>Native upstream uncertainty</h2><p class="note">Mean ± 1.96 × sample standard deviation across networks; observation noise is excluded. The predictive comparisons above include likelihood noise.</p><div id="native" class="author-grid"></div></section>'
    body+='<section><h2>Training histories</h2><div class="toolbar">'+loss_controls+'</div><p class="note">Author curves use optimizer steps; original curves use full training passes. Forces are gradients, rather than an additional scalar loss.</p><div id="histories" class="author-grid"></div></section>'
    body+='<section><h2>Settings and downloads</h2><div id="downloads"></div><p class="note">D’Angelo paper training recipe: 50 networks, 2 × 50 ReLU, Adam LR 0.01, 15,000 gradient steps, batch size 64, seed 42; standard Gaussian likelihood and N(0,I) prior. Existing datasets are kept fixed. On author x sin x, the released generator uses noise standard deviation 0.25; the paper’s Gaussian variance notation is ambiguous relative to that code. Other datasets transfer this training recipe. FoRDE and BDKN are extensions.</p><h3>Upstream implementation</h3><p><a href="https://github.com/'+LOCK['repository']+'/tree/'+LOCK['revision']+'">Pinned author source · '+LOCK['revision'][:12]+'</a></p><ul>'+''.join('<li>'+escape(item)+'</li>' for item in COMPATIBILITY)+'</ul><p class="note">Our-settings runs change architecture, input preprocessing, learning rate and update budget. The upstream scalar head, fixed-noise likelihood, prior and update rule are retained. BDKN extensions are ours.</p></section>'
    body+='<section class="definitions"><h2>Author updates</h2>'+equations+'<p>gᵢ: gradient passed to Adam; θᵢ: network parameters; N: training count; B: batch size; H: network count; σ: fixed likelihood standard deviation (1); fᵢ: network outputs on the shared batch; Jᵢ: parameter Jacobian; zᵢ: parameters for weight-space repulsion or outputs for functional repulsion; s²: kernel bandwidth; sg: stop-gradient. Self pairs are included. Functional prior scores use SSGE with regularization 0.01.</p></section>'
    body+='<section><h2>Released-code driving scale</h2><p>For scalar regression the pinned D’Angelo code broadcasts a likelihood tensor of shape H × 1 with a parameter-prior tensor of shape H. Its summed posterior therefore multiplies likelihood/prior driving gradients by H, as shown above; the independent author ensemble has the same factor. This released behavior is retained. The FoRDE hybrid and its matched JAX control use the direct scalar objective D + P, without that broadcasting factor. Compare the hybrid with its JAX control to isolate repulsion.</p></section>'
    body+='<section class="definitions"><h2>D’Angelo regression setup + author FoRDE repulsion</h2><p><a href="https://github.com/AaltoPML/FoRDE/tree/e8f9d7418be43064ffdc247ba7e62a61149da542">Pinned FoRDE JAX source · e8f9d7418be4</a>. The released FoRDE experiments classify images. This combination retains D’Angelo’s scalar regression model, fixed Gaussian noise (std 1), N(0,I) parameter prior, actual initial parameter arrays, shared batch stream and Adam settings. Only the diversity term is replaced by the original FoRDE repulsion function. Its PCA geometry is fitted on training inputs only. This is a regression combination, not a published FoRDE regression experiment.</p>'
    body+=latex(r'u_{ib}=\frac{g_{ib}V}{\sqrt{\lVert g_{ib}V\rVert^2+\epsilon}}\odot s,\quad d_{ijb}=\lVert u_{ib}-\operatorname{sg}(u_{jb})\rVert^2,\quad h_{jb}=\frac{\operatorname{median}_{i}\operatorname{sg}(d_{ijb})}{\log H}+10^{-12}')
    body+=latex(r'R=\frac1B\sum_i\left[\operatorname{LSE}_{j,b}\!\left(-\frac{d_{ijb}}{h_{jb}}\right)-\log B\right],\quad L=\frac{N}{2B}\sum_{i,b}(y_b-\mu_i(x_b))^2+\frac12\lVert\Theta\rVert^2+R+\mathrm{const}')
    body+='<p>g: mean-input derivative; V: training PCA directions; s: square-root PCA eigenvalues; ε = 10⁻²⁴; H: networks; N: training count; B: batch size; LSE: log-sum-exp. The control sets R to zero and shares initial parameters, batches, prior and Adam settings. In one dimension normalization can leave almost only derivative signs; zero or tiny repulsive parameter gradients are retained and plotted.</p><p class="note">Upstream preset: D’Angelo README configuration (100 networks, 10×10 hidden units, LR 0.01, 5,000 steps). Low/high: 5 networks, 50×50 units, LR 0.001/0.1, update counts matching 2,000/40 training passes. All use Adam (0.9, 0.999, ε = 10⁻⁸). BDKN is our separate frozen-network extension. Timings include first-use JAX compilation.</p></section>'
    scripts=''  # Load only the selected dataset's large figure payload on demand.
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Author code · Regression comparison</title><link rel="stylesheet" href="author.css"><link rel="stylesheet" href="vendor/katex/katex.min.css"></head><body><main><nav aria-label="Report pages"><a href="index.html">Results</a><a href="figures.html">Predictions</a><a href="#settings-comparison">Settings comparison</a><a href="adaptive.html">Adaptive repulsion</a><a href="forde_gradients.html">Gradient normalization</a><a data-author-nav href="author.html" aria-current="page">Author code</a></nav>'+body+'</main><script src="vendor/katex/katex.min.js"></script><script src="vendor/plotly/plotly-basic-2.35.2.min.js"></script><script src="author_catalog.js"></script>'+scripts+'<script src="author.js"></script></body></html>'
    write_text(output/'author.html',html)
    write_text(output/'author.css',CSS+'\n.author-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.author-card{min-width:0;border:1px solid #d9e0ea;border-radius:7px;padding:10px}.author-plot{height:360px}.author-status{padding:15px;color:#596579}h3{font-size:1rem}.toolbar{display:flex;flex-wrap:wrap;gap:14px}#downloads>div{padding:10px;border-bottom:1px solid #ddd}@media(max-width:900px){.author-grid{grid-template-columns:1fr}}')
    write_text(output/'author.js',(ROOT/'author_ui.js').read_text())
    write_text(output/'author_manifest.json',json.dumps(catalog,indent=2))
    fields=['dataset','preset','method','repulsion_weight','repulsion_inputs','rmse','nll','training_seconds','native_rmse','native_nll']
    import io
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for case in cases.values():
        for row in case['records'].values():
            metric=row['metrics'];native=metric.get('native',{})
            writer.writerow(dict(dataset=case['dataset'],preset=row['comparison_preset'],
                repulsion_weight=row['config'].get('repulsion_strength',''),
                repulsion_inputs=row['config'].get('repulsion_location','training') if 'repulsion_strength' in row['config'] else '',
                method=row['label'],rmse=metric['rmse'],nll=metric['nll'],training_seconds=metric['training_seconds'],
                native_rmse=native.get('rmse',''),native_nll=native.get('nll','')))
    write_text(output/'author_results.csv',buffer.getvalue())
    install_navigation(output)
    print(f'Author page updated: {sum(j["status"]=="complete" for j in jobs)}/{len(jobs)} new fits; {len(records)} saved prediction records; {len(errors)} unavailable groups',flush=True)


def build(runs,output,forde_runs=None):
    import fcntl
    lock=ROOT/'.cache/author_report_build.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        _build(runs,output,forde_runs or ROOT/'results/official_forde')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=ROOT/'results/author')
    parser.add_argument('--output',type=Path,default=ROOT/'website/dist')
    parser.add_argument('--forde-runs',type=Path,default=ROOT/'results/official_forde')
    args=parser.parse_args();build(args.runs,args.output,args.forde_runs)
