"""Compact offline comparison UI and a registry of completed experiment runs."""
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache'/'matplotlib'))
import numpy as np
import matplotlib.pyplot as plt

CSS = '''
:root{--ink:#172033;--muted:#5d6678;--line:#d9dee8;--accent:#243b6b}*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1440px;margin:auto;padding:20px}header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 22px;background:#15284d;color:white;border-radius:9px}header h1{font-size:1.35rem;margin:0}header a{color:white;font-size:.9rem}h2{font-size:1.2rem;margin:0 0 12px}h3{font-size:1rem;margin:0 0 8px}a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 24px;margin:14px 0;padding:12px 16px;background:white;border:1px solid var(--line);border-radius:8px}nav a{text-decoration:none}nav a[aria-current]{font-weight:750;text-decoration:underline;text-decoration-thickness:3px}.context,.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px}.context{background:white;border:1px solid var(--line);padding:12px 16px;border-radius:8px;margin-top:14px}.context label{font-weight:600;font-size:.85rem}.choices{display:flex;flex-wrap:wrap;gap:6px}.choice-control{display:flex;flex-direction:column;gap:5px}.choice-label{font-size:.85rem;font-weight:600}.choices button{padding:5px 10px;font-size:.85rem;cursor:pointer}.choices button[aria-pressed="true"]{background:#243b6b;color:white;border-color:#243b6b}.choices button:disabled{cursor:default}.parameters td{white-space:normal;text-align:left}.katex{font-size:1.05em}.toolbar{margin:14px 0}select,button{font:inherit;padding:6px 9px;border:1px solid #aab6c7;background:white;border-radius:5px}select{max-width:100%}section{background:white;border:1px solid var(--line);border-radius:9px;margin:16px 0;padding:20px}.note,.muted,.plot-caption,footer{color:var(--muted);font-size:.85rem}.note{margin:8px 0}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px}table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,tbody th{text-align:left}thead th{background:#eaf0f8;color:#1f345d}tbody th{background:#fafbfc}tr:last-child>*{border-bottom:0}.metric a{color:inherit;text-decoration:none}.overview thead th{white-space:normal;min-width:90px}.overview th:first-child{position:sticky;left:0;z-index:1}.metric{color:#172033;font-variant-numeric:tabular-nums}.epoch{color:var(--muted)}th button{padding:0;background:none;border:0;color:inherit;font:inherit;font-weight:700;cursor:pointer;text-decoration:underline;text-underline-offset:3px}.legend{display:flex;align-items:center;gap:8px;margin:8px 0 12px;font-size:.8rem;color:var(--muted)}.gradient{display:inline-block;width:110px;height:12px;background:linear-gradient(90deg,hsl(120 60% 88%),hsl(60 60% 88%),hsl(0 60% 88%));border:1px solid #d7dfd8;border-radius:3px}.weight-gallery{display:grid;grid-template-columns:repeat(var(--weights),minmax(220px,1fr));gap:12px;overflow-x:auto;margin:12px 0 22px;padding-bottom:8px}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.wide{grid-column:1/-1}figure{margin:0;border:1px solid var(--line);border-radius:7px;overflow:hidden}img{display:block;width:100%;height:auto}figcaption{background:#edf2f9;padding:9px 12px;font-size:.9rem;font-weight:600}.plot-caption{padding:0 12px;margin:7px 0}.downloads{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:.85rem}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.equation{font-family:Cambria,"STIX Two Math",serif;font-size:1.05rem;line-height:1.8;overflow-x:auto;padding:12px 16px;margin:12px 0;background:#f5f7fa;border-left:3px solid #5477b8}.definitions h3{margin-top:20px}.definitions p,.definitions li{max-width:1000px}.formula{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;border-radius:6px;padding:16px;font-size:.9rem}.settings td{white-space:normal}.sr-only{position:absolute;left:-10000px}footer{margin:20px 0}.context-group{display:flex;align-items:center;gap:8px}button,a,select{outline-offset:4px}[hidden]{display:none!important}.pair-title{margin-top:20px}code{overflow-wrap:anywhere}
@media(max-width:760px){main{padding:10px}header{padding:12px 14px}header h1{font-size:1.1rem}.context{align-items:stretch;gap:10px}.context-group{flex:1 1 100%;justify-content:space-between}.context select{width:65%;max-width:none}section{padding:12px}.gallery{grid-template-columns:1fr}th,td{padding:8px}nav{gap:10px 16px}.toolbar{gap:8px}}
'''
JS = r"""

function initializeChoice(node){
 node.value=node.dataset.value;
 node.addEventListener('click',event=>{
  const button=event.target.closest('button[data-value]');
  if(!button || button.disabled || !node.contains(button) || button.dataset.value===node.value)return;
  node.value=button.dataset.value;node.dataset.value=node.value;
  node.querySelectorAll('button[data-value]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  node.dispatchEvent(new Event('change'));
 });
}
document.querySelectorAll('.choices').forEach(initializeChoice);
function reportAsset(file){return globalThis.REPORT_ASSETS?.[file] || file;}

function sortMetrics(column){const body=document.querySelector('#metrics tbody');const rows=[...body.rows];const previous=body.dataset.sort;const direction=previous===column+'-asc'?-1:1;rows.sort((a,b)=>direction*(Number(a.cells[column].dataset.value)-Number(b.cells[column].dataset.value)));rows.forEach(row=>body.appendChild(row));body.dataset.sort=column+(direction===1?'-asc':'-desc');document.getElementById('sort-status').textContent='Sorted by '+({1:'RMSE',2:'NLL',3:'training seconds'}[column])+(direction===1?' ascending':' descending');}
const filter=document.getElementById('model-filter');if(filter)filter.addEventListener('change',()=>{document.querySelectorAll('[data-model]').forEach(card=>card.hidden=filter.value!=='all'&&card.dataset.model!==filter.value);});
const lossX=document.getElementById('loss-xscale'),lossY=document.getElementById('loss-yscale');
function updateLossScales(){const x=lossX.value,y=lossY.value;const suffix=x==='log'?(y==='log'?'_xylog':'_xlog'):(y==='log'?'_ylog':'');document.querySelectorAll('[data-loss-base]').forEach(img=>{const file=reportAsset('assets/'+img.dataset.lossBase+suffix+'.png');img.src=file;img.parentElement.href=file;const yLabel=y==='linear'?'linear':(img.dataset.logY==='symlog'?'signed log (symlog; zeros/negatives preserved)':'log');img.closest('figure').querySelector('[data-scale-note]').textContent='X: '+x+' · Y: '+yLabel;});}
if(lossX&&lossY){lossX.addEventListener('change',updateLossScales);lossY.addEventListener('change',updateLossScales);updateLossScales();}

const observations=document.getElementById('observation-mode');
function updateObservations(){const mode=observations.value;const suffix=mode==='train'?'':'_'+mode;document.querySelectorAll('[data-observation-base]').forEach(img=>{const file=reportAsset('assets/'+img.dataset.observationBase+suffix+'.png');img.src=file;img.parentElement.href=file;});document.getElementById('observation-status').textContent=mode==='none'?'Observations hidden':'Showing '+({train:'training',test:'test',both:'training and test'}[mode])+' observations';}
if(observations)observations.addEventListener('change',updateObservations);


const aggregationMethod=document.getElementById('aggregation-method');
if(aggregationMethod){
 function filterAggregation(){document.querySelectorAll('[data-aggregation-method]').forEach(card=>card.hidden=aggregationMethod.value!=='all'&&card.dataset.aggregationMethod!==aggregationMethod.value);}
 const selected=location.hash.match(/^#aggregation-(bdkn|frde|wrde|forde|all)$/)?.[1]||'bdkn';
 aggregationMethod.value=selected;aggregationMethod.dataset.value=selected;
 aggregationMethod.querySelectorAll('button[data-value]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.value===selected)));
 filterAggregation();aggregationMethod.addEventListener('change',()=>{filterAggregation();location.hash='#aggregation-'+aggregationMethod.value;});
}

const lossMethod=document.getElementById('loss-method');
if(lossMethod)lossMethod.addEventListener('change',()=>document.querySelectorAll('[data-loss-method]').forEach(figure=>figure.hidden=lossMethod.value!=='all'&&figure.dataset.lossMethod!==lossMethod.value));

if(globalThis.katex)document.querySelectorAll('[data-tex]').forEach(element=>katex.render(element.dataset.tex,element,{displayMode:element.tagName==='DIV',throwOnError:false}));

const catalogNode=document.getElementById('experiment-catalog');
if(catalogNode){
 const state=JSON.parse(catalogNode.textContent),entries=state.experiments;
 entries.forEach(entry=>entry.repulsion_location??='training');
 const datasetSelect=document.getElementById('dataset-select'),lrSelect=document.getElementById('lr-select'),strengthSelect=document.getElementById('strength-select'),locationSelect=document.getElementById('location-select');
 function rateLabel(value){return value==='0.1'?'High · LR 0.1 · 40 epochs':value==='0.001'?'Low · NN LR 0.001':value;}
 function fill(select,values,preferred){select.replaceChildren(...values.map(value=>{const option=document.createElement('button');option.type='button';option.dataset.value=value;option.textContent=select===lrSelect?rateLabel(value):select===locationSelect?({training:'Training',random:'Random',combined:'Training + random'}[value]||value):value;return option}));select.value=values.includes(preferred)?preferred:values[0];select.dataset.value=select.value;select.querySelectorAll('button[data-value]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.value===select.value)));}
 function switchExperiment(changed){
  const overview=state.page==='index.html',active=entries.find(entry=>entry.id===state.active);
  const dataset=overview?active.dataset:datasetSelect.value;
  const candidates=overview?entries:entries.filter(entry=>entry.dataset===dataset);
  if(changed==='dataset')fill(lrSelect,[...new Set(candidates.map(entry=>entry.learning_rate))].sort((a,b)=>Number(a)-Number(b)),lrSelect.value);
  const atRate=candidates.filter(entry=>entry.learning_rate===lrSelect.value);
  if(locationSelect)fill(locationSelect,['training','random','combined'].filter(value=>atRate.some(entry=>entry.repulsion_location===value)),locationSelect.value);
  const available=atRate.filter(entry=>entry.repulsion_location===(locationSelect?.value||'training'));
  if(changed!=='strength')fill(strengthSelect,[...new Set(available.map(entry=>entry.repulsion))].sort((a,b)=>Number(a)-Number(b)),strengthSelect.value);
  const matching=available.filter(entry=>entry.repulsion===strengthSelect.value);
  const target=matching.find(entry=>entry.dataset===dataset)||matching[0];
  if(target)location.assign(state.root+target.directory+state.page+location.hash);
 }
 datasetSelect.addEventListener('change',()=>switchExperiment('dataset'));
 lrSelect.addEventListener('change',()=>switchExperiment('lr'));
 strengthSelect.addEventListener('change',()=>switchExperiment('strength'));
 if(locationSelect)locationSelect.addEventListener('change',()=>switchExperiment('location'));
}

"""


def load_registry(path, root):
    registry=json.loads(Path(path).read_text())
    entries=registry['experiments'];datasets=registry['datasets'];ids=set();keys=set()
    if not entries: raise ValueError('Register at least one completed experiment')
    for entry in entries:
        identifier=entry['id']
        if not re.fullmatch(r'[a-z][a-z0-9-]*',identifier) or identifier in ids:raise ValueError('Experiment IDs must be unique lowercase slugs')
        ids.add(identifier)
        if entry['dataset'] not in datasets:raise ValueError('Unknown dataset: '+entry['dataset'])
        entry['learning_rate']=format(float(entry['learning_rate']),'.8g')
        entry['repulsion']=format(float(entry['repulsion']),'.8g')
        entry.setdefault('repulsion_location','training')
        if entry['repulsion_location'] not in ('training','random','combined'):raise ValueError('Unknown repulsion location')
        key=(entry['dataset'],entry['learning_rate'],entry['repulsion'],entry['repulsion_location'])
        if key in keys:raise ValueError('Register only one active run for each dataset/LR/repulsion/location combination')
        keys.add(key)
        run=Path(entry['run']);entry['run']=str(run if run.is_absolute() else root/run)
    if registry['default'] not in ids:raise ValueError('The default experiment ID must be registered')
    for entry in entries:entry['directory']='' if entry['id']==registry['default'] else 'experiments/'+entry['id']+'/'
    return registry


def metric_color(value, values):
    finite=[float(v) for v in values if v is not None and np.isfinite(v)]
    if value is None or not np.isfinite(value) or not finite:return '#eef1f5'
    low,high=min(finite),max(finite)
    rank=sum(v<float(value) for v in finite)+(sum(v==float(value) for v in finite)-1)/2
    fraction=.5 if high==low else rank/(len(finite)-1)
    return f'hsl({120*(1-fraction):.2f} 60% 88%)'


def metric_cell(value, values, digits=5, signed=False):
    if value is None:return '<td class="metric" data-value="Infinity">—</td>'
    text=f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    return f'<td class="metric" data-value="{value}" style="background:{metric_color(value,values)}">{text}</td>'


def legend():
    return '<div class="legend"><span>Better</span><span class="gradient" aria-hidden="true"></span><span>Worse · relative rank within each column</span></div>'


def image(name,caption,wide=False,observation=False):
    attr=f' data-observation-base="{name[:-4]}"' if observation else ''
    return f'<figure class="{"wide" if wide else ""}"><a href="assets/{name}" target="_blank" rel="noopener"><img src="assets/{name}"{attr} alt="{escape(caption)}" loading="lazy"></a><figcaption>{escape(caption)}</figcaption></figure>'


def choices(identifier,label,values,selected,disabled=False):
    buttons=''.join(f'<button type="button" data-value="{escape(str(value))}" aria-pressed="{str(str(value)==str(selected)).lower()}"'+(' disabled' if disabled else '')+f'>{escape(str(text))}</button>' for value,text in values)
    return f'<div class="choice-control"><span class="choice-label" id="{identifier}-label">{escape(label)}</span><div class="choices" id="{identifier}" role="group" aria-labelledby="{identifier}-label" data-value="{escape(str(selected))}">{buttons}</div></div>'

def short_label(name):
    return name.replace('Functional repulsion','FRDE').replace('Weight repulsion','WRDE').replace('First-order repulsion','FoRDE').replace('Deep ensemble','DE')

def latex(expression,display=True):
    tag='div' if display else 'span';css='equation' if display else 'math-inline'
    return f'<{tag} class="{css}" data-tex="{escape(expression,quote=True)}"><code>{escape(expression)}</code></{tag}>'


def observation_controls():
    return '<div class="toolbar">'+choices('observation-mode','Observations',[('train','Training'),('test','Test'),('both','Both'),('none','None')],'train')+'<span id="observation-status" class="note" aria-live="polite"></span></div>'


def page(ctx,title,filename,body):
    entry=ctx['entry'];registry=ctx['registry'];prefix='' if not entry['directory'] else '../../'
    catalog=[{k:e[k] for k in ['id','dataset','learning_rate','repulsion','repulsion_location','directory']} for e in registry['experiments']]
    datasets=registry['datasets']
    dataset_values=[(key,value['label']) for key,value in datasets.items() if any(e['dataset']==key for e in catalog)]
    rates=sorted({e['learning_rate'] for e in catalog if filename=='index.html' or e['dataset']==entry['dataset']},key=float)
    strengths=sorted({e['repulsion'] for e in catalog if (filename=='index.html' or e['dataset']==entry['dataset']) and e['learning_rate']==entry['learning_rate']},key=float)
    controls=choices('dataset-select','Dataset',[('all','All datasets')] if filename=='index.html' else dataset_values,'all' if filename=='index.html' else entry['dataset'],filename=='index.html')+choices('lr-select','Training setting',[(v,'High · LR 0.1 · 40 epochs' if v=='0.1' else 'Low · NN LR 0.001' if v=='0.001' else v) for v in rates],entry['learning_rate'])
    controls+=choices('strength-select','Repulsion weight',[('all','All weights')] if filename=='repulsion.html' else [(v,v) for v in strengths],'all' if filename=='repulsion.html' else entry['repulsion'],filename=='repulsion.html')
    locations=[v for v in ('training','random','combined') if any(e['repulsion_location']==v and e['learning_rate']==entry['learning_rate'] and (filename=='index.html' or e['dataset']==entry['dataset']) for e in catalog)]
    controls+=choices('location-select','Repulsion location',[(v,{'training':'Training','random':'Random','combined':'Training + random'}[v]) for v in locations],entry['repulsion_location'])
    links=[('index.html','Results'),('figures.html','Predictions'),('de_bdkn_results.html','DE ↔ BDKN'),('repulsion.html','Compare weights'),('locations.html','Compare locations'),('losses.html','Losses'),('aggregation.html','Aggregation weights'),('dataset.html','Data'),('settings.html','Settings')]
    nav=''.join(f'<a href="{file}"'+(' aria-current="page"' if filename==file else '')+f'>{label}</a>' for file,label in links)
    state=json.dumps(dict(experiments=catalog,datasets={k:v['label'] for k,v in datasets.items()},active=entry['id'],root=prefix,page=filename)).replace('<','\\u003c')
    math_head=f'<link rel="stylesheet" href="{prefix}vendor/katex/katex.min.css">' if filename in ('losses.html','repulsion.html','dataset.html','aggregation.html') else ''
    math_script=f'<script src="{prefix}vendor/katex/katex.min.js"></script>' if math_head else ''
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} · Regression comparison</title><link rel="stylesheet" href="style.css">{math_head}</head><body><main><header><h1>Regression comparison</h1><a href="{prefix+'downloads/all_results.csv' if filename=='index.html' else 'downloads/metrics.csv'}" download>Download results</a></header><div class="context">{controls}</div><nav aria-label="Report pages">{nav}</nav>{body}</main><script id="experiment-catalog" type="application/json">{state}</script><script src="asset-map.js"></script>{math_script}<script src="report.js"></script></body></html>'


def model_table(ctx):
    rows=ctx['rows'];configs=ctx['configs'];names=ctx['names']
    table='<div class="table-wrap"><table id="metrics"><thead><tr><th>Method</th><th><button onclick="sortMetrics(1)">RMSE ↕</button></th><th><button onclick="sortMetrics(2)">NLL ↕</button></th><th><button onclick="sortMetrics(3)">Training seconds ↕</button></th><th>Epochs</th><th>Learning rate</th></tr></thead><tbody>'
    for row in rows:
        key=row['method'];method=key[:-9] if key.endswith('_ensemble') and key!='deep_ensemble' else key
        cfg=configs[method]
        table+=f'<tr><th scope="row"><a href="figures.html#{key}">{names[key]}</a></th>'
        for metric,digits in [('rmse',5),('nll',5),('training_seconds',2)]:table+=metric_cell(row.get(metric),[r.get(metric) for r in rows],digits)
        table+=f'<td class="epoch">{cfg["epochs"]:,}</td><td>{cfg["lr"]:g}</td></tr>'
    return table+'</tbody></table></div><span id="sort-status" class="sr-only" aria-live="polite"></span>'


def loss_chart(chart):
    key=chart['name'];y_label='log' if chart['log_y_scale']=='log' else 'signed log'
    return f'<figure class="loss-chart" data-loss-method="{chart["method"]}"><a href="assets/{key}_xylog.png" target="_blank" rel="noopener"><img src="assets/{key}_xylog.png" data-loss-base="{key}" data-log-y="{chart["log_y_scale"]}" alt="{escape(chart["caption"])}" loading="lazy"></a><figcaption>{escape(chart["caption"])}</figcaption><p class="plot-caption" data-scale-note>X: log · Y: {y_label}</p></figure>'


def loss_controls():
    return '<div class="toolbar">'+choices('loss-xscale','Loss X',[('linear','Linear'),('log','Log')],'log')+choices('loss-yscale','Loss Y',[('linear','Linear'),('log','Log / signed log')],'log')+'</div>'


def loss_definitions():
    def symbols(items):
        return '<div class="table-wrap"><table class="parameters"><tbody>'+''.join('<tr><th scope="row">'+latex(symbol,False)+'</th><td>'+escape(description)+'</td></tr>' for symbol,description in items)+'</tbody></table></div>'

    methods=[
        ('deep-ensemble','Deep ensemble (DE)',[
            r'\mathcal L_{\mathrm{DE}}=\sum_{i=1}^{H}\frac{1}{2B}\sum_{b=1}^{B}\left[\log\!\left(2\pi v_i(\widetilde x_{ib})\right)+\frac{(y_{ib}-\mu_i(\widetilde x_{ib}))^2}{v_i(\widetilde x_{ib})}\right]'
        ],[(r'H,\ B','Number of networks and batch size.'),(r'\widetilde x_{ib},y_{ib}','Preprocessed input and target in network i’s batch.'),(r'\mu_i,\ v_i','Network i’s predicted mean and observation variance.')]),
        ('bdkn','BDKN',[
            r'\mathcal L_1=\mathcal L_{\mathrm{DE}},\qquad \Phi_{ni}=\mu_i(\widetilde x_n)',
            r'\Lambda_0=\sigma_p^{-2}I,\quad\Lambda_n=\Phi^\top\Phi+\Lambda_0,\quad m_n=\Lambda_n^{-1}\Phi^\top y',
            r'\alpha_n=\alpha_0+\frac n2,\qquad\beta_n=\beta_0+\frac12\left(\lVert y-\Phi m_n\rVert^2+\sigma_p^{-2}\lVert m_n\rVert^2\right)',
            r'\mathcal L_2=-\log\Gamma(\alpha_n)+\alpha_n\log\beta_n-\frac12(\log|\Lambda_0|-\log|\Lambda_n|)+\frac n2\log(2\pi)'
        ],[(r'\Phi,\ n,\ y','Frozen ensemble feature matrix, training count, and target vector.'),(r'\sigma_p^2','Prior weight variance learned in stage two.'),(r'\Lambda_0,\Lambda_n,m_n','Prior precision, posterior precision, and posterior mean.'),(r'\alpha_0,\beta_0;\ \alpha_n,\beta_n','Inverse-gamma prior and posterior parameters; prior-only constants omitted from the plotted loss.')]),
        ('frde','Functional repulsion (FRDE)',[
            r'\mathcal L_1=\mathcal L_{\mathrm{DE}}+\lambda R_{\mathrm F},\qquad z_i=(\mu_i(\widetilde s_b))_{b=1}^{P}',
            r'd_{ij}=\lVert z_i-\operatorname{sg}(z_j)\rVert^2,\quad h=\operatorname{sg}\!\left(\max\left\{\frac{\operatorname{med}_{i,j}d_{ij}}{\log(H+1)},\varepsilon_h\right\}\right)',
            r'R_{\mathrm F}=\sum_{i=1}^{H}\log\left[\sum_{j=1}^{H}\exp(-d_{ij}/h)\right]'
        ],[(r'z_i,\widetilde s_b, P','Mean predictions, shared preprocessed repulsion inputs, and their count.'),(r'R_{\mathrm F},\lambda','Unweighted functional repulsion and its objective weight.'),(r'\mathcal L_2','BDKN stage-two loss above, using the trained FRDE features.')]),
        ('wrde','Weight repulsion (WRDE)',[
            r'\mathcal L_1=\mathcal L_{\mathrm{DE}}+\lambda R_{\mathrm W},\qquad z_i=\theta_i',
            r'd_{ij}=\lVert\theta_i-\operatorname{sg}(\theta_j)\rVert^2,\quad h=\operatorname{sg}\!\left(\max\left\{\frac{\operatorname{med}_{i,j}d_{ij}}{\log(H+1)},\varepsilon_h\right\}\right)',
            r'R_{\mathrm W}=\sum_{i=1}^{H}\log\left[\sum_{j=1}^{H}\exp(-d_{ij}/h)\right]'
        ],[(r'\theta_i','All trainable parameters of network i, including both output heads.'),(r'R_{\mathrm W},\lambda','Unweighted weight repulsion and its objective weight.'),(r'\mathcal L_2','BDKN stage-two loss above, using the trained WRDE features.')]),
        ('forde','First-order repulsion (FoRDE, 1D)',[
            r'\mathcal L_1=\mathcal L_{\mathrm{DE}}+\lambda R_{\mathrm G},\quad g_i=\left(\frac{\partial\mu_i(\widetilde s_b)}{\partial\widetilde s_b}\right)_{b=1}^{P},\quad z_i=\frac{g_i}{\sqrt{\lVert g_i\rVert^2+\varepsilon_g}}',
            r'd_{ij}=\lVert z_i-\operatorname{sg}(z_j)\rVert^2,\quad h_j=\operatorname{sg}\!\left(\max\left\{\frac{\operatorname{med}_{i}d_{ij}}{\log H},\varepsilon_h\right\}\right)',
            r'R_{\mathrm G}=\sum_{i=1}^{H}\log\left[\sum_{j=1}^{H}\exp(-d_{ij}/h_j)\right]'
        ],[(r'g_i,z_i','Input-gradient vector on the selected repulsion inputs and its normalized particle.'),(r'h_j','Bandwidth for reference network j.'),(r'R_{\mathrm G},\lambda','Unweighted first-order repulsion and its objective weight.'),(r'\mathcal L_2','BDKN stage-two loss above, using the trained FoRDE features.')]),
        ('gp','Gaussian process (GP)',[
            r'\mathcal L_{\mathrm{GP}}=\frac{r^\top K_y^{-1}r+\log|K_y|+n\log(2\pi)}{2n},\quad r=y-m(X),\quad K_y=K(X,X)+\sigma^2I'
        ],[(r'K,m,\sigma^2','Configured linear-plus-RBF covariance, mean function, and observation noise variance.'),(r'X,y,n','Preprocessed training inputs, targets, and training count.')]),
        ('dkl','Deep kernel learning (DKL)',[
            r'\mathcal L_{\mathrm{DKL}}=\frac{r^\top K_{y,\psi}^{-1}r+\log|K_{y,\psi}|+n\log(2\pi)}{2n},\quad K_{y,\psi}=K^{\mathrm{SKI}}(f_\psi(X),f_\psi(X))+\sigma^2I'
        ],[(r'f_\psi,K^{\mathrm{SKI}}','Learned neural features and the configured grid-interpolated RBF covariance.'),(r'r,m,\sigma^2','Residual y − m(X), mean function, and observation noise variance.')]),
        ('ldbll','LDBLL',[
            r'\mathcal L_{\mathrm{LDBLL}}=-\frac1n\log p(y\mid\Phi)+\frac cn\underset{x\in\mathcal I}{\operatorname{mean}}\,D_{\mathrm{KL}}\!\left[p(\partial_x f\mid x)\,\Vert\,q(\partial_x f\mid x)\right]'
        ],[(r'\Phi,n','Learned Bayesian linear-layer features and training count.'),(r'p(y\mid\Phi)','Gaussian Bayesian linear-layer marginal likelihood; its negative per observation is the likelihood curve.'),(r'p(\partial_x f\mid x),q(\partial_x f\mid x)','Derivative prior and posterior defining the KL penalty.'),(r'c,\mathcal I','Derivative scaling factor and index points (perturbed training inputs).')]),
        ('vbll','VBLL',[
            r'\mathcal L_{\mathrm{VBLL}}=-\frac1B\sum_{b=1}^{B}\mathbb E_{q(w)}\log\mathcal N(y_b;w^\top\phi_b,\sigma^2)+\frac1n\left[D_{\mathrm{KL}}(q(w)\Vert p(w))-\Omega(\sigma^{-2})\right]',
            r'\Omega(\tau)=a\log\tau-\frac{s}{2}\tau'
        ],[(r'\phi_b,q(w),p(w)','Neural features, Gaussian variational last-layer posterior, and Gaussian weight prior.'),(r'\sigma^2,\tau','Learned observation variance and precision.'),(r'a,s','Configured noise-prior degrees-of-freedom coefficient and Wishart scale.'),(r'B,n','Batch size and training count; regularization weight is 1/n.')])
    ]
    body='<section id="loss-definitions"><h2>Loss definitions by method</h2><nav aria-label="Loss definitions">'+''.join('<a href="#definition-'+key+'">'+title+'</a>' for key,title,_,_ in methods)+'</nav>'
    for key,title,equations,parameters in methods:
        body+='<article id="definition-'+key+'"><h3>'+title+'</h3>'+''.join(latex(eq) for eq in equations)+symbols(parameters)+'</article>'
    body+='<h3>Shared repulsion symbols</h3>'+symbols([
        (r'd_{ij},h,\operatorname{med}','Squared particle distance, bandwidth, and median; pairs include i = j.'),
        (r'P,\widetilde s_b','Repulsion count and input: B training points, B random domain points, or their 2B-point union.'),
        (r'\operatorname{sg}','Stop-gradient: treated as fixed during differentiation.'),
        (r'\varepsilon_h=10^{-6},\ \varepsilon_g=10^{-8}','Bandwidth floor and gradient normalization constant.'),
        (r'R_{\mathrm F},R_{\mathrm W},R_{\mathrm G}','Separate repulsion panels show R; objective panels show the full loss including λR.')])
    return body+'</section>'



_OVERVIEW_CACHE={}


def overview_tables(ctx):
    """Rows compare methods within one dataset; never color across target scales."""
    registry=ctx['registry'];active=ctx['entry'];names=ctx['names']
    prefix='' if not active['directory'] else '../../'
    cache_key=(json.dumps(registry,sort_keys=True),active['learning_rate'],active['repulsion'],active.get('repulsion_location','training'))
    if cache_key not in _OVERVIEW_CACHE:
        selected=[]
        for entry in registry['experiments']:
            if entry['learning_rate']!=active['learning_rate'] or entry['repulsion']!=active['repulsion'] or entry.get('repulsion_location','training')!=active.get('repulsion_location','training'):continue
            rows=[]
            for folder in Path(entry['run']).iterdir():
                if folder.is_dir() and (folder/'metrics.json').exists():rows+=json.loads((folder/'metrics.json').read_text())
            if {r['method'] for r in rows}!=set(names):raise ValueError('Incomplete overview profile: '+entry['id'])
            selected.append((entry,{r['method']:r for r in rows}))
        _OVERVIEW_CACHE[cache_key]=selected
    selected=_OVERVIEW_CACHE[cache_key]
    budget='40 epochs for every method' if active.get('epochs')==40 else '2,000 epochs; DKL and LDBLL 10,000'
    body='<section><h2>Results across datasets</h2><p class="note">Fixed held-out observations over each full plotting domain · lower is better.</p><p class="note">'+budget+' · BDKN aggregation: 100 steps.</p><nav aria-label="Metrics"><a href="#rmse">RMSE</a><a href="#nll">NLL</a><a href="#training_seconds">Training time</a></nav></section>'
    for metric,label,digits in [('rmse','Test RMSE',5),('nll','Test NLL',5),('training_seconds','Training time · seconds',2)]:
        body+='<section id="'+metric+'"><h2>'+label+'</h2>'+legend().replace('within each column','within each dataset row')+'<div class="table-wrap"><table class="overview" data-metric="'+metric+'"><thead><tr><th>Dataset</th>'
        body+=''.join('<th>'+escape(short_label(name))+'</th>' for name in names.values())+'</tr></thead><tbody>'
        for entry,lookup in selected:
            directory=prefix+entry['directory'];label=registry['datasets'][entry['dataset']]['label']
            body+='<tr><th scope="row"><a href="'+directory+'figures.html">'+escape(label)+'</a></th>'
            values=[lookup[key].get(metric) for key in names]
            for key in names:
                cell=metric_cell(lookup[key].get(metric),values,digits)
                # Keep the heatmap and machine-readable value on the cell.
                start=cell.index('>')+1
                cell=cell[:start]+'<a href="'+directory+'figures.html#'+key+'">'+cell[start:-5]+'</a></td>'
                body+=cell
            body+='</tr>'
        body+='</tbody></table></div></section>'
    return body


def render_report(ctx):
    entry=ctx['entry'];output=ctx['output'];names=ctx['names'];rows=ctx['rows'];configs=ctx['configs'];lookup={r['method']:r for r in rows};dataset=ctx['registry']['datasets'][ctx['entry']['dataset']]
    body=overview_tables(ctx)
    (output/'index.html').write_text(page(ctx,'Results','index.html',body))
    (output/'main_results.html').write_text(page(ctx,'Results','index.html',body))
    body='<section><h2>Predictions</h2><p class="note">Individual panels show their full Y range. The all-methods overview uses shared axes.</p>'+choices('model-filter','Method',[('all','All')]+[(key,short_label(label)) for key,label in names.items()],'all')+observation_controls()+image('all_models.png','All methods · same axes · 95% predictive intervals',True,True)+'<div class="gallery">'
    for key,label in names.items():body+=f'<article id="{key}" data-model="{key}">'+image(key+'.png',label,observation=True)+'</article>'
    body+='</div></section>'
    (output/'figures.html').write_text(page(ctx,'Predictions','figures.html',body))
    body='<section><h2>DE ↔ BDKN</h2><p class="note">Each pair uses the same trained ensemble.</p>'+observation_controls()+'</section>'
    for label,de,bayesian in ctx['pairs']:
        a,b=lookup[de],lookup[bayesian]
        table='<div class="table-wrap"><table><thead><tr><th>Stage</th><th>RMSE</th><th>NLL</th><th>Training seconds</th></tr></thead><tbody>'
        for stage,r in [('DE',a),('BDKN',b)]:
            table+='<tr><th>'+stage+'</th>'+''.join(metric_cell(r.get(k),[a.get(k),b.get(k)],2 if k=='training_seconds' else 5) for k in ['rmse','nll','training_seconds'])+'</tr>'
        table+='</tbody></table></div>'
        body+='<section><h2>'+label+'</h2>'+table+'<div class="gallery">'+image(de+'.png','DE',observation=True)+image(bayesian+'.png','BDKN',observation=True)+image(bayesian+'_pair.png',label+' overlay',True,True)+'</div></section>'
    (output/'de_bdkn_results.html').write_text(page(ctx,'DE ↔ BDKN','de_bdkn_results.html',body))
    body='<section><h2>Training losses</h2><div class="toolbar">'+choices('loss-method','Method',[('all','All')]+[(m,short_label(names.get(m,m))) for m in configs],'all')+loss_controls()+'</div><div class="gallery">'
    for chart in ctx['histories']:
        key=chart['name'];method=chart['method']
        body+=loss_chart(chart)
    body+='</div></section>'
    body+=loss_definitions()
    (output/'losses.html').write_text(page(ctx,'Losses','losses.html',body))
    formula=dataset.get('formula','Formula not provided for this dataset.')
    data=ctx['data'];metadata=json.loads(str(data['metadata'])) if 'metadata' in data else {}
    formula_math=dataset.get('formula_latex',dataset.get('formula') if 'sampling' in dataset else None)
    formula_block=latex('f(x)='+formula_math) if formula_math else '<pre class="formula">'+escape(formula)+'</pre>'
    sampling=metadata.get('sampling',dataset.get('sampling','Uniform(-1, 1)'))
    noise=metadata.get('noise_std',dataset.get('noise_std',.1))
    body='<section><h2>'+escape(dataset['label'])+'</h2>'+image('data.png','Training and test observations',True)+f'<p class="note">{len(data["x_train"])} training · {len(data["x_test"])} test · seed {metadata.get("seed","—")}</p><h3>Generating function</h3>'+formula_block+(latex(r'\sigma(t)=\frac{1}{1+e^{-t}}') if ctx['entry']['dataset']=='steps' else '')+(latex(r'y=f(x)') if noise==0 else latex(r'\varepsilon\sim\mathcal N(0,\sigma_\varepsilon^2),\quad y=f(x)+\varepsilon,\quad \sigma_\varepsilon='+format(noise,'.8g')))+'<p>'+escape(sampling)+'</p><p class="note">Test inputs: independent uniform observations over the full displayed domain. '+('NumPy normalized sinc: sin(πt)/(πt), with sinc(0) = 1.' if 'sinc' in ctx['entry']['dataset'] else '')+'</p><a href="downloads/toy_1d.npz" download>Download dataset</a></section>'
    (output/'dataset.html').write_text(page(ctx,'Data','dataset.html',body))
    body='<section><h2>Settings</h2><div class="table-wrap"><table class="settings"><thead><tr><th>Method</th><th>Epochs</th><th>Learning rate</th><th>Hidden layers / kernel</th><th>Repulsion</th><th>Repulsion inputs</th><th>Device</th></tr></thead><tbody>'
    for method,cfg in configs.items():
        arch=cfg.get('hidden_dims') or ([cfg['hidden_dim']]*cfg['hidden_layers'] if 'hidden_dim' in cfg else cfg.get('kernel'))
        location_label='Location-independent' if method=='wrde' else ({'training':'Training · B points','random':'Random · B points','combined':'Training + random · 2B points'}[entry['repulsion_location']] if method in ('frde','forde') else '—')
        body+=f'<tr><th>{names.get(method,method)}</th><td>{cfg["epochs"]:,}</td><td>{cfg["lr"]:g}</td><td>{escape(str(arch))}</td><td>{cfg.get("repulsion_strength","—")}</td><td>{location_label}</td><td>{cfg["device"]}</td></tr>'
    body+='</tbody></table></div><p class="note">B is the actual training minibatch size, up to 32. Random points redraw each step, are distinct from all training inputs and one another, and cover the displayed domain. Weight repulsion is location-independent.</p><div class="downloads"><a href="downloads/settings.json" download>Settings JSON</a><a href="downloads/metrics.csv" download>Metrics CSV</a><a href="downloads/provenance.json" download>Source provenance</a></div></section>'
    (output/'settings.html').write_text(page(ctx,'Settings','settings.html',body))
    render_strengths(ctx)
    from aggregation_report import render as render_aggregation
    render_aggregation(ctx)

_STRENGTH_CACHE={}


def weight_plot_base(method,key,weight):
    return 'weight_'+key+'_r'+weight.replace('.','p')


def generated_plot_exists(ctx,filename):
    if filename.is_file():return True
    if '_asset_map' not in ctx:
        path=ctx['output']/'asset-map.js'
        ctx['_asset_map']=json.loads(path.read_text().split('=',1)[1].strip().removesuffix(';')) if path.exists() else {}
    relative=ctx['_asset_map'].get('assets/'+filename.name)
    return bool(relative and (ctx['output']/relative).is_file())


def render_strengths(ctx):
    """One panel per weight; compare values without overlaying predictions."""
    entry=ctx['entry'];names=ctx['names']
    related=sorted([e for e in ctx['registry']['experiments'] if e['dataset']==entry['dataset'] and e['learning_rate']==entry['learning_rate'] and e['repulsion_location']==entry['repulsion_location']],key=lambda e:float(e['repulsion']))
    group=(entry['dataset'],entry['learning_rate'],entry['repulsion_location']);assets=ctx['output']/'assets'
    if group in _STRENGTH_CACHE:
        origin,comparison,repulsion_charts=_STRENGTH_CACHE[group]
        for filename in origin.glob('weight_*.png'):
            if filename.parent!=assets:shutil.copy2(filename,assets/filename.name)
    else:
        comparison={};repulsion_charts={};data=ctx['data'];x=data['x_grid'].reshape(-1)
        import csv
        for method in ['frde','wrde','forde']:
            comparison[method]={};repulsion_charts[method]=[]
            for key,stage in [(method+'_ensemble','DE'),(method,'BDKN')]:
                series=[]
                for e in related:
                    source=Path(e['run'])/method;cfg=json.loads((source/'config.json').read_text())
                    if not np.isclose(cfg['repulsion_strength'],float(e['repulsion'])):raise ValueError('Repulsion label does not match the saved fit')
                    row=next(r for r in json.loads((source/'metrics.json').read_text()) if r['method']==key)
                    if row['dataset_sha256']!=ctx['dataset_hash']:raise ValueError('Cannot compare different datasets under one label')
                    with np.load(source/key/'predictions.npz',allow_pickle=False) as a:pred={k:a[k] for k in a.files}
                    series.append((e,row,pred))
                    base=weight_plot_base(method,key,e['repulsion'])
                    for mode,suffix in [('train',''),('test','_test'),('both','_both'),('none','_none')]:
                        filename=assets/(base+suffix+'.png')
                        if ctx.get('ui_only') and generated_plot_exists(ctx,filename):continue
                        fig,ax=plt.subplots(figsize=(6,4))
                        if mode in ('test','both'):ax.scatter(data['x_test'],data['y_test'],s=6,color='#b36a30',alpha=.25)
                        if mode in ('train','both'):ax.scatter(data['x_train'],data['y_train'],s=12,color='#172033',alpha=.5)
                        ax.plot(x,data['f_grid'].reshape(-1),'k--',lw=1.2,label='True function')
                        ax.plot(x,pred['grid_mean'],color='#396ca9',lw=1.5,label='Prediction')
                        ax.fill_between(x,pred['grid_lower'],pred['grid_upper'],color='#396ca9',alpha=.18,label='95% interval')
                        ax.set(xlim=(float(x.min()),float(x.max())),xlabel='x',ylabel='y',title='λ = '+e['repulsion'])
                        ax.legend(fontsize=7);ax.grid(alpha=.12);fig.tight_layout();fig.savefig(filename,dpi=140,bbox_inches='tight');plt.close(fig)
                comparison[method][stage]=series
            curves=[]
            for e in related:
                with (Path(e['run'])/method/'history.csv').open() as handle:records=[r for r in csv.DictReader(handle) if r['stage']=='1']
                curves.append((e['repulsion'],np.array([int(r['epoch']) for r in records]),np.array([float(r['repulsion']) for r in records])))
            log_y='log' if all(np.all(y>0) for _,_,y in curves) else 'symlog'
            for weight,x_epoch,y_loss in curves:
                base=weight_plot_base(method,method,weight)+'_repulsion_history'
                for xlog,ylog,suffix in [(False,False,''),(True,False,'_xlog'),(False,True,'_ylog'),(True,True,'_xylog')]:
                    filename=assets/(base+suffix+'.png')
                    if ctx.get('ui_only') and generated_plot_exists(ctx,filename):continue
                    fig,ax=plt.subplots(figsize=(6,4));ax.plot(x_epoch,y_loss,color='#396ca9')
                    if xlog:ax.set_xscale('log')
                    if ylog:
                        if log_y=='symlog':ax.set_yscale('symlog',linthresh=1e-8)
                        else:ax.set_yscale('log')
                    ax.set(xlabel='Epoch'+(' · log' if xlog else ''),ylabel='Repulsion loss R'+(' · '+log_y if ylog else ''),title='λ = '+weight)
                    ax.grid(alpha=.15,which='both');fig.tight_layout();fig.savefig(filename,dpi=140,bbox_inches='tight');plt.close(fig)
                repulsion_charts[method].append(dict(name=base,caption='λ = '+weight,log_y_scale=log_y,method=method))
        _STRENGTH_CACHE[group]=(assets,comparison,repulsion_charts)
    body='<section><h2>Compare repulsion weights</h2>'+observation_controls()+loss_controls()+'<p class="note">One panel per weight. Prediction panels use their own full Y range.</p></section>'
    for method,stages in comparison.items():
        body+='<section><h2>'+names[method].replace(' · BDKN','')+'</h2>'+legend()
        for stage,series in stages.items():
            body+='<h3>'+stage+'</h3><div class="table-wrap"><table><thead><tr><th>Repulsion weight</th><th>RMSE</th><th>NLL</th><th>Training seconds</th></tr></thead><tbody>'
            for e,row,pred in series:
                body+='<tr><th>'+e['repulsion']+'</th>'+''.join(metric_cell(row.get(k),[r.get(k) for _,r,_ in series],2 if k=='training_seconds' else 5) for k in ['rmse','nll','training_seconds'])+'</tr>'
            body+='</tbody></table></div><div class="weight-gallery" style="--weights:'+str(len(series))+'">'
            key=method+'_ensemble' if stage=='DE' else method
            for e,row,pred in series:body+=image(weight_plot_base(method,key,e['repulsion'])+'.png','λ = '+e['repulsion'],observation=True)
            body+='</div>'
        body+='<h3>Repulsion loss R only</h3><div class="weight-gallery" style="--weights:'+str(len(related))+'">'+''.join(loss_chart(chart) for chart in repulsion_charts[method])+'</div></section>'
    body+=loss_definitions()
    (ctx['output']/'repulsion.html').write_text(page(ctx,'Compare weights','repulsion.html',body))


# Pre-rendered variants share byte-identical images across experiment profiles.
# Resolve logical plot names through a small offline asset map rather than
# publishing duplicate copies of the same figures.


def deduplicate_assets(root, registry):
    root=Path(root);shared=root/'shared-assets';shared.mkdir(exist_ok=True)
    used=set()
    for entry in registry['experiments']:
        folder=root/entry['directory'];prefix='' if not entry['directory'] else '../../'
        mapping={};map_file=folder/'asset-map.js'
        if map_file.exists():
            mapping=json.loads(map_file.read_text().split('=',1)[1].strip().removesuffix(';'))
            mapping={key:value for key,value in mapping.items() if not key.startswith('assets/strength_')}
        for source in (folder/'assets').glob('*.png'):
            digest=hashlib.sha256(source.read_bytes()).hexdigest();filename=digest+'.png';target=shared/filename
            if not target.exists():shutil.copy2(source,target)
            mapping['assets/'+source.name]=prefix+'shared-assets/'+filename
            used.add(filename);source.unlink()
        for html in folder.glob('*.html'):
            # Do not recurse into other profiles while processing the default.
            text=html.read_text()
            text=re.sub(r'(src|href)="(assets/[^\"]+\.png)"',lambda match:match[1]+'="'+mapping[match[2]]+'"',text)
            html.write_text(text)
        map_file.write_text('globalThis.REPORT_ASSETS = '+json.dumps(mapping,separators=(',',':'))+';\n')
        used.update(Path(value).name for value in mapping.values() if 'shared-assets/' in value)
    for file in shared.glob('*.png'):
        if file.name not in used:file.unlink()
    print(f'Shared plot assets: {len(used)} unique images',flush=True)

