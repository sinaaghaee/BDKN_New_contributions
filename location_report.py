"""Offline side-by-side repulsion-location predictions and saved loss histories."""
from pathlib import Path
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

LOCATIONS=('training','random','combined')
LABELS={'training':'Training','random':'Random','combined':'Training + random'}
METHODS=('frde','forde')


def comparison_groups(registry):
    groups={}
    for entry in registry['experiments']:
        key=(entry['dataset'],entry['learning_rate'],entry['repulsion'])
        groups.setdefault(key,{})[entry.get('repulsion_location','training')]=entry
    return groups


def group_digest(entries):
    digest=hashlib.sha256(b'location-comparison-v1')
    first=next(iter(entries.values()))
    data_path=Path(json.loads((Path(first['run'])/'run.json').read_text())['data'])
    digest.update(data_path.read_bytes())
    for location in LOCATIONS:
        if location not in entries:continue
        entry=entries[location];digest.update(location.encode())
        for method in METHODS:
            for key in (method,method+'_ensemble'):
                digest.update((Path(entry['run'])/method/key/'predictions.npz').read_bytes())
    return digest.hexdigest()


def build_group(args):
    entries,destination=args;destination=Path(destination)
    if (destination/'complete.json').exists():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True,exist_ok=True)
    first=next(iter(entries.values()));run=Path(first['run'])
    with np.load(json.loads((run/'run.json').read_text())['data']) as saved:data={k:saved[k] for k in saved.files}
    x=data['x_grid'].reshape(-1)
    for method in METHODS:
        for stage in ('ensemble','bdkn'):
            key=method+'_ensemble' if stage=='ensemble' else method
            predictions={}
            for location,entry in entries.items():
                with np.load(Path(entry['run'])/method/key/'predictions.npz') as saved:predictions[location]={k:saved[k] for k in saved.files}
            values=np.concatenate([data['y_train'].reshape(-1),data['y_test'].reshape(-1),data['f_grid'].reshape(-1)]+[p[k] for p in predictions.values() for k in ('grid_lower','grid_upper')])
            lo,hi=float(values.min()),float(values.max());pad=.04*max(hi-lo,1e-6)
            for location,pred in predictions.items():
                for mode,suffix in (('train',''),('test','_test'),('both','_both'),('none','_none')):
                    fig,ax=plt.subplots(figsize=(6,4))
                    ax.plot(x,data['f_grid'].reshape(-1),'k--',lw=1,label='True function')
                    ax.plot(x,pred['grid_mean'],color='#396ca9',lw=1.5,label='Prediction')
                    ax.fill_between(x,pred['grid_lower'],pred['grid_upper'],color='#396ca9',alpha=.18,label='95% interval')
                    for split,color,size,alpha in (('train','#172033',12,.45),('test','#b36a30',6,.25)):
                        if mode in (split,'both'):ax.scatter(data['x_'+split],data['y_'+split],color=color,s=size,alpha=alpha,label=split.capitalize()+' observations')
                    ax.set(xlabel='x',ylabel='y',xlim=(x.min(),x.max()),ylim=(lo-pad,hi+pad),title=LABELS[location]);ax.grid(alpha=.12);ax.legend(fontsize=7)
                    fig.tight_layout();fig.savefig(destination/f'{method}_{stage}_{location}{suffix}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    (destination/'complete.json').write_text(json.dumps(dict(locations=list(entries)))+'\n')


def generate(registry,root,workers=4):
    groups=comparison_groups(registry);jobs=[];index={}
    for key,entries in groups.items():
        digest=group_digest(entries);index[key]=digest
        jobs.append((entries,str(root/'location-assets'/digest)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,_ in enumerate(pool.map(build_group,jobs),1):
            if i%10==0 or i==len(jobs):print(f'Location comparisons: {i}/{len(jobs)}',flush=True)
    return index


def render(ctx):
    """Called after all profile assets exist; histories reuse the original plots."""
    from report_ui import page,image,loss_chart,observation_controls,loss_controls,metric_cell,legend,choices
    entry=ctx['entry'];root=ctx['root'];output=ctx['output'];registry=ctx['registry']
    entries=comparison_groups(registry)[(entry['dataset'],entry['learning_rate'],entry['repulsion'])]
    locations=[location for location in LOCATIONS if location in entries]
    digest=ctx['location_index'][(entry['dataset'],entry['learning_rate'],entry['repulsion'])]
    mapping_path=output/'asset-map.js';mapping=json.loads(mapping_path.read_text().split('=',1)[1].strip().removesuffix(';'))
    def alias(name,target):mapping['assets/'+name]=os.path.relpath(target,output)
    body='<section><h2>Compare repulsion locations</h2>'+choices('model-filter','Method',[('all','All'),('frde','Functional repulsion'),('forde','First-order repulsion'),('wrde','Weight repulsion')],'all')+observation_controls()+loss_controls()+'<p class="note">Random points redraw every step and exclude all training inputs. Training and random use B points; combined uses 2B. Prediction axes match within each DE or BDKN row.</p></section>'
    for method in METHODS:
        body+='<section data-model="'+method+'"><h2>'+ctx['names'][method].replace(' · BDKN','')+'</h2>'+legend()
        for stage,title in (('ensemble','DE'),('bdkn','BDKN')):
            key=method+'_ensemble' if stage=='ensemble' else method
            rows=[]
            for location in locations:
                rows.append(next(r for r in json.loads((Path(entries[location]['run'])/method/'metrics.json').read_text()) if r['method']==key))
            body+='<h3>'+title+'</h3><div class="table-wrap"><table><thead><tr><th>Repulsion inputs</th><th>RMSE</th><th>NLL</th><th>Training seconds</th></tr></thead><tbody>'
            for location,row in zip(locations,rows):
                body+='<tr><th>'+LABELS[location]+'</th>'+''.join(metric_cell(row.get(k),[r.get(k) for r in rows],2 if k=='training_seconds' else 5) for k in ('rmse','nll','training_seconds'))+'</tr>'
            body+='</tbody></table></div><div class="weight-gallery" style="--weights:'+str(len(locations))+'">'
            for location in locations:
                name=f'location_{method}_{stage}_{location}'
                for suffix in ('','_test','_both','_none'):alias(name+suffix+'.png',root/'location-assets'/digest/f'{method}_{stage}_{location}{suffix}.png')
                body+=image(name+'.png',LABELS[location],observation=True)
            body+='</div>'
        for history,title in (('_history','Stage 1 · total loss and data fit'),('_repulsion_history','Repulsion loss R only'),('_stage2_history','Stage 2 · Bayesian marginal NLL')):
            body+='<h3>'+title+'</h3><div class="weight-gallery" style="--weights:'+str(len(locations))+'">'
            for location in locations:
                other=root/entries[location]['directory'];manifest=json.loads((other/'report_manifest.json').read_text());source_map=json.loads((other/'asset-map.js').read_text().split('=',1)[1].strip().removesuffix(';'))
                chart=next(c for c in manifest['history_charts'] if c['name']==method+history)
                name=f'location_{location}_{method}{history}'
                for suffix in ('','_xlog','_ylog','_xylog'):
                    filename=method+history+suffix+'.png';target=other/source_map.get('assets/'+filename,'assets/'+filename)
                    alias(name+suffix+'.png',target.resolve())
                body+=loss_chart(dict(chart,name=name,caption=LABELS[location]))
            body+='</div>'
        body+='</section>'
    weight_entry=entries.get('training',next(iter(entries.values())))
    weight_rows=json.loads((Path(weight_entry['run'])/'wrde/metrics.json').read_text())
    body+='<section data-model="wrde"><h2>Weight repulsion · location-independent</h2><p class="note">The loss compares network parameters directly. These same results apply to every location.</p><div class="table-wrap"><table><thead><tr><th>Stage</th><th>RMSE</th><th>NLL</th><th>Training seconds</th></tr></thead><tbody>'
    for row in weight_rows:
        body+='<tr><th>'+('DE' if row['method'].endswith('_ensemble') else 'BDKN')+'</th>'+''.join(metric_cell(row.get(k),[r.get(k) for r in weight_rows],2 if k=='training_seconds' else 5) for k in ('rmse','nll','training_seconds'))+'</tr>'
    body+='</tbody></table></div><div class="gallery">'
    other=root/weight_entry['directory'];other_map=json.loads((other/'asset-map.js').read_text().split('=',1)[1].strip().removesuffix(';'))
    for key,label in (('wrde_ensemble','DE'),('wrde','BDKN')):
        name='location_independent_'+key
        for suffix in ('','_test','_both','_none'):alias(name+suffix+'.png',(other/other_map['assets/'+key+suffix+'.png']).resolve())
        body+=image(name+'.png',label,observation=True)
    body+='</div></section>'
    mapping_path.write_text('globalThis.REPORT_ASSETS = '+json.dumps(mapping,separators=(',',':'))+';\n')
    (output/'locations.html').write_text(page(ctx,'Compare locations','locations.html',body))


def render_all(registry,root,index):
    from build_website import NAMES
    for entry in registry['experiments']:
        render(dict(entry=entry,registry=registry,root=root,output=root/entry['directory'],names=NAMES,location_index=index))


def prune_unused(root,index):
    """Remove superseded generated comparisons after their pages are refreshed."""
    import shutil
    used=set(index.values())
    for folder in (root/'location-assets').iterdir():
        if (folder.is_dir() and len(folder.name)==64 and all(c in '0123456789abcdef' for c in folder.name)
                and folder.name not in used and (folder/'complete.json').exists()):
            shutil.rmtree(folder)
