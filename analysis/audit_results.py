"""Read-only comparison of saved toy and benchmark results; never starts training.

Dataset means are the comparison units. Repeated baselines are deduplicated.
Sweeps are reported per fixed setting, never by choosing a test-set winner.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, json, re, sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

HERE = Path(__file__).resolve().parent
TOY = HERE.parent
SOURCE = TOY.parent / 'BDKN_Experiments'
sys.path.insert(0, str(SOURCE))
from scripts.forde_domain import report as domain
sys.path.insert(0, str(SOURCE/'scripts/comparison_report'))
import generate_complete_comparison as legacy

def read(p): return json.loads(Path(p).read_text())
def save_csv(name, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def summary(rows, group_keys):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in group_keys)].append(r)
    out=[]
    for key, rr in sorted(groups.items()):
        item=dict(zip(group_keys,key));item['n']=len(rr)
        for metric in ('rmse','nll'):
            a=np.array([r[metric] for r in rr]);b=np.array([r['base_'+metric] for r in rr])
            diff=a-b;tol=1e-9+1e-7*np.abs(b)
            item[metric+'_wins']=int((diff < -tol).sum())
            item[metric+'_ties']=int((np.abs(diff)<=tol).sum())
            item[metric+'_median_delta']=float(np.median(diff))
            if metric=='rmse':item['median_rmse_pct']=float(np.median(100*diff/b))
        item['both_wins']=sum(r['rmse']<r['base_rmse'] and r['nll']<r['base_nll'] for r in rr)
        out.append(item)
    return out

def paired(row, reference, **labels):
    if not all(np.isfinite(v) for v in [row.get('rmse',np.nan),row.get('nll',np.nan),reference.get('rmse',np.nan),reference.get('nll',np.nan)]):return None
    return dict(**labels,**{k:row[k] for k in ('rmse','nll')},**{'base_'+k:reference[k] for k in ('rmse','nll')})

def main():
    # Exact split-level values embedded in the existing benchmark website.
    splits=[];baseline={}; inventory=Counter()
    for p in sorted((SOURCE/'website/pages/datasets').glob('*.html')):
        m=re.search(r'<script type="application/json" id="plot-data">(.*?)</script>',p.read_text(),re.S)
        if not m:continue
        d=json.loads(m[1]);ds='sarcos_WAT' if d['dataset']=='sarcos' else d['dataset']
        for lr,ss in d['settings'].items():
            for sp,s in ss.items():
                ens=s.get('details',{}).get('ensemble',{})
                for method,v in ens.items():
                    if 'test' not in v:continue
                    method_run=s['run']
                    for family in ('FRDE','WRDE','FoRDE'):
                        if method in ('DE-'+family,'BDKN-'+family):method_run=s.get(family.lower()+'_run')
                    row=dict(dataset=ds,scope=d['scope'],lr=lr,split=int(sp),method=method,run=method_run,baseline_run=s['run'],source_page=str(p),**v['test'],**{'train_'+k:value for k,value in v.get('train',{}).items()})
                    splits.append(row)
                    if method in ('DE','BDKN'):baseline[d['scope'],ds,lr,int(sp),method]=row
    save_csv('benchmark_split_metrics.csv',splits)
    cells=defaultdict(list)
    for r in splits:cells[r['scope'],r['dataset'],r['lr'],r['method']].append(r)
    means={key:{k:float(np.mean([r[k] for r in rr])) for k in ('rmse','nll','coverage','width') if all(r.get(k) is not None for r in rr)}|{'n_splits':len(rr)} for key,rr in cells.items()}
    # Other methods use the exact summary sources selected by the current report.
    sarcos={(r['scope'],r['lr'],r['method']):r for r in read(SOURCE/'website/pages/hetero_assets/sarcos_baselines/selected_runs.json')}
    other=[]
    for scope,ds,lr in sorted({k[:3] for k in means}):
        for method in ('LDBLL','GP','DKL','BLL','MFVI','VBLL'):
            if ds=='sarcos_new':
                rec=sarcos.get((scope,lr,method));run=Path(rec['run']) if rec and rec.get('run') else None
            else:
                name='sarcos' if ds=='sarcos_WAT' else ds
                run=(legacy.standard_baseline_run if scope=='standard' else legacy.gap_baseline_run)(lr,name,method)
            if run is None:continue
            vals={k:legacy.read_metric(run,method,k.upper()) for k in ('rmse','nll')}
            if any(v.mean is None for v in vals.values()):continue
            v={k:x.mean for k,x in vals.items()}
            for ref in ('DE','BDKN'):
                b=means.get((scope,ds,lr,ref))
                if b:
                    rr=paired(v,b,dataset=ds,scope=scope,lr=lr,method=method,baseline=ref,run=str(run))
                    if rr:other.append(rr)
    save_csv('benchmark_other_methods.csv',other)
    save_csv('benchmark_other_methods_summary.csv',summary(other,['scope','lr','method','baseline']))
    variants=[]
    for scope,ds,lr in sorted({k[:3] for k in means}):
        if ds=='sarcos_new':continue
        name='sarcos' if ds=='sarcos_WAT' else ds
        for variant in (legacy.STANDARD_VARIANTS if scope=='standard' else legacy.GAP_VARIANTS)[1:]:
            run=legacy.variant_run(scope,lr,name,variant)
            if not run:continue
            cfg=read(run/'config.json')
            v={k:legacy.read_metric(run,'BDKN',k.upper()).mean for k in ('rmse','nll')}
            own={k:legacy.read_metric(run,'DE',k.upper()).mean for k in ('rmse','nll')}
            if any(x is None for x in [*v.values(),*own.values()]):continue
            for ref in ('DE','BDKN','own_DE'):
                b=own if ref=='own_DE' else means.get((scope,ds,lr,ref))
                if b:
                    rr=paired(v,b,dataset=ds,scope=scope,lr=lr,method=variant.short,baseline=ref,run=str(run),stage1_epochs=cfg.get('n_epochs_step1'),stage2_epochs=cfg.get('n_epochs_step2'),n_nets=cfg.get('n_nets'))
                    if rr:variants.append(rr)
    save_csv('benchmark_archived_variants.csv',variants)
    save_csv('benchmark_archived_variants_summary.csv',summary(variants,['scope','lr','method','baseline']))
    rows=[]
    for (scope,ds,lr,method),v in means.items():
        refs=['DE']
        if method.startswith('BDKN-') and ('DE-'+method[5:]) in {k[3] for k in means}:refs+=['DE-'+method[5:]]
        if method!='BDKN':refs+=['BDKN']
        for ref in dict.fromkeys(refs):
            b=means.get((scope,ds,lr,ref))
            if not b or method==ref or b['n_splits']!=v['n_splits']:continue
            if {r['split'] for r in cells[scope,ds,lr,method]}!={r['split'] for r in cells[scope,ds,lr,ref]}:continue
            rr=paired(v,b,dataset=ds,scope=scope,lr=lr,method=method,baseline=ref,n_splits=v['n_splits'],coverage=v.get('coverage'),base_coverage=b.get('coverage'),width=v.get('width'),base_width=b.get('width'))
            if rr:rows.append(rr)
    save_csv('benchmark_dataset_comparisons.csv',rows)
    save_csv('benchmark_summary.csv',summary(rows,['scope','lr','method','baseline']))
    # Fixed domain and adaptive FoRDE: deduplicate jobs shared by launch selections.
    sweep=[];status={};errors=[]
    for folder,kind in [('experiments_FoRDE_domain','domain'),('experiments_FoRDE_adaptive','adaptive')]:
        manifest=read(SOURCE/folder/'manifest.json')
        paths=sorted({p for e in manifest['experiments'] for p in e['jobs']})
        jobs=[read(p) for p in paths];status[kind]=dict(Counter(j['status'] for j in jobs))
        for j in jobs:
            if j['status'] not in ('complete','reused'):continue
            cfg=j['config'];lr='low' if cfg['learning_rate_step1']==.001 else 'high'
            ds='sarcos_WAT' if j['dataset']=='sarcos' else j['dataset']
            records=[]
            try:
                for sp in j['expected_splits']:
                    rec=domain.job_record(j,sp)
                    if rec is None:break
                    if not all((j['scope'],ds,lr,sp,stage) in baseline for stage in ('DE','BDKN')):break
                    records.append((sp,rec))
                if len(records)!=len(j['expected_splits']):continue
                for stage in ('DE','BDKN'):
                    val={k:float(np.mean([r[stage+'_test'][k] for _,r in records])) for k in ('rmse','nll')}
                    for ref in ('DE','BDKN','own_DE'):
                        if stage=='DE' and ref=='own_DE':continue
                        b={k:float(np.mean([r['DE_test'][k] if ref=='own_DE' else baseline[j['scope'],ds,lr,sp,ref][k] for sp,r in records])) for k in ('rmse','nll')}
                        rr=paired(val,b,dataset=ds,scope=j['scope'],lr=lr,method=kind+'_'+stage,baseline=ref,weight=str(j.get('weight','adaptive')),location=j['location'],n_splits=len(records),job_id=j['id'])
                        if rr:sweep.append(rr)
            except Exception as e:errors.append(dict(job=j['id'],error=str(e)))
    save_csv('benchmark_forde_sweep.csv',sweep)
    save_csv('benchmark_forde_sweep_summary.csv',summary(sweep,['scope','lr','method','weight','location','baseline']))
    # Nearby FRDE / FoRDE export consists of complete dataset aggregates.
    nearby=defaultdict(dict)
    with (SOURCE/'website/pages/nearby_results.csv').open() as f:
        for r in csv.DictReader(f):
            if r['stage_partition'].endswith('_test') and r['metric'] in ('rmse','nll'):
                nearby[tuple(r[k] for k in ('scope','lr','dataset','family','weight','location','stage_partition'))][r['metric']]=float(r['mean'])
    near=[]
    for (scope,lr,ds,fam,w,loc,stage),v in nearby.items():
        ds='sarcos_WAT' if ds=='sarcos' else ds
        for ref in ('DE','BDKN','own_DE'):
            if ref=='own_DE':b=nearby.get((scope,lr,ds if ds!='sarcos_WAT' else 'sarcos_WAT',fam,w,loc,'DE_test'))
            else:b=means.get((scope,ds,lr,ref))
            if not b or (stage=='DE_test' and ref=='own_DE'):continue
            rr=paired(v,b,dataset=ds,scope=scope,lr=lr,method=fam+'_'+stage[:-5],weight=w,location=loc,baseline=ref)
            if rr:near.append(rr)
    save_csv('benchmark_nearby_comparisons.csv',near)
    save_csv('benchmark_nearby_summary.csv',summary(near,['scope','lr','method','weight','location','baseline']))
    # Toy exported results, deduplicating baselines repeated over weight profiles.
    toy=[]
    for name in ('downloads/all_results.csv','adaptive_results.csv','forde_raw_results.csv'):
        with (TOY/'website/dist'/name).open() as f:
            for r in csv.DictReader(f):
                r['lr']=r.get('setting') or ('low' if float(r['nn_learning_rate'])==.001 else 'high')
                r['location']=r.get('location',r.get('repulsion_location','training'))
                r['weight']=r.get('weight',r.get('repulsion','adaptive'))
                r['rmse']=float(r['rmse']);r['nll']=float(r['nll']);toy.append(r)
    tb={}
    for r in toy:
        if r['method'] in ('deep_ensemble','bdkn'):
            key=r['dataset'],r['lr'],r['method']
            if key in tb:
                np.testing.assert_allclose([r['rmse'],r['nll']],[tb[key]['rmse'],tb[key]['nll']],rtol=1e-10,atol=1e-10)
                assert r['dataset_sha256']==tb[key]['dataset_sha256']
            tb[key]=r
    print('Toy baseline method counts',Counter(r['method'] for r in toy))
    tr=[]
    for r in toy:
        method=r['method'];own=method+'_ensemble' if not method.endswith('_ensemble') else None
        if method=='bdkn':own='deep_ensemble'
        for ref in ['deep_ensemble','bdkn']+(['own_DE'] if own and any(x['method']==own for x in toy) else []):
            if method==ref:continue
            b=tb.get((r['dataset'],r['lr'],ref)) if ref!='own_DE' else next((x for x in toy if all(x[k]==r[k] for k in ('dataset','lr','weight','location')) and x['method']==own),None)
            if not b:continue
            rr=paired(r,b,dataset=r['dataset'],lr=r['lr'],method=method,baseline=ref,weight=r['weight'],location=r['location'])
            if rr:tr.append(rr)
    tr=list({tuple(r.items()):r for r in tr}.values())
    save_csv('toy_comparisons.csv',tr)
    save_csv('toy_summary.csv',summary(tr,['lr','method','weight','location','baseline']))
    metadata=dict(created=datetime.now(ZoneInfo('America/Chicago')).isoformat(),benchmark_splits=len(splits),benchmark_dataset_means=len(means),other_method_comparisons=len(other),archived_variant_comparisons=len(variants),sweep_status=status,sweep_errors=errors,toy_export_rows=len(toy))
    (HERE/'audit_inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
