"""Read saved BDKN posteriors and render offline coefficient diagnostics; no fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t
import torch
import bdkn

VERSION=1
METHODS=('bdkn','frde','wrde','forde')
COLORS=['#396ca9','#d17b35','#42916c','#a35a91','#9b8b35']


def statistics(state):
    mean=state['mean'].detach().cpu().double().numpy().reshape(-1)
    precision=state['precision']
    if isinstance(precision,dict):
        q=precision['right_vectors'].detach().cpu().double().numpy()
        eigen=precision['eigenvalues'].detach().cpu().double().numpy()
        if np.any(eigen<=0):raise ValueError('Nonpositive posterior precision')
        inv=(q/eigen[None,:])@q.T
    else:
        matrix=precision.detach().cpu().double().numpy();np.linalg.cholesky(matrix)
        inv=np.linalg.solve(matrix,np.eye(len(matrix)))
    alpha,beta,sigma_p2=[float(state[k]) for k in ('alpha','beta','sigma_p2')]
    if alpha<=1 or beta<=0 or sigma_p2<=0:raise ValueError('Posterior has undefined coefficient variance')
    df=2*alpha;scale=np.sqrt(beta/alpha*np.diag(inv));sd=np.sqrt(beta/(alpha-1)*np.diag(inv))
    quantile=t.ppf(.975,df)
    correlation=inv/np.sqrt(np.outer(np.diag(inv),np.diag(inv)))
    if not np.isfinite(np.r_[mean,scale,sd]).all() or np.any(scale<=0):raise ValueError('Invalid coefficient posterior')
    return dict(mean=mean,scale=scale,sd=sd,low=mean-quantile*scale,high=mean+quantile*scale,
                probability_positive=t.sf(-mean/scale,df),df=df,noise_variance=beta/(alpha-1),
                conditional_prior_sd=np.sqrt(beta/(alpha-1)*sigma_p2),correlation=correlation,
                covariance_factor=inv,sigma_p2=sigma_p2)


def digest_for(folder,data_path):
    digest=hashlib.sha256(str(VERSION).encode())
    for name in ('config.json','stage1.pt','posterior.pt','preprocessing.npz'):
        digest.update((folder/name).read_bytes())
    digest.update(Path(data_path).read_bytes())
    return digest.hexdigest()


def prepare(folder,data_path,destination):
    torch.set_num_threads(1)
    cfg=json.loads((folder/'config.json').read_text())
    if cfg['phi_add_identity'] or cfg['phi_add_bias']:raise ValueError('Coefficient diagnostics require network-only features')
    state=torch.load(folder/'posterior.pt',map_location='cpu',weights_only=True);stats=statistics(state)
    artifact=torch.load(folder/'stage1.pt',map_location='cpu',weights_only=True)['ensemble']
    model=bdkn.BatchedSingleHeadEnsemble(artifact['n_nets'],bdkn.SingleHeadSpec(**artifact['spec']))
    model.load_state_dict(artifact['state_dict']);model.eval()
    with np.load(data_path,allow_pickle=False) as saved:data={key:saved[key] for key in saved.files}
    x,_,ym,ys=bdkn.preprocess(data,cfg,torch.device('cpu'))
    with np.load(folder/'preprocessing.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(x['x_train'].numpy(),saved['x_train_model'])
    heads={};variances={}
    with torch.inference_mode():
        for split in ('train','test','grid'):
            batches=[model(batch) for batch in x['x_'+split].split(512)]
            heads[split]=torch.cat([item[0] for item in batches],dim=1).T.double().numpy()
            variances[split]=torch.cat([item[1] for item in batches],dim=1).T.double().numpy()*ys**2
    method=json.loads((folder/'metrics.json').read_text())[-1]['method']
    reconstructed={split:ym+ys*(heads[split]@stats['mean']) for split in ('test','grid')}
    errors={}
    for split in ('test','grid'):
        with np.load(folder/method/'predictions.npz',allow_pickle=False) as saved:
            original=saved[split+'_mean'];errors[split]=float(np.max(np.abs(reconstructed[split]-original)))
            # Float32 feature evaluation can differ slightly between CUDA and CPU.
            scale=max(1.,float(np.std(data['y_train'])),float(np.max(np.abs(original))))
            np.testing.assert_allclose(reconstructed[split],original,rtol=3e-4,atol=3e-4*scale)
    contributions=ys*heads['test']*stats['mean'][None,:]
    raw=np.mean(np.abs(contributions),axis=0);share=raw/raw.sum() if raw.sum()>0 else np.full(len(raw),1/len(raw))
    raw_heads=ym+ys*heads['test'];target=data['y_test'].reshape(-1,1)
    rmse=np.sqrt(np.mean((raw_heads-target)**2,axis=0))
    nll=-norm.logpdf(target,loc=raw_heads,scale=np.sqrt(variances['test']+cfg['jitter_eps'])).mean(axis=0)
    with np.errstate(invalid='ignore',divide='ignore'):function_corr=np.corrcoef(heads['train'],rowvar=False)
    rows=[dict(network=i+1,mean=float(stats['mean'][i]),ci95_low=float(stats['low'][i]),ci95_high=float(stats['high'][i]),
               posterior_sd=float(stats['sd'][i]),probability_positive=float(stats['probability_positive'][i]),
               mean_absolute_test_contribution_share=float(share[i]),test_rmse=float(rmse[i]),test_nll=float(nll[i])) for i in range(len(raw))]
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'weights.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(version=VERSION,method=method,rows=rows,de_weight=1/len(raw),degrees_of_freedom=stats['df'],
                 sigma_p2=stats['sigma_p2'],noise_variance_mean=stats['noise_variance'],
                 coefficient_correlation=stats['correlation'].tolist(),
                 function_correlation=np.where(np.isfinite(function_corr),function_corr,np.nan).tolist(),
                 reconstructed_mean_max_absolute_error=errors,
                 posterior='Exact marginal Student-t coefficients',
                 prior_reference='Conditional Gaussian prior at fitted posterior mean noise variance; not an unconditional prior density')
    # JSON null marks an undefined correlation for a constant network function.
    payload['function_correlation']=[[float(v) if np.isfinite(v) else None for v in row] for row in function_corr]
    (destination/'statistics.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(destination/'functions.npz',x_grid=data['x_grid'],head_means=ym+ys*heads['grid'],
                        contributions=ys*heads['grid']*stats['mean'][None,:],bdkn_mean=reconstructed['grid'],
                        coefficient_mean=stats['mean'],coefficient_covariance=stats['noise_variance']*stats['covariance_factor'])
    plot(stats,data,heads,ym,ys,function_corr,destination)
    return payload


def save(fig,destination,name):
    fig.tight_layout();fig.savefig(destination/(name+'.png'),dpi=125,bbox_inches='tight');plt.close(fig)


def plot(stats,data,heads,ym,ys,function_corr,destination):
    h=len(stats['mean']);labels=[f'Network {i+1}' for i in range(h)];colors=[COLORS[i%len(COLORS)] for i in range(h)]
    fig,ax=plt.subplots(figsize=(8,4))
    for i in range(h):ax.errorbar(stats['mean'][i],i,xerr=[[stats['mean'][i]-stats['low'][i]],[stats['high'][i]-stats['mean'][i]]],fmt='o',color=colors[i],capsize=4)
    ax.axvline(0,color='.4',lw=1,label='Zero');ax.axvline(1/h,color='.4',ls='--',label=f'DE: 1/{h}')
    ax.set(yticks=range(h),yticklabels=labels,xlabel='Coefficient β · posterior mean and marginal 95% interval',title='Aggregation coefficients');ax.invert_yaxis();ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
    save(fig,destination,'coefficients')
    fig,axes=plt.subplots(2,3,figsize=(12,6));axes=np.asarray(axes).reshape(-1)
    s=stats['conditional_prior_sd'];grid=np.linspace(-4*s,4*s,500)
    axes[0].plot(grid,norm.pdf(grid,scale=s),'--',color='.4');axes[0].set(title='Conditional prior · fitted noise',xlabel='β · prior X range',ylabel='Density')
    for i in range(h):
        lower=float(t.ppf(.005,stats['df'],loc=stats['mean'][i],scale=stats['scale'][i]));upper=float(t.ppf(.995,stats['df'],loc=stats['mean'][i],scale=stats['scale'][i]));pad=.04*(upper-lower)
        ax=axes[i+1];grid=np.sort(np.r_[np.linspace(lower-pad,upper+pad,600),np.linspace(stats['low'][i],stats['high'][i],300)])
        density=t.pdf(grid,stats['df'],loc=stats['mean'][i],scale=stats['scale'][i]);ax.plot(grid,density,color=colors[i])
        ax.fill_between(grid,0,density,where=(grid>=stats['low'][i])&(grid<=stats['high'][i]),color=colors[i],alpha=.2)
        ax.axvline(0,color='.5',lw=.8);ax.axvline(1/h,color='.5',ls='--',lw=.8)
        ax.set(title=f'{labels[i]} · P(β > 0) = {stats["probability_positive"][i]:.3f}',xlabel='β · central 99% posterior range',ylabel='Density',xlim=(lower-pad,upper+pad))
    for ax in axes:ax.grid(alpha=.12)
    fig.suptitle('Exact Student-t coefficient marginals · shaded 95% credible intervals')
    save(fig,destination,'densities')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,matrix,title in zip(axes,[stats['correlation'],function_corr],['Posterior coefficient correlation','Training-function correlation']):
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dddddd')
        im=ax.imshow(np.ma.masked_invalid(matrix),vmin=-1,vmax=1,cmap=cmap)
        ax.set(xticks=range(h),xticklabels=range(1,h+1),yticks=range(h),yticklabels=range(1,h+1),xlabel='Network',ylabel='Network',title=title)
        for i in range(h):
            for j in range(h):ax.text(j,i,f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else '—',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    save(fig,destination,'correlations')
    grid=data['x_grid'].reshape(-1);pred=ym+ys*(heads['grid']@stats['mean'])
    for mode,suffix in [('train',''),('test','_test'),('both','_both'),('none','_none')]:
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for i in range(h):
            axes[0].plot(grid,ym+ys*heads['grid'][:,i],color=colors[i],lw=1,alpha=.8,label=labels[i])
            axes[1].plot(grid,ys*heads['grid'][:,i]*stats['mean'][i],color=colors[i],lw=1,label=f'β{i+1} mean × feature {i+1}')
        axes[0].plot(grid,data['f_grid'].reshape(-1),'k--',lw=1,label='True function')
        axes[0].plot(grid,ym+ys*heads['grid'].mean(axis=1),color='.4',ls=':',lw=2,label='DE mean')
        axes[0].plot(grid,pred,color='black',lw=1.8,label='BDKN mean')
        axes[1].plot(grid,pred,color='black',lw=1.8,label='Sum + target offset')
        for mode_key,color,size,alpha in [('train','#172033',12,.45),('test','#b36a30',6,.25)]:
            if mode in (mode_key,'both'):axes[0].scatter(data['x_'+mode_key],data['y_'+mode_key],color=color,s=size,alpha=alpha,label=mode_key.capitalize()+' observations')
        axes[0].set(title='Individual networks and combined predictions',ylabel='y');axes[1].set(title='Posterior-mean contributions to the combined prediction',xlabel='x',ylabel='Contribution in target units')
        for ax in axes:ax.set_xlim(grid.min(),grid.max());ax.grid(alpha=.12);ax.legend(fontsize=7,ncol=3)
        save(fig,destination,'functions'+suffix)


def build_one(args):
    folder,data_path,destination=args
    return prepare(Path(folder),Path(data_path),Path(destination))['method']


def generate(registry,root,workers=2):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    jobs={};references={}
    for entry in registry['experiments']:
        run=Path(entry['run']);data=Path(json.loads((run/'run.json').read_text())['data']);references[entry['id']]={}
        for method in METHODS:
            folder=run/method;digest=digest_for(folder,data);destination=root/'aggregation-assets'/digest
            references[entry['id']][method]=digest
            if not (destination/'complete.json').exists():jobs[digest]=(str(folder),str(data),str(destination))
    root.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures={executor.submit(build_one,args):digest for digest,args in jobs.items()}
        for count,future in enumerate(as_completed(futures),1):
            method=future.result();digest=futures[future]
            (root/'aggregation-assets'/digest/'complete.json').write_text(json.dumps(dict(version=VERSION,method=method))+'\n')
            if count%10==0 or count==len(jobs):print(f'Aggregation diagnostics: {count}/{len(jobs)} unique fits',flush=True)
    (root/'aggregation-index.json').write_text(json.dumps(references,indent=2)+'\n')
    return references


def render(ctx):
    from report_ui import page,image,choices,observation_controls,metric_cell,latex
    entry=ctx['entry'];output=ctx['output'];root=output if not entry['directory'] else output.parent.parent
    index_path=root/'aggregation-index.json'
    if not index_path.exists():return
    index=json.loads(index_path.read_text());prefix='' if not entry['directory'] else '../../'
    mapping_path=output/'asset-map.js';mapping=json.loads(mapping_path.read_text().split('=',1)[1].strip().removesuffix(';')) if mapping_path.exists() else {}
    body='<section><h2>Aggregation weights</h2>'+choices('aggregation-method','BDKN variant',[('all','All')]+[(m,ctx['names'][m]) for m in METHODS],'bdkn')+observation_controls()+latex(r'\widehat y(x)=y_{\rm mean}+y_{\rm std}\sum_{h=1}^{H}\beta_h\mu_h(x)')+'<p class="note">Coefficients can be negative and need not sum to one. Intervals describe coefficient uncertainty; the dashed reference is DE’s fixed 1/H weight.</p></section>'
    for method in METHODS:
        digest=index[entry['id']][method];folder=root/'aggregation-assets'/digest;stats=json.loads((folder/'statistics.json').read_text())
        for name in ['coefficients','densities','correlations','functions','functions_test','functions_both','functions_none']:
            mapping['assets/aggregation_'+method+'_'+name+'.png']=prefix+'aggregation-assets/'+digest+'/'+name+'.png'
        body+='<section id="aggregation-'+method+'" data-aggregation-method="'+method+'"'+(' hidden' if method!='bdkn' else '')+'><h2>'+ctx['names'][method]+'</h2><div class="gallery">'
        for name,caption,wide,obs in [('coefficients','Posterior mean and 95% coefficient intervals',False,False),('correlations','Coefficient and function correlations',False,False),('densities','Coefficient distributions · conditional prior and exact Student-t posterior',True,False),('functions','Individual networks and their weighted contributions',True,True)]:
            body+=image('aggregation_'+method+'_'+name+'.png',caption,wide,obs)
        body+='</div><p class="note">Each density panel uses its own X range for readability; coefficient intervals above share a common axis. Prior reference conditions on fitted noise. Contribution shares use held-out inputs; test targets are used only for scoring.</p><div class="table-wrap"><table><thead><tr><th>Network</th><th>β mean</th><th>95% low</th><th>95% high</th><th>P(β &gt; 0)</th><th>Contribution share</th><th>Test RMSE</th><th>Test NLL</th></tr></thead><tbody>'
        for row in stats['rows']:
            body+='<tr><th>'+str(row['network'])+'</th>'+''.join('<td>'+format(row[k],'.5f')+'</td>' for k in ('mean','ci95_low','ci95_high','probability_positive','mean_absolute_test_contribution_share'))
            for k in ('test_rmse','test_nll'):body+=metric_cell(row[k],[r[k] for r in stats['rows']])
            body+='</tr>'
        base=prefix+'aggregation-assets/'+digest+'/'
        body+='</tbody></table></div><p class="downloads"><a href="'+base+'weights.csv" download>Weights CSV</a><a href="'+base+'statistics.json" download>Posterior statistics</a><a href="'+base+'functions.npz" download>Functions and contributions</a></p></section>'
    mapping_path.write_text('globalThis.REPORT_ASSETS = '+json.dumps(mapping,separators=(',',':'))+';\n')
    (output/'aggregation.html').write_text(page(ctx,'Aggregation weights','aggregation.html',body))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registry',default='website/experiments.json');parser.add_argument('--output',default='website/dist');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    from report_ui import load_registry
    generate(load_registry(args.registry,Path.cwd()),Path(args.output).resolve(),args.workers)
