"""Apply the existing PyTorch BDKN stage to frozen author JAX mean functions."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))


def finalize(folder,device):
    import numpy as np
    import torch
    from scipy.stats import t
    import bdkn as b
    from official_forde import atomic_json
    torch.set_num_threads(1);device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable for BDKN')
    cfg=json.loads((folder/'config.json').read_text())
    with np.load(cfg['data'],allow_pickle=False) as saved: data={key:saved[key] for key in saved.files}
    with np.load(folder/'members.npz',allow_pickle=False) as saved:
        phi={key:torch.tensor(saved[key].T,dtype=torch.float64,device=device) for key in saved.files}
    y=torch.tensor(data['y_train'],dtype=torch.float64,device=device)
    design=b.prepare_bll_design(phi['x_train'],y)
    log_sigma=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64,device=device))
    optimizer=torch.optim.Adam([log_sigma],lr=cfg['stage2_lr'])
    with (folder/'history.csv').open() as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;history=list(reader)
    def clock():
        if device.type=='cuda':torch.cuda.synchronize(device)
        return time.perf_counter()
    start=clock()
    for step in range(1,cfg['stage2_steps']+1):
        optimizer.zero_grad();loss=b.nig_nll_feature(design,y,log_sigma,0.,0.)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite BDKN marginal likelihood')
        loss.backward();optimizer.step()
        row={key:0 for key in fields};row.update(stage=2,epoch=step,objective=float(loss.detach()),likelihood=float(loss.detach()),lr=cfg['stage2_lr'])
        history.append(row)
    posterior=b.compute_bll_posterior(design,y,log_sigma.exp().detach(),0.,0.)
    seconds=clock()-start;df=float(2*posterior[2])
    label=('author_forde' if cfg['method']=='forde' else 'author_forde_control')+'_bdkn'
    target=folder/label;target.mkdir()
    arrays={key:data[key] for key in ('x_test','y_test','x_grid','f_grid')};arrays['df']=df
    for split in ('test','grid'):
        mean,lower,upper=b.bll_student_t_mu_lo_hi(phi['x_'+split],*posterior)
        array=lambda tensor:tensor.detach().cpu().numpy().reshape(-1)
        for key,value in dict(mean=mean,lower=lower,upper=upper,scale=(upper-lower)/(2*t.ppf(.975,df))).items():
            arrays[split+'_'+key]=array(value)
            if not np.isfinite(arrays[split+'_'+key]).all():raise FloatingPointError('Invalid BDKN predictions')
        if np.any(arrays[split+'_scale']<=0):raise FloatingPointError('Nonpositive BDKN scale')
    def score(indices,truth):
        mu=arrays['test_mean'][indices];scale=arrays['test_scale'][indices]
        return dict(rmse=float(np.sqrt(np.mean((truth.reshape(-1)-mu)**2))),
                    nll=float(-t.logpdf(truth.reshape(-1),df=df,loc=mu,scale=scale).mean()))
    row=dict(method=label,**score(slice(None),data['y_test']),dataset_sha256=cfg['dataset_sha256'],
        predictive_family='student_t',training_seconds=cfg['stage1_training_seconds']+seconds,
        stage1_training_seconds=cfg['stage1_training_seconds'],stage2_training_seconds=seconds)
    if 'native_test_indices' in data:row['native']=score(data['native_test_indices'],data['y_native'])
    if not np.isfinite([row['rmse'],row['nll']]).all():raise FloatingPointError('Invalid BDKN metrics')
    np.savez_compressed(target/'predictions.npz',**arrays);atomic_json(target/'metrics.json',row)
    m,p,a,beta=posterior
    precision=p.cpu() if isinstance(p,torch.Tensor) else dict(right_vectors=p.right_vectors.cpu(),eigenvalues=p.eigenvalues.cpu())
    torch.save(dict(mean=m.cpu(),precision=precision,alpha=a.cpu(),beta=beta.cpu(),sigma_p2=log_sigma.exp().detach().cpu()),folder/'posterior.pt')
    with (folder/'history.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(history)
    rows=json.loads((folder/'stage1_metrics.json').read_text())+[row]
    atomic_json(folder/'metrics.json',rows)  # Completion marker comes last.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();finalize(args.folder,args.device)
