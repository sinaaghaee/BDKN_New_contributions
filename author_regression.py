"""Standalone upstream regression runner; edit CONFIG, or use run_author.py.

Training calls the pinned upstream training_1d_regre.train function itself.
Observation hooks export results without adding random draws or optimizer steps.
"""
CONFIG = {
    'method': 'kde_f_WGD',  # SGD, kde_WGD, kde_f_WGD (upstream identifiers)
    'preset': 'upstream',  # upstream, low, high
    'data': 'data/author-xsine.npz',
    'output': 'results/author/manual',
    'device': 'cuda:0',
    'bdkn': True,
    'cpu_threads': 1,
    'stage2_steps': 100,
    'stage2_lr': .1,
}

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
LABELS = {'SGD': 'author_sgd', 'kde_WGD': 'author_kde_wgd', 'kde_f_WGD': 'author_kde_f_wgd'}


def settings(preset, n_train):
    if preset == 'upstream':
        return dict(steps=5000, lr=.01, n_particles=100, size_hidden=10, seed=42, normalize_x=False)
    if preset not in ('low', 'high'):
        raise ValueError(preset)
    passes = 2000 if preset == 'low' else 40
    return dict(steps=passes*math.ceil(n_train/32), lr=.001 if preset == 'low' else .1,
                n_particles=5, size_hidden=50, seed=123, normalize_x=True, passes=passes)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fit(cfg, override_steps=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import norm, t
    from author_code import load_upstream, array_handler, LOCK, COMPATIBILITY
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; author experiments will not silently run on CPU')
    data_path = Path(cfg['data']).resolve()
    with np.load(data_path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    meta = json.loads(str(data['metadata']))
    dataset_name = meta.get('dataset', 'steps' if data_path.name=='toy_1d.npz' else data_path.stem)
    selected = settings(cfg['preset'], len(data['x_train']))
    selected.update(cfg.get('run_settings',{}))
    if override_steps is not None:
        selected['steps'] = override_steps
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = load_upstream().modules
    arguments = ['--epochs', str(selected['steps']), '--lr', str(selected['lr']),
        '--n_particles', str(selected['n_particles']), '--size_hidden', str(selected['size_hidden']),
        '--num_hidden', '2', '--method', cfg['method'], '--prior_variance', '1',
        '--annealing_steps', '1000', '--batch_size', '32', '--dataset', 'toy_reg',
        '--ann_sch', 'None', '--random_seed', str(selected['seed'])]
    author_config = upstream['utils.config'].configuration(arguments)
    # This is the entrypoint's RNG/model/data construction order.
    torch.manual_seed(author_config.random_seed)
    handler, x_offset, x_scale = array_handler(data, native=dataset_name=='author-xsine',
                                             normalize=selected['normalize_x'])
    net = upstream['models.MLP'].Net([data['x_train'].shape[1], selected['size_hidden'], selected['size_hidden'], 1],
                                    classification=False, act=F.relu, out_act=None).to(device)
    ensemble = upstream['models.ensemble'].Ensemble(device=device, net=net,
                                                  n_particles=selected['n_particles'])
    np.savez_compressed(output/'initial_particles.npz', particles=ensemble.particles.cpu().numpy())
    config = dict(cfg, **selected, dataset=dataset_name, dataset_sha256=digest(data_path),
                  data=str(data_path), source_revision=LOCK['revision'], compatibility=COMPATIBILITY,
                  upstream_config=vars(author_config), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, smoke_test=override_steps is not None)
    atomic_json(output/'config.json', config)
    atomic_json(output/'source.json', LOCK)
    history, scalars = [], []
    class Writer:
        def add_scalar(self, tag, value, step):
            if isinstance(value,torch.Tensor): value=value.detach()
            scalars.append(dict(tag=tag, value=float(value), step=step))
    trainer = upstream['training.training_1d_regre']
    factory = upstream['methods.method_utils'].create_method
    def observed_factory(*args, **kwargs):
        method = factory(*args, **kwargs)
        original_step = method.step
        def step(W, X, T, *step_args, **step_kwargs):
            result = original_step(W, X, T, *step_args, **step_kwargs)
            if not torch.isfinite(W).all():
                raise FloatingPointError('Nonfinite upstream particles; no settings were changed')
            with torch.no_grad():
                objective = -method.P.log_prob(W, X, T).mean().item()
                mse_half, _ = method.P.log_prob(W, X, T, return_loss=True)
            driving = float(result[0].detach().abs().mean()) if result is not None else 0.
            repulsive = float(result[1].detach().abs().mean()) if result is not None else 0.
            if not np.isfinite([objective, float(mse_half), driving, repulsive]).all():
                raise FloatingPointError('Nonfinite upstream objective/force')
            history.append(dict(stage=1, epoch=len(history)+1, objective=objective,
                                likelihood=float(mse_half), driving=driving, repulsive_force=repulsive))
            return result
        method.step = step
        return method
    # Plotting/logging consumes no training RNG. Native bands are exported below.
    trainer.create_method = observed_factory
    original_plot = trainer.plot_predictive_distributions
    trainer.plot_predictive_distributions = lambda *a, **kw: None
    def clock():
        if device.type == 'cuda': torch.cuda.synchronize(device)
        return time.perf_counter()
    start = clock()
    try:
        trainer.train(handler, ensemble, device, author_config, Writer())
    finally:
        trainer.create_method = factory
        trainer.plot_predictive_distributions = original_plot
    stage1_seconds = clock()-start
    torch.save(dict(particles=ensemble.particles.detach().cpu(), layers=net.layer_sizes,
                    config=config, training_seconds=stage1_seconds), output/'stage1.pt')
    tensors = {key: torch.as_tensor((data[key]-x_offset)/x_scale,
                dtype=torch.float32, device=device) for key in ('x_train', 'x_test', 'x_grid')}
    with torch.no_grad():
        heads = {key: ensemble.forward(value)[0].squeeze(-1).detach()
                 for key, value in tensors.items()}
    np.savez_compressed(output/'members.npz', **{key:value.cpu().numpy() for key,value in heads.items()})
    np.savez_compressed(output/'preprocessing.npz', x_mean=x_offset, x_std=x_scale)
    metrics = []
    label = LABELS[cfg['method']]
    def export(name, predictions, family, df=None, stage2_seconds=0.):
        folder = output/name
        folder.mkdir()
        arrays = dict(x_test=data['x_test'], y_test=data['y_test'], x_grid=data['x_grid'],
                      f_grid=data['f_grid'], df=np.nan if df is None else df)
        for split, p in predictions.items():
            for key, value in p.items():
                a = np.asarray(value).reshape(-1)
                if not np.isfinite(a).all(): raise FloatingPointError('Nonfinite exported prediction')
                arrays[split+'_'+key] = a
        for split in ('test', 'grid'):
            if (arrays[split+'_scale'] <= 0).any(): raise FloatingPointError('Nonpositive predictive scale')
        kwargs = {} if df is None else dict(df=df)
        distribution = norm if df is None else t
        def score(indices, targets):
            mean, scale = arrays['test_mean'][indices], arrays['test_scale'][indices]
            return dict(rmse=float(np.sqrt(np.mean((targets.reshape(-1)-mean)**2))),
                        nll=float(-distribution.logpdf(targets.reshape(-1),loc=mean,scale=scale,**kwargs).mean()))
        row = dict(method=name, **score(slice(None), data['y_test']), predictive_family=family,
                   dataset_sha256=config['dataset_sha256'], training_seconds=stage1_seconds+stage2_seconds,
                   stage1_training_seconds=stage1_seconds, stage2_training_seconds=stage2_seconds)
        if 'native_test_indices' in data:
            row['native'] = score(data['native_test_indices'], data['y_native'])
        if not np.isfinite([row['rmse'],row['nll']]).all(): raise FloatingPointError('Invalid metrics')
        np.savez_compressed(folder/'predictions.npz', **arrays)
        atomic_json(folder/'metrics.json', row)
        metrics.append(row)
    predictions = {}
    for split in ('test', 'grid'):
        values = heads['x_'+split].cpu().numpy().astype(np.float64)
        mean = values.mean(0)
        scale = np.sqrt(values.var(0)+author_config.pred_dist_std**2)
        predictions[split] = dict(mean=mean, scale=scale, lower=mean-1.959963984540054*scale,
                                  upper=mean+1.959963984540054*scale)
    export(label, predictions, 'gaussian')
    grid_members = heads['x_grid'].cpu().numpy()
    np.savez_compressed(output/'native_uncertainty.npz', x=data['x_grid'],
                       mean=grid_members.mean(0), std=grid_members.std(0, ddof=1))
    if cfg['bdkn']:
        import bdkn as b
        y = torch.as_tensor(data['y_train'],dtype=torch.float64,device=device)
        features = {key:value.T.to(torch.float64) for key,value in heads.items()}
        design = b.prepare_bll_design(features['x_train'],y)
        log_sigma = torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64,device=device))
        optimizer = torch.optim.Adam([log_sigma],lr=cfg['stage2_lr'])
        start2 = clock()
        for step in range(1,cfg['stage2_steps']+1):
            optimizer.zero_grad()
            loss = b.nig_nll_feature(design,y,log_sigma,0.,0.)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite BDKN second stage')
            loss.backward(); optimizer.step()
            history.append(dict(stage=2,epoch=step,objective=float(loss.detach()),
                                likelihood=float(loss.detach()),driving=0.,repulsive_force=0.))
        posterior = b.compute_bll_posterior(design,y,log_sigma.exp().detach(),0.,0.)
        seconds2 = clock()-start2
        df = float(2*posterior[2])
        predictions = {}
        for split in ('test','grid'):
            mean,lo,hi = b.bll_student_t_mu_lo_hi(features['x_'+split],*posterior)
            a = lambda value: value.detach().cpu().numpy().reshape(-1)
            predictions[split] = dict(mean=a(mean), lower=a(lo), upper=a(hi), scale=a((hi-lo)/(2*t.ppf(.975,df))))
        export(label+'_bdkn',predictions,'student_t',df,seconds2)
        m,p,a,beta = posterior
        precision = p.cpu() if isinstance(p,torch.Tensor) else dict(right_vectors=p.right_vectors.cpu(),eigenvalues=p.eigenvalues.cpu())
        torch.save(dict(mean=m.cpu(),precision=precision,alpha=a.cpu(),beta=beta.cpu(),
                        sigma_p2=log_sigma.exp().detach().cpu()),output/'posterior.pt')
    with (output/'history.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['stage','epoch','objective','likelihood','driving','repulsive_force'])
        writer.writeheader(); writer.writerows(history)
    atomic_json(output/'upstream_scalars.json', scalars)
    atomic_json(output/'metrics.json', metrics)  # Written last; completion marker.
    print(json.dumps(metrics),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=LABELS,default=CONFIG['method'])
    parser.add_argument('--preset',choices=['upstream','low','high'],default=CONFIG['preset'])
    parser.add_argument('--data',default=CONFIG['data'])
    parser.add_argument('--output',default=CONFIG['output'])
    parser.add_argument('--device',default=CONFIG['device'])
    parser.add_argument('--steps',type=int,help='Smoke-test override; never counted as a full result')
    parser.add_argument('--overrides',help='Runner-generated JSON with explicit training/stage-two overrides')
    args = parser.parse_args()
    cfg = CONFIG | vars(args)
    if args.overrides: cfg.update(json.loads(Path(args.overrides).read_text()))
    steps = cfg.pop('steps')
    if steps is not None and steps < 1: parser.error('--steps must be positive')
    try:
        fit(cfg,steps)
    except Exception as error:
        path = Path(cfg['output'])
        if path.exists(): atomic_json(path/'failure.json',dict(error=type(error).__name__,message=str(error)))
        raise


if __name__ == '__main__': main()
