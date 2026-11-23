"""Author FoRDE JAX core with an explicitly separate scalar regression adapter.

Run through run_official_forde.py, or edit CONFIG and supply a PyTorch Python
via --torch-python for the unchanged BDKN second stage.
"""
CONFIG = {
    'recipe': 'dangelo',  # D'Angelo regression setup + literal author FoRDE repulsion.
    'preset': 'low',  # low, high, author_style
    'method': 'forde',  # forde, control (same setup, zero repulsion)
    'data': 'data/author-xsine.npz',
    'output': 'results/official_forde/manual',
    'device': 'cuda:0',
    'seed': 123,
    'batch_size': 32,
    'hidden_dim': 50,
    'normalization_eps': 1e-24,  # README experiment setting, preserved.
    'weight_decay': 5e-4,
    'likelihood_std': 1.,
    'stage2_steps': 100,
    'stage2_lr': .1,
}

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from official_forde_core import LOCK, repulsion_function, optimizer, scheduler, training_pca


def settings(preset):
    return {'low':dict(epochs=2000,lr=.001,n_particles=5,lr_ratio=1.),
            'high':dict(epochs=40,lr=.1,n_particles=5,lr_ratio=1.),
            'author_style':dict(epochs=300,lr=.1,n_particles=10,lr_ratio=.01)}[preset].copy()


def dangelo_settings(preset,n_train):
    from author_regression import settings as reference
    selected=reference(preset,n_train)
    return dict(epochs=selected['steps'],steps=selected['steps'],lr=selected['lr'],
        n_particles=selected['n_particles'],hidden_dim=selected['size_hidden'],seed=selected['seed'],
        lr_ratio=1.,normalize_x=selected['normalize_x'],batch_size=selected.get('batch_size',32))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def make_model(hidden_dim=50):
    import haiku as hk
    import jax
    def forward(x): return hk.nets.MLP([hidden_dim,hidden_dim,1],activation=jax.nn.relu)(x)
    return hk.without_apply_rng(hk.transform(forward))


def make_step(model,n_members,batch_size,eps,wd,repulsion_enabled,pca_vectors,pca_values,likelihood_std=1.,dangelo=False,n_train=1):
    """Mirror the author's nested VJP and single-device optimizer update."""
    import jax
    import jax.numpy as jnp
    dimension=pca_vectors.shape[0]
    core=repulsion_function(n_members,batch_size,dimension,eps,pca_vectors,pca_values)
    _,opt_update,get_params,get_velocity=optimizer()
    if dangelo:
        import optax
        adam=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8,eps_root=0.)
    def outputs(params,x):
        def member(p):
            mu,vjp=jax.vjp(lambda inputs:model.apply(p,inputs),x)
            derivative=vjp(jnp.ones_like(mu))[0]
            return mu[:,0],derivative
        return jax.vmap(member)(params)
    def squared_norm(tree): return sum(jnp.sum(jnp.square(v)) for v in jax.tree_util.tree_leaves(tree))
    def step(params,velocity,x,y,lr):
        (mu,jacobian),vjp=jax.vjp(lambda p:outputs(p,x),params)
        def gaussian_loss(value):
            if dangelo:
                return .5*jnp.square((value-y[:,0][None,:])/likelihood_std).mean(1).sum()*n_train
            return (.5*jnp.square((value-y[:,0][None,:])/likelihood_std)
                    +jnp.log(likelihood_std)+.5*jnp.log(2*jnp.pi)).mean(1).sum()
        data_loss,mu_grad=jax.value_and_grad(gaussian_loss)(mu)
        native_jacobian=jacobian[:,None,:,:]  # Upstream H × devices × B × D.
        (repulsion,median),jac_grad=jax.value_and_grad(core,has_aux=True)(native_jacobian)
        if not repulsion_enabled:
            repulsion=jnp.zeros_like(repulsion);jac_grad=jnp.zeros_like(jac_grad)
        jac_grad=jac_grad[:,0]
        # Separate diagnostic VJPs do not change the actual combined VJP.
        data_grads=vjp((mu_grad,jnp.zeros_like(jacobian)))[0]
        if dangelo:data_grads=jax.tree_util.tree_map(lambda g,p:g+wd*p,data_grads,params)
        rep_grads=vjp((jnp.zeros_like(mu),jac_grad))[0]
        grads=vjp((mu_grad,jac_grad))[0]
        grads=jax.tree_util.tree_map(lambda g,p:g+wd*p,grads,params)
        if dangelo:
            updates,new_velocity=adam.update(grads,velocity,params)
            new_params=jax.tree_util.tree_map(lambda p,u:p-lr*u,params,updates)
        else:
            updated=jax.tree_util.tree_map(lambda g,p,v:opt_update(lr,g,p,v),grads,params,velocity)
            leaf=lambda value:isinstance(value,tuple) and len(value)==2
            new_params=jax.tree_util.tree_map(get_params,updated,is_leaf=leaf)
            new_velocity=jax.tree_util.tree_map(get_velocity,updated,is_leaf=leaf)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(p)) for p in jax.tree_util.tree_leaves(new_params)]))
        prior_constant=sum(p.size for p in jax.tree_util.tree_leaves(params))*.5*jnp.log(2*jnp.pi) if dangelo else 0.
        diagnostics=jnp.array([data_loss+repulsion+.5*wd*squared_norm(params)+prior_constant,data_loss,repulsion,
            median,jnp.sqrt(squared_norm(data_grads)),jnp.sqrt(squared_norm(rep_grads)),finite])
        return new_params,new_velocity,diagnostics
    return jax.jit(step)


def fit(cfg,torch_python,override_epochs=None,skip_bdkn=False,initialization=None):
    # Limit JAX's CPU thread pools per concurrent process before initialization.
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[os.getpid()%len(cpus)]})
    if cfg['device']=='cuda:0':
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible not in (None,'0'): raise RuntimeError('Physical GPU must be 0')
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['JAX_PLATFORMS']='cuda'
    else: os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import numpy as np
    from scipy.stats import norm
    if cfg['device']=='cuda:0' and not all(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Requested GPU unavailable; no CPU fallback')
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'.cache/jax_forde'))
    path=Path(cfg['data']).resolve();output=Path(cfg['output']).resolve()
    output.mkdir(parents=True,exist_ok=False)
    with np.load(path,allow_pickle=False) as saved: data={k:saved[k] for k in saved.files}
    dangelo=cfg.get('recipe')=='dangelo'
    selected=dangelo_settings(cfg['preset'],len(data['x_train'])) if dangelo else settings(cfg['preset'])
    selected.update(cfg.get('run_settings',{}))
    if override_epochs is not None:selected['epochs']=override_epochs
    cfg=cfg|selected
    if dangelo:cfg['weight_decay']=1.  # Exact N(0,I) parameter prior, rather than image weight decay.
    metadata=json.loads(str(data['metadata']))
    dataset=metadata.get('dataset','steps' if path.name=='toy_1d.npz' else path.stem)
    normalize=selected.get('normalize_x',True)
    offset=data['x_train'].mean(0) if normalize else np.zeros(data['x_train'].shape[1])
    scale=data['x_train'].std(0) if normalize else np.ones(data['x_train'].shape[1]);scale=np.where(scale==0,1.,scale)
    x={key:np.asarray((data[key]-offset)/scale,dtype=np.float32) for key in ('x_train','x_test','x_grid')}
    y=np.asarray(data['y_train'],dtype=np.float32)
    pca_vectors,pca_values=training_pca(x['x_train'])
    model=make_model(cfg['hidden_dim'])
    # Match the authors' independent per-network JAX key splitting.
    _,*keys=jax.random.split(jax.random.PRNGKey(cfg['seed']),selected['n_particles']+1)
    params=jax.vmap(model.init,(0,None))(jnp.stack(keys),jnp.ones((1,data['x_train'].shape[1]),dtype=jnp.float32))
    init_data=None
    if dangelo:
        if initialization is None:raise ValueError('D\'Angelo recipe requires --initialization prepared by prepare_dangelo.py')
        with np.load(initialization,allow_pickle=False) as saved:init_data={k:saved[k] for k in saved.files}
        init_meta=json.loads(str(init_data['metadata']))
        if init_meta['dataset_sha256']!=digest(path) or init_meta['preset']!=cfg['preset']:raise ValueError('Initialization/data mismatch')
        particles=init_data['particles'];shapes=json.loads(str(init_data['shapes']))
        flat={};position=0
        for layer in range(3):
            wshape,bshape=shapes[2*layer:2*layer+2];count=int(np.prod(wshape))
            weights=particles[:,position:position+count].reshape((len(particles),*wshape));position+=count
            count=int(np.prod(bshape));bias=particles[:,position:position+count].reshape((len(particles),*bshape));position+=count
            flat[f'mlp/~/linear_{layer}']=dict(w=jnp.array(weights.transpose(0,2,1)),b=jnp.array(bias))
        params=flat
    if dangelo:
        import optax
        velocity=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8,eps_root=0.).init(params)
    else:
        opt_init,_,_,_=optimizer();velocity=jax.tree_util.tree_map(opt_init,params)
    def save_params(filename,values):
        flat=hk.data_structures.to_mutable_dict(values)
        np.savez_compressed(filename,**{layer+'/'+name:np.asarray(value) for layer,weights in flat.items() for name,value in weights.items()})
    save_params(output/'initial_params.npz',params)
    config=dict(cfg,data=str(path),dataset=dataset,dataset_sha256=digest(path),
        source_revision=LOCK['revision'],smoke_test=override_epochs is not None,
        stage2_enabled=not skip_bdkn,jax_version=jax.__version__,haiku_version=hk.__version__,
        numpy_version=np.__version__,pca=True,normalize_x=normalize,
        initialization=str(initialization) if initialization else None,
        initialization_sha256=digest(initialization) if initialization else None,
        adaptation='Dangelo scalar Gaussian regression, N(0,I) prior, Adam, original initialization/batches; author FoRDE repulsion' if dangelo else 'Native FoRDE optimization with scalar Gaussian regression',
        source_sha256={name:digest(ROOT/name) for name in ('official_forde.py','official_forde_core.py','official_forde_bayes.py')})
    atomic_json(output/'config.json',config);atomic_json(output/'source.json',LOCK)
    np.savez_compressed(output/'preprocessing.npz',x_mean=offset,x_std=scale,pca_vectors=pca_vectors,pca_values=pca_values)
    lrs=scheduler(selected['lr'],selected['epochs'],selected['lr_ratio']).lrs
    batchers={};rng=np.random.RandomState(cfg['seed']);history=[]
    if dangelo:
        from official_forde_core import dangelo_batch_generator
        np.random.set_state(('MT19937',init_data['rng_keys'],int(init_data['rng_pos']),int(init_data['rng_has_gauss']),float(init_data['rng_cached_gaussian'])))
        train_indices=dangelo_batch_generator(np.arange(len(y)))
        test_indices=dangelo_batch_generator(np.arange(init_meta['native_test_count']))
    # Timings explicitly include first-use JIT compilation, as well as execution.
    jax.block_until_ready(params);start=time.perf_counter()
    for epoch in range(selected['epochs']):
        if dangelo:
            batches=[np.fromiter(train_indices,int,count=cfg['batch_size'])]
            np.fromiter(test_indices,int,count=cfg['batch_size'])  # Preserve upstream test RNG consumption.
        else:
            permutation=rng.permutation(len(y));batches=[permutation[pos:pos+cfg['batch_size']] for pos in range(0,len(y),cfg['batch_size'])]
        diagnostics=[]
        for indices in batches:
            b=len(indices)
            if b not in batchers:
                batchers[b]=make_step(model,selected['n_particles'],b,cfg['normalization_eps'],cfg['weight_decay'],
                    cfg['method']=='forde',jnp.array(pca_vectors),jnp.array(pca_values),cfg['likelihood_std'],dangelo,len(y))
            params,velocity,values=batchers[b](params,velocity,jnp.array(x['x_train'][indices]),jnp.array(y[indices]),lrs[epoch])
            values=np.asarray(values)
            if not np.isfinite(values).all() or values[-1]!=1: raise FloatingPointError('Nonfinite JAX update; settings unchanged')
            diagnostics.append(values)
        d=np.mean(diagnostics,axis=0)
        history.append(dict(stage=1,epoch=epoch+1,objective=float(d[0]),likelihood=float(d[1]),repulsion=float(d[2]),
            median=float(d[3]),driving=float(d[4]),repulsive_force=float(d[5]),lr=float(lrs[epoch])))
        if epoch==0 or (epoch+1)%100==0: print(f'{cfg["method"]}: epoch {epoch+1}/{selected["epochs"]}',flush=True)
    jax.block_until_ready(params);seconds=time.perf_counter()-start
    save_params(output/'stage1_params.npz',params)
    config.update(stage1_training_seconds=seconds,timing_includes_jit=True)
    atomic_json(output/'config.json',config)
    members={}
    predict=jax.jit(jax.vmap(model.apply,(0,None)))
    for key,values in x.items():
        parts=[np.asarray(predict(params,jnp.array(values[start:start+512])))[...,0] for start in range(0,len(values),512)]
        members[key]=np.concatenate(parts,axis=1)
    np.savez_compressed(output/'members.npz',**members)
    label='author_forde' if cfg['method']=='forde' else 'author_forde_control'
    folder=output/label;folder.mkdir()
    predictions={key:data[key] for key in ('x_test','y_test','x_grid','f_grid')}
    predictions['df']=np.nan
    for split in ('test','grid'):
        values=members['x_'+split].astype(np.float64);mu=values.mean(0)
        sigma=np.sqrt(values.var(0)+cfg['likelihood_std']**2)
        predictions.update({split+'_mean':mu,split+'_scale':sigma,split+'_lower':mu-1.959963984540054*sigma,
                            split+'_upper':mu+1.959963984540054*sigma})
    np.savez_compressed(folder/'predictions.npz',**predictions)
    def score(indices,target):
        mu=predictions['test_mean'][indices];sigma=predictions['test_scale'][indices]
        return dict(rmse=float(np.sqrt(np.mean((target.reshape(-1)-mu)**2))),
                    nll=float(-norm.logpdf(target.reshape(-1),loc=mu,scale=sigma).mean()))
    row=dict(method=label,**score(slice(None),data['y_test']),dataset_sha256=config['dataset_sha256'],
        predictive_family='gaussian',training_seconds=seconds,stage1_training_seconds=seconds,stage2_training_seconds=0.)
    if 'native_test_indices' in data: row['native']=score(data['native_test_indices'],data['y_native'])
    if not np.isfinite([row['rmse'],row['nll']]).all(): raise FloatingPointError('Invalid Gaussian metrics')
    atomic_json(folder/'metrics.json',row)
    atomic_json(output/'stage1_metrics.json',[row])
    np.savez_compressed(output/'native_uncertainty.npz',x=data['x_grid'],mean=members['x_grid'].mean(0),std=members['x_grid'].std(0,ddof=1))
    with (output/'history.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    if skip_bdkn: atomic_json(output/'metrics.json',[row])
    else:
        # Release live executables/arrays before the separate PyTorch process.
        del params,velocity,batchers,predict
        jax.clear_caches()
        subprocess.run([torch_python,str(ROOT/'official_forde_bayes.py'),'--folder',str(output),'--device',cfg['device']],check=True)
    print('Saved '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','output','preset','method','device','recipe'): parser.add_argument('--'+key,default=CONFIG[key])
    parser.add_argument('--initialization',type=Path)
    parser.add_argument('--overrides',help='Runner-generated JSON with explicit training/stage-two overrides')
    parser.add_argument('--torch-python',default=os.environ.get('BDKN_PYTHON',sys.executable))
    parser.add_argument('--epochs',type=int,help='Smoke budget; not accepted as a full result')
    parser.add_argument('--skip-bdkn',action='store_true')
    args=parser.parse_args()
    if args.preset not in ('paper','upstream','low','high','author_style') or args.method not in ('forde','control') or args.device not in ('cpu','cuda:0'):
        parser.error('Invalid preset/method/device')
    if args.epochs is not None and args.epochs<1: parser.error('epochs must be positive')
    cfg=CONFIG|{key:getattr(args,key) for key in ('data','output','preset','method','device','recipe')}
    if args.overrides:cfg.update(json.loads(Path(args.overrides).read_text()))
    try: fit(cfg,args.torch_python,args.epochs,args.skip_bdkn,args.initialization)
    except Exception as error:
        folder=Path(cfg['output'])
        if folder.exists(): atomic_json(folder/'failure.json',dict(error=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__': main()
