"""Scalar 1D datasets from the paper and local toy scripts; edit settings below."""
CONFIG = {
    'seed': 123,
    'n_train': 100,
    'n_test': 1000,
    'n_grid': 2000,
    'noise_std': 0.1,  # Paper noise VARIANCE is 0.01.
    'output': 'data/toy_1d.npz',
}
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT = Path(__file__).resolve().parent


def ground_truth(x):
    """Exact smoothed-step formula and oscillation from section 4.1."""
    left, middle, right = expit(200*(x+.6)), expit(200*x), expit(200*(x-.4))
    return .3*(1-left) + .9*(left-middle) - .6*(middle-right) + .01*np.sin(50*np.sin(10*x))


def make_dataset(config):
    for key in ('n_train', 'n_test', 'n_grid'):
        if config[key] < 2: raise ValueError(f'{key} must be at least 2')
    if not np.isfinite(config['noise_std']) or config['noise_std'] < 0:
        raise ValueError('noise_std must be finite and nonnegative')
    # Independent streams keep the test set fixed if only training size changes.
    streams = np.random.SeedSequence(config['seed']).spawn(4)
    tx, tn, vx, vn = map(np.random.default_rng, streams)
    x_train = tx.uniform(-1,1,(config['n_train'],1))
    x_test = vx.uniform(-1,1,(config['n_test'],1))
    x_grid = np.linspace(-1,1,config['n_grid']).reshape(-1,1)
    f_train, f_test, f_grid = map(ground_truth,(x_train,x_test,x_grid))
    return dict(x_train=x_train,y_train=f_train+tn.normal(0,config['noise_std'],f_train.shape),
                x_test=x_test,y_test=f_test+vn.normal(0,config['noise_std'],f_test.shape),
                f_train=f_train,f_test=f_test,x_grid=x_grid,f_grid=f_grid,
                metadata=np.array(json.dumps(config,sort_keys=True)))


# Source settings are active script configurations, not generator argument defaults.
# Each dataset stays scalar so every existing standalone method can read it.
DATASETS = {
    'steps': dict(label='Smoothed steps', n_train=100, noise_std=.1, domain=(-1.,1.), formula=r'0.3[1-\sigma(200(x+0.6))]+0.9[\sigma(200(x+0.6))-\sigma(200x)]-0.6[\sigma(200x)-\sigma(200(x-0.4))]+0.01\sin(50\sin(10x))', source='arXiv:2505.18526 section 4.1', sampling='Uniform(-1, 1)'),
    'cubic': dict(label='Cubic', n_train=20, noise_std=6., domain=(-6.,6.), formula=r'x^3', source='BDKN/Clean_Code/toy_example/x^3with gp.py', sampling='2.5 * clip(Normal(0,1), -2, 2)'),
    'forrester': dict(label='Forrester', n_train=20, noise_std=.1, domain=(0.,1.), formula=r'(6x-2)^2\sin(12x-4)', source='BDKN/Clean_Code/toy_example/forrester.py', sampling='10 equally spaced inputs in each of [0.2,0.4] and [0.65,0.85]'),
    'he': dict(label='He toy', n_train=20, noise_std=.2, domain=(-6.,6.), formula=r'x\sin x', source='BDKN/Clean_Code/toy_example/jeffreuys_with gp.py', sampling='10 equally spaced inputs in each of [-2,-0.67] and [0.67,2]'),
    'amplitude-sine': dict(label='Amplitude-varying sine', n_train=40, noise_std=.3, domain=(-3.,3.), formula=r'\frac12(x+3)\sin(1.5x)', source='BLLN/Benchamrk/LDBLL/toy_problem.py', sampling='20 uniform inputs in each of [-1.25,-0.75] and [0.75,1.25]'),
    'linear-sinc': dict(label='Linear + sinc', n_train=5, noise_std=.01, domain=(-1.2,1.2), formula=r'2x-2\operatorname{sinc}(5x)', source='BLLN/Benchamrk/LDBLL/active_learning.py', sampling='Five initial observations selected from a 1,000-point Uniform(-1,1) bank'),
    'square-wave': dict(label='Fourier square wave', n_train=50, noise_std=.01, domain=(-4.,4.), formula=r'\frac4\pi\sum_{k=0}^{19}\frac{\sin((2k+1)\pi x/2.5)}{2k+1}', source='BLLN/Benchamrk/LDBLL/training_viz.py', sampling='Five equally spaced inputs in [-2,2], each repeated 10 times'),
    'shifted-sinc': dict(label='Shifted sinc', n_train=4, noise_std=0., domain=(-3.,3.), formula=r'\operatorname{sinc}(6(x+1))', source='BLLN/Benchamrk/LDBLL/bayes_opt.py', sampling='Four initial Uniform(-3,3) observations'),
    'sine': dict(label='Sine', n_train=100, noise_std=.1, domain=(-3.,3.), formula=r'\sin x', source='BLLN/Benchamrk/LDBLL/toy_problem_multivariate.py', sampling='100 Uniform(-2,2) inputs; shared with cosine'),
    'cosine': dict(label='Cosine', n_train=100, noise_std=.3, domain=(-3.,3.), formula=r'\cos x', source='BLLN/Benchamrk/LDBLL/toy_problem_multivariate.py', sampling='100 Uniform(-2,2) inputs; shared with sine'),
}


def dataset_function(name, x):
    if name == 'steps': return ground_truth(x)
    if name == 'cubic': return x**3
    if name == 'forrester': return (6*x-2)**2*np.sin(12*x-4)
    if name == 'he': return x*np.sin(x)
    if name == 'amplitude-sine': return .5*(x+3)*np.sin(1.5*x)
    if name == 'linear-sinc': return 2*x-2*np.sinc(5*x)
    if name == 'square-wave': return sum(4*np.sin((2*k+1)*np.pi*x/2.5)/((2*k+1)*np.pi) for k in range(20))
    if name == 'shifted-sinc': return np.sinc(6*(x+1))
    if name == 'sine': return np.sin(x)
    if name == 'cosine': return np.cos(x)
    raise ValueError('Unknown dataset: '+name)


def make_named_dataset(name, seed=123, n_test=1000, n_grid=2000, n_train=None):
    cfg=dict(DATASETS[name], dataset=name, seed=seed, n_test=n_test, n_grid=n_grid)
    if n_train is not None: cfg['n_train']=n_train
    n=cfg['n_train']
    if n<2 or n_test<2 or n_grid<2: raise ValueError('Dataset counts must be at least two')
    if name=='steps':
        # Keep the old API/stream layout byte-for-byte compatible for step arrays.
        arrays=make_dataset(CONFIG|dict(seed=seed,n_train=n,n_test=n_test,n_grid=n_grid))
    else:
        tx,tn,vx,vn=map(np.random.default_rng,np.random.SeedSequence(seed).spawn(4))
        if name=='cubic': x_train=2.5*np.clip(tx.normal(size=(n,1)),-2,2)
        elif name in ('forrester','he','amplitude-sine'):
            intervals={'forrester':((.2,.4),(.65,.85)), 'he':((-2,-.67),(.67,2)), 'amplitude-sine':((-1.25,-.75),(.75,1.25))}[name]
            counts=(n//2,n-n//2)
            x_train=np.concatenate([(tx.uniform(a,b,size=c) if name=='amplitude-sine' else np.linspace(a,b,c)) for (a,b),c in zip(intervals,counts)]).reshape(-1,1)
        elif name=='square-wave':
            if n%5: raise ValueError('Square-wave training count must be divisible by five')
            x_train=np.repeat(np.linspace(-2,2,5),n//5).reshape(-1,1)
        elif name=='linear-sinc':
            bank=tx.uniform(-1,1,(1000,1));x_train=bank[tx.permutation(len(bank))[:n]]
            if n>1000: raise ValueError('Linear-sinc bank has only 1,000 observations')
        else:
            bounds=(-2,2) if name in ('sine','cosine') else cfg['domain']
            x_train=tx.uniform(*bounds,(n,1))
        x_test=vx.uniform(*cfg['domain'],(n_test,1));x_grid=np.linspace(*cfg['domain'],n_grid).reshape(-1,1)
        f_train,f_test,f_grid=[dataset_function(name,x) for x in (x_train,x_test,x_grid)]
        # Sine and cosine share X but use independent noise streams.
        if name=='cosine': tn,vn=map(np.random.default_rng,np.random.SeedSequence([seed,1]).spawn(2))
        arrays=dict(x_train=x_train,y_train=f_train+tn.normal(0,cfg['noise_std'],f_train.shape),
                    x_test=x_test,y_test=f_test+vn.normal(0,cfg['noise_std'],f_test.shape),
                    f_train=f_train,f_test=f_test,x_grid=x_grid,f_grid=f_grid)
    cfg['test_sampling']='Independent Uniform over the full plotting domain'
    cfg['sinc_definition']='sinc(t) = sin(pi*t)/(pi*t), sinc(0) = 1'
    arrays['metadata']=np.array(json.dumps(cfg,sort_keys=True))
    return arrays


def save_named_dataset(name, path, seed=123, **counts):
    path=Path(path)
    arrays=make_named_dataset(name,seed=seed,**counts)
    if path.exists():
        # Reusing data requires identity, not just a matching file name.
        with np.load(path,allow_pickle=False) as old:
            for key in arrays:
                if key!='metadata' and (key not in old or not np.array_equal(old[key],arrays[key])):
                    raise ValueError('Existing dataset does not match requested settings: '+str(path))
        return path
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(str(arrays['metadata'])+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS,default='steps')
    parser.add_argument('--output')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--n-train',type=int)
    parser.add_argument('--n-test',type=int,default=1000)
    parser.add_argument('--n-grid',type=int,default=2000)
    parser.add_argument('--force',action='store_true',help='Intentionally replace an existing dataset')
    args=parser.parse_args()
    path=Path(args.output or ('data/toy_1d.npz' if args.dataset=='steps' else 'data/'+args.dataset+'.npz'))
    if not path.is_absolute(): path=ROOT/path
    if path.exists() and not args.force: raise FileExistsError(f'{path} exists; use --force intentionally')
    data=make_named_dataset(args.dataset,args.seed,args.n_test,args.n_grid,args.n_train)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
    path.with_suffix('.json').write_text(str(data['metadata'])+'\n')
    print(f"Saved {path}: {len(data['x_train'])} train / {len(data['x_test'])} test")


if __name__ == '__main__': main()
