"""Dataset only from arXiv:2505.18526 section 4.1; edit CONFIG and run this file."""
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
    if not np.isfinite(config['noise_std']) or config['noise_std'] <= 0:
        raise ValueError('noise_std must be finite and positive')
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=CONFIG['output'])
    parser.add_argument('--seed',type=int,default=CONFIG['seed'])
    for name in ('train','test','grid'):
        parser.add_argument('--n-'+name,type=int,default=CONFIG['n_'+name])
    parser.add_argument('--force',action='store_true',help='Intentionally replace an existing dataset')
    args=parser.parse_args()
    cfg=dict(CONFIG,seed=args.seed,n_train=args.n_train,n_test=args.n_test,n_grid=args.n_grid)
    path=Path(args.output)
    if not path.is_absolute(): path=ROOT/path
    cfg['output']=str(path.resolve())
    if path.exists() and not args.force: raise FileExistsError(f'{path} exists; use --force intentionally')
    data=make_dataset(cfg); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**data)
    path.with_suffix('.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(f"Saved {path}: {cfg['n_train']} train / {cfg['n_test']} test")


if __name__ == '__main__':
    main()
