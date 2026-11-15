"""Run every standalone method on identical data and save a visual comparison."""
# --- EDIT SETTINGS HERE ---
METHODS = ['bdkn','deep_ensemble','frde','wrde','forde','gp','dkl','ldbll','vbll']
DEVICE = 'auto'  # Or 'cpu', 'cuda:0', etc.
DATA_PATH = 'data/toy_1d.npz'
OUTPUT_ROOT = 'results'
REUSE_BDKN_ENSEMBLE = True  # Reuse only when all stage-one settings match.

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def comparison(output,data_path):
    """Check dataset identity and aggregate only this run's successful methods."""
    expected=hashlib.sha256(data_path.read_bytes()).hexdigest()
    with np.load(data_path,allow_pickle=False) as data:
        train_x,train_y=data['x_train'],data['y_train']
    rows,panels=[],[]
    for method in METHODS:
        path=output/method/'metrics.json'
        if not path.exists(): continue
        for row in json.loads(path.read_text()):
            if row['dataset_sha256'] != expected: raise ValueError(f"Dataset mismatch: {row['method']}")
            rows.append(row)
            with np.load(output/method/row['method']/'predictions.npz',allow_pickle=False) as pred:
                panels.append((row,{k:pred[k] for k in pred.files}))
    if not rows: raise RuntimeError('No successful results; see method logs')
    with (output/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    nrows=math.ceil(len(panels)/3)
    fig,axes=plt.subplots(nrows,3,figsize=(16,3.6*nrows),sharex=True,sharey=True,squeeze=False)
    lo=min(float(p['grid_lower'].min()) for _,p in panels)
    hi=max(float(p['grid_upper'].max()) for _,p in panels)
    lo,hi=min(lo,float(train_y.min())),max(hi,float(train_y.max()))
    pad=.03*max(hi-lo,1)
    for ax,(row,pred) in zip(axes.flat,panels):
        x=pred['x_grid'].reshape(-1)
        ax.fill_between(x,pred['grid_lower'],pred['grid_upper'],alpha=.22,label='95% predictive interval')
        ax.plot(x,pred['grid_mean'],label='Prediction')
        ax.plot(x,pred['f_grid'].reshape(-1),'k--',lw=1,label='True function')
        ax.scatter(train_x,train_y,s=8,c='black',alpha=.4,label='Training data')
        ax.set(xlim=(-1,1),ylim=(lo-pad,hi+pad),xlabel='x',ylabel='y',
               title=f"{row['method']} | RMSE {row['rmse']:.4f} | NLL {row['nll']:.4f}")
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
    fig.suptitle('Same training/test data • 95% observation-predictive intervals')
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=180); fig.savefig(output/'comparison.pdf'); plt.close(fig)
    for row in rows: print(f"{row['method']:18s} RMSE={row['rmse']:.5f} NLL={row['nll']:.5f}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--nn-learning-rate',type=float,help='Override neural-model LR; GP keeps its configured LR')
    parser.add_argument('--repulsion-strength',type=float,help='Override strength for selected repulsive methods')
    parser.add_argument('--device',default=DEVICE)
    parser.add_argument('--data',default=DATA_PATH)
    parser.add_argument('--output',help='Exact, new run directory')
    parser.add_argument('--epochs',type=int,help='Override all budgets for smoke tests')
    args=parser.parse_args()
    if args.nn_learning_rate is not None and (not np.isfinite(args.nn_learning_rate) or args.nn_learning_rate<=0): parser.error('--nn-learning-rate must be finite and positive')
    if args.repulsion_strength is not None and (not np.isfinite(args.repulsion_strength) or args.repulsion_strength<0): parser.error('--repulsion-strength must be finite and nonnegative')
    if args.epochs is not None and args.epochs<1: parser.error('--epochs must be positive')
    data_path=resolve(args.data)
    if not data_path.exists():
        subprocess.run([sys.executable,str(ROOT/'generate_data.py'),'--output',str(data_path)],check=True)
    output=resolve(args.output) if args.output else resolve(OUTPUT_ROOT)/datetime.now(timezone.utc).strftime('comparison_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'run.json').write_text(json.dumps(vars(args)|{'python':sys.executable,'data':str(data_path),
        'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()},indent=2)+'\n')
    failures=[]
    # BDKN precedes DE so its stage-one artifact can be reused.
    for method in [m for m in METHODS if m in args.methods]:
        command=[sys.executable,str(ROOT/('vbll_regression.py' if method=='vbll' else method+'.py')),'--data',str(data_path),'--output',str(output/method),'--device',args.device]
        if args.nn_learning_rate is not None and method!='gp': command+=['--learning-rate',str(args.nn_learning_rate)]
        if args.repulsion_strength is not None and method in ('frde','wrde','forde'): command+=['--repulsion-strength',str(args.repulsion_strength)]
        if args.epochs is not None: command+=['--epochs',str(args.epochs)]
        if method=='deep_ensemble' and REUSE_BDKN_ENSEMBLE and (output/'bdkn'/'stage1.pt').exists():
            command+=['--reuse-stage1',str(output/'bdkn'/'stage1.pt')]
        print(f'Running {method}…',flush=True)
        with (output/(method+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in child.stdout:
                    print(line,end='',flush=True); log.write(line)
                code=child.wait()
            except BaseException:
                child.terminate(); child.wait(); raise
        if code: failures.append(dict(method=method,exit_code=code))
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    comparison(output,data_path)
    print(f'Comparison saved to {output}')
    if failures: raise SystemExit(f'Methods failed (see logs): {failures}')


if __name__ == '__main__':
    main()
