"""Cross-framework check: actual source initialization, batching and Adam."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def verify(jax_python):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from author_code import load_upstream,array_handler
    from generate_data import make_author_dataset
    from prepare_dangelo import prepare
    root=Path(__file__).resolve().parent;torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary);data_path=folder/'data.npz';data=make_author_dataset(n_grid=100)
        np.savez_compressed(data_path,**data);initial=folder/'initial.npz';prepare(data_path,'high',initial)
        subprocess.run([jax_python,str(root/'official_forde.py'),'--data',str(data_path),'--output',str(folder/'fit'),
            '--preset','high','--method','control','--device','cpu','--epochs','3','--skip-bdkn','--initialization',str(initial)],check=True)
        with np.load(initial) as saved:W=torch.tensor(saved['particles'])
        modules=load_upstream().modules;cfg=modules['utils.config'].configuration(['--method','SGD','--pred_dist_std','1'])
        net=modules['models.MLP'].Net([1,50,50,1],act=F.relu)
        ensemble=modules['models.ensemble'].Ensemble('cpu',net,particles=W)
        posterior=modules['utils.distributions'].Unorm_post(ensemble,
            torch.distributions.Normal(torch.zeros(net.num_params),torch.ones(net.num_params)),cfg,len(data['x_train']))
        handler,_,_=array_handler(data,True,True);optimizer=torch.optim.Adam([W],lr=.1)
        method=modules['methods.SVGD'].SGD(posterior,optimizer)
        for _ in range(3):
            x,y=handler.next_train_batch(32);handler.next_test_batch(32)
            method.step(W,torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32))
        def flatten(path):
            with np.load(path) as saved:
                values=[]
                for i in range(3):
                    values.extend((saved[f'mlp/~/linear_{i}/w'].transpose(0,2,1).reshape(5,-1),
                                   saved[f'mlp/~/linear_{i}/b'].reshape(5,-1)))
            return np.concatenate(values,axis=1)
        with np.load(initial) as saved:np.testing.assert_array_equal(flatten(folder/'fit/initial_params.npz'),saved['particles'])
        actual=flatten(folder/'fit/stage1_params.npz')
        difference=float(np.abs(actual-W.numpy()).max())
        # Different float32 GEMM/reduction and bias-correction implementations
        # need numerical tolerance; this does not assert bitwise JAX=PyTorch.
        np.testing.assert_allclose(actual,W.numpy(),rtol=2e-4,atol=1e-4)
        print('PASS actual initial weights and 3-step source Adam comparison; max difference',difference)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jax-python',default=str(Path(__file__).resolve().parent/'.venv-forde/bin/python'))
    verify(p.parse_args().jax_python)
