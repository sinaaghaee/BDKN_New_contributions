"""Export actual D'Angelo initialization and batch RNG state for the JAX adapter."""
import argparse
import json
from pathlib import Path


def prepare(data_path,preset,output,run_settings=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from author_code import load_upstream,array_handler
    from author_regression import settings,digest
    torch.set_num_threads(1)
    with np.load(data_path,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
    selected=settings(preset,len(data['x_train']));selected.update(run_settings or {});torch.manual_seed(selected['seed'])
    handler,_,_=array_handler(data,native='x_native' in data,normalize=selected['normalize_x'])
    state=np.random.get_state()
    modules=load_upstream().modules
    net=modules['models.MLP'].Net([data['x_train'].shape[1],selected['size_hidden'],selected['size_hidden'],1],
                                  classification=False,act=F.relu,out_act=None)
    ensemble=modules['models.ensemble'].Ensemble(device='cpu',net=net,n_particles=selected['n_particles'])
    metadata=dict(selected,input_dim=data['x_train'].shape[1],dataset_sha256=digest(data_path),
                  native_test_count=handler.num_test_samples,preset=preset)
    values=dict(particles=ensemble.particles.numpy(),shapes=np.array(json.dumps(net.param_shapes)),
                metadata=np.array(json.dumps(metadata)),rng_keys=state[1],rng_pos=state[2],
                rng_has_gauss=state[3],rng_cached_gaussian=state[4])
    if output.exists():
        with np.load(output,allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k],v) for k,v in values.items()):raise ValueError('D\'Angelo initialization changed: '+str(output))
        return
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**values)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--preset',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.data,args.preset,args.output)
