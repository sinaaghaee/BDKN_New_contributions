"""Load exact, hash-verified JAX source blocks from the FoRDE authors.

The nested repulsion function is compiled directly from its original AST,
without translating its mathematics or importing the classification launcher.
"""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
LOCK=json.loads((ROOT/'official_forde_source.lock.json').read_text())
CACHE=ROOT/'.cache/official_forde'


def blob_sha(content):
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def prepare_source():
    for name,sha in LOCK['files'].items():
        path=CACHE/name
        if not path.exists():
            url=f'https://raw.githubusercontent.com/{LOCK["repository"]}/{LOCK["revision"]}/{name}'
            with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
            if blob_sha(data)!=sha: raise ValueError('Downloaded upstream source mismatch: '+name)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        if blob_sha(path.read_bytes())!=sha: raise ValueError('Cached upstream source modified: '+name)
    return CACHE


def source_block(name):
    prepare_source()
    tree=ast.parse((CACHE/'train_forde.py').read_text())
    found=[node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name]
    if len(found)!=1: raise ValueError('Ambiguous upstream source block: '+name)
    return found[0]


def repulsion_function(n_members,batch_size,dimension=1,eps=1e-24,eigenvectors=None,eigenvalues=None):
    import jax
    import jax.numpy as jnp
    env=dict(jax=jax,jnp=jnp,n_members=n_members,n_devices=1,batch_size=batch_size,
             height=dimension,width=1,channel=1,eps=eps,eigenvectors=eigenvectors,eigenvalues=eigenvalues)
    node=source_block('get_repulsive_term')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(CACHE/'train_forde.py'),'exec'),env)
    return env['get_repulsive_term']


def scheduler(init_lr,epochs,lr_ratio):
    import jax.numpy as jnp
    env=dict(jnp=jnp)
    exec(compile(ast.Module(body=[source_block('LrScheduler')],type_ignores=[]),str(CACHE/'train_forde.py'),'exec'),env)
    return env['LrScheduler'](init_lr,epochs,(.5,.9),lr_ratio,0)


def optimizer():
    prepare_source()
    env={}
    exec(compile((CACHE/'optimizers.py').read_text(),str(CACHE/'optimizers.py'),'exec'),env)
    return env['nesterov_weight_decay'](mass=.9,weight_decay=0.)


def training_pca(inputs):
    """Same second moment, SVD and sqrt eigenvalues as calculate_pca.py."""
    import numpy as np
    covariance=inputs.T@inputs/(len(inputs)-1)
    vectors,values,_=np.linalg.svd(covariance)
    return vectors.astype(np.float32),np.sqrt(values).astype(np.float32)


def dangelo_batch_generator(indices):
    """Execute the pinned D'Angelo index generator verbatim (no framework imports)."""
    import numpy as np
    from author_code import prepare_source,CACHE as D_CACHE
    prepare_source()
    tree=ast.parse((D_CACHE/'data/dataset.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_random_batch')
    node.decorator_list=[]  # Remove only the class staticmethod descriptor.
    env=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(D_CACHE/'data/dataset.py'),'exec'),env)
    return env['_get_random_batch'](indices)


if __name__=='__main__': print(prepare_source())
