"""Verify literal upstream JAX blocks and the separate regression adapter."""
import ast
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
if hasattr(os,'sched_getaffinity'):
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[0]})
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from official_forde_core import LOCK,CACHE,prepare_source,blob_sha,source_block,repulsion_function,optimizer,scheduler,training_pca
from official_forde import make_model,make_step,settings


class OfficialFoRDE(unittest.TestCase):
    def test_exact_source_and_extracted_ast(self):
        prepare_source()
        for path,sha in LOCK['files'].items():self.assertEqual(blob_sha((CACHE/path).read_bytes()),sha)
        tree=ast.parse((CACHE/'train_forde.py').read_text())
        original=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_repulsive_term')
        self.assertEqual(ast.dump(original),ast.dump(source_block('get_repulsive_term')))

    def test_repulsion_and_derivative_multidimensional(self):
        h,b,d=4,5,3;eps=1e-24
        jac=jax.random.normal(jax.random.PRNGKey(15),(h,1,b,d))
        vectors=jnp.eye(d);weights=jnp.array([.4,1.,2.])
        actual=repulsion_function(h,b,d,eps,vectors,weights)
        def reference(value):
            u=value[:,0]@vectors
            u=u/jnp.sqrt(jnp.square(u).sum(2,keepdims=True)+eps)
            u=u*weights
            distance=jnp.square(u[:,None]-jax.lax.stop_gradient(u)[None,:]).sum(-1)
            bandwidth=jnp.median(jax.lax.stop_gradient(distance),axis=0)/jnp.log(h)+1e-12
            return (jax.scipy.special.logsumexp(-distance/bandwidth,axis=(1,2))-jnp.log(b)).sum()/b
        value,gradient=jax.value_and_grad(lambda x:actual(x)[0])(jac)
        expected,expected_gradient=jax.value_and_grad(reference)(jac)
        np.testing.assert_allclose(value,expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(gradient,expected_gradient,rtol=1e-5,atol=1e-6)

    def test_scalar_sign_behavior_is_not_modified(self):
        jac=jnp.array([[[[2.],[3.]]],[[[4.],[5.]]],[[[-2.],[-3.]]]])
        core=repulsion_function(3,2,1,1e-24)
        gradient=jax.grad(lambda x:core(x)[0])(jac)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        # Exact powers of two/same signs can have zero derivatives; do not
        # demand a nonzero force or introduce an artificial normalization floor.
        self.assertLess(float(jnp.linalg.norm(gradient)),1e-5)

    def test_pca_matches_author_second_moment(self):
        x=np.array([[1.],[2.],[4.]],dtype=np.float32)
        vectors,values=training_pca(x)
        np.testing.assert_array_equal(vectors,np.ones((1,1),dtype=np.float32))
        np.testing.assert_allclose(values,np.sqrt((x.T@x)/2).reshape(-1))

    def test_nesterov_and_schedule(self):
        init,update,get_params,get_velocity=optimizer()
        x=jnp.array([1.,2.]);v=init(x)
        for g in (jnp.array([.3,-.2]),jnp.array([.1,.4])):
            expected_v=.9*v+g;expected_x=x-.01*(.9*expected_v+g)
            state=update(.01,g,x,v)
            np.testing.assert_allclose(get_params(state),expected_x)
            np.testing.assert_allclose(get_velocity(state),expected_v)
            x,v=get_params(state),get_velocity(state)
        lrs=np.asarray(scheduler(.1,10,.01).lrs)
        np.testing.assert_allclose(lrs[:6],.1)
        np.testing.assert_allclose(lrs[7],.0505,rtol=1e-5)
        np.testing.assert_allclose(lrs[9],.001,rtol=1e-5)
        np.testing.assert_allclose(np.asarray(scheduler(.001,10,1.).lrs),.001)

    def test_control_gaussian_gradients_and_initialization(self):
        model=make_model(4);h=3
        keys=jax.random.split(jax.random.PRNGKey(123),h+1)[1:]
        params=jax.vmap(model.init,(0,None))(keys,jnp.ones((1,1)))
        matching=jax.vmap(model.init,(0,None))(keys,jnp.ones((1,1)))
        for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(matching)):np.testing.assert_array_equal(a,b)
        x=jnp.array([[-.7],[.5],[1.]]);y=jnp.array([[.2],[1.],[.6]])
        init,update,get_params,_=optimizer();velocity=jax.tree_util.tree_map(init,params)
        step=make_step(model,h,3,1e-24,5e-4,False,jnp.ones((1,1)),jnp.ones(1))
        actual,_,diag=step(params,velocity,x,y,.001)
        def loss(p):
            mu=jax.vmap(model.apply,(0,None))(p,x)[...,0]
            fit=(.5*jnp.square(mu-y[:,0][None,:])+.5*jnp.log(2*jnp.pi)).mean(1).sum()
            return fit+.5*5e-4*sum(jnp.sum(v*v) for v in jax.tree_util.tree_leaves(p))
        grad=jax.grad(loss)(params)
        expected=jax.tree_util.tree_map(lambda p,g,v:get_params(update(.001,g,p,v)),params,grad,velocity)
        for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
        self.assertEqual(float(diag[2]),0.);self.assertEqual(float(diag[5]),0.)

    def test_presets(self):
        self.assertEqual(settings('low')['epochs'],2000)
        self.assertEqual(settings('high')['epochs'],40)
        self.assertEqual(settings('author_style'),dict(epochs=300,lr=.1,n_particles=10,lr_ratio=.01))


if __name__=='__main__':unittest.main()
