"""Numerical and trajectory checks for the actual pinned upstream integration."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from author_code import load_upstream, original_data, LOCK, CACHE, blob_sha, array_handler
from author_regression import CONFIG, fit, settings
from generate_data import make_author_dataset


class AuthorParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.upstream=load_upstream().modules

    def test_pinned_sources(self):
        for path,sha in LOCK['files'].items():
            self.assertEqual(blob_sha((CACHE/path).read_bytes()),sha)

    def test_exact_native_data_and_separate_noise(self):
        data=make_author_dataset()
        rng=np.random.RandomState(42)
        x=np.vstack((rng.uniform(1.5,2.5,(45,1)),rng.uniform(4.5,6.,(45,1))))
        y=x*np.sin(x)+rng.normal(0,.25,(90,1))
        np.testing.assert_array_equal(data['x_train'],x)
        np.testing.assert_array_equal(data['y_train'],y)
        x_native=np.linspace(0,7,100).reshape(-1,1)
        np.testing.assert_array_equal(data['x_native'],x_native)
        np.testing.assert_array_equal(data['y_native'],x_native*np.sin(x_native))
        np.testing.assert_array_equal(data['x_test'][:100],x_native)
        self.assertFalse(np.array_equal(data['y_test'][:100],data['y_native']))

    def test_kde_gradient_against_closed_form(self):
        W=torch.tensor([[0.,1.],[2.,-1.],[3.,2.]],dtype=torch.float64,requires_grad=True)
        kernel=self.upstream['utils.kernel'].RBF()
        K=kernel(W,W.detach())
        gradient=torch.autograd.grad(K.sum(),W)[0]/K.sum(1,keepdim=True)
        distances=((W.detach()[:,None]-W.detach()[None])**2).sum(-1)
        median=torch.quantile(distances.flatten(),.5)
        gamma=torch.log(torch.tensor(4.,dtype=torch.float64))/median
        expected=(-2*gamma*(K.detach()[:,:,None]*(W.detach()[:,None]-W.detach()[None])).sum(1)/K.detach().sum(1,keepdim=True))
        torch.testing.assert_close(gradient,expected,atol=1e-7,rtol=1e-7)

    def test_ssge_compatibility_against_general_eig(self):
        # General eig is the mathematical operation used by legacy torch.eig.
        # Compare the final score, which is invariant to eigenpair order/sign.
        torch.manual_seed(9)
        xm=torch.randn(7,4,dtype=torch.float64)
        x=torch.randn(5,4,dtype=torch.float64)
        kernel=self.upstream['utils.kernel'].RBF()
        estimator=self.upstream['utils.SSGE_squeeze'].SpectralSteinEstimator(.01,None,kernel,device='cpu')
        actual=estimator.compute_score_gradients(x,xm)
        reference=xm.detach().requires_grad_(True)
        gram=kernel(reference,reference.detach())
        grad=torch.autograd.grad(gram.sum(),reference)[0]
        eigen,vectors=torch.linalg.eig(gram.detach()+.01*torch.eye(len(xm)))
        self.assertLess(float(eigen.imag.abs().max()),1e-12)
        eigen,vectors=eigen.real,vectors.real
        root=torch.sqrt(torch.tensor(len(xm),dtype=torch.float32))
        beta=(-root*vectors.T@(-grad/len(xm)))/eigen[:,None]
        phi=(root*kernel(x,xm)@vectors)/eigen
        torch.testing.assert_close(actual,phi@beta,atol=1e-8,rtol=1e-8)

    def test_export_hooks_do_not_change_native_training(self):
        trainer=self.upstream['training.training_1d_regre']
        factory=self.upstream['methods.method_utils'].create_method
        for method in ('SGD','kde_WGD','kde_f_WGD'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                arrays=make_author_dataset(n_grid=200)
                data=root/'data.npz'
                np.savez_compressed(data,**arrays)
                cfg=CONFIG|dict(data=str(data),output=str(root/'fit'),device='cpu',
                                method=method,preset='upstream',bdkn=False)
                with contextlib.redirect_stdout(io.StringIO()): fit(cfg,3)
                saved=torch.load(root/'fit/stage1.pt',weights_only=True)
                upstream_config=type('Config',(),saved['config']['upstream_config'])()
                torch.manual_seed(upstream_config.random_seed)
                handler,_,_=array_handler(arrays,native=True)
                model=self.upstream['models.MLP'].Net([1,10,10,1],classification=False,act=F.relu,out_act=None)
                ensemble=self.upstream['models.ensemble'].Ensemble(device='cpu',net=model,n_particles=100)
                writer=type('Writer',(),{'add_scalar':lambda *a,**kw:None})()
                plot=trainer.plot_predictive_distributions
                trainer.plot_predictive_distributions=lambda *a,**kw:None
                trainer.create_method=factory
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.train(handler,ensemble,torch.device('cpu'),upstream_config,writer)
                finally:
                    trainer.plot_predictive_distributions=plot
                torch.testing.assert_close(saved['particles'],ensemble.particles,rtol=0,atol=0)
                rows=json.loads((root/'fit/metrics.json').read_text())
                self.assertTrue(all(np.isfinite([r['rmse'],r['nll'],r['native']['nll']]).all() for r in rows))
                with np.load(root/'fit'/rows[0]['method']/'predictions.npz') as predictions:
                    self.assertTrue(np.all(predictions['grid_scale']>=1.))

    def test_update_budgets(self):
        self.assertEqual(settings('upstream',90)['steps'],5000)
        self.assertEqual(settings('low',90)['steps'],6000)
        self.assertEqual(settings('high',90)['steps'],120)

    def test_legacy_steps_metadata(self):
        from generate_data import make_dataset,CONFIG as DATA_CONFIG
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'toy_1d.npz'
            np.savez_compressed(path,**make_dataset(DATA_CONFIG))
            cfg=CONFIG|dict(data=str(path),output=str(root/'fit'),device='cpu',method='SGD',preset='high',bdkn=False)
            with contextlib.redirect_stdout(io.StringIO()): fit(cfg,2)
            self.assertEqual(json.loads((root/'fit/config.json').read_text())['dataset'],'steps')


if __name__=='__main__': unittest.main()
