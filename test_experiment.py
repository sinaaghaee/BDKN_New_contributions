"""Numerical checks. Optional original-code parity: set BDKN_SOURCE_REPO.

Run with: python -m unittest -v test_experiment
The old repository is used only for explicit parity tests, never by experiments.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import norm, t

import bdkn
import deep_ensemble
import forde
import frde
import generate_data
import gp
import ldbll
import wrde

ROOT=Path(__file__).resolve().parent
SOURCE=os.environ.get('BDKN_SOURCE_REPO')
torch.set_num_threads(1)


def ensemble(module):
    torch.manual_seed(12)
    return module.build_batched_ensemble(3,1,8,2,'Tanh','none',1e-6,torch.device('cpu'))


class DatasetTests(unittest.TestCase):
    def test_formula_and_reproducibility(self):
        x=np.array([[-1.],[-.6],[-.2],[0.],[.4],[.8]])
        s=lambda z:expit(200*z)
        expected=.3*(1-s(x+.6))+.9*(s(x+.6)-s(x))-.6*(s(x)-s(x-.4))+.01*np.sin(50*np.sin(10*x))
        np.testing.assert_array_equal(generate_data.ground_truth(x),expected)
        first=generate_data.make_dataset(generate_data.CONFIG)
        second=generate_data.make_dataset(generate_data.CONFIG)
        for key in first: np.testing.assert_array_equal(first[key],second[key])
        self.assertEqual(first['x_train'].shape,(100,1))
        self.assertEqual(first['x_test'].shape,(1000,1))
        self.assertEqual(first['x_grid'].shape,(2000,1))
        self.assertTrue(np.all(np.abs(first['x_train'])<=1))
        self.assertFalse(np.array_equal(first['x_train'],first['x_test'][:100]))

    def test_noise_variance_and_independent_streams(self):
        cfg=generate_data.CONFIG|dict(n_train=100000)
        a=generate_data.make_dataset(cfg)
        self.assertAlmostEqual(float(np.var(a['y_train']-a['f_train'])),.01,delta=.0002)
        b=generate_data.make_dataset(generate_data.CONFIG)
        np.testing.assert_array_equal(a['x_test'],b['x_test'])
        np.testing.assert_array_equal(a['y_test'],b['y_test'])

    def test_training_only_preprocessing(self):
        data=generate_data.make_dataset(generate_data.CONFIG)
        altered={k:v.copy() for k,v in data.items()}
        altered['x_test'][:]=999; altered['y_test'][:]=-999
        config=bdkn.CONFIG|dict(normalize_y=True)
        xa,ya,ma,sa=bdkn.preprocess(data,config,torch.device('cpu'))
        xb,yb,mb,sb=bdkn.preprocess(altered,config,torch.device('cpu'))
        torch.testing.assert_close(xa['x_train'],xb['x_train'],rtol=0,atol=0)
        torch.testing.assert_close(ya,yb,rtol=0,atol=0)
        self.assertEqual((ma,sa),(mb,sb))


class MathTests(unittest.TestCase):
    def test_zero_repulsion_and_gradients(self):
        bx=torch.linspace(-1,1,12).reshape(3,4,1); by=bx.squeeze(-1)**2
        for module in (frde,wrde,forde):
            with self.subTest(method=module.__name__):
                plain,repulsive=ensemble(bdkn),ensemble(module)
                base=bdkn.gaussian_nll_batched(*plain(bx),by).sum(); base.backward()
                loss,_,diag=module.stage1_objective(repulsive,bx,by,strength=0.)
                loss.backward()
                torch.testing.assert_close(loss,base,rtol=0,atol=0)
                for a,b in zip(plain.parameters(),repulsive.parameters()):
                    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
                self.assertEqual(float(diag[1]),0.)

    def test_student_t_and_gaussian_scores(self):
        torch.manual_seed(5)
        phi=torch.randn(20,3,dtype=torch.float64); y=torch.randn(20,1,dtype=torch.float64)
        design=bdkn.prepare_bll_design(phi,y)
        posterior=bdkn.compute_bll_posterior(design,y,1.,0.,0.)
        rmse,nll,mean,scale=bdkn.student_t_pred(phi,y,*posterior,1.7,2.3)
        df=float(2*posterior[2])
        self.assertAlmostEqual(nll,float(-t.logpdf(y.numpy().reshape(-1),df,loc=mean.numpy(),scale=scale.numpy()).mean()),places=10)
        mu,lo,hi=bdkn.bll_student_t_mu_lo_hi(phi,*posterior)
        np.testing.assert_allclose((hi-lo).numpy()/2,t.ppf(.975,df)*(scale.numpy()/2.3))
        self.assertTrue(np.isfinite(rmse))
        means=torch.randn(3,20); variances=torch.ones_like(means)*.1
        r,n=bdkn.compute_deep_ensemble_nll_rmse_from_head_outputs(means,variances,y.float(),torch.tensor(1.7),torch.tensor(2.3))
        m,v=bdkn.predictive_moments_from_head_outputs(means,variances,1e-6)
        expected=-norm.logpdf(y.numpy().reshape(-1),loc=(m*2.3+1.7).numpy(),scale=torch.sqrt(v*2.3**2+1e-6).numpy()).mean()
        self.assertAlmostEqual(n,expected,places=5)

    def test_documented_repulsion_formulas_match_values(self):
        from scipy.special import logsumexp
        import math
        particles=np.array([[0.,0.],[1.,0.],[0.,2.]])
        distance=((particles[:,None]-particles[None])**2).sum(-1)
        bandwidth=max(np.quantile(distance,.5)/math.log(4),1e-6)
        expected=logsumexp(-distance/bandwidth,axis=1).sum()
        for module in (frde,wrde):
            actual=module.repulsive_kde(torch.tensor(particles,dtype=torch.float64))[0]
            self.assertAlmostEqual(float(actual),expected,places=12)
        gradients=np.array([[[1.],[2.]],[[-1.],[.5]],[[2.],[-1.]]])
        field=gradients.reshape(3,-1)
        normalized=field/np.sqrt((field**2).sum(-1,keepdims=True)+1e-8)
        distance=((normalized[:,None]-normalized[None])**2).sum(-1)
        bandwidth=np.maximum(np.quantile(distance,.5,axis=0)/math.log(3),1e-6)
        expected=logsumexp(-distance/bandwidth,axis=1).sum()
        actual=forde.gradient_repulsive_kde(torch.tensor(gradients,dtype=torch.float64))[0]
        self.assertAlmostEqual(float(actual),expected,places=12)

    def test_documented_bayesian_marginal_loss_matches_values(self):
        from scipy.special import gammaln
        rng=np.random.default_rng(4);phi=rng.normal(size=(12,3));y=rng.normal(size=12);sigma=.7
        prior=np.eye(3)/sigma;precision=phi.T@phi+prior;mean=np.linalg.solve(precision,phi.T@y)
        alpha=len(y)/2;beta=.5*(np.sum((y-phi@mean)**2)+np.sum(mean**2)/sigma)
        expected=-(gammaln(alpha)-alpha*np.log(beta)+.5*(np.linalg.slogdet(prior)[1]-np.linalg.slogdet(precision)[1])-.5*len(y)*np.log(2*np.pi))
        actual=bdkn.nig_nll_feature(torch.tensor(phi),torch.tensor(y),torch.tensor(np.log(sigma)),0.,0.)
        self.assertAlmostEqual(float(actual),expected,places=11)

    def test_reuse_signature_detects_changes(self):
        cfg=bdkn.CONFIG|dict(dataset_sha256='abc')
        baseline=bdkn.stage1_signature(cfg)
        for key in bdkn.STAGE1_KEYS:
            changed=dict(cfg); changed[key]='changed'
            self.assertNotEqual(baseline,bdkn.stage1_signature(changed))
        self.assertNotEqual(baseline,bdkn.stage1_signature(cfg|dict(dataset_sha256='xyz')))
        self.assertEqual(baseline,deep_ensemble.stage1_signature(deep_ensemble.CONFIG|dict(dataset_sha256='abc')))

    def test_reuse_rejects_repulsive_or_unidentified_checkpoint(self):
        cfg=deep_ensemble.CONFIG|dict(dataset_sha256='abc')
        signature=deep_ensemble.stage1_signature(cfg)
        for method in ('bdkn','deep_ensemble'):
            self.assertTrue(deep_ensemble.can_reuse_stage1(dict(method=method,signature=signature),cfg))
        for method in ('frde','wrde','forde',None):
            self.assertFalse(deep_ensemble.can_reuse_stage1(dict(method=method,signature=signature),cfg))


class WebsiteHistoryTests(unittest.TestCase):
    def test_both_stages_and_independent_log_axes_preserve_losses(self):
        import build_website
        captured={}
        records=[dict(stage='1',epoch=str(i+1),objective=str(value)) for i,value in enumerate([2.,0.,-1.])]
        records += [dict(stage='2',epoch=str(i+1),objective=str(value)) for i,value in enumerate([-10.,-12.])]
        def capture(fig,name):
            ax=fig.axes[0]
            captured[name]=(ax.get_xscale(),ax.get_yscale(),ax.lines[0].get_ydata().copy())
            build_website.plt.close(fig)
        charts=build_website.make_history_plots(records,'bdkn',capture)
        self.assertEqual(len(charts),2)
        self.assertEqual([c['points'] for c in charts],[3,2])
        for base,expected in [('bdkn_history',[2.,0.,-1.]),('bdkn_stage2_history',[-10.,-12.])]:
            for suffix,xscale,yscale in [('', 'linear','linear'),('_xlog','log','linear'),('_ylog','linear','symlog'),('_xylog','log','symlog')]:
                x,y,loss=captured[base+suffix+'.png']
                self.assertEqual((x,y),(xscale,yscale))
                np.testing.assert_array_equal(loss,expected)

    def test_repulsion_loss_is_a_separate_unweighted_curve(self):
        import build_website
        captured={}
        records=[dict(stage='1',epoch=str(i+1),objective=str(3+i),likelihood=str(2+i),repulsion=str(r)) for i,r in enumerate([0.,.2,.4])]
        records += [dict(stage='2',epoch='1',objective='-10',likelihood='-10',repulsion='0')]
        def capture(fig,name):
            captured[name]=fig.axes[0].lines[0].get_ydata().copy()
            build_website.plt.close(fig)
        charts=build_website.make_history_plots(records,'frde',capture)
        self.assertEqual(len(charts),3)
        np.testing.assert_array_equal(captured['frde_repulsion_history_xylog.png'],[0.,.2,.4])
        self.assertEqual(next(c for c in charts if c['name']=='frde_repulsion_history')['points'],3)

    def test_positive_loss_uses_ordinary_log(self):
        import build_website
        captured={}
        def capture(fig,name):
            captured[name]=fig.axes[0].get_yscale()
            build_website.plt.close(fig)
        charts=build_website.make_history_plots([dict(epoch='1',train_loss='2'),dict(epoch='2',train_loss='1')],'vbll',capture)
        self.assertEqual(charts[0]['log_y_scale'],'log')
        self.assertEqual(captured['vbll_history_ylog.png'],'log')


class ReportComparisonTests(unittest.TestCase):
    def test_loss_axes_default_to_log(self):
        from report_ui import loss_controls,loss_chart
        controls=loss_controls()
        self.assertEqual(controls.count('data-value="log"'),4)
        self.assertNotIn('<select',controls)
        chart=loss_chart(dict(name='frde_repulsion_history',method='frde',caption='Repulsion loss R',log_y_scale='symlog'))
        self.assertIn('frde_repulsion_history_xylog.png',chart)
        self.assertIn('X: log',chart)
        self.assertIn('Y: signed log',chart)

    def test_metric_colors_are_best_to_worst_and_handle_ties(self):
        from report_ui import metric_color
        self.assertEqual(metric_color(-1,[-1,0,2]),'hsl(120.00 60% 88%)')
        self.assertEqual(metric_color(2,[-1,0,2]),'hsl(0.00 60% 88%)')
        self.assertEqual(metric_color(1,[1,1]),'hsl(60.00 60% 88%)')
        self.assertEqual(metric_color(None,[1,2]),'#eef1f5')
        self.assertEqual(metric_color(8,[.1,.11,.2,8,127]),'hsl(30.00 60% 88%)')

    def test_registry_has_unique_combinations_and_safe_paths(self):
        from report_ui import load_registry
        import tempfile
        registry=json.loads((ROOT/'website/experiments.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'registry.json';path.write_text(json.dumps(registry))
            parsed=load_registry(path,ROOT)
            self.assertEqual([e['repulsion'] for e in parsed['experiments'] if e['dataset']=='steps'],['0.001','0.01','0.1','1','10'])
            self.assertEqual(parsed['experiments'][0]['directory'],'')
            self.assertTrue(parsed['experiments'][1]['directory'].startswith('experiments/'))
            registry['experiments'].append(dict(registry['experiments'][0],id='duplicate'))
            path.write_text(json.dumps(registry))
            with self.assertRaises(ValueError):load_registry(path,ROOT)

    def test_compact_page_and_relative_profile_switching(self):
        from report_ui import page,load_registry
        registry=load_registry(ROOT/'website/experiments.json',ROOT)
        entry=registry['experiments'][1]
        html=page(dict(entry=entry,registry=registry),'Losses','losses.html','<section>Losses</section>')
        self.assertNotIn('class="stats"',html)
        self.assertNotIn('Lowest RMSE',html)
        self.assertIn('id="dataset-select"',html)
        self.assertIn('id="lr-select"',html)
        self.assertIn('id="strength-select"',html)
        self.assertIn('"root": "../../"',html)
        self.assertIn('"page": "losses.html"',html)
        self.assertNotIn('Δ',html)
        self.assertNotIn('<select',html)
        self.assertNotIn('<details',html)
        self.assertNotIn('<footer',html)


@unittest.skipUnless(SOURCE,'Set BDKN_SOURCE_REPO to run original-code parity checks')
class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,SOURCE)

    def test_source_provenance(self):
        provenance=json.loads((ROOT/'provenance.json').read_text())
        for path,expected in provenance['files'].items():
            self.assertEqual(hashlib.sha256((Path(SOURCE)/path).read_bytes()).hexdigest(),expected,path)

    def test_repulsive_objectives_and_gradients(self):
        bx=torch.linspace(-1,1,12).reshape(3,4,1); by=bx.squeeze(-1)**2
        for local,pkg in ((frde,'src_FRDE'),(wrde,'src_WRDE'),(forde,'src_FoRDE')):
            with self.subTest(method=pkg):
                original=importlib.import_module(pkg+'.bdkn_vectorized.repulsion')
                left,right=ensemble(local),ensemble(local)
                a,_,da=local.stage1_objective(left,bx,by)
                b,_,db=original.stage1_objective(right,bx,by)
                a.backward(); b.backward()
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(da,db,rtol=0,atol=0)
                for p,q in zip(left.parameters(),right.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_original_ensemble_and_bayesian_posterior(self):
        original=importlib.import_module('src.bdkn_vectorized.vectorized_de')
        orig_bll=importlib.import_module('src.bdkn_vectorized.bll')
        a=ensemble(bdkn); b=ensemble(original)
        x=torch.linspace(-1,1,10).reshape(-1,1)
        for p,q in zip(a(x),b(x)): torch.testing.assert_close(p,q,rtol=0,atol=0)
        phi=a(x)[0].T.detach().double(); y=x.double()**2
        pa=bdkn.compute_bll_posterior(bdkn.prepare_bll_design(phi,y),y,1.,0.,0.)
        pb=orig_bll.compute_bll_posterior(orig_bll.prepare_bll_design(phi,y),y,1.,0.,0.)
        ma,lo,hi=bdkn.bll_student_t_mu_lo_hi(phi,*pa)
        mb,lb,hb=orig_bll.bll_student_t_mu_lo_hi(phi,*pb)
        for p,q in ((ma,mb),(lo,lb),(hi,hb)): torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_ldbll_training_and_predictions(self):
        original=importlib.import_module('src.common.legacy_ldbll.regression.bll')
        x=torch.linspace(-1,1,20).reshape(-1,1); y=torch.sin(x*3)
        def fit(module):
            torch.manual_seed(12); np.random.seed(12)
            model=module.GaussianBayesianLastLayer(1,1,optimize_prior=True,hidden_dims=(5,5),hidden_bias=True)
            model.fit(x,y,num_epochs=2,learning_rate=.0002,latent_derivatives=True,add_epsilon_noise=True,disable_tqdm=True,log_every=3)
            return model.predict(x)
        a,b=fit(ldbll),fit(original)
        for p,q in zip(a,b): torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_gp_training_and_predictions(self):
        original=importlib.import_module('src.common.legacy_ldbll.regression.gp')
        x=torch.linspace(-1,1,20).reshape(-1,1); y=torch.sin(x*3)
        def fit(module):
            torch.manual_seed(12)
            model=module.GaussianProcess(1,1)
            model.fit(x,y,num_epochs=2,learning_rate=.01,enable_early_stopping=False,disable_tqdm=True,log_every=3)
            return model.predict(x)
        for a,b in zip(fit(gp),fit(original)): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_dkl_model_and_gradients(self):
        import dkl
        original=importlib.import_module('src.dkl.train')
        x=torch.linspace(-1,1,20).reshape(-1,1); y=torch.sin(x*3).squeeze(-1)
        def compute(module):
            torch.manual_seed(12)
            likelihood=module.gpytorch.likelihoods.GaussianLikelihood()
            model=module.DeepKernelGP(x,y,likelihood,1,[5,5],1,'ReLU','rbf',30,4)
            value=-module.gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)(model(x),y)
            value.backward()
            return value,[p.grad for p in model.parameters()]
        (a,ga),(b,gb)=compute(dkl),compute(original)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for p,q in zip(ga,gb): torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_vbll_model_and_gradients(self):
        import vbll_regression
        original=importlib.import_module('src.vbll.train')
        x=torch.linspace(-1,1,20).reshape(-1,1); y=torch.sin(x*3)
        def compute(module):
            torch.manual_seed(12)
            model=module.VBLLRegressor(1,5,2,torch.nn.ReLU,1/len(x),1.,1.)
            out=model(x); loss=out.train_loss_fn(y); loss.backward()
            return out.predictive.mean,out.predictive.covariance,loss,[p.grad for p in model.parameters()]
        a,b=compute(vbll_regression),compute(original)
        for p,q in zip(a[:3],b[:3]): torch.testing.assert_close(p,q,rtol=0,atol=0)
        for p,q in zip(a[3],b[3]): torch.testing.assert_close(p,q,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
