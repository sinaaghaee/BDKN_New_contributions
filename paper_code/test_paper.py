"""Check the published recipe and exact short-run parity with author training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from author_regression import CONFIG, fit, settings
from author_code import load_upstream, array_handler
from generate_data import make_author_dataset


class PaperRecipe(unittest.TestCase):
    def test_published_settings_and_native_trajectory(self):
        torch.set_num_threads(1)
        self.assertEqual(settings('paper',90),dict(steps=15000,lr=.01,n_particles=50,
            size_hidden=50,seed=42,normalize_x=False,batch_size=64))
        modules=load_upstream().modules
        trainer=modules['training.training_1d_regre']
        factory=modules['methods.method_utils'].create_method
        for method in ('SGD','kde_WGD','kde_f_WGD'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);data=make_author_dataset(n_grid=200)
                np.savez_compressed(root/'data.npz',**data)
                cfg=CONFIG|dict(data=str(root/'data.npz'),output=str(root/'fit'),
                    device='cpu',method=method,preset='paper',bdkn=False)
                with contextlib.redirect_stdout(io.StringIO()):fit(cfg,2)
                saved=torch.load(root/'fit/stage1.pt',weights_only=True)
                args=type('Args',(),saved['config']['upstream_config'])()
                self.assertEqual(args.batch_size,64)
                self.assertEqual(args.pred_dist_std,1.)
                self.assertEqual(args.prior_variance,1.)
                torch.manual_seed(42)
                handler,_,_=array_handler(data,native=True)
                model=modules['models.MLP'].Net([1,50,50,1],classification=False,act=F.relu,out_act=None)
                ensemble=modules['models.ensemble'].Ensemble(device='cpu',net=model,n_particles=50)
                writer=type('Writer',(),{'add_scalar':lambda *a,**kw:None})()
                plot=trainer.plot_predictive_distributions
                trainer.plot_predictive_distributions=lambda *a,**kw:None
                trainer.create_method=factory
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.train(handler,ensemble,torch.device('cpu'),args,writer)
                finally:trainer.plot_predictive_distributions=plot
                torch.testing.assert_close(saved['particles'],ensemble.particles,rtol=0,atol=0)
                rows=json.loads((root/'fit/metrics.json').read_text())
                self.assertTrue(all(np.isfinite([r['rmse'],r['nll']]).all() for r in rows))


if __name__=='__main__':unittest.main()
