"""Math, sampling, standalone output, resume, and offline adaptive-report checks."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import generate_data
import run_adaptive as runner

MODULES = [importlib.import_module('adaptive_' + name) for name in runner.METHODS]
torch.set_num_threads(1)


def make_ensemble(module, norm='none'):
    torch.manual_seed(12)
    return module.build_batched_ensemble(3, 1, 8, 2, 'Tanh', norm, 1e-6, torch.device('cpu'))


def arguments(output):
    return argparse.Namespace(datasets=['steps'], settings=['low', 'high'], methods=runner.METHODS,
        locations=runner.LOCATIONS, output=output, epochs=2, stage2_steps=3,
        target_ratio=.8, dual_lr=.1, lambda_initial=0., device='cpu', workers=2, update_website=False)


class AdaptiveMathTests(unittest.TestCase):
    def test_similarity_and_full_pair_gradient(self):
        for module in MODULES:
            with self.subTest(method=module.METHOD):
                p = torch.tensor([[0., 1.], [1., 2.], [-1., 1.]], dtype=torch.float64, requires_grad=True)
                h = .7
                actual = module.kernel_similarity(p, h)
                explicit = sum(torch.exp(-(p[i] - p[j]).square().sum() / h)
                    for i in range(3) for j in range(3) if i != j) / 6
                torch.testing.assert_close(actual, explicit)
                grad = torch.autograd.grad(actual, p, retain_graph=True)[0]
                expected = torch.autograd.grad(explicit, p)[0]
                torch.testing.assert_close(grad, expected)
                self.assertTrue(torch.autograd.gradcheck(lambda x: module.kernel_similarity(x, h), (p,)))
                # No self pairs: as members separate, C can approach zero.
                self.assertLess(float(module.kernel_similarity(torch.tensor([[0.], [100.]]), 1.)), 1e-10)

    def test_bandwidth_target_and_multiplier(self):
        for module in MODULES:
            p = torch.tensor([[0.], [1.], [3.]])
            h, initial, target = module.initial_constraint(p, module.CONFIG)
            self.assertEqual(h, 4.)
            self.assertAlmostEqual(target, .8 * initial)
            self.assertAlmostEqual(module.update_multiplier(.2, .8, .5, .1), .23)
            self.assertAlmostEqual(module.update_multiplier(.2, .2, .5, .1), .17)
            self.assertEqual(module.update_multiplier(0., .2, .5, .1), 0.)
            with self.assertRaises(ValueError):
                module.validate_adaptive(module.CONFIG | {'target_ratio': 1.})

    def test_references_rng_exclusion_and_common_draws(self):
        train = torch.tensor([[-1.], [0.], [0.], [1.]])
        grid = torch.linspace(-2, 2, 25).reshape(-1, 1)
        previous = None
        for module in MODULES:
            state = torch.random.get_rng_state().clone()
            random = module.fixed_reference_inputs(train, grid, 123, 'random')
            combined = module.fixed_reference_inputs(train, grid, 123, 'combined')
            torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
            self.assertEqual(random.shape, train.shape)
            self.assertEqual(len(torch.unique(random)), len(train))
            self.assertFalse(torch.isin(random, train).any())
            self.assertTrue(((random >= -2) & (random <= 2)).all())
            torch.testing.assert_close(combined, torch.cat((train, random)))
            if previous is not None:
                torch.testing.assert_close(random, previous, rtol=0, atol=0)
            previous = random

    def test_fit_gradient_at_zero_and_live_repulsion(self):
        bx = torch.linspace(-1, 1, 12).reshape(3, 4, 1)
        by = bx.squeeze(-1).square()
        reference = torch.linspace(-1, 1, 9).reshape(-1, 1)
        for module in MODULES:
            model = make_ensemble(module)
            cfg = module.CONFIG | {'n_nets': 3}
            h, initial, target = module.initial_constraint(module.particle_representation(model, reference, cfg), cfg)
            before = torch.random.get_rng_state().clone()
            loss, diag = module.adaptive_objective(model, bx, by, reference, cfg, h, target, 0.)
            grads = torch.autograd.grad(loss, list(model.parameters()))
            torch.random.set_rng_state(before)
            fit = module.gaussian_nll_batched(*model(bx), by).sum()
            expected = torch.autograd.grad(fit, list(model.parameters()))
            for actual, wanted in zip(grads, expected):
                torch.testing.assert_close(actual, wanted)
            similarity = module.kernel_similarity(module.particle_representation(model, reference, cfg), h)
            rep = torch.autograd.grad(similarity, list(model.parameters()), allow_unused=True)
            self.assertTrue(any(g is not None and g.abs().max() > 0 for g in rep))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in rep))

    def test_test_targets_never_affect_constraint(self):
        data = generate_data.make_named_dataset('steps')
        altered = copy.deepcopy(data)
        altered['x_test'][:] = 1e6
        altered['y_test'][:] = -1e6
        for module in MODULES:
            a, _, _, _ = module.preprocess(data, module.CONFIG, torch.device('cpu'))
            b, _, _, _ = module.preprocess(altered, module.CONFIG, torch.device('cpu'))
            for location in runner.LOCATIONS:
                ra = module.fixed_reference_inputs(a['x_train'], a['x_grid'], 123, location)
                rb = module.fixed_reference_inputs(b['x_train'], b['x_grid'], 123, location)
                torch.testing.assert_close(ra, rb, rtol=0, atol=0)


class RunnerTests(unittest.TestCase):
    def test_memory_admission_reserves_starting_contexts(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        output = [SimpleNamespace(stdout='16000, GPU-zero\n'),
                  SimpleNamespace(stdout='11, 400, GPU-zero\n99, 10000, GPU-zero\n12, 500, GPU-one\n')]
        with patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(runner.subprocess, 'run', side_effect=output) as query:
            limit = runner.concurrency_limit('cuda:0', 'all', {11: None, 12: None}, 78)
        self.assertEqual(limit, (16000 + 400 - 2048) // 450)
        self.assertEqual(query.call_args_list[0].args[0][1], '--id=0')
        self.assertEqual(runner.concurrency_limit('cuda:0', 2, {}, 78), 2)
        self.assertEqual(runner.concurrency_limit('cpu', 'all', {}, 78), 78)

    def test_readonly_plan_counts_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            args = arguments(output)
            jobs = runner.make_jobs(args)
            self.assertFalse(output.exists())
            self.assertEqual(len(jobs), 14)
            self.assertEqual(sum(j['method'] == 'wrde' for j in jobs), 2)
            self.assertEqual(len({j['fingerprint'] for j in jobs}), 14)
            args.target_ratio = .7
            self.assertTrue(set(j['fingerprint'] for j in jobs).isdisjoint(j['fingerprint'] for j in runner.make_jobs(args)))
            for job in jobs:
                self.assertNotIn('--repulsion-strength', runner.command(job, 'cpu'))


class ReportTests(unittest.TestCase):
    def test_baseline_configuration_history_and_data_identity(self):
        import adaptive_report as report
        import csv
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'run' / 'bdkn'
            prediction_folder = folder / 'bdkn'
            prediction_folder.mkdir(parents=True)
            data = generate_data.make_named_dataset('steps', n_test=8, n_grid=10)
            cfg = MODULES[0].CONFIG | {'dataset_sha256': 'fixture-data', 'epochs': 2, 'stage2_steps': 1}
            (folder / 'config.json').write_text(json.dumps(cfg))
            metric = dict(method='bdkn', rmse=.1234, nll=-.2345, training_seconds=2., dataset_sha256='fixture-data')
            (folder / 'metrics.json').write_text(json.dumps([metric]))
            history = [dict(stage=1, epoch=i, objective=1., likelihood=1.) for i in (1, 2)]
            history.append(dict(stage=2, epoch=1, objective=.5, likelihood=.5))
            with (folder / 'history.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            arrays = {key: data[key] for key in ('x_grid', 'x_test', 'y_test', 'f_grid')}
            for split in ('grid', 'test'):
                mean = data['f_' + split].reshape(-1)
                arrays.update({split + '_mean': mean, split + '_lower': mean - .2,
                    split + '_upper': mean + .2, split + '_scale': np.full(len(mean), .1)})
            np.savez_compressed(prediction_folder / 'predictions.npz', **arrays)
            registry = {'experiments': [dict(dataset='steps', learning_rate=.001, repulsion=.001,
                run=str(root / 'run'))]}
            with patch.object(report, 'ROOT', root):
                matched = report.baseline_for(registry, 'steps', cfg, 'bdkn', 'combined', .001, data, root / 'site', {})
                self.assertEqual(matched['stages']['bdkn']['metrics']['rmse'], .1234)
                self.assertEqual(matched['stages']['bdkn']['metrics']['nll'], -.2345)
                for change in ({'epochs': 3}, {'normalize_y': True}, {'stage2_steps': 2}):
                    rejected = report.baseline_for(registry, 'steps', cfg | change, 'bdkn', 'training', .001, data, root / 'site', {})
                    self.assertIn('unavailable', rejected)
                altered = {k: v.copy() for k, v in data.items()}
                altered['y_test'][0] += 1
                rejected = report.baseline_for(registry, 'steps', cfg, 'bdkn', 'training', .001, altered, root / 'site', {})
                self.assertIn('unavailable', rejected)

    def test_signed_histories_and_offline_navigation(self):
        import adaptive_report as report
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / 'history.csv'
            history.write_text('stage,epoch,objective,likelihood,lambda_after\n1,1,-2,1,0\n1,2,-1,1,0.1\n2,1,-3,-3,\n')
            saved = report.history_data(history)
            self.assertEqual(saved['1']['objective'], [-2., -1.])
            self.assertEqual(saved['1']['lambda_after'], [0., .1])
            self.assertEqual(saved['2']['count'], 1)
            self.assertNotIn('lambda_after', saved['2'])
            page = root / 'experiments/example/index.html'
            page.parent.mkdir(parents=True)
            page.write_text('<nav aria-label="Report pages"><a href="figures.html">Predictions</a></nav>')
            report.install_navigation(root)
            self.assertIn('href="../../adaptive.html"', page.read_text())
            before = page.read_bytes()
            report.install_navigation(root)
            self.assertEqual(before, page.read_bytes())


if __name__ == '__main__':
    unittest.main()
