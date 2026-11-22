"""Numerical checks for the single-change raw-gradient FoRDE ablation."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import forde
import forde_raw
import run_forde_raw as runner

torch.set_num_threads(1)


class RawGradientTests(unittest.TestCase):
    def test_simultaneous_oom_retries_keep_pool_budget_bounded(self):
        import run_adaptive as pool
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            jobs = [dict(id=str(i), dataset='steps', setting='high', method='forde_raw', location='training',
                signature={'epochs': 1}, fingerprint=str(i), folder=str(output / 'runs' / str(i)), status='pending') for i in range(2)]
            args = argparse.Namespace(output=output, device='cpu', workers=2, update_website=False,
                retry_memory_errors=True, memory_per_fit=512, memory_reserve=3072)
            launches = []
            clock = [1000.]
            def now():
                clock[0] += 65
                return clock[0]
            class Child:
                def __init__(self, number): self.pid = 100 + number; self.code = 1 if number <= 2 else 0
                def poll(self): return self.code
            def launch(command, **kwargs):
                launches.append(command)
                if len(launches) <= 2:
                    kwargs['stdout'].write('CUDA error: out of memory\n'); kwargs['stdout'].flush()
                return Child(len(launches))
            with patch.object(pool.subprocess, 'Popen', side_effect=launch), patch.object(pool.time, 'sleep'), patch.object(pool.time, 'time', side_effect=now):
                pool.execute(args, jobs, validator=lambda job: False, marker=lambda job: None, command_builder=lambda job, device: ['fake-trainer'])
            self.assertEqual(len(launches), 4)
            self.assertEqual(args.memory_per_fit, 768)
            self.assertTrue(all(j['status'] == 'complete' and j['oom_retries'] == 1 for j in jobs))

    def test_raw_kernel_and_backward_match_explicit_formula(self):
        g = torch.tensor([[[1.], [2.], [3.]], [[2.], [0.], [-1.]], [[-2.], [1.], [4.]]], dtype=torch.float64, requires_grad=True)
        loss, bandwidth, distance = forde_raw.gradient_repulsive_kde(g)
        particles = g.reshape(3, 1, -1)
        d = (particles[:, None] - particles.detach()[None]).square().sum(-1)
        h = (torch.quantile(d.detach(), .5, dim=0) / np.log(3)).clamp_min(1e-6)
        expected = torch.logsumexp((-d / h).flatten(1), dim=1).sum()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(bandwidth, h, rtol=0, atol=0)
        expected_gradient = torch.autograd.grad(expected, g, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(loss, g)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        normalized_loss = forde.gradient_repulsive_kde(g)[0]
        self.assertNotAlmostEqual(float(loss.detach()), float(normalized_loss.detach()), places=5)
        metadata = forde_raw.repulsion_metadata(.001, 1e-6)
        self.assertEqual(metadata['repulsion_normalization'], 'none')
        self.assertEqual(metadata['gradient_representation'], 'raw')
        self.assertTrue(metadata['repulsion_reference_detached'])
        self.assertTrue(metadata['repulsion_include_self'])

    def test_normalization_epsilon_is_unused(self):
        torch.manual_seed(5)
        g = torch.randn(5, 13, 1, dtype=torch.float64, requires_grad=True)
        first = forde_raw.gradient_repulsive_kde(g, normalization_eps=1e-8)[0]
        second = forde_raw.gradient_repulsive_kde(g, normalization_eps=100.)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        a = torch.autograd.grad(first, g, retain_graph=True)[0]
        b = torch.autograd.grad(second, g)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_strength_reproduces_original_fit_gradients(self):
        torch.manual_seed(12)
        model = forde.build_batched_ensemble(3, 1, 8, 2, 'Tanh', 'none', 1e-6, torch.device('cpu'))
        raw = copy.deepcopy(model)
        bx = torch.linspace(-1, 1, 18).reshape(3, 6, 1)
        by = bx.squeeze(-1).sin()
        a = forde.stage1_objective(model, bx, by, strength=0.)[0]
        b = forde_raw.stage1_objective(raw, bx, by, strength=0.)[0]
        a.backward(); b.backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for x, y in zip(model.parameters(), raw.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_sampling_is_identical_and_excludes_training(self):
        train = torch.tensor([[-1.], [0.], [.5], [1.]])
        grid = torch.linspace(-2, 2, 30).reshape(-1, 1)
        for mode in ('training', 'random', 'combined'):
            before = torch.random.get_rng_state().clone()
            a = forde.RepulsionPointSampler(train, grid, 123).sample(train, mode)
            b = forde_raw.RepulsionPointSampler(train, grid, 123).sample(train, mode)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
            if mode != 'training':
                sampled = b if mode == 'random' else b[len(train):]
                self.assertFalse(torch.isin(sampled, train).any())

    def test_full_matrix_matches_saved_normalized_settings_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'absent'
            args = argparse.Namespace(datasets=runner.DATASETS, settings=runner.SETTINGS, weights=runner.WEIGHTS,
                locations=runner.LOCATIONS, output=output, epochs=None, stage2_steps=None)
            jobs = runner.make_jobs(args)
            self.assertEqual(len(jobs), 300)
            self.assertFalse(output.exists())
            for job in jobs:
                cfg = runner.read_json(Path(job['normalized_reference']) / 'config.json')
                for key, value in job['signature'].items():
                    if key not in ('source_sha256', 'gradient_representation'):
                        self.assertEqual(cfg.get(key, forde.CONFIG.get(key)), value, (job['id'], key))


if __name__ == '__main__':
    unittest.main()
