"""Location sampling contracts and independent repulsive gradients."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import frde
import forde
import run_suite

class LocationTests(unittest.TestCase):
    def test_matching_streams_counts_bounds_and_uniqueness(self):
        train=torch.tensor([[-2.],[-.2],[.4],[1.]])
        grid=torch.tensor([[-3.],[3.]])
        for module in (frde,forde):
            a=module.RepulsionPointSampler(train,grid,123)
            b=module.RepulsionPointSampler(train,grid,123)
            state=torch.random.get_rng_state().clone()
            first=a.sample(train[:3],'random');mixed=b.sample(train[:3],'combined')
            torch.testing.assert_close(first,mixed[3:]);torch.testing.assert_close(mixed[:3],train[:3])
            self.assertEqual(len(mixed),6);self.assertEqual(len(first.unique()),3)
            self.assertFalse(torch.isin(first,train).any());self.assertTrue(((first>=-3)&(first<=3)).all())
            self.assertFalse(torch.equal(first,a.sample(train[:3],'random')))
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            self.assertIs(a.sample(train,'training'),train)
        torch.testing.assert_close(frde.RepulsionPointSampler(train,grid,9).sample(train,'random'),forde.RepulsionPointSampler(train,grid,9).sample(train,'random'))

    def test_training_path_and_random_gradients(self):
        for module in (frde,forde):
            torch.manual_seed(19)
            model=module.build_batched_ensemble(5,1,12,2,'ReLU','none',1e-6,torch.device('cpu'))
            x=torch.linspace(-1,1,8).reshape(-1,1);bx=x.expand(5,-1,-1).clone();by=bx.sin().squeeze(-1)
            cfg=dict(module.CONFIG);sampler=module.RepulsionPointSampler(x,torch.tensor([[-2.],[2.]]),123)
            old=module.training_objective(model,bx,by,cfg)
            same=module.training_objective(model,bx,by,cfg,x)
            torch.testing.assert_close(old[0],same[0],rtol=0,atol=0)
            for location in ('random','combined'):
                model.zero_grad();points=sampler.sample(x,location)
                loss,per_head,diag=module.training_objective(model,bx,by,cfg,points)
                repulsion_grad=torch.autograd.grad(loss-per_head.sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
                self.assertGreater(sum(g.abs().sum().item() for g in repulsion_grad if g is not None),1e-12)
                self.assertTrue(torch.isfinite(diag).all());loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None),0)
                torch.testing.assert_close(diag[0],old[2][0])

    def test_configs_and_legacy_defaults(self):
        for method in ('frde','forde'):
            cfg=run_suite.expected_config(method,None,.001,.1,repulsion_location='combined')
            self.assertEqual(cfg['repulsion_location'],'combined')
        self.assertNotIn('repulsion_location',run_suite.expected_config('wrde',None,.001,.1))

class LocationReportTests(unittest.TestCase):
    def test_registry_accepts_locations_and_rejects_duplicate_identity(self):
        import report_ui
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'registry.json'
            entries=[dict(id='toy-'+location,dataset='toy',learning_rate=.001,repulsion=.01,run='results/'+location,repulsion_location=location) for location in ('training','random','combined')]
            registry=dict(default='toy-training',datasets=dict(toy=dict(label='Toy')),experiments=entries)
            path.write_text(json.dumps(registry));loaded=report_ui.load_registry(path,root)
            self.assertEqual(len(loaded['experiments']),3)
            registry['experiments'].append(dict(entries[0],id='duplicate'))
            path.write_text(json.dumps(registry))
            with self.assertRaises(ValueError):report_ui.load_registry(path,root)

    def test_saved_location_mismatch_rejected(self):
        import shutil
        source=Path('/tmp/bdkn-location-smoke-frde')
        if not source.exists():self.skipTest('GPU smoke artifact not present')
        data=Path('data/toy_1d.npz').resolve()
        self.assertTrue(run_suite.validate_fit(source,'frde',data,2,.001,.001,repulsion_location='random'))
        with self.assertRaises(ValueError):run_suite.validate_fit(source,'frde',data,2,.001,.001,repulsion_location='combined')

    def test_collision_rejection_and_random_state_independence(self):
        for module in (frde,forde):
            generator=torch.Generator().manual_seed(123+1729)
            collision=(2*torch.rand(8,generator=generator,dtype=torch.float64)-1).float()[0]
            train=torch.tensor([[collision],[0.]])
            sampler=module.RepulsionPointSampler(train,torch.tensor([[-1.],[1.]]),123)
            points=sampler.sample(train,'random')
            self.assertFalse(torch.isin(points,train).any())
            self.assertEqual(len(points.unique()),len(train))

    def test_plot_cache_identity_and_grouping(self):
        from build_website import plot_key
        from location_report import comparison_groups
        hashes=dict(frde='abc',frde_ensemble='def',deep_ensemble='ghi',bdkn='jkl')
        history=dict(frde='xyz')
        self.assertEqual(plot_key('data','frde.png',hashes,history,[]),('data','frde.png','abc'))
        self.assertEqual(plot_key('data','frde_pair_test.png',hashes,history,[]),('data','frde_pair_test.png','abcdef'))
        self.assertEqual(plot_key('data','frde_history_xylog.png',hashes,history,[]),('data','frde_history_xylog.png','xyz'))
        entries=[dict(dataset='toy',learning_rate='0.001',repulsion='0.1',repulsion_location=location) for location in ('training','random','combined')]
        groups=comparison_groups(dict(experiments=entries))
        self.assertEqual(set(groups[('toy','0.001','0.1')]),{'training','random','combined'})


    def test_narrow_domain_duplicates_and_insufficient_support(self):
        eps=torch.finfo(torch.float32).eps
        for module in (frde,forde):
            train=torch.tensor([[1.],[1.+eps]]).repeat(3,1)
            sampler=module.RepulsionPointSampler(train,torch.tensor([[1.],[1.+8*eps]]),123)
            points=sampler.sample(train,'random')
            self.assertEqual(len(points.unique()),len(train))
            self.assertFalse(torch.isin(points,train).any())
            impossible=module.RepulsionPointSampler(train,torch.tensor([[1.],[1.+eps]]),123)
            with self.assertRaises(RuntimeError):impossible.sample(train,'random')

    def test_preprocessed_domain_includes_gap_and_extrapolation(self):
        import generate_data
        data=generate_data.make_named_dataset('forrester')
        for module in (frde,forde):
            x,_,_,_=module.preprocess(data,dict(module.CONFIG),torch.device('cpu'))
            sampler=module.RepulsionPointSampler(x['x_train'],x['x_grid'],123)
            points=torch.cat([sampler.sample(x['x_train'],'random') for _ in range(20)])
            raw=(points-sampler.low)/(sampler.high-sampler.low)
            self.assertTrue((raw<.2).any());self.assertTrue((raw>.85).any())
            self.assertTrue(((raw>.4)&(raw<.65)).any())
            self.assertFalse(torch.isin(points,x['x_train']).any())


    def test_registry_update_preserves_old_urls_and_distinguishes_locations(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'website').mkdir()
            old=dict(id='legacy-step-profile',dataset='steps',learning_rate=.001,repulsion=.001,run='old')
            registry=dict(default=old['id'],datasets={},experiments=[old])
            path=root/'website/experiments.json';path.write_text(json.dumps(registry))
            training=root/'old';training.mkdir();(training/'run.json').write_text('{}')
            random_run=root/'random';random_run.mkdir();(random_run/'run.json').write_text('{"repulsion_location":"random"}')
            profiles=[('steps',.001,training,training,None),('steps',.001,random_run,training,None)]
            with patch.object(run_suite,'ROOT',root):
                run_suite.update_registry(profiles,.001)
                run_suite.update_registry(profiles,.001)
            saved=json.loads(path.read_text())
            self.assertEqual(len(saved['experiments']),2)
            self.assertEqual(saved['experiments'][0]['id'],'legacy-step-profile')
            self.assertEqual({e['repulsion_location'] for e in saved['experiments']},{'training','random'})


if __name__=='__main__':unittest.main()
