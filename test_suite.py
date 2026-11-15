"""Dataset contracts, resume validation, and cross-dataset report checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import generate_data as datasets
import run_suite as suite
import report_ui as ui


class DatasetSuiteTests(unittest.TestCase):
    def test_source_shapes_domains_and_reproducibility(self):
        for name,cfg in datasets.DATASETS.items():
            with self.subTest(dataset=name):
                a=datasets.make_named_dataset(name);b=datasets.make_named_dataset(name)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                self.assertEqual(a['x_train'].shape,(cfg['n_train'],1))
                self.assertEqual(a['x_test'].shape,(1000,1))
                self.assertEqual(a['x_grid'].shape,(2000,1))
                self.assertEqual(tuple(a['x_grid'][[0,-1],0]),cfg['domain'])
                for key in ('x_train','x_test','x_grid','y_train','y_test','f_grid'):self.assertTrue(np.isfinite(a[key]).all())
                np.testing.assert_array_equal(a['f_grid'],datasets.dataset_function(name,a['x_grid']))
                self.assertFalse(np.array_equal(a['x_train'],a['x_test'][:len(a['x_train'])]))
                if cfg['noise_std']==0:np.testing.assert_array_equal(a['y_train'],a['f_train']);np.testing.assert_array_equal(a['y_test'],a['f_test'])

    def test_source_sampling_constraints(self):
        for name,intervals in [('forrester',[(.2,.4),(.65,.85)]),('he',[(-2,-.67),(.67,2)]),('amplitude-sine',[(-1.25,-.75),(.75,1.25)])]:
            x=datasets.make_named_dataset(name)['x_train'][:,0]
            n=len(x)//2
            for part,(lo,hi) in zip((x[:n],x[n:]),intervals):self.assertTrue(((part>=lo)&(part<=hi)).all())
            if name!='amplitude-sine':np.testing.assert_allclose(x[:n],np.linspace(*intervals[0],n))
        wave=datasets.make_named_dataset('square-wave')
        values,counts=np.unique(wave['x_train'],return_counts=True)
        np.testing.assert_array_equal(values,np.linspace(-2,2,5));np.testing.assert_array_equal(counts,[10]*5)
        sine=datasets.make_named_dataset('sine');cosine=datasets.make_named_dataset('cosine')
        np.testing.assert_array_equal(sine['x_train'],cosine['x_train'])
        np.testing.assert_array_equal(sine['x_test'],cosine['x_test'])
        self.assertFalse(np.allclose((sine['y_train']-sine['f_train'])/.1,(cosine['y_train']-cosine['f_train'])/.3))

    def test_legacy_steps_unchanged_and_test_stream_independent(self):
        old=datasets.make_dataset(datasets.CONFIG);new=datasets.make_named_dataset('steps')
        for key in old:
            if key!='metadata':np.testing.assert_array_equal(old[key],new[key])
        for name in datasets.DATASETS:
            if name=='square-wave':other_n=100
            else:other_n=20
            a=datasets.make_named_dataset(name);b=datasets.make_named_dataset(name,n_train=other_n)
            np.testing.assert_array_equal(a['x_test'],b['x_test']);np.testing.assert_array_equal(a['y_test'],b['y_test'])

    def test_formula_reference_points(self):
        x=np.array([[-1.],[0.],[.2],[1.]])
        np.testing.assert_allclose(datasets.dataset_function('cubic',x),x*x*x)
        np.testing.assert_allclose(datasets.dataset_function('he',x),x*np.sin(x))
        self.assertAlmostEqual(float(datasets.dataset_function('forrester',np.array([[.5]]))[0,0]),np.sin(2))
        self.assertEqual(float(datasets.dataset_function('shifted-sinc',np.array([[-1.]]))[0,0]),1.)
        self.assertEqual(float(datasets.dataset_function('linear-sinc',np.array([[0.]]))[0,0]),-2.)
        np.testing.assert_allclose(datasets.dataset_function('square-wave',x),sum(4*np.sin(np.pi*n*x/2.5)/(np.pi*n) for n in range(1,40,2)))

    def test_existing_dataset_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.npz';datasets.save_named_dataset('he',path)
            before=path.read_bytes();datasets.save_named_dataset('he',path)
            self.assertEqual(before,path.read_bytes())
            with self.assertRaises(ValueError):datasets.save_named_dataset('he',path,seed=99)


class ResumeAndReportTests(unittest.TestCase):
    def make_fit(self,root):
        data=root/'data.npz';datasets.save_named_dataset('he',data,n_test=10,n_grid=12)
        cfg=suite.expected_config('gp',2,.001,.001)|dict(dataset_sha256=suite.file_hash(data))
        folder=root/'gp';folder.mkdir();(folder/'model.pt').write_bytes(b'checkpoint fixture');suite.write_json(folder/'config.json',cfg)
        row=dict(method='gp',dataset_sha256=suite.file_hash(data),rmse=1.,nll=2.,training_seconds=3.)
        suite.write_json(folder/'metrics.json',[row]);pred=folder/'gp';pred.mkdir()
        with np.load(data) as d:
            arrays={split+'_'+key:np.ones(len(d['x_'+split])) for split in ('test','grid') for key in ('mean','scale','lower','upper')}
            arrays.update({key:d[key] for key in ('x_test','x_grid','f_grid')})
        np.savez_compressed(pred/'predictions.npz',**arrays)
        (folder/'history.csv').write_text('epoch,objective\n1,2\n2,1\n')
        return folder,data

    def test_atomic_status_writes_from_parallel_workers(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            with ThreadPoolExecutor(max_workers=8) as workers:
                list(workers.map(lambda i:suite.write_json(path,dict(number=i,payload='x'*1000)),range(100)))
            result=suite.read_json(path)
            self.assertIn(result['number'],range(100))
            self.assertEqual(result['payload'],'x'*1000)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_high_setting_overrides_gp_and_retains_bayesian_budget(self):
        for method in suite.METHODS:
            with self.subTest(method=method):
                high=suite.expected_config(method,40,.1,.01,gp_lr=.1)
                self.assertEqual(high['epochs'],40)
                self.assertEqual(high['lr'],.1)
                if method in ['bdkn']+suite.REPULSIVE:
                    self.assertEqual(high['stage2_steps'],100)
                    self.assertEqual(high['stage2_lr'],.1)
        self.assertEqual(suite.expected_config('gp',None,.001,.001)['lr'],.01)

    def test_resume_validates_settings_and_complete_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folder,data=self.make_fit(Path(directory))
            self.assertTrue(suite.validate_fit(folder,'gp',data,2))
            with self.assertRaises(ValueError):suite.validate_fit(folder,'gp',data,3)
            (folder/'history.csv').write_text('epoch,objective\n1,2\n')
            self.assertFalse(suite.validate_fit(folder,'gp',data,2))

    def test_overview_rows_columns_links_and_row_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries=[]
            for i,name in enumerate(('he','cubic')):
                run=root/name;run.mkdir();folder=run/'gp';folder.mkdir()
                rows=[dict(method='gp',rmse=1.+i*100,nll=2.,training_seconds=1.),dict(method='bdkn',rmse=2.+i*100,nll=1.,training_seconds=2.)]
                suite.write_json(folder/'metrics.json',rows)
                entries.append(dict(id=name,dataset=name,learning_rate='0.001',repulsion='0.001',directory='' if i==0 else 'experiments/cubic/',run=str(run)))
            ctx=dict(registry=dict(experiments=entries,datasets={key:dict(label=key) for key in ('he','cubic')}),entry=entries[0],names=dict(gp='GP',bdkn='BDKN'))
            html=ui.overview_tables(ctx)
            self.assertEqual(html.count('class="overview"'),3)
            self.assertEqual(html.count('scope="row"'),6)
            self.assertIn('experiments/cubic/figures.html#gp',html)
            # Both rows rank GP first for RMSE despite different absolute scales.
            self.assertEqual(html.count('background:hsl(120.00 60% 88%)'),6)
            self.assertIn('within each dataset row',html)


if __name__=='__main__':unittest.main()
