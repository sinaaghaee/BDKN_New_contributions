"""Check the generated offline site against its actual saved experiment results."""
import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit,unquote

import numpy as np
from report_ui import load_registry

ROOT=Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self,text):
        super().__init__();self.links=[];self.images=[];self.math=[];self.groups={};self.current_group=None;self.tables=[];self.table=None;self.row=None;self.cell=None;self.cells=[]
        self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ('a','script','link','img'):
            value=a.get('href',a.get('src'))
            if value:self.links.append(value)
        if tag=='img':self.images.append(a)
        if 'data-tex' in a:self.math.append(a['data-tex'])
        if 'choices' in a.get('class','').split():self.current_group=a['id'];self.groups[self.current_group]=dict(value=a['data-value'],buttons=[])
        if tag=='button' and 'data-value' in a and self.current_group:self.groups[self.current_group]['buttons'].append(a)
        if tag=='table' and 'overview' in a.get('class','').split():self.table=dict(metric=a['data-metric'],rows=[]);self.tables.append(self.table)
        if tag=='tr' and self.table is not None:self.row=[]
        if tag in ('td','th') and self.row is not None:self.cell=a;self.row.append(a)
        if tag=='td' and 'data-value' in a:self.cells.append(a)
    def handle_endtag(self,tag):
        if tag=='div' and self.current_group:self.current_group=None
        if tag=='tr' and self.row is not None:
            self.table['rows'].append(self.row);self.row=None
        if tag=='table':self.table=None



def verify_fits(registry):
    import run_suite as suite
    common={};checked=0
    for entry in registry['experiments']:
        run=Path(entry['run']);manifest=suite.read_json(run/'run.json');data=Path(manifest['data'])
        assert not suite.read_json(run/'failures.json'),run
        for method in suite.METHODS:
            assert suite.validate_fit(run/method,method,data,entry.get('epochs'),float(entry['learning_rate']),float(entry['repulsion']),entry.get('gp_learning_rate')),(entry['id'],method)
            checked+=1
            from scipy.stats import norm, t
            for row in suite.read_json(run/method/'metrics.json'):
                with np.load(run/method/row['method']/'predictions.npz',allow_pickle=False) as prediction, np.load(data,allow_pickle=False) as target:
                    truth=target['y_test'].reshape(-1)
                    np.testing.assert_array_equal(prediction['y_test'].reshape(-1),truth)
                    rmse=float(np.sqrt(np.mean((truth-prediction['test_mean'])**2)))
                    distribution=norm if row['predictive_family']=='gaussian' else t
                    kwargs={} if row['predictive_family']=='gaussian' else dict(df=float(prediction['df']))
                    nll=float(-distribution.logpdf(truth,loc=prediction['test_mean'],scale=prediction['test_scale'],**kwargs).mean())
                    np.testing.assert_allclose([row['rmse'],row['nll']],[rmse,nll],rtol=1e-9,atol=1e-9)
            if method not in suite.REPULSIVE:
                key=(entry['dataset'],entry['learning_rate'],method)
                digest=suite.file_hash(run/method/'metrics.json')
                assert common.setdefault(key,digest)==digest,(entry['id'],method,'baseline changed across weights')
        bdkn=suite.read_json(run/'bdkn/metrics.json')[0];de=suite.read_json(run/'deep_ensemble/metrics.json')[0]
        assert de['training_reused'] and de['training_seconds']==bdkn['stage1_training_seconds'],entry['id']
    print(f'PASS: {checked} registered method fits, budgets, shared data, complete histories/checkpoints, and reused baseline costs')


def verify(root,registry,fixture=None):
    all_math=set();pages=0
    for entry in registry['experiments']:
        folder=root/entry['directory']
        maps=json.loads((folder/'asset-map.js').read_text().split('=',1)[1].strip().removesuffix(';'))
        for path in maps.values():assert (folder/path).is_file(),(entry['id'],path)
        for file in folder.glob('*.html'):
            text=file.read_text();parsed=Page(text);pages+=1;all_math.update(parsed.math)
            assert '<select' not in text and '<details' not in text and 'Δ BDKN' not in text,file
            for url in parsed.links:
                parts=urlsplit(url)
                if parts.scheme or parts.netloc or not parts.path:continue
                target=(folder/unquote(parts.path)).resolve()
                assert target.is_relative_to(root.resolve()),(file,url,'outside site')
                assert target.is_file(),(file,url,'missing')
            for group in ('loss-xscale','loss-yscale'):
                if group in parsed.groups:assert parsed.groups[group]['value']=='log',(file,group)
            if file.name=='index.html':
                matching=[e for e in registry['experiments'] if e['learning_rate']==entry['learning_rate'] and e['repulsion']==entry['repulsion']]
                assert len(parsed.tables)==3,file
                for table in parsed.tables:
                    assert len(table['rows'])==len(matching)+1,(file,table['metric'])
                    assert all(len(row)==13 for row in table['rows']),(file,table['metric'],'columns')
                    for other,row in zip(matching,table['rows'][1:]):
                        lookup={}
                        for model in Path(other['run']).iterdir():
                            if model.is_dir() and (model/'metrics.json').exists():lookup.update({r['method']:r for r in json.loads((model/'metrics.json').read_text())})
                        from build_website import NAMES
                        values=[lookup[name][table['metric']] for name in NAMES]
                        np.testing.assert_array_equal([float(c['data-value']) for c in row[1:]],values)
                        from report_ui import metric_color
                        for cell,value in zip(row[1:],values):assert cell['style']=='background:'+metric_color(value,values)
        assert (folder/'report_manifest.json').is_file(),folder
    with (root/'downloads/all_results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(registry['experiments'])*12
    if fixture:
        # The browser control harness uses a complete real profile and actual assets.
        selected=next((e for e in registry['experiments'] if e['dataset']=='he'),registry['experiments'][0])
        folder=root/selected['directory'];texts={name:(folder/name).read_text() for name in ('index.html','figures.html','losses.html','repulsion.html')}
        objects={name:Page(text) for name,text in texts.items()}
        state=lambda text:json.loads(re.search(r'<script id="experiment-catalog" type="application/json">(.*?)</script>',text).group(1))
        Path(fixture).write_text(json.dumps(dict(root=str(folder.resolve()),states={name:state(text) for name,text in texts.items()},groups={name:p.groups for name,p in objects.items()},images={name:p.images for name,p in objects.items()},math=sorted(all_math))))
    print(f'PASS: {pages} pages, {len(registry["datasets"])} datasets, {len(registry["experiments"])} profiles, {len(rows)} metric rows, {len(all_math)} distinct LaTeX expressions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='website/dist');parser.add_argument('--registry',default='website/experiments.json');parser.add_argument('--fixture');parser.add_argument('--fits',action='store_true')
    args=parser.parse_args();registry=load_registry(args.registry,ROOT)
    if args.fits:verify_fits(registry)
    verify(Path(args.output).resolve(),registry,args.fixture)
