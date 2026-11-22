"""Saved-prediction diagnostic: keep BDKN means/df, replace variance with DE's.

This counterfactual is an analysis only, not a new trained method or website entry.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from scipy.stats import norm, t
from audit_results import SOURCE, save_csv

records=json.loads((SOURCE/'website/pages/hetero_assets/selected_runs.json').read_text())
rows=[]
for record in records:
    if record['lr']!='low' or record['status']!='complete':continue
    for split in record['expected']:
        folder=Path(record['run'])/f'split_{split:03d}/comparison'
        with np.load(folder/'predictions.npz') as z:
            y=z['test_observed'].ravel().astype(float)
            dm=z['test_DE_mean'].ravel().astype(float)
            dv=z['test_DE_predictive_variance'].ravel().astype(float)
            bm=z['test_BDKN_constant_mean'].ravel().astype(float)
            bv=z['test_BDKN_constant_predictive_variance'].ravel().astype(float)
            bs=z['test_BDKN_constant_scale_squared'].ravel().astype(float)
            df=float(z['test_BDKN_constant_df'])
            # Total DE variance = mean member variance + member-mean disagreement.
            heads=z['test_head_mean'].astype(float)
            alea=z['test_head_variance'].astype(float).mean(0)
        nll_de=float(-norm.logpdf(y,dm,np.sqrt(dv)).mean())
        nll_b=float(-t.logpdf(y,df,loc=bm,scale=np.sqrt(bs)).mean())
        nll_swap=float(-t.logpdf(y,df,loc=bm,scale=np.sqrt(dv*(df-2)/df)).mean())
        saved=json.loads((folder/'metrics.json').read_text())['test']
        np.testing.assert_allclose([nll_de,nll_b],[saved['DE']['nll'],saved['BDKN_constant']['nll']],rtol=2e-5,atol=2e-5)
        rows.append(dict(dataset=record['dataset'],scope=record['scope'],split=split,nll_de=nll_de,nll_bdkn=nll_b,nll_bdkn_with_de_variance=nll_swap,
                         median_bdkn_to_de_variance=float(np.median(bv/dv)),mean_epistemic_fraction=float(np.mean(np.var(heads,axis=0)/(alea+np.var(heads,axis=0)))),
                         de_rmse=float(np.mean((y-dm)**2)**.5),bdkn_rmse=float(np.mean((y-bm)**2)**.5),
                         de_standardized_squared_error=float(np.mean((y-dm)**2/dv)),bdkn_standardized_squared_error=float(np.mean((y-bm)**2/bv))))
save_csv('variance_diagnostic_splits.csv',rows)
groups=defaultdict(list)
for row in rows:groups[row['scope'],row['dataset']].append(row)
out=[]
for (scope,dataset),rr in sorted(groups.items()):
    out.append(dict(scope=scope,dataset=dataset,n_splits=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('dataset','scope','split')}))
save_csv('variance_diagnostic_datasets.csv',out)
print(json.dumps(out,indent=2))
