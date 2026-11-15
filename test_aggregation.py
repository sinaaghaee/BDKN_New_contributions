"""Exact coefficient posterior checks, independent of plot appearance."""
import unittest
import numpy as np
from scipy.stats import t
import torch
from aggregation_report import statistics


class AggregationTests(unittest.TestCase):
    def test_matrix_and_spectral_posterior_agree_with_closed_form(self):
        precision=np.array([[4.,1.,.2],[1.,3.,.1],[.2,.1,2.]])
        mean=np.array([-1.,.2,2.])
        state=dict(mean=torch.tensor(mean[:,None]),precision=torch.tensor(precision),alpha=torch.tensor(6.),beta=torch.tensor(3.),sigma_p2=torch.tensor(2.))
        result=statistics(state)
        inv=np.linalg.inv(precision);scale=np.sqrt(.5*np.diag(inv))
        np.testing.assert_allclose(result['scale'],scale)
        np.testing.assert_allclose(result['sd']**2,.6*np.diag(inv))
        np.testing.assert_allclose(result['low'],mean-t.ppf(.975,12)*scale)
        np.testing.assert_allclose(result['high'],mean+t.ppf(.975,12)*scale)
        np.testing.assert_allclose(result['probability_positive'],t.sf(-mean/scale,12))
        # These are coefficient marginals, with no predictive observation-noise +1.
        self.assertFalse(np.allclose(scale,np.sqrt(.5*(1+np.diag(inv)))))
        eigen,vectors=np.linalg.eigh(precision)
        state['precision']=dict(right_vectors=torch.tensor(vectors),eigenvalues=torch.tensor(eigen))
        spectral=statistics(state)
        for key in ('mean','scale','sd','low','high','correlation','covariance_factor'):
            np.testing.assert_allclose(spectral[key],result[key],rtol=1e-12,atol=1e-12)
        self.assertEqual(result['mean'][0],-1.)
        self.assertNotAlmostEqual(float(result['mean'].sum()),1.)

    def test_invalid_posterior_rejected(self):
        state=dict(mean=torch.zeros(2,1),precision=torch.eye(2),alpha=torch.tensor(1.),beta=torch.tensor(1.),sigma_p2=torch.tensor(1.))
        with self.assertRaises(ValueError):statistics(state)
        state['alpha']=torch.tensor(3.)
        state['precision']=dict(right_vectors=torch.eye(2),eigenvalues=torch.tensor([1.,0.]))
        with self.assertRaises(ValueError):statistics(state)


if __name__=='__main__':unittest.main()
