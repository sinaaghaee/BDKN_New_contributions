"""Pinned upstream code loader. Cached upstream files remain byte-for-byte intact.

Compatibility changes occur in memory: np.int -> int, the factory's import
alias, symmetric torch.eig -> torch.linalg.eigh, and removal of a broken
duplicate SGD.step that overwrites the author's working definition.
No losses, gradients, bandwidths, priors, jitter or optimizer settings are changed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import types
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
LOCK = json.loads((ROOT / 'author_source.lock.json').read_text())
CACHE = ROOT / '.cache/repulsive_ensembles'
COMPATIBILITY = [
    'data/dataset.py: np.int replaced with built-in int (same integer dtype)',
    'methods/method_utils.py: f_SVGD resolved to methods/f_SVGD.py',
    'utils/SSGE_squeeze.py: symmetric torch.eig replaced by torch.linalg.eigh; all eigenpairs retained, no clipping',
    'methods/SVGD.py: remove broken duplicate SGD.step; retain first upstream SGD.step verbatim',
]


def blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def prepare_source():
    """Fetch only missing pinned files; verify every Git blob before execution."""
    for name, expected in LOCK['files'].items():
        path = CACHE / name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/{LOCK["repository"]}/{LOCK["revision"]}/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if blob_sha(content) != expected:
                raise ValueError('Upstream source hash mismatch: ' + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if blob_sha(path.read_bytes()) != expected:
            raise ValueError('Cached upstream source was modified: ' + str(path))
    return CACHE


def legacy_symmetric_eig(matrix, eigenvectors=True):
    """Legacy eig's (real, imaginary) result for the upstream symmetric Gram."""
    import torch
    values, vectors = torch.linalg.eigh(matrix)
    return torch.stack((values, torch.zeros_like(values)), dim=1), vectors


def load_upstream():
    """Load actual upstream modules, isolated from this repo's scripts."""
    if '_author_loaded' in sys.modules:
        return sys.modules['_author_loaded']
    prepare_source()
    for name in ('models', 'methods', 'utils', 'data', 'data.toy_regression', 'training'):
        package = types.ModuleType(name)
        package.__path__ = [str(CACHE / name.replace('.', '/'))]
        sys.modules[name] = package
    def module(name, relative=None):
        path = CACHE / (relative or name.replace('.', '/') + '.py')
        source = path.read_text()
        if name == 'data.dataset':
            source = source.replace('np.int,', 'int,')
        if name == 'utils.SSGE_squeeze':
            source = source.replace('torch.eig(Kxx, eigenvectors=True)', '_legacy_symmetric_eig(Kxx, eigenvectors=True)')
        if name == 'methods.SVGD':
            start = source.index('class SGD:')
            duplicate = source.index('    def step(self, W,X,T,step,X_add = None):', start)
            end = source.index('class SGLD:', duplicate)
            source = source[:duplicate] + source[end:]
        result = types.ModuleType(name)
        result.__file__ = str(path)
        result._legacy_symmetric_eig = legacy_symmetric_eig
        sys.modules[name] = result
        exec(compile(source, str(path), 'exec'), result.__dict__)
        return result
    modules = {}
    for name in ('utils.utils', 'utils.kernel', 'utils.SSGE_squeeze', 'utils.distributions',
                 'utils.config', 'models.MLP', 'models.ensemble', 'methods.SVGD',
                 'methods.WGD', 'methods.f_SVGD'):
        modules[name] = module(name)
    sys.modules['f_SVGD'] = modules['methods.f_SVGD']
    for name in ('methods.method_utils', 'data.dataset', 'data.toy_regression.regression1d_data',
                 'data.toy_regression.generete_regression', 'training.training_1d_regre'):
        modules[name] = module(name)
    result = types.SimpleNamespace(modules=modules)
    sys.modules['_author_loaded'] = result
    return result


def original_data():
    return load_upstream().modules['data.toy_regression.generete_regression'].generate_1d_dataset(
        show_plots=False, task_set=4, data_random_seed=42)


def array_handler(arrays, native=False, normalize=False):
    """Use upstream Dataset batching, including its test-batch RNG consumption."""
    import numpy as np
    upstream = load_upstream()
    if native:
        handler = original_data()
        assert np.array_equal(handler.get_train_inputs(), arrays['x_train'])
        assert np.array_equal(handler.get_train_outputs(), arrays['y_train'])
    else:
        np.random.seed(42)
        cls = upstream.modules['data.toy_regression.regression1d_data'].ToyRegression
        handler = cls.__new__(cls)
        upstream.modules['data.dataset'].Dataset.__init__(handler)
        n = len(arrays['x_train'])
        test_x=arrays.get('batch_test_x',arrays['x_test'])
        test_y=arrays.get('batch_test_y',arrays['y_test'])
        handler._data.update(classification=False, sequence=False, in_shape=[1], out_shape=[1],
            in_data=np.concatenate((arrays['x_train'], test_x)),
            out_data=np.concatenate((arrays['y_train'], test_y)),
            train_inds=np.arange(n), test_inds=np.arange(n, n + len(test_x)))
    offset = arrays['x_train'].mean(0) if normalize else np.zeros(1)
    scale = arrays['x_train'].std(0) if normalize else np.ones(1)
    scale = np.where(scale == 0, 1., scale)
    if normalize:
        handler._data['in_data'] = (handler._data['in_data'] - offset) / scale
    return handler, offset, scale


if __name__ == '__main__':
    print(prepare_source())
    print('Verified upstream revision:', LOCK['revision'])
