"""Build the adaptive comparison in the existing offline report; never train models."""
import argparse
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import numpy as np
from run_adaptive import ROOT, read_json, atomic_json, validate_fit, digest

NAMES = {'frde': 'Functional', 'wrde': 'Weight', 'forde': 'FoRDE'}
MATCH_KEYS = ('seed', 'epochs', 'lr', 'batch_size', 'n_nets', 'hidden_dim', 'hidden_layers',
              'activation', 'norm_type', 'jitter_eps', 'normalize_x', 'normalize_y', 'x_norm_type')


def write_text(path, text):
    """Atomic replace, so an open file:// report never reads half-written assets."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        path.chmod(0o644)
        return
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.chmod(0o644)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pack_prediction(path, data):
    with np.load(path, allow_pickle=False) as p:
        for key in ('x_grid', 'x_test', 'y_test', 'f_grid'):
            if not np.array_equal(p[key], data[key]):
                raise ValueError('Prediction/data mismatch: ' + str(path))
        indexes = np.linspace(0, len(p['x_grid']) - 1, min(400, len(p['x_grid'])), dtype=int)
        result = {key: p['grid_' + key][indexes].tolist() for key in ('mean', 'lower', 'upper')}
        for split in ('grid', 'test'):
            for key in ('mean', 'lower', 'upper', 'scale'):
                values = p[split + '_' + key]
                if not np.isfinite(values).all() or len(values) != len(data['x_' + split]):
                    raise ValueError('Invalid prediction array: ' + str(path))
            if np.any(p[split + '_scale'] <= 0) or np.any(p[split + '_lower'] > p[split + '_upper']):
                raise ValueError('Invalid prediction interval: ' + str(path))
        result['coverage'] = float(np.mean((data['y_test'].reshape(-1) >= p['test_lower']) & (data['y_test'].reshape(-1) <= p['test_upper'])))
        result['interval_width'] = float(np.mean(p['test_upper'] - p['test_lower']))
    return result


def history_data(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for stage in ('1', '2'):
        values = [r for r in rows if r['stage'] == stage]
        if not values:
            continue
        indexes = np.unique(np.r_[0, np.linspace(0, len(values) - 1, min(500, len(values)), dtype=int), len(values) - 1])
        result[stage] = {key: [float(values[i][key]) for i in indexes] for key in values[0]
            if key != 'stage' and all(r.get(key, '') != '' for r in values)}
        result[stage]['count'] = len(values)
    return result


def copy_downloads(folder, output, identifier, labels):
    target = output / 'adaptive_assets' / 'downloads' / identifier
    target.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'history.csv', 'metrics.json']
    if (folder / 'reference_inputs.npz').exists():
        files += ['reference_inputs.npz']
    files += [label + '/predictions.npz' for label in labels]
    links = []
    for file in files:
        source = folder / file
        destination = target / file.replace('/', '_')
        if not destination.exists() or digest(source) != digest(destination):
            shutil.copy2(source, destination)
        links.append(dict(label=file, path=destination.relative_to(output).as_posix()))
    return links


def baseline_for(registry, dataset, cfg, method, location, weight, data, output, cache):
    """Only reuse a baseline with matching data and relevant training configuration."""
    location = location if method in ('frde', 'forde') else 'training'
    candidates = [e for e in registry['experiments'] if e['dataset'] == dataset
        and float(e['learning_rate']) == cfg['lr'] and float(e['repulsion']) == weight
        and e.get('repulsion_location', 'training') == location]
    errors = []
    for entry in candidates:
        folder = ROOT / entry['run'] / method
        key = (str(folder), cfg['dataset_sha256'], tuple(cfg[k] for k in MATCH_KEYS),
            tuple(cfg.get(k) for k in ('stage2_steps', 'stage2_lr', 'alpha0', 'beta0', 'learn_sigma_p2', 'Beta_var_prior', 'phi_add_identity', 'phi_add_bias', 'bll_dtype')), weight, location)
        if key in cache:
            return cache[key]
        try:
            saved = read_json(folder / 'config.json')
            if saved['dataset_sha256'] != cfg['dataset_sha256'] or any(saved.get(k) != cfg[k] for k in MATCH_KEYS):
                errors.append('training configuration differs')
                continue
            if method in NAMES and (saved.get('repulsion_strength') != weight or saved.get('repulsion_location', 'training') != location):
                errors.append('fixed repulsion weight or location differs')
                continue
            if method != 'deep_ensemble' and any(saved.get(k) != cfg[k] for k in ('stage2_steps', 'stage2_lr', 'alpha0', 'beta0', 'learn_sigma_p2', 'Beta_var_prior', 'phi_add_identity', 'phi_add_bias', 'bll_dtype')):
                errors.append('BDKN configuration differs')
                continue
            labels = [method] if method in ('bdkn', 'deep_ensemble') else [method + '_ensemble', method]
            metrics = {r['method']: r for r in read_json(folder / 'metrics.json')}
            histories = history_data(folder / 'history.csv')
            if histories.get('1', {}).get('count') != cfg['epochs']:
                raise ValueError('Incomplete baseline first-stage history')
            if method != 'deep_ensemble' and histories.get('2', {}).get('count') != cfg['stage2_steps']:
                raise ValueError('Incomplete baseline second-stage history')
            packed = {}
            for label in labels:
                row = metrics[label]
                if row['dataset_sha256'] != cfg['dataset_sha256'] or not np.isfinite([row['rmse'], row['nll'], row['training_seconds']]).all():
                    raise ValueError('Invalid baseline metrics')
                prediction = pack_prediction(folder / label / 'predictions.npz', data)
                stage = 'de' if label.endswith('_ensemble') or label == 'deep_ensemble' else 'bdkn'
                packed[stage] = dict(metrics=row | {k: prediction[k] for k in ('coverage', 'interval_width')}, prediction=prediction)
            identifier = hashlib.sha256(str(folder).encode()).hexdigest()[:16]
            result = dict(stages=packed, history=histories, config=saved, source=entry['run'] + '/' + method,
                downloads=copy_downloads(folder, output, 'fixed-' + identifier, labels))
            cache[key] = result
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    return dict(unavailable='; '.join(sorted(set(errors))) or 'No matching saved baseline')


def install_navigation(output):
    """Link already generated profiles without rebuilding their numerical figures."""
    for path in output.rglob('*.html'):
        if path.name == 'adaptive.html':
            continue
        text = path.read_text()
        if 'data-adaptive-nav' in text:
            path.chmod(0o644)
            continue
        link = os.path.relpath(output / 'adaptive.html', path.parent)
        text = re.sub(r'(<nav aria-label="Report pages">)(.*?)</nav>',
            lambda m: m[1] + m[2] + f'<a data-adaptive-nav href="{link}">Adaptive repulsion</a></nav>', text, count=1, flags=re.S)
        write_text(path, text)


def definitions():
    from report_ui import latex
    body = '<section id="definitions"><h2>Adaptive loss definitions</h2>'
    for title, expression in (
        ('Functional repulsion', r'z_h=(\mu_h(s_b))_{b=1}^{P}'),
        ('Weight repulsion', r'z_h=\operatorname{vec}(\theta_h)'),
        ('FoRDE', r'g_h=(\partial\mu_h(s_b)/\partial s_b)_{b=1}^{P},\qquad z_h=g_h/\sqrt{\|g_h\|^2+\varepsilon_g}')):
        body += '<h3>' + title + '</h3>' + latex(expression)
    for expression in (
        r'C(\Theta)=\frac{1}{H(H-1)}\sum_{i\ne j}\exp\!\left(-\frac{\|z_i-z_j\|^2}{h_0}\right)',
        r'h_0=\max\{\operatorname{median}_{i<j}\|z_i^{(0)}-z_j^{(0)}\|^2,\varepsilon_h\},\qquad c=rC(\Theta_0)',
        r'\min_\Theta L_{\mathrm{fit}}(\Theta)\quad\mathrm{subject\ to}\quad C(\Theta)\le c',
        r'\mathcal L(\Theta,\lambda)=L_{\mathrm{fit}}(\Theta)+\lambda(C(\Theta)-c)',
        r'\lambda_{t+1}=\max\{0,\lambda_t+\eta_\lambda[C(\Theta_{t+1})-c]\}',
        r'L_{\mathrm{fit}}=\sum_{h=1}^{H}\frac{1}{2B}\sum_{b=1}^{B}\left[\log(2\pi v_h(x_{hb}))+(y_{hb}-\mu_h(x_{hb}))^2/v_h(x_{hb})\right]'):
        body += latex(expression)
    body += '<p>H: network count; P: fixed reference count; μ and v: predicted mean and variance; θ: trainable parameters; r: target ratio (default 0.8); ηλ: multiplier step size (default 0.1); λ₀ = 0; εh = 10⁻⁶; εg = 10⁻⁸. Both sides of every pair receive gradients. The bandwidth, reference inputs, and target remain fixed.</p>'
    body += '<p>Training uses all training inputs. Random uses an equally sized fixed uniform domain set excluding training inputs. Combined uses their union. Weight repulsion is input-independent. Stage two retains the existing BDKN Bayesian marginal-NLL objective on frozen trained features.</p>'
    body += '<p>Stage-one curves average pre-update minibatch losses. Similarity and violation curves use the full reference set after each epoch; λ after the update is shown separately. The Lagrangian includes −λc and can be negative.</p>'
    body += '<p>This is a proposed constrained-diversity extension, distinct from the saved KDE objectives. Target ratio, bandwidth rule, and step sizes are experimental choices. Nonconvex training has no asserted global-convergence or uncertainty-calibration guarantee.</p>'
    body += '<p>References: <a href="https://web.stanford.edu/~boyd/cvxbook/">Boyd–Vandenberghe, Chapter 5</a>; <a href="https://arxiv.org/abs/1812.05905">Haarnoja et al., Section 5</a>; <a href="https://arxiv.org/abs/2306.02775">Trinh et al., FoRDE</a>.</p></section>'
    return body


def build(runs, output, registry_path=None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(runs / 'manifest.json')
    registry = read_json(registry_path or ROOT / 'website/experiments.json')
    jobs = manifest['jobs']
    catalog = dict(updated=datetime.now(timezone.utc).isoformat(), status=manifest['status'],
        datasets=registry['datasets'], cases=[], jobs=[{k: j[k] for k in ('id', 'dataset', 'setting', 'method', 'location', 'status')} for j in jobs])
    cache = {}
    groups = {}
    for job in jobs:
        groups.setdefault((job['dataset'], job['setting']), []).append(job)
    summaries = []
    for (dataset, setting), group in groups.items():
        data_path = Path(group[0]['data'])
        with np.load(data_path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        cfg = group[0]['signature']
        configs_by_method = {job['method']: job['signature'] for job in group}
        grid_index = np.linspace(0, len(data['x_grid']) - 1, min(400, len(data['x_grid'])), dtype=int)
        test_index = np.linspace(0, len(data['x_test']) - 1, min(400, len(data['x_test'])), dtype=int)
        case = dict(dataset=dataset, setting=setting, lr=cfg['lr'], epochs=cfg['epochs'], adaptive={}, fixed={},
            data=dict(x=data['x_grid'][grid_index, 0].tolist(), truth=data['f_grid'][grid_index, 0].tolist(),
                train_x=data['x_train'][:, 0].tolist(), train_y=data['y_train'][:, 0].tolist(),
                test_x=data['x_test'][test_index, 0].tolist(), test_y=data['y_test'][test_index, 0].tolist()))
        for job in group:
            method, location = job['method'], job['location']
            entry = dict(status=job['status'], unavailable='Adaptive fit ' + job['status'])
            if job['status'] == 'complete':
                if not validate_fit(job):
                    entry = dict(status='invalid', unavailable='Saved adaptive fit failed validation')
                else:
                    folder = Path(job['folder'])
                    saved = read_json(folder / 'config.json')
                    stages = {}
                    labels = ['adaptive_' + method + suffix for suffix in ('_ensemble', '')]
                    for row in read_json(folder / 'metrics.json'):
                        prediction = pack_prediction(folder / row['method'] / 'predictions.npz', data)
                        stage = 'de' if row['method'].endswith('_ensemble') else 'bdkn'
                        stages[stage] = dict(metrics=row | {k: prediction[k] for k in ('coverage', 'interval_width')}, prediction=prediction)
                    entry = dict(status='complete', stages=stages, history=history_data(folder / 'history.csv'), config=saved,
                        downloads=copy_downloads(folder, output, job['id'], labels))
            case['adaptive'][method + '/' + location] = entry
        weights = sorted({float(e['repulsion']) for e in registry['experiments'] if e['dataset'] == dataset and float(e['learning_rate']) == cfg['lr']})
        for weight in weights:
            weight_key = format(weight, '.8g')
            case['fixed'][weight_key] = {}
            for method in ('deep_ensemble', 'bdkn', 'frde', 'wrde', 'forde'):
                for location in (('training', 'random', 'combined') if method in ('frde', 'forde') else ('training',)):
                    case['fixed'][weight_key][method + '/' + location] = baseline_for(
                        registry, dataset, configs_by_method.get(method, cfg), method, location, weight, data, output, cache)
        # Keep an offline summary for all-dataset tables, lazy-load the actual curves.
        summary = {k: case[k] for k in ('dataset', 'setting', 'lr', 'epochs')}
        for policy in ('fixed', 'adaptive'):
            def small(entry):
                return {k: v for k, v in entry.items() if k in ('status', 'unavailable')} | {
                    'stages': {stage: {'metrics': item['metrics']} for stage, item in entry.get('stages', {}).items()}}
            summary[policy] = {k: {key: small(value) for key, value in entry.items()} for k, entry in case[policy].items()} if policy == 'fixed' else {key: small(value) for key, value in case[policy].items()}
        raw = json.dumps(case, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
        token = hashlib.sha256(raw.encode()).hexdigest()[:20]
        filename = f'adaptive_assets/{dataset}-{setting}-{token}.js'
        write_text(output / filename, 'window.ADAPTIVE_CASE(' + raw + ');\n')
        summary['asset'] = filename
        catalog['cases'].append(summary)
        summaries += [dict(dataset=dataset, setting=setting, base_method=method, location=location, stage=stage) | record['metrics'] for key, entry in case['adaptive'].items() for method, location in [key.split('/')]
            for stage, record in entry.get('stages', {}).items()]
    atomic_json(output / 'adaptive_manifest.json', catalog)
    write_text(output / 'adaptive_catalog.js', 'window.ADAPTIVE_CATALOG=' + json.dumps(catalog, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c') + ';\n')
    fields = list(dict.fromkeys(k for row in summaries for k in row)) or ['dataset', 'setting', 'method', 'location', 'stage', 'rmse', 'nll', 'training_seconds']
    import io
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summaries)
    write_text(output / 'adaptive_results.csv', handle.getvalue())
    for vendor in ('katex', 'plotly'):
        shutil.copytree(ROOT / 'website/vendor' / vendor, output / 'vendor' / vendor, dirs_exist_ok=True)
    write_text(output / 'adaptive.js', (ROOT / 'adaptive_ui.js').read_text())
    from report_ui import CSS
    write_text(output / 'adaptive.css', CSS + '\n.adaptive-panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.adaptive-plot{height:330px;min-width:0}.adaptive-card{min-width:0;border:1px solid #dce3ec;padding:12px;border-radius:6px}.adaptive-card h3{margin:0 0 8px}.adaptive-controller{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.adaptive-losses{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.adaptive-status{padding:8px 0}.adaptive-plot .modebar{font-size:14px}@media(max-width:1000px){.adaptive-panels,.adaptive-controller,.adaptive-losses{grid-template-columns:1fr}}')
    from report_ui import choices
    controls = choices('dataset', 'Dataset', [(k, v['label']) for k, v in registry['datasets'].items() if any(j['dataset'] == k for j in jobs)], jobs[0]['dataset'])
    controls += choices('setting', 'Training setting', [(s, f'{s.capitalize()} · LR {next(j["signature"]["lr"] for j in jobs if j["setting"] == s)} · {next(j["signature"]["epochs"] for j in jobs if j["setting"] == s)} epochs') for s in dict.fromkeys(j['setting'] for j in jobs)], jobs[0]['setting'])
    controls += choices('location', 'Reference inputs', [('training', 'Training'), ('random', 'Random'), ('combined', 'Training + random')], 'training')
    controls += choices('stage', 'Prediction plots', [('de', 'DE'), ('bdkn', 'BDKN')], 'de')
    controls += choices('weight', 'Fixed KDE baseline weight', [(format(w, '.8g'), format(w, '.8g')) for w in sorted({float(e['repulsion']) for e in registry['experiments']})], '0.001')
    controls += choices('observations', 'Observations', [('train', 'Training'), ('test', 'Test'), ('both', 'Both'), ('none', 'None')], 'train')
    loss_controls = choices('xscale', 'Loss X', [('linear', 'Linear'), ('log', 'Log')], 'log') + choices('yscale', 'Loss Y', [('linear', 'Linear'), ('log', 'Log / signed log')], 'log')
    body = '<section><h2>Adaptive repulsion</h2><p id="progress" aria-live="polite"></p><p id="load-status" aria-live="polite"></p><div class="toolbar">' + controls + '</div><nav><a href="#tables">Results</a><a href="#predictions">Predictions</a><a href="#training">Training</a><a href="#definitions">Loss definitions</a><a href="#settings">Settings and downloads</a></nav></section>'
    body += '<div id="tables"></div><section id="predictions"><h2>Predictions</h2><p>95% predictive intervals · matching axes within each method row.</p><div id="prediction-panels"></div></section>'
    body += '<section id="training"><h2>Training histories</h2><div class="toolbar">' + loss_controls + '</div><div id="training-panels"></div></section>'
    body += definitions() + '<section id="settings"><h2>Settings and downloads</h2><button id="show-settings" type="button" aria-expanded="false">Show settings</button><div id="resolved-settings" hidden></div><p><a href="adaptive_results.csv" download>Adaptive results CSV</a></p></section>'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adaptive repulsion · Regression comparison</title><link rel="stylesheet" href="adaptive.css"><link rel="stylesheet" href="vendor/katex/katex.min.css"></head><body><main><header><h1>Regression comparison</h1></header><nav aria-label="Report pages"><a href="index.html">Results</a><a href="figures.html">Predictions</a><a href="adaptive.html" aria-current="page">Adaptive repulsion</a></nav>' + body + '</main><script src="vendor/katex/katex.min.js"></script><script src="vendor/plotly/plotly-basic-2.35.2.min.js"></script><script src="adaptive_catalog.js"></script><script src="adaptive.js"></script></body></html>'
    if (output / 'forde_gradients.html').exists():
        html = html.replace('<nav aria-label="Report pages">', '<nav aria-label="Report pages"><a data-forde-raw-nav href="forde_gradients.html">Gradient normalization</a>', 1)
    write_text(output / 'adaptive.html', html)
    install_navigation(output)
    complete = sum(j['status'] == 'complete' for j in jobs)
    print(f'Adaptive website: {complete}/{len(jobs)} completed fits · {output}/adaptive.html', flush=True)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'results/adaptive')
    parser.add_argument('--output', type=Path, default=ROOT / 'website/dist')
    args = parser.parse_args()
    build(args.runs, args.output)


if __name__ == '__main__':
    main()
