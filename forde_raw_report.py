"""Offline FoRDE gradient-normalization ablation; read saved results only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

import numpy as np
from run_adaptive import ROOT, read_json, atomic_json, digest
from run_forde_raw import validate_fit
from adaptive_report import write_text, pack_prediction, history_data, baseline_for
from report_ui import CSS, choices, latex


def downloads(folder, output, identifier):
    target = output / 'forde_raw_assets/downloads' / identifier
    target.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'history.csv', 'metrics.json', 'forde_raw/predictions.npz', 'forde_raw_ensemble/predictions.npz']
    links = []
    for name in files:
        destination = target / name.replace('/', '_')
        if not destination.exists() or digest(folder / name) != digest(destination):
            shutil.copy2(folder / name, destination)
        links.append(dict(label=name, path=destination.relative_to(output).as_posix()))
    return links


def install_navigation(output):
    for path in output.rglob('*.html'):
        if path.name == 'forde_gradients.html':
            continue
        text = path.read_text()
        if 'data-forde-raw-nav' in text:
            continue
        import os
        href = os.path.relpath(output / 'forde_gradients.html', path.parent)
        link = f'<a data-forde-raw-nav href="{href}">Gradient normalization</a>'
        if '<nav aria-label="Report pages">' in text:
            text = re.sub(r'(<nav aria-label="Report pages">.*?)</nav>', lambda m: m[1] + link + '</nav>', text, count=1, flags=re.S)
            write_text(path, text)


def build(runs, output):
    manifest = read_json(runs / 'manifest.json')
    registry = read_json(ROOT / 'website/experiments.json')
    groups = {}
    for job in manifest['jobs']:
        groups.setdefault((job['dataset'], job['setting']), []).append(job)
    catalog = dict(status=manifest['status'], datasets=registry['datasets'], cases=[],
        jobs=[{k: j[k] for k in ('id', 'dataset', 'setting', 'weight', 'location', 'status')} for j in manifest['jobs']])
    cache = {}
    rows = []
    for (dataset, setting), jobs in groups.items():
        with np.load(jobs[0]['data'], allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        gi = np.linspace(0, len(data['x_grid']) - 1, min(400, len(data['x_grid'])), dtype=int)
        ti = np.linspace(0, len(data['x_test']) - 1, min(400, len(data['x_test'])), dtype=int)
        case = dict(dataset=dataset, setting=setting, variants={}, data=dict(
            x=data['x_grid'][gi, 0].tolist(), truth=data['f_grid'][gi, 0].tolist(),
            train_x=data['x_train'][:, 0].tolist(), train_y=data['y_train'][:, 0].tolist(),
            test_x=data['x_test'][ti, 0].tolist(), test_y=data['y_test'][ti, 0].tolist()))
        for job in jobs:
            cfg = job['signature']
            normalized = baseline_for(registry, dataset, cfg, 'forde', job['location'], job['weight'], data, output, cache)
            if digest(Path(job['normalized_reference']) / 'config.json') != job['normalized_config_sha256']:
                normalized = dict(unavailable='Normalized reference configuration changed after this selection')
            original = {}
            for stage, method in [('de', 'deep_ensemble'), ('bdkn', 'bdkn')]:
                entry = baseline_for(registry, dataset, cfg, method, 'training', job['weight'], data, output, cache)
                original[stage] = entry
            raw = dict(status=job['status'], unavailable=job.get('error') or 'Raw-gradient fit ' + job['status'])
            if job['status'] == 'complete':
                if not validate_fit(job):
                    raw = dict(status='invalid', unavailable='Raw-gradient artifact validation failed')
                else:
                    folder = Path(job['folder'])
                    stages = {}
                    for row in read_json(folder / 'metrics.json'):
                        stage = 'de' if row['method'].endswith('_ensemble') else 'bdkn'
                        prediction = pack_prediction(folder / row['method'] / 'predictions.npz', data)
                        stages[stage] = dict(prediction=prediction, metrics=row | {k: prediction[k] for k in ('coverage', 'interval_width')})
                        rows.append(dict(dataset=dataset, setting=setting, weight=job['weight'], location=job['location'], stage=stage) | row)
                    raw = dict(status='complete', stages=stages, history=history_data(folder / 'history.csv'),
                        config=read_json(folder / 'config.json'), downloads=downloads(folder, output, job['id']))
            case['variants'].setdefault(format(job['weight'], '.8g'), {})[job['location']] = dict(original=original, normalized=normalized, raw=raw)
        summary = dict(dataset=dataset, setting=setting, variants={})
        for weight, locations in case['variants'].items():
            for location, variant in locations.items():
                def small(entry):
                    return dict(unavailable=entry.get('unavailable'), stages={stage: {'metrics': record['metrics']} for stage, record in entry.get('stages', {}).items()})
                summary['variants'].setdefault(weight, {})[location] = dict(
                    original={stage: small(entry) for stage, entry in variant['original'].items()},
                    normalized=small(variant['normalized']), raw=small(variant['raw']))
        text = 'window.FORDE_RAW_CASE(' + json.dumps(case, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c') + ');\n'
        filename = 'forde_raw_assets/' + hashlib.sha256(text.encode()).hexdigest() + '.js'
        write_text(output / filename, text)
        summary['asset'] = filename
        catalog['cases'].append(summary)
    atomic_json(output / 'forde_raw_manifest.json', catalog)
    write_text(output / 'forde_raw_catalog.js', 'window.FORDE_RAW_CATALOG=' + json.dumps(catalog, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c') + ';\n')
    handle = io.StringIO()
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['dataset', 'setting', 'weight', 'location', 'stage', 'rmse', 'nll', 'training_seconds']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    write_text(output / 'forde_raw_results.csv', handle.getvalue())
    for vendor in ('katex', 'plotly'):
        shutil.copytree(ROOT / 'website/vendor' / vendor, output / 'vendor' / vendor, dirs_exist_ok=True)
    write_text(output / 'forde_raw.js', (ROOT / 'forde_raw_ui.js').read_text())
    write_text(output / 'forde_raw.css', CSS + '\n.raw-panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.raw-losses{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.adaptive-card{min-width:0;border:1px solid #dce3ec;padding:12px;border-radius:6px}.adaptive-plot{height:340px;min-width:0}.adaptive-status{padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:1000px){.raw-panels,.raw-losses{grid-template-columns:1fr}}')
    jobs = manifest['jobs']
    controls = choices('dataset', 'Dataset', [(k, registry['datasets'][k]['label']) for k in dict.fromkeys(j['dataset'] for j in jobs)], jobs[0]['dataset'])
    controls += choices('setting', 'Training setting', [(k, f'{k.capitalize()} · LR {next(j["signature"]["lr"] for j in jobs if j["setting"] == k):g} · {next(j["signature"]["epochs"] for j in jobs if j["setting"] == k):,} epochs') for k in dict.fromkeys(j['setting'] for j in jobs)], jobs[0]['setting'])
    controls += choices('weight', 'Repulsion weight', [(format(w, '.8g'), format(w, '.8g')) for w in sorted({j['weight'] for j in jobs})], format(jobs[0]['weight'], '.8g'))
    controls += choices('location', 'Repulsion inputs', [(k, {'training': 'Training', 'random': 'Random', 'combined': 'Training + random'}[k]) for k in dict.fromkeys(j['location'] for j in jobs)], jobs[0]['location'])
    controls += choices('stage', 'Prediction plots', [('de', 'DE'), ('bdkn', 'BDKN')], 'de')
    controls += choices('observations', 'Observations', [('train', 'Training'), ('test', 'Test'), ('both', 'Both'), ('none', 'None')], 'train')
    axes = choices('xscale', 'Loss X', [('linear', 'Linear'), ('log', 'Log')], 'log') + choices('yscale', 'Loss Y', [('linear', 'Linear'), ('log', 'Log / signed log')], 'log')
    formulas = ''.join(latex(eq) for eq in [r'g_i=(\partial\mu_i(\widetilde s_b)/\partial\widetilde s_b)_{b=1}^{P}',
        r'z_i^{\mathrm{normalized}}=g_i/\sqrt{\|g_i\|^2+\varepsilon_g},\qquad z_i^{\mathrm{raw}}=g_i',
        r'd_{ij}=\|z_i-\operatorname{sg}(z_j)\|^2,\qquad h_j=\operatorname{sg}\!\left(\max\{\operatorname{median}_{i}d_{ij}/\log H,\varepsilon_h\}\right)',
        r'L_1=L_{\mathrm{fit}}+\lambda\sum_i\log\!\left[\sum_j\exp(-d_{ij}/h_j)\right]'])
    body = '<section><h2>FoRDE gradient normalization</h2><p id="progress" aria-live="polite"></p><div class="toolbar">' + controls + '</div><nav><a href="#tables">Results</a><a href="#predictions">Predictions</a><a href="#training">Losses</a><a href="#definitions">Math</a><a href="#settings">Settings</a></nav></section>'
    body += '<div id="tables"></div><section id="predictions"><h2>Predictions</h2><p id="load-status" aria-live="polite"></p><div class="raw-panels" id="prediction-panels"></div></section>'
    body += '<section id="training"><h2>Training histories</h2><div class="toolbar">' + axes + '</div><div class="raw-losses" id="training-panels"></div></section>'
    body += '<section id="definitions"><h2>One change: raw input gradients</h2>' + formulas + '<p>The complete 1D gradient field remains one particle. The raw variant retains gradient magnitude. Bandwidth estimation, stop-gradient references, self terms, likelihood, input preprocessing, random-point sampling, and Bayesian stage two retain their matched normalized settings. Random points redraw each optimizer step and exclude all training inputs; combined uses B training plus B random inputs.</p></section>'
    body += '<section id="settings"><h2>Settings and downloads</h2><button id="show-settings" aria-expanded="false">Show settings</button><div id="resolved-settings" hidden></div><p><a href="forde_raw_results.csv" download>Raw-gradient results CSV</a></p></section>'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gradient normalization · Regression comparison</title><link rel="stylesheet" href="forde_raw.css"><link rel="stylesheet" href="vendor/katex/katex.min.css"></head><body><main><header><h1>Regression comparison</h1></header><nav aria-label="Report pages"><a href="index.html">Results</a><a href="adaptive.html">Adaptive repulsion</a><a href="forde_gradients.html" aria-current="page">Gradient normalization</a></nav>' + body + '</main><script src="vendor/katex/katex.min.js"></script><script src="vendor/plotly/plotly-basic-2.35.2.min.js"></script><script src="forde_raw_catalog.js"></script><script src="forde_raw.js"></script></body></html>'
    write_text(output / 'forde_gradients.html', html)
    install_navigation(output)
    print(f'Gradient comparison: {sum(j["status"] == "complete" for j in jobs)}/{len(jobs)} fits · {output}/forde_gradients.html', flush=True)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'results/forde_raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'website/dist')
    args = parser.parse_args()
    build(args.runs, args.output)


if __name__ == '__main__':
    main()
