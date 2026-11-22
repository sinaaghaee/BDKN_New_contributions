/* Offline interactive plots; scientific predictions/metrics are read from saved fits. */
'use strict';
const catalog = window.ADAPTIVE_CATALOG;
const methodNames = {frde: 'Functional', wrde: 'Weight', forde: 'FoRDE'};
const controls = ['dataset', 'setting', 'location', 'stage', 'weight', 'observations', 'xscale', 'yscale'];
const state = Object.fromEntries(controls.map(id => [id, document.getElementById(id).dataset.value]));
const cases = new Map();
let currentCase;
let plotCounter = 0;
const escapeHTML = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
try {
  const requested = new URLSearchParams(location.hash.slice(1));
  for (const id of controls) {
    const value = requested.get(id);
    if (value && [...document.getElementById(id).querySelectorAll('button')].some(b => b.dataset.value === value)) state[id] = value;
  }
  if (!requested.has('dataset') && !requested.has('setting')) {
    const ready = catalog.cases.find(c => Object.values(c.adaptive).some(e => e.status === 'complete'));
    if (ready) {
      state.dataset = ready.dataset; state.setting = ready.setting;
      if (!requested.has('location')) {
        const available = Object.entries(ready.adaptive).filter(([, e]) => e.status === 'complete');
        const preferred = available.find(([key]) => key.endsWith('/training')) || available[0];
        state.location = preferred[0].split('/')[1];
      }
    }
  }
} catch (_) {}
function selectButtons() {
  for (const id of controls) {
    const group = document.getElementById(id);
    group.dataset.value = state[id];
    group.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.value === state[id])));
  }
}
function storeSelection() {
  history.replaceState(null, '', '#' + new URLSearchParams(state).toString());
}
function entryFor(c, method, policy, stage = state.stage) {
  const location = method === 'wrde' || method === 'original' ? 'training' : state.location;
  if (policy === 'adaptive') return c.adaptive[method + '/' + location];
  const fixed = c.fixed[state.weight] || {};
  const key = method === 'original' ? (stage === 'de' ? 'deep_ensemble' : 'bdkn') : method;
  return fixed[key + '/' + location];
}
function stageOf(entry) { return entry?.stages?.[state.stage]; }
function color(value, values, coverage = false) {
  if (!Number.isFinite(value)) return '#eef1f5';
  const score = v => coverage ? Math.abs(v - .95) : v;
  const finite = values.filter(Number.isFinite).map(score), v = score(value);
  if (finite.length <= 1) return 'hsl(60 60% 88%)';
  const rank = finite.filter(x => x < v).length + (finite.filter(x => x === v).length - 1) / 2;
  return `hsl(${120 * (1 - rank / (finite.length - 1))} 60% 88%)`;
}
function renderTables() {
  const groups = [['Original', 'original', 'fixed'], ...Object.entries(methodNames).flatMap(([m, n]) => [[n + ' · fixed', m, 'fixed'], [n + ' · adaptive', m, 'adaptive']])];
  const columns = groups.flatMap(([name, method, policy]) => ['de', 'bdkn'].map(stage => [name, method, policy, stage]));
  let html = '';
  for (const [metric, title, digits] of [['rmse', 'Test RMSE', 5], ['nll', 'Test NLL', 5], ['training_seconds', 'Training time · seconds', 2], ['coverage', '95% interval coverage', 3], ['interval_width', 'Mean predictive interval width', 3]]) {
    html += `<section><h2>${title}</h2><p class="note">${metric === 'coverage' ? 'Closer to 0.95 is better.' : metric === 'interval_width' ? 'Width is shown without a best/worst ranking.' : 'Lower is better · colors rank methods within each dataset.'}</p><div class="table-wrap"><table data-metric="${metric}"><thead><tr><th rowspan="2" scope="col">Dataset</th>${groups.map(([name]) => `<th colspan="2" scope="colgroup">${name}</th>`).join('')}</tr><tr>${columns.map(([, , , stage]) => `<th scope="col">${stage.toUpperCase()}</th>`).join('')}</tr></thead><tbody>`;
    for (const c of catalog.cases.filter(c => c.setting === state.setting)) {
      const values = columns.map(([, m, p, stage]) => entryFor(c, m, p, stage)?.stages?.[stage]?.metrics?.[metric]);
      html += `<tr><th>${escapeHTML(catalog.datasets[c.dataset]?.label || c.dataset)}</th>` + values.map((value, index) => {
        const [, method, policy, stage] = columns[index], entry = entryFor(c, method, policy, stage);
        const reason = entry?.unavailable || 'No matching saved result';
        return Number.isFinite(value) ? `<td style="background:${metric === 'interval_width' ? '#fff' : color(value, values, metric === 'coverage')}">${value.toFixed(digits)}</td>` : `<td title="${escapeHTML(reason)}">—</td>`;
      }).join('') + '</tr>';
    }
    html += '</tbody></table></div></section>';
  }
  document.getElementById('tables').innerHTML = html;
}
function slot(parent, title) {
  const card = document.createElement('div'); card.className = 'adaptive-card';
  const h = document.createElement('h3'); h.textContent = title; card.appendChild(h);
  const node = document.createElement('div'); node.className = 'adaptive-plot'; node.id = 'adaptive-plot-' + (++plotCounter);
  card.appendChild(node); parent.appendChild(card); return node;
}
function missing(node, entry) {
  node.className = 'adaptive-status'; node.textContent = entry?.unavailable || 'No matching saved result';
}
const plotConfig = {responsive: true, displaylogo: false, toImageButtonOptions: {format: 'svg', filename: 'adaptive-comparison'}};
function prediction(node, record, data, limits) {
  if (!record) return;
  const p = record.prediction, x = data.x;
  const traces = [
    {x, y: p.lower, mode: 'lines', line: {width: 0}, hoverinfo: 'skip', showlegend: false},
    {x, y: p.upper, mode: 'lines', line: {width: 0}, fill: 'tonexty', fillcolor: 'rgba(57,108,169,.18)', name: '95% interval'},
    {x, y: p.mean, mode: 'lines', line: {color: '#396ca9', width: 2}, name: 'Prediction'},
    {x, y: data.truth, mode: 'lines', line: {color: '#172033', dash: 'dash', width: 1.5}, name: 'True function'}
  ];
  for (const [split, label, color] of [['train', 'Training', '#172033'], ['test', 'Test', '#b36a30']]) {
    if (state.observations === split || state.observations === 'both') traces.push({x: data[split + '_x'], y: data[split + '_y'], mode: 'markers', marker: {size: split === 'train' ? 5 : 3, color, opacity: .5}, name: label});
  }
  Plotly.newPlot(node, traces, {margin: {t: 10, l: 50, r: 12, b: 75}, xaxis: {title: 'x', range: [x[0], x.at(-1)]}, yaxis: {title: 'y', range: limits}, legend: {orientation: 'h', y: -.25}, hovermode: 'closest'}, plotConfig);
}
function chart(node, curves, ylabel, xlabel = 'Epoch') {
  curves = curves.filter(c => c.x?.length && c.y?.length);
  if (!curves.length) { missing(node); return; }
  const values = curves.flatMap(c => c.y), signed = state.yscale === 'log' && values.some(v => v <= 0);
  const transform = y => signed ? Math.sign(y) * Math.log10(1 + Math.abs(y) / .001) : y;
  const yaxis = {title: ylabel + (signed ? ' · signed log' : ''), type: state.yscale === 'log' && !signed ? 'log' : 'linear', autorange: true};
  if (signed) {
    const ticks = [0, ...[-1, 1].flatMap(s => [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6].map(p => s * 10 ** p))].sort((a, b) => a - b);
    const min = Math.min(...values), max = Math.max(...values);
    const active = ticks.filter(t => t >= min && t <= max);
    yaxis.tickvals = active.map(transform); yaxis.ticktext = active.map(v => String(Number(v.toPrecision(2))));
  }
  const traces = curves.map((c, i) => ({x: c.x, y: c.y.map(transform), customdata: c.y, mode: 'lines', name: c.name,
    line: {width: 2, dash: c.dash || 'solid', color: c.color || ['#396ca9', '#c06c32', '#5b8c5a', '#8c6bb1'][i % 4]},
    hovertemplate: '%{x}: %{customdata:.6g}<extra>%{fullData.name}</extra>'}));
  Plotly.newPlot(node, traces, {margin: {t: 10, l: 65, r: 12, b: 90}, xaxis: {title: xlabel, type: state.xscale}, yaxis, legend: {orientation: 'h', y: -.3}}, plotConfig);
}
function series(entry, field, name, stage = '1', extra = {}) {
  const h = entry?.history?.[stage];
  return {x: h?.epoch, y: h?.[field], name, ...extra};
}
function clearPlots(parent) {
  parent.querySelectorAll('.js-plotly-plot').forEach(node => Plotly.purge(node));
  parent.replaceChildren();
}
function renderCase() {
  if (!currentCase || currentCase.dataset !== state.dataset || currentCase.setting !== state.setting) return;
  const c = currentCase, data = c.data;
  const predictions = document.getElementById('prediction-panels'), training = document.getElementById('training-panels');
  clearPlots(predictions); clearPlots(training);
  for (const [method, name] of Object.entries(methodNames)) {
    const original = entryFor(c, 'original', 'fixed'), fixed = entryFor(c, method, 'fixed'), adaptive = entryFor(c, method, 'adaptive');
    const heading = document.createElement('h3'); heading.textContent = name + (method === 'wrde' ? ' · input-independent' : ''); predictions.appendChild(heading);
    const row = document.createElement('div'); row.className = 'adaptive-panels'; predictions.appendChild(row);
    const entries = [['Original', original], ['Fixed KDE · λ = ' + state.weight, fixed], ['Adaptive constraint', adaptive]];
    const records = entries.map(([, e]) => stageOf(e)).filter(Boolean);
    const ys = [...data.truth, ...data.train_y, ...data.test_y, ...records.flatMap(r => [...r.prediction.lower, ...r.prediction.upper])];
    const low = Math.min(...ys), high = Math.max(...ys), pad = .04 * Math.max(high - low, 1e-6), limits = [low - pad, high + pad];
    for (const [label, entry] of entries) {
      const node = slot(row, label + ' · ' + state.stage.toUpperCase()), record = stageOf(entry);
      if (record) prediction(node, record, data, limits); else missing(node, entry);
    }
    const section = document.createElement('article'); training.appendChild(section);
    const h = document.createElement('h3'); h.textContent = name; section.appendChild(h);
    const losses = document.createElement('div'); losses.className = 'adaptive-losses'; section.appendChild(losses);
    chart(slot(losses, 'Stage 1 · total objectives'), [series(fixed, 'objective', 'Fixed KDE: NLL + λR'), series(adaptive, 'objective', 'Adaptive: NLL + λ(C − c)')], 'Objective');
    chart(slot(losses, 'Stage 1 · data fit'), [series(fixed, 'likelihood', 'Fixed KDE'), series(adaptive, 'likelihood', 'Adaptive')], 'Data-fit NLL');
    chart(slot(losses, 'Fixed KDE repulsion'), [series(fixed, 'repulsion', 'Unweighted R')], 'R');
    chart(slot(losses, 'Stage 2 · BDKN marginal NLL'), [series(fixed, 'objective', 'Fixed KDE', '2'), series(adaptive, 'objective', 'Adaptive', '2')], 'Marginal NLL', 'Optimization step');
    const controller = document.createElement('div'); controller.className = 'adaptive-controller'; section.appendChild(controller);
    chart(slot(controller, 'Adaptive multiplier'), [series(adaptive, 'lambda_before', 'λ before update'), series(adaptive, 'lambda_after', 'λ after update')], 'λ');
    chart(slot(controller, 'Similarity and target'), [series(adaptive, 'similarity', 'Epoch-end C'), series(adaptive, 'target', 'Fixed target c', '1', {dash: 'dash'})], 'Similarity');
    const violation = series(adaptive, 'violation', 'C − c');
    chart(slot(controller, 'Constraint violation'), [violation, {x: violation.x, y: violation.x?.map(() => 0), name: 'Satisfied at or below zero', dash: 'dash'}], 'C − c');
    if (adaptive?.config?.controller_final) {
      const p = document.createElement('p'); p.className = 'note';
      const final = adaptive.config.controller_final;
      p.textContent = `Final constraint ${final.satisfied ? 'satisfied' : 'violated'} · C − c = ${final.violation.toPrecision(4)} · λ = ${final.multiplier.toPrecision(4)}`;
      section.appendChild(p);
    }
  }
  const settings = document.getElementById('resolved-settings'); settings.replaceChildren();
  for (const [method, name] of Object.entries(methodNames)) {
    for (const [policy, entry] of [['Fixed KDE', entryFor(c, method, 'fixed')], ['Adaptive', entryFor(c, method, 'adaptive')]]) {
      const h = document.createElement('h3'); h.textContent = name + ' · ' + policy; settings.appendChild(h);
      if (!entry?.config) { const p = document.createElement('p'); p.textContent = entry?.unavailable || 'Unavailable'; settings.appendChild(p); continue; }
      const pre = document.createElement('pre'); pre.textContent = JSON.stringify(entry.config, null, 2); settings.appendChild(pre);
      const downloads = document.createElement('div'); downloads.className = 'downloads';
      for (const link of entry.downloads || []) { const a = document.createElement('a'); a.href = link.path; a.download = ''; a.textContent = link.label; downloads.appendChild(a); }
      settings.appendChild(downloads);
    }
  }
}
window.ADAPTIVE_CASE = data => {
  cases.set(data.dataset + '/' + data.setting, data);
  if (data.dataset === state.dataset && data.setting === state.setting) {
    currentCase = data; document.getElementById('load-status').textContent = ''; renderCase();
  }
};
function loadCase() {
  const c = catalog.cases.find(c => c.dataset === state.dataset && c.setting === state.setting);
  if (!c) { document.getElementById('load-status').textContent = 'No selected configuration'; return; }
  const key = c.dataset + '/' + c.setting;
  if (cases.has(key)) { currentCase = cases.get(key); renderCase(); return; }
  document.getElementById('load-status').textContent = 'Loading saved results…';
  const script = document.createElement('script'); script.src = c.asset;
  script.onerror = () => { document.getElementById('load-status').textContent = 'Could not load local results: ' + c.asset; };
  document.body.appendChild(script);
}
for (const id of controls) {
  document.getElementById(id).addEventListener('click', event => {
    const button = event.target.closest('button[data-value]');
    if (!button || button.disabled || state[id] === button.dataset.value) return;
    state[id] = button.dataset.value; selectButtons(); storeSelection();
    renderTables(); loadCase();
  });
}
document.getElementById('show-settings').addEventListener('click', event => {
  const target = document.getElementById('resolved-settings'); target.hidden = !target.hidden;
  event.target.setAttribute('aria-expanded', String(!target.hidden)); event.target.textContent = target.hidden ? 'Show settings' : 'Hide settings';
});
document.querySelectorAll('[data-tex]').forEach(node => katex.render(node.dataset.tex, node, {displayMode: node.tagName === 'DIV', throwOnError: false}));
const counts = {};
catalog.jobs.forEach(j => counts[j.status] = (counts[j.status] || 0) + 1);
document.getElementById('progress').textContent = `${counts.complete || 0} / ${catalog.jobs.length} fits complete` + (counts.running ? ` · ${counts.running} running` : '') + (counts.failed ? ` · ${counts.failed} failed` : '') + ` · ${catalog.status} · updated ${new Date(catalog.updated).toLocaleString()}`;
selectButtons(); renderTables(); loadCase();
if (catalog.status === 'running') setTimeout(() => location.reload(), 120000);
