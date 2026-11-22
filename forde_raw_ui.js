/* Fixed-weight FoRDE ablation: the only model change is removal of gradient normalization. */
'use strict';
const catalog=window.FORDE_RAW_CATALOG;
const controls=['dataset','setting','weight','location','stage','observations','xscale','yscale'];
const state=Object.fromEntries(controls.map(id=>[id,document.getElementById(id).dataset.value]));
const cases=new Map();let currentCase,plotCounter=0;
const escapeHTML=text=>String(text??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const requested=new URLSearchParams(location.hash.slice(1));
for(const id of controls){const value=requested.get(id);if(value&&[...document.getElementById(id).querySelectorAll('button')].some(b=>b.dataset.value===value))state[id]=value}
if(!requested.has('dataset')&&!requested.has('setting')){
 const ready=catalog.cases.find(c=>Object.values(c.variants).some(l=>Object.values(l).some(v=>v.raw?.stages?.de)));
 if(ready){state.dataset=ready.dataset;state.setting=ready.setting}
}
function variant(c){return c.variants[state.weight]?.[state.location]}
function entry(v,policy,stage){return policy==='original'?v?.original?.[stage]:v?.[policy]}
function record(v,policy,stage){return entry(v,policy,stage)?.stages?.[stage]}
function selectButtons(){for(const id of controls){const root=document.getElementById(id);root.dataset.value=state[id];root.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===state[id])))}}
function color(value, values, coverage = false) {
  if (!Number.isFinite(value)) return '#eef1f5';
  const score = v => coverage ? Math.abs(v - .95) : v;
  const finite = values.filter(Number.isFinite).map(score), v = score(value);
  if (finite.length <= 1) return 'hsl(60 60% 88%)';
  const rank = finite.filter(x => x < v).length + (finite.filter(x => x === v).length - 1) / 2;
  return `hsl(${120 * (1 - rank / (finite.length - 1))} 60% 88%)`;
}

function renderTables(){
 let html='';const groups=[['Original','original'],['Normalized FoRDE','normalized'],['Raw-gradient FoRDE','raw']];
 const columns=groups.flatMap(([label,policy])=>['de','bdkn'].map(stage=>({label,policy,stage})));
 for(const [metric,title,digits] of [['rmse','Test RMSE',5],['nll','Test NLL',5],['training_seconds','Training time · seconds',2],['coverage','95% interval coverage',3],['interval_width','Mean predictive interval width',3]]){
  html+=`<section><h2>${title}</h2><p class="note">${metric==='coverage'?'Closer to 0.95 is better.':metric==='interval_width'?'Width is shown without a best/worst ranking.':'Lower is better · colors rank within each dataset.'}</p><div class="table-wrap"><table data-metric="${metric}"><thead><tr><th rowspan="2">Dataset</th>${groups.map(([label])=>`<th colspan="2">${label}</th>`).join('')}</tr><tr>${columns.map(c=>`<th>${c.stage.toUpperCase()}</th>`).join('')}</tr></thead><tbody>`;
  for(const c of catalog.cases.filter(c=>c.setting===state.setting)){
   const v=variant(c),values=columns.map(col=>record(v,col.policy,col.stage)?.metrics?.[metric]);
   html+=`<tr><th>${escapeHTML(catalog.datasets[c.dataset].label)}</th>`+columns.map((col,i)=>{
    const value=values[i],e=entry(v,col.policy,col.stage),reason=e?.unavailable||'No matching saved result';
    return Number.isFinite(value)?`<td style="background:${metric==='interval_width'?'#fff':color(value,values,metric==='coverage')}">${value.toFixed(digits)}</td>`:`<td title="${escapeHTML(reason)}">—</td>`;
   }).join('')+'</tr>';
  }
  html+='</tbody></table></div></section>';
 }
 document.getElementById('tables').innerHTML=html;
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

function series(e,key,name,stage='1'){const h=e?.history?.[stage];return {x:h?.epoch,y:h?.[key],name}}
function renderCase(){
 if(!currentCase||currentCase.dataset!==state.dataset||currentCase.setting!==state.setting)return;
 const c=currentCase,v=variant(c),data=c.data,root=document.getElementById('prediction-panels'),loss=document.getElementById('training-panels');clearPlots(root);clearPlots(loss);
 const groups=[['Original','original'],['Normalized FoRDE','normalized'],['Raw-gradient FoRDE','raw']];
 const records=groups.map(([,p])=>record(v,p,state.stage)).filter(Boolean);
 const bounds=[...data.truth,...data.train_y,...data.test_y,...records.flatMap(r=>[...r.prediction.lower,...r.prediction.upper])];const lo=Math.min(...bounds),hi=Math.max(...bounds),pad=.04*Math.max(hi-lo,1e-6);
 for(const [title,policy] of groups){const node=slot(root,title+' · '+state.stage.toUpperCase()),r=record(v,policy,state.stage);if(r)prediction(node,r,data,[lo-pad,hi+pad]);else missing(node,entry(v,policy,state.stage))}
 const a=v?.normalized,b=v?.raw;
 chart(slot(loss,'Stage 1 · total loss'),[series(a,'objective','Normalized'),series(b,'objective','Raw gradients')],'NLL + λR');
 chart(slot(loss,'Stage 1 · data fit'),[series(a,'likelihood','Normalized'),series(b,'likelihood','Raw gradients')],'Data-fit NLL');
 chart(slot(loss,'KDE repulsion loss'),[series(a,'repulsion','Normalized'),series(b,'repulsion','Raw gradients')],'Unweighted R');
 chart(slot(loss,'Stage 2 · BDKN marginal NLL'),[series(a,'objective','Normalized','2'),series(b,'objective','Raw gradients','2')],'Marginal NLL','Optimization step');
 const settings=document.getElementById('resolved-settings');settings.replaceChildren();
 for(const [title,policy] of groups){const e=entry(v,policy,state.stage),h=document.createElement('h3');h.textContent=title;settings.append(h);if(!e?.config){const p=document.createElement('p');p.textContent=e?.unavailable||'Unavailable';settings.append(p);continue}
  const pre=document.createElement('pre');pre.textContent=JSON.stringify(e.config,null,2);settings.append(pre);const links=document.createElement('div');links.className='downloads';for(const item of e.downloads||[]){const a=document.createElement('a');a.href=item.path;a.download='';a.textContent=item.label;links.append(a)}settings.append(links);
 }
}
window.FORDE_RAW_CASE=c=>{cases.set(c.dataset+'/'+c.setting,c);if(c.dataset===state.dataset&&c.setting===state.setting){currentCase=c;document.getElementById('load-status').textContent='';renderCase()}};
function loadCase(){const c=catalog.cases.find(c=>c.dataset===state.dataset&&c.setting===state.setting);if(!c){document.getElementById('load-status').textContent='No selected experiment';return}const key=c.dataset+'/'+c.setting;if(cases.has(key)){currentCase=cases.get(key);renderCase();return}document.getElementById('load-status').textContent='Loading saved results…';const script=document.createElement('script');script.src=c.asset;script.onerror=()=>document.getElementById('load-status').textContent='Cannot load local result asset';document.body.append(script)}
for(const id of controls)document.getElementById(id).addEventListener('click',event=>{const b=event.target.closest('button[data-value]');if(!b||b.disabled||b.dataset.value===state[id])return;state[id]=b.dataset.value;selectButtons();history.replaceState(null,'','#'+new URLSearchParams(state));renderTables();loadCase()});
document.getElementById('show-settings').addEventListener('click',event=>{const target=document.getElementById('resolved-settings');target.hidden=!target.hidden;event.target.textContent=target.hidden?'Show settings':'Hide settings';event.target.setAttribute('aria-expanded',String(!target.hidden))});
document.querySelectorAll('[data-tex]').forEach(n=>katex.render(n.dataset.tex,n,{displayMode:n.tagName==='DIV',throwOnError:false}));
const counts={};catalog.jobs.forEach(j=>counts[j.status]=(counts[j.status]||0)+1);
document.getElementById('progress').textContent=`${counts.complete||0} / ${catalog.jobs.length} raw-gradient fits complete · ${counts.running||0} running · ${counts.pending||0} queued`+(counts.failed?` · ${counts.failed} failed`:'');
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',event=>{const target=document.querySelector(a.getAttribute('href'));if(target){event.preventDefault();target.scrollIntoView({behavior:'smooth'})}}));
selectButtons();history.replaceState(null,'','#'+new URLSearchParams(state));renderTables();loadCase();if(catalog.status==='running')setTimeout(()=>location.reload(),120000);
