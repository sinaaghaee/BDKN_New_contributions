/* Offline author-code comparison. All predictions come from saved fits. */
'use strict';
const catalog=window.AUTHOR_CATALOG, cases=window.AUTHOR_CASES||(window.AUTHOR_CASES={});
const loading=new Map();
function loadCase(dataset){
  if(cases[dataset])return Promise.resolve(cases[dataset]);
  if(!loading.has(dataset))loading.set(dataset,new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='author_assets/'+encodeURIComponent(dataset)+'.js';script.onload=()=>resolve(cases[dataset]);script.onerror=()=>{loading.delete(dataset);reject(new Error('Could not load saved dataset figures'));};document.body.appendChild(script);}));
  return loading.get(dataset);
}
const ids=['dataset','preset','original-setting','weight','location','evaluation','observations','stage','xscale','yscale'];
const state=Object.fromEntries(ids.map(id=>[id,document.getElementById(id).dataset.value]));
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const plotConfig={responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'author-comparison'}};
const groups=[['Author baseline','author','SGD',['de','bdkn']],['Author KDE-WGD','author','kde_WGD',['de','bdkn']],['Author functional KDE-WGD','author','kde_f_WGD',['de','bdkn']],['D’Angelo + author FoRDE','author','official_forde',['de','bdkn']],['FoRDE matched control','author','official_forde_control',['de','bdkn']],['Original','original','original',['de','bdkn']],['Functional','original','frde',['de','bdkn']],['Weight','original','wrde',['de','bdkn']],['Our FoRDE','original','forde',['de','bdkn']],...['gp','dkl','ldbll','vbll'].map(m=>[m.toUpperCase(),'original',m,['de']])];
try{const saved=new URLSearchParams(location.hash.slice(1));for(const id of ids){const value=saved.get(id);if(value&&[...document.getElementById(id).querySelectorAll('button')].some(b=>b.dataset.value===value))state[id]=value;}}catch(_){}
function lookup(c,kind,method,stage){
  if(!c)return {};
  let entry;
  if(kind==='author')entry=c.author[state.preset+'/'+method];
  else{
    if(method==='original')method=stage==='de'?'deep_ensemble':'bdkn';
    const repulsive=['frde','wrde','forde'].includes(method);
    const input=['frde','forde'].includes(method)?state.location:'training';
    entry=c.original[`${state['original-setting']}/${repulsive?state.weight:'any'}/${input}/${method}`];
  }
  const record=(entry?.ids||[]).map(id=>c.records[id]).find(r=>r.stage===stage);
  return {record,entry};
}
function selected(c){return groups.flatMap(([title,kind,method,stages])=>stages.map(stage=>({title,kind,method,stage,...lookup(c,kind,method,stage)})));}
function ranked(value,values){const finite=values.filter(Number.isFinite);if(finite.length<2)return 'hsl(60 60% 88%)';const rank=finite.filter(v=>v<value).length+(finite.filter(v=>v===value).length-1)/2;return `hsl(${120*(1-rank/(finite.length-1))} 60% 88%)`;}
function metric(record,name){if(!record)return undefined;if(state.evaluation==='native'&&name!=='training_seconds')return record.metrics.native?.[name];return record.metrics[name];}
function tables(){
  let html='';
  for(const [name,title,digits] of [['rmse','Test RMSE',5],['nll','Test NLL',5],['training_seconds','Training time · seconds',2]]){
    html+=`<section><h2>${title}</h2><p class="note">Lower is better · green to red within each dataset. BDKN time includes ensemble training and the second stage.</p><div class="table-wrap"><table data-metric="${name}"><thead><tr><th rowspan="2">Dataset</th>${groups.map(([title,,,stages])=>`<th colspan="${stages.length}">${title}</th>`).join('')}</tr><tr>${groups.flatMap(([,,,stages])=>stages.map(s=>`<th>${s==='bdkn'?'BDKN':'DE'}</th>`)).join('')}</tr></thead><tbody>`;
    for(const [key,label] of Object.entries(catalog.datasets)){
      const entries=selected(catalog.overview[key]),values=entries.map(e=>metric(e.record,name));
      html+=`<tr><th><button data-dataset="${escapeHTML(key)}">${escapeHTML(label)}</button></th>`+entries.map((entry,i)=>Number.isFinite(values[i])?`<td style="background:${ranked(values[i],values)}">${values[i].toFixed(digits)}</td>`:`<td title="${escapeHTML(entry.entry?.error||entry.entry?.status||'No saved result for this selection')}">—</td>`).join('')+'</tr>';
    }
    html+='</tbody></table></div></section>';
  }
  document.getElementById('tables').innerHTML=html;
  document.querySelectorAll('[data-dataset]').forEach(button=>button.onclick=()=>{state.dataset=button.dataset.dataset;render();});
}
function clear(id){const node=document.getElementById(id);node.querySelectorAll('.js-plotly-plot').forEach(plot=>Plotly.purge(plot));node.replaceChildren();return node;}
function card(parent,title){const wrapper=document.createElement('div');wrapper.className='author-card';const h=document.createElement('h3');h.textContent=title;wrapper.appendChild(h);const plot=document.createElement('div');plot.className='author-plot';wrapper.appendChild(plot);parent.appendChild(wrapper);return plot;}
function unavailable(node,entry){node.className='author-status';node.textContent=entry?.error||({running:'Running',pending:'Pending GPU memory / execution',failed:'Fit failed; see run log',unavailable:'Saved result unavailable'}[entry?.status]||'No saved result for this selection');}
function observations(c){const data=c.data, traces=[];for(const [split,label,color] of [['train','Training','#172033'],['test','Test','#aa6622']]){
  if(state.observations!==split&&state.observations!=='both')continue;
  const native=split==='test'&&state.evaluation==='native'&&data.native_x;
  traces.push({x:native?data.native_x:data[split+'_x'],y:native?data.native_y:data[split+'_y'],mode:'markers',marker:{size:split==='train'?5:3,opacity:.55,color},name:label});
}return traces;}
function prediction(node,c,record,native=false){
  const x=c.data.x,p=native?{mean:record.native.mean,lower:record.native.mean.map((m,i)=>m-1.96*record.native.std[i]),upper:record.native.mean.map((m,i)=>m+1.96*record.native.std[i])}:record.prediction;
  const traces=[{x,y:p.lower,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},
    {x,y:p.upper,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(57,108,169,.18)',name:native?'Native ensemble band':'95% predictive interval'},
    {x,y:p.mean,mode:'lines',line:{color:'#396ca9',width:2},name:'Prediction'},
    {x,y:c.data.truth,mode:'lines',line:{color:'#172033',dash:'dash'},name:'True function'},...observations(c)];
  Plotly.newPlot(node,traces,{margin:{t:10,l:50,r:15,b:75},xaxis:{title:'x'},yaxis:{title:'y'},legend:{orientation:'h',y:-.25}},plotConfig);
}
function drawMembers(node,c,record){
  const traces=record.members.map((y,i)=>({x:c.data.x,y,mode:'lines',name:'Network '+(i+1),showlegend:record.members.length<=10,line:{width:1},opacity:.7}));
  traces.push({x:c.data.x,y:c.data.truth,mode:'lines',name:'True function',line:{color:'black',dash:'dash',width:2}},...observations(c));
  Plotly.newPlot(node,traces,{margin:{t:10,l:50,r:15,b:75},xaxis:{title:'x'},yaxis:{title:'Network mean'},legend:{orientation:'h',y:-.25}},plotConfig);
}
function historyPlot(node,history,keys,stage){
  const used=keys.filter(([key])=>Array.isArray(history[key]));
  const signed=state.yscale==='log'&&used.some(([key])=>history[key].some(v=>v<=0));
  const traces=used.map(([key,name])=>({x:history.epoch,y:history[key].map(v=>signed?Math.sign(v)*Math.log10(1+Math.abs(v)):v),mode:'lines',name}));
  Plotly.newPlot(node,traces,{margin:{t:10,l:65,r:15,b:75},xaxis:{title:stage==='1'?'Training progress':'Stage-two step',type:state.xscale},yaxis:{title:signed?'Signed log₁₀(1 + |value|)':'Value',type:state.yscale==='log'&&!signed?'log':'linear'},legend:{orientation:'h',y:-.25}},plotConfig);
}
function figures(c){
  const predictions=clear('predictions'),members=clear('members'),native=clear('native'),histories=clear('histories');
  const downloads=document.getElementById('downloads');downloads.replaceChildren();
  const seen=new Set();
  for(const entry of selected(c)){
    if(state.stage==='both'||entry.stage===state.stage){const node=card(predictions,entry.record?.title||entry.title+' · '+entry.stage.toUpperCase());if(entry.record)prediction(node,c,entry.record);else unavailable(node,entry.entry);}
    if(!entry.record)continue;
    const record=entry.record,key=record.downloads.find(d=>d.label==='config.json')?.path||record.id;
    if(seen.has(key))continue;seen.add(key);
    if(record.members)drawMembers(card(members,entry.title+` · ${record.members.length} networks`),c,record);
    if(record.native)prediction(card(native,entry.title),c,record,true);
    for(const [stage,h] of Object.entries(record.history)){
      const title=entry.title+' · stage '+stage;
      if(stage==='2')historyPlot(card(histories,title+' · BDKN loss'),h,[['objective','Marginal-likelihood loss']],stage);
      else if(record.label.startsWith('author_forde')){
        historyPlot(card(histories,title+' · objective'),h,[['objective','Combined objective'],['likelihood','Gaussian data fit']],stage);
        historyPlot(card(histories,title+' · repulsion loss'),h,[['repulsion','Author repulsion loss']],stage);
        historyPlot(card(histories,title+' · parameter-gradient magnitudes'),h,[['driving','Data + prior gradient'],['repulsive_force','Repulsive gradient']],stage);
      }else if(entry.kind==='author'){
        historyPlot(card(histories,title+' · posterior objective'),h,[['objective','Negative log posterior'],['likelihood','Half MSE']],stage);
        if(h.repulsive_force?.some(v=>v!==0))historyPlot(card(histories,title+' · forces'),h,[['driving','Driving force magnitude'],['repulsive_force','Repulsive force magnitude']],stage);
      }else{
        historyPlot(card(histories,title+' · objective'),h,[['objective','Objective'],['likelihood','Data-fit loss']],stage);
        if(h.repulsion?.some(v=>v!==0))historyPlot(card(histories,title+' · repulsion loss'),h,[['repulsion','Repulsion loss']],stage);
      }
    }
    const block=document.createElement('div');const heading=document.createElement('strong');heading.textContent=entry.title;block.appendChild(heading);
    const config=document.createElement('p');config.className='note';const cfg=record.config;
    config.textContent=record.label.startsWith('author_forde')?`${cfg.preset} · ${cfg.epochs} optimizer steps · LR ${cfg.lr} · ${cfg.n_particles} networks · 2 × ${cfg.hidden_dim} ReLU · JAX/Adam · fixed noise std 1`:(entry.kind==='author'?`${cfg.preset} · ${cfg.steps} optimizer steps · LR ${cfg.lr} · ${cfg.n_particles} networks · 2 × ${cfg.size_hidden} ReLU · fixed noise std 1`:`${cfg.epochs} epochs · LR ${cfg.lr} · ${cfg.n_nets||'—'} networks`);
    block.appendChild(config);const links=document.createElement('div');links.className='downloads';
    for(const download of record.downloads){const a=document.createElement('a');a.href=download.path;a.download='';a.textContent=download.label;links.appendChild(a);}block.appendChild(links);downloads.appendChild(block);
  }
  if(!native.children.length)native.textContent='Native upstream bands will appear when author fits complete.';
}
async function render(){
  const requested=state.dataset;
  const c=await loadCase(requested);if(state.dataset!==requested)return;
  if(!c.data.native_x&&state.evaluation==='native')state.evaluation='noisy';
  for(const id of ids){const group=document.getElementById(id);group.dataset.value=state[id];group.querySelectorAll('button').forEach(button=>{button.setAttribute('aria-pressed',String(button.dataset.value===state[id]));if(id==='evaluation'&&button.dataset.value==='native')button.disabled=!c.data.native_x;});}
  history.replaceState(null,'','#'+new URLSearchParams(state));
  const complete=catalog.jobs.filter(j=>j.status==='complete').length,running=catalog.jobs.filter(j=>j.status==='running').length,failed=catalog.jobs.filter(j=>j.status==='failed').length;
  document.getElementById('progress').textContent=`${complete} / ${catalog.jobs.length} new fits complete · ${running} running · ${failed} failed · ${catalog.status}. Reload to load newly completed results.`;
  document.getElementById('setup-note').textContent=state.preset==='paper'?`D’Angelo paper training recipe: 50 networks, 2 × 50 ReLU, Adam LR 0.01, 15,000 gradient steps, batch size 64, seed 42, raw inputs, fixed likelihood std 1 and N(0,I) prior. Existing datasets are unchanged. FoRDE and BDKN are extensions. Original comparison: ${state['original-setting']} LR.`:state.preset==='upstream'?`D’Angelo README setup for all author-method columns: 100 networks, 2 × 10 ReLU, LR 0.01 and 5,000 optimizer steps, raw inputs. The FoRDE combination keeps the regression setup and replaces only repulsion. Original comparison: ${state['original-setting']} LR.`:`D’Angelo setup with our ${state.preset} settings: 5 networks, 2 × 50 ReLU, standardized inputs, LR ${state.preset==='low'?'0.001; 2,000':'0.1; 40'} training passes converted to optimizer steps. Author FoRDE uses the same prior, initialization, batch stream and Adam settings, with the released FoRDE repulsion. Original comparison: ${state['original-setting']} LR.`;
  const formula=document.getElementById('data-formula');formula.replaceChildren();if(window.katex)katex.render('f(x)='+c.metadata.formula,formula,{throwOnError:false});
  tables();figures(c);
}
for(const id of ids)document.getElementById(id).addEventListener('click',event=>{const button=event.target.closest('button[data-value]');if(!button||button.disabled)return;state[id]=button.dataset.value;if(id==='preset'&&['low','high'].includes(state.preset))state['original-setting']=state.preset;render();});
if(window.katex)document.querySelectorAll('[data-tex]').forEach(node=>katex.render(node.dataset.tex,node,{displayMode:node.tagName==='DIV',throwOnError:false}));
render();
