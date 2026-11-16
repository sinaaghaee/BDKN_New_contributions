function initializeChoice(node){
 node.value=node.dataset.value;
 node.addEventListener('click',event=>{
  const button=event.target.closest('button[data-value]');
  if(!button || button.disabled || !node.contains(button) || button.dataset.value===node.value)return;
  node.value=button.dataset.value;node.dataset.value=node.value;
  node.querySelectorAll('button[data-value]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  node.dispatchEvent(new Event('change'));
 });
}
document.querySelectorAll('.choices').forEach(initializeChoice);
function reportAsset(file){return globalThis.REPORT_ASSETS?.[file] || file;}

function sortMetrics(column){const body=document.querySelector('#metrics tbody');const rows=[...body.rows];const previous=body.dataset.sort;const direction=previous===column+'-asc'?-1:1;rows.sort((a,b)=>direction*(Number(a.cells[column].dataset.value)-Number(b.cells[column].dataset.value)));rows.forEach(row=>body.appendChild(row));body.dataset.sort=column+(direction===1?'-asc':'-desc');document.getElementById('sort-status').textContent='Sorted by '+({1:'RMSE',2:'NLL',3:'training seconds'}[column])+(direction===1?' ascending':' descending');}
const filter=document.getElementById('model-filter');if(filter)filter.addEventListener('change',()=>{document.querySelectorAll('[data-model]').forEach(card=>card.hidden=filter.value!=='all'&&card.dataset.model!==filter.value);});
const lossX=document.getElementById('loss-xscale'),lossY=document.getElementById('loss-yscale');
function updateLossScales(){const x=lossX.value,y=lossY.value;const suffix=x==='log'?(y==='log'?'_xylog':'_xlog'):(y==='log'?'_ylog':'');document.querySelectorAll('[data-loss-base]').forEach(img=>{const file=reportAsset('assets/'+img.dataset.lossBase+suffix+'.png');img.src=file;img.parentElement.href=file;const yLabel=y==='linear'?'linear':(img.dataset.logY==='symlog'?'signed log (symlog; zeros/negatives preserved)':'log');img.closest('figure').querySelector('[data-scale-note]').textContent='X: '+x+' · Y: '+yLabel;});}
if(lossX&&lossY){lossX.addEventListener('change',updateLossScales);lossY.addEventListener('change',updateLossScales);updateLossScales();}

const observations=document.getElementById('observation-mode');
function updateObservations(){const mode=observations.value;const suffix=mode==='train'?'':'_'+mode;document.querySelectorAll('[data-observation-base]').forEach(img=>{const file=reportAsset('assets/'+img.dataset.observationBase+suffix+'.png');img.src=file;img.parentElement.href=file;});document.getElementById('observation-status').textContent=mode==='none'?'Observations hidden':'Showing '+({train:'training',test:'test',both:'training and test'}[mode])+' observations';}
if(observations)observations.addEventListener('change',updateObservations);


const aggregationMethod=document.getElementById('aggregation-method');
if(aggregationMethod){
 function filterAggregation(){document.querySelectorAll('[data-aggregation-method]').forEach(card=>card.hidden=aggregationMethod.value!=='all'&&card.dataset.aggregationMethod!==aggregationMethod.value);}
 const selected=location.hash.match(/^#aggregation-(bdkn|frde|wrde|forde|all)$/)?.[1]||'bdkn';
 aggregationMethod.value=selected;aggregationMethod.dataset.value=selected;
 aggregationMethod.querySelectorAll('button[data-value]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.value===selected)));
 filterAggregation();aggregationMethod.addEventListener('change',()=>{filterAggregation();location.hash='#aggregation-'+aggregationMethod.value;});
}

const lossMethod=document.getElementById('loss-method');
if(lossMethod)lossMethod.addEventListener('change',()=>document.querySelectorAll('[data-loss-method]').forEach(figure=>figure.hidden=lossMethod.value!=='all'&&figure.dataset.lossMethod!==lossMethod.value));

if(globalThis.katex)document.querySelectorAll('[data-tex]').forEach(element=>katex.render(element.dataset.tex,element,{displayMode:element.tagName==='DIV',throwOnError:false}));

const catalogNode=document.getElementById('experiment-catalog');
if(catalogNode){
 const state=JSON.parse(catalogNode.textContent),entries=state.experiments;
 entries.forEach(entry=>entry.repulsion_location??='training');
 const datasetSelect=document.getElementById('dataset-select'),lrSelect=document.getElementById('lr-select'),strengthSelect=document.getElementById('strength-select'),locationSelect=document.getElementById('location-select');
 function rateLabel(value){return value==='0.1'?'High · LR 0.1 · 40 epochs':value==='0.001'?'Low · NN LR 0.001':value;}
 function fill(select,values,preferred){select.replaceChildren(...values.map(value=>{const option=document.createElement('button');option.type='button';option.dataset.value=value;option.textContent=select===lrSelect?rateLabel(value):select===locationSelect?({training:'Training',random:'Random',combined:'Training + random'}[value]||value):value;return option}));select.value=values.includes(preferred)?preferred:values[0];select.dataset.value=select.value;select.querySelectorAll('button[data-value]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.value===select.value)));}
 function switchExperiment(changed){
  const overview=state.page==='index.html',active=entries.find(entry=>entry.id===state.active);
  const dataset=overview?active.dataset:datasetSelect.value;
  const candidates=overview?entries:entries.filter(entry=>entry.dataset===dataset);
  if(changed==='dataset')fill(lrSelect,[...new Set(candidates.map(entry=>entry.learning_rate))].sort((a,b)=>Number(a)-Number(b)),lrSelect.value);
  const atRate=candidates.filter(entry=>entry.learning_rate===lrSelect.value);
  if(locationSelect)fill(locationSelect,['training','random','combined'].filter(value=>atRate.some(entry=>entry.repulsion_location===value)),locationSelect.value);
  const available=atRate.filter(entry=>entry.repulsion_location===(locationSelect?.value||'training'));
  if(changed!=='strength')fill(strengthSelect,[...new Set(available.map(entry=>entry.repulsion))].sort((a,b)=>Number(a)-Number(b)),strengthSelect.value);
  const matching=available.filter(entry=>entry.repulsion===strengthSelect.value);
  const target=matching.find(entry=>entry.dataset===dataset)||matching[0];
  if(target)location.assign(state.root+target.directory+state.page+location.hash);
 }
 datasetSelect.addEventListener('change',()=>switchExperiment('dataset'));
 lrSelect.addEventListener('change',()=>switchExperiment('lr'));
 strengthSelect.addEventListener('change',()=>switchExperiment('strength'));
 if(locationSelect)locationSelect.addEventListener('change',()=>switchExperiment('location'));
}
