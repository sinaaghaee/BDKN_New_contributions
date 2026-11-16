// Execute the generated report controls against real page/asset fixtures.
// Usage: node verify_report_controls.mjs /tmp/report-fixture.json
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const require=createRequire(import.meta.url);
const katex=require('./website/vendor/katex/katex.min.js');
for(const expression of fixture.math)katex.renderToString(expression,{throwOnError:true});
console.log(`PASS: ${fixture.math.length} distinct LaTeX expressions`);
function button(value,disabled=false){return {dataset:{value},disabled,setAttribute(k,v){this[k]=v},closest(){return this}}}
function group(data){return {value:data.value,dataset:{value:data.value},buttons:data.buttons.map(a=>button(a['data-value'],'disabled' in a)),events:{},addEventListener(name,fn){this.events[name]=fn},querySelectorAll(){return this.buttons},contains(b){return this.buttons.includes(b)},replaceChildren(...buttons){this.buttons=buttons},dispatchEvent(event){this.events[event.type]?.(event)},click(value){const b=this.buttons.find(b=>b.dataset.value===value);assert.ok(b,'Missing button '+value);this.events.click({target:b})}}}
function environment(page,mutate){
 const state=structuredClone(fixture.states[page]);if(mutate)mutate(state);
 const elements=Object.fromEntries(Object.entries(fixture.groups[page]).map(([key,value])=>[key,group(value)]));
 elements['experiment-catalog']={textContent:JSON.stringify(state)};
 elements['observation-status']={textContent:''};
 const images=fixture.images[page].map(a=>({dataset:{lossBase:a['data-loss-base'],logY:a['data-log-y'],observationBase:a['data-observation-base']},src:a.src,parentElement:{href:a.src},note:{textContent:''},closest(){return {querySelector:()=>this.note}}}));
 const cards=state.experiments.length?['bdkn','frde','forde','wrde','gp'].map(model=>({dataset:{model,lossMethod:model},hidden:false})):[];
 const aggregationCards=['bdkn','frde','wrde','forde'].map(method=>({dataset:{aggregationMethod:method},hidden:false}));
 const location={hash:'#frde',assigned:null,assign(url){this.assigned=url}};
 const document={getElementById:id=>elements[id]??null,querySelectorAll(selector){if(selector==='.choices')return Object.values(elements).filter(e=>e.dataset);if(selector==='[data-loss-base]')return images.filter(i=>i.dataset.lossBase);if(selector==='[data-observation-base]')return images.filter(i=>i.dataset.observationBase);if(selector==='[data-aggregation-method]')return aggregationCards;if(selector==='[data-model]'||selector==='[data-loss-method]')return cards;return []},createElement(){return button('')}};
 const context={document,location,Event:class{constructor(type){this.type=type}}};vm.createContext(context);
 const assetMap=path.join(fixture.root,'asset-map.js');if(fs.existsSync(assetMap))vm.runInContext(fs.readFileSync(assetMap,'utf8'),context);
 vm.runInContext(fs.readFileSync(path.join(fixture.root,'report.js'),'utf8'),context);
 return {elements,images,state,location,context,cards,aggregationCards};
}
for(const page of ['losses.html','repulsion.html','locations.html']){
 const env=environment(page);const loss=env.images.filter(i=>i.dataset.lossBase);assert.ok(loss.length);
 for(const [x,y,suffix] of [['linear','linear',''],['log','linear','_xlog'],['linear','log','_ylog'],['log','log','_xylog']]){
  env.elements['loss-xscale'].click(x);env.elements['loss-yscale'].click(y);
  for(const img of loss){assert.equal(img.src,env.context.reportAsset('assets/'+img.dataset.lossBase+suffix+'.png'));assert.equal(img.parentElement.href,img.src);assert.ok(fs.existsSync(path.resolve(fixture.root,img.src)));}
 }
 console.log('PASS: all loss-axis button combinations on '+page);
}
for(const page of ['figures.html','repulsion.html','aggregation.html','locations.html']){
 const env=environment(page);const obs=env.images.filter(i=>i.dataset.observationBase);assert.ok(obs.length);
 for(const mode of ['test','both','none','train']){
  env.elements['observation-mode'].click(mode);
  for(const img of obs){assert.equal(img.src,env.context.reportAsset('assets/'+img.dataset.observationBase+(mode==='train'?'':'_'+mode)+'.png'));assert.equal(img.parentElement.href,img.src);assert.ok(fs.existsSync(path.resolve(fixture.root,img.src)));}
 }
 console.log('PASS: all observation buttons on '+page);
}
for(const page of ['index.html','figures.html','losses.html','aggregation.html','locations.html']){
 const env=environment(page);const active=env.state.experiments.find(e=>e.id===env.state.active);
 for(const weight of ['0.01','0.1','1','10','0.001']){
  if(!env.elements['strength-select'].buttons.some(b=>b.dataset.value===weight))continue;
  env.elements['strength-select'].click(weight);
  const target=env.state.experiments.find(e=>e.dataset===active.dataset&&e.learning_rate===active.learning_rate&&e.repulsion===weight);
  if(env.location.assigned===null)continue; // Clicking an already selected button is inert.
  assert.equal(env.location.assigned,env.state.root+target.directory+page+'#frde');
  assert.ok(fs.existsSync(path.resolve(fixture.root,env.state.root+target.directory+page)));
 }
 if(page==='index.html'){
  assert.equal(env.elements['dataset-select'].value,'all');assert.ok(env.elements['dataset-select'].buttons[0].disabled);
 }else{
  for(const dataset of env.elements['dataset-select'].buttons.map(button=>button.dataset.value)){
   env.elements['dataset-select'].click(dataset);
   const target=env.state.experiments.find(e=>e.dataset===dataset&&e.learning_rate===env.elements['lr-select'].value&&e.repulsion===env.elements['strength-select'].value);
   if(env.location.assigned===null)continue;
   assert.ok(target);assert.equal(env.location.assigned,env.state.root+target.directory+page+'#frde');
   assert.ok(fs.existsSync(path.resolve(fixture.root,env.state.root+target.directory+page)));
  }
 }
 console.log('PASS: real dataset/weight navigation on '+page);
}
{
 const env=environment('figures.html');env.elements['model-filter'].click('gp');assert.ok(env.cards.every(card=>card.hidden===(card.dataset.model!=='gp')));env.elements['model-filter'].click('all');assert.ok(env.cards.every(card=>!card.hidden));
 const loss=environment('losses.html');loss.elements['loss-method'].click('bdkn');assert.ok(loss.cards.every(card=>card.hidden===(card.dataset.lossMethod!=='bdkn')));
 console.log('PASS: prediction and loss method filters');
}

{
 const env=environment('index.html',state=>state.experiments.push({id:'future-lr',dataset:'future',learning_rate:'0.02',repulsion:'0.1',directory:'experiments/future-lr/'}));
 env.elements['lr-select'].buttons.push(button('0.02'));env.elements['lr-select'].click('0.02');
 assert.equal(env.location.assigned,env.state.root+'experiments/future-lr/index.html#frde');
 assert.equal(env.elements['strength-select'].value,'0.1');
 console.log('PASS: overview LR switching when the active dataset has no run at that LR');
}

for(const page of ['index.html','figures.html','losses.html','aggregation.html','locations.html']){
 const env=environment(page);
 for(const rate of env.elements['lr-select'].buttons.map(button=>button.dataset.value).reverse()){
  env.elements['lr-select'].click(rate);
  if(env.location.assigned===null)continue;
  const activeDataset=env.state.experiments.find(e=>e.id===env.state.active).dataset;
  const target=env.state.experiments.find(e=>e.dataset===activeDataset&&e.learning_rate===rate&&e.repulsion===env.elements['strength-select'].value);
  assert.ok(target);assert.equal(env.location.assigned,env.state.root+target.directory+page+'#frde');
  assert.ok(fs.existsSync(path.resolve(fixture.root,env.state.root+target.directory+page)));
 }
 console.log('PASS: real low/high training-setting switches on '+page);
}

{
 const env=environment('aggregation.html');
 for(const method of ['frde','wrde','forde','bdkn','all']){
  env.elements['aggregation-method'].click(method);
  assert.ok(env.aggregationCards.every(card=>card.hidden===(method!=='all'&&card.dataset.aggregationMethod!==method)));
 }
 console.log('PASS: aggregation-variant filters and selected variant URL');
}

for(const page of ['index.html','figures.html','losses.html','aggregation.html','repulsion.html','locations.html']){
 const env=environment(page);const active=env.state.experiments.find(e=>e.id===env.state.active);
 for(const pointLocation of env.elements['location-select'].buttons.map(b=>b.dataset.value)){
  env.elements['location-select'].click(pointLocation);
  const target=env.state.experiments.find(e=>e.dataset===active.dataset&&e.learning_rate===env.elements['lr-select'].value&&e.repulsion===env.elements['strength-select'].value&&e.repulsion_location===pointLocation);
  if(env.location.assigned===null)continue;
  assert.ok(target);assert.equal(env.location.assigned,env.state.root+target.directory+page+'#frde');
  assert.ok(fs.existsSync(path.resolve(fixture.root,env.state.root+target.directory+page)));
 }
 console.log('PASS: repulsion-location navigation on '+page);
}

{
 const env=environment('locations.html');
 for(const method of ['frde','forde','wrde','all']){
  env.elements['model-filter'].click(method);
  assert.ok(env.cards.every(card=>card.hidden===(method!=='all'&&card.dataset.model!==method)));
 }
 console.log('PASS: location-comparison method filter');
}
for(const page of ['index.html','figures.html','losses.html','aggregation.html','locations.html']){
 for(const mode of ['random','combined']){
  const env=environment(page);
  if(!env.elements['location-select'].buttons.some(b=>b.dataset.value===mode))continue;
  env.elements['location-select'].click(mode);
  for(const id of ['dataset-select','lr-select','strength-select']){
   const control=env.elements[id];const alternate=control.buttons.find(b=>!b.disabled&&b.dataset.value!==control.value);
   if(!alternate)continue;
   control.click(alternate.dataset.value);
   assert.equal(env.elements['location-select'].value,mode);
   const dataset=page==='index.html'?env.state.experiments.find(e=>e.id===env.state.active).dataset:env.elements['dataset-select'].value;
   const target=env.state.experiments.find(e=>e.dataset===dataset&&e.learning_rate===env.elements['lr-select'].value&&e.repulsion===env.elements['strength-select'].value&&e.repulsion_location===mode);
   assert.ok(target);assert.equal(env.location.assigned,env.state.root+target.directory+page+'#frde');
  }
 }
 console.log('PASS: location preserved across other filters on '+page);
}
