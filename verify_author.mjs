// Offline browser verification. Existing Node 22+, Chrome debugging port 9361.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const page=process.argv[2],port=Number(process.argv[3]||9361);
const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target=targets.find(t=>t.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>ws.onopen=resolve);
let next=0;const pending=new Map(),exceptions=[],urls=[];
ws.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);message.error?p.reject(new Error(JSON.stringify(message.error))):p.resolve(message.result);}if(message.method==='Runtime.exceptionThrown')exceptions.push(message.params);if(message.method==='Network.requestWillBeSent')urls.push(message.params.request.url);};
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function waitFor(expression){for(let i=0;i<600;i++){if(await evaluate(expression))return;await new Promise(resolve=>setTimeout(resolve,100));}throw new Error('Timeout: '+expression);}
await send('Runtime.enable');await send('Network.enable');await send('Page.enable');
await send('Page.navigate',{url:pathToFileURL(page).href+'#dataset=steps&preset=low&original-setting=low'});
await waitFor('document.querySelectorAll(".js-plotly-plot").length>=10');
assert.equal(await evaluate('document.querySelectorAll("select,details").length'),0);
assert.equal(await evaluate('document.querySelectorAll(".katex-error").length'),0);
assert.equal(await evaluate('document.querySelectorAll("table[data-metric]").length'),3);
assert.equal(await evaluate('document.querySelectorAll("table[data-metric=rmse] tbody tr").length'),11);
for(const [id,values] of Object.entries({preset:['upstream','high','low'],stage:['de','bdkn','both'],observations:['test','both','none','train'],weight:['0.01','10','0.001'],location:['random','combined','training'],xscale:['linear','log'],yscale:['linear','log']})){
  for(const value of values){await evaluate(`document.querySelector('#${id} button[data-value="${value}"]').click()`);await waitFor(`document.getElementById('${id}').dataset.value==='${value}' && document.querySelectorAll('.js-plotly-plot').length>=10`);}
  console.log('PASS',id);
}
await evaluate('document.querySelector("#dataset button[data-value=author-xsine]").click()');
await waitFor('document.getElementById("dataset").dataset.value==="author-xsine"');
await evaluate('document.querySelector("#evaluation button[data-value=native]").click()');
await waitFor('document.getElementById("evaluation").dataset.value==="native"');
assert.equal(await evaluate('document.querySelector("#evaluation button[data-value=native]").disabled'),false);
await evaluate('document.querySelector("#dataset button[data-value=he]").click()');
await waitFor('document.getElementById("dataset").dataset.value==="he"');
assert.equal(await evaluate('document.getElementById("evaluation").dataset.value'),'noisy');
assert.equal(await evaluate('document.querySelector("#evaluation button[data-value=native]").disabled'),true);
assert.ok(await evaluate('document.querySelectorAll("#downloads a[download]").length')>0);
assert.equal(await evaluate('[...document.querySelectorAll("td")].some(n=>n.textContent.includes("NaN"))'),false);
assert.equal(urls.filter(url=>/^https?:/.test(url)).length,0,'Report must work without external network assets');
assert.equal(exceptions.length,0,JSON.stringify(exceptions));
if(process.argv[4]){await send('Emulation.setDeviceMetricsOverride',{width:1480,height:1100,deviceScaleFactor:1,mobile:false});const image=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(process.argv[4],Buffer.from(image.data,'base64'));}
console.log('PASS offline tables, formulas, controls, plots, downloads and test selection');
ws.close();
