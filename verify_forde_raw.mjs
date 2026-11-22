// Real-browser offline controls/plots check. Node 22+, Chrome debugging port.
// node verify_forde_raw.mjs /absolute/path/to/forde_gradients.html 9347 [screenshot.png]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const page = process.argv[2], port = Number(process.argv[3] || 9347);
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find(t => t.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, {once: true}));
let next = 0;
const pending = new Map(), exceptions = [], urls = [];
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const item = pending.get(message.id);
    if (item) { pending.delete(message.id); message.error ? item.reject(message.error) : item.resolve(message.result); }
  }
  if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
  if (message.method === 'Network.requestWillBeSent') urls.push(message.params.request.url);
});
function send(method, params = {}) {
  return new Promise((resolve, reject) => { const id = ++next; pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params})); });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression) {
  for (let i = 0; i < 150; i++) {
    if (await evaluate(expression)) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + expression);
}
await send('Runtime.enable'); await send('Network.enable'); await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', {width: 1480, height: 1100, deviceScaleFactor: 1, mobile: false});
await send('Page.navigate', {url: pathToFileURL(page).href});
await waitFor('document.querySelectorAll(".js-plotly-plot").length >= 5');
assert.equal(await evaluate('document.querySelectorAll("select,details").length'), 0);
assert.equal(await evaluate('document.querySelectorAll(".katex-error").length'), 0);
assert.equal(await evaluate('document.querySelectorAll("table[data-metric]").length'), 5);
assert.equal(await evaluate('document.querySelectorAll("table[data-metric] thead tr:last-child th").length'), 30);
const datasets = await evaluate('[...document.querySelectorAll("#dataset button")].map(b=>b.dataset.value)');
for (const [id, values] of Object.entries({dataset: datasets, setting: ['high', 'low'], location: ['training', 'random', 'combined'], stage: ['de', 'bdkn'], observations: ['test', 'both', 'none', 'train'], weight: ['0.01', '0.1', '1', '10', '0.001'], xscale: ['linear', 'log'], yscale: ['linear', 'log']})) {
  for (const value of values) {
    const exists = await evaluate(`!!document.querySelector('#${id} button[data-value="${value}"]')`);
    if (!exists) continue;
    await evaluate(`document.querySelector('#${id} button[data-value="${value}"]').click()`);
    await waitFor(`document.getElementById('${id}').dataset.value === '${value}' && document.querySelectorAll('.js-plotly-plot').length >= 5`);
  }
  console.log('PASS: ' + id + ' controls');
}
await evaluate('document.getElementById("show-settings").click()');
assert.equal(await evaluate('document.getElementById("resolved-settings").hidden'), false);
assert.ok(await evaluate('document.querySelectorAll("#resolved-settings a[download]").length') > 0);
assert.ok(await evaluate('[...document.querySelectorAll("#tables td[style]")].every(n => n.textContent !== "NaN")'));
assert.equal(urls.filter(url => /^https?:/.test(url)).length, 0, 'Report must not request network assets');
assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
await evaluate('document.getElementById("show-settings").click()');
await send('Page.reload');
await waitFor('document.querySelectorAll(".js-plotly-plot").length >= 5');
assert.equal(await evaluate('document.getElementById("location").dataset.value'), 'combined');
assert.equal(await evaluate('document.getElementById("stage").dataset.value'), 'bdkn');
console.log('PASS: raw/normalized paired tables, offline plots, math, downloads, and selection persistence');
if (process.argv[4]) {
  await send('Emulation.setDeviceMetricsOverride', {width: 1480, height: 1100, deviceScaleFactor: 1, mobile: false});
  const screenshot = await send('Page.captureScreenshot', {format: 'png'});
  fs.writeFileSync(process.argv[4], Buffer.from(screenshot.data, 'base64'));
}
ws.close();
