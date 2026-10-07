const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('frontend/frontend_dashboard.html', 'utf8');
function source(name) {
  const start = html.search(new RegExp(`    (?:async )?function ${name}\\(`));
  assert.ok(start >= 0);
  return html.slice(start, html.indexOf('\n    }', start) + 6);
}
function harness() {
  const drawn = [];
  const state = {connected:true, lastArducamAt:0, arducamFrameTimes:[], expandedStream:'',
    mediaRender:{arducam:{busy:false, pending:null, bitmap:null, generation:0}, video:{bitmap:{old:true}}}};
  const scope = {state, Blob, Date, els:{arducamFrame:{setAttribute(){}}, arducamEmpty:{},
    arducamStatus:{}, videoFrame:{}, expandedMediaCanvas:{}},
    drawBitmap:(canvas, image) => drawn.push({canvas, image}),
    formatImageMime:() => 'image/jpeg', addEvent(){},
    setBadge:(el, text) => {el.textContent=text;},
    closeMediaModal:() => {state.expandedStream='';},
    decodeMediaBlob:async () => ({close(){}})};
  vm.createContext(scope);
  for (const name of ['streamCanvas','processMediaFrames','renderMediaImageBytes','updateArducamStatus'])
    vm.runInContext(source(name), scope);
  return {...scope, drawn, scope};
}

test('dashboard inline scripts parse', () => {
  for (const script of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
});

test('Arducam decoding uses its own canvas and drops replaced pending frames', async () => {
  const h = harness();
  let finish;
  const first = {close(){this.closed=true;}};
  const final = {close(){}};
  let count = 0;
  h.scope.decodeMediaBlob = () => ++count === 1 ? new Promise(resolve => {finish=resolve;}) : Promise.resolve(final);
  h.renderMediaImageBytes(new Uint8Array([1]), 'jpg', 'arducam');
  h.renderMediaImageBytes(new Uint8Array([2]), 'jpg', 'arducam');
  h.renderMediaImageBytes(new Uint8Array([3]), 'jpg', 'arducam');
  finish(first);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(count, 2);
  assert.equal(h.drawn.length, 2);
  assert.equal(h.drawn[0].canvas, h.els.arducamFrame);
  assert.equal(h.state.mediaRender.arducam.bitmap, final);
  assert.equal(first.closed, true);
  assert.equal(h.state.mediaRender.video.bitmap.old, true);
  assert.equal(h.els.arducamEmpty.hidden, true);
});

test('late decode after disconnect cannot restore an old Arducam frame', async () => {
  const h = harness();
  let finish;
  const image = {close(){this.closed=true;}};
  h.scope.decodeMediaBlob = () => new Promise(resolve => {finish=resolve;});
  h.renderMediaImageBytes(new Uint8Array([1]), 'jpg', 'arducam');
  h.state.mediaRender.arducam.generation += 1;
  finish(image);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(image.closed, true);
  assert.equal(h.drawn.length, 0);
});

test('missing or interrupted camera never appears live or remains expanded', () => {
  const h = harness();
  h.updateArducamStatus();
  assert.equal(h.els.arducamStatus.textContent, 'Esperando cámara');
  h.state.lastArducamAt = Date.now() - 4000;
  h.state.expandedStream = 'arducam';
  h.updateArducamStatus();
  assert.equal(h.els.arducamStatus.textContent, 'Señal interrumpida');
  assert.equal(h.els.arducamEmpty.hidden, false);
  assert.equal(h.state.expandedStream, '');
  h.state.connected = false;
  h.updateArducamStatus();
  assert.equal(h.els.arducamStatus.textContent, 'Sin conexión');
});
