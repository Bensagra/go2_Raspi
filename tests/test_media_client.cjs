const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('frontend/go2-media-controls.js', 'utf8');

function harness() {
  const events = new EventTarget();
  const document = new EventTarget();
  let processorClass;
  const packets = [];
  const track = Object.assign(new EventTarget(), { stop() { this.stopped = true; } });
  const stream = { getTracks: () => [track] };
  const socketList = [];
  const contexts = [];
  const nodes = [];
  const audioBase = () => ({ connect(next) { return next; }, disconnect() {} });
  class Context {
    constructor() { contexts.push(this); this.destination = {}; this.audioWorklet = {
      addModule: async url => {
        const code = await (await fetch(url)).text();
        vm.runInNewContext(code, { sampleRate: 48000, AudioWorkletProcessor: class {
          constructor() { this.port = { postMessage: buffer => packets.push(buffer) }; }
        }, registerProcessor: (name, Class) => { processorClass = Class; } });
      },
    }; }
    resume() { return Promise.resolve(); }
    close() { this.closed = true; return Promise.resolve(); }
    createMediaStreamSource() { return audioBase(); }
    createGain() { return { ...audioBase(), gain: { value: 1 } }; }
  }
  class WorkletNode {
    constructor() { Object.assign(this, audioBase(), { port: {} }); nodes.push(this); }
  }
  class Socket {
    static OPEN = 1;
    constructor(url) { this.url = String(url); this.readyState = 1; this.bufferedAmount = 0;
      this.sent = []; socketList.push(this);
      setTimeout(() => this.onmessage?.({data:'{"status":"ready"}'}), 0);
    }
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; this.onclose?.({code:1000}); }
  }
  const window = events;
  Object.assign(window, { isSecureContext: true, AudioWorkletNode: WorkletNode });
  const navigator = { mediaDevices: { getUserMedia: async () => stream } };
  const network = { fetch: () => { throw new Error('Unexpected HTTP request'); } };
  vm.runInNewContext(source, { window, document, navigator, AudioContext: Context,
    AudioWorkletNode: WorkletNode, WebSocket: Socket, URL, Blob, setTimeout, clearTimeout,
    setInterval, clearInterval, console, AbortController,
    crypto: { getRandomValues: value => require('node:crypto').webcrypto.getRandomValues(value) },
    fetch: (...args) => network.fetch(...args) });
  return { api: window.Go2Media, window, document, navigator, track, stream, socketList,
    contexts, nodes, packets, network, processor: () => new processorClass() };
}
const options = { apiBase: 'https://host.example/prefix/', token: 'secret', robotId: 'dog' };

test('PTT uses prefix, streams only when ready, closes mic on blur and disposes resources', async () => {
  const h = harness();
  const statuses = [];
  const client = new h.api.TalkClient(s => statuses.push(s));
  await client.start(options);
  assert.equal(new URL(h.socketList[0].url).pathname, '/prefix/ws/talk/dog');
  assert.equal(statuses.at(-1).status, 'talking');
  const pcm = new ArrayBuffer(1280);
  h.nodes[0].port.onmessage({data:pcm});
  assert.equal(h.socketList[0].sent[0], pcm);
  h.window.dispatchEvent(new Event('blur'));
  assert.equal(client.active, false);
  assert.equal(h.track.stopped, true);
  assert.equal(h.contexts[0].closed, true);
  assert.equal(h.socketList[0].sent.at(-1), '{"type":"stop"}');
  client.dispose();
});

test('releasing while microphone permission is pending cannot reopen the microphone', async () => {
  const h = harness();
  let grant;
  h.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { grant = resolve; });
  const client = new h.api.TalkClient();
  const start = client.start(options);
  await new Promise(resolve => setImmediate(resolve));
  client.stop();
  grant(h.stream);
  await start;
  assert.equal(h.track.stopped, true);
  assert.equal(h.socketList.length, 0);
  assert.equal(client.active, false);
  client.dispose();
});

test('worklet resamples 48 kHz into 16 kHz PCM16 LE in continuous 40 ms packets', async () => {
  const h = harness();
  const client = new h.api.TalkClient();
  await client.start(options);
  const processor = h.processor();
  for (let i = 0; i < 375; i++) processor.process([[new Float32Array(128).fill(0.5)]]);
  assert.equal(h.packets.length, 25); // 1 second => 25 packets, 640 samples each
  for (const buffer of h.packets) {
    assert.equal(buffer.byteLength, 1280);
    assert.equal(new DataView(buffer).getInt16(0, true), 16384);
  }
  client.dispose();
});

test('slow network stops sending instead of accumulating delayed speech', async () => {
  const h = harness();
  const statuses = [];
  const client = new h.api.TalkClient(s => statuses.push(s));
  await client.start(options);
  h.socketList[0].bufferedAmount = 7000;
  h.nodes[0].port.onmessage({data:new ArrayBuffer(1280)});
  assert.equal(client.active, false);
  assert.equal(h.track.stopped, true);
  assert.match(statuses.at(-1).error, /Red lenta/);
  client.dispose();
});

test('snapshot rejects missing/stale frames and copies pixels before asynchronous encoding', async () => {
  const h = harness();
  await assert.rejects(h.api.captureCanvas({}, {robotId:'dog', lastFrameAt:0}), /cuadro reciente/);
  await assert.rejects(h.api.captureCanvas({}, {robotId:'dog', lastFrameAt:Date.now()-5000}), /cuadro reciente/);
  let copied = false;
  h.document.createElement = () => ({getContext: () => ({drawImage: () => {copied = true;}}),
    toBlob: done => {assert.equal(copied, true); done(new Blob(['png']));}});
  const shot = await h.api.captureCanvas({width:640, height:360}, {robotId:'dog', lastFrameAt:Date.now()});
  assert.equal(shot.width, 640);
  assert.equal(shot.height, 360);
  assert.match(shot.filename, /^dog_.*\.png$/);
});

test('light works without randomUUID and waits past accepted to the executed ACK', async () => {
  const h = harness();
  let commandId;
  let polls = 0;
  h.network.fetch = async (url, options) => {
    assert.equal(options.headers.Authorization, 'Bearer secret');
    if (options.method === 'POST') {
      const command = JSON.parse(options.body);
      assert.equal(command.type, 'set_flashlight');
      assert.equal(command.payload.brightness, 10);
      commandId = command.command_id;
      return {ok:true, json: async () => ({status:'queued', command_id:commandId})};
    }
    assert.equal(new URL(url).pathname, '/prefix/api/robots/dog/replay');
    return {ok:true, json: async () => ({acks:[{command_id:commandId,
      status: ++polls === 1 ? 'accepted' : 'executed', result:{brightness:10, enabled:true}}]})};
  };
  const result = await h.api.confirmedCommand(options, 'set_flashlight', {brightness:10});
  assert.equal(result.enabled, true);
  assert.equal(polls, 2);
  assert.match(commandId, /^web-[a-f0-9]{32}$/);
});

test('light surfaces robot rejection instead of treating queued as success', async () => {
  const h = harness();
  let commandId;
  h.network.fetch = async (url, options) => {
    if (options.method === 'POST') {
      commandId = JSON.parse(options.body).command_id;
      return {ok:true, json: async () => ({status:'queued'})};
    }
    return {ok:true, json: async () => ({acks:[{command_id:commandId, status:'error', reason:'VUI rejected'}]})};
  };
  await assert.rejects(h.api.confirmedCommand(options, 'set_flashlight', {brightness:10}), /VUI rejected/);
});
