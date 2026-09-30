const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('frontend/frontend_dashboard.html', 'utf8');
// Execute the actual production handlers with a fake renderer and async decoder.
function functionSource(name) {
  const marker = new RegExp(`    (?:async )?function ${name}\\(`);
  const start = html.search(marker);
  assert.ok(start >= 0);
  const end = html.indexOf('\n    }', start) + 6;
  return html.slice(start, end);
}
function harness() {
  const state = {robotId:'dog', lastLidarAt:0, lidarStats:{}, netStats:{},
    liveLidar: {generation:'', retiredGenerations:new Set(), decodeGeneration:0,
      voxels:new Map(), path:[], voxelSize:0.08, maxPoints:100, pose:{x:0,y:0,yaw:0}, viewer:{}},
    lidarRec:{playing:false, recording:false, frames:[]}, mesh:{indexCount:100, metadata:{},gl:null}};
  const decoded = new Map();
  const scope = {state, els:{liveLidarStats:{},meshStatus:{}},
    decodeLidarCloud: header => decoded.get(header.id),
    stopReplay: () => { state.lidarRec.playing=false; }, updateRecorderUi(){}, requestLiveRender(){},
    resetLiveView(){}, requestMeshRender(){}, setBadge(){}, addEvent(){}, recordLidarFrame(){},
    parseNum: (value, fallback=0) => Number.isFinite(Number(value)) ? Number(value) : fallback,
    Date, Math};
  vm.createContext(scope);
  for (const name of ['clearLiveLidar','useLidarGeneration','handleLidarFrame','applyLiveLidar'])
    vm.runInContext(functionSource(name), scope);
  return {state, decoded, ...scope};
}
const cloud = x => ({points:new Float32Array([x,0,0]), colors:null});

test('late decode from prior mission cannot insert old-room points', async () => {
  const h = harness();
  let finishOld;
  h.decoded.set('old', new Promise(resolve => {finishOld=resolve;}));
  const pending = h.handleLidarFrame({id:'old',map_generation:'mission-a',mode:'keyframe'}, new Uint8Array());
  h.useLidarGeneration('mission-b');
  h.decoded.set('new', Promise.resolve(cloud(50)));
  await h.handleLidarFrame({id:'new',map_generation:'mission-b',mode:'keyframe'}, new Uint8Array());
  finishOld(cloud(1));
  await pending;
  assert.equal(h.state.liveLidar.voxels.size, 1);
  assert.equal([...h.state.liveLidar.voxels.values()][0][0], 50);
  assert.equal(h.useLidarGeneration('mission-a'), false);
});

test('keyframe before reset notification does not clear fresh points twice', async () => {
  const h = harness();
  h.decoded.set('new', Promise.resolve(cloud(50)));
  await h.handleLidarFrame({id:'new',map_generation:'mission-b',mode:'keyframe'}, new Uint8Array());
  assert.equal(h.useLidarGeneration('mission-b'), true);
  assert.equal(h.state.liveLidar.voxels.size, 1);
});

test('mission generation clears path, mesh and replay while preserving exportable recording', () => {
  const h = harness();
  h.state.liveLidar.path = [[3,4]];
  h.state.liveLidar.voxels.set('old', [1,2,3]);
  h.state.lidarRec.playing = h.state.lidarRec.recording = true;
  h.state.lidarRec.frames = ['saved-local-frame'];
  h.useLidarGeneration('new-mission');
  assert.equal(h.state.liveLidar.voxels.size,0);
  assert.equal(h.state.liveLidar.path.length,0);
  assert.equal(h.state.mesh.indexCount,0);
  assert.equal(h.state.lidarRec.playing,false);
  assert.equal(h.state.lidarRec.recording,false);
  assert.deepEqual(h.state.lidarRec.frames,['saved-local-frame']);
});
