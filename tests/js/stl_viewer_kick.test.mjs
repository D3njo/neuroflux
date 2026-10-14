/**
 * Wayland STL viewer: kick timers and the bootstrap animation frame must not
 * paint a WebGLRenderer after _stlStopView disposes it.
 *
 * rAF is held back (the Wayland "no frames until a click" case) while
 * setTimeout still runs, which is the window that used to last ~1.5s.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const html = fs.readFileSync(path.join(root, 'src/neuroflux/data/neuroflux.html'), 'utf8');
const start = html.indexOf('let _stlRenderer');
const end = html.indexOf('function togStlAutoRot()');
if (start < 0 || end < start) {
  throw new Error('Could not locate the STL viewer script in neuroflux.html');
}

const factory = new Function(
  `${html.slice(start, end)}
   return { init: _initStlScene, stop: _stlStopView };`,
);

const renderers = [];
let timers = [];
let rafQueue = [];
let now = 0;
let timerSeq = 1;
let rafSeq = 1;

function el(id) {
  return {
    id,
    textContent: '',
    clientWidth: 800,
    clientHeight: 600,
    parentElement: id === 'stl-viewer-canvas' ? el.parent : null,
    style: {},
  };
}
el.parent = { id: 'stl-viewer-inner', clientWidth: 800, clientHeight: 600 };

function vec() {
  return { set() { return this; } };
}

function installDom() {
  const elements = {
    'stl-viewer-canvas': el('stl-viewer-canvas'),
    'stl-viewer-info': el('stl-viewer-info'),
  };
  globalThis.document = {
    getElementById(id) {
      return elements[id] || el(id);
    },
  };
  globalThis.window = globalThis;
  globalThis.devicePixelRatio = 1;
  globalThis.addEventListener = () => {};
  globalThis.removeEventListener = () => {};
  globalThis.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  globalThis.THREE = {
    WebGLRenderer: class {
      constructor() {
        this.domElement = {};
        this.disposed = false;
        this.renders = 0;
        renderers.push(this);
      }
      setPixelRatio() {}
      setClearColor() {}
      setSize() {}
      render() {
        if (this.disposed) {
          throw new Error('WebGLRenderer.render called after dispose');
        }
        this.renders += 1;
      }
      dispose() {
        this.disposed = true;
      }
    },
    Scene: class { add() {} },
    PerspectiveCamera: class {
      constructor() { this.position = vec(); }
      updateProjectionMatrix() {}
      lookAt() {}
    },
    AmbientLight: class {},
    DirectionalLight: class { constructor() { this.position = vec(); } },
    MeshPhongMaterial: class {},
    Mesh: class {},
    DoubleSide: 2,
    STLLoader: class {
      parse() {
        return {
          computeVertexNormals() {},
          center() {},
          computeBoundingSphere() {},
          computeBoundingBox() {},
          boundingSphere: { radius: 12 },
          boundingBox: {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 10, y: 20, z: 30 },
          },
          index: null,
          attributes: { position: { count: 9 } },
        };
      }
    },
    OrbitControls: class {
      constructor() {
        this.enableDamping = false;
        this.dampingFactor = 0;
        this.autoRotate = false;
        this.autoRotateSpeed = 0;
      }
      update() {}
    },
  };

  globalThis.setTimeout = (fn, ms) => {
    const id = timerSeq++;
    timers.push({ id, at: now + ms, fn });
    return id;
  };
  globalThis.clearTimeout = (id) => {
    timers = timers.filter((t) => t.id !== id);
  };
  globalThis.requestAnimationFrame = (cb) => {
    const id = rafSeq++;
    rafQueue.push({ id, cb });
    return id;
  };
  globalThis.cancelAnimationFrame = (id) => {
    rafQueue = rafQueue.filter((r) => r.id !== id);
  };
}

function resetQueues() {
  timers = [];
  rafQueue = [];
  renderers.length = 0;
  now = 0;
}

function flushTimers(ms) {
  const end = now + ms;
  while (true) {
    let idx = -1;
    for (let i = 0; i < timers.length; i++) {
      const better = idx < 0
        || timers[i].at < timers[idx].at
        || (timers[i].at === timers[idx].at && timers[i].id < timers[idx].id);
      if (timers[i].at <= end && better) idx = i;
    }
    if (idx < 0) break;
    const t = timers[idx];
    timers.splice(idx, 1);
    now = t.at;
    t.fn();
  }
  now = end;
}

function flushRaf() {
  const batch = rafQueue.splice(0, rafQueue.length);
  for (const frame of batch) frame.cb();
}

function viewer() {
  // `id` is defined earlier in the page script; the extracted slice expects it.
  globalThis.id = (i) => document.getElementById(i);
  return factory();
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

installDom();

const checks = [];
function check(name, fn) {
  checks.push([name, fn]);
}

check('kicks paint while frames are withheld, then stop once rAF runs', () => {
  const { init, stop } = viewer();
  init(new ArrayBuffer(4));
  flushTimers(100);
  const renderer = renderers[0];
  assert(renderer.renders === 3, `expected 3 kick paints, got ${renderer.renders}`);
  flushRaf();
  const afterFrame = renderer.renders;
  assert(afterFrame > 3, 'bootstrap frame should paint');
  flushTimers(1500);
  assert(
    renderer.renders === afterFrame,
    `kick kept painting after the first frame (${renderer.renders} vs ${afterFrame})`,
  );
  stop();
});

check('close cancels the pending kick and the bootstrap frame', () => {
  const { init, stop } = viewer();
  init(new ArrayBuffer(4));
  assert(timers.length === 1, `expected 1 kick timer, got ${timers.length}`);
  assert(rafQueue.length === 1, `expected 1 bootstrap frame, got ${rafQueue.length}`);
  const renderer = renderers[0];
  stop();
  assert(timers.length === 0, `kick timer still queued after stop (${timers.length})`);
  assert(rafQueue.length === 0, `bootstrap frame still queued after stop (${rafQueue.length})`);
  assert(renderer.disposed, 'renderer should be disposed');
  flushTimers(1500);
  flushRaf();
  assert(renderer.renders === 0, `disposed renderer was painted ${renderer.renders} time(s)`);
});

check('reopen within the kick window does not paint the disposed renderer', () => {
  const { init, stop } = viewer();
  init(new ArrayBuffer(4));
  flushTimers(0);
  const first = renderers[0];
  const painted = first.renders;
  assert(painted >= 1, 'first viewer should kick-paint before close');
  const escapedKick = timers[0];
  const escapedFrame = rafQueue[0];
  stop();
  init(new ArrayBuffer(4));
  const second = renderers[1];
  assert(timers.length === 1, 'new viewer should own the only kick timer');
  assert(rafQueue.length === 1, 'new viewer should own the only bootstrap frame');
  // A callback that already escaped clearTimeout must no-op, and must not
  // drop the new viewer's timer id.
  escapedKick.fn();
  escapedFrame.cb();
  assert(first.renders === painted, 'escaped callback painted the disposed renderer');
  assert(timers.length === 1, 'escaped kick clobbered the new timer');
  flushTimers(1500);
  assert(second.renders >= 1, 'new viewer should still kick-paint');
  assert(first.renders === painted, 'old kick painted the disposed renderer after reopen');
  const live = second.renders;
  stop();
  flushTimers(1500);
  flushRaf();
  assert(second.renders === live, 'kick painted the second renderer after it was disposed');
});

let failed = 0;
for (const [name, fn] of checks) {
  resetQueues();
  try {
    fn();
    console.log(`ok  ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL ${name}\n${err && err.stack ? err.stack : err}`);
  }
}
if (failed) {
  process.exit(1);
}
