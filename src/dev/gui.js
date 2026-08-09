/**
 * The tuning panel.
 *
 * Every control here binds STRAIGHT to the live tuning object — no shadow state, no apply
 * button, no serialisation. That is only possible because the values it touches are read fresh
 * every frame: the grade and the glow are uniforms (see FrameUniforms.write), the camera's
 * projection is rebuilt when its inputs change, and the rest are plain objects the renderer
 * dereferences per frame. A control bound to something baked at pipeline creation would be a
 * slider that does nothing, which is worse than no slider, so anything in that category either
 * got moved onto the uniform first or is not exposed here.
 *
 * Loaded LAZILY, on the first press of `g`. lil-gui is 59 KB and this is a debug surface, so
 * it has no business being in the startup path of a renderer whose whole point is frame time.
 *
 * lil-gui (MIT) is vendored rather than fetched from a CDN: this project has no build step and
 * no package manager, and it should keep working with the network off.
 */

import GUI from '../../vendor/lil-gui.esm.js';
import { VIEWS } from '../passes/debugview.js';
import { MODELS } from '../scenes/modelview.js';
import { FILM, GLOW, FLARE, CAMERA, AURORA, VOLUME, TEMPORAL, QUALITY } from '../scene/tuning.js';

const STORE = 'beep.presets';

/** Saved presets, as name -> lil-gui state. Absent or unreadable storage is not an error. */
function loadStore() {
  try { return JSON.parse(localStorage.getItem(STORE) ?? '{}') ?? {}; } catch { return {}; }
}
function saveStore(all) {
  try { localStorage.setItem(STORE, JSON.stringify(all)); return true; } catch { return false; }
}

/**
 * @param {import('../renderer.js').Renderer} renderer
 * @param {object} [live]  per-frame readings the host loop owns — currently `{ fps }`. Passed in
 *                         because the loop knows the frame rate and the renderer does not.
 * @returns {GUI}
 */
export function buildGui(renderer, live = {}) {
  const gui = new GUI({ title: 'beep beep beep', width: 300 });

  // ---- scene ----
  // FIRST, because it decides what every control below is acting on. The model selector only means
  // anything in the viewer, so it is shown and hidden with the scene rather than sitting greyed out.
  const sceneFolder = gui.addFolder('Scene');
  const sceneProxy = {
    scene: renderer.sceneKey,
    model: MODELS[renderer.scenes.modelview.model].key,
  };

  // SCENE FIRST, THEN MODEL. The model row is hidden outside the viewer, so with it added first the
  // scene row jumped up a line every time you switched away from the viewer and back — the control
  // you just used moving out from under the pointer. A row that appears and disappears has to be
  // BELOW the one that controls it.
  const sceneCtl = sceneFolder
    .add(sceneProxy, 'scene', Object.fromEntries(
      Object.entries(renderer.scenes).map(([key, sc]) => [sc.constructor.label, key])))
    .onChange((key) => { renderer.setScene(key); syncModelVisibility(); });

  const modelCtl = sceneFolder
    .add(sceneProxy, 'model', Object.fromEntries(MODELS.map((m) => [m.label, m.key])))
    .onChange((key) => {
      renderer.scenes.modelview.model = MODELS.findIndex((m) => m.key === key);
      // The two models are different sizes, so the camera re-frames; the history describes the old
      // framing and would smear across the switch.
      renderer.resetHistory();
    });

  // lil-gui's own show/hide, not a style on the DOM node. A controller's `domElement.parentElement` is
  // the FOLDER's children container, so setting display there hides every control in the folder.
  const syncModelVisibility = () => { modelCtl.show(renderer.sceneKey === 'modelview'); };
  syncModelVisibility();

  // WHAT IS ACTUALLY BEING DRAWN, in triangles.
  //
  // Reported for the ACTIVE SCENE rather than only for the model viewer, which needs no special case
  // and answers a question worth asking in both: the viewer says how heavy the model you are
  // inspecting is, and the planetoid says what the rings, hull and satellites cost together at the
  // level and cull state of the frame that just went out.
  //
  // Counts follow the LOD selection and the instance count, so they are what the GPU was asked for,
  // not what the buffers hold. `draws` is post-cull — the gap between it and the object count is the
  // frustum test doing its job.
  const geo = { triangles: 0, vertices: 0, draws: 0, lod: '' };
  const geoCtrls = [];
  for (const k of ['triangles', 'vertices', 'draws', 'lod']) {
    geoCtrls.push(sceneFolder.add(geo, k).disable());
  }

  // ---- grade ----
  // First, because it is the one set of controls you reach for while LOOKING at the image
  // rather than while reasoning about it.
  const film = gui.addFolder('Film');
  // Kelvin, and it reads like a camera's white balance: up warms, down cools. 1800 is candlelight and
  // 12000 is deep shade, which is wider than anyone needs and cheap to allow. 3200 is where Eastman
  // 50T actually sits; the default of 4750 is where the hand-picked balance it replaced sat.
  film.add(FILM, 'temperature', 1800, 12000, 25).name('temperature (K)');
  film.add(FILM, 'toneMap', { AgX: 1, Hable: 0 }).name('display transform');
  // Folders a PRESET covers, keyed by a stable string of our own rather than by lil-gui's
  // private `_title`: the key ends up in saved JSON, so it must not move if a folder is
  // renamed or reordered.
  const artFolders = [['film', film]];
  film.add(FILM, 'gain', 0.05, 3.0, 0.01).name('gain (pre-grade)');
  film.add(FILM, 'exposure', 0.1, 6.0, 0.01).name('exposure (curve)');
  film.add(FILM, 'white', 2.0, 24.0, 0.1).name('white point');
  film.add(FILM, 'contrast', 0.5, 2.0, 0.01);
  film.add(FILM, 'blackLift', 0.0, 0.12, 0.001).name('black lift');
  film.add(FILM, 'saturation', 0.0, 2.0, 0.01);
  film.add(FILM, 'halation', 0.0, 3.0, 0.01);
  film.add(FILM, 'vignette', 0.0, 1.0, 0.01).name('vignette (cos^4)');
  film.add(FILM, 'grain', 0.0, 0.12, 0.001);

  // ---- glow ----
  const glow = gui.addFolder('Glow').close();
  artFolders.push(['glow', glow]);
  glow.add(GLOW, 'threshold', 0.0, 6.0, 0.01);
  glow.add(GLOW, 'strength', 0.0, 0.8, 0.005);
  // Both of these are baked into the per-level parameter buffers, which are only written on
  // resize — so they need an explicit rebuild rather than just a new value.
  glow.add(GLOW, 'radius', 0.25, 3.0, 0.01)
    .onChange(() => renderer.passes.bloom.invalidate());
  glow.add(GLOW, 'levelWeight', 0.2, 1.6, 0.01).name('level weight')
    .onChange(() => renderer.passes.bloom.invalidate());
  glow.add(FLARE, 'strength', 0.0, 2.0, 0.01).name('flare');

  // ---- camera ----
  const cam = gui.addFolder('Camera').close();
  artFolders.push(['camera', cam]);
  cam.add(CAMERA, 'diagonalFov', 40, 140, 0.5).name('fov (diagonal)');
  cam.add(CAMERA, 'distance', 2.0, 12.0, 0.05);
  cam.add(CAMERA, 'zoom', 0.0, 3.0, 0.01).name('dolly amplitude');
  cam.add(CAMERA, 'roll', 0.0, 0.6, 0.01);

  // ---- planet mesh (rasterized scene) ----
  {
    const pm = gui.addFolder('Planet Mesh');
    import('../passes/PlanetMeshManager.js').then(({ PLANET_MESH }) => {
      pm.add(PLANET_MESH, 'ny', 8, 256, 8).name('resolution');
    });
  }

  // ---- auroras ----
  // The flow parameters are read per step, so they take effect on the next frame — but the
  // ribbons already in the buffer were integrated under the OLD ones, so a big change takes a
  // ribbon lifetime to fully show. `reseed` skips that wait.
  const aur = gui.addFolder('Auroras').close();
  artFolders.push(['aurora', aur]);
  aur.add(AURORA, 'speed', 0.05, 1.0, 0.01);
  aur.add(AURORA, 'maxTurn', 0.1, 4.0, 0.05).name('max turn (rad/s)');
  aur.add(AURORA, 'curlScale', 0.05, 1.2, 0.01).name('curl scale');
  aur.add(AURORA, 'curlDrift', 0.0, 0.5, 0.005).name('curl drift');
  aur.add(AURORA, 'gain', 0.0, 3.0, 0.01);
  aur.add(AURORA, 'grazeFade', 0.05, 1.0, 0.01).name('graze fade');
  aur.add(AURORA, 'rays', 0.0, 1.0, 0.01).name('ray depth');
  aur.add({ reseed: () => renderer.aurora.reseed() }, 'reseed');

  // ---- atmosphere ----
  //
  // Three levels, and they are uniforms rather than constants precisely so these sliders work —
  // the rest of VOLUME is geometry (step count, falloff, the shadow penumbrae) and stays a
  // compile-time constant, so it is not exposed rather than shipped as controls that do nothing.
  const vol = gui.addFolder('Atmosphere').close();
  artFolders.push(['volume', vol]);
  vol.add(VOLUME, 'sigma', 0.0, 3.0, 0.01).name('thickness');
  vol.add(VOLUME, 'g', 0.0, 0.9, 0.01).name('forward scatter');
  vol.add(VOLUME, 'ringOpacity', 0.0, 1.0, 0.01).name('ring shadow');

  // ---- temporal ----
  const taa = gui.addFolder('Temporal').close();
  artFolders.push(['temporal', taa]);
  taa.add(TEMPORAL, 'blend', 0.02, 1.0, 0.01);
  taa.add(TEMPORAL, 'clipGamma', 0.5, 4.0, 0.05).name('clip gamma');
  taa.add(TEMPORAL, 'clipGammaUpsample', 0.5, 4.0, 0.05).name('clip gamma (TAAU)');
  taa.add(TEMPORAL, 'depthGate', 0.001, 0.06, 0.001).name('depth gate');
  taa.add(TEMPORAL, 'weightMax', 1, 32, 1).name('weight max');
  taa.add(TEMPORAL, 'taauSigma', 0.4, 2.0, 0.01).name('TAAU sigma');
  // Reallocates the accumulation buffer, so it goes through the renderer rather than
  // being poked directly.
  taa.add(QUALITY, 'taau').name('upsampling').onChange(() => renderer.resize());
  taa.add(QUALITY, 'renderScale', 0.25, 1.0, 0.05).name('render scale')
    .onChange(() => renderer.resize());
  // Reallocates the additive target. Off by default; `beep.additive()` measures both sides of
  // the trade, and on a contended machine the cost side of it does not resolve.
  taa.add(QUALITY, 'additiveDisplayRes').name('additive @ display res')
    .onChange(() => renderer.resize());
  // The controller reads `renderScale` back every window, so dragging that slider while this is on
  // is a suggestion rather than a setting. `reset` stops it reacting to samples from before.
  taa.add(QUALITY, 'dynamicRes').name('dynamic resolution')
    .onChange(() => renderer.dynres.reset());
  taa.add(QUALITY, 'dynamicTargetMs', 4, 40, 0.5).name('target gpu ms')
    .onChange(() => renderer.dynres.reset());

  // ---- monitors ----
  //
  // Read-only: disabled so they cannot be dragged, since a slider that writes to a derived value
  // is a trap. They are refreshed on a timer — see the note on `refresh` for why not `.listen()`.
  //
  // Deliberately NOT the per-pass timings: the corner HUD already shows those, and duplicating
  // them here would mean two places to keep honest. What this adds is the state you otherwise
  // have to reconstruct from three tuning values — which grid each stage is actually running at.
  const mon = { fps: 0, resolution: '', accumulation: '', additive: '', converged: 0, gpu: 0 };
  const monitor = gui.addFolder('Monitor');
  const monCtrls = [];
  const readonly = (key, name) => monCtrls.push(monitor.add(mon, key).name(name).disable());
  readonly('fps', 'fps');
  readonly('gpu', 'gpu total (ms)');
  readonly('resolution', 'render');
  readonly('accumulation', 'accumulate');
  readonly('additive', 'additive');
  readonly('converged', 'frames accumulated');

  // `updateDisplay` explicitly rather than lil-gui's `.listen()`. Two reasons, and the second
  // is the one that matters: a controller snapshots its value when it is created, so mutating
  // the bound object afterwards shows nothing until something re-reads it; and `.listen()`
  // re-reads on requestAnimationFrame, which ties the panel's cadence to a clock that stops in a
  // background tab and is throttled in an inactive one. Driving it from the same interval that
  // computes the values keeps the two in step and depends on nothing else.
  const refresh = () => {
    const t = renderer.targets;
    mon.fps = Math.round(live.fps ?? 0);
    mon.gpu = +renderer.profiler.total().toFixed(2);
    // The panel can be opened before the first resize has allocated anything, and a monitor that
    // says `undefinedxundefined` for half a second reads as a bug in the thing being monitored.
    const size = (w, h) => (w && h ? `${w}x${h}` : 'not allocated');
    mon.resolution = size(t.width, t.height);
    mon.accumulation = size(t.accumWidth, t.accumHeight);
    mon.additive = size(t.addWidth, t.addHeight);
    mon.converged = renderer.accumFrames;
    for (const c of monCtrls) c.updateDisplay();

    // PULL THE SELECTORS BACK INTO STEP, for the same reason the buffer dropdown does it below: the
    // panel is not the only writer. Everything here is reachable from `window.beep`, and this project
    // expects you to use that — a dropdown reading "satellite" over a picture of the ship is the panel
    // lying about the thing it exists to report.
    if (sceneProxy.scene !== renderer.sceneKey) {
      sceneProxy.scene = renderer.sceneKey;
      sceneCtl.updateDisplay();
      syncModelVisibility();
    }
    const modelKey = MODELS[renderer.scenes.modelview.model]?.key;
    if (modelKey && sceneProxy.model !== modelKey) {
      sceneProxy.model = modelKey;
      modelCtl.updateDisplay();
    }

    let tris = 0;
    let verts = 0;
    let draws = 0;
    const levels = [];
    for (const m of renderer.scene.solidPasses) {
      if (!m.mesh) continue;
      const instances = m.spec.instances ?? 1;
      tris += (m.mesh.indexCount / 3) * instances;
      verts += m.mesh.vertexCount * instances;
      draws += m.drawn ?? 0;
      if (m.meshes && m.meshes.length > 1) levels.push(`${m.spec.label} L${m.lod}/${m.meshes.length - 1}`);
    }
    geo.triangles = tris.toLocaleString();
    geo.vertices = verts.toLocaleString();
    geo.draws = draws;
    // Only meshes with a chain have a level worth reporting; everything else would just say L0/0.
    geo.lod = levels.length ? levels.join(', ') : 'single level';
    for (const c of geoCtrls) c.updateDisplay();
    // The `b` key writes renderer.debugView directly, so pull it back in rather than assuming the
    // dropdown is the only writer.
    if (viewState.buffer !== renderer.debugView) {
      viewState.buffer = renderer.debugView;
      viewCtrl.updateDisplay();
    }
  };
  // ---- traces ----
  //
  // lil-gui has no graph widget, so this is a canvas appended to the folder's own children
  // container. The one piece of raw DOM in here, and it earns it: a single number cannot show a
  // regression that only appears every few seconds, and the frame time on this renderer is spiky
  // enough that a median hides real stalls.
  //
  // Two traces on one canvas, each auto-scaled to its own recent maximum rather than to a fixed
  // range — the interesting thing about frame time is its SHAPE, and a fixed axis either clips the
  // spikes or flattens everything into the bottom pixel.
  const TRACE_N = 120;
  const traces = [
    { key: 'gpu', label: 'gpu ms', colour: '#7fd4ff', data: new Float32Array(TRACE_N) },
    { key: 'fps', label: 'fps', colour: '#ffd27f', data: new Float32Array(TRACE_N) },
  ];
  let traceHead = 0;
  const plot = document.createElement('canvas');
  plot.width = 280;
  plot.height = 64;
  plot.style.cssText = 'display:block;width:calc(100% - 12px);height:64px;margin:4px 6px 6px;'
    + 'background:rgba(0,0,0,0.25);border-radius:2px';
  monitor.$children.appendChild(plot);
  const g2d = plot.getContext('2d');

  const drawTraces = () => {
    const w = plot.width, h = plot.height;
    g2d.clearRect(0, 0, w, h);
    for (const tr of traces) {
      let hi = 0;
      for (const v of tr.data) hi = Math.max(hi, v);
      if (hi <= 0) continue;
      g2d.strokeStyle = tr.colour;
      g2d.lineWidth = 1;
      g2d.beginPath();
      for (let i = 0; i < TRACE_N; i++) {
        // Oldest sample first, so the trace reads left to right in time.
        const v = tr.data[(traceHead + i) % TRACE_N];
        const x = (i / (TRACE_N - 1)) * (w - 1);
        const y = h - 1 - (v / hi) * (h - 3);
        if (i === 0) g2d.moveTo(x, y); else g2d.lineTo(x, y);
      }
      g2d.stroke();
      g2d.fillStyle = tr.colour;
      g2d.font = '9px ui-monospace, monospace';
      g2d.fillText(`${tr.label} peak ${hi.toFixed(hi < 10 ? 2 : 0)}`,
        4, 10 + traces.indexOf(tr) * 10);
    }
  };

  const sample = () => {
    traces[0].data[traceHead] = mon.gpu;
    traces[1].data[traceHead] = mon.fps;
    traceHead = (traceHead + 1) % TRACE_N;
    drawTraces();
  };

  // ---- measurements ----
  //
  // The instruments are console-only otherwise, which means knowing they exist and what to type.
  // Each button runs one and leaves its headline here; the full report still goes to the console,
  // because a single number is a summary and the reports exist to stop it being read alone.
  const result = { last: 'idle' };
  // ---- the buffer viewer ----
  //
  // A control rather than a readout, so the panel can select as well as report - and kept in step
  // with the `b` key by `refresh` below, because either can change it. The list comes from
  // passes/debugview.js so the panel cannot drift from what the pass can actually show.
  const viewNames = { 'off (composite)': -1 };
  VIEWS.forEach((v, i) => { viewNames[v.name] = i; });
  const viewState = { buffer: -1 };
  const viewCtrl = gui.add(viewState, 'buffer', viewNames).name('show buffer')
    .onChange((v) => { renderer.debugView = v; });

  // AFTER `viewState` and `viewCtrl`, not before. `refresh` closes over both to keep the dropdown in
  // step with the `b` key, so calling it earlier hit their temporal dead zone and threw
  // "Cannot access 'viewState' before initialization" — which rejected `buildGui`'s promise and left
  // the panel silently absent rather than visibly broken. The first call has to follow the last
  // declaration it reads.
  refresh();
  // Twice a second: these are for reading, and a value that changes every frame is unreadable.
  // The trace samples at the same rate, so 120 points is a minute of history.
  const timer = setInterval(() => { refresh(); sample(); }, 500);

  const measure = gui.addFolder('Measure').close();
  const resultCtrl = measure.add(result, 'last').name('result').disable();
  const run = (label, fn, headline) => measure.add({
    [label]: async () => {
      result.last = `${label}...`;
      resultCtrl.updateDisplay();
      try {
        result.last = headline(await fn());
      } catch (e) {
        result.last = `failed: ${e.message}`;
        console.error(e);
      }
      resultCtrl.updateDisplay();
    },
  }, label);
  const pct = (v) => `${(v * 100).toFixed(3)}%`;
  run('stability', () => window.beep.bench({ stability: true, frames: 90 }),
      (r) => `residual ${pct(r.stability.medianRelative)}`);
  run('lag', () => window.beep.lag(), (r) => `${r.lagOverNoise.toFixed(2)}x noise`);
  run('sub-pixel', () => window.beep.subpixel(),
      (r) => r.rows.map((x) => `${x.target} ${x.worstStep}%`).join('  '));
  run('detail', () => window.beep.detail(),
      (r) => `taau retains ${(r.configs.find((c) => c.name === 'taau').retained * 100).toFixed(0)}%`);
  run('field evals', () => window.beep.evals(), (r) => `${r.meanEvalsPerPixel.toFixed(2)} /px`);

  // ---- presets ----
  //
  // A preset captures the ART STATE and nothing else. That distinction is the whole design, and
  // getting it wrong was a real bug: `gui.save()` walks EVERY folder, so the first version stored
  // Monitor, Measure and Presets along with the grade — and since the preset picker is itself a
  // saved controller, loading a preset re-entered the picker's own onChange and the load never
  // completed. Snapshotting the tuning folders explicitly makes that impossible, and it is also
  // just correct: "which preset was selected" is not part of a look.
  //
  // `load` goes through setValue, so the onChange hooks above still fire — a preset that changes
  // `renderScale` really does reallocate the targets.
  const snapshot = () => {
    const folders = {};
    for (const [key, f] of artFolders) folders[key] = f.save();
    return { folders };
  };
  const restore = (snap) => {
    for (const [key, f] of artFolders) {
      const data = snap?.folders?.[key];
      if (data) f.load(data);
    }
  };

  // Captured before anything has been touched, so "defaults" means what the file says rather
  // than whatever happened to be live when the panel was first opened.
  const defaults = snapshot();
  const presets = gui.addFolder('Presets').close();
  const state = { name: 'my look', saved: '' };
  let savedCtrl = null;
  const listNames = () => {
    state.saved = Object.keys(loadStore()).join(', ') || '(none)';
    savedCtrl?.updateDisplay();
  };

  presets.add(state, 'name');
  presets.add({
    save: () => {
      const n = state.name.trim();
      if (!n) { return; }
      const all = loadStore();
      all[n] = snapshot();
      if (!saveStore(all)) { console.warn('presets: localStorage unavailable, not saved'); return; }
      listNames();
    },
  }, 'save').name('save as name');
  presets.add({
    load: () => {
      const snap = loadStore()[state.name.trim()];
      if (!snap) { console.warn(`presets: no preset named "${state.name}"`); return; }
      restore(snap);
    },
  }, 'load').name('load name');
  presets.add({
    remove: () => {
      const all = loadStore();
      delete all[state.name.trim()];
      saveStore(all);
      listNames();
    },
  }, 'remove').name('delete name');
  presets.add({ reset: () => restore(defaults) }, 'reset').name('reset to file defaults');
  presets.add({
    copy: () => {
      const json = JSON.stringify(snapshot(), null, 2);
      navigator.clipboard?.writeText(json).catch(() => {});
      console.log(json);
    },
  }, 'copy').name('copy JSON to clipboard');
  // A plain read-only list rather than a dropdown: a dropdown's options have to be rebuilt every
  // time the set changes, which in lil-gui means destroying and re-adding the controller — and
  // that reorders the folder. Typing a name is one more keystroke and nothing to go wrong.
  savedCtrl = presets.add(state, 'saved').name('saved presets').disable();
  listNames();

  // The interval outlives the panel unless something stops it, and `g` destroys the panel.
  const destroy = gui.destroy.bind(gui);
  gui.destroy = () => { clearInterval(timer); destroy(); };

  return gui;
}
