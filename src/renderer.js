/**
 * The frame graph.
 *
 * This file owns the ORDER of things and nothing else — each pass knows how to
 * record itself but not when. That split is what makes the pipeline expandable:
 * adding a pass means writing one class and one line here.
 *
 * Frame order, and why:
 *
 *   tilecull  -> which tiles can reach the body at all
 *   raymarch  -> one jittered HDR sample + depth tag. The body, then the analytic
 *                satellites, then the player SHIP — each narrowing the next one's
 *                tmax, so the whole set resolves front-to-back with no sorting
 *   rings     -> rasterised OPAQUE geometry into `solid` + an exact per-pixel motion
 *                vector; hardware depth resolves it against itself
 *   taa       -> resolve solid vs body by depth, then blend into history — by motion
 *                vector where one exists, by camera reprojection otherwise
 *   ember sim -> particle state + compacted draw args (independent; could run
 *                earlier, but keeping it next to its draw is easier to follow)
 *   ember draw-> additive billboards, soft-occluded by the scene depth
 *   contrail  -> the ship's trail, into the SAME additive target
 *   railgun   -> beam helices, likewise
 *   aurora    -> curl-noise ribbons, likewise — the contrail's geometry, generated
 *                paths instead of a recorded one
 *   bloom     -> prefilter(scene+embers) then the down/up pyramid
 *   flare     -> ghosts and streaks from the blurred pyramid
 *   composite -> layers + limb glow + grade + grain, to the swapchain
 *
 * The rule that keeps the temporal side honest: anything camera-relative or
 * world-space dynamic must NOT enter the accumulation buffer, because reprojection
 * assumes static geometry. That is why embers, flares, the explosion fireballs, the
 * limb glow and the explosion fireballs all live downstream of TAA.
 *
 * The RINGS are the exception that proves the rule rather than breaking it: they move
 * on their own, but their motion is analytic, so they can hand TAA an exact motion
 * vector instead of asking it to guess. Anything that can say where it was last frame
 * is welcome in the accumulation buffer.
 */

import { Targets } from './core/targets.js';
import { FrameUniforms } from './core/uniforms.js';
import { Profiler } from './core/profiler.js';
import { ShaderCache } from './core/wgsl.js';
import { Camera } from './scene/camera.js';
import { PlanetoidScene } from './scenes/planetoid.js';
import { ModelViewScene } from './scenes/modelview.js';
import { SHIP_MESH } from './scene/ship_sdf.js';
import { DebugViewPass, VIEWS } from './passes/debugview.js';
import { DynamicRes } from './scene/dynres.js';
import { PULSE, QUALITY, SUNS, FILM, GLOW, FLARE, AURORA, VOLUME,
         TEMPORAL, MARCH, PROBE, wgslDefines } from './scene/tuning.js';
import { extractFrustum } from './core/frustum.js';
import { selectLod } from './core/lod.js';
import { TaaPass } from './passes/taa.js';
import { BloomPass } from './passes/bloom.js';
import { LensFlarePass } from './passes/lensflare.js';
import { CompositePass } from './passes/composite.js';

/** Hammersley-ish low-discrepancy jitter: converges far faster than white noise. */
function radicalInverse(bits) {
  let r = 0, f = 0.5, b = bits;
  for (let i = 0; i < 16; i++) { r += (b & 1) * f; b >>= 1; f *= 0.5; }
  return r;
}


export class Renderer {
  constructor(gpu) {
    this.gpu = gpu;
    this.shaders = new ShaderCache(gpu.device);
    this.targets = new Targets(gpu.device);
    this.uniforms = new FrameUniforms(gpu.device);
    this.profiler = new Profiler(gpu.device, { enabled: gpu.caps.timestamps });
    this.camera = new Camera();

    // THE SCENES. Both are constructed up front and both are initialised, so switching is a property
    // change rather than a load: the alternative stalls the one frame where a stall is most obviously
    // the switch's fault. They share the ship's contoured mesh through a memo in planetoid.js, so the
    // second one costs GPU buffers rather than another hundred milliseconds of contouring.
    this.scenes = {
      planetoid: new PlanetoidScene(gpu, this.targets, this.shaders),
      modelview: new ModelViewScene(gpu, this.targets, this.shaders),
    };
    this.sceneKey = 'planetoid';

    this.frameIndex = 0;
    this.dynres = new DynamicRes();
    this._warnedNoTimestamps = false;
    /** Set by the freeze control: dt becomes exactly 0, so no simulation advances. */
    this.held = false;
    /** Index into VIEWS, or -1 for the normal composite. See passes/debugview.js. */
    this.debugView = -1;
    /** Five frustum planes, reused every frame. See core/frustum.js for why five. */
    this._frustum = new Float32Array(20);
    // The controller is FED, not polling: one genuine sample per resolved frame — see dynres.js.
    this.profiler.onFrame = (ms) => { if (QUALITY.dynamicRes) this.dynres.sample(ms); };
    this.accumFrames = 0;
    this.prevTime = 0;

    // Scratch, reused every frame. A renderer that allocates per frame hands the
    // collector a steady drip of short-lived objects, and a GC pause is a dropped
    // frame — which on an accumulating renderer also means a visible hitch, since
    // the camera jumps further between samples than the history expects.
    this._sunA = [0, 0];
    this._sunB = [0, 0];
    this._jitter = [0, 0];
    // The render context handed to the scene each frame. Allocated once and refilled, for the same
    // reason every other scratch object here is: a per-frame object is a per-frame allocation.
    this._rc = {
      renderer: this, gpu: this.gpu, targets: this.targets, camera: this.camera,
      shaders: this.shaders, frameBGL: null,
      time: 0, dt: 0, input: null, frustum: null,
    };
    // Declared in full, every field the uniform writer reads. Two reasons: this literal is
    // the frame-state contract, so a reader should not have to scan `frame()` to learn what
    // it contains; and adding keys to an object after construction changes its shape, which
    // is exactly the per-frame deoptimisation the scratch objects above exist to avoid.
    this._state = {
      camera: null, time: 0, viewMode: 0, focusDist: 0, width: 0, height: 0,
      accumWidth: 0, accumHeight: 0, addWidth: 0, addHeight: 0,
      beat: 0, life: 0, frameIndex: 0, dt: 0, jitter: null, lens: null,
      historyValid: false, dragging: false, exposure: 0,
      sunA: null, sunB: null, ship: null,
      taa: TEMPORAL, march: MARCH, probe: PROBE, renderScale: 0,
      grade: FILM, glow: GLOW, flareStrength: 0, aurora: AURORA, auroraPhase: 0,
      volume: VOLUME,
      // Contributed by the active scene's `writeState`. Declared here anyway: this literal is the
      // frame-state contract, and a field that only exists in one scene is exactly the kind that
      // silently carries the other scene's last value.
      modelView: false, modelSpin: 0, modelPrevSpin: 0,
    };

    // ONLY THE SHARED CHAIN. Everything that draws a world belongs to a scene; what is left here is
    // what every scene gets for free — resolve, bloom, flare, grade, and the buffer viewer.
    this.passes = {
      taa: new TaaPass(gpu, this.targets, this.shaders),
      bloom: new BloomPass(gpu, this.targets, this.shaders),
      flare: new LensFlarePass(gpu, this.targets, this.shaders),
      composite: new CompositePass(gpu, this.targets, this.shaders),
      debugview: new DebugViewPass(gpu, this.targets, this.shaders),
    };
  }

  /** The active scene. */
  get scene() { return this.scenes[this.sceneKey]; }

  /**
   * The player ship, when the active scene has one.
   *
   * A forwarding accessor rather than a field, because the ship belongs to the planetoid now. It exists
   * because the benchmark suppresses auto-fire through it, and a probe reaching for `renderer.ship`
   * should get `undefined` in a scene without one rather than a stale object from another.
   */
  get ship() { return this.scene.ship; }

  /** Switch scenes. Both are already initialised, so this is a property change. */
  setScene(key) {
    if (!this.scenes[key] || key === this.sceneKey) return;
    this.sceneKey = key;
    // The history describes a world that no longer exists: reprojecting into it would smear the old
    // scene across the first frames of the new one.
    this.resetHistory();
  }

  async init() {
    const d = this.gpu.device;
    // One layout for group 0, shared by every pass, so the frame bind group is
    // interchangeable across pipelines and never has to be rebuilt per pass.
    this.frameBGL = FrameUniforms.bindGroupLayout(d);
    this.frameBG = this.uniforms.bindGroup(this.frameBGL);
    this._rc.frameBGL = this.frameBGL;
    const rc = this._rc;

    await Promise.all([
      this.passes.taa.init(this.frameBGL),
      this.passes.bloom.init(this.frameBGL),
      this.passes.flare.init(this.frameBGL),
      this.passes.composite.init(this.frameBGL, this.gpu.format),
      this.passes.debugview.init(this.frameBGL, this.gpu.format, wgslDefines()),
      // Every scene, not just the active one — see the constructor for why.
      ...Object.values(this.scenes).map((sc) => sc.init(rc)),
    ]);
  }

  /** Invalidate the temporal history — after a resize, there is nothing valid. */
  resetHistory() {
    this.accumFrames = 0;
  }

  resize() {
    // `syncSize` reports whether the CANVAS changed, which is not the same question as
    // whether the TARGETS need rebuilding. Gating the second on the first meant any
    // setting that changes target sizes without changing the window — `renderScale`, and
    // now `QUALITY.taau` — was silently ignored until something else forced a resize.
    // Targets owns its own idempotence, so just ask it every frame; the check is a couple
    // of integer compares.
    this.gpu.syncSize(QUALITY.maxWidth);
    if (!this.gpu.width) return;
    if (this.targets.resize(this.gpu.width, this.gpu.height)) this.resetHistory();
  }

  /**
   * @param {number} time seconds
   * @param {object} input pointer state
   */
  frame(time, input) {
    this.resize();
    const t = this.targets;
    if (!t.width) return;

    // HELD means dt is exactly zero, so nothing integrates at all.
    //
    // The floor of 1e-4 below is what stops a repeated clock from being a true freeze: every sim
    // still advanced 0.1ms per frame, so the embers, contrails and ribbons crept sub-pixel forever.
    // Being additive and drawn after the resolve they get no antialiasing, so that creep showed up
    // as isolated pixels swinging by ~100 of 255 levels between consecutive frames of a scene that
    // was supposed to be still - which made the frozen baseline a slow-motion scene rather than a
    // stopped one. Nothing divides by dt except one guarded line in ship.js, so zero is safe.
    const dt = this.held ? 0 : Math.min(0.1, Math.max(1e-4, time - this.prevTime));

    // DYNAMIC RESOLUTION, decided here and applied by the next frame's `resize`. Deliberately not
    // applied mid-frame: reallocating render targets between passes is the hitch this feature
    // exists to avoid, and a one-frame delay costs nothing. GPU time when the timestamps are
    // available, wall time only as a fallback — see dynres.js for why that distinction matters.
    if (QUALITY.dynamicRes && this.profiler.enabled) {
      const step = this.dynres.update();
      if (step !== null) {
        QUALITY.renderScale = step.scale;
        QUALITY.additiveDisplayRes = step.additive;
      }
    } else if (QUALITY.dynamicRes && !this._warnedNoTimestamps) {
      // Once, not every frame. Wall time cannot substitute — see dynres.js.
      this._warnedNoTimestamps = true;
      console.warn('dynamic resolution needs timestamp queries, which this adapter lacks; holding renderScale');
    }

    // Projection first: it depends only on the viewport, and the camera folds it into the view matrix
    // it is about to build. Then the scene advances its own world and places the camera — which of
    // those two things happens is the scene's business, not this method's.
    this.camera.setViewport(t.width, t.height);

    const rc = this._rc;
    rc.time = time;
    rc.dt = dt;
    rc.input = input;
    this.scene.update(rc);

    // Sun screen positions, for the flare pass to anchor its streaks. 1e3 is the
    // sentinel the shader reads as "behind the camera", written in place so the
    // miss path allocates nothing either.
    if (!this.camera.projectDirection(SUNS.a.dir, this._sunA)) {
      this._sunA[0] = 1e3; this._sunA[1] = 1e3;
    }
    if (!this.camera.projectDirection(SUNS.b.dir, this._sunB)) {
      this._sunB[0] = 1e3; this._sunB[1] = 1e3;
    }

    const fi = this.frameIndex;
    const st = this._state;

    // Pixel jitter drives the AA; the lens offset drives the bokeh. Both are
    // low-discrepancy so a handful of frames already looks converged.
    this._jitter[0] = ((fi * 0.618033988) % 1) - 0.5;
    this._jitter[1] = radicalInverse(fi) - 0.5;
    if (PROBE.zeroJitter) { this._jitter[0] = 0; this._jitter[1] = 0; }

    st.camera = this.camera;

    st.focusDist = this.camera.focusDistance();

    st.viewMode = this.debugView >= 0 ? VIEWS[this.debugView].mode : 0;
    st.time = time;
    st.width = t.width;
    st.height = t.height;
    st.addWidth = t.addWidth;
    st.addHeight = t.addHeight;
    st.accumWidth = t.accumWidth;
    st.accumHeight = t.accumHeight;
    st.beat = time * (PULSE.bpm / 60);
    st.life = time * PULSE.lifeRate;
    st.frameIndex = fi;
    st.dt = dt;
    st.jitter = this._jitter;
    st.historyValid = this.accumFrames > 0;
    st.dragging = input.dragging;
    // Pre-grade linear gain. Its own control, not a second application of `exposure` — see
    // the note on FILM.gain for why those were split.
    st.exposure = FILM.gain;
    st.sunA = this._sunA;
    st.sunB = this._sunB;
    st.taa = TEMPORAL;      // read live, so console tweaks take effect immediately
    st.march = MARCH;
    st.probe = PROBE;
    st.renderScale = QUALITY.renderScale;
    st.grade = FILM;
    st.glow = GLOW;
    st.flareStrength = FLARE.strength;
    st.aurora = AURORA;
    st.volume = VOLUME;
    // The scene's own fields last, so it can override anything above it.
    this.scene.writeState(st, rc);
    this.uniforms.write(st);

    const encoder = this.gpu.device.createCommandEncoder({ label: 'frame' });
    this.profiler.beginFrame();
    const p = this.profiler;
    const { taa, bloom, flare, composite } = this.passes;
    const solids = this.scene.solidPasses;

    // DEBUG GROUPS around the phases, not around every pass.
    //
    // Each pass already labels its own render pass, which is what a capture lists; what a capture
    // cannot infer is the STRUCTURE - which passes belong to the same phase, and therefore which of
    // them a change should be expected to move. These names are the frame graph's own phases, so a
    // capture reads the way the architecture document describes it. Free when no tool is attached.

    // Frustum planes once per frame, from the same matrix the vertex stages use, into a buffer that is
    // reused rather than reallocated.
    extractFrustum(this.camera.viewProj, this._frustum);
    rc.frustum = this._frustum;

    // LOD and the wireframe view apply to whatever the scene lists, because both are properties of how
    // this renderer draws meshes rather than of what any scene contains.
    const view = this.debugView >= 0 ? VIEWS[this.debugView] : null;
    const wire = !!view?.wireframe;
    for (const m of solids) {
      m.wireframe = wire;
      if (wire) m.prepareWireframe(this.frameBGL, wgslDefines());
      if (!m.meshes || m.meshes.length < 2) continue;
      // The distance is to the object's bounding sphere CENTRE, which the sphere-based culling already
      // knows how to place — so selection and culling share one fact about where an object is.
      const s0 = m.spec.worldSphere?.(0, m.mesh.ranges[0]);
      const c = this.camera.current.pos;
      const dist = s0
        ? Math.hypot(s0[0] - c[0], s0[1] - c[1], s0[2] - c[2])
        // No sphere: the object is wherever the scene put it, and the only distance available is to
        // whatever the camera is looking at. Good enough for a model on a turntable.
        : this.camera.distance;
      m.lod = selectLod(m.meshes, dist, 1.2, Math.hypot(t.width, t.height), SHIP_MESH.lodErrorPx);
    }

    encoder.pushDebugGroup('scene');
    this.scene.recordWorld(encoder, this.frameBG, p, rc);
    encoder.popDebugGroup();

    encoder.pushDebugGroup('resolve');
    taa.record(encoder, this.frameBG, p);
    encoder.popDebugGroup();

    encoder.pushDebugGroup('additive');
    this.scene.recordAdditive(encoder, this.frameBG, p, rc);
    encoder.popDebugGroup();

    encoder.pushDebugGroup('post');
    bloom.record(encoder, this.frameBG, p);
    flare.record(encoder, this.frameBG, p);

    const surface = this.gpu.context.getCurrentTexture().createView();
    // The buffer viewer REPLACES the composite rather than drawing over it: the point is to see the
    // buffer, not the buffer under a film grade. Everything upstream still ran, so the timings on the
    // HUD stay comparable to a normal frame.
    const shown = this.debugView >= 0 && this.debugView < VIEWS.length
      ? this.passes.debugview.record(encoder, this.frameBG, surface, this.debugView, p)
      : false;
    if (!shown) {
      composite.record(encoder, this.frameBG, surface,
        bloom.resultView, flare.resultView, p);
    }

    encoder.popDebugGroup();

    p.resolve(encoder);
    this.gpu.device.queue.submit([encoder.finish()]);
    p.readback();                          // fire and forget

    t.swapAccum();
    this.frameIndex++;
    this.accumFrames++;
    this.prevTime = time;
  }

  destroy() {
    for (const sc of Object.values(this.scenes)) sc.destroy();
    this.targets.destroy();
    this.profiler.destroy();
  }
}

