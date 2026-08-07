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
import { Ship } from './scene/ship.js';
import { Contrail } from './scene/contrail.js';
import { AdditivePass } from './passes/additive.js';
import { DebugViewPass, VIEWS } from './passes/debugview.js';
import { Railgun } from './scene/railgun.js';
import { Aurora } from './scene/aurora.js';
import { DynamicRes } from './scene/dynres.js';
import { PULSE, QUALITY, SUNS, FILM, GLOW, FLARE, AURORA, VOLUME, RINGS, CONTRAIL, RAIL, SHIP,
         SATELLITES, TEMPORAL, MARCH, PROBE, wgslDefines } from './scene/tuning.js';
import { SolidMeshPass } from './passes/solidmesh.js';
import { Mesh } from './core/mesh.js';
import { extractFrustum } from './core/frustum.js';
import { rectTube, box, concatMeshes } from './scene/meshgen.js';
import { shipTree, SHIP_MESH } from './scene/ship_sdf.js';
import { compile, bounds } from './scene/sdf/nodes.js';
import { dualContour, resolutionForScreen } from './scene/sdf/dualcontour.js';
import { ringDims } from './scene/tuning.js';
import { ScenePass } from './passes/scene.js';
import { TaaPass } from './passes/taa.js';
import { BloomPass } from './passes/bloom.js';
import { LensFlarePass } from './passes/lensflare.js';
import { EmberPass } from './passes/embers.js';
import { CompositePass } from './passes/composite.js';

/** Hammersley-ish low-discrepancy jitter: converges far faster than white noise. */
function radicalInverse(bits) {
  let r = 0, f = 0.5, b = bits;
  for (let i = 0; i < 16; i++) { r += (b & 1) * f; b >>= 1; f *= 0.5; }
  return r;
}


/**
 * Mesh the ship's SDF, once, at a resolution derived from how large a cell may look on screen.
 *
 * RESOLUTION IS DERIVED, NOT FIXED, so the mesh suits the window rather than the machine it was authored
 * on — and because the same call at other resolutions is the LOD chain, whenever that is wanted. Capped
 * because this runs at startup: measured on the ship's tree, 40 cells is 3540 triangles in 43 ms and 64
 * cells is 8836 in 101 ms, and a hull that is usually 40 px across does not need the second one.
 *
 * The mesh is scaled AFTER contouring rather than by scaling the tree: a uniform scale of a distance
 * field is exact, but scaling the tree would also scale every blend width and fillet, which are authored
 * in body units on purpose.
 */
function buildShipMesh() {
  const tree = shipTree();
  const b = bounds(tree);
  const size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) * SHIP_MESH.scale;
  const res = Math.min(
    SHIP_MESH.maxResolution,
    resolutionForScreen({
      size,
      distance: SHIP_MESH.viewDistance,
      focal: 1.2,
      diagonalPx: 2500,
      errorPx: SHIP_MESH.errorPx,
    }),
  );
  const mesh = dualContour(compile(tree), { bounds: b, resolution: res });

  // Body units -> world units, in place. A uniform scale leaves normals unchanged, which is the whole
  // reason `scale` is uniform-only.
  const positions = new Float32Array(mesh.positions.length);
  for (let i = 0; i < positions.length; i++) positions[i] = mesh.positions[i] * SHIP_MESH.scale;

  // Through `concatMeshes` even though there is one object, rather than hand-rolling the ids and the
  // empty `extra`. It is the same three lines either way until it is not: the first version wrote them by
  // hand and silently skipped the per-object RANGE, so the ship fell back to a bounding sphere of
  // Infinity and was never culled. One code path produces the ids, the extra data and the bounds
  // together, so a mesh cannot arrive half-described.
  return concatMeshes([{
    positions,
    normals: mesh.normals,
    extra: new Float32Array(mesh.vertexCount * 4),
    indices: mesh.indices,
  }]);
}

export class Renderer {
  constructor(gpu) {
    this.gpu = gpu;
    this.shaders = new ShaderCache(gpu.device);
    this.targets = new Targets(gpu.device);
    this.uniforms = new FrameUniforms(gpu.device);
    this.profiler = new Profiler(gpu.device, { enabled: gpu.caps.timestamps });
    this.camera = new Camera();
    this.ship = new Ship();
    this.contrail = new Contrail(gpu.device);
    this.railgun = new Railgun(gpu.device);
    this.aurora = new Aurora(gpu.device);

    this.frameIndex = 0;
    this.dynres = new DynamicRes();
    this._warnedNoTimestamps = false;
    this._fireSlot = -1;
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
    };

    this.passes = {
      scene: new ScenePass(gpu, this.targets, this.shaders),
      taa: new TaaPass(gpu, this.targets, this.shaders),
      embers: new EmberPass(gpu, this.targets, this.shaders),
      bloom: new BloomPass(gpu, this.targets, this.shaders),
      flare: new LensFlarePass(gpu, this.targets, this.shaders),
      composite: new CompositePass(gpu, this.targets, this.shaders),
      debugview: new DebugViewPass(gpu, this.targets, this.shaders),
      // THE SHIP, meshed from its SDF. Drawn after the rings and appending rather than clearing, which
      // is what the solid layer was always meant to support - the pass header promised that another
      // kind of solid would be a draw rather than another target, and this is that promise being cashed.
      shipmesh: new SolidMeshPass(gpu, this.targets, this.shaders, {
        label: 'ship-mesh',
        shader: 'shipmesh.wgsl',
        clear: false,
        mesh: () => new Mesh(gpu.device, buildShipMesh(), 'ship'),
        // The hull's sphere rides with the ship. Object-space radius times the mesh scale, centred on
        // the ship's position — the orientation cannot change a sphere.
        worldSphere: (_id, r) => [
          this.ship.pos[0], this.ship.pos[1], this.ship.pos[2],
          r.radius * SHIP_MESH.scale,
        ],
      }),
      rings: new SolidMeshPass(gpu, this.targets, this.shaders, {
        label: 'rings',
        shader: 'rings.wgsl',
        // Built at init, not here: the three hoops concatenated into one buffer, each vertex tagged
        // with its ring index so one draw covers all of them and the vertex shader fetches each
        // ring's precessing basis. Baked in ring-local space, which is what keeps the motion vectors
        // exact - see meshgen.js.
        mesh: () => new Mesh(gpu.device, concatMeshes(
          Array.from({ length: RINGS.count }, (_, i) => rectTube({
            segments: RINGS.segments, ...ringDims(i),
          }))), 'rings'),
        // A hoop's sphere is centred on the world origin and its radius does not depend on the
        // precessing basis, so this needs nothing the CPU does not already have. The object-space radius
        // from the mesh is already exactly that.
        worldSphere: (_id, r) => [0, 0, 0, r.radius],
      }),
      // Fifteen boxes - five satellites, a bus and two array wings each - as ONE unit cube drawn fifteen
      // times, each instance scaled and placed by the orbital frame its own instance index selects. They
      // were analytic boxes intersected inside the scene march, which every ray in the frame paid for
      // whether or not it went near one; see satmesh.wgsl for the rest of that trade.
      //
      // No `worldSphere`, so nothing is culled: the orbits are evaluated in WGSL and never reach the CPU,
      // and a second copy of them in JavaScript to reject 180 triangles would cost more than it saves.
      satellites: new SolidMeshPass(gpu, this.targets, this.shaders, {
        label: 'satellites',
        shader: 'satmesh.wgsl',
        clear: false,
        mesh: () => new Mesh(gpu.device, concatMeshes([box()]), 'satellites'),
        instances: SATELLITES.count * 3,
      }),
      // Three instanced additive draws that differ only in their data — see AdditivePass. The
      // `source` thunks are called when bind groups are built rather than captured now, so an
      // owner is free to reallocate its buffer.
      contrail: new AdditivePass(gpu, this.targets, this.shaders, {
        label: 'contrail',
        shader: 'contrail.wgsl',
        vertices: (CONTRAIL.samples - 1) * 6,
        instances: 2,                         // one per nacelle, selected by instance index
        source: () => this.contrail.buffer,
      }),
      railgun: new AdditivePass(gpu, this.targets, this.shaders, {
        label: 'railgun',
        shader: 'railgun.wgsl',
        vertices: RAIL.segments * 6,
        instances: RAIL.pool * 2,             // two strands per shot
        source: () => this.railgun.buffer,
      }),
      aurora: new AdditivePass(gpu, this.targets, this.shaders, {
        label: 'aurora',
        shader: 'aurora.wgsl',
        vertices: (AURORA.samples - 1) * 6,
        instances: AURORA.ribbons,
        source: () => this.aurora.buffer,
      }),
    };

    // THE SOLID MESH PASSES, IN DRAW ORDER, as one list.
    //
    // Three things need this order and this membership: init, recording, and the wireframe flag. Listing
    // them once means adding a fourth generated mesh cannot half-land — which it could when the order
    // was spelled out three times, and did: the first solid to draw is the one that CLEARS the layer, so
    // getting the list right in one place and wrong in another is a cleared buffer, not an error.
    this.solidPasses = [this.passes.rings, this.passes.shipmesh, this.passes.satellites];
  }

  async init() {
    const d = this.gpu.device;
    // One layout for group 0, shared by every pass, so the frame bind group is
    // interchangeable across pipelines and never has to be rebuilt per pass.
    this.frameBGL = FrameUniforms.bindGroupLayout(d);
    this.frameBG = this.uniforms.bindGroup(this.frameBGL);

    await Promise.all([
      this.passes.scene.init(this.frameBGL),
      this.passes.taa.init(this.frameBGL),
      this.passes.embers.init(this.frameBGL),
      this.passes.bloom.init(this.frameBGL),
      this.passes.flare.init(this.frameBGL),
      this.passes.composite.init(this.frameBGL, this.gpu.format),
      this.passes.debugview.init(this.frameBGL, this.gpu.format, wgslDefines()),
      ...this.solidPasses.map((m) => m.init(this.frameBGL, wgslDefines())),
      this.passes.contrail.init(this.frameBGL, wgslDefines()),
      this.passes.railgun.init(this.frameBGL, wgslDefines()),
      this.passes.aurora.init(this.frameBGL, wgslDefines()),
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

    // Projection first: it depends only on the viewport, and the camera folds it
    // into the view matrix it is about to build.
    this.ship.update(dt, input.cmd);
    this.contrail.update(dt, this.ship);
    // AUTO-FIRE while the ship is still cruising, so there is something to look at before anyone
    // touches a key. A one-frame PULSE per period, not a held boolean: the rail gun deliberately
    // triggers on a rising edge — holding it would fire once and then never again, which is the
    // behaviour its own comment exists to explain.
    let fire = !!input.cmd.fire;
    if (!this.ship.flown) {
      const slot = Math.floor(time / SHIP.autoFireEvery);
      if (slot !== this._fireSlot) { this._fireSlot = slot; fire = true; }
    }
    this.railgun.update(time, this.ship, fire);
    this.aurora.update(dt, time);
    this.camera.setViewport(t.width, t.height);
    this.camera.update(time, dt, input, this.ship);

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
    st.ship = this.ship;
    st.taa = TEMPORAL;      // read live, so console tweaks take effect immediately
    st.march = MARCH;
    st.probe = PROBE;
    st.renderScale = QUALITY.renderScale;
    st.grade = FILM;
    st.glow = GLOW;
    st.flareStrength = FLARE.strength;
    st.aurora = AURORA;
    st.volume = VOLUME;
    st.auroraPhase = this.aurora.emitPhase;
    this.uniforms.write(st);

    const encoder = this.gpu.device.createCommandEncoder({ label: 'frame' });
    this.profiler.beginFrame();
    const p = this.profiler;
    const { scene, taa, embers, bloom, flare, composite } = this.passes;

    // DEBUG GROUPS around the phases, not around every pass.
    //
    // Each pass already labels its own render pass, which is what a capture lists; what a capture
    // cannot infer is the STRUCTURE - which passes belong to the same phase, and therefore which of
    // them a change should be expected to move. These four names are the frame graph's own phases, so
    // a capture reads the way the file above documents it. Free when no tool is attached.
    // Frustum planes once per frame, from the same matrix the vertex stages use, into a buffer that is
    // reused rather than reallocated.
    extractFrustum(this.camera.viewProj, this._frustum);

    encoder.pushDebugGroup('scene');
    scene.record(encoder, this.frameBG, p);
    // Solids BEFORE taa: they carry exact motion vectors, so TAA can accumulate
    // them, which is what anti-aliases their silhouettes and puts them in the bloom.
    // The wireframe view swaps every solid mesh onto its line-list pipeline. Built on demand the first
    // time the view is selected, and the flag only takes effect once it exists, so selecting it never
    // stalls the frame waiting on a pipeline compile.
    const view = this.debugView >= 0 ? VIEWS[this.debugView] : null;
    const wire = !!view?.wireframe;
    for (const m of this.solidPasses) {
      m.wireframe = wire;
      if (wire) m.prepareWireframe(this.frameBGL, wgslDefines());
    }
    for (const m of this.solidPasses) m.record(encoder, this.frameBG, p, this._frustum);
    // After the rings, into the same layer and the same depth buffer, so the two occlude each other by
    // hardware depth rather than by anyone sorting them.
    encoder.popDebugGroup();

    encoder.pushDebugGroup('resolve');
    taa.record(encoder, this.frameBG, p);
    encoder.popDebugGroup();

    encoder.pushDebugGroup('additive');
    embers.simulate(encoder, this.frameBG, p);
    embers.record(encoder, this.frameBG, p);
    // Into the same additive target, right after the particles.
    this.passes.contrail.record(encoder, this.frameBG, p);
    this.passes.railgun.record(encoder, this.frameBG, p);
    this.passes.aurora.record(encoder, this.frameBG, p);
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
    this.contrail.destroy();
    this.railgun.destroy();
    this.aurora.destroy();
    this.passes.embers.destroy?.();
    this.targets.destroy();
    this.profiler.destroy();
  }
}

