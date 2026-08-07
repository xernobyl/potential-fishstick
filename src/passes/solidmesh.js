import { MESH_VERTEX_LAYOUT } from '../core/mesh.js';
import { sphereVisible } from '../core/frustum.js';

/**
 * The project's one pass for RASTERISED OPAQUE geometry: a generated mesh into the solid target.
 *
 * It used to be the rings pass, and the rename is the point. It now takes a mesh and a shader, so a
 * second generated shape is `new SolidMeshPass(...)` with a different mesh and a different fragment
 * stage - not another pass, another target, another binding and another branch in the composite. The
 * vertex front end those shaders share lives in `mesh_vertex.wgsl`; the geometry comes from
 * `meshgen.js` through `core/mesh.js`.
 *
 * It draws into `targets.solid`, whose alpha carries linear view distance, and the
 * composite resolves that against the raymarched body by comparing distances. The
 * name is deliberate: this is the second feature to need exactly this plumbing (a
 * cube lattice was the first), so the target and the resolve are generic and adding
 * another kind of solid means adding a draw here rather than another target, another
 * binding and another branch in the composite.
 *
 * A real depth attachment, because self-occlusion — a ring passing behind itself and
 * behind its neighbours — is what hardware depth is for, and the alternative is
 * sorting geometry every frame.
 *
 * Deliberately NOT in the accumulation buffer: these move every frame and
 * reprojection assumes static geometry. Being rasterised they need no accumulation
 * to be clean, so staying out costs nothing.
 */

export class SolidMeshPass {
  /**
   * @param {object} spec
   * @param {string} spec.label   pass, pipeline and profiler-scope name
   * @param {string} spec.shader  WGSL module exposing `vs` and `fs`
   * @param {() => import('../core/mesh.js').Mesh} spec.mesh  resolved at init, so the caller can
   *        build the geometry lazily rather than before the device exists
   * @param {() => (object|object[])} spec.mesh  one mesh, or an LOD chain finest-first. A chain makes
   *        `pass.lod` meaningful; see core/lod.js for how the level is chosen.
   * @param {number|((range:object)=>number)} [spec.instances]  draw this many times, the vertex stage
   *        placing each from `@builtin(instance_index)`. A FUNCTION when the ranges need different
   *        counts: a satellite is one bus and two array wings, so the bus range is drawn once per
   *        satellite and the panel range twice.
   * @param {(id: number, range: object) => number[]} [spec.worldSphere]  where this object's bounding
   *        sphere is in the world, as [x, y, z, r]. Supplying it is what opts an object into frustum
   *        culling: only the caller knows the transform, and some transforms (the satellites' orbits)
   *        live in WGSL and never reach the CPU at all.
   */
  constructor(gpu, targets, shaders, spec) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.spec = spec;
    this.generation = -1;
    this.pipeline = null;
  }

  async init(frameBGL, defines) {
    const d = this.gpu.device;
    // The spec may hand back one mesh or a chain of them, finest first. One is the common case and it
    // becomes a chain of length one, so nothing downstream needs to know which it got.
    const built = this.spec.mesh();
    this.meshes = Array.isArray(built) ? built : [built];
    this.lod = 0;
    this.mesh = this.meshes[0];
    const module = await this.shaders.module(this.spec.shader, defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: this.spec.label,
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL] }),
      vertex: { module, entryPoint: 'vs', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [
          { format: 'rgba16float' },   // colour, alpha = linear view distance
          { format: 'rgba16float' },   // xy pixel delta, z previous view distance
        ],
      },
      // BACKFACE CULLING, which the geometry now earns.
      //
      // This used to be 'none', on the grounds that a ring is thin and its inner face is legitimately
      // visible through the hoop. That was true of the shape and false about the surface: a
      // rectangular-section tube is CLOSED, so what you see looking through the hoop is the tube's far
      // outer wall, not its inside. Culling removes only the faces pointing away from the eye, which
      // are behind an opaque wall in every case.
      //
      // It was also the only thing hiding two winding bugs - the ring mesh's, and four inverted
      // triangles in the dual contourer - both of which are now fixed and asserted rather than
      // tolerated. Roughly half the fragments of a closed mesh are back faces, so this is the cheapest
      // fragment saving available, and from here on a winding mistake is visible instead of latent.
      // `frontFace` is spelled out rather than left to the default. It IS 'ccw' by default, so this
      // changes nothing — but back-face culling only removes the right half of the geometry if the
      // generators wind counter-clockwise seen from outside, and that is asserted in dev/meshgen.mjs
      // and dev/octree.mjs. Writing it here puts the convention next to the thing that depends on it
      // instead of in the spec.
      primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'back' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        // GREATER, because the projection is reverse-Z: the near plane is at 1 and distance
        // tends to 0, so nearer is LARGER. This and the 0.0 clear below have to move together
        // with the matrix — leave either behind and the rings occlude themselves inside out,
        // which reads as the far side of a hoop drawing over the near side.
        depthCompare: 'greater',
      },
    });
  }

  /**
   * The wireframe pipeline, built on first use.
   *
   * SAME SHADERS, SAME LAYOUT, only the topology differs — so the wireframe shows the real material at
   * the real vertices, and a mesh whose vertex stage is wrong looks wrong here in exactly the way it
   * looks wrong shaded. A dedicated flat-colour wire shader would have hidden that, which is the
   * opposite of what a debug view is for.
   *
   * Depth writes OFF, and since no wireframe pass writes depth either, the buffer stays at its clear
   * value and every line passes the `greater-equal` test. The view is therefore X-RAY: you see the far
   * side of a hoop through the near side. That is deliberate and it is most of the value — a wireframe
   * you cannot see through tells you about the silhouette, which the shaded view already showed; one
   * you can see through tells you about the tessellation, which nothing else does.
   */
  async #wirePipelineFor(frameBGL, defines) {
    if (this.wirePipeline) return this.wirePipeline;
    const d = this.gpu.device;
    const module = await this.shaders.module(this.spec.shader, defines);
    this.wirePipeline = await d.createRenderPipelineAsync({
      label: `${this.spec.label}-wire`,
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL] }),
      vertex: { module, entryPoint: 'vs', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format: 'rgba16float' }, { format: 'rgba16float' }],
      },
      // No culling: a line has no facing, and half a wireframe is not a wireframe.
      primitive: { topology: 'line-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'greater-equal',
      },
    });
    return this.wirePipeline;
  }

  /** Build the wireframe pipeline ahead of the frame that needs it, so the view does not stutter on. */
  prepareWireframe(frameBGL, defines) {
    return this.#wirePipelineFor(frameBGL, defines);
  }

  #sync() {
    if (this.generation === this.targets.generation) return;
    this.generation = this.targets.generation;
    this.colourView = this.targets.solid.createView();
    this.motionView = this.targets.motion.createView();
    this.depthView = this.targets.solidDepth.createView();
  }

  /**
   * Release the GPU buffers this pass owns.
   *
   * A pass owns its meshes, so it is the one that can free them. Without this `renderer.destroy()`
   * looked complete while leaving every vertex, index and wireframe-edge buffer behind — the ship's
   * LOD chain alone is four of each. Pipelines and bind groups are not destroyable in WebGPU and are
   * collected with the device, so meshes are the whole of it.
   */
  destroy() {
    for (const m of this.meshes ?? []) m.destroy();
    this.meshes = [];
    this.mesh = null;
  }

  /**
   * @param {Float32Array} [planes] frustum planes; without them nothing is culled
   */
  record(encoder, frameBG, profiler, planes) {
    if (!this.pipeline) return;
    this.#sync();
    // FIRST SOLID CLEARS, THE REST LOAD. Two passes both clearing would wipe the first, and the target
    // is a shared layer rather than one pass's private buffer — which is what the file header promised
    // when it said another kind of solid means adding a draw rather than another target.
    const load = this.spec.clear === false;
    const pass = encoder.beginRenderPass({
      label: this.spec.label,
      colorAttachments: [
        {
          view: this.colourView,
          // Alpha 0 is the "nothing here" sentinel downstream.
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: load ? 'load' : 'clear',
          storeOp: 'store',
        },
        {
          view: this.motionView,
          // LOAD, not clear. The scene pass has already filled this with the sentinel
          // and with the ship's motion; clearing here would wipe the ship's.
          loadOp: 'load',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 0.0,          // reverse-Z: nothing is further than 0
        depthLoadOp: load ? 'load' : 'clear',
        // STORE, not discard, once more than one pass draws solids: the second needs the first's
        // depth to occlude against. Discarding was correct while the rings were alone.
        depthStoreOp: 'store',
      },
      ...profiler.scope(this.spec.label),
    });
    const spread = this.spec.instances ?? 1;
    const instancesFor = typeof spread === 'function' ? spread : () => spread;
    // The level chosen for this frame. Set by the renderer, which owns the camera; the pass only has to
    // honour it. Clamped rather than trusted, so a stale index from a resize cannot index off the end.
    this.mesh = this.meshes[Math.min(Math.max(this.lod, 0), this.meshes.length - 1)];
    pass.setBindGroup(0, frameBG);

    // WIREFRAME, when the debug view asks for it and its pipeline is ready. Everything is drawn,
    // unculled: the point of the view is to see the triangles, and an object that culling removed is
    // exactly the thing you would want to notice was missing.
    if (this.wireframe && this.wirePipeline) {
      pass.setPipeline(this.wirePipeline);
      this.mesh.drawWireframe(pass, this.gpu.device, instancesFor(this.mesh.ranges[0]));
      this.drawn = this.mesh.ranges.length;
      pass.end();
      return;
    }

    pass.setPipeline(this.pipeline);

    // PER-OBJECT FRUSTUM CULLING, when the caller supplies both the planes and a way to place each
    // object's sphere in the world. Without either it draws everything, which is what a mesh with no
    // meaningful per-object transform wants.
    //
    // The sphere is asked for rather than derived here because only the caller knows the transform: the
    // rings' basis is evaluated in WGSL and never reaches the CPU, but their sphere is origin-centred and
    // rotation-invariant, so the CPU needs no basis at all to place it. That is the payoff of choosing
    // spheres.
    this.mesh.bind(pass);
    const sphereOf = this.spec.worldSphere;
    // Whether to cull is decided by whether a sphere is available, which needs no separate flag and no
    // instancing special case: a mesh drawn many times from GPU-side transforms cannot offer one, so it
    // falls through to drawing everything, which is the right answer for it.
    if (planes && sphereOf) {
      this.drawn = 0;
      for (const r of this.mesh.ranges) {
        const s = sphereOf(r.id, r);
        if (s && !sphereVisible(planes, s[0], s[1], s[2], s[3])) continue;
        this.mesh.drawRange(pass, r, instancesFor(r));
        this.drawn++;
      }
    } else {
      // One draw per range even without culling, because the counts may differ between them.
      this.mesh.bind(pass);
      this.drawn = 0;
      for (const r of this.mesh.ranges) {
        this.mesh.drawRange(pass, r, instancesFor(r));
        this.drawn++;
      }
    }
    pass.end();
  }
}
