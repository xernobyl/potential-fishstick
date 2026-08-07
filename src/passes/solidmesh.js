import { MESH_VERTEX_LAYOUT } from '../core/mesh.js';

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
    this.mesh = this.spec.mesh();
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
      primitive: { topology: 'triangle-list', cullMode: 'back' },
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

  #sync() {
    if (this.generation === this.targets.generation) return;
    this.generation = this.targets.generation;
    this.colourView = this.targets.solid.createView();
    this.motionView = this.targets.motion.createView();
    this.depthView = this.targets.solidDepth.createView();
  }

  record(encoder, frameBG, profiler) {
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
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    this.mesh.draw(pass);
    pass.end();
  }
}
