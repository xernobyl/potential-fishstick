/**
 * The rings pass — and more generally, the project's one pass for RASTERISED OPAQUE
 * geometry.
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

/** Four faces per segment, two triangles each. */
const VERTS_PER_SEGMENT = 4 * 6;

export class RingsPass {
  constructor(gpu, targets, shaders, tuning) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.count = tuning.count;
    this.vertexCount = VERTS_PER_SEGMENT * tuning.segments;
    this.generation = -1;
    this.pipeline = null;
  }

  async init(frameBGL, defines) {
    const d = this.gpu.device;
    const module = await this.shaders.module('rings.wgsl', defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'rings',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [
          { format: 'rgba16float' },   // colour, alpha = linear view distance
          { format: 'rgba16float' },   // xy pixel delta, z previous view distance
        ],
      },
      // No culling: a ring is thin and its inner face is legitimately visible
      // through the hoop from most angles, so back faces are real surface here.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
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
    const pass = encoder.beginRenderPass({
      label: 'rings',
      colorAttachments: [
        {
          view: this.colourView,
          // Alpha 0 is the "nothing here" sentinel downstream.
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
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
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',      // never read after this pass
      },
      ...profiler.scope('rings'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.draw(this.vertexCount, this.count);
    pass.end();
  }
}
