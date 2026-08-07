/**
 * One instanced additive draw into the shared additive target.
 *
 * This replaced three classes — the contrail, the auroras, the rail guns — that were the same
 * ~100 lines three times over. The aurora and contrail files were byte-identical modulo
 * identifiers and TWO lines: the shader name and where the instance count came from. The rail gun
 * differed by nothing structural at all. That is not a family of passes, it is one pass with three
 * parameter sets, and the duplication was actively harmful rather than merely untidy: every fix
 * had to be applied three times, and the per-parity bind-group bug this class documents was
 * originally found in the contrail and then found AGAIN in the aurora, because the copy did not
 * come with the reasoning.
 *
 * What each caller supplies is data, not behaviour:
 *   label        for the pass, the pipeline and the bind groups
 *   shader       the WGSL module; it must expose `vs` and `fs`
 *   vertices     vertices per instance
 *   instances    how many
 *   source       () => GPUBuffer, called at bind-group build time rather than stored, because
 *                the owner may reallocate and the indirection is what keeps that safe
 *
 * Everything the three had in common is here once: the additive blend, the load-not-clear so the
 * layer accumulates, the absent depth attachment, and the per-parity bind groups.
 */

export class AdditivePass {
  /**
   * @param {object} spec {label, shader, vertices, instances, source}
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
    this.bgl = d.createBindGroupLayout({
      label: `${this.spec.label}-bgl`,
      entries: [
        // The geometry's source: positions, shots, whatever the owner keeps.
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        // The resolved scene, for the soft depth occlusion every one of these does. Unfilterable
        // because the alpha channel is a depth TAG and interpolating two tags yields neither.
        {
          binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const module = await this.shaders.module(this.spec.shader, defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: this.spec.label,
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          // Premultiplied additive: the shader folds alpha in, so src is ONE.
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one' },
            alpha: { srcFactor: 'one', dstFactor: 'one' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  #sync() {
    if (this.generation === this.targets.generation) return;
    this.generation = this.targets.generation;
    this.view = this.targets.ember.createView();
    // ONE BIND GROUP PER ACCUMULATION PARITY. The ping-pong flips every frame while
    // `targets.generation` only changes on resize, so a single cached bind group would point at
    // the wrong half of it half the time — and it must be the buffer written THIS frame, or the
    // depth used for occlusion is a frame stale. Both bugs happened here, and then happened again
    // in the copy of this file that no longer exists.
    this.bindGroups = [0, 1].map((i) => this.gpu.device.createBindGroup({
      label: `${this.spec.label}-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.spec.source() } },
        { binding: 1, resource: this.targets.accum[i].createView() },
      ],
    }));
  }

  record(encoder, frameBG, profiler) {
    if (!this.pipeline) return;
    this.#sync();
    const pass = encoder.beginRenderPass({
      label: this.spec.label,
      colorAttachments: [{
        view: this.view,
        // Load, not clear: this target is a LAYER and each of these adds to it.
        loadOp: 'load',
        storeOp: 'store',
      }],
      // No depth attachment. Additive and translucent, so there is nothing to depth-test
      // against itself, and occlusion against the scene is per-fragment from the stored depth.
      ...profiler.scope(this.spec.label),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[this.targets.accumIndex]);
    pass.draw(this.spec.vertices, this.spec.instances);
    pass.end();
  }
}
