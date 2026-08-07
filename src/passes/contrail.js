/**
 * The contrail pass.
 *
 * Two ribbons in ONE instanced draw — the instance index selects which half of the
 * buffer it reads, so a second trail costs no extra binding, pass or upload.
 *
 * Draws into the EMBER target rather than allocating one of its own: that target is
 * already additive, already at render resolution, and already composited downstream of
 * TAA — which is exactly where a fast-changing translucent effect belongs. `loadOp:
 * 'load'` preserves the particles drawn just before it.
 *
 * No depth attachment. The ribbon is translucent and additive, so it has nothing to
 * depth-test against itself, and its occlusion against the scene is done per-fragment
 * from the stored linear depth.
 */

import { CONTRAIL } from '../scene/tuning.js';

export class ContrailPass {
  constructor(gpu, targets, shaders, contrail) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.contrail = contrail;
    this.vertexCount = (CONTRAIL.samples - 1) * 6;
    this.ribbons = 2;                 // one per nacelle, selected by instance index
    this.generation = -1;
    this.pipeline = null;
  }

  async init(frameBGL, defines) {
    const d = this.gpu.device;
    this.bgl = d.createBindGroupLayout({
      label: 'contrail-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        {
          binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const module = await this.shaders.module('contrail.wgsl', defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'contrail',
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
      // The ribbon is two-sided by construction.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  #sync() {
    if (this.generation === this.targets.generation) return;
    this.generation = this.targets.generation;
    this.view = this.targets.ember.createView();
    // ONE BIND GROUP PER ACCUMULATION PARITY, matching the ember pass.
    //
    // Two bugs lived here. It bound the buffer written LAST frame, so the depth used for
    // occlusion was always one frame stale. And it cached that binding against
    // `targets.generation`, which only changes on resize — while the ping-pong flips
    // every frame, so the cached view pointed at the wrong half of it half the time.
    // Prebuilding both and selecting by parity fixes both and allocates nothing.
    this.bindGroups = [0, 1].map((i) => this.gpu.device.createBindGroup({
      label: `contrail-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.contrail.buffer } },
        { binding: 1, resource: this.targets.accum[i].createView() },
      ],
    }));
  }

  record(encoder, frameBG, profiler) {
    if (!this.pipeline) return;
    this.#sync();
    const pass = encoder.beginRenderPass({
      label: 'contrail',
      colorAttachments: [{
        view: this.view,
        // LOAD: the embers were drawn into this a moment ago.
        loadOp: 'load',
        storeOp: 'store',
      }],
      ...profiler.scope('contrail'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[this.targets.accumIndex]);
    pass.draw(this.vertexCount, this.ribbons);
    pass.end();
  }
}
