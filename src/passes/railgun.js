/**
 * The rail gun pass. Same target and blend as the contrail — additive, into the ember
 * buffer, downstream of TAA — so a second beam effect costs no new target and no new
 * composite branch.
 *
 * Instances are shots x strands: the instance index divides by two to pick the shot and
 * takes the remainder as which strand of the helix, so the double spiral needs no extra
 * geometry or bindings.
 */

import { RAIL } from '../scene/tuning.js';

export class RailgunPass {
  constructor(gpu, targets, shaders, railgun) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.railgun = railgun;
    this.vertexCount = RAIL.segments * 6;
    this.instances = RAIL.pool * 2;      // two strands per shot
    this.generation = -1;
    this.pipeline = null;
  }

  async init(frameBGL, defines) {
    const d = this.gpu.device;
    this.bgl = d.createBindGroupLayout({
      label: 'railgun-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        {
          binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const module = await this.shaders.module('railgun.wgsl', defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'railgun',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
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
    // ONE BIND GROUP PER ACCUMULATION PARITY, matching the ember pass.
    //
    // Two bugs lived here. It bound the buffer written LAST frame, so the depth used for
    // occlusion was always one frame stale. And it cached that binding against
    // `targets.generation`, which only changes on resize — while the ping-pong flips
    // every frame, so the cached view pointed at the wrong half of it half the time.
    // Prebuilding both and selecting by parity fixes both and allocates nothing.
    this.bindGroups = [0, 1].map((i) => this.gpu.device.createBindGroup({
      label: `railgun-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.railgun.buffer } },
        { binding: 1, resource: this.targets.accum[i].createView() },
      ],
    }));
  }

  record(encoder, frameBG, profiler) {
    if (!this.pipeline) return;
    this.#sync();
    const pass = encoder.beginRenderPass({
      label: 'railgun',
      colorAttachments: [{ view: this.view, loadOp: 'load', storeOp: 'store' }],
      ...profiler.scope('railgun'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[this.targets.accumIndex]);
    pass.draw(this.vertexCount, this.instances);
    pass.end();
  }
}
