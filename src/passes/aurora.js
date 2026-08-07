/**
 * The aurora pass.
 *
 * Every ribbon in ONE instanced draw — the instance index selects both its slice of the
 * position buffer and its colour from the palette, so a sixth ribbon costs one more instance
 * and nothing else. Same arrangement as the contrail, which is the point: this is that pass
 * with a different source of positions.
 *
 * Draws into the EMBER target rather than allocating its own. That target is already additive,
 * already at render resolution, and already composited downstream of TAA — which is where
 * anything this fast-changing has to live, because reprojection assumes static geometry and a
 * ribbon that appears from nothing would trail a ghost through the accumulation buffer.
 * `loadOp: 'load'` preserves the particles and the contrail drawn just before it.
 *
 * No depth attachment: additive and translucent, so it has nothing to depth-test against
 * itself, and its occlusion against the scene is per-fragment from the stored linear depth.
 */

import { AURORA } from '../scene/tuning.js';

export class AuroraPass {
  constructor(gpu, targets, shaders, aurora) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.aurora = aurora;
    this.vertexCount = (AURORA.samples - 1) * 6;
    this.ribbons = AURORA.ribbons;
    this.generation = -1;
    this.pipeline = null;
  }

  async init(frameBGL, defines) {
    const d = this.gpu.device;
    this.bgl = d.createBindGroupLayout({
      label: 'aurora-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        {
          binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const module = await this.shaders.module('aurora.wgsl', defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'aurora',
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
    // `targets.generation` only changes on resize, so a single cached bind group would point
    // at the wrong half of it half the time — and it must be the buffer written THIS frame, or
    // the depth used for occlusion is a frame stale. Both bugs happened here in the contrail.
    this.bindGroups = [0, 1].map((i) => this.gpu.device.createBindGroup({
      label: `aurora-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.aurora.buffer } },
        { binding: 1, resource: this.targets.accum[i].createView() },
      ],
    }));
  }

  record(encoder, frameBG, profiler) {
    if (!this.pipeline) return;
    this.#sync();
    const pass = encoder.beginRenderPass({
      label: 'aurora',
      colorAttachments: [{
        view: this.view,
        loadOp: 'load',              // embers and the contrail are already in here
        storeOp: 'store',
      }],
      ...profiler.scope('aurora'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[this.targets.accumIndex]);
    pass.draw(this.vertexCount, this.ribbons);
    pass.end();
  }
}
