/**
 * Composite: assemble the layers and apply the film response, straight into the
 * swapchain. The only pass that knows about display encoding.
 */

import { wgslDefines } from '../scene/tuning.js';

export class CompositePass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
  }

  async init(frameBGL, surfaceFormat) {
    const d = this.gpu.device;

    this.bgl = d.createBindGroupLayout({
      label: 'composite-bgl',
      entries: [
        // Scene and embers are at RENDER resolution while this pass runs at
        // display resolution, so both are filtered on the way up — declared
        // filterable, and the shader still point-samples the scene's alpha
        // separately because that channel is a depth tag, not a colour.
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const mod = await this.shaders.module('composite.wgsl', wgslDefines());
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'composite',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: surfaceFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  #sync(bloomView, flareView) {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    this.bindGroups = [0, 1].map((i) => d.createBindGroup({
      label: `composite-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: t.accum[i].createView() },
        { binding: 1, resource: t.ember.createView() },
        { binding: 2, resource: bloomView },
        { binding: 3, resource: flareView },
        { binding: 4, resource: t.linear },
      ],
    }));
  }

  record(encoder, frameBG, surfaceView, bloomView, flareView, profiler) {
    this.#sync(bloomView, flareView);
    const t = this.targets;

    const pass = encoder.beginRenderPass({
      label: 'composite',
      colorAttachments: [{
        view: surfaceView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      ...profiler.scope('composite'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[t.accumIndex]);
    pass.draw(3);
    pass.end();
  }
}
