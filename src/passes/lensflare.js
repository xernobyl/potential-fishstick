/**
 * Lens flares, at quarter resolution.
 *
 * Source is the bloom pyramid's level 1 — already thresholded and blurred, which
 * is exactly the input a flare wants, so the cost is a few taps rather than a
 * second threshold-and-blur chain of its own.
 */

import { wgslDefines } from '../scene/tuning.js';

export class LensFlarePass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
  }

  async init(frameBGL) {
    const d = this.gpu.device;

    this.bgl = d.createBindGroupLayout({
      label: 'flare-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const mod = await this.shaders.module('lensflare.wgsl', wgslDefines());
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'lensflare',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    // Level 1 rather than 0: one step blurrier, which suppresses the sparkle
    // that would otherwise crawl through the ghosts as the camera drifts.
    // clamped low as well as high: see the note in targets.js about the pyramid
    // never being empty — this refuses to index past either end regardless.
    const srcLevel = Math.max(0, Math.min(1, t.bloom.length - 1));
    this.view = t.flare.createView();
    this.bindGroup = d.createBindGroup({
      label: 'flare-bg',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: t.bloom[srcLevel].texture.createView() },
        { binding: 1, resource: t.linear },
      ],
    });
  }

  get resultView() { return this.view; }

  record(encoder, frameBG, profiler) {
    this.#sync();
    if (!this.targets.bloom.length) return;

    const pass = encoder.beginRenderPass({
      label: 'lensflare',
      colorAttachments: [{
        view: this.view,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      ...profiler.scope('flare'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroup);
    pass.draw(3);
    pass.end();
  }
}
