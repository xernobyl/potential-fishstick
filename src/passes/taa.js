/**
 * Temporal accumulation.
 *
 * Reads the raw sample plus last frame's accumulation, writes this frame's.
 * The ping-pong lives in Targets; this pass just needs two bind groups, one per
 * parity, so nothing is rebuilt per frame.
 */

import { wgslDefines } from '../scene/tuning.js';

export class TaaPass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
  }

  async init(frameBGL) {
    const d = this.gpu.device;

    this.bgl = d.createBindGroupLayout({
      label: 'taa-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        {
          binding: 2, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
        },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        // Accumulated weight: last frame's in, this frame's out.
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        {
          binding: 7, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' },
        },
      ],
    });

    const mod = await this.shaders.module('taa.wgsl', wgslDefines());
    this.pipeline = await d.createComputePipelineAsync({
      label: 'taa',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      compute: { module: mod, entryPoint: 'main' },
    });
  }

  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    // One bind group per ping-pong parity: index i writes accum[i] and reads
    // accum[1-i]. Prebuilt so a frame never allocates.
    this.bindGroups = [0, 1].map((i) => d.createBindGroup({
      label: `taa-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: t.sceneRaw.createView() },
        { binding: 1, resource: t.accum[1 - i].createView() },
        { binding: 2, resource: t.accum[i].createView() },
        { binding: 3, resource: t.linear },
        { binding: 4, resource: t.solid.createView() },
        { binding: 5, resource: t.motion.createView() },
        { binding: 6, resource: t.accumWeight[1 - i].createView() },
        { binding: 7, resource: t.accumWeight[i].createView() },
      ],
    }));
  }

  record(encoder, frameBG, profiler) {
    this.#sync();
    const t = this.targets;
    const pass = encoder.beginComputePass({ label: 'taa', ...profiler.scope('taa') });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroups[t.accumIndex]);
    // Over the ACCUMULATION grid, not the render grid: with temporal upsampling there is
    // one invocation per OUTPUT pixel, each gathering the render-res samples that landed
    // near it. Dispatching over the render grid instead would leave three quarters of the
    // history untouched.
    pass.dispatchWorkgroups(Math.ceil(t.accumWidth / 8), Math.ceil(t.accumHeight / 8));
    pass.end();
  }
}
