/**
 * The scene pass: tile classification, then the raymarch.
 *
 * Both are compute. The tile pass writes one flag per 8x8 tile; the raymarch
 * reads it and skips the march for tiles that cannot reach the body. Because the
 * flag is shared by the whole workgroup, every invocation in it takes the same
 * branch, so there is no divergence cost either.
 *
 * A coarse world-space distance volume used to live here as a third stage, to let the
 * march stride through empty space without evaluating the field. It was correct — verified
 * to leave the image unchanged — and it did not pay. See the note in tuning.js on why the
 * march turned out not to be step-count-bound.
 */

import { QUALITY, wgslDefines } from '../scene/tuning.js';

export class ScenePass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
  }

  async init(frameBGL) {
    const d = this.gpu.device;
    const defines = wgslDefines();

    this.cullBGL = d.createBindGroupLayout({
      label: 'tilecull-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.marchBGL = d.createBindGroupLayout({
      label: 'raymarch-bgl',
      entries: [
        {
          binding: 0, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
        },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        {
          binding: 2, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float' },
        },
      ],
    });

    const [cullMod, marchMod] = await Promise.all([
      this.shaders.module('tilecull.wgsl', defines),
      this.shaders.module('raymarch.wgsl', defines),
    ]);

    // Async creation so a long shader compile cannot stall the first frame.
    [this.cullPipeline, this.marchPipeline] = await Promise.all([
      d.createComputePipelineAsync({
        label: 'tilecull',
        layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.cullBGL] }),
        compute: { module: cullMod, entryPoint: 'main' },
      }),
      d.createComputePipelineAsync({
        label: 'raymarch',
        layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.marchBGL] }),
        compute: { module: marchMod, entryPoint: 'main' },
      }),
    ]);
  }

  /** Bind groups hold concrete views, so they must be rebuilt after a resize. */
  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    this.cullBG = d.createBindGroup({
      label: 'tilecull-bg',
      layout: this.cullBGL,
      entries: [{ binding: 0, resource: { buffer: t.tileFlags } }],
    });
    this.marchBG = d.createBindGroup({
      label: 'raymarch-bg',
      layout: this.marchBGL,
      entries: [
        { binding: 0, resource: t.sceneRaw.createView() },
        { binding: 1, resource: { buffer: t.tileFlags } },
        { binding: 2, resource: t.motion.createView() },
      ],
    });
  }

  record(encoder, frameBG, profiler) {
    this.#sync();
    const t = this.targets;

    {
      const pass = encoder.beginComputePass({
        label: 'tilecull',
        ...profiler.scope('tilecull'),
      });
      pass.setPipeline(this.cullPipeline);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this.cullBG);
      // One thread per tile, in 8x8 thread groups.
      pass.dispatchWorkgroups(Math.ceil(t.tilesX / 8), Math.ceil(t.tilesY / 8));
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({
        label: 'raymarch',
        ...profiler.scope('raymarch'),
      });
      pass.setPipeline(this.marchPipeline);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this.marchBG);
      // One workgroup per tile: the workgroup size and QUALITY.tile must agree.
      pass.dispatchWorkgroups(
        Math.ceil(t.width / QUALITY.tile),
        Math.ceil(t.height / QUALITY.tile),
      );
      pass.end();
    }
  }
}
