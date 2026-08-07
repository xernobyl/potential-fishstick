/**
 * Glow: a dual-filter (Kawase) pyramid.
 *
 *   prefilter : scene + embers -> level 0, thresholded with a soft knee
 *   down      : 0 -> 1 -> 2 ... halving each time
 *   up        : coarsest -> ... -> 0, ADDITIVELY blended
 *
 * The upsample uses the pipeline's blend state to accumulate, so each coarser
 * level lands on the finer one with no separate combine pass. Level 0 therefore
 * ends up holding the full reconstruction, and the composite reads only that.
 *
 * All three stages share one bind group layout so the pipelines can differ only
 * in their fragment entry point and blend state.
 */

import { GLOW, wgslDefines } from '../scene/tuning.js';

const PARAM_SIZE = 16;   // vec2 texel + f32 radius + f32 weight

export class BloomPass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
    this.paramBuffers = [];
    // Snapshotted rather than read live: these two are baked into the per-level parameter
    // buffers, which are written on resize, so changing them at runtime needs an explicit
    // rebuild. `invalidate()` is that rebuild — the debug panel calls it.
    this.radius = GLOW.radius;
    this.levelWeight = GLOW.levelWeight;
  }

  async init(frameBGL) {
    const d = this.gpu.device;

    this.bgl = d.createBindGroupLayout({
      label: 'bloom-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: PARAM_SIZE } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    const mod = await this.shaders.module('bloom.wgsl', wgslDefines());
    const layout = d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] });
    const target = { format: 'rgba16float' };

    const make = (label, entry, blend) => d.createRenderPipelineAsync({
      label,
      layout,
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: {
        module: mod,
        entryPoint: entry,
        targets: [blend ? { ...target, blend } : target],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Additive, so the upsample accumulates into the finer level in place.
    const additive = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };

    [this.prefilterPipeline, this.downPipeline, this.upPipeline] = await Promise.all([
      make('bloom-prefilter', 'fsPrefilter', null),
      make('bloom-down', 'fsDown', null),
      make('bloom-up', 'fsUp', additive),
    ]);
  }

  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    for (const b of this.paramBuffers) b.destroy();
    this.paramBuffers = [];
    this.views = t.bloom.map((l) => l.texture.createView());

    // A parameter buffer per stage instance. Written once here, never per frame:
    // the values depend only on the target sizes.
    const makeParams = (label, texelW, texelH, radius, weight) => {
      const buf = d.createBuffer({
        label,
        size: PARAM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      d.queue.writeBuffer(buf, 0, new Float32Array([texelW, texelH, radius, weight]));
      this.paramBuffers.push(buf);
      return buf;
    };

    const sceneView = t.accum.map((a) => a.createView());
    this.sceneViews = sceneView;
    const emberView = t.ember.createView();

    // ---- prefilter: scene(+embers) -> level 0 ----
    //
    // The texel size is the ACCUMULATION buffer's, not the render target's. It used to be the
    // latter, which was simply wrong whenever temporal upsampling made the two different sizes —
    // harmless only because nothing read it. It is read now: the prefilter box-averages its
    // whole footprint, and `ratio` tells it how wide that is.
    const ratio = t.accumWidth / t.bloom[0].width;
    this.prefilterBG = [0, 1].map((i) => d.createBindGroup({
      label: `bloom-prefilter-bg-${i}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: sceneView[i] },
        { binding: 1, resource: t.linear },
        {
          binding: 2,
          resource: {
            buffer: makeParams('bloom-pre-params',
              1 / t.accumWidth, 1 / t.accumHeight, ratio, 1),
          },
        },
        { binding: 3, resource: emberView },
      ],
    }));

    // ---- downsample: level i-1 -> level i ----
    this.down = [];
    for (let i = 1; i < t.bloom.length; i++) {
      const src = t.bloom[i - 1];
      this.down.push({
        bindGroup: d.createBindGroup({
          label: `bloom-down-bg-${i}`,
          layout: this.bgl,
          entries: [
            { binding: 0, resource: this.views[i - 1] },
            { binding: 1, resource: t.linear },
            {
              binding: 2,
              resource: {
                buffer: makeParams(`bloom-down-params-${i}`,
                  1 / src.width, 1 / src.height, this.radius, 1),
              },
            },
            // Unused by fsDown, but the layout is shared; bind something valid.
            { binding: 3, resource: emberView },
          ],
        }),
        view: this.views[i],
      });
    }

    // ---- upsample: level i -> level i-1, additive ----
    this.up = [];
    for (let i = t.bloom.length - 1; i >= 1; i--) {
      const src = t.bloom[i];
      this.up.push({
        bindGroup: d.createBindGroup({
          label: `bloom-up-bg-${i}`,
          layout: this.bgl,
          entries: [
            { binding: 0, resource: this.views[i] },
            { binding: 1, resource: t.linear },
            {
              binding: 2,
              resource: {
                buffer: makeParams(`bloom-up-params-${i}`,
                  1 / src.width, 1 / src.height, this.radius, this.levelWeight),
              },
            },
            { binding: 3, resource: emberView },
          ],
        }),
        view: this.views[i - 1],
      });
    }
  }

  /** Level 0 holds the final reconstruction. */
  get resultView() { return this.views[0]; }

  /** Re-read the parameters that live in the per-level uniform buffers. */
  invalidate() {
    this.radius = GLOW.radius;
    this.levelWeight = GLOW.levelWeight;
    this.generation = -1;
  }

  record(encoder, frameBG, profiler) {
    this.#sync();
    const t = this.targets;
    if (!t.bloom.length) return;

    const draw = (label, view, pipeline, bindGroup, load) => {
      const pass = encoder.beginRenderPass({
        label,
        colorAttachments: [{
          view,
          loadOp: load ? 'load' : 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, bindGroup);
      pass.draw(3);
      pass.end();
    };

    const scope = profiler.scope('bloom');

    // Prefilter and downsample can share one profiler scope; splitting every
    // level would exhaust the query set for little insight.
    {
      const pass = encoder.beginRenderPass({
        label: 'bloom-prefilter',
        colorAttachments: [{
          view: this.views[0], loadOp: 'clear', storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
        ...scope,
      });
      pass.setPipeline(this.prefilterPipeline);
      pass.setBindGroup(0, frameBG);
      // accumIndex is the target TAA just WROTE this frame; the swap happens at
      // end of frame, so downstream reads that index, not the other one.
      pass.setBindGroup(1, this.prefilterBG[t.accumIndex]);
      pass.draw(3);
      pass.end();
    }

    for (let i = 0; i < this.down.length; i++) {
      draw(`bloom-down-${i}`, this.down[i].view, this.downPipeline, this.down[i].bindGroup, false);
    }
    for (let i = 0; i < this.up.length; i++) {
      // load, not clear: this is the accumulation.
      draw(`bloom-up-${i}`, this.up[i].view, this.upPipeline, this.up[i].bindGroup, true);
    }
  }
}
