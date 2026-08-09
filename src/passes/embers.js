/**
 * Ember particles: bake the sprite once, simulate on the GPU, draw indirect.
 *
 * Nothing about the particles ever touches the CPU after start-up. State lives in
 * a storage buffer, the live set is compacted by the simulation, and the draw's
 * instance count is read straight out of a GPU buffer via drawIndirect — so the
 * CPU never even learns how many are alive.
 */

import { EMBERS, wgslDefines } from '../scene/tuning.js';

/** Ember struct: vec3 pos + f32 size + vec3 tint + f32 alpha, std layout. */
const EMBER_STRIDE = 32;
const DRAW_ARGS_SIZE = 16;      // 4 x u32

export class EmberPass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
    /** All one-shot GPU resources created during init, for the destroy path. */
    this._owned = [];
  }

  async init(frameBGL) {
    const d = this.gpu.device;
    const defines = wgslDefines();

    // ---- buffers ----
    this.emberBuffer = this._own(d.createBuffer({
      label: 'embers',
      size: EMBERS.count * EMBER_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    }));
    this.liveBuffer = this._own(d.createBuffer({
      label: 'ember-live-list',
      size: EMBERS.count * 4,
      usage: GPUBufferUsage.STORAGE,
    }));
    this.drawArgs = this._own(d.createBuffer({
      label: 'ember-draw-args',
      size: DRAW_ARGS_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    }));

    // ---- simulation ----
    this.simBGL = d.createBindGroupLayout({
      label: 'ember-sim-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const simMod = await this.shaders.module('ember_sim.wgsl', defines);
    const simLayout = d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.simBGL] });
    [this.resetPipeline, this.simPipeline] = await Promise.all([
      d.createComputePipelineAsync({
        label: 'ember-sim-reset', layout: simLayout,
        compute: { module: simMod, entryPoint: 'reset' },
      }),
      d.createComputePipelineAsync({
        label: 'ember-sim', layout: simLayout,
        compute: { module: simMod, entryPoint: 'simulate' },
      }),
    ]);
    this.simBG = d.createBindGroup({
      label: 'ember-sim-bg',
      layout: this.simBGL,
      entries: [
        { binding: 0, resource: { buffer: this.emberBuffer } },
        { binding: 1, resource: { buffer: this.liveBuffer } },
        { binding: 2, resource: { buffer: this.drawArgs } },
      ],
    });

    // ---- sprite, baked once ----
    await this.#bakeSprite(defines);

    // ---- draw ----
    this.drawBGL = d.createBindGroupLayout({
      label: 'ember-draw-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const drawMod = await this.shaders.module('ember_draw.wgsl', defines);
    this.drawPipeline = await d.createRenderPipelineAsync({
      label: 'ember-draw',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.drawBGL] }),
      vertex: { module: drawMod, entryPoint: 'vs' },
      fragment: {
        module: drawMod,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  /**
   * Render the sprite into a small mipped texture, once.
   * Every mote shares one appearance, so evaluating its profile per pixel per
   * particle every frame would be waste; this turns it into one bilinear tap,
   * and the mips stop distant motes aliasing.
   */
  async #bakeSprite(defines) {
    const d = this.gpu.device;
    const size = EMBERS.spriteSize;
    const mipCount = Math.floor(Math.log2(size)) + 1;

    this.sprite = this._own(d.createTexture({
      label: 'ember-sprite',
      size: { width: size, height: size },
      format: 'rgba16float',
      mipLevelCount: mipCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    }));

    const mod = await this.shaders.module('ember_sprite.wgsl', defines);
    const bakePipeline = await d.createRenderPipelineAsync({
      label: 'ember-sprite-bake',
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });

    const enc = d.createCommandEncoder({ label: 'ember-sprite-bake' });
    const pass = enc.beginRenderPass({
      label: 'ember-sprite-bake-pass',
      colorAttachments: [{
        view: this.sprite.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.setPipeline(bakePipeline);
    pass.draw(3);
    pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();

    // Mip generation: render each level from the one above with a linear sampler.
    await this.#generateMips(this.sprite, size, mipCount);

    this.spriteView = this.sprite.createView();
    this.spriteSampler = this._own(d.createSampler({
      label: 'ember-sprite-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    }));
  }

  async #generateMips(texture, size, mipCount) {
    const d = this.gpu.device;
    const mod = await this.shaders.module('blit.wgsl', {});
    const pipeline = await d.createRenderPipelineAsync({
      label: 'mip-blit',
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });

    // One-shot resources scoped to this chain; destroyed after the submit completes
    // because they serve no further purpose.
    const owned = [];

    try {
      const sampler = d.createSampler({
        label: 'mip-blit-sampler',
        magFilter: 'linear',
        minFilter: 'linear',
      });
      owned.push(sampler);

      const enc = d.createCommandEncoder({ label: 'mip-chain' });
      for (let m = 1; m < mipCount; m++) {
        const mipView = texture.createView({ baseMipLevel: m, mipLevelCount: 1 });
        const srcView = texture.createView({ baseMipLevel: m - 1, mipLevelCount: 1 });

        const bg = d.createBindGroup({
          label: `mip-blit-bg-${m}`,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: srcView },
            { binding: 1, resource: sampler },
          ],
        });
        owned.push(bg);

        const pass = enc.beginRenderPass({
          label: `mip-blit-pass-${m}`,
          colorAttachments: [{
            view: mipView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }

      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
    } finally {
      for (const r of owned) {
        if (typeof r.destroy === 'function') r.destroy();
      }
    }
  }

  _own(resource) {
    this._owned.push(resource);
    return resource;
  }

  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;

    this.view = t.ember.createView();
    // One per accumulation parity: the fragment shader reads the scene's depth
    // out of whichever accumulation target was written this frame.
    this.drawBG = [0, 1].map((i) => d.createBindGroup({
      label: `ember-draw-bg-${i}`,
      layout: this.drawBGL,
      entries: [
        { binding: 0, resource: { buffer: this.emberBuffer } },
        { binding: 1, resource: { buffer: this.liveBuffer } },
        { binding: 2, resource: this.spriteView },
        { binding: 3, resource: this.spriteSampler },
        { binding: 4, resource: t.accum[i].createView() },
      ],
    }));
  }

  /** Simulate. Must run before the draw, and before anything reads the buffers. */
  simulate(encoder, frameBG, profiler) {
    const pass = encoder.beginComputePass({
      label: 'ember-sim',
      ...profiler.scope('embers'),
    });
    pass.setPipeline(this.resetPipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.simBG);
    pass.dispatchWorkgroups(1);

    pass.setPipeline(this.simPipeline);
    pass.dispatchWorkgroups(Math.ceil(EMBERS.count / 64));
    pass.end();
  }

  /** Draw the billboards into their own additive target. */
  record(encoder, frameBG, profiler) {
    this.#sync();
    const t = this.targets;
    const pass = encoder.beginRenderPass({
      label: 'ember-draw',
      colorAttachments: [{
        view: this.view,
        loadOp: 'clear',          // dynamic: nothing to carry over between frames
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      ...profiler.scope('ember-draw'),
    });
    pass.setPipeline(this.drawPipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.drawBG[t.accumIndex]);
    pass.drawIndirect(this.drawArgs, 0);
    pass.end();
  }

  destroy() {
    for (const r of this._owned) {
      if (r && typeof r.destroy === 'function') r.destroy();
    }
    this._owned.length = 0;
  }
}
