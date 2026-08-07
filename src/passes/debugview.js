/**
 * Put one intermediate buffer on the screen, instead of the composite.
 *
 * A renderer whose whole subject is what the temporal resolve does to an image needs a way to look
 * at its inputs. Everything before this was indirect — the instruments read buffers back and report
 * statistics, which is right for deciding whether something changed but useless for seeing WHERE.
 *
 * It replaces the composite for the frame rather than drawing over it: the point is to see the
 * buffer, not the buffer under a film grade. Every pass upstream still runs, so the frame is the
 * real one and the timings on the HUD stay comparable.
 *
 * Bind groups are rebuilt whenever the target set is reallocated OR the selection changes, and both
 * are cheap because there is one view in play at a time. The accumulation is the exception that has
 * to be handled per parity, as everywhere else in this codebase — see `AdditivePass` for the bug
 * that taught it.
 */

/**
 * The buffers worth looking at, in cycle order.
 *
 *   pick   () => GPUTexture, resolved per frame because a resize replaces the object
 *   mode    matches the MODE_* constants in debugview.wgsl
 */
export const VIEWS = [
  { name: 'scene (pre-resolve)', mode: 0, pick: (t) => t.sceneRaw },
  { name: 'accumulation', mode: 0, pick: (t) => t.accumRead },
  { name: 'depth tag', mode: 2, pick: (t) => t.accumRead },
  { name: 'motion vectors', mode: 1, pick: (t) => t.motion },
  { name: 'additive layer', mode: 0, pick: (t) => t.ember },
  { name: 'solid (rings)', mode: 0, pick: (t) => t.solid },
];

export class DebugViewPass {
  constructor(gpu, targets, shaders) {
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.pipeline = null;
    this.generation = -1;
    this.index = -1;
    this.bindGroup = null;
  }

  async init(frameBGL, surfaceFormat, defines) {
    const d = this.gpu.device;
    this.bgl = d.createBindGroupLayout({
      label: 'debugview-bgl',
      entries: [
        // Filterable: this is a viewer, and a linear tap is what makes a half-resolution buffer
        // legible when it is stretched to the window.
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const module = await this.shaders.module('debugview.wgsl', defines);
    this.pipeline = await d.createRenderPipelineAsync({
      label: 'debugview',
      layout: d.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: surfaceFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  #sync(index) {
    const t = this.targets;
    // The accumulation ping-pongs, so its parity is part of the identity of what is bound.
    const key = `${index}:${t.accumIndex}`;
    if (this.generation === t.generation && this.key === key) return;
    this.generation = t.generation;
    this.key = key;
    this.bindGroup = this.gpu.device.createBindGroup({
      label: 'debugview-bg',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: VIEWS[index].pick(t).createView() },
        { binding: 1, resource: t.linear },
      ],
    });
  }

  /** @param {number} index into VIEWS */
  record(encoder, frameBG, surfaceView, index, profiler) {
    if (!this.pipeline) return false;
    this.#sync(index);
    const pass = encoder.beginRenderPass({
      label: 'debugview',
      colorAttachments: [{
        view: surfaceView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      ...profiler.scope('debugview'),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frameBG);
    pass.setBindGroup(1, this.bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }
}
