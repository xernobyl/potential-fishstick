/**
 * Device / surface bring-up.
 *
 * Owns exactly one thing: getting a usable GPUDevice and a configured canvas
 * context, plus reporting what the adapter can actually do. Nothing in here
 * knows about the scene — passes ask `caps` what is available and adapt.
 */

/** Features we take if offered. Everything here must be optional at runtime. */
const WANTED_FEATURES = [
  'timestamp-query',      // GPU-side pass timings for the profiler
  'shader-f16',           // half precision in the heavy inner loops
  'subgroups',            // wave-level reductions (tile cull)
  'indirect-first-instance',
  'float32-filterable',
];

export class Gpu {
  constructor({ device, adapter, context, canvas, format, caps }) {
    this.device = device;
    this.adapter = adapter;
    this.context = context;
    this.canvas = canvas;
    this.format = format;
    this.caps = caps;
    /** Bumped whenever the drawing surface changes size. Passes watch this. */
    this.generation = 0;
    this.width = 1;
    this.height = 1;
  }

  static async create(canvas, { onLost } = {}) {
    if (!navigator.gpu) {
      throw new Error(
        'WebGPU is not available in this browser.\n' +
        'Chrome/Edge 113+, Safari 18+, or Firefox 141+ on a supported GPU.'
      );
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) throw new Error('No suitable GPU adapter was found.');

    const requiredFeatures = WANTED_FEATURES.filter((f) => adapter.features.has(f));

    // Ask for the headroom this renderer actually wants, but never more than the
    // adapter offers — requesting an unsupported limit rejects device creation.
    const want = {
      maxTextureDimension2D: 8192,
      maxStorageBufferBindingSize: 1 << 28,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
    };
    const requiredLimits = {};
    for (const [k, v] of Object.entries(want)) {
      const avail = adapter.limits[k];
      if (typeof avail === 'number') requiredLimits[k] = Math.min(v, avail);
    }

    const device = await adapter.requestDevice({
      label: 'beep-device',
      requiredFeatures,
      requiredLimits,
    });

    const caps = {
      timestamps: device.features.has('timestamp-query'),
      f16: device.features.has('shader-f16'),
      subgroups: device.features.has('subgroups'),
      indirectFirstInstance: device.features.has('indirect-first-instance'),
      limits: device.limits,
      adapterInfo: adapter.info ?? {},
    };

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not acquire a "webgpu" canvas context.');

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: 'opaque',              // no compositing work we do not need
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const gpu = new Gpu({ device, adapter, context, canvas, format, caps });

    device.lost.then((info) => {
      // 'destroyed' is us tearing down on purpose; anything else is a real fault.
      if (info.reason !== 'destroyed') onLost?.(info);
    });

    return gpu;
  }

  /**
   * Resize the drawing buffer to match CSS size, clamped for sanity.
   * Returns true when the size actually changed, so callers can reallocate.
   *
   * `renderScale` is applied to the *internal* buffers by the target manager,
   * not here: the swapchain always matches the display.
   */
  syncSize(maxWidth = 2560) {
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const cssW = Math.max(1, this.canvas.clientWidth);
    const cssH = Math.max(1, this.canvas.clientHeight);
    const scale = Math.min(dpr, maxWidth / cssW);
    const w = Math.max(4, Math.min(this.caps.limits.maxTextureDimension2D, Math.floor(cssW * scale)));
    const h = Math.max(4, Math.min(this.caps.limits.maxTextureDimension2D, Math.floor(cssH * scale)));
    if (w === this.width && h === this.height) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
    this.generation++;
    return true;
  }

  destroy() {
    this.device?.destroy();
  }
}
