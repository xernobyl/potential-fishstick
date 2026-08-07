/**
 * Deterministic benchmark, driven WITHOUT requestAnimationFrame.
 *
 * The reason this exists rather than reading the on-screen HUD: rAF is pinned to
 * the display and browsers throttle or stop it entirely for a hidden or
 * backgrounded canvas, so the HUD can sit at its first frame forever while the
 * renderer is perfectly healthy. Worse, when the HUD *does* tick, its numbers are
 * exponentially smoothed and contended with whatever else is on the GPU, which is
 * how the same pass came out 60x apart across two readings here. Neither state is
 * something you can optimise against.
 *
 * So: drive `renderer.frame()` from a plain loop, on a SYNTHETIC clock. That
 * sidesteps rAF completely and makes the camera path identical run to run, which
 * is what makes two builds comparable at all. Report the distribution — median
 * and p95 — not a smoothed scalar.
 *
 * `stability` mode freezes the clock instead of advancing it and reads the
 * accumulation buffer back over the wire. With the camera and the animation both held
 * still, every remaining frame-to-frame difference in that buffer is the sampler: the
 * pixel jitter, the aperture offset and the accumulation converging. That is the
 * residual that reads as shimmer.
 *
 * It measures the BUFFER, not the canvas, and that distinction is the whole point —
 * see the note on FrameSampler below.
 */


/**
 * Reads a centred patch of the accumulation buffer back to the CPU.
 *
 * This exists because the obvious approach does not work. Scraping the canvas with
 * `drawImage` reads whatever the compositor happens to be holding, which is not
 * synchronised with the frames being driven — the same frozen scene measured 2.757 and
 * then 1.416, and turning depth of field OFF made the number go UP, which is
 * physically impossible. A measurement that disagrees with itself is worse than none,
 * because it invites conclusions.
 *
 * Copying the texture and awaiting the queue is deterministic: the numbers are the
 * actual contents of the buffer TAA wrote, in linear HDR, before any tone mapping or
 * grain could mask or invent variation.
 */
const PATCH_MAX = 256;

class FrameSampler {
  /**
   * @param {number} [wantW] explicit patch width. Needed for resolution-matched comparison:
   *   two configs whose accumulation buffers differ in size must read patches covering the
   *   same SCREEN REGION, which means different pixel counts. Must be a multiple of 32.
   */
  constructor(device, texW, texH, wantW, wantH) {
    // bytesPerRow must be a multiple of 256, and rgba16float is 8 bytes per texel, so
    // the width has to be a multiple of 32.
    const capW = wantW ?? Math.min(PATCH_MAX, texW);
    const w = Math.max(32, Math.min(capW, texW) - (Math.min(capW, texW) % 32));
    this.w = w;
    this.h = Math.max(1, Math.min(wantH ?? 144, texH));
    this.bytesPerRow = this.w * 8;
    this.device = device;
    this.buffer = device.createBuffer({
      label: 'stability-readback',
      size: this.bytesPerRow * this.h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  record(encoder, texture, texW, texH) {
    encoder.copyTextureToBuffer(
      {
        texture,
        origin: {
          x: Math.max(0, Math.floor((texW - this.w) / 2)),
          y: Math.max(0, Math.floor((texH - this.h) / 2)),
        },
      },
      { buffer: this.buffer, bytesPerRow: this.bytesPerRow, rowsPerImage: this.h },
      { width: this.w, height: this.h },
    );
  }

  async read() {
    await this.buffer.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(this.buffer.getMappedRange()).slice();
    this.buffer.unmap();
    return raw;
  }

  destroy() { this.buffer.destroy(); }
}

/** IEEE half -> float. Written out because Float16Array is not universally available. */
function halfToFloat(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const man = bits & 0x03ff;
  if (exp === 0) return sign * man * 5.960464477539063e-8;
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * (man + 1024) * Math.pow(2, exp - 25);
}

/**
 * Mean absolute RGB difference between two patches, and the mean magnitude it should
 * be judged against. Alpha is skipped: it is a depth TAG, not a colour, and a change
 * in it is a change of surface rather than of shading.
 */
function patchDelta(a, b) {
  let diff = 0;
  let mag = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const x = halfToFloat(a[i + c]);
      const y = halfToFloat(b[i + c]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      diff += Math.abs(x - y);
      mag += Math.abs(x);
      n++;
    }
  }
  return { diff: n ? diff / n : 0, mag: n ? mag / n : 0 };
}

/** Decode a half-float RGBA patch to f32, so it can be resampled arithmetically. */
function decodeHalf(patch) {
  const out = new Float32Array(patch.length);
  for (let i = 0; i < patch.length; i++) out[i] = halfToFloat(patch[i]);
  return out;
}

/**
 * Bilinear RGBA resample. Exists for ONE purpose: making two configurations comparable when
 * their accumulation buffers are different sizes.
 *
 * Sharpness is a per-pixel measure, so it cannot be compared across resolutions — at twice the
 * linear resolution a fixed image has half the per-pixel gradient, and that alone would make
 * an upsampling path look 2x softer than it is. Bringing both to a common grid first is what
 * makes the comparison mean anything, and it is deliberately the SAME operation the composite
 * would apply when upscaling the low-resolution path for display.
 */
function resampleRGBA(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.max(0, (y + 0.5) * (sh / dh) - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(sh - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.max(0, (x + 0.5) * (sw / dw) - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(sw - 1, x0 + 1), fx = sx - x0;
      const i00 = (y0 * sw + x0) * 4, i10 = (y0 * sw + x1) * 4;
      const i01 = (y1 * sw + x0) * 4, i11 = (y1 * sw + x1) * 4;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = src[i00 + c] + (src[i10 + c] - src[i00 + c]) * fx;
        const b = src[i01 + c] + (src[i11 + c] - src[i01 + c]) * fx;
        out[o + c] = a + (b - a) * fy;
      }
    }
  }
  return out;
}

/** Sharpness of an already-decoded f32 RGBA patch. */
function sharpnessOf(f, w, h) {
  const luma = (i) => 0.2126 * f[i] + 0.7152 * f[i + 1] + 0.0722 * f[i + 2];
  let grad = 0, mag = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const c = luma(i);
      if (!Number.isFinite(c)) continue;
      const gx = luma(i + 4) - luma(i - 4);
      const gy = luma(i + w * 4) - luma(i - w * 4);
      if (!Number.isFinite(gx) || !Number.isFinite(gy)) continue;
      grad += Math.abs(gx) + Math.abs(gy);
      mag += c;
      n++;
    }
  }
  return (!n || mag <= 1e-6) ? 0 : grad / mag;
}

/**
 * Retained high-frequency detail: mean central-difference gradient of luma, relative to
 * mean luma. Higher is sharper.
 *
 * This is the number that says whether a history reconstruction filter is doing its job.
 *
 * Note what it is NOT sensitive to, because the obvious guess is wrong: freezing the
 * camera does not neutralise the filter. Reprojection starts from the JITTERED sample's
 * hit point, so the history fetch sits at a fractional offset every frame whether or not
 * anything is moving — the jitter supplies the offset by itself. Measured: the converged
 * reference is 0.3629 with Catmull-Rom and 0.3196 with bilinear, on an identical frozen
 * scene. A filter comparison is therefore valid in both the moving and the static case.
 *
 * Relative to brightness, so a dim frame and a bright one are comparable. Noise inflates
 * it, which is why the absolute value means little — but noise inflates it EQUALLY for
 * two filters sampling the same sequence, so the comparison between them still holds.
 */
function patchSharpness(patch, w, h) {
  return sharpnessOf(decodeHalf(patch), w, h);
}

/**
 * Mean of the red channel over a patch. Used with the field-evaluation probe, where the
 * raymarch writes its per-pixel work count as radiance: read this back, multiply by the
 * probe divisor, and the answer is the mean number of field evaluations per pixel.
 *
 * Exists because three march optimisations in a row measured as noise, which meant the
 * cost model they were aimed at was wrong. Counting beats reasoning about it.
 */
function patchMeanRed(patch) {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < patch.length; i += 4) {
    const v = halfToFloat(patch[i]);
    if (!Number.isFinite(v)) continue;
    sum += v;
    n++;
  }
  return n ? sum / n : 0;
}

/**
 * Mean field evaluations per pixel, counted in the shader.
 *
 * Renders ONE frame with the probe on (not converged — the count is deterministic for a
 * given camera, and accumulating it would average it with whatever the history holds).
 */
export async function fieldEvalCount(renderer, gpu, probe, opts = {}) {
  const time = opts.time ?? 6.0;
  const div = opts.div ?? 100.0;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);
  const sampler = new FrameSampler(gpu.device,
    renderer.targets.accumWidth, renderer.targets.accumHeight);
  const was = probe.showFieldEvals;
  try {
    probe.showFieldEvals = 1;
    renderer.resetHistory();
    const patch = await converge(renderer, gpu, sampler, input, time, 1);
    return { meanEvalsPerPixel: patchMeanRed(patch) * div, patch: `${sampler.w}x${sampler.h}` };
  } finally {
    probe.showFieldEvals = was;
    sampler.destroy();
  }
}

/** Median of an array. Mutates by sorting — callers here always pass a private copy. */
function median(a) {
  if (!a.length) return 0;
  a.sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) * 0.5;
}

function percentile(a, p) {
  if (!a.length) return 0;
  const sorted = [...a].sort((x, y) => x - y);
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

/**
 * The deterministic input the measurements drive the renderer with.
 *
 * `everUsed:false` selects the organic drift and `chase:false` keeps the free camera,
 * both pure functions of `time` — so the whole path is reproducible from the clock
 * alone, which is what makes two builds comparable at all. `cmd` must be present: the
 * renderer steps the ship every frame and a missing command threw, which silently broke
 * every benchmark run once the ship landed.
 *
 * @param {object} [cmd] override the flight command, to measure under manoeuvre
 */
export function benchInput(cmd) {
  return {
    dragging: false, everUsed: false, chase: false,
    x: 0, y: 0, width: 1, height: 1,
    cmd: { pitch: 0, yaw: 0, roll: 0, thrust: 0, ...cmd },
  };
}

/**
 * Layout may not have given the canvas a size yet — the interactive path gets there via
 * its ResizeObserver, but a measurement can run straight out of boot and must not
 * measure a degenerate 4x4 target.
 */
async function ensureSize(renderer) {
  for (let i = 0; i < 10 && renderer.targets.width < 32; i++) {
    renderer.resize();
    if (renderer.targets.width >= 32) break;
    await new Promise((r) => setTimeout(r, 16));
  }
  if (renderer.targets.width < 32) {
    throw new Error(`canvas has no usable size (${renderer.targets.width}x${renderer.targets.height})`);
  }
}

/**
 * Render `n` frames on a synthetic clock, then read the accumulation buffer once.
 *
 * The size handed to `record` is the ACCUMULATION buffer's, which is not `targets.width` — that
 * distinction was a real bug and it silently corrupted every resolution comparison. `record`
 * centres the patch using the size it is given, so passing the render size while copying from a
 * display-resolution accumulation buffer read a patch centred at a QUARTER of the frame instead
 * of the middle. Two configs then measured two different regions and were reported side by side.
 * The "temporal upsampling looks 39% less sharp on the real scene" result came from here.
 */
async function converge(renderer, gpu, sampler, input, time, n) {
  for (let i = 0; i < n; i++) renderer.frame(time, input);
  const enc = gpu.device.createCommandEncoder({ label: 'converge-copy' });
  const t = renderer.targets;
  sampler.record(enc, t.accumRead, t.accumWidth, t.accumHeight);
  gpu.device.queue.submit([enc.finish()]);
  await gpu.device.queue.onSubmittedWorkDone();
  return sampler.read();
}

/**
 * @param {import('../renderer.js').Renderer} renderer
 * @param {object} gpu
 * @param {object} [opts]
 * @param {number} [opts.frames]   measured frames
 * @param {number} [opts.warmup]   frames discarded first (pipeline warm, TAA converged)
 * @param {number} [opts.startTime] synthetic clock start, seconds
 * @param {boolean} [opts.stability] freeze the clock and measure residual motion
 */
export async function benchmark(renderer, gpu, opts = {}) {
  const frames = opts.frames ?? 120;
  const warmup = opts.warmup ?? 30;
  const startTime = opts.startTime ?? 6.0;
  const stability = !!opts.stability;
  /** Drain the GPU queue every frame, so pass timestamps are attributable. Costs
   *  throughput; see the note at FLUSH. */
  const serial = !!opts.serial;
  const dt = 1 / 60;

  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  // Stability mode reads the accumulation buffer back and diffs consecutive frames.
  let sampler = null;
  let prevPatch = null;
  const deltas = [];
  const rels = [];
  if (stability) {
    sampler = new FrameSampler(gpu.device,
      renderer.targets.accumWidth, renderer.targets.accumHeight);
  }

  const passes = new Map();
  const wall = [];
  const prof = renderer.profiler;
  const prevOnSample = prof.onSample;

  let collecting = false;
  prof.onSample = (label, ms) => {
    if (!collecting) return;
    let a = passes.get(label);
    if (!a) passes.set(label, (a = []));
    a.push(ms);
  };

  // Do NOT wait for the GPU after every frame.
  //
  // Serialising per frame looks tidier and gives a clean per-frame wall time, but
  // it wrecks the pass breakdown: the GPU goes idle between submissions, and the
  // timestamps of the passes that follow a long one end up spanning the
  // scheduling gap. It showed up unmistakably — a pass the HUD had at 0.03 ms
  // reported 6 ms, and the pass medians summed to 68 ms inside a 31 ms frame,
  // which is impossible when passes run serially. Keeping the queue fed measures
  // the pipeline as it actually runs; the queue is drained every FLUSH frames so
  // it cannot grow without bound.
  //
  // `serial` drains the queue every frame instead, which isolates a frame from whatever
  // depth the queue happens to have. It is NOT needed for the pass breakdown to be
  // attributable — that was a duration-arithmetic bug, fixed in profiler.js — and it
  // inflates frame time by a full stall, so a serial run must never be read as
  // throughput. Useful only for pinning down one frame in isolation.
  const FLUSH = serial ? 1 : 8;
  const runStart = performance.now();
  let segStart = runStart;
  try {
    for (let i = 0; i < warmup + frames; i++) {
      collecting = i >= warmup;
      const time = stability ? startTime : startTime + i * dt;
      renderer.frame(time, input);
      if (serial) { await gpu.device.queue.onSubmittedWorkDone(); }

      if (sampler) {
        // Separate encoder AFTER the frame: `accumRead` is the buffer the frame just
        // wrote, since renderer.frame() swaps the ping-pong on its way out.
        const enc = gpu.device.createCommandEncoder({ label: 'stability-copy' });
        sampler.record(enc, renderer.targets.accumRead,
                       renderer.targets.accumWidth, renderer.targets.accumHeight);
        gpu.device.queue.submit([enc.finish()]);
        await gpu.device.queue.onSubmittedWorkDone();
        const patch = await sampler.read();
        if (prevPatch && collecting) {
          const d = patchDelta(patch, prevPatch);
          deltas.push(d.diff);
          rels.push(d.mag > 1e-6 ? d.diff / d.mag : 0);
        }
        prevPatch = patch;
      }

      await prof.readback();

      if ((i + 1) % FLUSH === 0) {
        await gpu.device.queue.onSubmittedWorkDone();
        const now = performance.now();
        if (collecting) wall.push((now - segStart) / FLUSH);
        segStart = now;
      }
    }
    await gpu.device.queue.onSubmittedWorkDone();
  } finally {
    collecting = false;
    prof.onSample = prevOnSample;
  }
  const stabilityResult = sampler
    ? {
      // Linear-HDR units, and the same figure as a FRACTION of the patch's own
      // brightness — which is the number that means something, since a bright scene
      // and a dim one cannot be compared in absolute terms.
      medianDelta: median([...deltas]),
      p95Delta: percentile(deltas, 0.95),
      medianRelative: median([...rels]),
      samples: deltas.length,
      patch: `${sampler.w}x${sampler.h}`,
    }
    : null;
  sampler?.destroy();
  const meanFrameMs = (performance.now() - runStart) / (warmup + frames);

  // Preserve the order the passes were recorded in rather than Map order, so the
  // report reads like the frame graph.
  const rows = [...passes.entries()].map(([label, a]) => ({
    label,
    median: median([...a]),
    p95: percentile(a, 0.95),
    samples: a.length,
  }));
  const gpuTotal = rows.reduce((s, r) => s + r.median, 0);

  // Passes reporting the SAME median to the nanosecond did not each measure something —
  // one interval was reported N times, which is what pipelined submission does to
  // timestamps. This is a far sharper contamination signal than the sum heuristic
  // below, because it names the passes involved instead of condemning the whole table.
  const byTime = new Map();
  for (const r of rows) {
    // Zero is not an alias. Several passes here genuinely fall below the counter's
    // 65 us resolution and can only report zero; flagging those as a shared interval
    // was a false positive that buried the real ones.
    if (r.median <= 0) continue;
    const k = r.median.toFixed(6);
    if (!byTime.has(k)) byTime.set(k, []);
    byTime.get(k).push(r.label);
  }
  const aliased = [...byTime.values()].filter((g) => g.length > 1);

  return {
    mode: stability ? 'stability' : 'throughput',
    serial,
    frames,
    resolution: `${renderer.targets.width}x${renderer.targets.height}`,
    timestamps: gpu.caps.timestamps,
    passes: rows,
    gpuTotalMs: gpuTotal,
    // In serial mode this INCLUDES a per-frame stall, so it is not the throughput
    // figure. Named to be awkward to misread.
    meanFrameMs,
    frameMsIsThroughput: !serial,
    wallMedianMs: median([...wall]),
    wallP95Ms: percentile(wall, 0.95),
    // Passes run serially on the GPU, so their times cannot sum to more than a
    // frame. When they do, the timestamps caught scheduling gaps rather than work,
    // and the breakdown must not be believed — better to say so than to publish a
    // plausible-looking number that is arithmetically impossible.
    contaminated: gpu.caps.timestamps && (aliased.length > 0 || gpuTotal > meanFrameMs * 1.2),
    /** Groups of passes that reported one shared interval. Empty is the healthy case. */
    aliasedPasses: aliased,
    stability: stabilityResult,
  };
}

/**
 * Do two configurations render the SAME IMAGE?
 *
 * The question an accelerator has to answer. A conservative distance bound is supposed to
 * change how long the march takes and nothing else; if it ever over-estimates, rays tunnel
 * through the surface and punch holes. Those holes appear only at particular angles, so
 * looking at one screenshot proves very little — and a timing measurement proves nothing
 * at all, because a bound that skips the surface entirely is gloriously fast.
 *
 * Converge at a fixed instant, apply `mutate`, converge again at the same instant, and
 * diff. Converged rather than single-frame so the sampling noise averages out of both
 * sides; the residual floor is returned alongside so "identical" can be judged against
 * what identical actually costs.
 *
 * @param {() => void} mutate      switch to the configuration under test
 * @param {() => void} restore     switch back
 */
export async function compareConfigs(renderer, gpu, mutate, restore, opts = {}) {
  const settle = opts.settle ?? 48;
  const time = opts.time ?? 6.0;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  const sampler = new FrameSampler(gpu.device,
    renderer.targets.accumWidth, renderer.targets.accumHeight);
  try {
    renderer.resetHistory();
    const a = await converge(renderer, gpu, sampler, input, time, settle);
    // A second converged frame under the SAME config, to price the noise floor.
    const a2 = await converge(renderer, gpu, sampler, input, time, 1);

    mutate();
    try {
      renderer.resetHistory();
      const b = await converge(renderer, gpu, sampler, input, time, settle);
      const diff = patchDelta(b, a);
      const floor = patchDelta(a2, a);
      const rel = diff.mag > 1e-6 ? diff.diff / diff.mag : 0;
      const relFloor = floor.mag > 1e-6 ? floor.diff / floor.mag : 0;
      return {
        relative: rel,
        noiseFloor: relFloor,
        /** Above ~2x the floor the two configurations are rendering different images. */
        overFloor: relFloor > 1e-9 ? rel / relFloor : Infinity,
        sharpnessA: patchSharpness(a, sampler.w, sampler.h),
        sharpnessB: patchSharpness(b, sampler.w, sampler.h),
      };
    } finally {
      restore();
    }
  } finally {
    sampler.destroy();
  }
}

/**
 * Sharpness of two configurations, measured on a COMMON display-resolution grid.
 *
 * This is the instrument the upsampling work needed and kept going without. Sharpness is a
 * per-pixel gradient measure, so comparing a 1280x720 accumulation buffer against a 2560x1440
 * one directly is meaningless — the second has half the per-pixel gradient for the same image,
 * which would make any upsampler look 2x softer than it is. So: read each config's patch over
 * the SAME screen region (which means different pixel counts), bring both to the same grid,
 * and only then measure. The resample is bilinear, deliberately the same operation the
 * composite performs when it upscales the low-resolution path for display — so the comparison
 * is "upsampling versus what it actually replaces", not against an idealisation.
 *
 * Each entry of `configs` is `{ name, apply }`; `apply` sets the configuration and may change
 * target sizes, since it is followed by a resize and a fresh convergence.
 */
export async function matchedSharpness(renderer, gpu, configs, opts = {}) {
  const settle = opts.settle ?? 48;
  const time = opts.time ?? 6.0;
  // The comparison grid, in DISPLAY pixels. Width must stay a multiple of 32 after being
  // scaled down for the low-resolution config, so 256 -> 128 -> 64 all remain valid.
  const gridW = opts.gridW ?? 256;
  const gridH = opts.gridH ?? 144;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);
  const displayW = renderer.targets.displayWidth;

  const out = [];
  for (const cfg of configs) {
    cfg.apply();
    renderer.resize();
    const t = renderer.targets;
    // Same screen region at whatever resolution this config accumulates in.
    const k = t.accumWidth / displayW;
    const pw = Math.max(32, Math.round(gridW * k / 32) * 32);
    const ph = Math.max(2, Math.round(gridH * k));
    const sampler = new FrameSampler(gpu.device, t.accumWidth, t.accumHeight, pw, ph);
    try {
      renderer.resetHistory();
      const patch = await converge(renderer, gpu, sampler, input, time, settle);
      const f = decodeHalf(patch);
      const common = (sampler.w === gridW && sampler.h === gridH)
        ? f
        : resampleRGBA(f, sampler.w, sampler.h, gridW, gridH);
      out.push({
        name: cfg.name,
        accum: `${t.accumWidth}x${t.accumHeight}`,
        readPatch: `${sampler.w}x${sampler.h}`,
        sharpness: sharpnessOf(common, gridW, gridH),
      });
    } finally {
      sampler.destroy();
    }
  }
  const best = out.reduce((a, b) => (b.sharpness > a.sharpness ? b : a), out[0]);
  return { grid: `${gridW}x${gridH}`, configs: out, sharpest: best.name };
}

/** Luma of a decoded f32 RGBA patch, as a single plane. */
function lumaPlane(f, w, h) {
  const L = new Float32Array(w * h);
  for (let i = 0, j = 0; i < L.length; i++, j += 4) {
    const v = 0.2126 * f[j] + 0.7152 * f[j + 1] + 0.0722 * f[j + 2];
    L[i] = Number.isFinite(v) ? v : 0;
  }
  return L;
}

/** Separable Gaussian blur of a single plane, clamped at the edges. */
function blurPlane(L, w, h, sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v; sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const clamp = (v, hi) => Math.min(hi, Math.max(0, v));
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += k[i + r] * L[y * w + clamp(x + i, w - 1)];
      tmp[y * w + x] = a;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += k[i + r] * tmp[clamp(y + i, h - 1) * w + x];
      out[y * w + x] = a;
    }
  }
  return out;
}

/** The high-frequency band: luma minus its own blur. */
function highPass(f, w, h, sigma) {
  const L = lumaPlane(f, w, h);
  const B = blurPlane(L, w, h, sigma);
  const H = new Float32Array(w * h);
  for (let i = 0; i < H.length; i++) H[i] = L[i] - B[i];
  return { L, H };
}

/**
 * DETAIL vs NOISE: does the extra sharpness a reconstruction produces match the truth?
 *
 * The question `matchedSharpness` cannot answer, and the reason the temporal-upsampling verdict
 * stayed open. Sharpness is the magnitude of an image's high-frequency band, and noise lives in
 * exactly that band — so a sharper number can mean recovered detail or added grain, and nothing
 * about the number itself distinguishes them. On the synthetic pattern the upsampler measured 7.6x
 * more resolved detail; on the real scene it measured 39% LESS sharp, and neither figure could be
 * trusted because both metrics were blind in the same way.
 *
 * The fix is a GROUND TRUTH and a projection rather than a magnitude. Render the same instant at
 * native display resolution and let the temporal filter converge: with the clock frozen the jitter
 * sequence supersamples the frame, so what it settles to is the correctly antialiased image. Then
 * for any candidate X, decompose its high-frequency band against the reference's:
 *
 *     signal = <H(X), H(R)> / <H(R), H(R)>      how much of the TRUE detail is present, 1 = all
 *     noise  = ||H(X) - signal * H(R)|| / ||H(R)||    high-frequency energy that is not detail
 *
 * These are orthogonal by construction, so magnitude^2 = signal^2 + noise^2 — the same total that
 * a sharpness metric reports, now split into the two halves that were being confused. A blurry
 * image has low signal and low noise; a grainy one has signal near 1 and high noise; only the
 * combination tells you which.
 *
 * A SECOND, INDEPENDENT reference is measured alongside, converged from a fresh history so its
 * jitter subsequence differs. It is the same configuration as the reference, so its true signal is
 * 1 and its true noise is 0 — whatever it actually reports is the method's own floor, since the
 * reference carries its own residual and that residual is uncorrelated with everyone else's.
 * Read every noise figure against it, exactly as lag is read against stability.
 *
 * @param {object} quality     the QUALITY tuning block, mutated and restored here. Passed in
 *                             rather than imported, as `fieldEvalCount` takes PROBE — this
 *                             module stays free of tuning imports so it cannot disagree with
 *                             the renderer about which object is live.
 * @param {object} [opts]
 * @param {number} [opts.settle]  frames to converge each configuration
 * @param {number} [opts.sigma]   high-pass cutoff, in display pixels
 * @param {number} [opts.gridW]   comparison window, display pixels
 */
export async function detailSnr(renderer, gpu, quality, opts = {}) {
  const settle = opts.settle ?? 48;
  const time = opts.time ?? 6.0;
  const sigma = opts.sigma ?? 1.5;
  const gridW = opts.gridW ?? 256;
  const gridH = opts.gridH ?? 144;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);
  const displayW = renderer.targets.displayWidth;

  // `dynamicRes` off for the duration: it would move the very scale being held fixed here.
  const was = { taau: quality.taau, renderScale: quality.renderScale, dyn: quality.dynamicRes };
  quality.dynamicRes = false;
  // The reference is measured FIRST and LAST is restored; the control repeats the reference's
  // configuration so its only difference from the reference is which jitter samples it saw.
  const configs = [
    { name: 'reference', taau: false, scale: 1.0 },
    { name: 'control', taau: false, scale: 1.0 },
    { name: 'taau', taau: true, scale: was.renderScale },
    { name: 'lowres', taau: false, scale: was.renderScale },
  ];

  const grids = [];
  try {
    for (const cfg of configs) {
      quality.taau = cfg.taau;
      quality.renderScale = cfg.scale;
      renderer.resize();
      const t = renderer.targets;
      // The same screen region, at whatever resolution this configuration accumulates in.
      const k = t.accumWidth / displayW;
      const pw = Math.max(32, Math.round(gridW * k / 32) * 32);
      const ph = Math.max(2, Math.round(gridH * k));
      const sampler = new FrameSampler(gpu.device, t.accumWidth, t.accumHeight, pw, ph);
      try {
        renderer.resetHistory();
        const patch = await converge(renderer, gpu, sampler, input, time, settle);
        const f = decodeHalf(patch);
        // Resampling the low-resolution config up to the common grid IS the naive upscale it
        // stands for, so the baseline is measured as it would actually be displayed.
        const common = (sampler.w === gridW && sampler.h === gridH)
          ? f : resampleRGBA(f, sampler.w, sampler.h, gridW, gridH);
        grids.push({
          name: cfg.name,
          accum: `${t.accumWidth}x${t.accumHeight}`,
          ...highPass(common, gridW, gridH, sigma),
        });
      } finally {
        sampler.destroy();
      }
    }
  } finally {
    quality.taau = was.taau;
    quality.renderScale = was.renderScale;
    quality.dynamicRes = was.dyn;
    renderer.resize();
    renderer.resetHistory();
  }

  const ref = grids[0];
  // Exclude a margin: the blur clamps at the edges, so the band is not meaningful there.
  const m = Math.max(2, Math.ceil(sigma * 3));
  const dot = (A, B) => {
    let a = 0;
    for (let y = m; y < gridH - m; y++) {
      for (let x = m; x < gridW - m; x++) { const i = y * gridW + x; a += A[i] * B[i]; }
    }
    return a;
  };
  const rr = dot(ref.H, ref.H);
  const refNorm = Math.sqrt(rr);

  const out = grids.map((g) => {
    const signal = rr > 0 ? dot(g.H, ref.H) / rr : 0;
    const gg = dot(g.H, g.H);
    // ||H - signal*Href||^2 = <H,H> - 2*signal*<H,Href> + signal^2*<Href,Href>
    const residual2 = Math.max(0, gg - 2 * signal * dot(g.H, ref.H) + signal * signal * rr);
    // Plain error against the truth, over the whole band, for the standard PSNR view.
    let se = 0, n = 0, peak = 0;
    for (let y = m; y < gridH - m; y++) {
      for (let x = m; x < gridW - m; x++) {
        const i = y * gridW + x;
        const d = g.L[i] - ref.L[i];
        se += d * d; n++;
        peak = Math.max(peak, ref.L[i]);
      }
    }
    const rmse = Math.sqrt(se / Math.max(n, 1));
    return {
      name: g.name,
      accum: g.accum,
      magnitude: +(Math.sqrt(gg) / Math.max(refNorm, 1e-12)).toFixed(4),
      signal: +signal.toFixed(4),
      noise: +(Math.sqrt(residual2) / Math.max(refNorm, 1e-12)).toFixed(4),
      psnr: rmse > 0 ? +(20 * Math.log10(Math.max(peak, 1e-6) / rmse)).toFixed(2) : Infinity,
    };
  });

  // CALIBRATED against the control, which is the only apples-to-apples comparison available.
  //
  // The reference is itself a converged render, so part of its high-frequency band is its own
  // residual noise rather than detail. An independent run of the same configuration correlates
  // only with the shared part, so the control's `signal` estimates what fraction of the
  // reference's band is real detail — 1 would mean fully converged. Dividing every config's
  // signal by the control's therefore expresses it as a fraction of NATIVE detail retained,
  // which is the number that answers "what does upsampling cost".
  const ctl = out[1].signal;
  for (const c of out) c.retained = ctl > 1e-6 ? +(c.signal / ctl).toFixed(4) : 0;

  return {
    grid: `${gridW}x${gridH}`, sigma, settle,
    // Named out loud, because every noise figure above is meaningless without it.
    noiseFloor: out[1].noise,
    convergence: ctl,
    configs: out,
  };
}

/**
 * ALIASING in the additive layer, and what removing it costs.
 *
 * The layer that carries the particles, the contrail, the rail guns and the auroras is drawn
 * after the temporal resolve — it has to be, since reprojection assumes static geometry and
 * every mote would ghost — so nothing antialiases it. At render resolution it is then upscaled
 * along with its own aliasing. This measures whether that is actually visible and what fixing
 * it would cost, rather than leaving it as a plausible suspicion.
 *
 * Method: converge the same instant with the layer at display resolution and at render
 * resolution, upscale the second exactly as the composite does, and compare their
 * high-frequency bands. The display-resolution render is the reference, because for a layer
 * with no history there is nothing better to compare against — the question is not "is it
 * correct" but "does drawing it small change what you see". Frame cost is reported beside it,
 * since that is the other half of the decision.
 *
 * @param {object} quality  the QUALITY tuning block, mutated and restored here
 */
export async function additiveAliasing(renderer, gpu, quality, opts = {}) {
  const settle = opts.settle ?? 32;
  const time = opts.time ?? 6.0;
  const sigma = opts.sigma ?? 1.5;
  const gridW = opts.gridW ?? 256;
  const gridH = opts.gridH ?? 144;
  const frames = opts.frames ?? 40;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  const was = quality.additiveDisplayRes;
  const wasDyn = quality.dynamicRes;
  quality.dynamicRes = false;
  const runs = [];
  try {
    for (const displayRes of [true, false]) {
      quality.additiveDisplayRes = displayRes;
      renderer.resize();
      const t = renderer.targets;
      const k = t.addWidth / t.displayWidth;
      const pw = Math.max(32, Math.round(gridW * k / 32) * 32);
      const ph = Math.max(2, Math.round(gridH * k));
      const sampler = new FrameSampler(gpu.device, t.addWidth, t.addHeight, pw, ph);
      try {
        renderer.resetHistory();
        for (let i = 0; i < settle; i++) renderer.frame(time, input);
        const enc = gpu.device.createCommandEncoder({ label: 'additive-copy' });
        sampler.record(enc, t.ember, t.addWidth, t.addHeight);
        gpu.device.queue.submit([enc.finish()]);
        await gpu.device.queue.onSubmittedWorkDone();
        const f = decodeHalf(await sampler.read());
        const common = (sampler.w === gridW && sampler.h === gridH)
          ? f : resampleRGBA(f, sampler.w, sampler.h, gridW, gridH);
        // Wall-clock cost of the same instant, so the trade is stated in one place.
        const t0 = performance.now();
        for (let i = 0; i < frames; i++) renderer.frame(time, input);
        await gpu.device.queue.onSubmittedWorkDone();
        const ms = (performance.now() - t0) / frames;
        runs.push({
          name: displayRes ? 'display-res' : 'render-res',
          size: `${t.addWidth}x${t.addHeight}`,
          frameMs: +ms.toFixed(3),
          ...highPass(common, gridW, gridH, sigma),
        });
      } finally {
        sampler.destroy();
      }
    }
  } finally {
    quality.additiveDisplayRes = was;
    quality.dynamicRes = wasDyn;
    renderer.resize();
    renderer.resetHistory();
  }

  const [ref, low] = runs;
  const m = Math.max(2, Math.ceil(sigma * 3));
  let hr = 0, hl = 0, dh = 0, lref = 0, ldiff = 0, n = 0;
  for (let y = m; y < gridH - m; y++) {
    for (let x = m; x < gridW - m; x++) {
      const i = y * gridW + x;
      hr += ref.H[i] * ref.H[i];
      hl += low.H[i] * low.H[i];
      dh += (low.H[i] - ref.H[i]) ** 2;
      lref += ref.L[i];
      ldiff += Math.abs(low.L[i] - ref.L[i]);
      n++;
    }
  }
  return {
    grid: `${gridW}x${gridH}`, sigma, settle,
    runs: runs.map((r) => ({ name: r.name, size: r.size, frameMs: r.frameMs })),
    // How much of the layer's own high-frequency structure the low-resolution render gets wrong.
    bandError: +(Math.sqrt(dh / Math.max(hr, 1e-20))).toFixed(4),
    bandRatio: +(Math.sqrt(hl / Math.max(hr, 1e-20))).toFixed(4),
    // And in plain brightness terms, relative to the layer's own mean.
    meanError: +(ldiff / Math.max(lref, 1e-20)).toFixed(4),
    costMs: +(runs[0].frameMs - runs[1].frameMs).toFixed(3),
  };
}

/**
 * A PNG of the frame the renderer is showing.
 *
 * The swapchain texture is released when it is presented, so `drawImage` on a WebGPU canvas hands
 * back transparent black — there is no browser-side way to screenshot one. It has to be copied out
 * during the frame that drew it, which is why this lives here and why the canvas is configured
 * with COPY_SRC.
 *
 * The surface format is BGRA on most platforms and the canvas is `opaque`, so the channels are
 * swizzled and alpha forced on the way into the ImageData.
 */
export async function grabFrame(renderer, gpu, scale = 1) {
  await ensureSize(renderer);
  const input = benchInput();
  // One frame, so the texture being copied is the one just drawn into.
  renderer.frame(performance.now() / 1000, input);
  const tex = gpu.context.getCurrentTexture();
  const w = tex.width, h = tex.height;
  const bpr = Math.ceil(w * 4 / 256) * 256;
  const buf = gpu.device.createBuffer({
    label: 'screenshot',
    size: bpr * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = gpu.device.createCommandEncoder({ label: 'screenshot-copy' });
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
  gpu.device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange().slice(0));
  buf.unmap();
  buf.destroy();

  const img = new ImageData(w, h);
  const bgra = gpu.format.startsWith('bgra');
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * bpr + x * 4;
      const o = (y * w + x) * 4;
      img.data[o] = bgra ? src[i + 2] : src[i];
      img.data[o + 1] = src[i + 1];
      img.data[o + 2] = bgra ? src[i] : src[i + 2];
      img.data[o + 3] = 255;                 // the context is opaque; alpha is not meaningful
    }
  }
  const ow = Math.max(1, Math.round(w * scale));
  const oh = Math.max(1, Math.round(h * scale));
  const full = new OffscreenCanvas(w, h);
  full.getContext('2d').putImageData(img, 0, 0);
  if (ow === w && oh === h) return full.convertToBlob({ type: 'image/png' });
  const small = new OffscreenCanvas(ow, oh);
  const c2 = small.getContext('2d');
  c2.imageSmoothingQuality = 'high';
  c2.drawImage(full, 0, 0, ow, oh);
  return small.convertToBlob({ type: 'image/png' });
}

/**
 * SUB-PIXEL STABILITY: does the image flicker when the camera moves less than one pixel?
 *
 * The measurement every "is it crawling?" question actually wants, and the one the other
 * instruments cannot give. `stability` freezes the camera, so it sees sampling noise but no
 * aliasing. `detail` compares against a ground truth, so it sees blur and grain but says nothing
 * about behaviour under motion. Crawl is neither: it is what happens when a feature slides across
 * the sampling grid, and it only appears if you slide it.
 *
 * So: shift the camera across ONE pixel in `steps` fractions, render each deterministically, and
 * ask how much the image changes. Under a pure translation a properly filtered image is
 * essentially invariant — its mean and its high-frequency energy do not care where the grid sits.
 * An aliased one beats against the grid, and both quantities wobble. That wobble, as a coefficient
 * of variation, IS the crawl.
 *
 * Two things make this honest, and both are the point:
 *
 * NO RESAMPLING. The obvious method — shift, then shift back and diff — cannot work, because
 * resampling by a fractional offset blurs by an amount that depends on the fraction, and that
 * dependence is indistinguishable from the aliasing being measured. Comparing translation-INVARIANT
 * scalars instead needs no compensation at all.
 *
 * NO NOISE. Every per-frame random source has to go, or it swamps the signal outright:
 *   - `CAMERA.aperture` offsets the ray ORIGIN every frame from a disk sequence indexed by frame
 *     number. It is the reason the residual never converges away.
 *   - `FILM.grain` is per-pixel noise. It happens to be applied in the composite, which this
 *     bypasses by reading the target directly, but it is zeroed anyway so the intent is on the
 *     record rather than resting on which pass reads what.
 *
 * The TAA jitter is the exception, and getting it wrong is the trap. It is ITSELF a sub-pixel
 * shift, so it looks like exactly the noise to switch off — but it is also the ANTIALIASING, and
 * disabling it measures the renderer with its AA turned off, which reports the marched image as
 * wildly unstable and means nothing. It is made REPEATABLE instead: `frameIndex` is pinned before
 * each offset's render, so every offset sees the identical jitter subsequence and the only thing
 * that differs between them is the camera shift asked for.
 * The clock is frozen too, which stops the animation without stopping the particle and ribbon
 * simulations — those advance on the clamped minimum dt, about 0.1 ms per frame, so over a whole
 * run they drift by a few milliseconds' worth of motion. Not zero, but three orders of magnitude
 * below a pixel.
 *
 * @param {object} knobs  { quality, camera, film, probe } — the live tuning blocks, mutated and
 *                        restored here. Passed in rather than imported, as the others are.
 */
export async function subPixelStability(renderer, gpu, knobs, opts = {}) {
  const { quality, camera: cam, film, probe } = knobs;
  const steps = opts.steps ?? 8;
  // Enough for the accumulation to actually converge; a handful of frames measures a
  // half-built history instead of the image.
  const settle = opts.settle ?? 40;
  const time = opts.time ?? 6.0;
  const sigma = opts.sigma ?? 1.5;
  const gridW = opts.gridW ?? 256;
  const gridH = opts.gridH ?? 144;
  const span = opts.span ?? 1.0;            // how far to slide, in DISPLAY pixels
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  // The accumulation buffer is measured only once: the additive layer's resolution cannot affect
  // it, which the first version of this confirmed by reporting identical figures for both — a
  // useful check, and then a waste of half the runtime.
  const configs = opts.configs ?? [
    { name: 'additive @ render res', taau: true, additive: false, targets: ['ember', 'accum'] },
    { name: 'additive @ display res', taau: true, additive: true, targets: ['ember'] },
  ];

  const saved = {
    aperture: cam.aperture, grain: film.grain, zeroJitter: probe.zeroJitter,
    offX: cam.frameOffset[0], offY: cam.frameOffset[1],
    taau: quality.taau, additive: quality.additiveDisplayRes, dyn: quality.dynamicRes,
  };
  cam.aperture = 0;
  film.grain = 0;
  quality.dynamicRes = false;
  // Deliberately NOT zeroJitter — see above. Left as the caller had it, and forced OFF so a
  // stale probe setting cannot silently remove the antialiasing under measurement.
  probe.zeroJitter = false;
  const fiBase = renderer.frameIndex;

  const out = [];
  try {
    for (const cfg of configs) {
      quality.taau = cfg.taau;
      quality.additiveDisplayRes = cfg.additive;
      renderer.resize();
      const t = renderer.targets;
      // One display pixel is 2/diagonal in the shared screen space, and a screen-space shift of
      // `d` is produced by a frameOffset of `-d` (see the projection: the forward axis lands at
      // screen -frameOffset). Expressed in DISPLAY pixels so the number means the same thing to
      // every configuration regardless of what it renders at.
      const diagPx = Math.hypot(t.displayWidth, t.displayHeight);
      const perPixel = 2 / diagPx;

      for (const which of (cfg.targets ?? ['ember'])) {
        const texW = which === 'ember' ? t.addWidth : t.accumWidth;
        const texH = which === 'ember' ? t.addHeight : t.accumHeight;
        const k = texW / t.displayWidth;
        const pw = Math.max(32, Math.round(gridW * k / 32) * 32);
        const ph = Math.max(2, Math.round(gridH * k));
        const sampler = new FrameSampler(gpu.device, texW, texH, pw, ph);
        const means = [];
        const bands = [];
        try {
          for (let i = 0; i < steps; i++) {
            const d = (i / steps) * span;
            cam.frameOffset[0] = saved.offX - d * perPixel;
            // `setViewport` runs every frame and now keys its early-out on frameOffset, so the
            // projection rebuilds on its own; nothing here has to reach past the camera.
            //
            // Pinning the frame counter is what makes the jitter repeatable rather than absent:
            // it indexes the jitter sequence, so every offset now walks the identical one.
            renderer.frameIndex = fiBase;
            renderer.resetHistory();
            for (let f = 0; f < settle; f++) renderer.frame(time, input);
            const enc = gpu.device.createCommandEncoder({ label: 'subpixel-copy' });
            const tex = which === 'ember' ? renderer.targets.ember : renderer.targets.accumRead;
            sampler.record(enc, tex, texW, texH);
            gpu.device.queue.submit([enc.finish()]);
            await gpu.device.queue.onSubmittedWorkDone();
            const f32 = decodeHalf(await sampler.read());
            const common = (sampler.w === gridW && sampler.h === gridH)
              ? f32 : resampleRGBA(f32, sampler.w, sampler.h, gridW, gridH);
            const { L, H } = highPass(common, gridW, gridH, sigma);
            // Interior only: content enters and leaves at the patch border as the image slides,
            // and that is a boundary effect, not aliasing.
            const m = Math.max(2, Math.ceil(sigma * 3)) + 2;
            let sl = 0, sh = 0, n = 0;
            for (let y = m; y < gridH - m; y++) {
              for (let x = m; x < gridW - m; x++) {
                const j = y * gridW + x;
                sl += L[j]; sh += H[j] * H[j]; n++;
              }
            }
            means.push(sl / n);
            bands.push(Math.sqrt(sh / n));
          }
        } finally {
          sampler.destroy();
        }
        const cv = (a) => {
          const mu = a.reduce((x, y) => x + y, 0) / a.length;
          if (Math.abs(mu) < 1e-12) return 0;
          const v = a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length;
          return Math.sqrt(v) / mu;
        };
        // The largest jump between ADJACENT sub-pixel positions. Perceptually this is the one
        // that matters: a smooth drift across a pixel is invisible, a step is a flicker.
        let worst = 0;
        for (let i = 1; i < means.length; i++) {
          worst = Math.max(worst, Math.abs(means[i] - means[i - 1]));
        }
        const muM = means.reduce((x, y) => x + y, 0) / means.length;
        out.push({
          config: cfg.name,
          target: which,
          size: `${texW}x${texH}`,
          meanCV: +(cv(means) * 100).toFixed(3),
          bandCV: +(cv(bands) * 100).toFixed(3),
          worstStep: +(muM > 1e-12 ? (worst / muM) * 100 : 0).toFixed(3),
        });
      }
    }
  } finally {
    cam.aperture = saved.aperture;
    film.grain = saved.grain;
    probe.zeroJitter = saved.zeroJitter;
    cam.frameOffset[0] = saved.offX;
    cam.frameOffset[1] = saved.offY;
    quality.taau = saved.taau;
    quality.additiveDisplayRes = saved.additive;
    quality.dynamicRes = saved.dyn;
    renderer.resize();
    renderer.resetHistory();
  }

  return { steps, span, settle, grid: `${gridW}x${gridH}`, sigma, rows: out };
}

/**
 * TEMPORAL LAG: how far the accumulated image trails the instant it claims to show.
 *
 * This is the measurement the harness was missing, and its absence is exactly why the
 * TAA depth gate could not safely be loosened any further. `stability` proves the image
 * is STEADY — and a badly ghosting accumulator is extremely steady, because it is mostly
 * showing you the past. Steadiness and correctness are different axes, and no single
 * frame-to-frame difference can see both. Every "is it ghosting?" question up to now was
 * answered by looking at screenshots.
 *
 * Method, per sample:
 *   1. Advance the clock normally so TAA accumulates over a MOVING scene, then read the
 *      accumulation buffer. That is what the renderer actually shows at time T.
 *   2. Freeze the clock at exactly T, reset the history, and converge from scratch. With
 *      nothing moving, every reprojection is valid by construction, so what TAA settles
 *      to is the correct anti-aliased image for that instant — the sampling noise
 *      averaged away and no lag in it at all.
 *   3. Their difference is the lag.
 *
 * Converging the reference rather than rendering one un-accumulated frame is the whole
 * trick. A single frame carries the full per-sample noise, ~3.3% here, which would swamp
 * the signal entirely. Averaging drives that toward zero while leaving lag untouched.
 *
 * Read the result AGAINST the stability figure, never on its own: lag at or below the
 * residual is indistinguishable from noise, and lag several times higher is real
 * trailing. Both numbers are returned together so the comparison cannot be skipped.
 *
 * Each sample necessarily destroys the history it just measured, so the next one re-warms
 * from scratch. The clock advances across samples regardless, so the ship and the
 * contrail keep evolving and the samples land on genuinely different scene states.
 *
 * @param {object} [opts]
 * @param {number} [opts.samples]  how many instants to measure
 * @param {number} [opts.warmup]   animated frames before each capture
 * @param {number} [opts.settle]   frozen frames to build each reference
 * @param {object} [opts.cmd]      flight command, to measure lag under manoeuvre
 */
export async function lagMetric(renderer, gpu, opts = {}) {
  const samples = opts.samples ?? 6;
  const warmup = opts.warmup ?? 40;
  const settle = opts.settle ?? 48;
  const startTime = opts.startTime ?? 6.0;
  const dt = 1 / 60;

  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  const sampler = new FrameSampler(gpu.device,
    renderer.targets.accumWidth, renderer.targets.accumHeight);
  const lags = [];
  const residuals = [];
  const sharps = [];
  const refSharps = [];
  let time = startTime;

  try {
    const rel = (a, b) => {
      const d = patchDelta(a, b);
      return d.mag > 1e-6 ? d.diff / d.mag : 0;
    };

    for (let s = 0; s < samples; s++) {
      // ---- 1. the real pipeline, accumulating over motion, up to exactly T ----
      renderer.resetHistory();
      for (let i = 0; i < warmup; i++) {
        renderer.frame(time, input);
        time += dt;
      }
      const T = time;
      const live = await converge(renderer, gpu, sampler, input, T, 1);

      // ---- 2. the converged reference for THAT SAME instant ----
      // Same T, not T+dt: an off-by-one here injects a frame of real scene motion into
      // the difference and reports it as lag.
      //
      // resetHistory first, because the point is a reference with no history in it —
      // reusing the accumulated buffer would carry the very lag being measured into the
      // baseline it is measured against.
      renderer.resetHistory();
      const ref = await converge(renderer, gpu, sampler, input, T, settle);

      // ---- 3. the noise floor, also at T ----
      // One further FROZEN frame. It must be frozen: measured from two animated frames
      // instead, this picks up legitimate scene motion and reports ~7% where the real
      // sampling noise is ~3.3%, which flatters the ratio into meaninglessness.
      const ref2 = await converge(renderer, gpu, sampler, input, T, 1);

      lags.push(rel(live, ref));
      residuals.push(rel(ref2, ref));
      // Sharpness of the LIVE buffer, not the reference: the reference is converged with
      // a frozen camera, where every history filter degenerates to the same identity tap.
      // Only the moving case resamples at fractional offsets, so only it can tell them
      // apart. Paired with lag from the same frames, which is the point — a sharper
      // reconstruction that buys its detail back by trailing has not helped.
      sharps.push(patchSharpness(live, sampler.w, sampler.h));
      refSharps.push(patchSharpness(ref, sampler.w, sampler.h));
      time += dt;
    }
  } finally {
    sampler.destroy();
  }

  const medLag = median([...lags]);
  const medRes = median([...residuals]);
  return {
    mode: 'lag',
    samples: lags.length,
    patch: `${sampler.w}x${sampler.h}`,
    /** Median relative difference from the converged truth. */
    medianLag: medLag,
    p95Lag: percentile(lags, 0.95),
    /** The noise floor the above must be read against. */
    medianResidual: medRes,
    /** Lag in units of the noise floor. At or under ~1.3 it is not distinguishable. */
    lagOverNoise: medRes > 1e-9 ? medLag / medRes : Infinity,
    /** Detail retained while MOVING — the figure a history filter is judged on. */
    sharpness: median([...sharps]),
    /** The same measure on the converged reference, as a scale for the above. Every
     *  history filter collapses to the identity tap here, so this is filter-independent
     *  and moves only if the SCENE or the sampling changed. */
    refSharpness: median([...refSharps]),
    perSample: lags.map((v, i) => ({
      lag: v, residual: residuals[i], sharpness: sharps[i],
    })),
  };
}

/**
 * Residual aperture parallax, analytically.
 *
 * This measures the actual mechanism behind the camera shake rather than a proxy
 * for it. The lens offset displaces the ray ORIGIN, so what the viewer perceives
 * as wobble is the drift of the accumulated MEAN of those offsets — if the running
 * mean sat exactly at the disk centre every frame, the bokeh would be smooth and
 * the image would not move at all.
 *
 * So: replay the exponential accumulation the TAA actually performs over the lens
 * sample sequence, and report how far its mean strays from centre. Closed form, no
 * GPU, no eyeballing, and directly comparable between sampler designs.
 *
 * Note what it does NOT cover: shader sampling noise (AO, SSS, transmission) also
 * shimmers, and that shows up in `benchmark({stability:true})` instead.
 *
 * @param {Float32Array} disk   xy pairs, unit disk
 * @param {number} blend        TAA weight of the new sample
 * @param {number} aperture     world-space lens radius
 */
export function lensResidual(disk, blend, aperture) {
  const n = disk.length / 2;
  let mx = 0;
  let my = 0;
  let worst = 0;
  // Several cycles, so the measurement reflects the steady state rather than the
  // transient from starting at the centre.
  const iterations = n * 12;
  for (let i = 0; i < iterations; i++) {
    const k = (i % n) * 2;
    mx += (disk[k] - mx) * blend;
    my += (disk[k + 1] - my) * blend;
    if (i > n * 4) worst = Math.max(worst, Math.hypot(mx, my));
  }
  return {
    cycle: n,
    blend,
    worstOffsetFractionOfAperture: worst,
    worstOffsetWorldUnits: worst * aperture,
  };
}

/**
 * Capture two SEQUENTIAL frames of the accumulation buffer and report where the
 * residual actually is.
 *
 * A single scalar ("4.2% of brightness changes per frame") says something is wrong but
 * not what. Splitting the same difference by surface class and by brightness says
 * whether the instability is everywhere or only on the body, only at edges, or only in
 * the bright core — and those have completely different causes. It also draws the
 * amplified difference into an on-screen canvas, because some patterns (a grid, a band,
 * a single object) are obvious to the eye and invisible in a mean.
 *
 * The clock is frozen across both frames, so anything that differs is the sampler and
 * the accumulation, never the animation.
 */
export async function dumpFrames(renderer, gpu, opts = {}) {
  const time = opts.time ?? 9.0;
  const settle = opts.settle ?? 45;
  const input = {
    dragging: false, everUsed: false, chase: false,
    x: 0, y: 0, width: 1, height: 1,
    cmd: { pitch: 0, yaw: 0, roll: 0, thrust: 0 },
  };

  for (let i = 0; i < 10 && renderer.targets.width < 32; i++) {
    renderer.resize();
    await new Promise((r) => setTimeout(r, 16));
  }
  // The ACCUMULATION buffer's size, since that is the texture being copied — see `converge`.
  const W = renderer.targets.accumWidth;
  const H = renderer.targets.accumHeight;
  const sampler = new FrameSampler(gpu.device, W, H);

  // Let the accumulation settle on a frozen scene first.
  for (let i = 0; i < settle; i++) renderer.frame(time, input);

  const capture = async () => {
    renderer.frame(time, input);
    const enc = gpu.device.createCommandEncoder({ label: 'dump' });
    sampler.record(enc, renderer.targets.accumRead, W, H);
    gpu.device.queue.submit([enc.finish()]);
    await gpu.device.queue.onSubmittedWorkDone();
    return sampler.read();
  };
  const a = await capture();
  const b = await capture();

  // Alpha is the depth TAG, so it also tells us WHAT each pixel is.
  const cls = { background: [0, 0], body: [0, 0], dynamic: [0, 0] };
  const band = { dark: [0, 0], mid: [0, 0], bright: [0, 0] };
  let peak = 0;
  let peakAt = null;
  const w = sampler.w;
  const diffImg = new Float32Array(a.length / 4);

  for (let i = 0, px = 0; i < a.length; i += 4, px++) {
    let d = 0;
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const x = halfToFloat(a[i + c]);
      const y = halfToFloat(b[i + c]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      d += Math.abs(x - y);
      m += Math.abs(x);
    }
    d /= 3; m /= 3;
    diffImg[px] = d;
    if (d > peak) { peak = d; peakAt = [px % w, Math.floor(px / w)]; }

    const tag = halfToFloat(a[i + 3]);
    const k = tag < -50 ? 'dynamic' : tag < 0 ? 'background' : 'body';
    cls[k][0] += d; cls[k][1]++;
    const bk = m < 0.05 ? 'dark' : m < 0.5 ? 'mid' : 'bright';
    band[bk][0] += d; band[bk][1]++;
  }

  const avg = ([sum, n]) => (n ? +(sum / n).toExponential(3) : null);
  const pct = ([, n]) => +((n / (a.length / 4)) * 100).toFixed(1);

  // Draw the amplified difference where it can be looked at.
  if (opts.draw !== false && typeof document !== 'undefined') {
    let cv = document.getElementById('taa-dump');
    if (!cv) {
      cv = document.createElement('canvas');
      cv.id = 'taa-dump';
      Object.assign(cv.style, {
        position: 'fixed', right: '8px', bottom: '8px', zIndex: 50,
        border: '1px solid #444', imageRendering: 'pixelated',
        width: `${sampler.w}px`, height: `${sampler.h}px`,
      });
      document.body.appendChild(cv);
    }
    cv.width = sampler.w; cv.height = sampler.h;
    const g = cv.getContext('2d');
    const img = g.createImageData(sampler.w, sampler.h);
    const gain = opts.gain ?? 40;
    for (let px = 0; px < diffImg.length; px++) {
      const v = Math.min(255, diffImg[px] * gain * 255);
      img.data[px * 4] = v; img.data[px * 4 + 1] = v;
      img.data[px * 4 + 2] = v; img.data[px * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }

  sampler.destroy();
  return {
    patch: `${sampler.w}x${sampler.h}`,
    byClass: {
      background: { meanDelta: avg(cls.background), coverage: pct(cls.background) },
      body: { meanDelta: avg(cls.body), coverage: pct(cls.body) },
      dynamic: { meanDelta: avg(cls.dynamic), coverage: pct(cls.dynamic) },
    },
    byBrightness: {
      dark: { meanDelta: avg(band.dark), coverage: pct(band.dark) },
      mid: { meanDelta: avg(band.mid), coverage: pct(band.mid) },
      bright: { meanDelta: avg(band.bright), coverage: pct(band.bright) },
    },
    peakDelta: +peak.toExponential(3),
    peakAt,
  };
}

/**
 * TEMPORAL stability of the resolved image, under camera motion.
 *
 * Every other probe here holds the world still, which is the right call for measuring aliasing but
 * makes them blind to the thing that actually looks wrong: shimmer that only appears once the
 * camera moves. `subPixelStability` slides the frame across a pixel with a frozen clock;
 * `lagMetric` measures how fast the history converges. Neither answers "does the settled image sit
 * still while flying", and that is the complaint this exists to quantify.
 *
 * It reads the ACCUMULATION buffer, not the canvas: post-process grain and the additive layers are
 * downstream of the resolve, so including them would measure noise the TAA never sees. Grain is
 * zeroed anyway, and the aperture with it.
 *
 * THE METRIC IS A SECOND DIFFERENCE IN TIME, and the first version of this got it wrong in a way
 * worth recording. A plain frame-to-frame difference under a moving camera is dominated by the
 * motion itself: an edge sweeping across a pixel changes it a lot, legitimately. Because that term
 * scales with image GRADIENT, the first version scored native resolution 8x worse than half
 * resolution — it was measuring sharpness, not stability, and a blurrier image always wins.
 *
 * |L(t+1) - 2L(t) + L(t-1)| cancels any value that is RAMPING linearly, which is what a smooth
 * translation looks like per pixel, while flicker and crawl survive it intact. Normalised per pixel
 * against that pixel's own temporal mean — floored at a fraction of the frame mean, so the near-
 * black sky cannot divide a small absolute wobble into a huge relative one — and reduced with a
 * MEDIAN over pixels, because a handful of specular pixels are not what "everything is shaking"
 * means.
 *
 * The three conditions are a diagnosis rather than a score, and the whole point is the RATIO
 * between them:
 *
 *   frozen  same clock every frame, so nothing moves and every reprojection is the identity.
 *           This is the pipeline's own noise floor. It should be near zero; whatever is here is
 *           per-frame randomness that the resolve is not converging.
 *   drift   the ambient camera, clock advancing. Slow motion, mostly rotation about the subject.
 *   chase   the follow camera behind a flying ship. Fast translation, and the reason the report
 *           came in.
 *
 * frozen high      => something reseeds per frame and TAA cannot average it out.
 * chase >> drift   => the cost is in reprojection or in history rejection, not in the shading.
 *
 * THE SCENE MUST BE STATIONARY or the numbers are not comparable, and this is easy to get wrong.
 * The ship auto-fires every few seconds while cruising, and a rail gun going off mid-run moves the
 * figure by several multiples — enough that one run read 10.7% where the run before it read 1.8%
 * with nothing changed. The ship is marked as flown here, which is exactly what suppresses the
 * autopilot's fire (see Ship#flown), so a run measures the steady state rather than an event.
 *
 * Even so: trust comparisons made WITHIN one call, in one ordering, and distrust them across
 * calls. The particle and ribbon simulations carry state forward from whatever ran before, and
 * nothing here rewinds them.
 *
 * @param {object} knobs { quality, camera, film, probe }
 */
export async function temporalShake(renderer, gpu, knobs, opts = {}) {
  const { quality, camera: cam, film } = knobs;
  const settle = opts.settle ?? 48;
  const frames = opts.frames ?? 60;
  const time = opts.time ?? 6.0;
  const dt = opts.dt ?? 1 / 60;
  const gridW = opts.gridW ?? 320;
  const gridH = opts.gridH ?? 180;
  await ensureSize(renderer);

  const saved = { aperture: cam.aperture, grain: film.grain, dyn: quality.dynamicRes,
                  zoom: cam.zoom, roll: cam.roll };
  cam.aperture = 0;
  film.grain = 0;
  quality.dynamicRes = false;      // a rung change mid-run would be measured as shake

  // `animated` is the one that splits the diagnosis. `drift` moves the camera AND advances the
  // world, so on its own it cannot say which of the two the resolve is failing on. Pinning the
  // camera while the clock runs isolates the animated geometry: reprojection assumes the world is
  // static, and this planet pulses.
  const conditions = opts.conditions ?? [
    { name: 'frozen', advance: false, chase: false },
    { name: 'animated', advance: true, chase: false, pin: true },
    { name: 'drift', advance: true, chase: false },
    { name: 'chase', advance: true, chase: true },
  ];

  const out = [];
  try {
    const t = renderer.targets;
    const sampler = new FrameSampler(gpu.device, t.accumWidth, t.accumHeight, gridW, gridH);
    try {
      for (const c of conditions) {
        // `chase` needs a ship that has been flown, or the camera stays on its ambient path and
        // the condition silently measures `drift` twice.
        const input = benchInput(opts.cmd);
        input.chase = c.chase;
        if (c.chase) { input.cmd.thrust = 1; }
        // Autopilot fire OFF for the duration: a shot during the window is worth several multiples
        // of the figure being measured. Marking the ship flown is the switch for it, and it also
        // stops the barrel roll, which is a second event the metric would otherwise pick up.
        renderer.ship.markFlown();
        input.cmd.fire = false;
        if (c.pin) {
          // The arcball branch takes yaw and pitch straight from the pointer, so a fixed pointer
          // is a fixed camera — except for the dolly and roll, which breathe on the clock and
          // would smuggle camera motion back into the condition that exists to exclude it.
          input.everUsed = true;
          input.x = input.width * 0.5;
          input.y = input.height * 0.5;
          cam.zoom = 0;
          cam.roll = 0;
        } else {
          cam.zoom = saved.zoom;
          cam.roll = saved.roll;
        }

        renderer.resetHistory();
        for (let f = 0; f < settle; f++) renderer.frame(time + (c.advance ? f * dt : 0), input);

        // Two frames of history, because the metric needs three samples in time.
        let m2 = null, m1 = null;
        const n = sampler.w * sampler.h;
        const band = opts.tagBand ?? null;
        // Counted per frame, not sticky. Sticky selected 88% of the frame with large satellites
        // sweeping across it over 20 frames, which is not a mask - it is the whole image with extra
        // steps. A pixel qualifies only if the object was there for MOST of the run, so the
        // measurement is about pixels the object actually occupies rather than ones it passed over.
        const bandHits = band ? new Uint16Array(n) : null;
        const bandNeed = opts.bandFraction ?? 0.7;
        const d2sum = new Float64Array(n);     // per-pixel sum of |second difference|
        const dsum = new Float64Array(n);      // per-pixel sum of |first difference|, for context
        const lsum = new Float64Array(n);      // per-pixel temporal mean, the normaliser
        let counted = 0;
        for (let f = 0; f < frames; f++) {
          renderer.frame(time + (c.advance ? (settle + f) * dt : 0), input);
          const enc = gpu.device.createCommandEncoder({ label: 'shake-copy' });
          sampler.record(enc, renderer.targets.accumRead, t.accumWidth, t.accumHeight);
          gpu.device.queue.submit([enc.finish()]);
          await gpu.device.queue.onSubmittedWorkDone();
          const rgba = decodeHalf(await sampler.read());
          const L = lumaPlane(rgba, sampler.w, sampler.h);
          // SCOPE BY DEPTH TAG when asked. The accumulation buffer's alpha is the hit distance, so a
          // band around an object's distance selects that object's pixels - which is the only way to
          // measure something that covers a small share of the frame. A median over the whole image
          // cannot see a change confined to the satellites however large the change is, and three
          // ablations were run against that blind spot before this existed.
          if (band) {
            for (let i = 0; i < n; i++) {
              const a = rgba[i * 4 + 3];
              if (a >= band[0] && a <= band[1]) bandHits[i]++;
            }
          }
          for (let i = 0; i < n; i++) lsum[i] += L[i];
          if (m1) for (let i = 0; i < n; i++) dsum[i] += Math.abs(L[i] - m1[i]);
          if (m2) {
            for (let i = 0; i < n; i++) d2sum[i] += Math.abs(L[i] - 2 * m1[i] + m2[i]);
            counted++;
          }
          m2 = m1; m1 = L;
        }

        let frameMean = 0;
        for (let i = 0; i < n; i++) frameMean += lsum[i];
        frameMean /= n * frames;
        // The floor is what keeps the sky out of it: a pixel at 1% of the frame mean has no
        // business reporting a 300% relative wobble off a rounding-sized absolute one.
        const floor = Math.max(frameMean * 0.25, 1e-6);
        const flick = new Array(n);
        const rel = new Array(n);
        for (let i = 0; i < n; i++) {
          const den = Math.max(lsum[i] / frames, floor);
          flick[i] = (d2sum[i] / Math.max(counted, 1)) / den;
          rel[i] = (dsum[i] / Math.max(frames - 1, 1)) / den;
        }
        let keptFlick = flick;
        let keptRel = rel;
        let coverage = 1;
        if (band) {
          keptFlick = []; keptRel = [];
          const need = Math.max(1, Math.floor(frames * bandNeed));
          for (let i = 0; i < n; i++) {
            if (bandHits[i] >= need) { keptFlick.push(flick[i]); keptRel.push(rel[i]); }
          }
          coverage = keptFlick.length / n;
          // An empty mask would otherwise report a confident 0% - the worst possible failure for a
          // measurement, because it looks like a fix that worked perfectly.
          if (keptFlick.length === 0) throw new Error(`tagBand [${band}] selected no pixels`);
        }
        out.push({
          name: c.name,
          // The one that means shimmer.
          flick: median(keptFlick) * 100,
          flick95: percentile(keptFlick, 95) * 100,
          // Motion-dominated once the camera moves; useful only against `frozen`.
          mad: median(keptRel) * 100,
          // What share of the frame the band selected, so a suspiciously good number can be checked.
          coverage,
        });
      }
    } finally { sampler.destroy?.(); }
  } finally {
    cam.aperture = saved.aperture;
    film.grain = saved.grain;
    quality.dynamicRes = saved.dyn;
    cam.zoom = saved.zoom;
    cam.roll = saved.roll;
    renderer.resetHistory();
  }
  return out;
}

/**
 * Stability of the FINAL DISPLAYED IMAGE, with the scene stopped dead.
 *
 * Every other probe here reads the accumulation buffer, which is upstream of the composite - so
 * none of them can see the grade, the bloom, the flares, the additive layers or the grain, and for a
 * complaint phrased as "the picture shimmers" that is most of the pipeline missing. This reads the
 * swapchain, the way record.js does, so what it measures is what a viewer sees.
 *
 * IT STOPS EVERYTHING, which no earlier probe did either:
 *   - the clock is fixed, so the camera, rings, pulse, orbits and ship do not move;
 *   - `renderer.held` forces dt to exactly 0, so the particle and ribbon sims do not integrate.
 *     Without that, dt's 1e-4 floor advanced them 0.1ms per frame forever - a slow-motion scene
 *     rather than a stopped one.
 * What is left varying is the temporal jitter and the per-frame dither, which is precisely the
 * question: with nothing moving, is the PIPELINE still?
 *
 * MUST OWN THE LOOP. Measured from the console while the app's own loop was running, the numbers
 * came out 17x worse, because frames from the live clock landed between the two captures. The
 * `beep.still()` wrapper stops the loop first, exactly as the other instruments do; this function
 * assumes that has happened.
 *
 * Reports a DISTRIBUTION, not a mean. The mean frame-to-frame delta here is a third of one 255-level
 * step, which sounds like nothing and is - the shimmer is a small number of pixels swinging by 100
 * levels or more, and only counts past a threshold show that.
 */
export async function finalStability(renderer, gpu, knobs, opts = {}) {
  const { film, probe, camera: cam } = knobs;
  const settle = opts.settle ?? 45;
  const pairs = opts.pairs ?? 3;
  const time = opts.time ?? 7.0;
  const grain = opts.grain ?? false;      // OFF by default: real grain is meant to move
  // APERTURE OFF by default, as every older probe here does - and the first version of this one
  // forgot, which produced a wrong conclusion worth recording. `PROBE.zeroJitter` only zeroes
  // jitter.xy, the PIXEL jitter; jitter.zw is the lens offset for depth of field, and it resamples
  // the lens every frame. So "jitter off" left a per-frame random source running and the frozen
  // scene looked irreducibly noisy. With both off it is bit-exact: mean 0.000, peak 1 of 255.
  const aperture = opts.aperture ?? false;
  const input = benchInput(opts.cmd);
  await ensureSize(renderer);

  const canvas = gpu.canvas;
  const w = canvas.width;
  const h = canvas.height;
  if (w < 64 || h < 64) throw new Error(`canvas has no usable size (${w}x${h})`);
  const bpr = Math.ceil(w * 4 / 256) * 256;
  const buf = gpu.device.createBuffer({
    label: 'still-readback',
    size: bpr * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const saved = { grain: film.grain, held: renderer.held, zeroJitter: probe.zeroJitter,
                  aperture: cam.aperture };
  if (!grain) film.grain = 0;
  if (!aperture) cam.aperture = 0;
  renderer.held = true;
  if (opts.zeroJitter !== undefined) probe.zeroJitter = opts.zeroJitter ? 1 : 0;

  const grab = async () => {
    renderer.frame(time, input);
    const enc = gpu.device.createCommandEncoder({ label: 'still-copy' });
    enc.copyTextureToBuffer({ texture: gpu.context.getCurrentTexture() },
                            { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
    gpu.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap();
    return bytes;
  };

  try {
    renderer.resetHistory();
    for (let i = 0; i < settle; i++) renderer.frame(time, input);

    // FOUR consecutive frames, compared BOTH ways. With every per-frame input disabled and the scene
    // stopped, anything left has to be structural, and the ping-pong is the structure: the
    // accumulation, the embers and the bloom all alternate buffers by frame parity. A difference
    // between ADJACENT frames that disappears between SAME-PARITY frames is a period-2 alternation -
    // two images being shown in turn - which reads as constant shimmer at half the frame rate and
    // cannot be explained by any noise source, because there is no noise left.
    const f = [await grab(), await grab(), await grab(), await grab()];
    const compare = (a, b) => {
      let over4 = 0, over16 = 0, over48 = 0, peak = 0, sum = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * bpr + x * 4;
          const la = (a[i] + a[i + 1] + a[i + 2]) / 3;
          const lb = (b[i] + b[i + 1] + b[i + 2]) / 3;
          const d = Math.abs(la - lb);
          sum += d;
          if (d > peak) peak = d;
          if (d > 4) over4++;
          if (d > 16) over16++;
          if (d > 48) over48++;
        }
      }
      const px = w * h;
      return { mean: sum / px, peak, over4: (over4 / px) * 100,
               over16: (over16 / px) * 100, over48: (over48 / px) * 100 };
    };
    return {
      size: [w, h],
      adjacent: compare(f[0], f[1]),
      adjacent2: compare(f[1], f[2]),
      sameParity: compare(f[0], f[2]),
      sameParity2: compare(f[1], f[3]),
    };
  } finally {
    buf.destroy();
    film.grain = saved.grain;
    renderer.held = saved.held;
    probe.zeroJitter = saved.zeroJitter;
    cam.aperture = saved.aperture;
    renderer.resetHistory();
  }
}
