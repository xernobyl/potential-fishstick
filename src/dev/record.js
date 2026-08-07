/**
 * Record the scene to an MP4, OFFLINE.
 *
 * "Offline" is the whole point and it is what makes this different from screen capture. The frames
 * are driven on a SYNTHETIC clock at exactly 1/fps and each is stamped with the time it represents,
 * so a frame that takes 90 ms to march still occupies 33.3 ms of video. The output is constant
 * frame rate by construction, at whatever resolution is asked for, however slow the renderer is.
 *
 * That is only cheap here because `renderer.frame(time, input)` already takes its clock as an
 * argument — the same property every instrument in benchmark.js relies on. This is that loop with
 * an encoder attached.
 *
 * WHY NOT MediaRecorder. `canvas.captureStream()` plus MediaRecorder is the obvious route and it is
 * the wrong one: it is a realtime pipeline that timestamps frames by wall clock. At 25 fps of render
 * you get dropped and duplicated frames and a variable frame rate, and `requestFrame()` does not fix
 * the timestamps. WebCodecs hands you the timestamp, which is exactly the control this needs.
 *
 * WHY THE READBACK. A WebGPU canvas cannot be sampled by anything that treats it as an image source
 * — `drawImage` returns transparent black, because the swapchain texture is released at present
 * time — so `new VideoFrame(canvas)` is not available. The frame is copied out of the swapchain
 * texture during the frame that drew it, exactly as `grabFrame` does, and handed to `VideoFrame` as
 * a raw buffer. That also lets the surface's native BGRA go straight through with no swizzle: the
 * copy is padded to a 256-byte row stride, which `layout` describes rather than requiring a repack.
 */

import { Muxer, ArrayBufferTarget } from '../../vendor/mp4-muxer.mjs';
import { benchInput } from './benchmark.js';

/** Bitrate for a given pixel count and rate. ~0.1 bits per pixel per frame is visually clean for
 *  H.264 on this kind of content — smooth gradients and glow, little hard detail. */
function defaultBitrate(w, h, fps) {
  return Math.round(w * h * fps * 0.1);
}

/**
 * @param {object} opts
 * @param {number} [opts.seconds]  duration of the FINISHED video
 * @param {number} [opts.fps]
 * @param {number} [opts.width]    output resolution; the canvas is resized to it
 * @param {number} [opts.height]
 * @param {number} [opts.warmup]   frames rendered before capture starts, to converge the history
 * @param {number} [opts.startTime] synthetic clock at frame 0
 * @param {(p: {frame: number, total: number}) => void} [opts.onProgress]
 * @returns {Promise<{blob: Blob, frames: number, encodeMs: number}>}
 */
export async function recordVideo(renderer, gpu, quality, opts = {}) {
  const fps = opts.fps ?? 30;
  const seconds = opts.seconds ?? 15;
  const width = (opts.width ?? 1920) & ~1;      // H.264 wants even dimensions
  const height = (opts.height ?? 1080) & ~1;
  const total = Math.round(seconds * fps);
  // Longer than the interactive default. At full resolution the history has four times the samples to
  // converge and the first frame of the file is the one most likely to be looked at closely; the cost is
  // wall-clock time during an export that is already not real time.
  const warmup = opts.warmup ?? 96;
  const t0 = opts.startTime ?? 0;
  // The SAME neutral command the instruments use, rather than a second copy of one. Recording with
  // live input would let a stray keypress steer the shot, and the ship cruises on its own until
  // flown — so an empty command is exactly what shows the trails.
  const input = benchInput(opts.cmd);

  if (typeof VideoEncoder === 'undefined') {
    throw new Error('WebCodecs VideoEncoder is unavailable in this browser');
  }

  // H.264 High profile, level 4.0 — 1080p30 is inside it, and High is what every player expects.
  const codec = 'avc1.640028';
  const config = {
    codec,
    width,
    height,
    bitrate: opts.bitrate ?? defaultBitrate(width, height, fps),
    framerate: fps,
    // Length-prefixed rather than Annex-B, which is what an MP4 sample entry holds.
    avc: { format: 'avc' },
  };
  const support = await VideoEncoder.isConfigSupported(config);
  if (!support.supported) throw new Error(`encoder rejected ${codec} at ${width}x${height}`);

  const muxer = new Muxer({
    target: new ArrayBufferTarget(),
    video: { codec: 'avc', width, height, frameRate: fps },
    // In-memory: the whole file is assembled before writing, which puts the index at the FRONT so
    // the result is seekable and streamable rather than needing a full download before it plays.
    fastStart: 'in-memory',
  });

  let encodeError = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => { encodeError = e; },
  });
  encoder.configure(config);

  // ---- take over the canvas ----
  //
  // Saved and restored, all of it: the recorder owns the resolution and the quality settings for
  // the duration, and leaving any of them changed afterwards would be a trap.
  const canvas = gpu.canvas;
  const saved = {
    w: canvas.width, h: canvas.height,
    style: { width: canvas.style.width, height: canvas.style.height },
    maxWidth: quality.maxWidth,
    dynamicRes: quality.dynamicRes,
    scale: quality.renderScale,
    additive: quality.additiveDisplayRes,
  };

  // FULL RESOLUTION AND THE BEST QUALITY THE SETTINGS OFFER, for the whole recording.
  //
  // The interactive defaults are a frame-rate compromise: the march runs at `renderScale` (0.5, so a
  // quarter of the pixels) and TAAU upsamples it, which is the right trade at 60 fps and the wrong one
  // for a file that gets watched frame by frame. A recording is not real time — every frame is rendered
  // on a synthetic clock and read back, so it can cost whatever it costs.
  //
  // What this buys, concretely: the body is MARCHED at output resolution rather than reconstructed from
  // a quarter-resolution history, which is most of the silhouette detail; and the additive layer
  // (embers, contrails, rail guns, auroras) rasterises at display resolution rather than being upscaled,
  // which `beep.additive()` measured as the larger of the two effects on those elements.
  //
  // Restored in the `finally` below, all of it — leaving any of these changed would be a trap.
  quality.dynamicRes = false;      // it would change the image quality partway through the shot
  quality.renderScale = 1.0;       // march at output resolution, no temporal upsampling to do
  quality.additiveDisplayRes = true;
  // `maxWidth` caps the swapchain, and `syncSize` re-derives the canvas from its CSS box every
  // frame — so the cap is what actually pins the resolution here, not the backing store.
  quality.maxWidth = width;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const bpr = Math.ceil(width * 4 / 256) * 256;
  const rows = [];      // a small ring of readback buffers, so the GPU is not stalled every frame
  const RING = 3;
  for (let i = 0; i < RING; i++) {
    rows.push(gpu.device.createBuffer({
      label: `record-readback-${i}`,
      size: bpr * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }));
  }
  // BGRA straight from the surface where that is what it is, so no channel swizzle is needed.
  const format = gpu.format.startsWith('bgra') ? 'BGRA' : 'RGBA';

  const t0ms = performance.now();
  let frames = 0;
  try {
    renderer.resize();
    if (renderer.targets.displayWidth !== width) {
      // Not fatal — the encoder is configured for `width`, so a mismatch would corrupt every frame.
      throw new Error(`canvas settled at ${renderer.targets.displayWidth}px, not ${width}px`);
    }

    // WARM UP. The accumulation buffer needs to converge before frame 0, or the first second of the
    // video is the noisy transient that TAA exists to remove. Held at the start time so the warm-up
    // converges the frame that is about to be recorded rather than a different one.
    renderer.resetHistory();
    for (let i = 0; i < warmup; i++) renderer.frame(t0, input);

    for (let i = 0; i < total; i++) {
      if (encodeError) throw encodeError;
      const time = t0 + i / fps;
      renderer.frame(time, input);

      const buf = rows[i % RING];
      const tex = gpu.context.getCurrentTexture();
      const enc = gpu.device.createCommandEncoder({ label: 'record-copy' });
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr },
                              { width, height });
      gpu.device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      // Copied out of the mapped range: VideoFrame takes ownership of what it is given, and the
      // range is invalidated by unmap.
      const bytes = new Uint8Array(buf.getMappedRange().slice(0));
      buf.unmap();

      const frame = new VideoFrame(bytes, {
        format,
        codedWidth: width,
        codedHeight: height,
        // MICROSECONDS, and derived from the frame index rather than accumulated — accumulating a
        // rounded step is how a long capture drifts out of sync with its own frame rate.
        timestamp: Math.round((i * 1e6) / fps),
        duration: Math.round(1e6 / fps),
        layout: [{ offset: 0, stride: bpr }],
      });
      // A keyframe every second: enough for seeking without spending much bitrate on it.
      encoder.encode(frame, { keyFrame: i % fps === 0 });
      frame.close();

      // Backpressure. The encoder is asynchronous and the march is slow, so this rarely trips —
      // but without it a fast scene would queue hundreds of 1080p frames into memory.
      while (encoder.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 0));

      frames = i + 1;
      opts.onProgress?.({ frame: frames, total });
    }

    await encoder.flush();
    if (encodeError) throw encodeError;
    muxer.finalize();
    return {
      blob: new Blob([muxer.target.buffer], { type: 'video/mp4' }),
      frames,
      encodeMs: performance.now() - t0ms,
    };
  } finally {
    // Order matters: close the encoder before the buffers it might still be reading through.
    try { encoder.close(); } catch { /* already closed by flush's error path */ }
    for (const b of rows) b.destroy();
    quality.maxWidth = saved.maxWidth;
    quality.dynamicRes = saved.dynamicRes;
    quality.renderScale = saved.scale;
    quality.additiveDisplayRes = saved.additive;
    canvas.style.width = saved.style.width;
    canvas.style.height = saved.style.height;
    renderer.resize();
    renderer.resetHistory();
  }
}
