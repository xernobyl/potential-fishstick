/**
 * Entry point: bring up the device, wire input, run the loop.
 *
 * Deliberately thin. Anything that could belong to the renderer or the scene
 * lives there instead, so this file stays readable as "what the page does".
 */

import { Gpu } from './core/device.js';
import { Renderer, LENS_DISK } from './renderer.js';
import { Input } from './scene/input.js';
import * as TUNING from './scene/tuning.js';
import { benchmark, lensResidual, dumpFrames, lagMetric, compareConfigs, fieldEvalCount, matchedSharpness, detailSnr, additiveAliasing, subPixelStability, temporalShake, finalStability, grabFrame } from './dev/benchmark.js';

const canvas = document.getElementById('gpu');
const perfEl = document.getElementById('perf');
const fatalEl = document.getElementById('fatal');

function fatal(title, detail) {
  fatalEl.style.display = 'grid';
  fatalEl.innerHTML = '';
  const b = document.createElement('b');
  b.textContent = title;
  fatalEl.append(b, document.createTextNode(detail ?? ''));
  console.error(title, detail);
}

async function boot() {
  let gpu;
  try {
    gpu = await Gpu.create(canvas, {
      onLost: (info) => fatal('The GPU device was lost.', info.message),
    });
  } catch (e) {
    fatal('Cannot start.', e.message);
    return;
  }

  const renderer = new Renderer(gpu);
  try {
    await renderer.init();
  } catch (e) {
    fatal('Pipeline setup failed.', `${e.message}\n\nSee the console for shader diagnostics.`);
    return;
  }

  const input = new Input(canvas);

  // F toggles fullscreen, G the tuning panel. Bound on the document so they work
  // without the canvas having focus, and ignored while a modifier is held so they
  // never eat a browser shortcut.
  let gui = null;
  let recording = false;
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'f') {
      e.preventDefault();
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.().catch(() => {});
    } else if (k === 'r') {
      e.preventDefault();
      if (recording) { console.warn('already recording'); return; }
      recording = true;
      // Imported on demand, like the panel: a video encoder and a muxer have no business in the
      // startup path of a renderer.
      import('./dev/record.js').then(async (m) => {
        const wasRunning = running;
        running = false;
        let last = -1;
        try {
          console.log('recording 15s @ 1080p30 — the render is offline, so this takes longer than 15s');
          const r = await m.recordVideo(renderer, gpu, TUNING.QUALITY, {
            seconds: 15, fps: 30, width: 1920, height: 1080,
            onProgress: ({ frame, total }) => {
              const pc = Math.floor((frame / total) * 10);
              if (pc !== last) { last = pc; console.log(`  ${frame}/${total}`); }
            },
          });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(r.blob);
          a.download = 'beep-1080p30.mp4';
          a.click();
          URL.revokeObjectURL(a.href);
          console.log(`recorded ${r.frames} frames, ${(r.blob.size / 1e6).toFixed(1)} MB, `
            + `${(r.encodeMs / 1000).toFixed(1)}s wall`);
        } catch (err) {
          console.error('recording failed', err);
        } finally {
          recording = false;
          if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
        }
      });
    } else if (k === 't') {
      e.preventDefault();
      setFrozen(!frozen);
    } else if (k === 'g') {
      e.preventDefault();
      if (gui) { gui.destroy(); gui = null; return; }
      // Imported on demand: lil-gui is 59 KB of debug surface and has no business in the
      // startup path of a renderer whose whole subject is frame time.
      import('./dev/gui.js')
        .then((m) => { gui = m.buildGui(renderer, live); })
        .catch((err) => console.error('tuning panel failed to load', err));
    }
  });

  // Declared before anything that touches it: `let` is not hoisted-initialised,
  // so reading it earlier is a ReferenceError rather than an undefined.
  let running = true;

  // The benchmark drives renderer.frame() itself, so the rAF loop has to stand
  // down for the duration or the two interleave and both measurements are junk.
  async function bench(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await benchmark(renderer, gpu, opts);
      reportBenchmark(r);
      return r;
    } finally {
      if (wasRunning) {
        running = true;
        renderer.resetHistory();     // the synthetic clock jumped
        requestAnimationFrame(loop);
      }
    }
  }

  /** Both temporal measurements together, because neither means anything alone. */
  async function lag(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await lagMetric(renderer, gpu, opts);
      console.log(reportLag(r));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /** Does changing this config change the IMAGE? The check an accelerator needs. */
  async function compare(mutate, restore, opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await compareConfigs(renderer, gpu, mutate, restore, opts);
      console.log('[compare]', JSON.stringify(r, null, 2));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /**
   * Detail vs noise against a native-resolution ground truth — the one measurement that can tell
   * a reconstruction filter's recovered detail from the grain it adds. Slow on purpose: it
   * converges four configurations, one of them at full resolution.
   */
  async function detail(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await detailSnr(renderer, gpu, TUNING.QUALITY, opts);
      console.log(reportDetail(r));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /**
   * Does the image flicker when the camera moves less than one pixel? The crawl measurement.
   * Neutralises every per-frame random source first — see the note on the instrument.
   */
  async function subpixel(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await subPixelStability(renderer, gpu, {
        quality: TUNING.QUALITY, camera: TUNING.CAMERA, film: TUNING.FILM, probe: TUNING.PROBE,
      }, opts);
      console.log(reportSubpixel(r));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /**
   * Does the SETTLED image sit still while the camera moves? Reports the pipeline's noise floor
   * (frozen), the ambient camera, and the chase camera, so the ratio says where the shimmer is.
   */
  async function shake(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await temporalShake(renderer, gpu, {
        quality: TUNING.QUALITY, camera: TUNING.CAMERA, film: TUNING.FILM, probe: TUNING.PROBE,
      }, opts);
      console.log('[shake] flicker in the resolved image, % of each pixel\'s own level');
      console.log('  flick = |second difference| in time, which cancels smooth motion; mad is the');
      console.log('  plain frame-to-frame change, motion-dominated and only meaningful vs frozen.');
      for (const c of r) {
        console.log(`  ${c.name.padEnd(8)} flick ${c.flick.toFixed(3)}%  p95 ${c.flick95.toFixed(3)}%`
          + `   mad ${c.mad.toFixed(3)}%`);
      }
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /**
   * With the scene stopped DEAD - clock fixed, dt zero, so not one thing moves - does the final
   * displayed image still change? The only probe that reads the swapchain rather than the
   * accumulation buffer, so the only one that can see the grade, the additive layers and the grain.
   */
  async function still(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await finalStability(renderer, gpu, {
        film: TUNING.FILM, probe: TUNING.PROBE,
      }, opts);
      const row = (k, v) => console.log(`  ${k.padEnd(12)} mean ${v.mean.toFixed(3)}  peak `
        + `${v.peak.toFixed(0).padStart(3)}   >4 ${v.over4.toFixed(3)}%  >16 ${v.over16.toFixed(3)}%`
        + `  >48 ${v.over48.toFixed(4)}%`);
      console.log(`[still] ${r.size[0]}x${r.size[1]}, scene stopped dead`);
      row('adjacent', r.adjacent);
      row('adjacent2', r.adjacent2);
      row('sameParity', r.sameParity);
      row('sameParity2', r.sameParity2);
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /** Does drawing the additive layer at render resolution change what you see, and at what cost? */
  async function additive(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await additiveAliasing(renderer, gpu, TUNING.QUALITY, opts);
      console.log('[additive]', JSON.stringify(r, null, 2));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /**
   * Save a PNG of the current frame.
   *
   * Not a nicety: a WebGPU canvas cannot be captured with `drawImage`, which returns transparent
   * black because the swapchain texture is released at present time. The frame has to be copied
   * out of the texture during the frame that drew it, which means the renderer has to cooperate —
   * hence a dev helper rather than something the browser can do for you.
   */
  async function shot(scale = 1) {
    const wasRunning = running;
    running = false;
    try {
      const png = await grabFrame(renderer, gpu, scale);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(png);
      a.download = `beep-${renderer.targets.displayWidth}x${renderer.targets.displayHeight}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
      return `${(png.size / 1024).toFixed(0)} KB`;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  /** Mean field evaluations per pixel, counted in the shader rather than reasoned about. */
  async function evals(opts) {
    const wasRunning = running;
    running = false;
    try {
      const r = await fieldEvalCount(renderer, gpu, TUNING.PROBE,
        { div: TUNING.PROBE_EVAL_DIV, ...opts });
      console.log('[evals]', JSON.stringify(r));
      return r;
    } finally {
      if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
    }
  }

  // A handle for poking at the scene from the console. Tuning values are read
  // per frame, so most of them can be changed live.
  window.beep = {
    renderer, gpu, input, tuning: TUNING, bench, lag, compare, evals, detail, additive, subpixel, shake, still, shot,
    /** Freeze/unfreeze the clock and input; `t` does the same. Returns the new state. */
    freeze: (on) => setFrozen(on === undefined ? !frozen : !!on),
    /** Sharpness of several configs on one display-resolution grid. */
    sharp: async (configs, opts) => {
      const wasRunning = running;
      running = false;
      try {
        const r = await matchedSharpness(renderer, gpu, configs, opts);
        console.log('[sharp]', JSON.stringify(r, null, 2));
        return r;
      } finally {
        if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
      }
    },
    lens: () => lensResidual(LENS_DISK, TUNING.TEMPORAL.blend, TUNING.CAMERA.aperture),
    /** Two sequential frames of the accumulation buffer, and where they differ. */
    dump: async (opts) => {
      const wasRunning = running;
      running = false;
      try {
        const r = await dumpFrames(renderer, gpu, opts);
        console.log('[dump]', JSON.stringify(r, null, 2));
        return r;
      } finally {
        if (wasRunning) { running = true; renderer.resetHistory(); requestAnimationFrame(loop); }
      }
    },
  };

  // ?bench runs it headless and skips the interactive loop entirely, so it works
  // in a hidden or backgrounded tab where rAF never fires.
  const q = new URLSearchParams(location.search);
  if (q.has('bench')) {
    running = false;
    const frames = Number(q.get('frames')) || undefined;
    await bench({ frames, stability: q.get('bench') === 'stability' });
    return;
  }

  // Re-render on resize rather than waiting for the next tick, so the canvas
  // never shows a stretched stale frame.
  const ro = new ResizeObserver(() => renderer.resize());
  ro.observe(canvas);

  // Pause when hidden: a background tab still gets rAF in some browsers, and
  // burning GPU on an invisible canvas is rude.
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    if (running) {
      // The clock jumped; the history is stale by an unknown amount.
      renderer.resetHistory();
      requestAnimationFrame(loop);
    }
  });

  // NOT const: freezing rebases it on resume, so the scene continues from where it stopped instead
  // of jumping forward by however long it was held.
  let start = performance.now();
  let frozen = false;
  let frozenTime = 0;
  let frozenInput = null;
  let lastHud = 0;
  let frames = 0;
  let fps = 0;
  // Shared with the tuning panel, which displays it. One object rather than a getter so the
  // panel can `.listen()` to it without knowing anything about this loop.
  const live = { fps: 0 };

  /**
   * Hold or release the clock and the input.
   *
   * There for the question "does it still shimmer when NOTHING is moving?", which no amount of
   * staring at a moving scene can answer: with the world in motion, real motion and pipeline
   * shimmer look the same. `beep.shake()` measures the same state as its `frozen` condition; this is
   * the version you can look at.
   */
  function setFrozen(on) {
    if (on === frozen) return;
    frozen = on;
    // Not just the clock: dt is forced to zero too, or the particle sims keep creeping on the
    // clamped minimum step and the scene is in slow motion rather than stopped.
    renderer.held = frozen;
    if (frozen) {
      frozenTime = (performance.now() - start) / 1000;
      // A DEEP snapshot: `input.state()` returns one shared object that it refreshes in place, so
      // keeping the reference would keep tracking the live pointer - the opposite of frozen.
      const live = input.state(canvas);
      frozenInput = { ...live, cmd: { ...live.cmd } };
    } else {
      start = performance.now() - frozenTime * 1000;
    }
    // The accumulation is discarded either way: it was built under a different motion regime, and
    // letting it carry over would show a settling transient that belongs to the transition rather
    // than to whichever state is being judged.
    renderer.resetHistory();
    return frozen;
  }

  function loop(now) {
    if (!running) return;
    // FROZEN holds the clock AND the input. The clock alone stops the world - the camera drift, the
    // rings, the pulse, the orbits and the ship all derive from it, and dt falls to zero - but the
    // arcball reads the pointer directly, so without holding the input too, a mouse anywhere near
    // the canvas would still move the camera. Nothing in the scene moves in this state, which makes
    // it the state to look at when asking whether the PIPELINE is stable: everything left is the
    // temporal jitter and the per-frame dither.
    const time = frozen ? frozenTime : (now - start) / 1000;

    try {
      renderer.frame(time, frozen ? frozenInput : input.state(canvas));
    } catch (e) {
      fatal('Render failed.', e.message);
      return;
    }

    frames++;
    if (now - lastHud > 500) {
      fps = (frames * 1000) / (now - lastHud);
      live.fps = fps;
      frames = 0;
      lastHud = now;
      updateHud(renderer, gpu, fps);
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

/** Console report for the sub-pixel stability sweep. */
function reportSubpixel(r) {
  const lines = [
    `${r.steps} offsets across ${r.span} display px   ${r.settle} frames each   grid ${r.grid}`,
    '',
    'config                target   size          mean CV   band CV   worst step',
  ];
  for (const x of r.rows) {
    lines.push(`${x.config.padEnd(21)} ${x.target.padEnd(8)} ${x.size.padEnd(13)} `
      + `${(x.meanCV + '%').padStart(7)}   ${(x.bandCV + '%').padStart(7)}`
      + `   ${(x.worstStep + '%').padStart(10)}`);
  }
  lines.push('',
    '  mean CV     how much the mean brightness wobbles as the image slides across the grid.',
    '              A translation cannot change it, so anything here is aliasing.',
    '  band CV     the same for high-frequency energy — sensitive to thin features beating',
    '              against the sampling grid, which is what crawl looks like.',
    '  worst step  the largest jump between ADJACENT sub-pixel offsets, as a share of the mean.',
    '              The perceptual one: a smooth drift across a pixel is invisible, a step is not.');
  return lines.join('\n');
}

/** Console report for the detail/noise decomposition. */
function reportDetail(r) {
  const lines = [
    `grid ${r.grid}   sigma ${r.sigma} px   settle ${r.settle} frames`,
    '',
    'config       accum          magnitude   signal   noise    psnr   retained',
  ];
  for (const c of r.configs) {
    lines.push(`${c.name.padEnd(12)} ${c.accum.padEnd(14)} `
      + `${c.magnitude.toFixed(4).padStart(9)}   ${c.signal.toFixed(4).padStart(6)}`
      + `   ${c.noise.toFixed(4).padStart(6)}   ${String(c.psnr).padStart(6)}`
      + `   ${c.retained.toFixed(4).padStart(8)}`);
  }
  // Spelled out, because the raw numbers invite exactly the reading they exist to prevent.
  lines.push(
    '',
    `noise floor  ${r.noiseFloor.toFixed(4)} — the CONTROL is the reference's own configuration, so`,
    '             its true signal is 1 and its true noise 0. What it reports instead is this',
    "             method's floor, and no noise figure below it means anything.",
    '',
    '  magnitude  total high-frequency energy, 1 = the reference. This alone is what a',
    '             sharpness metric reports, and it cannot tell the next two apart.',
    '  signal     how much of the TRUE detail is present. Below 1 is detail lost.',
    '  noise      high-frequency energy that is not the truth. Read against the floor.',
    '             magnitude^2 = signal^2 + noise^2 — the two are orthogonal by construction.',
    `  retained   signal / the control's signal, i.e. the fraction of NATIVE detail kept.`,
    `             The control reads ${r.convergence.toFixed(4)} rather than 1, because the reference`,
    '             carries its own residual; dividing by it removes that from every row.');
  const taau = r.configs.find((c) => c.name === 'taau');
  const low = r.configs.find((c) => c.name === 'lowres');
  if (taau && low) {
    const dn = (taau.noise - low.noise) * 100;
    lines.push('',
      `verdict      upsampling retains ${(taau.retained * 100).toFixed(0)}% of native detail;`
      + ` a plain upscale retains ${(low.retained * 100).toFixed(0)}%,`,
      `             for ${dn >= 0 ? '+' : ''}${dn.toFixed(1)} points of extra noise`
      + ` and ${(taau.psnr - low.psnr).toFixed(2)} dB of PSNR`);
  }
  return lines.join('\n');
}

/** Console report for a benchmark run. */
function reportLag(r) {
  const pct = (v) => `${(v * 100).toFixed(3)}%`;
  // The verdict is spelled out rather than left to the reader, because the raw lag
  // number is meaningless without the noise floor beside it and that is precisely the
  // comparison it is tempting to skip.
  const n = r.lagOverNoise;
  const verdict = n <= 1.3 ? 'not distinguishable from noise'
    : n <= 2.0 ? 'slight — visible only on high-contrast edges'
      : n <= 4.0 ? 'real trailing'
        : 'heavy ghosting';
  return [
    `mode        lag   samples ${r.samples}   patch ${r.patch}`,
    `LAG         ${pct(r.medianLag)} median   ${pct(r.p95Lag)} p95   <- the regression signal`,
    `noise floor ${pct(r.medianResidual)}   (same frames, so directly comparable)`,
    `ratio       ${n.toFixed(2)}x noise — ${verdict}`,
    '            ^ a VISIBILITY test, not a regression signal. The ratio moves when either',
    '            term moves, and the floor moves far more: measured across render scales and',
    '            aperture settings, absolute lag held at 6.0-7.7% while the floor ran 0.65-2.33%,',
    '            so the ratio swung 2.6x-11.3x. Zeroing the aperture — which can only improve the',
    '            image — gives the WORST ratio of all, because it removes the noise it is divided',
    '            by. Compare absolute lag between builds; read the ratio only to ask whether the',
    '            lag that exists is visible above the sampling noise.',
    `sharpness   ${r.sharpness.toFixed(4)} moving   ${r.refSharpness.toFixed(4)} converged`,
    '            (higher is more retained detail; only the moving figure sees the filter)',
  ].join('\n');
}

function reportBenchmark(r) {
  const lines = [
    `mode        ${r.mode}${r.serial ? ' (serial)' : ''}`,
    `resolution  ${r.resolution}   frames ${r.frames}`,
    `frame       ${r.meanFrameMs.toFixed(3)} ms mean   ${r.wallMedianMs.toFixed(3)} median   ${r.wallP95Ms.toFixed(3)} p95`,
  ];
  if (!r.frameMsIsThroughput) {
    lines.push('            ^ serial mode stalls every frame — NOT a throughput figure');
  }
  if (!r.timestamps) {
    lines.push('(timestamp-query unavailable — GPU pass breakdown not possible)');
  } else {
    for (const p of r.passes) {
      lines.push(`  ${p.label.padEnd(11)} ${p.median.toFixed(3)} ms median   ${p.p95.toFixed(3)} p95`);
    }
    lines.push(`  ${'gpu total'.padEnd(11)} ${r.gpuTotalMs.toFixed(3)} ms`);
    if (r.stability) {
      const st = r.stability;
      lines.push(`stability   ${st.medianDelta.toExponential(2)} linear HDR median`
               + `   ${(st.medianRelative * 100).toFixed(3)}% of brightness`,
                 `            p95 ${st.p95Delta.toExponential(2)}   patch ${st.patch}`
               + `   n=${st.samples}`);
    }
    for (const g of r.aliasedPasses ?? []) {
      lines.push(`  !! ${g.join(' = ')} report ONE shared interval, not ${g.length} measurements`);
    }
    if (r.contaminated) {
      lines.push('  !! breakdown is not attributable — pipelined submission let pass',
                 '     intervals overlap. Re-run with {serial:true} for the split;',
                 '     trust THIS run only for the frame time.');
    }
  }
  console.log(lines.join('\n'));
}

function updateHud(renderer, gpu, fps) {
  const lines = [`${fps.toFixed(0)} fps   ${renderer.targets.width}x${renderer.targets.height}`];
  const report = renderer.profiler.report();
  if (report.length) {
    for (const [label, ms] of report) {
      lines.push(`${label.padEnd(11)} ${ms.toFixed(3)} ms`);
    }
    lines.push(`${'gpu total'.padEnd(11)} ${renderer.profiler.total().toFixed(3)} ms`);
  } else if (!gpu.caps.timestamps) {
    lines.push('(timestamp-query unavailable)');
  }
  perfEl.textContent = lines.join('\n');
}

boot();
