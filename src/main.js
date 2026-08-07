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
import { benchmark, lensResidual, dumpFrames, lagMetric, compareConfigs, fieldEvalCount, matchedSharpness, detailSnr, additiveAliasing } from './dev/benchmark.js';

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
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'f') {
      e.preventDefault();
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.().catch(() => {});
    } else if (k === 'g') {
      e.preventDefault();
      if (gui) { gui.destroy(); gui = null; return; }
      // Imported on demand: lil-gui is 59 KB of debug surface and has no business in the
      // startup path of a renderer whose whole subject is frame time.
      import('./dev/gui.js')
        .then((m) => { gui = m.buildGui(renderer); })
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
    renderer, gpu, input, tuning: TUNING, bench, lag, compare, evals, detail, additive,
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

  const start = performance.now();
  let lastHud = 0;
  let frames = 0;
  let fps = 0;

  function loop(now) {
    if (!running) return;
    const time = (now - start) / 1000;

    try {
      renderer.frame(time, input.state(canvas));
    } catch (e) {
      fatal('Render failed.', e.message);
      return;
    }

    frames++;
    if (now - lastHud > 500) {
      fps = (frames * 1000) / (now - lastHud);
      frames = 0;
      lastHud = now;
      updateHud(renderer, gpu, fps);
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
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
    `lag         ${pct(r.medianLag)} median   ${pct(r.p95Lag)} p95`,
    `noise floor ${pct(r.medianResidual)}   (same frames, so directly comparable)`,
    `ratio       ${n.toFixed(2)}x noise — ${verdict}`,
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
