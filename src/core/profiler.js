/**
 * GPU timing via timestamp queries.
 *
 * Wall-clock frame time tells you almost nothing about a GPU-bound renderer —
 * requestAnimationFrame is pinned to the display and the CPU is idle waiting.
 * Timestamps are the only way to see which pass actually costs what.
 *
 * Optional by design: when the feature is absent every method is a no-op and the
 * renderer runs unchanged.
 *
 * Readback uses a small pool of mappable buffers. A buffer being mapped cannot
 * be a copy destination, so reusing one immediately would either stall or throw;
 * the pool lets timings arrive a frame or two late instead, which is fine for a
 * HUD.
 */

const MAX_PASSES = 16;
const POOL = 3;

/**
 * Per-pass milliseconds from one frame's raw begin/end counters.
 *
 * The obvious `end - begin` is WRONG on at least one backend, and the raw counters say
 * so unambiguously. Measured here on Metal, in a single frame:
 *
 *   tilecull    begin  0.000  end  0.000
 *   raymarch    begin  0.000  end 34.865
 *   rings       begin  0.000  end 36.831
 *   taa         begin 36.831  end 37.290
 *   ember-draw  begin  8.192  end 37.290
 *
 * Four passes report `begin` at the IDENTICAL counter tick, and ember-draw spans 29 ms
 * while ending at the same instant taa does — for a pass the HUD has at 0.05 ms. The
 * beginning-of-pass writes are simply not per-pass here. The ends, though, are monotonic
 * and land where each pass actually finished.
 *
 * Passes on one queue execute serially, so the cost of pass i is the time between the
 * previous pass finishing and this one finishing. That reconstruction is self-consistent
 * by construction — the parts sum to the whole — and on the frame above it recovers
 * rings at 1.97 ms where `end - begin` claimed 36.8, and reproduces every small pass to
 * within the counter's resolution.
 *
 * When the begins ARE self-consistent (each at or after the previous end) they carry real
 * information about inter-pass gaps, so they are used directly and the reconstruction
 * steps aside.
 *
 * One hard limit remains: the counter granularity is 65536 ns on this hardware, so
 * ~0.066 ms is the floor. Several passes here are genuinely below it and can only be
 * reported as "under one tick" — which is the honest answer, and the reason the frame
 * total is the number to optimise against rather than any single small pass.
 */
export function passDurations(count, times) {
  // Are the BEGIN timestamps self-consistent — each at or after the previous pass's end?
  //
  // Pairs whose predecessor was never sampled are skipped rather than counted as passing.
  // Comparing against an unwritten end of 0 succeeds for any begin, so including those pairs
  // let a frame full of unwritten passes vouch for begins that are not actually usable, and
  // the function would then fall back to the `end - begin` arithmetic this exists to avoid.
  const beginsUsable = (() => {
    let prevEnd = 0n;
    for (let i = 0; i < count; i++) {
      const end = times[i * 2 + 1];
      if (end === 0n) { continue; }
      if (prevEnd !== 0n && times[i * 2] < prevEnd) { return false; }
      prevEnd = end;
    }
    return true;
  })();

  const out = new Array(count).fill(null);
  // The most recent end that was actually WRITTEN. Some passes come back 0/0 — the
  // counter was never sampled for them — and walking back to a zero end measures from
  // the start of the counter's epoch, which is how ember-draw came out at 10 ms for
  // 0.05 ms of work. An unwritten pass has an unknown cost (reported null, not zero) and
  // its successor necessarily absorbs it, which is the honest attribution: the pair's
  // combined cost is known, the split is not.
  let lastEnd = 0n;
  for (let i = 0; i < count; i++) {
    const begin = times[i * 2];
    const end = times[i * 2 + 1];
    if (end === 0n) { continue; }               // never sampled — no measurement exists
    const from = beginsUsable || lastEnd === 0n ? begin : lastEnd;
    lastEnd = end;
    const ns = Number(end - from);
    out[i] = Number.isFinite(ns) && ns >= 0 ? ns / 1e6 : null;
  }
  return out;
}

export class Profiler {
  constructor(device, { enabled }) {
    this.device = device;
    this.enabled = !!enabled;
    this.labels = [];
    /** label -> smoothed milliseconds */
    this.timings = new Map();

    /**
     * Optional raw-sample sink: `(label, ms)` for every resolved pass, before
     * smoothing. The HUD needs smoothing to be readable at all; anything trying
     * to MEASURE needs the distribution, and recovering it from an exponential
     * average is not possible. So the profiler keeps owning timing and the caller
     * owns statistics.
     */
    this.onSample = null;

    /**
     * Optional raw sink: `(labels, times)` with the unmodified u64 counter values,
     * begin/end interleaved. Exists because the DERIVED milliseconds cannot answer the
     * question "did two passes measure the same interval?", and on some backends they
     * do — see the note on `passDurations`. Diagnosing that needs the absolute counters.
     */
    this.onRaw = null;

    if (!this.enabled) return;

    // Two timestamps per pass: begin and end.
    this.capacity = MAX_PASSES * 2;
    this.querySet = device.createQuerySet({
      label: 'pass-timestamps',
      type: 'timestamp',
      count: this.capacity,
    });
    this.byteSize = this.capacity * 8;         // u64 each
    this.resolveBuffer = device.createBuffer({
      label: 'timestamp-resolve',
      size: this.byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.pool = [];
    for (let i = 0; i < POOL; i++) {
      this.pool.push({
        buffer: device.createBuffer({
          label: `timestamp-read-${i}`,
          size: this.byteSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        busy: false,
      });
    }
  }

  /** Call once per frame, before recording passes. */
  beginFrame() {
    this.labels.length = 0;
  }

  /**
   * Timestamp writes for a pass descriptor. Spread into the descriptor:
   *   { ...desc, ...profiler.scope('raymarch') }
   * Returns an empty object when profiling is off, so call sites stay identical.
   */
  scope(label) {
    if (!this.enabled) return {};
    const slot = this.labels.length;
    if (slot >= MAX_PASSES) return {};
    this.labels.push(label);
    return {
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: slot * 2,
        endOfPassWriteIndex: slot * 2 + 1,
      },
    };
  }

  /** Call after recording all passes, before submit. */
  resolve(encoder) {
    if (!this.enabled || !this.labels.length) return;
    const free = this.pool.find((p) => !p.busy);
    if (!free) return;                        // all in flight; skip this frame

    const count = this.labels.length * 2;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, free.buffer, 0, count * 8);
    free.busy = true;
    free.labels = this.labels.slice();
  }

  /** Call after submit. Fire and forget; results land when they land. */
  async readback() {
    if (!this.enabled) return;
    for (const slot of this.pool) {
      if (!slot.busy || slot.reading) continue;
      slot.reading = true;
      try {
        await slot.buffer.mapAsync(GPUMapMode.READ);
        const times = new BigUint64Array(slot.buffer.getMappedRange()).slice();
        slot.buffer.unmap();
        if (this.onRaw) this.onRaw(slot.labels, times);
        const durations = passDurations(slot.labels.length, times);
        for (let i = 0; i < slot.labels.length; i++) {
          const ms = durations[i];
          if (ms === null) continue;
          const label = slot.labels[i];
          if (this.onSample) this.onSample(label, ms);
          // Exponential smoothing: raw per-frame numbers are far too jittery to
          // read off a HUD.
          const prev = this.timings.get(label);
          this.timings.set(label, prev === undefined ? ms : prev * 0.9 + ms * 0.1);
        }
      } catch {
        /* device lost or buffer destroyed — drop this sample */
      } finally {
        slot.reading = false;
        slot.busy = false;
      }
    }
  }

  /** `[[label, ms], ...]` in the order the passes were recorded. */
  report() {
    return [...this.timings.entries()];
  }

  total() {
    let t = 0;
    for (const v of this.timings.values()) t += v;
    return t;
  }

  destroy() {
    if (!this.enabled) return;
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    for (const p of this.pool) p.buffer.destroy();
  }
}
