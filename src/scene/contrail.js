/**
 * Contrail: a ring buffer of where the ship has been.
 *
 * Trails are the one thing in this scene that CANNOT be stateless. Everything else here
 * — the rings, the detonations, the satellites — is a closed-form function of time, so
 * it needs no memory. A trail is by definition the path actually taken, and the ship's
 * state is integrated, so there is nothing to re-evaluate at t-n. Hence real history.
 *
 * Samples are emitted on a fixed TIME interval, not per frame and not per unit of
 * distance. Per frame would make the trail's length depend on frame rate. Per unit of
 * distance would give uniform spacing but a stopped ship would leave a trail frozen in
 * the air. A fixed interval gives speed-proportional spacing — a fast ship draws a long
 * trail, a stopped one lets its trail collapse and fade — which is what a contrail
 * actually does.
 *
 * Because the interval is fixed, AGE is implied by index and never stored: sample i of n
 * is exactly (n-1-i) intervals old. The spare component carries the throttle at
 * emission instead, so a trail thins out where the pilot was coasting.
 *
 * There are TWO ribbons, one per nacelle, held as two contiguous halves of one buffer
 * rather than two buffers. The draw is then a single instanced call where the instance
 * index picks the half — no second binding, no second pass, and the two trails cross and
 * separate on their own as the ship banks, because each nozzle is offset along the
 * hull's own lateral axis.
 */

import { CONTRAIL } from './tuning.js';

export class Contrail {
  constructor(device) {
    this.count = CONTRAIL.samples;
    this.ribbons = 2;
    // xyz position, w throttle at emission. Ribbon r occupies [r*count, (r+1)*count).
    this.cpu = new Float32Array(this.count * this.ribbons * 4);
    this.buffer = device.createBuffer({
      label: 'contrail',
      size: this.cpu.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device = device;
    this._acc = 0;
    this._primed = false;
  }

  /** @param {number} dt seconds @param {import('./ship.js').Ship} ship */
  update(dt, ship) {
    const a = this.cpu;
    const n = this.count;

    if (!this._primed) {
      // Fill both halves with the current position, so the first frames show a collapsed
      // trail rather than a streak from the world origin.
      for (let i = 0; i < n * this.ribbons; i++) {
        a[i * 4] = ship.pos[0]; a[i * 4 + 1] = ship.pos[1];
        a[i * 4 + 2] = ship.pos[2]; a[i * 4 + 3] = 0;
      }
      this._primed = true;
    }

    this._acc += dt;
    // Catch up if several intervals elapsed, but bounded: after a long stall, replaying
    // hundreds of intervals would just refill the buffer with one position anyway.
    let steps = 0;
    while (this._acc >= CONTRAIL.interval && steps < 4) {
      this._acc -= CONTRAIL.interval;
      steps++;
      const f = ship.forward();
      const r = ship.right();
      for (let rib = 0; rib < this.ribbons; rib++) {
        const base = rib * n * 4;
        // Shift this half down by one and append. n is a few dozen and this runs once
        // per interval, not per frame, so a copy beats the index arithmetic a true ring
        // would force into the shader.
        a.copyWithin(base, base + 4, base + n * 4);
        const o = base + (n - 1) * 4;
        // Behind the hull, and out to this nacelle. `right()` is the banked axis, so the
        // two trails swap sides through a roll exactly as the nozzles do.
        const side = rib === 0 ? -CONTRAIL.spread : CONTRAIL.spread;
        for (let k = 0; k < 3; k++) {
          a[o + k] = ship.pos[k] - f[k] * CONTRAIL.offset + r[k] * side;
        }
        a[o + 3] = ship.throttle;
      }
    }
    if (steps) this.device.queue.writeBuffer(this.buffer, 0, this.cpu);
  }

  destroy() { this.buffer.destroy(); }
}
