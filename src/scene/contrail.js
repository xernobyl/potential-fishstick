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
    /** Scratch for the live head, so a per-frame write allocates nothing. */
    this._head = [0, 0, 0];
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

    // WHERE A NOZZLE IS RIGHT NOW: the local offset turned by the ship's own ORIENTATION QUATERNION.
    //
    // Not rebuilt from forward/right/up, and that distinction was a real bug rather than a stylistic
    // one. `ship.right()` is the BANKED lateral axis but `ship.up()` is the UNBANKED surface normal —
    // the banked up exists only as a scratch on its way into the quaternion — so mixing them placed
    // the nozzle in no frame at all. The error is proportional to bank, which is why the trails looked
    // right flying straight and left from the side of the hull in a turn.
    //
    // `rot` is what the HULL is drawn with (see meshRigidFromQuat in mesh_vertex.wgsl), so using it
    // here makes the attachment correct by construction. There is one orientation, and this is it.
    const nozzleAt = (side, out) => {
      const q = ship.rot;
      // Local nozzle, in the hull's own axes, already in world units.
      const lx = side;
      const ly = CONTRAIL.rise;
      const lz = -CONTRAIL.offset;
      // v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v) — the two-cross-product form, no matrix.
      const cx = q[1] * lz - q[2] * ly;
      const cy = q[2] * lx - q[0] * lz;
      const cz = q[0] * ly - q[1] * lx;
      const tx = cx + q[3] * lx;
      const ty = cy + q[3] * ly;
      const tz = cz + q[3] * lz;
      out[0] = ship.pos[0] + lx + 2 * (q[1] * tz - q[2] * ty);
      out[1] = ship.pos[1] + ly + 2 * (q[2] * tx - q[0] * tz);
      out[2] = ship.pos[2] + lz + 2 * (q[0] * ty - q[1] * tx);
      return out;
    };

    this._acc += dt;
    // Catch up if several intervals elapsed, but bounded: after a long stall, replaying
    // hundreds of intervals would just refill the buffer with one position anyway.
    let steps = 0;
    while (this._acc >= CONTRAIL.interval && steps < 4) {
      this._acc -= CONTRAIL.interval;
      steps++;
      for (let rib = 0; rib < this.ribbons; rib++) {
        const base = rib * n * 4;
        // Shift this half down by one and append. n is a few dozen and this runs once
        // per interval, not per frame, so a copy beats the index arithmetic a true ring
        // would force into the shader.
        a.copyWithin(base, base + 4, base + n * 4);
      }
    }

    // THE HEAD IS LIVE, REWRITTEN EVERY FRAME. The interval above decides when a sample is FROZEN
    // into the history; it should never have decided where the trail starts.
    //
    // With the head left as the last frozen sample it lagged the ship by up to one interval — 45 ms,
    // which at cruise is a fair fraction of a hull length. Flying straight that lag sits directly
    // behind the nozzle and is invisible; in a turn the ship has rotated away from it, so the trail
    // appeared to leave from the SIDE of the hull rather than from the hole. The symptom looked like a
    // bad offset and was actually a stale one.
    //
    // `right()` is the banked axis, so the two trails swap sides through a roll exactly as the nozzles
    // do.
    for (let rib = 0; rib < this.ribbons; rib++) {
      const o = rib * n * 4 + (n - 1) * 4;
      nozzleAt(rib === 0 ? -CONTRAIL.spread : CONTRAIL.spread, this._head);
      a[o] = this._head[0]; a[o + 1] = this._head[1]; a[o + 2] = this._head[2];
      a[o + 3] = ship.throttle;
    }
    this.device.queue.writeBuffer(this.buffer, 0, this.cpu);
  }

  destroy() { this.buffer.destroy(); }
}
