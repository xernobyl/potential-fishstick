/**
 * Rail gun: a small pool of instant beams, alternating between the wing tips.
 *
 * Built on the contrail's shape — a storage buffer of samples drawn as camera-facing
 * ribbons — but the state is different in kind. A contrail is a HISTORY that is appended
 * to continuously; a rail shot is an EVENT with a birth time. So instead of a ring buffer
 * of positions, this is a ring of shots, each holding where it started, which way it
 * went, and when. Everything else about a shot's appearance is a function of its age,
 * which means the GPU needs two vec4s per shot and nothing more.
 *
 * Fired on the EDGE of the key, never while held. Auto-fire would make the pool cycle in
 * a fraction of a second and the alternation invisible; hammering the key is the point.
 *
 * The muzzle is stored in SHIP-LOCAL space, not world space, and that is the difference
 * between a beam that comes out of the gun and one that does not. Frozen in world space
 * at fire time, the near end stays put while the ship flies on — measured, 2.5 hull-radii
 * of separation by the time a shot fades, which reads as beams floating in the void. The
 * shader rebuilds the world origin from the ship's LIVE transform each frame, so the beam
 * stays welded to the wing tip. Only the DIRECTION is frozen at fire time, which is the
 * part that should not follow the ship: you aimed it once.
 */

import { RAIL } from './tuning.js';

const FLOATS_PER_SHOT = 8;   // vec4(localMuzzle.xyz, fireTime) + vec4(dir.xyz, side)

export class Railgun {
  constructor(device) {
    this.max = RAIL.pool;
    this.cpu = new Float32Array(this.max * FLOATS_PER_SHOT);
    this.buffer = device.createBuffer({
      label: 'railgun',
      size: this.cpu.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device = device;
    this.next = 0;
    /** Which wing fires next. Alternates on every shot, hence the name. */
    this.side = 1;
    // Fire times start far in the past so nothing is mid-flight at startup.
    for (let i = 0; i < this.max; i++) this.cpu[i * FLOATS_PER_SHOT + 3] = -1e3;
    this._dirty = true;
  }

  /**
   * @param {number} time seconds
   * @param {import('./ship.js').Ship} ship
   * @param {boolean} fire edge-triggered, already consumed by the caller
   */
  update(time, ship, fire) {
    if (fire) {
      const o = (this.next % this.max) * FLOATS_PER_SHOT;
      this.next++;
      this.side = -this.side;

      const f = ship.forward();
      // Local muzzle, in the hull's own axes: +X is the wing, +Z the nose. The shader
      // transforms it by the ship's current orientation every frame.
      this.cpu[o + 0] = RAIL.spread * this.side;
      this.cpu[o + 1] = RAIL.up;
      this.cpu[o + 2] = RAIL.forward;
      // Direction IS frozen — captured in world space at the moment of firing.
      for (let k = 0; k < 3; k++) this.cpu[o + 4 + k] = f[k];
      this.cpu[o + 3] = time;
      this.cpu[o + 7] = this.side;
      this._dirty = true;
    }

    // Uploaded only when something changed. A shot's whole animation is a function of
    // its age, so a beam in flight needs no per-frame writes at all.
    if (this._dirty) {
      this.device.queue.writeBuffer(this.buffer, 0, this.cpu);
      this._dirty = false;
    }
  }

  destroy() { this.buffer.destroy(); }
}
