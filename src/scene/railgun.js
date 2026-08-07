/**
 * Rail gun: a small pool of instant beams, alternating between the wing tips — and a POWER SHOT if you
 * hold the trigger first.
 *
 * Built on the contrail's shape — a storage buffer of samples drawn as camera-facing ribbons — but the
 * state is different in kind. A contrail is a HISTORY that is appended to continuously; a rail shot is
 * an EVENT with a birth time. So instead of a ring buffer of positions, this is a ring of shots, each
 * holding where it started, which way it went, when, and how hard. Everything else about a shot's
 * appearance is a function of its age, which means the GPU needs three vec4s per shot and nothing more.
 *
 * THE CHARGE IS INTEGRATED HERE, on the renderer's own dt rather than a wall clock. That matters more
 * than it looks: `freeze` and `pause` set dt to zero, and a charge on `performance.now()` would keep
 * filling through a frozen scene and fire the moment you unfroze it. One clock, and every instrument
 * keeps working.
 *
 * FIRING HAPPENS ON RELEASE, which is the only moment the charge is known. A tap therefore fires on
 * keyup rather than keydown — a few milliseconds later and imperceptible, and it is what lets one
 * button mean both "shoot" and "wind up".
 *
 * The muzzle is stored in SHIP-LOCAL space, not world space, and that is the difference between a beam
 * that comes out of the gun and one that does not. Frozen in world space at fire time, the near end
 * stays put while the ship flies on — measured, 2.5 hull-radii of separation by the time a shot fades,
 * which reads as beams floating in the void. The shader rebuilds the world origin from the ship's LIVE
 * transform each frame, so the beam stays welded to the wing tip. Only the DIRECTION is frozen at fire
 * time, which is the part that should not follow the ship: you aimed it once.
 */

import { RAIL } from './tuning.js';

/**
 * A deterministic stand-in for `Math.random()`, seeded on a number you already have.
 *
 * Everything else in this renderer is a closed form in time — that is what lets a benchmark replay a
 * scene and get the same pixels, and what the recorder relies on to render a file frame by frame off a
 * synthetic clock. `Math.random()` in the fire path quietly broke that: the same run twice would give a
 * shot a different colour, so an image comparison across two runs could differ for a reason that had
 * nothing to do with the change being measured.
 *
 * The fire time is already unique per shot, so hashing it gives the same variety for free.
 */
function hash1(x) {
  const s = Math.sin(x * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** vec4(localMuzzle.xyz, fireTime) + vec4(dir.xyz, side) + vec4(power, hue, seed, spare) */
const FLOATS_PER_SHOT = 12;

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

    /** Seconds the trigger has been held, 0 when it is not. */
    this.charge = 0;
    /** 0..1. What the shader draws as the muzzle glow, and what scales a power shot. */
    this.chargeFrac = 0;
    /**
     * The hue the charged shot will fire as, chosen when the trigger goes DOWN.
     *
     * Chosen at press rather than at release so the muzzle glow can preview it — the colour you charge
     * is the colour that leaves, which makes the wind-up feel like it belongs to the shot rather than
     * being a generic effect that happens to precede one.
     */
    this.chargeHue = 0;
    this._held = false;
    /** Set on the frame a power shot leaves, for the camera to kick off. Read and cleared. */
    this.kick = 0;

    // Fire times start far in the past so nothing is mid-flight at startup.
    for (let i = 0; i < this.max; i++) this.cpu[i * FLOATS_PER_SHOT + 3] = -1e3;
    this._dirty = true;
  }

  /**
   * Put one shot in the pool.
   *
   * @param {number} time  seconds
   * @param {import('./ship.js').Ship} ship
   * @param {number} power 0 for an ordinary shot, 0..1 for a charged one
   */
  fire(time, ship, power = 0) {
    const o = (this.next % this.max) * FLOATS_PER_SHOT;
    this.next++;
    this.side = -this.side;

    const f = ship.forward();
    // Local muzzle, in the hull's own axes: +X is the wing, +Z the nose. The shader transforms it by
    // the ship's current orientation every frame.
    //
    // A POWER SHOT LEAVES FROM THE CENTRELINE, not a wing tip. Both guns are feeding it, and a beam
    // this size coming off one wing looks like it should be spinning the ship.
    this.cpu[o + 0] = power > 0 ? 0 : RAIL.spread * this.side;
    this.cpu[o + 1] = RAIL.up;
    this.cpu[o + 2] = RAIL.forward;
    // Direction IS frozen — captured in world space at the moment of firing.
    for (let k = 0; k < 3; k++) this.cpu[o + 4 + k] = f[k];
    this.cpu[o + 3] = time;
    this.cpu[o + 7] = this.side;
    this.cpu[o + 8] = power;
    // The hue the charge previewed, or one derived from the fire time for an ordinary shot.
    this.cpu[o + 9] = power > 0 ? this.chargeHue : hash1(time);
    // A per-shot seed, so two shots never scatter their sparks the same way.
    this.cpu[o + 10] = hash1(time * 1.7 + 9.1) * 1000;
    this.cpu[o + 11] = 0;
    this._dirty = true;

    if (power > 0) this.kick = power;
  }

  /**
   * @param {number} time    seconds
   * @param {number} dt      seconds; exactly 0 when the scene is held
   * @param {import('./ship.js').Ship} ship
   * @param {boolean} trigger  whether the trigger is down THIS frame
   * @param {boolean} autoFire  a synthetic ordinary shot, from the autopilot
   */
  update(time, dt, ship, trigger, autoFire) {
    if (trigger && !this._held) {
      // Trigger down: start charging, and pick the colour now so the glow can show it.
      this.charge = 0;
      this.chargeHue = hash1(time * 2.3 + 4.7);
    } else if (trigger) {
      this.charge += dt;
    } else if (this._held) {
      // RELEASE. Past the threshold it is a power shot; below it, the tap it always was.
      const frac = Math.min(this.charge / RAIL.chargeTime, 1);
      this.fire(time, ship, frac >= RAIL.chargeMin ? frac : 0);
      this.charge = 0;
    }
    this._held = trigger;
    this.chargeFrac = trigger ? Math.min(this.charge / RAIL.chargeTime, 1) : 0;

    // The autopilot's shot, which never charges — it is there so something is happening before anyone
    // touches a key, and a charging autopilot would just be a slower one.
    if (autoFire) this.fire(time, ship, 0);

    // Uploaded only when something changed. A shot's whole animation is a function of its age, so a
    // beam in flight needs no per-frame writes at all.
    if (this._dirty) {
      this.device.queue.writeBuffer(this.buffer, 0, this.cpu);
      this._dirty = false;
    }
  }

  /** The camera's kick for this frame, consumed on read. */
  takeKick() {
    const k = this.kick;
    this.kick = 0;
    return k;
  }

  destroy() { this.buffer.destroy(); }
}
