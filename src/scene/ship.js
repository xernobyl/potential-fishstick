/**
 * The player ship: an old-school 2D game played on the surface of a sphere.
 *
 * Two DOF of position (anywhere on the shell), one of heading (which way it faces in
 * the local tangent plane), and no roll or pitch freedom at all. Left/right rotate the
 * heading, forward/back accelerate along it. That is Asteroids, wrapped onto a ball.
 *
 * ORIENTATION IS DERIVED, not integrated, and that is the load-bearing decision. Up is
 * the surface normal and the nose is the heading, always, by construction. An earlier
 * version carried a free torque-driven quaternion; it was the more "physical" model and
 * it was the wrong one, because a free rotation lets the ship tumble, and a tumbling
 * ship stops reading as a 2D game and starts reading as debris. Here tumbling is not
 * representable, so it cannot happen — and the quaternion is exactly unit for free,
 * with nothing to drift or renormalise.
 *
 * Moving across a sphere means the tangent plane turns underneath you, so position,
 * velocity and heading are all re-projected onto the new plane each step. That is a
 * first-order parallel transport: exact in the limit, and at these speeds and step
 * sizes indistinguishable from it.
 *
 * Bank is cosmetic and never feeds back into the motion. That separation is what lets
 * a constrained surface feel like flying.
 */

import { SHIP } from './tuning.js';

const v3 = () => new Float32Array(3);

function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  v[0] /= l; v[1] /= l; v[2] /= l;
  return v;
}

/** Remove the component of `v` along the unit vector `n`, in place. */
function reject(v, n) {
  const d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
  v[0] -= n[0] * d; v[1] -= n[1] * d; v[2] -= n[2] * d;
  return v;
}

function cross(out, a, b) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

/**
 * Rotation basis -> quaternion, given the world axes the body's X, Y, Z map to.
 * Branching on the largest diagonal term is not an optimisation: the single-case form
 * divides by a quantity that vanishes near a 180-degree rotation.
 */
function basisToQuat(out, x, y, z) {
  const m00 = x[0], m01 = y[0], m02 = z[0];
  const m10 = x[1], m11 = y[1], m12 = z[1];
  const m20 = x[2], m21 = y[2], m22 = z[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    out[0] = (m21 - m12) / s; out[1] = (m02 - m20) / s;
    out[2] = (m10 - m01) / s; out[3] = 0.25 * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    out[0] = 0.25 * s; out[1] = (m01 + m10) / s;
    out[2] = (m02 + m20) / s; out[3] = (m21 - m12) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    out[0] = (m01 + m10) / s; out[1] = 0.25 * s;
    out[2] = (m12 + m21) / s; out[3] = (m02 - m20) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    out[0] = (m02 + m20) / s; out[1] = (m12 + m21) / s;
    out[2] = 0.25 * s; out[3] = (m10 - m01) / s;
  }
  return out;
}

export class Ship {
  constructor() {
    /** World position, always at |pos| == SHIP.orbit. */
    this.pos = new Float32Array([0, 0, SHIP.orbit]);
    /** Unit heading, always tangent to the sphere at `pos`. The nose. */
    this.heading = new Float32Array([1, 0, 0]);
    /** World velocity, always tangent. */
    this.vel = v3();
    /** Turn rate about the local normal, rad/s. Torque-driven, so it has inertia. */
    this.turn = 0;
    /** Cosmetic lean, and the engine states the jets read. */
    this.bank = 0;
    this.throttle = 0;
    // Set once the player takes over; see `update`.
    this._flown = false;
    this.reverse = 0;

    /** Orientation, world <- body; body +Z nose, +Y up. Derived every frame. */
    this.rot = new Float32Array([0, 0, 0, 1]);
    /** Last frame's transform, so a hull hit can be mapped back exactly. Kept here
     *  rather than recomputed: the state is integrated, so there is no closed form to
     *  re-evaluate at t-dt the way the rings have. */
    this.prevPos = v3();
    this.prevRot = new Float32Array([0, 0, 0, 1]);
    /** What the thrusters react to: [unused, turn accel, bank rate]. */
    this.angAccel = v3();

    this._up = v3();
    this._right = v3();
    this._tmp = v3();
    this.update(0, { pitch: 0, yaw: 0, roll: 0, thrust: 0 });
  }

  forward() { return this.heading; }
  up() { return this._up; }
  /** Lateral axis. This is the BANKED one — it is what the orientation quaternion was
   *  built from — so anything attached to the hull rolls with it. */
  right() { return this._right; }
  speed() { return Math.hypot(this.vel[0], this.vel[1], this.vel[2]); }

  /**
   * @param {number} dt seconds
   * @param {{pitch:number, yaw:number, roll:number, thrust:number}} cmd each -1..1
   */
  update(dt, cmd) {
    const T = SHIP;
    // Snapshot before anything moves.
    this.prevPos.set(this.pos);
    this.prevRot.set(this.rot);
    const p = this.pos;
    const h = this.heading;
    const vel = this.vel;
    const up = this._up;
    const right = this._right;

    // The local normal, which is also the ship's up.
    up[0] = p[0]; up[1] = p[1]; up[2] = p[2];
    norm(up);

    // ---- turning: left/right, torque-driven so it winds up and coasts -----
    // The negation converts between two FORWARD conventions. Both are correct; neither
    // is a bug, and an earlier version of this comment was wrong to call one a defect.
    //
    // The ship's body frame is +Z-forward: +Z nose, +Y up, so +X = up x nose, and
    // det(right, up, nose) = +1. The camera's frame is -Z-forward, the usual convention
    // for a view basis: `right = fwd x upRef` (camera.js) evaluates to exactly +X for a
    // camera looking down -Z, which is the standard right-handed camera. Its
    // det(right, up, fwd) is -1 only because `fwd` IS the -Z axis there — the view basis
    // (right, up, -fwd) is properly right-handed. Verified numerically, both of them.
    //
    // Two conventions means one conversion, and this sign is it: rotating the heading
    // toward the body's +X reads as turning LEFT on screen, so the key's meaning and the
    // maths differ by a sign. Unifying them would mean respecifying the hull in -Z
    // space and moving every jet, RCS and muzzle offset with it — a lot of churn to
    // relocate a conversion rather than remove one. This negation is load-bearing, and
    // so is the one on `bank`.
    const turnAccel = -cmd.yaw * T.turnAccel - this.turn * T.turnDamp;
    this.turn += turnAccel * dt;
    // Rotate the heading about the normal. Small-angle rotation via the tangent basis
    // is enough here and avoids building a rotation matrix per step.
    cross(right, up, h);
    norm(right);
    const a = this.turn * dt;
    const ca = Math.cos(a), sa = Math.sin(a);
    for (let i = 0; i < 3; i++) h[i] = h[i] * ca + right[i] * sa;
    reject(h, up);
    norm(h);

    // ---- thrust: forward and reverse along the nose ----------------------
    // cmd.pitch is the up/down axis: up accelerates, down accelerates backwards.
    var fwdCmd = Math.max(0, cmd.pitch) + cmd.thrust;
    const revCmd = Math.max(0, -cmd.pitch);

    // CRUISE UNTIL FLOWN. With no input the ship used to sit still, which meant the contrails —
    // the whole reason it is here — only existed if someone was holding a key. It now runs at full
    // throttle until the player actually flies it, and then never again.
    //
    // Latched on FLIGHT input specifically, not on `input.everUsed`: that also latches on a drag,
    // so merely orbiting the camera to look at the thing would have stopped it.
    if (fwdCmd > 0 || revCmd > 0 || cmd.yaw !== 0) { this._flown = true; }
    if (!this._flown) { fwdCmd = 1; }
    this.throttle += (Math.min(1, fwdCmd) - this.throttle) * Math.min(1, dt * T.throttleRate);
    this.reverse += (revCmd - this.reverse) * Math.min(1, dt * T.throttleRate);

    const drive = this.throttle * T.thrust - this.reverse * T.reverseThrust;
    for (let i = 0; i < 3; i++) {
      vel[i] += h[i] * drive * dt;
      vel[i] -= vel[i] * T.linDamp * dt;        // drag stands in for a speed cap
      p[i] += vel[i] * dt;
    }

    // ---- back onto the sphere, and transport the frame with it ------------
    // Moving across a sphere turns the tangent plane underneath you, so velocity and
    // heading are re-projected onto the new one. Skip this and they drift out of the
    // surface and the ship slowly stops being tangent to anything.
    up[0] = p[0]; up[1] = p[1]; up[2] = p[2];
    norm(up);
    for (let i = 0; i < 3; i++) p[i] = up[i] * T.orbit;
    reject(vel, up);
    reject(h, up);
    norm(h);

    // ---- cosmetic lean ---------------------------------------------------
    const prevBank = this.bank;
    // Sign settled by MEASUREMENT in screen space, not by derivation: with a body frame
    // and a view frame that disagree on which way forward points (see the note on
    // `turnAccel`), the only trustworthy question is "does the roof lean the same way the
    // nose swings, as the player sees it". It does with this sign; it banked out of the
    // turn with the other one.
    this.bank += (this.turn * T.bankPerTurn - this.bank) * Math.min(1, dt * T.bankRate);

    // ---- derive the orientation ------------------------------------------
    cross(right, up, h);
    norm(right);
    // Bank INTO the turn, and the sign matters: with +X right, +Y up, +Z nose,
    // dropping the right wingtip rotates `up` TOWARD +right, not away from it. The
    // opposite signs banked the ship out of its turns, which is what reads as leaning
    // the wrong way.
    const cb = Math.cos(this.bank), sb = Math.sin(this.bank);
    const bu = this._tmp;
    for (let i = 0; i < 3; i++) {
      bu[i] = up[i] * cb + right[i] * sb;
      right[i] = right[i] * cb - up[i] * sb;
    }
    basisToQuat(this.rot, right, bu, h);

    // What the RCS shows — taken from the motion, not the keys, so the puffs fire to
    // arrest a turn as well as to start one.
    this.angAccel[0] = 0;
    this.angAccel[1] = turnAccel * T.rcsFromAccel;
    this.angAccel[2] = dt > 0 ? (this.bank - prevBank) / dt : 0;
  }
}
