/**
 * The camera: an organic drift, or an arcball while the pointer is down.
 *
 * Two responsibilities that must not drift apart:
 *  - produce the basis and matrices used to build rays this frame;
 *  - remember last frame's view-projection, because temporal reprojection needs
 *    the exact matrix the history was rendered with. Reconstructing it from
 *    `time - dt` would be subtly wrong the moment the user drags.
 *
 * The projection lives here as a MATRIX rather than as shader constants. That is
 * not tidiness: the shading ray, the reprojection and the ember billboards must
 * agree to the last bit, and three hand-rolled copies of the same dot products
 * is precisely how they stop agreeing. One matrix, one source of truth.
 *
 * Field of view is set against the screen DIAGONAL (see `screen` below), so the
 * subject is framed the same whether the window is wide, tall or square.
 */

import { CAMERA } from './tuning.js';
import * as m4 from './mat4.js';

const V = () => new Float32Array(3);

function normalize(o, x, y, z) {
  const l = Math.hypot(x, y, z) || 1;
  o[0] = x / l; o[1] = y / l; o[2] = z / l;
  return o;
}
function cross(o, a, b) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  o[0] = x; o[1] = y; o[2] = z;
  return o;
}

/** One camera basis: position plus an orthonormal right/up/forward. */
class Basis {
  constructor() {
    this.pos = V();
    this.right = V();
    this.up = V();
    this.fwd = V();
  }
  copyFrom(o) {
    this.pos.set(o.pos); this.right.set(o.right); this.up.set(o.up); this.fwd.set(o.fwd);
  }
}

export class Camera {
  constructor() {
    this.current = new Basis();
    this.previous = new Basis();
    this.distance = CAMERA.distance;
    this.target = V();
    this._tmp = V();
    this._tmp2 = V();
    this._first = true;

    /** Focal length in half-diagonal units: half-diagonal 1 at distance `focal`. */
    this.focal = 1 / Math.tan((CAMERA.diagonalFov * Math.PI) / 360);

    this.viewProj = m4.create();
    this.invViewProj = m4.create();
    this.prevViewProj = m4.create();

    /**
     * Shared screen space: half-diagonal 1, y up, origin centred.
     *   [invDiag, diag, sx, sy]
     * `sx`/`sy` are the half-extents, so hypot(sx, sy) === 1 always and the axes
     * merely share out a fixed diagonal. `diag` is the diagonal in pixels.
     */
    this.screen = new Float32Array(4);
    this._aspect = 0;
    this._proj = m4.create();
    this._invProj = m4.create();
    this._view = m4.create();
    this._invView = m4.create();
  }

  /**
   * Recompute the projection for a viewport. Cheap and idempotent — it early-outs
   * unless the aspect actually changed, so calling it every frame is free.
   */
  setViewport(width, height) {
    const diag = Math.hypot(width, height) || 1;
    const sx = width / diag;
    const sy = height / diag;
    this.screen[0] = 1 / diag;
    this.screen[1] = diag;
    this.screen[2] = sx;
    this.screen[3] = sy;

    const aspect = width / Math.max(1, height);
    // The field of view is part of the early-out key, not just the aspect. It is a tuning
    // value the debug panel can move, and gating only on aspect meant a FOV change did
    // nothing until the window was resized — a silent no-op control.
    const focal = 1 / Math.tan((CAMERA.diagonalFov * Math.PI) / 360);
    if (aspect === this._aspect && focal === this.focal) return;
    this._aspect = aspect;
    this.focal = focal;
    m4.projection(this._proj, this.focal, CAMERA.frameOffset, sx, sy);
    m4.projectionInverse(this._invProj, this.focal, CAMERA.frameOffset, sx, sy);
  }

  /**
   * @param {number} time    seconds
   * @param {object} input   { dragging, x, y, width, height, everUsed }
   */
  update(time, dt, input, ship = null) {
    this.previous.copyFrom(this.current);
    this.prevViewProj.set(this.viewProj);

    // Chase, once the player has flown. Eased rather than rigid: a rigid chase on a
    // curved surface swings the whole frame every time the ship turns, which is
    // nauseating — lag is what makes a follow camera readable. The look-at point leads
    // the ship along its heading so you see where you are going, not where you are.
    if (ship && input.chase) {
      const c = this.current;
      const sp = ship.pos;
      const sf = ship.forward();
      const su = ship.up();
      // Exponential smoothing with a real TIME CONSTANT, not a fixed per-frame
      // fraction. A constant fraction is frame-rate dependent: at 60 fps a 0.08 lag is
      // a ~200 ms time constant, at 25 fps the same number is ~480 ms. Frame time
      // always varies a little, so the camera lagged a different amount every frame,
      // and that is a camera that jitters for reasons nothing on screen explains.
      const k = 1 - Math.exp(-Math.max(dt, 1e-4) / CAMERA.chaseTau);
      for (let i = 0; i < 3; i++) {
        const want = sp[i] - sf[i] * CAMERA.chaseBack + su[i] * CAMERA.chaseUp;
        c.pos[i] += (want - c.pos[i]) * k;
        this._tmp[i] = sp[i] + sf[i] * CAMERA.chaseLead;
      }
      normalize(c.fwd, this._tmp[0] - c.pos[0], this._tmp[1] - c.pos[1], this._tmp[2] - c.pos[2]);
      // The ship's own up as the reference, so the horizon rolls with the surface
      // rather than staying pinned to world Y.
      const r = cross(this._tmp2, c.fwd, su);
      normalize(c.right, r[0], r[1], r[2]);
      cross(c.up, c.right, c.fwd);
      this.distance = CAMERA.chaseBack;
      m4.mul(this.viewProj, this._proj, m4.view(this._view, c));
      m4.mul(this.invViewProj, m4.viewInverse(this._invView, c), this._invProj);
      if (this._first) {
        this.previous.copyFrom(c);
        this.prevViewProj.set(this.viewProj);
        this._first = false;
      }
      return;
    }

    let yaw, pitch;
    const ta = this.target;

    // Dolly and roll are properties of the camera itself rather than of the drift,
    // so they apply while the user steers too — the scene keeps breathing, and the
    // arcball stays predictable because neither touches yaw or pitch.
    const TAU = Math.PI * 2;
    const dolly = CAMERA.zoom * (
        0.78 * Math.sin((TAU * time) / CAMERA.zoomPeriod)
      + 0.22 * Math.sin((TAU * time) / (CAMERA.zoomPeriod * 0.41) + 1.7));
    const roll = CAMERA.roll * (
        0.72 * Math.sin((TAU * time) / CAMERA.rollPeriod + 0.4)
      + 0.28 * Math.sin((TAU * time) / (CAMERA.rollPeriod * 0.37) + 2.7));

    if (input.everUsed) {
      // Arcball: yaw and pitch follow the pointer directly, so steering is exact.
      yaw = -(input.x / Math.max(1, input.width) - 0.5) * Math.PI * 2 * 1.6;
      pitch = clamp((input.y / Math.max(1, input.height) - 0.5) * 2.2, -1.15, 1.15);
      this.distance = CAMERA.distance + dolly;
      ta[0] = ta[1] = ta[2] = 0;
    } else {
      // Layered sines at incommensurate periods: the rate eases and gathers
      // instead of ticking round, and nothing ever visibly loops.
      const t = time;
      yaw = 0.13 * t + 0.22 * Math.sin(0.077 * t) + 0.09 * Math.sin(0.031 * t + 2.1);
      pitch = 0.2 + 0.16 * Math.sin(0.053 * t + 0.7) + 0.05 * Math.sin(0.121 * t + 1.9);
      this.distance = CAMERA.distance + dolly;
      ta[0] = 0.16 * Math.sin(0.045 * t + 0.3);
      ta[1] = 0.16 * Math.sin(0.062 * t + 1.7);
      ta[2] = 0.16 * Math.sin(0.038 * t + 2.9);
    }

    const c = this.current;
    const cp = Math.cos(pitch);
    c.pos[0] = ta[0] + this.distance * cp * Math.cos(yaw);
    c.pos[1] = ta[1] + this.distance * Math.sin(pitch);
    c.pos[2] = ta[2] + this.distance * cp * Math.sin(yaw);

    normalize(c.fwd, ta[0] - c.pos[0], ta[1] - c.pos[1], ta[2] - c.pos[2]);

    // Roll tilts the reference up-vector. Safe here because pitch is clamped
    // well inside ±90°, where fwd could become parallel to up and the cross
    // product would collapse.
    const upRef = normalize(this._tmp, Math.sin(roll), Math.cos(roll), 0);
    // Into scratch, and no spread: `cross(V(), ...)` allocated a Float32Array
    // every frame, and the spread allocated an iterator on top of it, both purely
    // to hand three numbers along.
    const r = cross(this._tmp2, c.fwd, upRef);
    normalize(c.right, r[0], r[1], r[2]);
    cross(c.up, c.right, c.fwd);

    m4.mul(this.viewProj, this._proj, m4.view(this._view, c));
    m4.mul(this.invViewProj, m4.viewInverse(this._invView, c), this._invProj);

    // On the very first frame there is no previous basis; make it identical so
    // reprojection has nothing to smear from.
    if (this._first) {
      this.previous.copyFrom(c);
      this.prevViewProj.set(this.viewProj);
      this._first = false;
    }
  }

  /** World-space focus distance for the thin lens. */
  focusDistance() {
    return Math.max(this.distance - CAMERA.focusPull, 0.5);
  }

  /**
   * Where a world DIRECTION lands in the shared screen space (half-diagonal 1,
   * y up), for anchoring the lens flares. Returns null when it is behind the
   * camera, which the caller must turn into an offscreen sentinel.
   *
   * w = 0 is what makes this a direction: it cancels the matrix's translation
   * column, which is exactly the rotation-only projection something at infinity
   * wants. Same matrix as the geometry, so a flare cannot drift off its sun.
   */
  projectDirection(dir, out = [0, 0]) {
    if (!m4.project(this.viewProj, dir, 0, out)) return null;
    out[0] *= this.screen[2];
    out[1] *= this.screen[3];
    return out;
  }
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
