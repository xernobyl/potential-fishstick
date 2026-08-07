/**
 * Auroras: ribbons drifting on a curl-noise flow, appearing and dissolving.
 *
 * Mechanically this is the contrail generalised — a ring buffer of past positions, sampled
 * on a fixed interval, drawn as a camera-facing ribbon. What changes is where the positions
 * come from: instead of recording a ship that is already moving, each ribbon has a walker
 * advected through a vector field, so the path is generated rather than observed.
 *
 * CURL noise, specifically, and the reason is that it is divergence-free by construction.
 * The curl of any smooth vector field has zero divergence, which means the flow neither
 * compresses nor expands — walkers released into it swirl and braid indefinitely instead of
 * piling into the attractors and dead spots that a raw noise field has everywhere. That is
 * the difference between "drifting ribbons" and "five streaks that converge and stop".
 *
 * These are STATEFUL, which is a deliberate exception to how the rest of this scene works.
 * The rings, the detonations and the satellites are closed-form functions of time and need no
 * memory. A path through a time-varying flow is an integral, so there is nothing to evaluate
 * at t-n — the same reason the contrail keeps real history. Two stateful things in the scene
 * now, both for the same reason, which is at least a consistent one.
 *
 * Each ribbon also carries a LIFE cycle, and that is what makes them read as something
 * arriving rather than as permanent scenery. The envelope fades the whole ribbon up and back
 * down; when it wraps, that ribbon is re-primed at a fresh position on the shell. Re-priming
 * matters: teleporting a walker without collapsing its history would draw one frame of
 * straight line across the entire scene.
 */

import { AURORA } from './tuning.js';

/** Emissions catchable in one frame. A backstop for a stall, not a working limit: dt is already
 *  clamped upstream, and at the current interval this covers most of a second. */
const MAX_EMIT = 4;

/** Cheap 3D value noise: hashed lattice, cubic (smoothstep) interpolation. */
function hash3(x, y, z) {
  // Three large odd multipliers, then fract of a sine-free product — cheap and stable in f64.
  let h = x * 374761393 + y * 668265263 + z * 1274126177;
  h = (h ^ (h >>> 13)) * 1274126177;
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

function valueNoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const w = zf * zf * (3 - 2 * zf);
  const lerp = (a, b, t) => a + (b - a) * t;
  const c = (i, j, k) => hash3(xi + i, yi + j, zi + k);
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
  const x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
  const x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 2 - 1;
}

/**
 * The vector potential itself, at three large offsets of one noise field.
 *
 * Using the same field three times over is fine as long as the offsets are far enough apart
 * that the samples are uncorrelated; reusing it without offsets would make the components equal
 * and the curl vanish. Read separately from `curlNoise` because the boundary treatment needs
 * psi as well as its curl — see `#step`.
 */
const POT_OFFSET = [13.1, 97.3, 211.7];

function potential(out, x, y, z, t) {
  for (let i = 0; i < 3; i++) {
    const o = POT_OFFSET[i];
    out[i] = valueNoise(x + o, y + o * 1.7, z + o * 0.3 + t);
  }
  return out;
}

/**
 * Curl of that potential, by central differences.
 */
function curlNoise(out, x, y, z, t) {
  const e = AURORA.curlEps;
  const n = (px, py, pz, o) => valueNoise(px + o, py + o * 1.7, pz + o * 0.3 + t);

  // Potential component X at y+-e and z+-e, etc. Each partial needs two lookups.
  const dFz_dy = (n(x, y + e, z, 211.7) - n(x, y - e, z, 211.7)) / (2 * e);
  const dFy_dz = (n(x, y, z + e, 97.3) - n(x, y, z - e, 97.3)) / (2 * e);
  const dFx_dz = (n(x, y, z + e, 13.1) - n(x, y, z - e, 13.1)) / (2 * e);
  const dFz_dx = (n(x + e, y, z, 211.7) - n(x - e, y, z, 211.7)) / (2 * e);
  const dFy_dx = (n(x + e, y, z, 97.3) - n(x - e, y, z, 97.3)) / (2 * e);
  const dFx_dy = (n(x, y + e, z, 13.1) - n(x, y - e, z, 13.1)) / (2 * e);

  out[0] = dFz_dy - dFy_dz;
  out[1] = dFx_dz - dFz_dx;
  out[2] = dFy_dx - dFx_dy;
  return out;
}

/**
 * Bridson's ramp: -1 below, +1 above, quintic between.
 *
 * Chosen for its DERIVATIVES rather than its shape. It is C2 at the ends (r' and r'' both
 * vanish at +-1) so the field it modulates has no crease where the ramp saturates, while its
 * slope at zero is a healthy 15/8 — and that non-zero slope at the boundary is the entire
 * mechanism in `shellField`. A smoothstep, whose slope vanishes at both ends, would not work.
 */
function ramp(x) {
  if (x >= 1) { return 1; }
  if (x <= -1) { return -1; }
  const x3 = x * x * x;
  return 1.875 * x - 1.25 * x3 + 0.375 * x3 * x * x;
}

function rampD(x) {
  if (x >= 1 || x <= -1) { return 0; }
  const x2 = x * x;
  return 1.875 - 3.75 * x2 + 1.875 * x2 * x2;
}

/**
 * The shell weighting and its radial derivative: [alpha, d alpha / dr].
 *
 * `alpha` is 1 through the middle of the shell and ramps to 0 at both faces, over `shellRamp`.
 * Multiplying the vector potential by it is what confines the flow — see `#step` for why that
 * works and why it is done to the potential rather than to the velocity.
 */
function shellField(out, r) {
  // Distance to the nearer face. The min has a crease at the mid-radius, but the ramp is
  // saturated flat there (2 * shellRamp < the shell's thickness), so the crease is in a region
  // where alpha is constant 1 and nothing propagates through it.
  const inner = r - AURORA.shellMin;
  const outer = AURORA.shellMax - r;
  const edge = Math.min(inner, outer);
  const sign = inner < outer ? 1 : -1;   // d(edge)/dr
  const x = edge / AURORA.shellRamp;
  out[0] = Math.max(0, ramp(x));
  out[1] = (x < 0 ? 0 : rampD(x) * sign) / AURORA.shellRamp;
  return out;
}

export class Aurora {
  constructor(device) {
    this.count = AURORA.samples;
    this.ribbons = AURORA.ribbons;
    // xyz position, w life envelope at emission. Ribbon r occupies [r*count, (r+1)*count),
    // exactly as the contrail lays out its two — so one instanced draw covers all of them and
    // the instance index picks the ribbon.
    this.cpu = new Float32Array(this.count * this.ribbons * 4);
    this.buffer = device.createBuffer({
      label: 'aurora',
      size: this.cpu.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device = device;

    this.pos = new Float32Array(this.ribbons * 3);
    this.phase = new Float32Array(this.ribbons);
    // Scratch FIRST. `#respawn` seeds a whole ribbon through the flow, so it needs `_curl`
    // already allocated — declaring it after the loop threw on construction, which the headless
    // walker test caught before it ever reached a GPU.
    this._acc = 0;
    this._primed = false;
    this._curl = new Float32Array(3);
    this._pot = new Float32Array(3);
    this._shell = new Float32Array(2);
    // Heading per walker, so the path has momentum and therefore bounded curvature.
    this.vel = new Float32Array(this.ribbons * 3);
    // Life phases spread evenly, so they do not all bloom together. Positions only: the first
    // `update` seeds the paths, because seeding integrates the flow and the flow is a function
    // of TIME, which the constructor does not have. Seeding here as well was ~12k noise
    // evaluations thrown away on the first frame.
    for (let r = 0; r < this.ribbons; r++) {
      this.phase[r] = r / this.ribbons;
      this.#place(r, r * 0.618);
    }
  }

  /** Place a walker somewhere on the shell and seed its ribbon from there. */
  #respawn(r, seed) {
    this.#place(r, seed);
    this.#seed(r, seed);
  }

  /** Just the placement, without the seeding — the constructor has no clock to seed against. */
  #place(r, seed) {
    // Fibonacci-ish spread over the sphere from a scalar seed, so a respawn cannot land in
    // the same place twice in a row and needs no random state.
    const u = (seed * 0.6180339887) % 1;
    const v = (seed * 0.7548776662) % 1;
    const ct = 1 - 2 * u;
    const st = Math.sqrt(Math.max(0, 1 - ct * ct));
    const ph = v * Math.PI * 2;
    const rad = AURORA.shellMin + (AURORA.shellMax - AURORA.shellMin) * ((seed * 0.4501) % 1);
    const p = this.pos;
    p[r * 3] = Math.cos(ph) * st * rad;
    p[r * 3 + 1] = ct * rad;
    p[r * 3 + 2] = Math.sin(ph) * st * rad;
    // Drop the heading too: it belonged to a walker somewhere else entirely, and carrying it
    // across would spend the first seconds of the new life turning around.
    this.vel[r * 3] = 0; this.vel[r * 3 + 1] = 0; this.vel[r * 3 + 2] = 0;
  }

  /**
   * Fill one ribbon's whole history by integrating the flow from its spawn point.
   *
   * The obvious thing is to collapse the ribbon to a point and let it draw itself out, which is
   * what the contrail does — correctly, because a contrail SHOULD start at the nozzle. It is
   * wrong here: filling 72 samples at a 0.11 s interval takes 7.9 seconds, so out of a 14 second
   * life a ribbon would spend more than half of it as a stub. Measured on screen, that is
   * exactly how it looked — short dashes rather than curtains.
   *
   * Integrating the path up front costs `samples` noise evaluations once per respawn, which is
   * nothing at five ribbons on a 14 second cycle, and the ribbon exists at full length from its
   * first visible frame. The envelope is zero at the wrap regardless, so the jump is unseen.
   */
  #seed(r, seedT) {
    const a = this.cpu;
    const n = this.count;
    const base = r * n * 4;
    const p = this.pos;
    const o = r * 3;
    // Walk forward through the flow, writing as we go, so index n-1 ends up the head. One fewer
    // STEP than write, so the walker finishes exactly on the head rather than an interval past
    // it — otherwise the first frame's live-head write snaps the last segment to full length.
    for (let i = 0; i < n; i++) {
      a[base + i * 4] = p[o];
      a[base + i * 4 + 1] = p[o + 1];
      a[base + i * 4 + 2] = p[o + 2];
      a[base + i * 4 + 3] = 0;        // rewritten every frame — see the note in `update`
      if (i < n - 1) { this.#step(o, AURORA.interval, seedT + i * AURORA.interval, r); }
    }
  }

  /**
   * How far through the current emission interval we are, 0..1.
   *
   * The shader adds this to every sample's age. Without it the whole ribbon's parameterisation
   * — width, fade, and the ray striation's phase — steps by one sample on every emission while
   * the geometry stays put, and that step was the shimmer.
   */
  get emitPhase() { return Math.min(1, this._acc / AURORA.interval); }

  /**
   * Advance one walker by `dt`. Two mechanisms, and they solve different halves of the problem.
   *
   * 1. THE FLOW RESPECTS THE SHELL, by construction (Bridson 2007, "Curl-Noise for Procedural
   *    Fluid Flow", section 3). The velocity is the curl of a vector potential, so multiplying
   *    the POTENTIAL by a weighting alpha(r) that vanishes at the shell's two faces gives
   *
   *        u = curl(alpha * psi) = alpha * curl(psi) + grad(alpha) x psi
   *
   *    and at a face, where alpha = 0, only the second term survives. grad(alpha) is radial
   *    there, so u is a cross product with the radius: exactly tangential, with no radial
   *    component at all. The walker cannot advect out of the shell, and the field is still a
   *    curl, so it is still divergence-free — no attractors, no dead spots.
   *
   *    Doing it to the potential is what makes that true. Projecting the VELOCITY onto the
   *    sphere would also stop the walker leaving, but projection is not divergence-free and
   *    reintroduces exactly the piling-up that curl noise is chosen to avoid.
   *
   * 2. THE TURN RATE IS BOUNDED, which is what the ribbon geometry needs. The walker carries a
   *    heading and steers it toward the flow by at most `maxTurn` radians per second, rather
   *    than adopting the flow direction outright.
   *
   *    Adopting it outright leaves the path's curvature unbounded, and worst exactly where the
   *    field is weakest: normalising a curl passing through zero turns a tiny noisy vector into
   *    a wildly swinging direction. Those were the sharp turns, and a ribbon cannot be expanded
   *    through one by any method — the miter limit only chooses how it fails. Bounding the turn
   *    rate bounds the path's minimum radius at speed / maxTurn, and keeping that several times
   *    the ribbon's half-width means the joint is never near the limit in the first place.
   *
   * The radial steering below is neither of those. It is the recovery path for a walker already
   * outside the shell, which the turn-rate lag makes possible even though the flow does not:
   * the heading trails the flow direction, and a trailing heading at a face has a small radial
   * component. It engages only outside, so the flow inside is untouched by it.
   */
  #step(o, dt, time, r) {
    const p = this.pos;
    const v = this.vel;
    const c = this._curl;
    const w = this._pot;
    const sh = this._shell;
    const s = AURORA.curlScale;
    const t = time * AURORA.curlDrift + r * 31.7;
    const sx = p[o] * s, sy = p[o + 1] * s, sz = p[o + 2] * s;

    // curlNoise differentiates in the SCALED coordinates it is handed, so its result is
    // d/d(scaled); the chain rule puts it back into world units. Both terms below have to be in
    // the same space or their balance is arbitrary.
    curlNoise(c, sx, sy, sz, t);
    potential(w, sx, sy, sz, t);
    const d = Math.hypot(p[o], p[o + 1], p[o + 2]) || 1;
    shellField(sh, d);
    const nx = p[o] / d, ny = p[o + 1] / d, nz = p[o + 2] / d;

    // u = alpha * curl(psi) + grad(alpha) x psi,  grad(alpha) = (d alpha / dr) * rhat
    let ux = sh[0] * c[0] * s + sh[1] * (ny * w[2] - nz * w[1]);
    let uy = sh[0] * c[1] * s + sh[1] * (nz * w[0] - nx * w[2]);
    let uz = sh[0] * c[2] * s + sh[1] * (nx * w[1] - ny * w[0]);

    // Desired direction: the flow, or hold course where the field has no opinion.
    const ul = Math.hypot(ux, uy, uz);
    let dx = ul > 1e-9 ? ux / ul : v[o];
    let dy = ul > 1e-9 ? uy / ul : v[o + 1];
    let dz = ul > 1e-9 ? uz / ul : v[o + 2];

    // Radial steering, over the same region the ramp covers and beyond it.
    //
    // The field alone is not quite enough, for two reasons that both come down to the walker
    // not being the flow. The heading TRAILS the flow direction by up to one turn-rate step, so
    // it still has a small radial component where the flow has none; and the tangential term is
    // a cross product with psi, which vanishes wherever psi happens to line up with the radius —
    // at such a spot the flow is momentarily nil, the walker holds its heading, and a heading
    // that was pointing out sails straight through the face.
    //
    // So the heading is bent inward as well, ramping in from where alpha starts falling rather
    // than waiting for the walker to be outside. Waiting was measurably too late: the turn
    // radius is 0.375 against a shell 0.7 thick, so a walker that only starts turning at the
    // face is already past it before it comes round, and the hard clamp was firing on every
    // approach — which put a 35 degree corner in a path otherwise bounded to 8.
    const outer = (d - (AURORA.shellMax - AURORA.shellRamp)) / AURORA.shellRamp;
    const inner = ((AURORA.shellMin + AURORA.shellRamp) - d) / AURORA.shellRamp;
    const depth = outer > 0 ? Math.min(outer, AURORA.shellSteerMax)
                : inner > 0 ? -Math.min(inner, AURORA.shellSteerMax) : 0;
    if (depth !== 0) {
      // Proportional-plus-DERIVATIVE. The position term alone is too late to be enough: the
      // walker only knows it should turn once it is already deep in the ramp, and turning
      // around costs it a turn radius of overshoot every time. The second term reacts to the
      // heading instead — an outward heading inside the ramp region is steered against
      // immediately, whatever the position — which is what actually bounds the excursion.
      const radial = v[o] * nx + v[o + 1] * ny + v[o + 2] * nz;
      const g = depth * AURORA.shellSteer
              + Math.min(1, Math.abs(depth)) * radial * AURORA.shellDamp;
      dx -= nx * g; dy -= ny * g; dz -= nz * g;
      const dl = Math.hypot(dx, dy, dz) || 1;
      dx /= dl; dy /= dl; dz /= dl;
    }

    // Rotate the current heading toward the desired one by at most maxTurn * dt.
    const vl = Math.hypot(v[o], v[o + 1], v[o + 2]);
    if (vl < 1e-9) { v[o] = dx; v[o + 1] = dy; v[o + 2] = dz; }
    else { v[o] /= vl; v[o + 1] /= vl; v[o + 2] /= vl; }
    const dot = Math.max(-1, Math.min(1, v[o] * dx + v[o + 1] * dy + v[o + 2] * dz));
    const maxAng = AURORA.maxTurn * dt;
    const ang = Math.acos(dot);
    if (ang > maxAng) {
      // Gram-Schmidt: the part of `desired` perpendicular to the current heading spans the
      // rotation plane, so this is an exact rotation by maxAng within it.
      const px = dx - v[o] * dot, py = dy - v[o + 1] * dot, pz = dz - v[o + 2] * dot;
      const pl = Math.hypot(px, py, pz);
      if (pl > 1e-9) {
        const ca = Math.cos(maxAng), sa = Math.sin(maxAng);
        v[o] = v[o] * ca + (px / pl) * sa;
        v[o + 1] = v[o + 1] * ca + (py / pl) * sa;
        v[o + 2] = v[o + 2] * ca + (pz / pl) * sa;
      }
    } else {
      v[o] = dx; v[o + 1] = dy; v[o + 2] = dz;
    }

    const step = AURORA.speed * dt;
    p[o] += v[o] * step; p[o + 1] += v[o + 1] * step; p[o + 2] += v[o + 2] * step;
    this.#confine(o);
  }

  /** Backstop only — the steering in `#step` should keep the walker inside on its own. */
  #confine(o) {
    const p = this.pos;
    const d = Math.hypot(p[o], p[o + 1], p[o + 2]) || 1;
    const lo = AURORA.shellMin - AURORA.shellSlack;
    const hi = AURORA.shellMax + AURORA.shellSlack;
    if (d >= lo && d <= hi) { return; }
    const sc = (d < lo ? lo : hi) / d;
    p[o] *= sc; p[o + 1] *= sc; p[o + 2] *= sc;
  }

  /** @param {number} dt seconds @param {number} time seconds */
  update(dt, time) {
    if (!this._primed) {
      // Seeded, not collapsed, so the first frames already show curtains.
      for (let r = 0; r < this.ribbons; r++) this.#seed(r, time + r * 3.7);
      this._primed = true;
    }

    // Advance the walkers every frame — the flow is continuous and integrating it on the
    // emission interval instead would make the path depend on that interval.
    for (let r = 0; r < this.ribbons; r++) {
      const o = r * 3;
      this.#step(o, dt, time, r);

      // Life. On wrap the ribbon is re-primed somewhere new; the envelope is zero at the
      // wrap so nothing is visible during the jump.
      const before = this.phase[r];
      this.phase[r] = before + dt / AURORA.lifetime;
      if (this.phase[r] >= 1) {
        this.phase[r] -= 1;
        this.#respawn(r, time * 7.31 + r * 2.17);
      }
    }

    const a = this.cpu;
    const n = this.count;

    this._acc += dt;
    let steps = 0;
    while (this._acc >= AURORA.interval && steps < MAX_EMIT) {
      this._acc -= AURORA.interval;
      steps++;
      // Shift by one sample. The head slot is left alone: it already holds the live position,
      // which IS the sample being committed, and the write below refreshes it.
      for (let r = 0; r < this.ribbons; r++) {
        const base = r * n * 4;
        a.copyWithin(base, base + 4, base + n * 4);
      }
    }
    // Long stalls must not leave a phase greater than one interval, or the oldest sample's age
    // runs past 1 and its fade term goes negative.
    if (this._acc >= AURORA.interval) { this._acc = AURORA.interval * 0.999; }

    // THE HEAD TRACKS THE WALKER EVERY FRAME, not once per emission.
    //
    // Writing it only on emission left the leading edge stationary for a whole interval and
    // then jumping the full sample spacing — about ten pixels, four and a half times a second.
    // Written continuously, the last segment instead grows from nothing to full length over the
    // interval and the shift that follows is exact, because the sample being committed is the
    // one already there. Together with the phase term in the shader's age, nothing about the
    // ring buffer's discreteness reaches the screen.
    for (let r = 0; r < this.ribbons; r++) {
      const oi = r * n * 4 + (n - 1) * 4;
      const o = r * 3;
      a[oi] = this.pos[o]; a[oi + 1] = this.pos[o + 1]; a[oi + 2] = this.pos[o + 2];
    }

    // The life envelope is a property of the WHOLE ribbon, so it is written across every
    // sample each frame rather than captured per sample at emission.
    //
    // Storing it at emission was wrong and the failure was total rather than subtle: a seeded
    // ribbon has no emission history, so every sample carried w = 0, the shader multiplied the
    // whole curtain by zero, and only the two or three samples added since the seed were ever
    // visible. On screen that is a short bright dash — which is exactly what it looked like.
    //
    // A contrail genuinely wants the per-sample version, because throttle at emission is a fact
    // about that moment. An aurora brightens and dims as one sheet, so a single current value is
    // both correct and simpler. 360 floats a frame; the upload is 5.7 KB.
    for (let r = 0; r < this.ribbons; r++) {
      const e = this.envelope(r);
      const base = r * n * 4;
      for (let i = 0; i < n; i++) a[base + i * 4 + 3] = e;
    }
    this.device.queue.writeBuffer(this.buffer, 0, a);
  }

  /**
   * Re-integrate every ribbon from scratch under the CURRENT flow parameters.
   *
   * The walkers read their tuning per step, so a change takes effect immediately — but the
   * samples already in the buffer were integrated under the old values and only leave as the
   * ribbon ages out, which at a 26 s lifetime makes a flow tweak look like it did almost
   * nothing. This is what makes the panel's flow sliders legible.
   */
  reseed() {
    for (let r = 0; r < this.ribbons; r++) this.#respawn(r, r * 0.618 + this.phase[r] * 7.7);
    this.device.queue.writeBuffer(this.buffer, 0, this.cpu);
  }

  /** Smooth 0 -> 1 -> 0 over one lifetime. Zero at both ends, so a respawn is invisible. */
  envelope(r) {
    const t = this.phase[r];
    const s = Math.sin(Math.PI * t);
    return s * s;
  }

  destroy() { this.buffer.destroy(); }
}
