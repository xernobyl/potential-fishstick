// ---------------------------------------------------------------------------
// The player ship: a small signed-distance field, marched in its OWN local space.
//
// Marching in local space is what makes an arbitrarily-oriented SDF cheap. The
// alternative — rotating the field — means rotating every primitive; instead the ray
// is rotated once by the inverse orientation, the field is evaluated in a fixed
// frame, and the hit normal is rotated back. A unit quaternion's inverse is its
// conjugate, so "once" is a handful of multiplies.
//
// Bounded by a sphere, so a ray that cannot reach the ship costs one quadratic. This
// runs inside the scene pass alongside the satellites, which is why that matters:
// the pass is already the frame's whole cost.
//
// Shape is deliberately hard-surface: a drawn-out hull, swept delta wings cut by a
// plane rather than modelled, a canopy blister offset off the spine, outboard
// nacelles and a dorsal blade. Jodorowsky by way of Star Fox — flat panels, a long
// nose, and everything readable in silhouette.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"

fn sdCapsule(p : vec3f, a : vec3f, b : vec3f, r : f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn sdBox(p : vec3f, b : vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdEllipsoid(p : vec3f, r : vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-5);
}

/// Rotate by a unit quaternion, and by its inverse (its conjugate).
fn qrotate(q : vec4f, v : vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn qrotateInv(q : vec4f, v : vec3f) -> vec3f {
  return qrotate(vec4f(-q.xyz, q.w), v);
}

/// The hull, in local space: nose along +Z, up along +Y.
fn sdShip(p : vec3f) -> f32 {
  // Fuselage. Squashing y BEFORE the capsule and dividing after gives a flattened
  // hull rather than a tube, which is the whole difference between a spacecraft and
  // a pipe. The divide keeps it a distance again (approximately, and it is a
  // uniform-enough scale that the marcher never notices).
  let sq = vec3f(1.0, 1.62, 1.0);
  var d = sdCapsule(p * sq, vec3f(0.0, 0.0, -0.42), vec3f(0.0, 0.0, 0.66), 0.105) / 1.62;

  // Wings, mirrored through x. Swept by CUTTING the leading edge with a plane rather
  // than by modelling a taper — one smooth-max instead of a shear, and the cut edge
  // is dead straight, which is what makes it read as a machined delta.
  var w = p;
  w.x = abs(w.x);
  var wing = sdBox(w - vec3f(0.30, 0.0, -0.05), vec3f(0.31, 0.015, 0.27));
  let sweep = dot(w - vec3f(0.05, 0.0, 0.21), normalize(vec3f(0.70, 0.0, 0.72)));
  wing = smax(wing, sweep, 0.035);
  // Notch out the trailing inboard corner, so the plan-form has a shape.
  wing = smax(wing, -sdBox(w - vec3f(0.64, 0.0, -0.42), vec3f(0.26, 0.10, 0.15)), 0.04);
  d = smin(d, wing, 0.05);

  // Canopy: offset off the spine, not centred on it.
  d = smin(d, sdEllipsoid(p - vec3f(0.0, 0.072, 0.20), vec3f(0.072, 0.052, 0.21)), 0.04);

  // Outboard nacelles, and a dorsal blade.
  var n = p;
  n.x = abs(n.x);
  d = smin(d, sdCapsule(n, vec3f(0.175, -0.012, -0.30), vec3f(0.175, -0.012, 0.10), 0.052), 0.045);
  d = smin(d, sdBox(p - vec3f(0.0, 0.135, -0.30), vec3f(0.011, 0.115, 0.155)), 0.03);
  return d;
}

/// The hull at its final size. Dividing the point and multiplying the result is a
/// uniform scale, which keeps the value a true distance — so the marcher, the normal
/// and the bounding sphere all stay consistent with one constant.
fn sdShipScaled(p : vec3f) -> f32 {
  return sdShip(p / SHIP_SCALE) * SHIP_SCALE;
}

struct ShipHit { hit : bool, t : f32, nor : vec3f, lp : vec3f };

fn shipNormal(lp : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0016 * SHIP_SCALE;
  return normalize(
      e.xyy * sdShipScaled(lp + e.xyy) + e.yyx * sdShipScaled(lp + e.yyx)
    + e.yxy * sdShipScaled(lp + e.yxy) + e.xxx * sdShipScaled(lp + e.xxx));
}

fn hitShip(ro : vec3f, rd : vec3f, tmax : f32) -> ShipHit {
  var out : ShipHit;
  out.hit = false;
  out.t = tmax;
  out.nor = vec3f(0.0, 1.0, 0.0);
  out.lp = vec3f(0.0);

  let q = frame.shipRot;
  let rel = ro - frame.shipPos.xyz;
  // One quadratic before any marching.
  let bs = iSphere(rel, rd, SHIP_BOUND);
  if (bs.y <= 0.0 || bs.x > tmax) { return out; }

  let lo = qrotateInv(q, rel);
  let ld = qrotateInv(q, rd);          // orientation only, so the direction stays unit

  var t = max(bs.x, 0.0);
  let tEnd = min(bs.y, tmax);
  for (var i = 0; i < SHIP_STEPS; i++) {
    let d = sdShipScaled(lo + ld * t);
    if (d < 0.0007 * SHIP_SCALE) {
      out.hit = true;
      out.t = t;
      out.lp = lo + ld * t;
      out.nor = qrotate(q, shipNormal(out.lp));
      return out;
    }
    t += d * 0.85;
    if (t > tEnd) { break; }
  }
  return out;
}

/// Hull shading: painted panels over metal, with the canopy as glass.
fn shadeShip(h : ShipHit, ro : vec3f, rd : vec3f) -> vec3f {
  let wp = ro + rd * h.t;
  let V = -rd;
  let N = h.nor;

  // Panel lines in LOCAL space, so they travel with the hull.
  // Authored-space local position: the hull is modelled at scale 1, so every feature
  // threshold below is expressed in those units and divides the scale out once here.
  let al = h.lp / SHIP_SCALE;
  let g = al * SHIP_PANEL;
  let cellId = floor(g);
  let ph = hash13(cellId);
  let e = abs(fract(g) - 0.5) * (1.0 - abs(normalize(al + 1e-5)));
  let seam = smoothstep(0.42, 0.50, max(max(e.x, e.y), e.z));

  // The canopy is glass: darker, smoother, and it sits above the spine at the nose.
  let isGlass = step(0.5, step(0.045, al.y) * step(al.z, 0.42) * step(-0.02, al.z));

  var alb = mix(SHIP_HULL * (0.85 + 0.30 * ph), SHIP_GLASS, isGlass);
  alb = mix(alb, alb * 0.45, seam);
  // A painted stripe along the wings, because a readable silhouette wants a graphic.
  let stripe = smoothstep(0.02, 0.0, abs(abs(al.x) - 0.42)) * step(0.1, abs(al.x));
  alb = mix(alb, SHIP_TRIM, stripe * (1.0 - isGlass));

  var f0 = mix(vec3f(0.06), vec3f(0.09), isGlass);
  var rough = mix(clamp(0.34 + 0.22 * ph + 0.25 * seam, 0.06, 1.0), 0.08, isGlass);

  var col = sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0);

  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  col += background(reflect(-V, N)) * fresnelSchlickRough(NoV, f0, rough) * SHIP_ENV;
  // The planetoid is right there and it is bright; ignoring it would flatten the hull.
  let toCore = -normalize(wp);
  col += SHIP_CORE_COL * max(dot(N, toCore), 0.0) / max(dot(wp, wp), 1.0)
       * (0.6 + 0.4 * heartbeat(beatPhase())) * alb;
  col += alb * SHIP_AMBIENT;
  return col;
}

/// Engine plumes and RCS puffs, as additive glow along the view ray.
///
/// Driven from the ANGULAR ACCELERATION the physics actually produced, not from the
/// key state: the puffs then show what the ship is doing rather than what was
/// pressed, so they fire to arrest a rotation as well as to start one. That is the
/// difference between thrusters and indicator lights.
fn shipJets(ro : vec3f, rd : vec3f, tmax : f32) -> vec3f {
  let q = frame.shipRot;
  let sp = frame.shipPos.xyz;
  let throttle = frame.shipPos.w;
  let ang = frame.shipJet.xyz;

  var sum = vec3f(0.0);

  // Two main plumes at the nacelle exhausts, stretched backwards along the hull.
  for (var i = 0; i < 2; i++) {
    let sx = select(-0.175, 0.175, i == 0);
    // Sampled at a few points down the plume so it reads as a cone, not a ball.
    for (var k = 0; k < SHIP_JET_LEN; k++) {
      let fk = f32(k);
      let local = vec3f(sx, -0.012, -0.33 - fk * 0.085 * (0.35 + throttle)) * SHIP_SCALE;
      let wpj = sp + qrotate(q, local);
      let oc = wpj - ro;
      let tc = clamp(dot(oc, rd), 0.0, tmax);
      let dist = length(oc - rd * tc);
      // Widening and cooling down its length, which is what makes a plume a plume
      // rather than a row of dots: the near cells are small and white-hot, the far
      // ones broad and dim.
      let r = (0.030 + fk * 0.019) * (0.5 + 0.8 * throttle) * SHIP_SCALE;
      let fade = throttle * max(0.0, 1.0 - fk / f32(SHIP_JET_LEN));
      let heat = mix(vec3f(1.9, 1.5, 1.1), SHIP_JET_COL, fk / f32(SHIP_JET_LEN));
      sum += heat * fade * exp(-sqr(dist / r) * 1.5);
    }
  }

  // Reverse: a short pair of plumes out of the FRONT, so backing off is visible.
  let rev = frame.shipJet.w;
  if (rev > 0.01) {
    for (var i = 0; i < 2; i++) {
      let sx = select(-0.10, 0.10, i == 0);
      let wpj = sp + qrotate(q, vec3f(sx, 0.0, 0.34) * SHIP_SCALE);
      let oc = wpj - ro;
      let tc = clamp(dot(oc, rd), 0.0, tmax);
      sum += vec3f(1.3, 0.85, 0.55) * rev
           * exp(-sqr(length(oc - rd * tc) / (0.032 * SHIP_SCALE)) * 2.0);
    }
  }

  // RCS: one puff per axis pair, on the side that would produce the measured torque.
  for (var a = 0; a < 3; a++) {
    let mag = ang[a];
    if (abs(mag) < 0.05) { continue; }
    // Pitch fires at the tail, yaw at the wingtips, roll at opposite tips.
    var local = vec3f(0.0);
    if (a == 0) { local = vec3f(0.0, 0.10 * sign(mag), -0.34); }
    else if (a == 1) { local = vec3f(0.46 * sign(mag), 0.0, 0.10); }
    else { local = vec3f(0.46 * sign(mag), 0.06, -0.24); }
    let wpj = sp + qrotate(q, local * SHIP_SCALE);
    let oc = wpj - ro;
    let tc = clamp(dot(oc, rd), 0.0, tmax);
    let dist = length(oc - rd * tc);
    let s = clamp(abs(mag) * SHIP_RCS_GAIN, 0.0, 1.4);
    sum += SHIP_RCS_COL * s * exp(-sqr(dist / (0.028 * SHIP_SCALE)) * 2.0);
  }
  return sum;
}
