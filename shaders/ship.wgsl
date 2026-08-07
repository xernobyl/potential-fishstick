// ---------------------------------------------------------------------------
// The ship's PLUMES: engine exhaust and RCS puffs, as additive glow along the view ray.
//
// The hull used to be here too, as a small signed-distance field marched in its own local space. It is
// a generated triangle mesh now — see scene/ship_sdf.js for the shape and shipmesh.wgsl for the
// material — and the whole marched implementation went with it: the field, its normal estimator, the
// bounding-sphere test, the step loop and the hull shading. Roughly ninety lines that nothing called.
//
// What stayed is what is genuinely volumetric. A plume has no surface to rasterise: it is emission
// integrated along the ray, so it belongs in the march, where a few Gaussian lobes sampled down the
// exhaust cost far less than any geometry that could stand in for them.
//
// Authored in the hull's local space at scale 1, with SHIP_SCALE dividing that out once — the same
// convention the mesh uses, so a nozzle in ship_sdf.js and a plume here are in the same units and
// stay lined up.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"

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
