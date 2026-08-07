// ---------------------------------------------------------------------------
// Ring GEOMETRY: where each hoop is, and which way it is facing.
//
// Extracted from rings.wgsl for the same reason brdf.wgsl and fibonacci.wgsl were
// extracted from the shaders that first needed them: something other than the renderer of
// a thing now wants to know where that thing IS. The volumetric scattering casts the rings'
// shadows through the atmosphere, and it can do that ANALYTICALLY — no shadow map, no depth
// pass — precisely because a ring's position and attitude are closed-form functions of the
// instance index and the clock. Pulling that out is what lets a second caller have it
// without also compiling a metal BRDF, a starfield and a detonation.
//
// Stateless, like the satellites and the detonations: nothing is stored between frames, so
// two passes evaluating this at the same `t` necessarily agree about where the rings are.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"

struct RingDef {
  radius : f32,
  halfW  : f32,   // radial half-thickness
  halfH  : f32,   // axial half-height
  ax     : vec3f, // in-plane basis
  ay     : vec3f,
  az     : vec3f, // the ring's axis
};

fn ringDefAt(i : u32, t : f32) -> RingDef {
  let fi = f32(i);
  let h0 = hash11(fi * 5.17 + 0.31);
  let h1 = hash11(fi * 9.71 + 4.13);

  var r : RingDef;
  // Concentric: each ring a fixed step further out, so they nest rather than
  // intersecting. Widths shrink outward, which reads as a hierarchy.
  r.radius = RING_R0 + fi * RING_GAP;
  // Mirrored by `ringDims` in tuning.js, which the mesh generator uses; the taper constants are
  // injected from there rather than written twice.
  r.halfW = RING_W * (1.0 - RING_WTAPER * fi);
  r.halfH = RING_H * (1.0 - RING_HTAPER * fi);

  // Pseudo-random axis, precessing on two incommensurate periods so no ring ever
  // returns to the same attitude and none of them share a rhythm.
  let a = TAU * h0 + t * RING_PRECESS * (0.6 + 0.8 * h0);
  let b = 1.1 + 1.4 * h1 + t * RING_PRECESS * 0.61 * (0.5 + h1);
  let sb = sin(b);
  r.az = normalize(vec3f(cos(a) * sb, cos(b), sin(a) * sb));

  // Any stable in-plane basis will do; the surface detail is what makes the spin
  // visible, so the basis is rotated about the axis by the spin angle here rather
  // than the geometry being re-derived per vertex.
  let spin = t * RING_SPIN * (0.7 + 0.9 * h1) + TAU * h0;
  // Pick a reference not parallel to the axis, then orthonormalise.
  // `ref` is a WGSL reserved keyword.
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(r.az.y) > 0.9);
  let e0 = normalize(cross(up, r.az));
  let e1 = cross(r.az, e0);
  r.ax = e0 * cos(spin) + e1 * sin(spin);
  r.ay = cross(r.az, r.ax);
  return r;
}
