// ---------------------------------------------------------------------------
// Ember simulation, and the indirect draw arguments.
//
// State lives in a storage buffer and never leaves the GPU: no readback, no
// per-frame upload. One invocation per particle.
//
// The pass also compacts the live particles into an index list and writes the
// draw call's instance count itself, so the vertex stage is only invoked for
// motes that are actually visible. That is what `drawIndirect` is for — the CPU
// never learns how many there are.
//
// Motion is derived from the beat phase rather than integrated, which is a
// deliberate trade: it costs one hash instead of a velocity, it cannot drift or
// blow up, and it stays perfectly in step with the body's pulse. The cost is
// that particles cannot collide or be pushed around — if that is ever wanted,
// this is the file that changes, and nothing else needs to know.
//
// Curl noise turbulence is layered on top of the radial drift: a divergence-free
// 3D vector field sampled from vnoise() via finite differences. It swirls the
// particles without clumping or dispersing them, and grows with the particle's
// age so the ejection is tight near the surface and turbulent far out.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sdf.wgsl"

struct Ember {
  pos   : vec3f,
  size  : f32,
  tint  : vec3f,
  alpha : f32,
};

struct DrawArgs {
  vertexCount   : u32,
  instanceCount : atomic<u32>,
  firstVertex   : u32,
  firstInstance : u32,
};

@group(1) @binding(0) var<storage, read_write> embers : array<Ember>;
@group(1) @binding(1) var<storage, read_write> liveList : array<u32>;
@group(1) @binding(2) var<storage, read_write> drawArgs : DrawArgs;

/// Divergence-free 3D vector field from the curl of a multi-octave noise
/// potential (fbm3). Sampled at `p` (world space), animated by `t`, with
/// feature scale `s`. Richer than single-octave curl — four octaves of
/// detail from tight swirls to broad arcs.
///
/// Evaluates fbm3 6 times (24 vnoise calls total) — heavier but produces
/// visible turbulence with no parameter tuning.
fn curlNoise(p : vec3f, t : f32, s : f32, eps : f32) -> vec3f {
  let sp = p * s + t;
  let h = eps * s;
  let nx_p = fbm3(sp + vec3f( h, 0, 0));
  let nx_m = fbm3(sp + vec3f(-h, 0, 0));
  let ny_p = fbm3(sp + vec3f(0,  h, 0));
  let ny_m = fbm3(sp + vec3f(0, -h, 0));
  let nz_p = fbm3(sp + vec3f(0, 0,  h));
  let nz_m = fbm3(sp + vec3f(0, 0, -h));
  let inv = 1.0 / (2.0 * eps);
  return vec3f(
    (ny_p - ny_m) - (nz_p - nz_m),
    (nz_p - nz_m) - (nx_p - nx_m),
    (nx_p - nx_m) - (ny_p - ny_m),
  ) * inv;
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x != 0u) { return; }
  drawArgs.vertexCount = 6u;              // two triangles per billboard
  atomicStore(&drawArgs.instanceCount, 0u);
  drawArgs.firstVertex = 0u;
  drawArgs.firstInstance = 0u;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= u32(EMBER_COUNT)) { return; }
  let fi = f32(i);

  // Each mote runs its own cycle, offset so they do not all leave together.
  let phase = hash11(fi * 1.37 + 0.5);
  let life = fract(beatPhase() * EMBER_RATE + phase);

  // Spread over a forward spherical-Fibonacci set so they leave evenly from all
  // sides, with a per-mote wobble so the pattern does not read as a lattice.
  var dir = sfPointFwd(fi, f32(EMBER_COUNT));
  let wob = vec3f(hash11(fi * 3.1), hash11(fi * 5.7), hash11(fi * 9.3)) - 0.5;
  dir = normalize(dir + wob * 0.22);

  // Drift outward, easing as they go: fast off the surface, slowing as they cool.
  let travel = EMBER_TRAVEL * (1.0 - pow(1.0 - life, 2.2));
  let radius = EMBER_R0 + travel;

  var e : Ember;
  e.pos = dir * radius;
  // Curl noise turbulence: divergence-free 3D swirl that grows with the
  // particle's age. Tight radial ejection near the surface, increasingly
  // turbulent as the motes cool and drift outward.
  let turb = curlNoise(e.pos, beatPhase() * 0.5, 1.1, 0.05);
  e.pos += turb * (EMBER_TURB * life * 0.7);
  // Swell then shrink, and a per-mote base size.
  e.size = EMBER_SIZE * (0.35 + 1.5 * sin(PI * life)) * (0.55 + 0.9 * hash11(fi * 2.11));
  // Cool from white-hot to deep ember as they travel.
  e.tint = EMBER_COL * mix(vec3f(1.25, 1.1, 0.95), vec3f(1.0, 0.55, 0.22), life);
  e.alpha = sin(PI * life);              // fade in and out
  embers[i] = e;

  // Compact: only motes with something to show get an instance. atomicAdd
  // returns this thread's slot, so no ordering or locking is needed.
  if (e.alpha > 0.02) {
    let slot = atomicAdd(&drawArgs.instanceCount, 1u);
    liveList[slot] = i;
  }
}
