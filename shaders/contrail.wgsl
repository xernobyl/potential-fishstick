// ---------------------------------------------------------------------------
// Contrail: a camera-facing ribbon through the ship's past positions.
//
// Two ribbons, one per nacelle. One quad per segment, expanded perpendicular to BOTH
// the path tangent and the view
// direction — which is what keeps a zero-thickness curve visible from any angle. No
// vertex buffer: the segment index and which side of the ribbon a vertex is on both
// come out of the vertex index.
//
// Drawn additively into the ember target, so it shares the particles' soft occlusion
// against the scene depth and needs no pass of its own.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "volumetric.wgsl"

@group(1) @binding(0) var<storage, read> trail : array<vec4f>;  // xyz pos, w throttle
@group(1) @binding(1) var sceneTex : texture_2d<f32>;           // alpha = depth tag

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
);

struct VOut {
  @location(15) atmo : vec3f,
  @builtin(position) pos : vec4f,
  @location(0) fade  : f32,
  @location(1) across : f32,
  @location(2) viewZ : f32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  let seg = vi / 6u;
  let c = QUAD[vi % 6u];
  let last = u32(TRAIL_COUNT) - 1u;
  // One ribbon per nacelle, stored as contiguous halves of a single buffer, so the
  // instance index is all it takes to pick one.
  let base = ii * u32(TRAIL_COUNT);
  let idx = min(seg + u32(c.x), last);
  let s = trail[base + idx];

  // Central difference for the tangent, clamped at this ribbon's own ends.
  let prev = trail[base + select(idx - 1u, 0u, idx == 0u)];
  let next = trail[base + min(idx + 1u, last)];
  var tan = next.xyz - prev.xyz;
  let tl = length(tan);
  // A stalled ship emits coincident samples, so the tangent can vanish. Any direction
  // will do then, because the ribbon has collapsed to a point and is fading anyway.
  tan = select(vec3f(0.0, 0.0, 1.0), tan / max(tl, 1e-6), tl > 1e-5);

  let toCam = normalize(frame.camPos.xyz - s.xyz);
  var side = cross(tan, toCam);
  let sl = length(side);
  side = select(vec3f(1.0, 0.0, 0.0), side / max(sl, 1e-6), sl > 1e-4);

  // Age is implied by index: the interval is fixed, so sample `idx` of `last` is
  // exactly (last - idx) intervals old. Nothing to store, nothing to drift.
  let age = f32(last - idx) / f32(last);
  let width = TRAIL_W0 + age * TRAIL_WGROW;

  var out : VOut;
  let wp = s.xyz + side * ((c.y * 2.0 - 1.0) * width);
  out.pos = frame.viewProj * vec4f(wp, 1.0);
  // ATMOSPHERE IN FRONT OF THIS, as extinction only. Additive elements sit deep inside the shell —
  // the embers are born just above the surface — and without this they read as if the air were not
  // there. Per VERTEX, not per fragment: this layer overdraws heavily and the integral is smooth.
  // See volTransmittance for why in-scattering is left out.
  let toEye = wp - frame.camPos.xyz;
  let dist = length(toEye);
  out.atmo = volTransmittance(frame.camPos.xyz, toEye / max(dist, 1e-6), dist);
  // Fades as it ages, and carries how hard the engine was working when it was laid
  // down — so the trail thins where the pilot was coasting.
  out.fade = pow(1.0 - age, TRAIL_FALLOFF) * (0.25 + 0.75 * s.w);
  out.across = c.y * 2.0 - 1.0;
  out.viewZ = length(wp - frame.camPos.xyz);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  // Soft across the ribbon, so it has no hard edge.
  let profile = 1.0 - in.across * in.across;

  // Soft occlusion against the scene, exactly as the particles do it: compare this
  // fragment's own view depth against the stored linear depth and dissolve rather than
  // clip, so a trail passing behind a ring fades instead of popping.
  let sceneW = textureLoad(sceneTex, toAccumPx(in.pos.xy), 0).a;
  let vis = smoothstep(0.0, 0.08, tagDepth(sceneW) - in.viewZ);

  let a = in.fade * profile * vis;
  // Cools as it disperses: hot near the nozzle, cold vapour further back.
  let col = mix(TRAIL_COLD, TRAIL_HOT, in.fade);
  return vec4f(col * a * TRAIL_GAIN * in.atmo, a);
}
