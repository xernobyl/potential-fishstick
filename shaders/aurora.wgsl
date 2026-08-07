// ---------------------------------------------------------------------------
// Auroras: curl-noise ribbons, drifting in and out of existence.
//
// The geometry is the contrail's, deliberately — one quad per segment, expanded
// perpendicular to both the path tangent and the view direction, no vertex buffer, the
// segment and the side both falling out of the vertex index. What differs is entirely in the
// shading: wider and softer, an emission palette instead of a heat gradient, and a striation
// along the ribbon so it reads as a curtain of rays rather than a tube.
//
// Colour comes from the INSTANCE INDEX, not from the buffer. Each ribbon's hue is therefore
// free — no per-sample storage, no second binding — and it stays fixed for that ribbon's
// whole life, which is what makes them read as five distinct things rather than one
// colour-cycling one.
//
// The palette is the real emission spectrum rather than an arbitrary spread of hues: 557.7 nm
// oxygen green (the dominant line), 630 nm oxygen red, 427.8 nm ionised-nitrogen violet, and
// the teal and magenta that appear where those overlap. Picking from physics is why it reads
// as aurora instead of as a rainbow.
//
// Drawn additively into the ember target, so it inherits the particles' soft occlusion and
// their place downstream of TAA — which is where anything this fast-changing belongs, since
// reprojection assumes static geometry.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"

@group(1) @binding(0) var<storage, read> ribbon : array<vec4f>;  // xyz pos, w current envelope
@group(1) @binding(1) var sceneTex : texture_2d<f32>;            // alpha = depth tag

/// Smooth 1D value noise: hashed lattice, cubic interpolation. Soft-edged, unlike a hash of a
/// floored coordinate — which is a square wave with random heights, i.e. a barcode.
fn smoothNoise1(x : f32) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(hash11(i), hash11(i + 1.0), u);
}

/**
 * Which way a ribbon segment opens out: perpendicular to both the path and the view ray, which
 * is the ordinary camera-facing trail construction.
 *
 * It is DEGENERATE when the segment points at the viewer — the cross product goes to zero and
 * its direction becomes arbitrary — and there is no formula that avoids that, because at that
 * configuration the ribbon genuinely has no preferred side. Two things were tried before the
 * one that works:
 *
 *   Snapping to a fixed axis at the degeneracy. That converts a degeneracy into a POP, which is
 *   worse: the ribbon visibly flicks over.
 *
 *   Blending toward a field-aligned axis as the segment turns toward the viewer. Physically
 *   motivated — auroral curtains really do hang along the field — but wrong twice over. It
 *   needed the camera-facing candidate's SIGN matched to the field one, and that match flips
 *   whenever the view direction lies in the curtain's plane, which is a common configuration
 *   and a new discontinuity of its own. And a purely field-aligned curtain, viewed from outside
 *   the shell, is seen almost exactly edge-on: correct for a photograph of a limb, and not what
 *   this scene wants.
 *
 * What works is to notice that at the degeneracy the quad projects to nothing anyway, and fade
 * it out — see AURORA.grazeFade, where the same quantity does double duty as the energy
 * correction for foreshortening. Then the arbitrary direction below is provably invisible, and
 * the construction is continuous everywhere it is visible, so no sign fixing is needed at all:
 * cross(dir, toCam) varies smoothly with dir, and the walker's bounded turn rate keeps dir
 * smooth along the curve.
 */
fn ribbonSide(dir : vec3f, toCam : vec3f) -> vec3f {
  let n = cross(dir, toCam);
  let l = length(n);
  return select(vec3f(1.0, 0.0, 0.0), n / l, l > 1e-5);
}

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
);

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) fade   : f32,
  @location(1) across : f32,
  @location(2) viewZ  : f32,
  @location(3) along  : f32,
  @location(4) tint   : vec3f,
  @location(5) graze  : f32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  let seg = vi / 6u;
  let c = QUAD[vi % 6u];
  let last = u32(AURORA_COUNT) - 1u;
  let base = ii * u32(AURORA_COUNT);
  let idx = min(seg + u32(c.x), last);
  let s = ribbon[base + idx];

  // MITER JOIN, which is the standard polyline expansion and what this needs that the
  // contrail does not.
  //
  // Taking one averaged tangent and offsetting by cross(tangent, toCam) — what the contrail
  // does — is correct only while the path turns gently relative to the ribbon's width. A ship
  // does. A curl-noise walker does not, so there the two adjacent quads' offsets disagree, and
  // the ribbon pinches on the inside of the turn and gaps on the outside.
  //
  // The fix is to offset along the BISECTOR of the two adjacent segment normals and divide by
  // cos(theta/2), which places the shared edge exactly where both quads want it. That factor
  // diverges as a turn approaches 180 degrees, so it is clamped — the miter limit — after which
  // the joint degrades to a bevel-ish crease instead of firing a spike across the screen.
  //
  // With the walker's turn rate now bounded (see AURORA.maxTurn) the measured joint angle stays
  // under 10 degrees, so the miter widens by 1.005x at worst and the limit is a guard rather
  // than a working part. It is kept because a limit you never reach costs nothing.
  let sPrev = ribbon[base + select(idx - 1u, 0u, idx == 0u)];
  let sNext = ribbon[base + min(idx + 1u, last)];

  var dirA = s.xyz - sPrev.xyz;      // incoming segment
  var dirB = sNext.xyz - s.xyz;      // outgoing segment
  // At the ends one of them does not exist; borrow the other so the cap is square. If both are
  // degenerate the samples are coincident and the segment has no extent to orient.
  if (length(dirA) < 1e-6) { dirA = dirB; }
  if (length(dirB) < 1e-6) { dirB = dirA; }
  if (length(dirA) < 1e-6) { dirA = vec3f(0.0, 0.0, 1.0); dirB = dirA; }
  dirA = normalize(dirA);
  dirB = normalize(dirB);

  let toCam = normalize(frame.camPos.xyz - s.xyz);
  let sa = ribbonSide(dirA, toCam);
  let sb = ribbonSide(dirB, toCam);
  var miter = sa + sb;
  let ml = length(miter);
  miter = select(sa, miter / ml, ml > 1e-5);
  let cosHalf = max(dot(miter, sa), AURORA_MITER_MIN);

  // AGE, CONTINUOUS — and the plus-phase is the whole point.
  //
  // The interval is fixed, so sample `idx` is (last - idx) intervals old and age could be read
  // straight off the index. That is what this did, and it was the single biggest source of the
  // shimmer. Every emission shifts the whole ring buffer, so a given point in the world keeps
  // its position and loses an index — its age, and therefore its width, its fade and its ray
  // phase, all jump by a full sample, four and a half times a second, while everything else
  // moves continuously. The ray striation was the worst of it: the phase stepped back 0.35 of
  // a noise cell on every shift while the drift term slid forward, i.e. a sawtooth.
  //
  // Adding the fraction of the way through the current interval makes it exact: the fraction
  // runs 0 -> 1 and resets to 0 precisely as (last - idx) gains its 1, so the sum is continuous
  // across the shift and the buffer's discreteness stops being visible at all.
  let ageSamples = f32(last - idx) + frame.aurora.w;
  // Over COUNT, not `last`, so a full-phase oldest sample still lands inside [0, 1).
  let age = ageSamples / f32(AURORA_COUNT);

  // Wider at the trailing end, so the curtain opens out behind its leading edge.
  let width = AURORA_W0 + age * AURORA_WGROW;

  var out : VOut;
  let wp = s.xyz + miter * ((c.y * 2.0 - 1.0) * width / cosHalf);
  out.pos = frame.viewProj * vec4f(wp, 1.0);
  // Two independent fades multiplied: how old this part of the ribbon is, and how bright the
  // ribbon is RIGHT NOW. The first shapes it along its length; the second is uniform across the
  // whole curtain, which is what makes one arrive and dissolve as a single sheet.
  out.fade = pow(max(1.0 - age, 0.0), AURORA_FALLOFF) * s.w;
  out.across = c.y * 2.0 - 1.0;
  // Along the ribbon in SAMPLES OF AGE, not in buffer index — continuous for the reason above,
  // and since the walker's speed is constant this is proportional to arc length, which is what
  // a striation should be pinned to.
  out.along = ageSamples;
  out.viewZ = length(wp - frame.camPos.xyz);
  out.tint = AURORA_PALETTE[ii % u32(AURORA_PALETTE_N)];
  // sin of the angle between the path and the view ray, which is exactly how much the segment
  // is foreshortened. See the fragment stage.
  out.graze = length(cross(normalize(dirA + dirB), toCam));
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  // Soft across the ribbon. Squared against the contrail's linear falloff, which concentrates
  // the light into a core with a wide dim skirt — the way a curtain looks edge-lit.
  let p = 1.0 - in.across * in.across;
  let profile = p * p;

  // RAY STRIATION. Real auroras are bundles of field-aligned rays, and that structure is most
  // of what identifies them — but it has to be SMOOTH and it has to move.
  //
  // The first attempt was `hash11(floor(along * freq))`: one random value per quantised band,
  // which is a hard-edged square wave with random heights. That is a barcode, and it looked like
  // one. Smooth value noise gives soft-edged rays; two octaves keeps them from reading as a
  // regular comb; and drifting them along the curtain over time is what makes the thing look
  // alive rather than painted on.
  let rt = in.along * AURORA_RAY_FREQ + frame.camPos.w * AURORA_RAY_DRIFT + in.tint.g * 37.0;
  let n1 = smoothNoise1(rt);
  let n2 = smoothNoise1(rt * 2.17 + 11.3);
  let rays = mix(1.0 - frame.aurora.y, 1.0, n1 * 0.65 + n2 * 0.35);

  // Soft occlusion against the scene, as the particles and the contrail do it: compare this
  // fragment's own view depth against the stored linear depth and dissolve rather than clip, so
  // a ribbon passing behind a ring fades instead of popping.
  let sceneW = textureLoad(sceneTex, toAccumPx(in.pos.xy), 0).a;
  let vis = smoothstep(0.0, 0.10, tagDepth(sceneW) - in.viewZ);

  // Near fade. The shell can bring a curtain close to the orbit camera and the chase camera can
  // fly straight through one, and either way an additive ribbon at arm's length is a
  // full-screen white blade. Fading it out by view distance covers both, and unlike shrinking
  // the shell it also covers the chase camera, whose position is not a function of the shell at
  // all.
  let near = smoothstep(AURORA_NEAR0, AURORA_NEAR1, in.viewZ);

  // FORESHORTENING, which is both the fix for the camera-facing degeneracy and the correct
  // radiometry for it — the same quantity, arrived at from two directions.
  //
  // A quad whose segment points at the viewer projects to almost nothing, so the same emitted
  // energy lands in a vanishing area and the ribbon blows out to a sliding bright blob exactly
  // where its orientation is also least defined. The projected area scales with sin(angle
  // between the path and the view ray), so scaling alpha by the same factor holds the energy
  // constant — and it reaches zero precisely where `ribbonSide` stops meaning anything, which
  // is what makes that arbitrary fallback direction unobservable.
  //
  // Linear rather than a smoothstep, deliberately: linear IS the energy correction, and a
  // smoothstep goes as the square near zero and would over-darken.
  let graze = min(1.0, in.graze / frame.aurora.z);

  let a = in.fade * profile * rays * vis * near * graze;

  // Hue shift toward the dim edges. Real auroras do this because different species emit at
  // different altitudes, and it keeps a single ribbon from looking flat — but the shift is derived
  // from the ribbon's OWN tint rather than being a fixed colour, because with nine hues in the
  // palette one shared fringe would tint every curtain the same and erase the variety. A channel
  // rotation is a 120-degree hue rotation: always a different hue, never a muddy one.
  let fringe = mix(in.tint, in.tint.brg, AURORA_FRINGE);
  let col = mix(fringe, in.tint, profile);
  return vec4f(col * a * frame.aurora.x, a);
}
