// ---------------------------------------------------------------------------
// Rail gun beams: instant, then fading — one instance per shot in the pool.
//
// The Quake II look is not the beam, it is the HELIX around it. A straight bright line
// reads as a laser; a spiral reads as something spun up a rail. So the ribbon's centre
// line is displaced around the beam axis as it advances, and a second strand runs half a
// turn out of phase, which is what makes it twist rather than wobble.
//
// Everything is derived from age. The buffer holds only where a shot started, which way
// it went, and when — so a beam in flight costs no CPU work and no uploads.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "volumetric.wgsl"

struct Shot {
  local : vec4f,    // xyz muzzle in SHIP-LOCAL space, w fire time
  dir   : vec4f,    // xyz world direction frozen at fire time, w which wing
  extra : vec4f,    // x power 0..1, y hue, z seed, w spare
};

/// A cosine palette: cheap, always in gamut, and it never produces the muddy midpoints a naive HSV
/// lerp does. One hue per shot, chosen on the CPU when the trigger goes down.
fn shotTint(hue : f32) -> vec3f {
  return 0.55 + 0.45 * cos(TAU * (hue + vec3f(0.0, 0.33, 0.67)));
}

/// Rotate by a unit quaternion.

/// The muzzle, NOW. Rebuilt from the ship's live transform every frame so the beam stays
/// welded to the wing tip rather than being left behind as the ship flies on.
fn muzzleOf(sh : Shot) -> vec3f {
  return frame.shipPos.xyz + qrotate(frame.shipRot, sh.local.xyz);
}

@group(1) @binding(0) var<storage, read> shots : array<Shot>;
@group(1) @binding(1) var sceneTex : texture_2d<f32>;   // alpha = depth tag

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
);

struct VOut {
  @location(15) atmo : vec3f,
  @builtin(position) pos : vec4f,
  @location(0) fade   : f32,
  @location(1) across : f32,
  @location(2) viewZ  : f32,
  @location(3) along  : f32,
  @location(4) tint   : vec3f,
  @location(5) power  : f32,
  @location(6) core   : f32,
};

/// Point on the helix at parameter t along a shot, for a given strand.
///
/// `strands` is how many are actually running, so the phase spacing closes up as a power shot adds
/// them — two strands sit half a turn apart, six sit a sixth of a turn apart, and the beam reads as a
/// braid rather than as two lines that gained company.
fn helixPoint(sh : Shot, t : f32, strand : f32, strands : f32, spin : f32) -> vec3f {
  let power = sh.extra.x;
  let length = RAIL_LENGTH * mix(1.0, RAIL_POWER_LENGTH, power);
  let turns = RAIL_TURNS * mix(1.0, RAIL_POWER_TURNS, power);

  // STRAND 0 OF A POWER SHOT IS THE CORE: dead straight, and the thing the rest orbit. Without it
  // there is no "main shot" for spirals to go around — six equal helices read as a braided rope,
  // which is a different and much less exciting object. Ordinary shots keep the original pair of
  // matched helices and no core, so nothing about the normal weapon changed.
  let isCore = power > 0.0 && strand < 0.5;
  // The spirals fan OUT: the innermost hugs the core, the outermost swings wide. Spacing them rather
  // than stacking them is what makes the shot read as having depth.
  let spiralIdx = max(strand - 1.0, 0.0);
  let spirals = max(strands - 1.0, 1.0);
  let fan = select(1.0, 0.3 + 1.1 * (spiralIdx / spirals), power > 0.0);
  let radius = select(RAIL_RADIUS * mix(1.0, RAIL_POWER_RADIUS, power) * fan, 0.0, isCore);
  let origin = muzzleOf(sh);
  let axis = normalize(sh.dir.xyz);
  // Any perpendicular basis; the beam is radially symmetric so the choice is free.
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
  let e0 = normalize(cross(up, axis));
  let e1 = cross(axis, e0);
  // Alternate strands counter-rotate on a POWER shot, so the spirals cross each other instead of all
  // sweeping the same way — the difference between a corkscrew and something being flung apart.
  //
  // Power only. An ordinary shot is a matched double helix wound the same way, which is the "spun up a
  // rail" look this file was written around; counter-rotating its two strands turned it into a
  // crossing X and quietly changed the weapon nobody asked to change.
  let sgn = select(1.0, -1.0, power > 0.0 && (u32(strand) & 1u) == 1u);
  let a = sgn * (t * turns * TAU + spin) + (strand / strands) * TAU;
  // Radius opens from the muzzle then tapers, so the beam has a shape rather than
  // being a uniform corkscrew.
  let rad = radius * sin(clamp(t, 0.0, 1.0) * PI) * (0.35 + 0.65 * t);
  return origin + axis * (t * length)
       + (e0 * cos(a) + e1 * sin(a)) * rad;
}

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  // Strands are interleaved into the instance index, always at the maximum. A normal shot runs two of
  // them and collapses the rest below, which costs one degenerate vertex each and keeps the draw a
  // single instanced call whatever is in the pool.
  let sh = shots[ii / u32(RAIL_STRANDS)];
  let strand = f32(ii % u32(RAIL_STRANDS));
  let power = sh.extra.x;
  // Two strands at rest, all of them at full charge — and the count has to be integral, or a strand
  // would fade in at a fractional index and read as a glitch rather than as more beam.
  let live = select(2.0, floor(mix(2.0, RAIL_STRANDS, power) + 0.5), power > 0.0);
  let age = (frame.camPos.w - sh.local.w) / (RAIL_LIFE * mix(1.0, RAIL_POWER_LIFE, power));
  let segs = u32(RAIL_SEGMENTS);
  let seg = vi / 6u;
  let c = QUAD[vi % 6u];
  let t = (f32(seg) + c.x) / f32(segs);

  var out : VOut;
  if (age < 0.0 || age > 1.0 || strand >= live) {
    // Dead or not yet fired: collapse offscreen rather than rasterising anything.
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.fade = 0.0; out.across = 0.0; out.viewZ = 0.0; out.along = 0.0;
    out.tint = vec3f(0.0); out.power = 0.0; out.core = 0.0;
    return out;
  }

  // The strands keep rotating as the shot decays, which is most of what makes it read
  // as spun rather than drawn.
  let spin = age * RAIL_SPIN;
  let p = helixPoint(sh, t, strand, live, spin);
  // Tangent by finite difference along the helix, so the ribbon follows the twist.
  let pN = helixPoint(sh, t + 0.02, strand, live, spin);
  var tan = pN - p;
  let tl = length(tan);
  tan = select(normalize(sh.dir.xyz), tan / max(tl, 1e-6), tl > 1e-6);

  let toCam = normalize(frame.camPos.xyz - p);
  var side = cross(tan, toCam);
  let sl = length(side);
  side = select(vec3f(1.0, 0.0, 0.0), side / max(sl, 1e-6), sl > 1e-4);

  let isCore = power > 0.0 && strand < 0.5;
  // The core is the thick one. Letting every strand grow with power made a shot that was uniformly
  // fat, where what reads as powerful is a bright solid centre with fine filaments whipping round it.
  let widthScale = select(mix(1.0, RAIL_POWER_WIDTH * 0.45, power), RAIL_POWER_WIDTH, isCore);
  let wp = p + side * ((c.y * 2.0 - 1.0) * RAIL_WIDTH * widthScale);
  out.pos = frame.viewProj * vec4f(wp, 1.0);
  // ATMOSPHERE IN FRONT OF THIS, as extinction only. Additive elements sit deep inside the shell —
  // the embers are born just above the surface — and without this they read as if the air were not
  // there. Per VERTEX, not per fragment: this layer overdraws heavily and the integral is smooth.
  // See volTransmittance for why in-scattering is left out.
  let toEye = wp - frame.camPos.xyz;
  let dist = length(toEye);
  out.atmo = volTransmittance(frame.camPos.xyz, toEye / max(dist, 1e-6), dist);
  // Snaps to full brightness and decays: a rail shot is an event, not a build-up.
  out.fade = pow(1.0 - age, RAIL_FALLOFF);
  out.across = c.y * 2.0 - 1.0;
  out.along = t;
  out.viewZ = length(wp - frame.camPos.xyz);
  out.tint = shotTint(sh.extra.y);
  out.power = power;
  out.core = select(0.0, 1.0, isCore);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  if (in.fade <= 0.0) { discard; }
  let profile = 1.0 - in.across * in.across;

  // Soft occlusion against the scene, as the particles and the contrail do it: dissolve
  // rather than clip, so a beam crossing behind a ring fades instead of popping.
  let sceneW = textureLoad(sceneTex, toAccumPx(in.pos.xy), 0).a;
  let vis = smoothstep(0.0, 0.06, tagDepth(sceneW) - in.viewZ);

  let a = in.fade * profile * vis;
  // Hot at the muzzle, cooling down the beam.
  let base = mix(RAIL_HOT, RAIL_COLD, in.along);
  // A POWER SHOT TAKES ITS OWN COLOUR. The core keeps a white-hot centre because it is the part that
  // should look like it is running out of headroom; the SPIRALS stay saturated, because they are the
  // part that has to still be that colour after the tone map has finished with them.
  // The core keeps a white centre and takes the tint at its EDGES, where the profile falls off. Mixing
  // uniformly gave a bar that clipped to white across its whole width and lost the colour the shot was
  // supposed to have.
  let tinted = in.tint * select(1.25, 0.35 + 1.1 * (1.0 - profile), in.core > 0.5);
  let col = mix(base, tinted, in.power * select(1.0, 0.8, in.core > 0.5));
  // The core is the bright one; the filaments only need to be seen, not to blow out. Driving every
  // strand at the full power gain clipped the whole beam to white and lost the colour entirely.
  let gain = RAIL_GAIN * mix(1.0, RAIL_POWER_GAIN * select(0.5, 0.7, in.core > 0.5), in.power);
  return vec4f(col * a * gain * in.atmo, a);
}
