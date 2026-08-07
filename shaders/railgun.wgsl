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
};

/// Rotate by a unit quaternion.
fn qrotate(q : vec4f, v : vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

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
};

/// Point on the helix at parameter t along a shot, for a given strand.
fn helixPoint(sh : Shot, t : f32, strand : f32, spin : f32) -> vec3f {
  let origin = muzzleOf(sh);
  let axis = normalize(sh.dir.xyz);
  // Any perpendicular basis; the beam is radially symmetric so the choice is free.
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
  let e0 = normalize(cross(up, axis));
  let e1 = cross(axis, e0);
  let a = t * RAIL_TURNS * TAU + strand * PI + spin;
  // Radius opens from the muzzle then tapers, so the beam has a shape rather than
  // being a uniform corkscrew.
  let rad = RAIL_RADIUS * sin(clamp(t, 0.0, 1.0) * PI) * (0.35 + 0.65 * t);
  return origin + axis * (t * RAIL_LENGTH)
       + (e0 * cos(a) + e1 * sin(a)) * rad;
}

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  // Two strands per shot, interleaved into the instance index.
  let sh = shots[ii / 2u];
  let strand = f32(ii % 2u);

  let age = (frame.camPos.w - sh.local.w) / RAIL_LIFE;
  let segs = u32(RAIL_SEGMENTS);
  let seg = vi / 6u;
  let c = QUAD[vi % 6u];
  let t = (f32(seg) + c.x) / f32(segs);

  var out : VOut;
  if (age < 0.0 || age > 1.0) {
    // Dead or not yet fired: collapse offscreen rather than rasterising anything.
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.fade = 0.0; out.across = 0.0; out.viewZ = 0.0; out.along = 0.0;
    return out;
  }

  // The strands keep rotating as the shot decays, which is most of what makes it read
  // as spun rather than drawn.
  let spin = age * RAIL_SPIN;
  let p = helixPoint(sh, t, strand, spin);
  // Tangent by finite difference along the helix, so the ribbon follows the twist.
  let pN = helixPoint(sh, t + 0.02, strand, spin);
  var tan = pN - p;
  let tl = length(tan);
  tan = select(normalize(sh.dir.xyz), tan / max(tl, 1e-6), tl > 1e-6);

  let toCam = normalize(frame.camPos.xyz - p);
  var side = cross(tan, toCam);
  let sl = length(side);
  side = select(vec3f(1.0, 0.0, 0.0), side / max(sl, 1e-6), sl > 1e-4);

  let wp = p + side * ((c.y * 2.0 - 1.0) * RAIL_WIDTH);
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
  let col = mix(RAIL_HOT, RAIL_COLD, in.along);
  return vec4f(col * a * RAIL_GAIN * in.atmo, a);
}
