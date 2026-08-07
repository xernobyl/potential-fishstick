// ---------------------------------------------------------------------------
// Volumetric single scattering in the atmosphere, WITH SHADOWS.
//
// What this replaces is worth stating first, because it is the point. The atmosphere used to be
// `limbGlow`: one smoothstep on the ray's perpendicular distance from the body, added in the
// composite. A halo painted around the silhouette. It could not be shadowed, it could not sit in
// front of anything, and it lived in the composite because a screen-space halo has no world
// position to reproject. This is the integral it was standing in for.
//
// SINGLE SCATTERING, ray-marched, and the modern froxel machinery is deliberately not here.
// Hillaire's froxel grid (Frostbite, SIGGRAPH 2015) is the right answer when the medium is
// arbitrary, animated and lit by many local lights, because it decouples the volume's cost from
// screen resolution and gives the volume its own temporal reprojection. None of that applies to a
// thin shell around one body lit by two distant suns: there is no local light to voxelise, the
// density is an analytic function of altitude, and a 3D grid would cost a target and two passes to
// win nothing. What IS borrowed is the part that matters at low sample counts — jittered sample
// placement resolved temporally, and analytic integration of each step rather than a midpoint sum.
//
// WHERE IT RUNS is the load-bearing decision. Upstream of the temporal resolve, in the passes that
// own a surface: the march pass for the body and the background, the ring pass for the rings. That
// buys two things. Every sample is world-anchored, so the accumulation buffer is averaging a real
// integral rather than a screen-space effect. And the step offset can be jittered per frame, which
// turns 12 steps into an effectively converged integral once TAA has averaged a handful of frames —
// the same trade the rest of this renderer makes everywhere else.
//
// The cost of that choice: each pass integrates only to its OWN surface, so both have to call it.
// A single call in the composite would need no cooperation, but it would also get no temporal
// averaging, and 12 unaveraged steps through a shadowed medium is exactly the banding-and-crawl
// this project spends its effort removing.
//
// SHADOWS ARE ANALYTIC. Both occluders are closed-form, so there is no shadow map at all:
//   - the rings, because `ringDefAt` gives every hoop's axis and radius from the clock, and a
//     shadow is then one ray-plane crossing and a radius compare;
//   - the body, as a soft sphere. It is really a lumpy field, and marching it per sample per sun
//     would be 12 x 2 x 16 field evaluations a pixel against the 12 the whole march costs now.
//     A soft terminator against the mean radius is wrong by less than the penumbra it replaces.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "ring_geom.wgsl"

struct Scatter {
  inScatter     : vec3f,
  transmittance : vec3f,
};

/// Density at a point: exponential in altitude above the body, as an atmosphere is.
///
/// Clamped at the surface rather than allowed to keep growing inward. The body's real surface is
/// below its mean radius in places, so a ray can legitimately be inside `R` without having hit
/// anything, and an unclamped exponential would put a bright bead there.
fn volDensity(p : vec3f) -> f32 {
  let alt = length(p) - R;
  return exp(-max(alt, 0.0) * VOL_FALLOFF);
}

/// How much of a sun reaches `p` past the body. Soft, and soft is the honest choice: the occluder
/// is a lumpy distance field, so a hard terminator would be confidently wrong where a smooth one
/// is only approximately right — and a real limb has a penumbra several times wider than the
/// error either way.
fn bodyShadow(p : vec3f, L : vec3f) -> f32 {
  // Perpendicular distance from the body's centre to the ray p + s*L, s >= 0. If the closest
  // approach lies behind `p` the nearest point of the allowed segment is `p` itself.
  let pl = dot(p, L);
  let perp = select(sqrt(max(dot(p, p) - pl * pl, 0.0)), length(p), pl >= 0.0);
  // VOL_BODY_R, not R. R is the field's MEAN radius and the lumps reach half again as far, so a
  // terminator at R lets light through where a lobe should be blocking it — a gap that grew with
  // the body when the world doubled. This sits between the two, which is the best a single sphere
  // can do against a silhouette that is not one.
  let rs = R * VOL_BODY_R;
  return smoothstep(rs * (1.0 - VOL_BODY_SOFT), rs * (1.0 + VOL_BODY_SOFT), perp);
}

/// How much of a sun reaches `p` past the rings. Exact, and cheap enough to do per sample.
///
/// A ring is an annulus in a plane through the origin, so the shadow test is: cross the plane
/// along the light direction, and ask whether the crossing radius lands inside the annulus. One
/// divide, one length and two smoothsteps per ring — against a shadow map, which would need a
/// pass, a target and a resolution compromise, and would still not be exact.
///
/// The penumbra widens with distance from the occluder, which is what a real one does and also
/// what stops the edge aliasing: a hard analytic annulus sweeping through a low-step march is a
/// crawling line, and this is a soft gradient several samples wide.
/// The only part of a ring the shadow test needs: its axis, its radius and its radial half-width.
/// Four floats and a fifth, against RingDef's eighteen — and none of it depends on where along the
/// ray you are, which is the whole point of `ringShadowSetup`.
struct RingOccluder {
  azr   : vec4f,   // xyz axis, w radius
  halfW : f32,
};

/// Evaluate every ring's attitude ONCE.
///
/// `ringDefAt` is a function of the ring index and the clock — not of position — so calling it
/// inside the march was pure waste: 12 steps times 2 suns times 3 rings is 72 evaluations per
/// pixel of a value that changes 3 times per frame. Each one costs a hash, three sin/cos pairs, a
/// cross product and two normalises. Hoisting it is the difference between the atmosphere costing
/// 2.1 ms and costing a fraction of that.
fn ringShadowSetup(t : f32) -> array<RingOccluder, RING_COUNT> {
  var out : array<RingOccluder, RING_COUNT>;
  for (var i = 0u; i < u32(RING_COUNT); i++) {
    let r = ringDefAt(i, t);
    out[i].azr = vec4f(r.az, r.radius);
    out[i].halfW = r.halfW;
  }
  return out;
}

fn ringShadow(rings : array<RingOccluder, RING_COUNT>, p : vec3f, L : vec3f) -> f32 {
  var sh = 1.0;
  for (var i = 0u; i < u32(RING_COUNT); i++) {
    let r = rings[i];
    let denom = dot(L, r.azr.xyz);
    // A light travelling parallel to the ring's plane never crosses it.
    if (abs(denom) < 1e-4) { continue; }
    let s = -dot(p, r.azr.xyz) / denom;
    // Only occluders BETWEEN the point and the sun.
    if (s <= 0.0) { continue; }
    let q = p + L * s;
    // `q` lies in the plane, so its length IS the in-plane radius.
    let rad = length(q);
    let soft = VOL_RING_SOFT * (1.0 + s * VOL_RING_SPREAD);
    let inner = smoothstep(r.azr.w - r.halfW - soft, r.azr.w - r.halfW + soft, rad);
    let outer = 1.0 - smoothstep(r.azr.w + r.halfW - soft, r.azr.w + r.halfW + soft, rad);
    sh *= 1.0 - frame.volume.y * inner * outer;
  }
  return sh;
}

/// Where in its step each sample sits: interleaved gradient noise (Jimenez), spatial only.
///
/// FRAME-STATIC, and the first version was not — it advanced by the golden ratio every frame so
/// the accumulation would converge the integral. That reasoning was wrong, and measurably so.
///
/// Injecting per-frame noise into a VARIANCE-CLIPPED accumulator does not just add noise that
/// averages away. The clip box is built from the 3x3 neighbourhood of the CURRENT sample, so a
/// noisier sample means a wider box, and a wider box admits more stale history — everywhere, not
/// only where the noise was. Measured: the accumulation buffer's high-frequency wobble under
/// sub-pixel camera motion went from 0.80% before this pass existed to 2.37% with the temporal
/// term, against 0.33% for the un-antialiased additive layer. The atmosphere covers most of the
/// screen, so this was the dominant shimmer in the whole image.
///
/// Static, the integral is a deterministic function of pixel and camera. It changes between frames
/// only by as much as the view genuinely changed, which is exactly what the resolve is built to
/// reproject — the same contract every other part of this renderer honours. The step count carries
/// the quality instead, and the spatial dither keeps it from reading as shells.
///
/// This also removes the THIRD independent source of per-frame randomness. There should be one: the
/// shared `frame.jitter`, expressed as a ray offset by the compute passes and as a clip offset by
/// the raster ones (see `jitterClip`). Two expressions of one number is inherent; a third number of
/// its own was not.
fn volStepOffset(px : vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
}

/**
 * EXTINCTION ONLY, along a segment. No shadows, no phase function, no in-scattering.
 *
 * For the additive layer — embers, contrails, rail guns — which is drawn AFTER the temporal resolve
 * and therefore cannot afford a jittered march: 12 unaveraged steps per fragment through a
 * heavily-overdrawn particle field would cost more than the whole atmosphere does. But an emissive
 * mote sitting deep in the air should still be dimmed and reddened by what is in front of it, and
 * that is transmittance. The in-scattering it displaces is second order for something that is
 * itself a light source, and it is the expensive half.
 *
 * Cheap enough to be honest about: density is smooth and monotonic in altitude and there are no
 * shadows in here to alias against, so a midpoint rule over a handful of samples is not an
 * approximation worth apologising for. Call it from the VERTEX stage — once per sprite or ribbon
 * vertex rather than once per fragment.
 */
fn volTransmittance(ro : vec3f, rd : vec3f, tMax : f32) -> vec3f {
  if (frame.volume.x <= 1e-5) { return vec3f(1.0); }
  let shell = iSphere(ro, rd, ATMO_R);
  if (shell.y <= 0.0) { return vec3f(1.0); }
  let t0 = max(shell.x, 0.0);
  let t1 = min(shell.y, tMax);
  if (t1 <= t0) { return vec3f(1.0); }

  let dt = (t1 - t0) / f32(VOL_TR_STEPS);
  var od = 0.0;
  for (var i = 0; i < VOL_TR_STEPS; i++) {
    od += volDensity(ro + rd * (t0 + (f32(i) + 0.5) * dt));
  }
  return exp(-od * dt * frame.volume.x * VOL_ALBEDO);
}

/**
 * The integral, from the camera to `tMax`.
 *
 * Returns in-scattered radiance and the transmittance behind it, so the caller composites as
 *     colour = colour * transmittance + inScatter
 * which is the only ordering that conserves energy when something is already there.
 */
fn volumetric(ro : vec3f, rd : vec3f, tMax : f32, px : vec2f) -> Scatter {
  var out : Scatter;
  out.inScatter = vec3f(0.0);
  out.transmittance = vec3f(1.0);

  // A zero-thickness medium is the natural way to switch this off, and making it genuinely free
  // rather than merely invisible is what lets the slider double as an A/B: the difference between
  // sigma 0 and sigma 0.55 is then exactly what the atmosphere costs.
  if (frame.volume.x <= 1e-5) { return out; }

  let shell = iSphere(ro, rd, ATMO_R);
  if (shell.y <= 0.0) { return out; }
  let t0 = max(shell.x, 0.0);
  let t1 = min(shell.y, tMax);
  if (t1 <= t0) { return out; }

  let dt = (t1 - t0) / f32(VOL_STEPS);
  let jit = volStepOffset(px);

  // The phase function is constant along the ray — the suns are directional — so it is evaluated
  // once rather than per step. CORNETTE-SHANKS rather than Henyey-Greenstein: same forward lobe
  // parameter, plus the (1 + cos^2) factor that HG lacks, which restores the backscatter a real
  // aerosol has. It is the standard improvement for Mie scattering and costs two multiplies.
  let ph1 = phaseCS(dot(rd, SUN1_DIR), frame.volume.z);
  let ph2 = phaseCS(dot(rd, SUN2_DIR), frame.volume.z);

  // Once, not per sample per sun. See ringShadowSetup.
  let rings = ringShadowSetup(frame.camPos.w);

  for (var i = 0; i < VOL_STEPS; i++) {
    let t = t0 + (f32(i) + jit) * dt;
    let p = ro + rd * t;
    let dens = volDensity(p);
    if (dens < 1e-4) { continue; }

    let lit = SUN1_COL * (ph1 * bodyShadow(p, SUN1_DIR) * ringShadow(rings, p, SUN1_DIR))
            + SUN2_COL * (ph2 * bodyShadow(p, SUN2_DIR) * ringShadow(rings, p, SUN2_DIR))
            + VOL_AMBIENT;

    // ANALYTIC integration of the step rather than a midpoint sum. Over a step of constant density
    // the source term integrates to S * (1 - Tr) / sigma_t, and with S = sigma_s * L the extinction
    // cancels — so this is both exact for the step and cheaper than the sum it replaces. It is what
    // stops a 12-step march banding as the optical depth rises.
    //
    // The extinction is SPECTRAL, and that is a correction rather than a refinement. It used to be
    // grey with the albedo applied to the scattered term instead, which meant the medium scattered
    // blue light out of the beam without ever removing it — energy conserved overall but not per
    // channel. Making VOL_ALBEDO the shape of sigma_t and treating the gas as purely scattering
    // (which it nearly is) gives `1 - Tr` as the scattered fraction directly, and reddens what is
    // seen THROUGH the atmosphere, which is the whole reason a sunset is red.
    let tr = exp(-dens * frame.volume.x * dt * VOL_ALBEDO);
    out.inScatter += out.transmittance * lit * (1.0 - tr);
    out.transmittance *= tr;

    // Nothing behind this point can contribute once the medium is opaque.
    if (max(out.transmittance.r, out.transmittance.b) < 0.003) { break; }
  }
  return out;
}
