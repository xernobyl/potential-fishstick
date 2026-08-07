// ---------------------------------------------------------------------------
// Hashes and value noise.
//
// These feed both the distance field and the shading, so they must be cheap and
// they must be stable: the SDF calls hash11 several times per candidate sphere,
// per march step.
// ---------------------------------------------------------------------------

fn hash11(p0 : f32) -> f32 {
  var p = fract(p0 * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

fn hash13(p0 : vec3f) -> f32 {
  var p = fract(p0 * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

fn hash21(p : vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(41.7, 289.1))) * 43758.5453);
}

fn vnoise(x : vec3f) -> f32 {
  let p = floor(x);
  var f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash13(p + vec3f(0.0, 0.0, 0.0)), hash13(p + vec3f(1.0, 0.0, 0.0)), f.x),
        mix(hash13(p + vec3f(0.0, 1.0, 0.0)), hash13(p + vec3f(1.0, 1.0, 0.0)), f.x), f.y),
    mix(mix(hash13(p + vec3f(0.0, 0.0, 1.0)), hash13(p + vec3f(1.0, 0.0, 1.0)), f.x),
        mix(hash13(p + vec3f(0.0, 1.0, 1.0)), hash13(p + vec3f(1.0, 1.0, 1.0)), f.x), f.y),
    f.z);
}

fn fbm3(p0 : vec3f) -> f32 {
  var p = p0;
  var a = 0.5;
  var s = 0.0;
  for (var i = 0; i < 4; i++) {
    s += a * vnoise(p);
    p = p * 2.03 + 1.7;
    a *= 0.5;
  }
  return s;
}

/// Fine two-octave noise, used as a real displacement: it puts grain on the
/// surface and fuzz on the silhouette.
/// Surface grain. The frequencies are divided by R, which is what makes them a property of the
/// BODY rather than of the world: a bare `p * 7.0` is 7 cycles per world unit, so doubling the
/// planet halves the grain's relative size. Against R it is 7 cycles per body radius and stays put.
/// How much of a noise octave survives the pixel footprint.
///
/// `fpPeriods` is the footprint measured in PERIODS of that octave: a pixel covering half a lattice
/// cell is at Nyquist, and past that the octave is no longer a signal - it is a sample of a function
/// varying faster than the grid can carry, which is exactly what aliases and, once the jitter moves
/// the sample every frame, what the temporal resolve has to admit as variance.
///
/// Faded rather than cut, because a hard cutoff is a visible boundary in space AND in time as the
/// camera moves toward something. The band from 0.35 to 0.9 of Nyquist is deliberately below 1: value
/// noise interpolates over its cell, so its effective content extends past the naive limit.
///
/// Callers mix toward the noise's MEAN by this weight, not toward zero - fading a 0..1 noise to zero
/// darkens the surface as it recedes, which reads as a lighting bug rather than as filtering.
fn bandWeight(fpPeriods : f32) -> f32 {
  return 1.0 - smoothstep(0.35, 0.9, fpPeriods);
}

/// Band-limited value noise. `k` is the frequency `x` was already scaled by, `fp` the world-space
/// pixel footprint, so `k * fp` is the footprint in periods.
fn vnoiseBL(x : vec3f, k : f32, fp : f32) -> f32 {
  let w = bandWeight(k * fp);
  if (w <= 0.0) { return 0.5; }
  return mix(0.5, vnoise(x), w);
}

/// The shading copy of the surface grain, band-limited per octave.
///
/// NOT for the field. `mapBodyAt` subtracts `detailNoise` from the distance, and filtering that by a
/// pixel footprint would make the GEOMETRY view-dependent: the same point would report different
/// distances to the march, to `calcNormal`, and to the reflection pass, which is a broken SDF rather
/// than a filtered one. Only shading may do this.
fn detailNoiseBL(p : vec3f, fp : f32) -> f32 {
  return 0.62 * vnoiseBL(p * (7.0 / R), 7.0 / R, fp)
       + 0.38 * vnoiseBL(p * (15.0 / R), 15.0 / R, fp);
}

fn detailNoise(p : vec3f) -> f32 {
  return 0.62 * vnoise(p * (7.0 / R)) + 0.38 * vnoise(p * (15.0 / R));
}

/// Radical inverse (van der Corput, base 2). A low-discrepancy sequence
/// converges far faster than white noise for the same number of samples, which
/// is exactly what the temporal jitter wants.
fn radicalInverse(bits0 : u32) -> f32 {
  var bits = bits0;
  var r = 0.0;
  var f = 0.5;
  for (var i = 0; i < 16; i++) {
    r += f32(bits & 1u) * f;
    bits >>= 1u;
    f *= 0.5;
  }
  return r;
}

/// Uniform point on a unit disk. Used for the lens aperture.
fn diskSample(u : vec2f) -> vec2f {
  let a = TAU * u.x;
  let r = sqrt(clamp(u.y, 0.0, 1.0));
  return r * vec2f(cos(a), sin(a));
}
