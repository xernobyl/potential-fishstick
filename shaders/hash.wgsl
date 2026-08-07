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
fn detailNoise(p : vec3f) -> f32 {
  return 0.62 * vnoise(p * 7.0) + 0.38 * vnoise(p * 15.0);
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
