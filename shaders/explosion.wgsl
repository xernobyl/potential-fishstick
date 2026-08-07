// ---------------------------------------------------------------------------
// Surface detonations.
//
// A fixed pool of BLAST_COUNT slots, each a PURE FUNCTION OF TIME — no buffers,
// no simulation pass, nothing to keep in sync. A slot's firing index is
// `floor(time/period)`, so hashing that index gives every firing a fresh site,
// and hashing it again decides whether that firing happens at all. That last part
// is what stops them arriving on a metronome.
//
// Two halves, and they live in different passes on purpose:
//
//  - the LIGHT the blast casts on a surface is real illumination, added alongside
//    the suns wherever a surface is shaded.
//
//    That light goes through the accumulation buffer, which lags by a few frames.
//    Acceptable here: variance clipping tracks a sudden brightening quickly (the
//    whole neighbourhood moves together, so the clamp box moves with it), and a
//    detonation lasts far longer than the lag.
//
//  - the FIREBALL itself is emissive and fast, and fast things must not enter the
//    accumulation buffer or they smear. So it is integrated in the composite pass,
//    downstream of TAA, alongside the limb glow and the flares.
//
// The colour track is the blast cooling as it expands: white-hot, then yellow,
// orange, and finally a deep red as the gas does work against its surroundings.
// Radius follows 1-exp(-kt): fast expansion that decelerates hard, which is what
// a real blast wave does once it has spent its overpressure.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "brdf.wgsl"

struct Blast {
  pos    : vec3f,
  radius : f32,
  colour : vec3f,
  /// Emissive intensity, already faded in and out over the life.
  power  : f32,
  /// 1 while this slot is mid-detonation, 0 otherwise.
  live   : f32,
  /// Normalised age in [0,1]. Exposed so the fireball can shape itself by it —
  /// the shock shell only exists early, the soft glow outlives it.
  age    : f32,
};

/// White-hot to deep red as the fireball expands and cools.
fn fireColour(age : f32) -> vec3f {
  let hot = mix(vec3f(1.00, 0.96, 0.88), vec3f(1.00, 0.58, 0.18),
                smoothstep(0.0, 0.32, age));
  return mix(hot, vec3f(0.60, 0.10, 0.02), smoothstep(0.28, 1.0, age));
}

fn blastAt(slot : i32) -> Blast {
  let fs = f32(slot);
  // Staggered so slots never fire together.
  let u = frame.camPos.w / BLAST_PERIOD + hash11(fs * 7.31 + 0.17);
  let cyc = floor(u);
  let seed = fs * 31.7 + cyc * 3.77;

  var b : Blast;
  // Normalised age within this firing. Past 1 the slot is dormant until the next.
  let age = fract(u) / BLAST_LIFE;
  // Only some firings happen, so the rhythm is sporadic rather than periodic.
  let fires = step(1.0 - BLAST_CHANCE, hash11(seed + 11.3));
  b.live = fires * step(age, 1.0);

  // A fresh site every firing: z uniform in [-1,1] with a uniform azimuth is the
  // correct uniform distribution on a sphere (equal-area in z).
  let z = hash11(seed) * 2.0 - 1.0;
  let a = TAU * hash11(seed + 5.1);
  let s = sqrt(max(1.0 - z * z, 0.0));
  b.pos = vec3f(cos(a) * s, sin(a) * s, z) * BLAST_SURF;

  // Blast-wave expansion: rapid, then stalling.
  b.radius = BLAST_R * (0.16 + 0.84 * (1.0 - exp(-age * 6.5)));
  b.colour = fireColour(age);
  // A hard flash on ignition over a slower burn, then a clean fade to nothing so
  // no slot ever pops off.
  let flash = exp(-age * 22.0);
  let burn = exp(-age * 2.4) * (1.0 - smoothstep(0.6, 1.0, age));
  b.power = b.live * (flash * 2.6 + burn) * BLAST_POWER;
  b.age = age;
  return b;
}

/// The visible fireball, integrated along the view ray.
///
/// Closest approach to a Gaussian ball rather than a marched volume: one dot
/// product per blast, and at this size on screen a proper march would not look
/// different. `maxT` is the scene depth, so a detonation on the far side of the
/// body is correctly hidden by it.
fn blastGlow(ro : vec3f, rd : vec3f, maxT : f32) -> vec3f {
  var sum = vec3f(0.0);
  for (var i = 0; i < BLAST_COUNT; i++) {
    let b = blastAt(i);
    if (b.power <= 0.0) { continue; }
    let oc = b.pos - ro;
    // Distance along the ray to the closest point, clamped to the visible span so
    // the glow neither leaks in front of the camera nor through the body.
    let tc = clamp(dot(oc, rd), 0.0, maxT);
    let closest = length(oc - rd * tc);
    let r = b.radius;
    // Three components, because a fireball is not one blob:
    //   the BODY of hot gas, broad and soft;
    //   a white-hot CORE, so ignition has a hard bright centre;
    //   the SHOCK SHELL, a thin bright rim at the wave front. The shell is what
    //   sells it as a detonation rather than a glow — it only exists while the
    //   front is still moving, so it fades out early and independently.
    let g = exp(-sqr(closest / r) * 1.15);
    let core = exp(-sqr(closest / (r * 0.42)) * 2.4);
    let shellW = r * 0.20;
    let shell = exp(-sqr((closest - r * 0.82) / shellW))
              * (1.0 - smoothstep(0.05, 0.5, b.age));
    sum += b.colour * b.power * (g + core * 2.0)
         + vec3f(1.0, 0.88, 0.72) * b.power * shell * 1.4;
  }
  return sum;
}

/// Direct light every live blast casts on a surface point. No shadow term: these
/// sit ON the surface they light, so the only thing that could occlude one is the
/// lobe it is already sitting on, and N.L handles that.
fn blastLight(p : vec3f, N : vec3f, V : vec3f,
              alb : vec3f, rough : f32, f0 : vec3f) -> vec3f {
  var sum = vec3f(0.0);
  for (var i = 0; i < BLAST_COUNT; i++) {
    let b = blastAt(i);
    if (b.power <= 0.0) { continue; }
    let d = b.pos - p;
    let dist2 = dot(d, d);
    // Softened inverse square: the fireball has real extent, so treating it as a
    // point singularity would blow out to infinity as the surface approaches it.
    let atten = 1.0 / (1.0 + dist2 / (b.radius * b.radius));
    let L = d * inverseSqrt(max(dist2, 1e-6));
    sum += sunLight(N, V, L, b.colour * b.power, alb, rough, f0, 1.0) * atten;
  }
  return sum;
}
