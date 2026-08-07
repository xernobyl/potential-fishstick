// ---------------------------------------------------------------------------
// The body: a pure fBM signed distance field, no base primitive.
//
//   + OCTAVES octaves of spheres, smooth-union'd. Octave 0 overlaps enough to
//     close the body on its own; finer octaves ride the surface it forms.
//   - HOLE_OCT octaves of holes on their own Fibonacci sets, smooth-subtracted.
//     Outward they get denser and relatively smaller: craters become pitting.
//   + a fine value-noise displacement for grain and a fuzzy silhouette.
//   * a heartbeat and a slow life cycle, per sphere.
//
// Placement is a spherical-Fibonacci spiral, and the *inverse* mapping
// (Keinert/Innmann/Sanger/Stamminger 2015) returns the few nearest points in
// O(1), so the field never iterates all N spheres.
// ---------------------------------------------------------------------------

//!include "hash.wgsl"
//!include "fibonacci.wgsl"

/// Each sphere grows and recedes on its own long cycle, so the body keeps
/// reorganising. Octave 0 and its holes are structural — the body would open up
/// if those vanished — so they only breathe slightly.
fn lifeEnvelope(i : f32, rotIdx : i32) -> f32 {
  var amt = 0.55;
  if (rotIdx == 0 || rotIdx == 4) { amt = 0.14; }
  let lph = hash11(i * 2.17 + f32(rotIdx) * 11.71);
  let life = fract(lifePhase() + lph);
  let env = smoothstep(0.0, 0.16, life) * smoothstep(1.0, 0.80, life);
  return 1.0 - amt * (1.0 - env);
}

/// Bounds on the per-sphere randomisation in `layerDist`, named so the marching
/// bound can be DERIVED from them rather than hand-computed. Every one of these
/// is the supremum of an expression a few lines further down; if you change one
/// there, change it here.
const SPHERE_SIZE_MAX : f32 = 1.20;   // sup of (0.85 + 0.35 * hash)
const PULSE_AMP_MAX   : f32 = 1.35;   // sup of (0.65 + 0.70 * hash)

/// The largest radius any sphere in a layer of nominal radius `rho` can reach:
/// the per-sphere size variation, then the heartbeat swell, whose amplitude
/// itself varies per sphere. `heartbeat` is normalised to peak at 1, so it
/// contributes no factor of its own.
///
/// Shared deliberately, and derived rather than measured. The marching bound and
/// the candidate rejection both need this number and they had drifted apart: the
/// bound used (1 + PULSE_R), a 5% UNDER-estimate, and an under-estimate here
/// clips geometry at the limb on every beat. One definition cannot disagree with
/// itself, and a derived one cannot rot when the pulse is retuned.
/// Slack, because this must be a STRICT upper bound and the terms above are not
/// all exact: HEARTBEAT_PEAK is a rounded decimal, and the shader evaluates in
/// f32 where the JS that derived it used f64. Derived-and-exact still leaves the
/// bound one rounding away from being violated, and being under by even a float
/// clips geometry. 1% costs a fractionally looser march and cannot be wrong.
const RADIUS_BOUND_SLACK : f32 = 1.01;

fn layerRadiusMax(rho : f32) -> f32 {
  return rho * SPHERE_SIZE_MAX * (1.0 + PULSE_R * PULSE_AMP_MAX) * RADIUS_BOUND_SLACK;
}

struct Layer { d : f32, capH : f32 };

/// Can a candidate at squared angular distance `ang2` still affect the answer?
///
/// A strict test, used twice per candidate with a progressively tighter `ang2`: once on the
/// cheap polar lower bound and once on the true value. `dLow` is monotonic in `ang2`, so a
/// lower bound on the angle gives a lower bound on the distance, and rejecting on that can
/// never discard a candidate that mattered.
fn sfMayMatter(ang2 : f32, best : f32, radial2 : f32, prScLo : f32,
               rrMax : f32, rrMax2 : f32, wantCap : bool, capMax : f32) -> bool {
  let dLow = sqrt(radial2 + prScLo * ang2) - rrMax;
  // The cap contribution vanishes once the sphere's angular rim is passed.
  let noCap = !wantCap || capMax * ang2 >= rrMax2;
  return dLow < best || !noCap;
}

/// One layer of spheres: nearest-surface distance, plus how far the layer lifts
/// the surface along `dir` so the next octave can ride on it.
/// Used for both the additive sphere layers and the subtracted hole layers.
fn layerDist(p : vec3f, dir : vec3f, n : f32, rotIdx : i32,
             rho : f32, surf : f32, wantCap : bool, nCand : i32) -> Layer {
  let c = sfCell(dir, n, rotIdx);
  var out : Layer;
  out.d = 1e5;
  out.capH = 0.0;

  // Conservative bounds on what the per-sphere animation can do, so a candidate
  // that provably cannot matter is rejected BEFORE any of it is evaluated. This
  // is the difference between paying for 9 animated spheres per layer and paying
  // for the 1-3 that can actually be nearest: the window is centred on the
  // nearest cell, so the very first candidate usually makes `out.d` tight enough
  // to reject the rest.
  //
  // Every bound here must be strict, never approximate — a candidate wrongly skipped
  // is a hole in the field, which the marcher renders as a hard seam.
  let rrMax = layerRadiusMax(rho);
  let scSlack = rho * (JITTER * 0.5 + PULSE_OFF * 1.35);
  let scLo = max(surf - scSlack, 0.0);
  let pr = length(p);
  let capMax = surf * surf;
  // Loop-invariant, so hoisted out of it rather than trusting the compiler to notice.
  let radial = pr - clamp(pr, scLo, surf + scSlack);
  let radial2 = radial * radial;
  let prScLo = pr * scLo;
  let rrMax2 = rrMax * rrMax;

  // EVERYTHING BELOW IS IN THE LAYER'S OWN FRAME. `p` and `dir` are rotated into it once here,
  // which is what lets the candidate stay unrotated — see sfDirLocal. A rotation preserves both
  // the lengths and the angles this loop asks for, so the answers are bit-comparable; the only
  // difference is that a 3x3 multiply per candidate became one per layer.
  let lp = c.lp;
  let lpp = c.rot * p;

  for (var j = 0; j < nCand; j++) {
    let pol = sfPolar(c, j);
    if (!pol.ok) { continue; }
    let i = pol.i;

    // TIER 1 — the POLAR bound, before the azimuth exists.
    //
    // The rejection test below only needs the angular distance, and the candidate's latitude
    // alone bounds that from below (see sfDotMax). So the cheap half of the candidate is enough
    // to throw most of them out, and the `fract`/`cos`/`sin` is never paid for those. The bound
    // is loose exactly when the azimuths differ a lot, which is when TIER 2 catches it.
    let ang2Lo = max(0.0, 2.0 - 2.0 * sfDotMax(c, pol));
    if (!sfMayMatter(ang2Lo, out.d, radial2, prScLo, rrMax, rrMax2, wantCap, capMax)) { continue; }

    let lcw = sfDirLocal(pol);

    // TIER 2 — the same test with the true angular distance, which is now tight.
    //
    // |p - cw*sc|^2 == (pr - sc)^2 + pr*sc*|dir - cw|^2 exactly; minimising the
    // two terms independently over the possible `sc` under-estimates the true
    // distance, which is the safe direction.
    let ang2 = max(0.0, 2.0 - 2.0 * dot(lp, lcw));      // |dir - cw|^2
    if (!sfMayMatter(ang2, out.d, radial2, prScLo, rrMax, rrMax2, wantCap, capMax)) { continue; }

    let jr = (hash11(i + f32(rotIdx) * 57.1) - 0.5) * JITTER * rho;
    var rr = rho * (0.85 + 0.35 * hash11(i * 1.7 + f32(rotIdx) * 13.3));

    // Heartbeat: each sphere swells and breathes on its own phase and amplitude,
    // so the body pulses organically rather than as one rigid balloon. Constant
    // per sphere, so the field stays smooth in space.
    let ph = hash11(i * 0.77 + f32(rotIdx) * 3.13);
    let amp = 0.65 + 0.70 * hash11(i * 1.31 + f32(rotIdx) * 7.71);
    let beat = heartbeat(beatPhase() - ph * PULSE_LAG);
    rr *= 1.0 + PULSE_R * amp * beat;
    rr *= lifeEnvelope(i, rotIdx);
    let sc = surf + jr + PULSE_OFF * rho * amp * (beat - 0.35);

    out.d = min(out.d, length(lpp - lcw * sc) - rr);   // hard min = true nearest field

    // Cap height as a *max of parabolas* rather than the exact spherical cap of
    // the nearest point. Two reasons, both about keeping the field marchable:
    // sqrt(rr^2-off^2) has infinite slope at a sphere's angular rim, and picking
    // only the nearest point makes capH JUMP wherever two neighbours of
    // different radius tie. Either one bends the ray steps into thin arc seams.
    // Hole layers discard it, so do not pay for it there.
    if (wantCap) {
      let u = surf * length(lp - lcw) / rr;        // 0 at the axis, 1 at the rim
      out.capH = max(out.capH, rr * max(0.0, 1.0 - u * u));
    }
  }
  return out;
}

/// The field.
///   maxOct  : octave budget. Shadows / AO / the interior march stop at 2.
///   doHoles : carve the hole layers. Off for shadows and AO — a hole-free field
///             is slightly *larger*, so it stays conservative for marching, and
///             the difference is invisible once blurred.
fn mapImpl(p : vec3f, maxOct : i32, doHoles : bool) -> f32 {
  let dir = normalize(p + 1e-6);
  var surf = R;
  var d = 1e5;                              // empty space; octave 0 forms the body
  var n = N0;
  var rho = RHO0;
  var hden = HOLEDEN;
  var hsc = HOLESC;

  for (var oct = 0; oct < OCTAVES; oct++) {
    if (oct >= maxOct) { break; }
    let layer = layerDist(p, dir, n, oct, rho, surf, true, SFN);
    d = smin(d, layer.d, rho * 0.55);

    if (doHoles && oct < HOLE_OCT) {
      let hole = layerDist(p, dir, n * hden, oct + 4, rho * hsc, surf + rho * HOLEOFF, false, SFN_HOLE);
      d = smax(d, -hole.d, rho * 0.35);
    }

    surf += layer.capH * 0.62;              // capH is a max, so it reads higher
    n *= NGROW;
    rho *= RHOFALL;
    hden *= HOLEDGROW;
    hsc *= HOLESFALL;
  }
  return d;
}

fn mapBody(p : vec3f) -> f32 {
  var d = mapImpl(p, OCTAVES, true);
  // Grain only near the surface (a real saving), faded in smoothly: a hard
  // cutoff is a step in the field, and the marcher renders that as stripes.
  // Against R: this band is a distance from the SURFACE, so it has to scale with the body.
  let w = 1.0 - smoothstep(R * 0.06, R * 0.20, d);
  if (w > 0.0) { d -= (detailNoise(p) - 0.5) * DETAIL * w; }
  return d;
}

/// Bounding radius, derived rather than guessed: the surface can be lifted by
/// 0.62*rr_max per octave (geometric, so bounded by the series), then the last
/// octave adds its own radius, jitter, pulse and grain on top.
/// rr_max = 1.2*rho because of the per-sphere radius variation.
fn bodyBound() -> f32 {
  let rrmax = layerRadiusMax(RHO0);
  // The radial breathing scales by the same per-sphere amplitude as the swell,
  // so it carries the 1.35 factor too; it was missing here.
  return R + rrmax * (0.62 / (1.0 - RHOFALL) + 1.0)
           + JITTER * RHO0 * 0.5 + PULSE_OFF * RHO0 * 1.35 + DETAIL;
}

fn calcNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0011;
  return normalize(
      e.xyy * mapBody(p + e.xyy) + e.yyx * mapBody(p + e.yyx)
    + e.yxy * mapBody(p + e.yxy) + e.xxx * mapBody(p + e.xxx));
}

fn softshadow(ro : vec3f, rd : vec3f, mint : f32, maxt : f32, w : f32) -> f32 {
  var res = 1.0;
  var t = mint;
  // 16 steps with a wide minimum: the result is heavily blurred by `w`, extra
  // steps buy nothing visible, and this runs twice per pixel (two suns).
  for (var i = 0; i < 16; i++) {
    let h = mapImpl(ro + rd * t, 2, false);
    res = min(res, h / (w * t));
    // Body-space lengths, hence against R — a fixed floor would take twice as many steps to
    // cross twice the body, and a fixed ceiling would step past it.
    t += clamp(h, R * 0.03, R * 0.30);
    if (res < -1.0 || t > maxt) { break; }
  }
  res = max(res, -1.0);
  return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}
