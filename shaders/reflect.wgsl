// ---------------------------------------------------------------------------
// Raymarched reflections, cheap enough to put on every metal surface.
//
// The metal was mirroring only `background`, i.e. the sky. That is what makes
// rendered metal look fake: the biggest, brightest thing in the scene sits right
// next to it and does not appear in it at all.
//
// Three things keep the cost down, and the first is the one that matters:
//
//  1. A BOUNDING SPHERE TEST before any marching. A reflection ray that misses the
//     body has nothing to hit but sky, and that is one quadratic instead of a march.
//     Most rays miss — the rings are outside the body looking outward — so most
//     pixels pay a dot product and a square root.
//  2. A COARSE field: one octave, no holes. Reflections here are seen in curved,
//     scratched, partly rough metal at a fraction of the screen, and the fine
//     octaves are invisible through that. This is the same reasoning the shadow and
//     AO taps already use.
//  3. A SHORT step budget with a relaxed step scale, because the coarse field has no
//     DETAIL noise in it and is therefore properly 1-Lipschitz — the near-surface
//     safety factor the primary march needs does not apply.
//
// On a hit the body is approximated rather than fully shaded: `shadeBody` does
// sphere-integral AO, transmission, an interior refraction march and a core term,
// and calling it from here would nest one expensive march inside another for
// something seen in a scratched mirror.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sdf.wgsl"
//!include "sky.wgsl"

/// Cheap stand-in for the body's appearance, for reflections only.
fn reflectedBody(p : vec3f, rd : vec3f) -> vec3f {
  // Normal from the coarse field, and a wider epsilon than the primary normal uses:
  // a tight one would resolve detail this field does not contain.
  let e = vec2f(1.0, -1.0) * 0.006;
  let n = normalize(
      e.xyy * mapImpl(p + e.xyy, 1, false) + e.yyx * mapImpl(p + e.yyx, 1, false)
    + e.yxy * mapImpl(p + e.yxy, 1, false) + e.xxx * mapImpl(p + e.xxx, 1, false));

  // The gummy's hue, without the noise fields that only read at full size.
  let candy = mix(vec3f(1.00, 0.12, 0.62), vec3f(0.52, 0.18, 1.00),
                  0.5 + 0.5 * sin(dot(p, vec3f(1.7, 2.1, 1.3))));

  var col = candy * 0.10 * (max(dot(n, SUN1_DIR), 0.0) * 0.7 + 0.25);
  col += SUN2_COL * candy * max(dot(n, SUN2_DIR), 0.0) * 0.05;
  // The core burning out through the shell — the brightest thing a reflection of
  // this body should carry, and the reason reflecting it is worth doing at all.
  let shell = max(length(p) - CORE_R, 0.0);
  col += CORE_COL * 0.16 * exp(-shell * CORE_ATT * 0.8)
       * (0.6 + 0.4 * heartbeat(beatPhase()));
  // A rim, so the silhouette in the reflection is not flat.
  col += candy * pow(clamp(1.0 + dot(rd, n), 0.0, 1.0), 3.0) * 0.35;
  return col;
}

/// One reflection ray. Returns the sky on a miss, so this drops straight into an
/// existing environment term without changing how it is weighted.
fn traceReflection(ro : vec3f, rd : vec3f) -> vec3f {
  let bound = bodyBound();
  let bs = iSphere(ro, rd, bound);
  // Behind us or missing entirely: sky, for one quadratic.
  if (bs.y <= 0.0) { return background(rd); }

  var t = max(bs.x, 0.0);
  for (var i = 0; i < REFL_STEPS; i++) {
    let p = ro + rd * t;
    let d = mapImpl(p, REFL_OCT, false);
    if (d < 0.004 * t) { return reflectedBody(p, rd); }
    // 0.9 rather than the primary march's 0.65: no DETAIL noise in this field, so
    // it is properly 1-Lipschitz and cannot overshoot at this scale.
    t += max(d, 0.008) * 0.9;
    if (t > bs.y) { break; }
  }
  return background(rd);
}
