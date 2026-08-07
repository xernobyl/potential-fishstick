// ---------------------------------------------------------------------------
// The sky.
//
// Split into `bgNebula` (smooth, fBm-dominated cost) and `bgSharp` (stars and
// suns, directional). Refraction wants a per-channel sample for dispersion, and
// re-running the fBm three times for a field that cannot resolve a 0.02 IOR
// shift would be pure waste.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "fibonacci.wgsl"

/// Nearest point of a spherical-Fibonacci set, for *point* queries rather than
/// the field. `nCand` walks a prefix of the centred window: 9 is exact, fewer is
/// approximate but much cheaper, and a rare miss here only nudges a cell edge —
/// it cannot crease a surface.
///
/// Returns the COSINE of the angle to the nearest point rather than the point itself. That is
/// all any caller wanted, and asking for it directly means the whole search can stay in the
/// layer's local frame: a rotation preserves angles, so the winner never has to be rotated back
/// into world space. On a starfield that runs over most of the screen, that is one 3x3 multiply
/// per candidate removed from a five-candidate loop.
struct NearSF { cosAng : f32, idx : f32 };

fn nearestSF(dir : vec3f, n : f32, rotIdx : i32, nCand : i32) -> NearSF {
  let c = sfCell(dir, n, rotIdx);
  var out : NearSF;
  out.cosAng = -1.0;
  out.idx = 0.0;
  for (var j = 0; j < nCand; j++) {
    let pol = sfPolar(c, j);
    if (!pol.ok) { continue; }
    // The polar bound is free here and rejects before the azimuth, exactly as in the field.
    if (sfDotMax(c, pol) <= out.cosAng) { continue; }
    let d = dot(c.lp, sfDirLocal(pol));
    if (d > out.cosAng) { out.cosAng = d; out.idx = pol.i; }
  }
  return out;
}

/// Crisp, evenly spread stars on their own Fibonacci set, with only ~1/4 of the
/// slots lit. 5 candidates rather than 9: this runs on every background pixel,
/// i.e. most of the screen, and a missed star is invisible when three quarters
/// of the slots are dark anyway.
fn starField(rd : vec3f) -> vec3f {
  let near = nearestSF(rd, 3800.0, 7, 5);
  let idx = near.idx;
  if (hash11(idx * 1.73 + 3.3) > 0.25) { return vec3f(0.0); }
  let b = hash11(idx * 0.31 + 7.7);
  let ang = acos(clamp(near.cosAng, -1.0, 1.0));
  let sz = mix(0.0006, 0.0020, b * b);
  let core = smoothstep(sz, sz * 0.35, ang);
  let glow = exp(-ang / (sz * 1.4)) * 0.14;
  let tint = mix(vec3f(0.72, 0.82, 1.0), vec3f(1.0, 0.90, 0.78), hash11(idx * 2.1 + 1.1));
  return tint * (core * 1.15 + glow) * (0.45 + 1.25 * b);
}

/// One sun: bright disk plus corona.
fn sunDisk(rd : vec3f, L : vec3f, warm : vec3f, core : vec3f, size : f32) -> vec3f {
  let sd = max(dot(rd, L), 0.0);
  var c = warm * pow(sd, 48.0) * 0.55;
  c += core * pow(sd, 380.0) * 1.60;
  c += core * smoothstep(1.0 - 1.6 * size, 1.0 - 0.8 * size, sd) * 9.0;
  return c;
}

fn bgNebula(rd : vec3f) -> vec3f {
  var col = vec3f(0.006, 0.008, 0.014);
  col += 0.020 * vec3f(0.35, 0.22, 0.5) * fbm3(rd * 3.0 + 11.0);
  col += 0.015 * vec3f(0.1, 0.3, 0.4) * fbm3(rd * 6.0 - 4.0);
  return col;
}

fn bgSharp(rd : vec3f) -> vec3f {
  var col = starField(rd);
  col += sunDisk(rd, SUN1_DIR, vec3f(1.00, 0.72, 0.42), vec3f(1.40, 1.24, 0.98), 0.0014);
  col += sunDisk(rd, SUN2_DIR, vec3f(0.62, 0.24, 1.00), vec3f(0.95, 0.62, 1.35), 0.0010);
  col += sunDisk(rd, SUN3_DIR, vec3f(0.30, 0.52, 0.78), vec3f(0.55, 0.78, 1.05), 0.0007);
  return col;
}

fn background(rd : vec3f) -> vec3f { return bgNebula(rd) + bgSharp(rd); }

/// Atmospheric limb glow. Camera-relative, so it is applied in the composite
/// pass rather than baked into the accumulation buffer: it cannot be reprojected
/// like a star at infinity, and accumulating it leaves soft ghost wedges
/// drifting across the sky. Keyed off ATMO_R and deliberately not off the
/// marching bound, so a geometry tweak cannot silently resize the atmosphere.
fn limbGlow(ro : vec3f, rd : vec3f) -> vec3f {
  let b2 = dot(ro, rd);
  let perp = sqrt(max(dot(ro, ro) - b2 * b2, 0.0));
  let front = step(b2, 0.0);
  let glow = smoothstep(ATMO_R, R + 0.15, perp) * front;
  return glow * glow * vec3f(0.16, 0.20, 0.36) * 0.7;
}
