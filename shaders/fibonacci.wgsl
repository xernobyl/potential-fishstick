// ---------------------------------------------------------------------------
// Spherical Fibonacci lattices, forward and inverse.
//
// Shared, and extracted for that reason: the body's distance field and the
// starfield both place points this way, but while this lived inside sdf.wgsl the
// only route to it was to include that file entire — so sky.wgsl pulled in the
// whole field, and every shader wanting a star or the sky (composite, satellites)
// compiled the entire marched body to get one. That is a dependency on
// the wrong thing: it costs compile time and register pressure in shaders that
// never march anything.
//
// The inverse mapping is Keinert/Innmann/Sanger/Stamminger 2015: given a direction
// it returns the lattice cell in O(1), so a field never iterates all N points.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"

// Per-layer rotations decorrelate the octaves. Slots:
//   0..2 sphere octaves   4..6 the matching hole layers   7 starfield
//
// LAYER_ROT and LAYER_IROT are injected TABLES. They used to be built by a pair of sin/cos
// and a 3x3 multiply inside sfCell; sfCell runs six times per field evaluation and the
// march evaluates the field ~60 times per pixel, so the GPU was recomputing the same eight
// matrices on the order of 360 times per pixel. The generator and its constants now live in
// src/scene/lattice.js, which also asserts each one really is a rotation — orthonormal,
// det +1 — since a silent slip there would shear the whole lattice into something that
// still looks plausible.
//
// The computed path is KEPT, behind PROBE.latticeTable, because "table beats maths" is not
// self-evident here: a dynamic index into 288 bytes of constant data can lower to scratch
// memory, and this hardware's transcendentals are cheap. Both paths, one measurement.
fn rotY(a : f32) -> mat3x3f {
  let c = cos(a); let s = sin(a);
  return mat3x3f(vec3f(c, 0.0, -s), vec3f(0.0, 1.0, 0.0), vec3f(s, 0.0, c));
}
fn rotX(a : f32) -> mat3x3f {
  let c = cos(a); let s = sin(a);
  return mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, c, -s), vec3f(0.0, s, c));
}
fn layerRotComputed(i : i32) -> mat3x3f {
  let f = f32(i);
  return rotY(f * LAYER_YAW_STEP + LAYER_YAW_BIAS) * rotX(f * LAYER_PITCH_STEP + LAYER_PITCH_BIAS);
}

// ---- inverse spherical Fibonacci ---------------------------------------
//
// `sfCell` solves the inverse mapping once, finding the lattice cell of `dir`.
// `sfCandidate` then walks a 3x3 window of cells around it.
//
// Why 3x3 and not the paper's 2x2: an SF point's true neighbourhood is
// hexagonal (~6 points), so a 2x2 window can miss the genuinely nearest point.
// In a *distance field* that miss is not a small error — the min jumps as the
// ray moves and creases the surface with thin seam lines.
//
// sin^2(theta) is also floored away from zero: otherwise log() blows up at each
// layer's poles, k collapses to its clamp, and the lattice there is garbage
// (which showed up as fan-shaped striations).

const SFN : i32 = 9;

/// Centre first, then the four edge neighbours, then the diagonals — so a caller
/// that only needs an approximate nearest can walk a PREFIX of this list.
///
/// Who may: the additive sphere layers must use all 9, because a missed nearest
/// point makes the min jump as the ray moves and creases the surface with seam
/// lines. A SUBTRACTED hole layer may use 5 — the same error shifts a crater rim
/// slightly instead of tearing the silhouette — and so may the starfield.
const SFWIN = array<vec2f, 9>(
  vec2f( 0.0,  0.0),
  vec2f( 1.0,  0.0), vec2f(-1.0,  0.0), vec2f( 0.0,  1.0), vec2f( 0.0, -1.0),
  vec2f( 1.0,  1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0), vec2f(-1.0, -1.0),
);

struct SFCell {
  rot  : mat3x3f,   // world -> layer local
  irot : mat3x3f,   // layer local -> world
  lp   : vec3f,     // the query direction, in layer-local space
  F    : vec2f,
  cc   : vec2f,
  m    : f32,
  n    : f32,
  // |lp.xy| — the query's equatorial magnitude, EXACTLY, not `sqrt(1 - lp.z^2)`.
  //
  // The difference is the difference between a bound that is provably safe and one that is
  // merely usually safe. `rot * dir` leaves |lp| a few ulps off unity, so 1 - lp.z^2 can come out
  // a hair BELOW the true |lp.xy|^2 — and the Cauchy-Schwarz step in `sfDotMax` then has a
  // slightly-too-small factor, which makes it a slightly-too-small upper bound, which is exactly
  // the direction that lets a candidate be wrongly skipped. A wrongly skipped candidate is a hole
  // in the field and renders as a hard seam. Taking the length outright removes the assumption,
  // and it is also TIGHTER, so it rejects marginally more.
  lxy  : f32,
};

fn sfCell(dir : vec3f, n : f32, rotIdx : i32) -> SFCell {
  var rot : mat3x3f;
  var irot : mat3x3f;
  if (frame.probe.x > 0.5) {
    rot = LAYER_ROT[rotIdx];
    irot = LAYER_IROT[rotIdx];
  } else {
    rot = layerRotComputed(rotIdx);
    irot = transpose(rot);
  }
  let lp = rot * dir;
  let ct = clamp(lp.z, -1.0, 1.0);
  let s2 = max(1.0 - ct * ct, 1e-7);        // never log(0) at the poles
  let m = 1.0 - 1.0 / n;
  let k = max(2.0, floor(log(n * PI * sqrt(5.0) * s2) / log(PHI + 1.0)));
  let Fk = pow(PHI, k) / sqrt(5.0);
  let F = vec2f(floor(Fk + 0.5), floor(Fk * PHI + 0.5));   // consecutive Fibonaccis
  let ka = 2.0 * F / n;
  let kb = TAU * (fract((F + 1.0) * PHI) - (PHI - 1.0));
  let iB = mat2x2f(vec2f(ka.y, -ka.x), vec2f(kb.y, -kb.x)) * (1.0 / (ka.y * kb.x - ka.x * kb.y));

  var c : SFCell;
  c.rot = rot;
  c.irot = irot;
  c.lp = lp;
  c.F = F;
  c.m = m;
  c.n = n;
  c.lxy = length(lp.xy);
  c.cc = floor(iB * vec2f(atan2(lp.y, lp.x), ct - m));
  return c;
}

// ---- evaluating a candidate, in two halves --------------------------------
//
// Split because the halves cost wildly different amounts and the caller can often reject
// between them. The polar half is a dot product, a multiply-add and a square root. The azimuth
// half is a `fract`, a `cos` and a `sin` — and it is the only part that needs the index's
// low bits, so it is also the part that cannot be bounded cheaply.
//
// `layerDist` used to pay both for every candidate in a 3x3 window, six layers deep, on every
// field evaluation. Most of those candidates are then thrown away against a distance bound
// that only needs the ANGLE — and the polar half alone bounds the angle. See `sfDotMax`.

struct SFPolar {
  ok  : bool,
  i   : f32,        // lattice index
  ict : f32,        // cos of the candidate's polar angle
  ist : f32,        // sin of the same
};

/// The cheap half: which lattice point, and how far up the axis it sits.
fn sfPolar(c : SFCell, j : i32) -> SFPolar {
  var o : SFPolar;
  o.i = dot(c.F, SFWIN[j] + c.cc);
  o.ict = 0.0;
  o.ist = 0.0;
  if (o.i < 0.0 || o.i > c.n - 0.5) { o.ok = false; return o; }
  o.ict = c.m - 2.0 * o.i / c.n;
  o.ist = sqrt(max(1.0 - o.ict * o.ict, 0.0));
  o.ok = true;
  return o;
}

/// The LARGEST dot(query, candidate) that any azimuth could produce.
///
/// Cauchy-Schwarz on the equatorial components. |lcw.xy| is exactly `ist` by construction, so
///
///     dot(lp, lcw) = lp.xy . lcw.xy + lp.z * ict  <=  |lp.xy| * ist + lp.z * ict
///
/// and the right-hand side is cos of the DIFFERENCE IN POLAR ANGLE — the great-circle distance
/// the two latitudes alone imply, attained when the azimuths coincide. So it is a true upper
/// bound on the dot product, i.e. a true LOWER bound on the angular distance, which is the safe
/// direction for rejecting a candidate: if the best case is already too far to matter, no azimuth
/// can rescue it. Costs two multiplies against a `fract`, a `cos`, a `sin` and a 3x3 multiply.
///
/// Note what is NOT assumed: `lp` need not be exactly unit, and `lp.z` is used raw rather than
/// clamped. Both matter — the inequality above is an identity plus one Cauchy-Schwarz step, and
/// substituting an approximation of |lp.xy| or a clamped lp.z would break the "upper" in upper
/// bound by a few ulps, which is all it takes to drop a sphere and seam the surface.
fn sfDotMax(c : SFCell, pol : SFPolar) -> f32 {
  return c.lp.z * pol.ict + c.lxy * pol.ist;
}

/// The expensive half, in LAYER-LOCAL space. Deliberately not rotated back to world: both hot
/// callers measure everything in the layer's own frame, where a rotation preserves every length
/// and angle they ask for, so rotating the CANDIDATE was a 3x3 multiply apiece that rotating
/// the query once replaces.
fn sfDirLocal(pol : SFPolar) -> vec3f {
  let ip = TAU * fract(pol.i * PHI);
  return vec3f(cos(ip) * pol.ist, sin(ip) * pol.ist, pol.ict);
}

struct SFHit { ok : bool, cw : vec3f, idx : f32 };

/// World-space convenience, for callers that genuinely need a world direction — the AO integral
/// does, because it works with world positions and normals it did not rotate.
fn sfCandidate(c : SFCell, j : i32) -> SFHit {
  let pol = sfPolar(c, j);
  var out : SFHit;
  out.idx = pol.i;
  out.ok = pol.ok;
  out.cw = select(vec3f(0.0, 0.0, 1.0), c.irot * sfDirLocal(pol), pol.ok);
  return out;
}

/// FORWARD mapping: index -> direction. The inverse is only needed to ask
/// "which point is nearest this direction"; to merely iterate the set (embers)
/// this is all it takes.
fn sfPointFwd(i : f32, n : f32) -> vec3f {
  let m = 1.0 - 1.0 / n;
  let ph = TAU * fract(i * PHI);
  let ct = m - 2.0 * i / n;
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  return vec3f(cos(ph) * st, sin(ph) * st, ct);
}
