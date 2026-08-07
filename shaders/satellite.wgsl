// ---------------------------------------------------------------------------
// Satellites: three boxes each, analytically intersected.
//
//   1 central cube  - the bus, wrapped in metallised film
//   2 flat cubes    - the solar array, one either side on a boom
//
// Analytic boxes rather than SDF boxes folded into the marched field, for the
// same reason the moon and debris are analytic: they are rigid bodies on their
// own orbits, the field is already the frame's whole cost, and an exact slab test
// gives a perfect edge and an exact normal for a fraction of one march step.
//
// The greeble is procedural SURFACE detail, not more geometry. Detail that only
// perturbs the normal and the material costs a few hashes on the pixels that hit
// a satellite, where the same detail as real boxes would multiply the ray tests.
// At the size these sit on screen, nothing is lost.
//
// Orientation is how a real bird flies: the bus is nadir-pointing (one face kept
// toward the planet) and the array rotates about its boom to track the sun. That
// is why the panels catch the light together and swing as the camera drifts,
// which is most of what makes them read as satellites rather than as debris.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "reflect.wgsl"

const SAT_MAT_BUS   : i32 = 0;
const SAT_MAT_PANEL : i32 = 1;

struct SatHit {
  hit   : bool,
  t     : f32,
  nor   : vec3f,   // world space
  local : vec3f,   // box space, for the greeble
  mat   : i32,
  seed  : f32,
  // Signed offset along the boom of the box that was hit: 0 for the bus, +-(boom + panel) for the
  // two arrays. Stored rather than re-derived because `local` is in the hit box's OWN space and so
  // cannot say which side of the bus that box was on - and the previous-frame transform needs it.
  boom  : f32,
};

/// Slab test against a box of half-extents `rad`, centred at the origin of the
/// space `ro`/`rd` are given in. Returns (tNear, normal); tNear < 0 is a miss.
/// The normal falls out of which slab was entered last, so it is exact.
fn boxIntersect(ro : vec3f, rd : vec3f, rad : vec3f) -> vec4f {
  // A component of exactly zero would make 1/rd infinite and 0*inf a NaN, and a
  // NaN here would survive into the accumulation buffer as a stuck pixel.
  let safe = select(rd, vec3f(1e-9), abs(rd) < vec3f(1e-9));
  let m = 1.0 / safe;
  let n = m * ro;
  let k = abs(m) * rad;
  let t1 = -n - k;
  let t2 = -n + k;
  let tN = max(max(t1.x, t1.y), t1.z);
  let tF = min(min(t2.x, t2.y), t2.z);
  if (tN > tF || tF < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
  let nor = -sign(safe) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  return vec4f(tN, nor);
}

/// One satellite's rigid frame: an orthonormal basis plus the array's own axis.
struct SatFrame {
  pos  : vec3f,
  bx   : vec3f,   // along track
  by   : vec3f,   // boom: the array extends along this
  bz   : vec3f,   // zenith, away from the planet
  pn   : vec3f,   // array face normal, sun-tracking about `by`
  pw   : vec3f,   // array width axis
};

/// One satellite's frame AT A GIVEN TIME. Parameterised on `t` rather than reading the clock,
/// because the exact motion vector needs this evaluated at the previous frame's time too - the same
/// reason `ringDefAt` takes a time. `satFrame` is the now-shaped wrapper.
fn satFrameAt(i : f32, t : f32) -> SatFrame {
  // Each on its own orbit: radius, inclination and phase all decorrelated, so
  // they never line up into a visible ring.
  let s = hash11(i * 4.31 + 0.7);
  let s2 = hash11(i * 9.17 + 3.1);
  let rad = SAT_ORB * (0.88 + 0.30 * s);
  let ang = t * SAT_RATE * (0.7 + 0.6 * s2) + i * 2.39996323;
  let inc = (s - 0.5) * 1.5;

  // Orbit in a plane tilted by `inc` about the along-track axis.
  let ci = cos(inc);
  let si = sin(inc);
  let p = vec3f(cos(ang) * rad, sin(ang) * rad * si, sin(ang) * rad * ci);

  var f : SatFrame;
  f.pos = p;
  f.bz = normalize(p);
  // Velocity direction: the derivative of the orbit, which keeps the bus
  // correctly oriented along track without any extra state.
  let vel = vec3f(-sin(ang), cos(ang) * si, cos(ang) * ci);
  f.bx = normalize(vel - f.bz * dot(vel, f.bz));
  f.by = cross(f.bz, f.bx);

  // The array tracks the sun by rotating about the boom, so its normal is the
  // sun direction with the boom component removed. Degenerate only when the sun
  // is straight down the boom, where any orientation is equally edge-on.
  let sp = SUN1_DIR - f.by * dot(SUN1_DIR, f.by);
  let spl = length(sp);
  f.pn = select(f.bz, sp / max(spl, 1e-5), spl > 1e-3);
  f.pw = normalize(cross(f.pn, f.by));
  return f;
}

fn satFrame(i : f32) -> SatFrame { return satFrameAt(i, frame.camPos.w); }

/// Where a satellite hit was ONE FRAME AGO, exactly. The orbits are analytic, so the previous
/// transform is the same evaluation at the previous time: no stored velocities, no extra buffer,
/// no estimate. This is what lets satellites carry a real motion vector instead of falling back to
/// same-pixel screen-space history, which is what made them crawl while the rings sat still.
fn satPrevWorld(h : SatHit, tPrev : f32) -> vec3f {
  let f  = satFrameAt(h.seed, frame.camPos.w);
  let pf = satFrameAt(h.seed, tPrev);
  if (h.mat == SAT_MAT_PANEL) {
    // The arrays rotate about the boom to track the sun, so their basis has to be mapped through
    // as well. Using the bus basis for them would leave that rotation uncompensated, which is a
    // slow error but exactly the kind that never averages out.
    let cNow  = f.pos  + f.by  * h.boom;
    let cPrev = pf.pos + pf.by * h.boom;
    return cPrev + pf.pn * h.local.x + pf.by * h.local.y + pf.pw * h.local.z;
  }
  return pf.pos + pf.bx * h.local.x + pf.by * h.local.y + pf.bz * h.local.z;
}

/// Test one box given its basis as rows, narrowing `out` when it is closer.
fn satBox(out : ptr<function, SatHit>, ro : vec3f, rd : vec3f,
          centre : vec3f, ax : vec3f, ay : vec3f, az : vec3f,
          rad : vec3f, mat : i32, seed : f32, boom : f32) {
  // Into box space. The basis is orthonormal, so the inverse is the transpose,
  // i.e. three dot products.
  let rel = ro - centre;
  let lo = vec3f(dot(rel, ax), dot(rel, ay), dot(rel, az));
  let ld = vec3f(dot(rd, ax), dot(rd, ay), dot(rd, az));
  let h = boxIntersect(lo, ld, rad);
  if (h.x < 0.0 || h.x >= (*out).t) { return; }
  (*out).t = h.x;
  (*out).nor = ax * h.y + ay * h.z + az * h.w;   // basis * localNormal
  (*out).local = lo + ld * h.x;
  (*out).mat = mat;
  (*out).seed = seed;
  (*out).boom = boom;
  (*out).hit = true;
}

fn hitSatellites(ro : vec3f, rd : vec3f, tmax : f32) -> SatHit {
  var out : SatHit;
  out.hit = false;
  out.t = tmax;
  out.nor = vec3f(0.0, 1.0, 0.0);
  out.local = vec3f(0.0);
  out.mat = SAT_MAT_BUS;
  out.seed = 0.0;
  out.boom = 0.0;

  for (var i = 0; i < SAT_COUNT; i++) {
    let fi = f32(i);
    let f = satFrame(fi);

    // Cheap reject: one sphere around the whole assembly. The array makes this
    // much wider than the bus, so it is worth the test before three slab tests.
    let reach = SAT_BOOM + SAT_PANEL_LEN;
    let bs = iSphere(ro - f.pos, rd, reach * 1.45);
    if (bs.y < 0.0 || bs.x > out.t) { continue; }

    satBox(&out, ro, rd, f.pos, f.bx, f.by, f.bz,
           vec3f(SAT_BUS), SAT_MAT_BUS, fi, 0.0);

    let panelRad = vec3f(SAT_PANEL_THICK, SAT_PANEL_LEN, SAT_PANEL_WIDE);
    let boom = SAT_BOOM + SAT_PANEL_LEN;
    let off = f.by * boom;
    // Panel space: x is the thin axis (its face normal), y the boom, z the width.
    satBox(&out, ro, rd, f.pos + off, f.pn, f.by, f.pw,
           panelRad, SAT_MAT_PANEL, fi, boom);
    satBox(&out, ro, rd, f.pos - off, f.pn, f.by, f.pw,
           panelRad, SAT_MAT_PANEL, fi, -boom);
  }
  return out;
}

// ---- shading -----------------------------------------------------------

/// Both materials are specular-dominated, so both want the environment, not just
/// the two suns. `background` already contains the nebula, the stars and the sun
/// discs, which is exactly the right thing to mirror.
fn satEnv(p : vec3f, N : vec3f, V : vec3f, f0 : vec3f, rough : f32) -> vec3f {
  let R = reflect(-V, N);
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  // Marched, so the planet actually appears in the solar panels — which for a
  // mirror-flat array is most of what there is to see.
  return traceReflection(p + N * 0.004, R) * fresnelSchlickRough(NoV, f0, rough);
}

fn shadeSatellite(h : SatHit, ro : vec3f, rd : vec3f) -> vec3f {
  let hitP = ro + rd * h.t;
  let V = -rd;
  var N = h.nor;
  var alb : vec3f;
  var f0 : vec3f;
  var rough : f32;
  var emis = vec3f(0.0);

  if (h.mat == SAT_MAT_PANEL) {
    // Solar cells: near-black silicon under a smooth cover glass, so almost
    // everything you see is the specular sheet. That is why real arrays read as
    // dark blue slabs face-on and as mirrors at a glance.
    let g = h.local.yz * SAT_CELL_SCALE;
    let cell = floor(g);
    let ch = hash21(cell + h.seed * 17.0);

    // Cell grid: the gaps between cells and the busbars crossing them.
    let e = abs(fract(g) - 0.5);
    let gap = smoothstep(0.36, 0.50, max(e.x, e.y));
    // Every fourth line is a thicker interconnect run.
    let bus = smoothstep(0.44, 0.50, abs(fract(g.x * 0.25) - 0.5));

    alb = SAT_PANEL_COL * (0.82 + 0.36 * ch);
    // The gaps show the substrate, the busbars show tinned metal.
    alb = mix(alb, SAT_PANEL_COL * 0.25, gap);
    f0 = mix(vec3f(0.05), vec3f(0.78, 0.79, 0.80), bus * 0.6);
    rough = mix(0.055 + 0.03 * ch, 0.28, max(gap, bus * 0.5));

    // The cover glass stays geometrically flat — that mirror-flatness is the
    // whole point of it, and perturbing it would cost the array its sheen.
    // A faint blue cast at grazing angles, from the anti-reflective coating.
    let NoV = clamp(dot(N, V), 0.0, 1.0);
    emis += SAT_PANEL_SHEEN * pow(1.0 - NoV, 4.0);
  } else {
    // The bus: multi-layer insulation. Metallised film is never flat — it is
    // crinkled, and the crinkle is most of what identifies it — so the normal is
    // perturbed by a tangent-space wobble, and the roughness rides along with it
    // because a crease scatters more than a flat span.
    let q = h.local * SAT_WRINKLE_SCALE;
    let wob = vec3f(hash13(q + 1.7), hash13(q + 5.3), hash13(q + 9.1)) - 0.5;
    let tang = wob - h.nor * dot(wob, h.nor);      // into the tangent plane
    N = normalize(h.nor + tang * SAT_WRINKLE);

    // Greeble. Hashing floor(local) gives per-face cells for free: the coordinate
    // along a face's own axis is constant across it, so it drops out of the cell
    // index without any face-selection logic.
    let g = h.local * SAT_GREEBLE_SCALE;
    let gh = hash13(floor(g) + h.seed * 31.0);

    // Seams between panels. Weighting by (1 - |n|) excludes the face's own axis,
    // whose fract() is constant and would otherwise seam a whole face at once.
    let w = 1.0 - abs(h.nor);
    let e = abs(fract(g) - 0.5) * w;
    let seam = smoothstep(0.40, 0.50, max(max(e.x, e.y), e.z));

    // Two real materials, picked per cell: gold-tinted kapton over most of it,
    // bare aluminised film on the rest.
    let gold = step(0.42, gh);
    f0 = mix(SAT_FILM_AL, SAT_FILM_AU, gold);
    alb = vec3f(0.0);                     // conductor: no diffuse lobe
    rough = 0.14 + 0.34 * gh + 0.22 * seam;

    // A few cells read as shadowed recesses - vents, thruster ports, harness.
    let recess = step(0.93, gh);
    f0 *= 1.0 - 0.75 * recess;
    rough = mix(rough, 0.62, recess);
    f0 *= 1.0 - 0.45 * seam;
  }

  var col = emis;
  // No shadow term: these are metres across and thousands away from the body, so
  // a shadow ray would be spent to change a handful of pixels.
  col += sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, 1.0);
  col += sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, 1.0);
  col += sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0);
  col += satEnv(hitP, N, V, f0, rough);
  // A little ambient so the unlit side is not pure black, as the planet's own
  // glow would bounce onto it.
  col += alb * SAT_AMBIENT + f0 * SAT_AMBIENT * 0.35;
  return col;
}
