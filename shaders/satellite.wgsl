// ---------------------------------------------------------------------------
// Satellites: the orbital mechanics and the surface materials.
//
// This file no longer intersects anything. The three boxes each satellite is made of are now REAL
// GEOMETRY - one unit cube, instanced fifteen times and scaled per instance - rasterised into the solid
// layer by satmesh.wgsl. What stays here is the part that is genuinely about satellites rather than
// about drawing: where each one is at a given time, and what its two materials look like.
//
// WHY THE SPLIT AND NOT ONE FILE. `satFrameAt` is needed by the vertex stage (to place the boxes now and
// one frame ago) and the materials by the fragment stage, so they would be separated anyway; keeping them
// together here means the ONE definition of a satellite's orbit is the one both stages read, which is
// what makes the motion vector exact instead of merely consistent.
//
// The greeble is procedural SURFACE detail, not more geometry, and that was true when the boxes were
// analytic and stays true now: detail that only perturbs the normal and the material costs a few hashes
// on the pixels that hit a satellite, where the same detail as real triangles would multiply them. At the
// size these sit on screen, nothing is lost.
//
// Orientation is how a real bird flies: the bus is nadir-pointing (one face kept toward the planet) and
// the array rotates about its boom to track the sun. That is why the panels catch the light together and
// swing as the camera drifts, which is most of what makes them read as satellites rather than as debris.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "reflect.wgsl"

const SAT_MAT_BUS   : i32 = 0;
const SAT_MAT_PANEL : i32 = 1;

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

/// One drawable box of one satellite: an oriented frame, a half-extent and a material.
///
/// Three parts per satellite - the bus and the two array wings - indexed the way an instanced draw
/// indexes them, so `satPart(i / 3, i % 3, t)` turns a flat instance number into a placement. Taking the
/// time as a parameter is what lets the vertex stage ask the same question about the PREVIOUS frame and
/// get an exact answer rather than an estimate.
struct SatPart {
  centre : vec3f,
  ax     : vec3f,   // the box's own x, y, z in world space
  ay     : vec3f,
  az     : vec3f,
  rad    : vec3f,   // half-extents along those axes
  mat    : i32,
};

fn satPart(sat : f32, part : u32, t : f32) -> SatPart {
  let f = satFrameAt(sat, t);
  var o : SatPart;
  if (part == 0u) {
    // The bus, on the orbital frame itself.
    o.centre = f.pos;
    o.ax = f.bx; o.ay = f.by; o.az = f.bz;
    o.rad = vec3f(SAT_BUS);
    o.mat = SAT_MAT_BUS;
  } else {
    // An array wing, out along the boom either side. Panel space is x thin (the face normal), y along
    // the boom, z across the width - and it rides the sun-tracking basis, not the bus one, which is the
    // rotation that has to be carried into the previous frame as well or the panels shear in the
    // history.
    let boom = select(-(SAT_BOOM + SAT_PANEL_LEN), SAT_BOOM + SAT_PANEL_LEN, part == 1u);
    o.centre = f.pos + f.by * boom;
    o.ax = f.pn; o.ay = f.by; o.az = f.pw;
    o.rad = vec3f(SAT_PANEL_THICK, SAT_PANEL_LEN, SAT_PANEL_WIDE);
    o.mat = SAT_MAT_PANEL;
  }
  return o;
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

/// The two satellite materials, given a point on a box.
///
/// Takes the surface rather than a hit record, so it is the same function whether the box was
/// intersected or rasterised. `local` is the point in the BOX's own space in real units - which is what
/// every feature below is authored against, and why the greeble does not swim when a satellite rotates.
fn shadeSatSurface(local : vec3f, nor : vec3f, mat : i32, seed : f32,
                   hitP : vec3f, V : vec3f) -> vec3f {
  var N = nor;
  var alb : vec3f;
  var f0 : vec3f;
  var rough : f32;
  var emis = vec3f(0.0);

  if (mat == SAT_MAT_PANEL) {
    // Solar cells: near-black silicon under a smooth cover glass, so almost
    // everything you see is the specular sheet. That is why real arrays read as
    // dark blue slabs face-on and as mirrors at a glance.
    let g = local.yz * SAT_CELL_SCALE;
    let cell = floor(g);
    let ch = hash21(cell + seed * 17.0);

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
    let q = local * SAT_WRINKLE_SCALE;
    let wob = vec3f(hash13(q + 1.7), hash13(q + 5.3), hash13(q + 9.1)) - 0.5;
    let tang = wob - nor * dot(wob, nor);      // into the tangent plane
    N = normalize(nor + tang * SAT_WRINKLE);

    // Greeble. Hashing floor(local) gives per-face cells for free: the coordinate
    // along a face's own axis is constant across it, so it drops out of the cell
    // index without any face-selection logic.
    let g = local * SAT_GREEBLE_SCALE;
    let gh = hash13(floor(g) + seed * 31.0);

    // Seams between panels. Weighting by (1 - |n|) excludes the face's own axis,
    // whose fract() is constant and would otherwise seam a whole face at once.
    let w = 1.0 - abs(nor);
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
