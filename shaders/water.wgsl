// ---------------------------------------------------------------------------
// The water planet: an ocean world, raymarched.
//
// A smooth water SPHERE displaced by a low-frequency 3D noise (the ocean
// swell), detailed with THREE Fibonacci spirals of tiny offset spheres that
// melt together. The spheres are small and densely placed, and every union is
// a smooth blend — never a hard min — so nothing reads as a coarse "big
// octave" or a separate ball.
//
// Motion: a few octaves of value noise, translated slowly in time, give the
// surface a gentle 3D wave — the classic flowing-water look rather than
// per-ball sine lumps.
//
// Shading is realistic PBR water: GGX sun glints, fresnel-weighted sky
// reflection, an absorbed deep-blue refracted body, analytic sphere AO with
// matching local thickness, and a wrapped-diffuse subsurface scattering term.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "fibonacci.wgsl"
//!include "brdf.wgsl"
//!include "sky.wgsl"

// ---- water field -----------------------------------------------------------

const WR       : f32 = 2.0;     // mean radius
const WN0      : f32 = 260.0;   // spheres per spiral
const WRHO0    : f32 = 0.04;    // tiny sphere radius
const W_SPIRALS: i32  = 3;      // three offset Fibonacci spirals

// 3D noise displacement. Amplitude, frequency and speed are live-tunable via
// the frame uniform (`frame.water`), read fresh each frame so the panel drives
// them.

/// The per-sphere radius and radial centre offset, from its lattice index and
/// spiral. Shared by the field and the AO, so the two can never disagree about
/// where a sphere is. Returns (rr, jr).
fn sphereParams(i : f32, rotIdx : i32) -> vec2f {
  let rr = WRHO0 * (0.6 + 0.8 * hash11(i * 1.7 + f32(rotIdx) * 13.3));
  let jr = (hash11(i + f32(rotIdx) * 57.1) - 0.5) * WRHO0 * 0.6;
  return vec2f(rr, jr);
}

/// A directional travelling wave: value noise that ADVECTS along `dir` with a
/// per-spiral phase offset, so the three spirals' wave fields move in different
/// directions and interfere into a living surface.
fn waveNoise(p : vec3f, t : f32, spiral : i32) -> f32 {
  // Three spirals, 120 apart: each advects its own copy of the noise along a
  // different rotational direction, so the layers slide past each other.
  var s = 0.0;
  var amp = 0.6;
  var f = 1.0;
  // The advection direction for this spiral, rotated by its 120 slot.
  let ang = f32(spiral) * (TAU / 3.0);
  let along = vec3f(cos(ang), 0.25, sin(ang));
  for (var i = 0; i < 3; i++) {
    let flow = p * frame.water.y - along * (t * frame.water.z) + f32(i) * 11.3;
    s += amp * vnoise(flow * f);
    f *= 2.2;
    amp *= 0.5;
  }
  return s;
}

/// The water surface: a sphere displaced by three travelling waves (one per
/// spiral, 120 apart), plus three spirals of tiny spheres that ride them.
fn waterMap(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);
  let r = length(p);
  let t = frame.camPos.w;

  // HOIST the three travelling waves. Each `waveNoise` is 3 vnoise octaves
  // (24 hash13s); evaluating it per sphere candidate was the dominant cost of
  // the whole march. It depends only on the RAY DIRECTION, so three calls once
  // here replace 27 calls inside the loops below.
  var swell = 0.0;
  var perSpiralSwell = array<f32, 3>();
  for (var k = 0; k < W_SPIRALS; k++) {
    let w = frame.water.x * (waveNoise(dir, t, k) - 0.5) * 2.0;
    perSpiralSwell[k] = w;
    swell += w;
  }
  swell *= 0.45;   // the three interfere; scale so the total stays bounded
  var d = r - WR - swell;

  // Three offset spirals of tiny spheres riding the hoisted swell. No noise is
  // evaluated in here — each sphere's centre rides `dir`'s wave, which is within
  // a cell of the sphere's own direction, so the reuse is visually exact.
  for (var s = 0; s < W_SPIRALS; s++) {
    let rotIdx = 4 + s;             // distinct lattice rotation per spiral
    let c = sfCell(dir, WN0, rotIdx);
    let lp = c.lp;
    let lpp = c.rot * p;
    for (var j = 0; j < 9; j++) {
      let pol = sfPolar(c, j);
      if (!pol.ok) { continue; }
      let lcw = sfDirLocal(pol);
      let i = pol.i;

      let pr = sphereParams(i, rotIdx);
      let rr = pr.x;
      let surf = WR + perSpiralSwell[s];
      let sc = surf + pr.y;

      let sphere = length(lpp - lcw * sc) - rr;
      d = smin(d, sphere, WRHO0 * 1.8);
    }
  }

  return d;
}

fn waterBound() -> f32 {
  return WR + 3.0 * frame.water.x + 0.25;
}

fn waterNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0008;
  return normalize(
      e.xyy * waterMap(p + e.xyy) + e.yyx * waterMap(p + e.yyx)
    + e.yxy * waterMap(p + e.yxy) + e.xxx * waterMap(p + e.xxx));
}

/// Analytic sphere visibility + local thickness from ONE pass over the
/// Fibonacci neighbours (Barre-Brisebois & Bouchard, GDC 2011, and the same
/// construction the gummy uses). `.x` is ambient visibility about +nor (AO);
/// `.y` is 1 - visibility about -nor, i.e. LOCAL THICKNESS — the distance
/// light must cross, which then drives Beer-Lambert extinction in the SSS.
fn waterAOThickness(pos : vec3f, nor : vec3f) -> vec2f {
  var vis = 1.0;
  var back = 1.0;
  let dir = normalize(pos);
  for (var s = 0; s < 2; s++) {
    let rotIdx = 4 + s;
    let c = sfCell(dir, WN0, rotIdx);
    for (var j = 0; j < 9; j++) {
      let cand = sfCandidate(c, j);
      if (!cand.ok) { continue; }
      let cw = cand.cw;
      let i = cand.idx;
      let pr = sphereParams(i, rotIdx);
      let rr = pr.x;
      let sph = vec4f(cw * (WR + pr.y), rr);
      let di = sph.xyz - pos;
      let l = length(di);
      let nl = dot(nor, di / max(l, 1e-4));
      let h = l / rr;
      let h2 = h * h;
      if (h2 <= 1.0) { continue; }
      let k2 = 1.0 - h2 * nl * nl;
      var res = max(0.0, nl) / h2;
      if (k2 > 0.001) {
        res = nl * acos(-nl * sqrt((h2 - 1.0) / (1.0 - nl * nl))) - sqrt(k2 * (h2 - 1.0));
        res = (res / h2 + atan2(sqrt(k2 / (h2 - 1.0)), 1.0)) / PI;
      }
      let fade = 1.0 - smoothstep(rr * rr * 9.0, rr * rr * 25.0, dot(di, di));
      if (fade > 0.0) {
        vis *= 1.0 - clamp(res * fade, 0.0, 1.0);
        // The SAME sphere integral about -nor — the local thickness. This is
        // the other half of the hemisphere, evaluated from the same `di`.
        var backRes = max(0.0, -nl) / h2;
        if (k2 > 0.001) {
          backRes = (-nl) * acos(nl * sqrt((h2 - 1.0) / (1.0 - nl * nl))) - sqrt(k2 * (h2 - 1.0));
          backRes = (backRes / h2 + atan2(sqrt(k2 / (h2 - 1.0)), 1.0)) / PI;
        }
        back *= 1.0 - clamp(backRes * fade, 0.0, 1.0);
      }
    }
  }
  return vec2f(vis, 1.0 - back);
}

// ---- water shading ---------------------------------------------------------

const WATER_F0    : vec3f = vec3f(0.02);   // water IOR ~1.33

/// A fine ripple added to the field normal, so the specular glints have a
/// liquid micro-scale rather than a glass-smooth mirror.
fn waterRipple(p : vec3f, t : f32) -> vec3f {
  let s = 0.02;
  let e = vec2f(1.0, -1.0) * s;
  let off = vec3f(0.0, t * 0.3, 0.0);
  let h = vnoise(p * 14.0 + off);
  return vec3f(
    vnoise((p + e.xyy) * 14.0 + off) - h,
    vnoise((p + e.yyx) * 14.0 + off) - h,
    vnoise((p + e.yxy) * 14.0 + off) - h,
  ) / s;
}

/// Realistic PBR water: GGX specular sun glints, fresnel-weighted sky
/// reflection, and a deep-blue absorbed sky seen through the refracted view.
fn shadeWater(p : vec3f, rd : vec3f) -> vec3f {
  let baseN = waterNormal(p);
  let N = normalize(baseN + normalize(waterRipple(p, frame.camPos.w)) * 0.35);
  let V = -rd;

  let f0 = WATER_F0;
  let rough = frame.water.w;

  // GGX specular from the three suns — the identity of water. A broad lobe for
  // the body reflection…
  let spec =
      sunLight(N, V, SUN1_DIR, SUN1_COL, vec3f(0.02), rough, f0, 1.0 - 1e-3)
    + sunLight(N, V, SUN2_DIR, SUN2_COL, vec3f(0.02), rough, f0, 1.0 - 1e-3)
    + sunLight(N, V, SUN3_DIR, SUN3_COL, vec3f(0.02), rough, f0, 0.6);

  // …plus a TIGHT second lobe: the micro-facet sparkle that makes water read as
  // water rather than wet glass. Much lower roughness, low weight, so it sits as
  // a bright glint riding the broad reflection where the surface is nearest.
  let sparkleRough = frame.water3.x;
  let sparkleW = frame.water3.y;
  let sparkle = sparkleW * (
      sunLight(N, V, SUN1_DIR, SUN1_COL, vec3f(0.02), sparkleRough, f0, 1.0 - 1e-3)
    + sunLight(N, V, SUN2_DIR, SUN2_COL, vec3f(0.02), sparkleRough, f0, 1.0 - 1e-3)
    + sunLight(N, V, SUN3_DIR, SUN3_COL, vec3f(0.02), sparkleRough, f0, 0.6));

  // Fresnel: how much of the view reflects (sky) vs refracts (water body).
  let F = fresnelSchlick(clamp(dot(N, V), 0.0, 1.0), f0);

  // Analytic AO (visibility about +N) and LOCAL THICKNESS (about -N) from one
  // pass over the neighbours.
  let aot = waterAOThickness(p, baseN);
  let ao = aot.x;
  let curveThick = clamp(aot.y * 1.5, 0.02, 1.0);

  // Sky reflection, tinted a little toward the water.
  let reflDir = reflect(rd, N);
  let refl = bgNebula(reflDir) + bgSharp(reflDir);

  // Refracted sky: deep blue, absorbed by the water column, darkened in the
  // crevices by the AO.
  let deepBlue = vec3f(0.01, 0.10, 0.24);
  let refr = (deepBlue + (bgNebula(N) + bgSharp(N)) * 0.15) * ao;

  // ---- subsurface scattering: sun light seen THROUGH the water ----
  //
  // Colin Barré-Brisebois' wrapped-diffuse "local thickness" translucency
  // (GDC 2011). The light vector is warped toward the normal, then the view dot
  // the negated warped light gives the translucent term. The thickness driving
  // Beer-Lambert absorption combines the sphere-integral local thickness with
  // the radius chord — creases are thin AND the far limb is thin, both glow.
  let chordThick = max(waterBound() - length(p), 0.0);
  let thick = mix(curveThick, chordThick, 0.5);
  let absorb = exp(-thick * vec3f(0.7, 0.3, 0.12));   // water absorbs red first
  let warp = vec3f(frame.water2.z);    // mDistortion: how far light wraps in
  let power = frame.water2.x;          // mLTPower: wrap sharpness
  let ssize = frame.water2.y;          // lSize: scale
  var sss = vec3f(0.0);
  sss += SUN1_COL * pow(clamp(dot(V, -normalize(SUN1_DIR + N * warp)), 0.0, 1.0), power) * ssize * absorb;
  sss += SUN2_COL * pow(clamp(dot(V, -normalize(SUN2_DIR + N * warp)), 0.0, 1.0), power) * ssize * absorb;
  sss += SUN3_COL * pow(clamp(dot(V, -normalize(SUN3_DIR + N * warp)), 0.0, 1.0), power) * ssize * absorb;

  var col = spec;
  col += sparkle;
  col += refl * F;
  col += refr * (vec3f(1.0) - F) * 0.9;
  // SSS passes through the thin limbs, tinted toward the (blue-shifted) water.
  col += sss * vec3f(0.35, 0.75, 1.0) * frame.water2.w;

  // Fresnel rim brightening at the silhouette.
  let fre = pow(clamp(1.0 + dot(rd, N), 0.0, 1.0), 4.0);
  col += fre * vec3f(0.25, 0.38, 0.52) * 0.6;

  return col;
}

// ---- passes ----------------------------------------------------------------

@group(1) @binding(0) var<storage, read_write> tileFlags : array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cull_main(@builtin(global_invocation_id) gid : vec3u) {
  let tiles = vec2u((vec2u(frame.res.xy) + u32(TILE) - 1u) / u32(TILE));
  if (gid.x >= tiles.x || gid.y >= tiles.y) { return; }
  let r = waterBound() * 1.05;
  var visible = false;
  for (var j = 0u; j < 5u; j++) {
    var off = vec2f(0.0);
    if (j == 0u) { off = vec2f(0.0, 0.0); }
    else if (j == 1u) { off = vec2f(f32(TILE), 0.0); }
    else if (j == 2u) { off = vec2f(0.0, f32(TILE)); }
    else if (j == 3u) { off = vec2f(f32(TILE), f32(TILE)); }
    else { off = vec2f(f32(TILE) * 0.5, f32(TILE) * 0.5); }
    let ray = cameraRay(vec2f(gid.xy) * f32(TILE) + off);
    if (iSphere(ray.o, ray.d, r).y > 0.0) { visible = true; break; }
  }
  tileFlags[gid.y * tiles.x + gid.x] = select(0u, 1u, visible);
}

@group(2) @binding(0) var outTex    : texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var motionTex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn march_main(@builtin(global_invocation_id) gid : vec3u,
              @builtin(workgroup_id) wid : vec3u) {
  let size = vec2u(frame.res.xy);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let tiles = (size + u32(TILE) - 1u) / u32(TILE);
  let mayHit = tileFlags[wid.y * tiles.x + wid.x] != 0u;

  let px = vec2f(gid.xy) + 0.5 + frame.jitter.xy;
  let ray = cameraRay(px);

  var col = background(ray.d);
  var depthTag = TAG_BG;

  if (mayHit) {
    let bs = iSphere(ray.o, ray.d, waterBound());
    if (bs.y > 0.0) {
      var t = max(bs.x, 0.0);
      let tmax = bs.y;
      var hit = false;
      for (var i = 0; i < 96; i++) {
        let d = waterMap(ray.o + ray.d * t);
        if (d < 0.002 * max(t, 0.1)) { hit = true; break; }
        t += d * 0.85;
        if (t > tmax) { break; }
      }
      if (hit) {
        let p = ray.o + ray.d * t;
        col = shadeWater(p, ray.d);
        depthTag = t;
      }
    }
  }

  textureStore(motionTex, vec2i(gid.xy), vec4f(MOTION_NONE, 0.0, 0.0, 0.0));
  textureStore(outTex, vec2i(gid.xy), vec4f(col, depthTag));
}
