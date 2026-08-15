// ---------------------------------------------------------------------------
// Chibi grass planet: a small round world covered in grass — rolling hills and
// a fine blade field. Adapted from David Hoskins' "Rolling hills"
// (Shadertoy Xsf3zX).
//
// Geometry: a sphere displaced by low-frequency fbm (the hills) and a finer,
// smooth value-noise blade field, so the surface has real volume that shades
// like turf. The field is continuous and uniform over the sphere — no Voronoi
// cell grid, so no visible facets or gaps.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"

const CR       : f32 = 1.0;     // planet radius
const HILL_AMP : f32 = 0.08;    // rolling-hill amplitude
const BLADE_H  : f32 = 0.018;   // grass blade height
const BLADE_F  : f32 = 55.0;    // blade field frequency

/// Rolling hills: a low-frequency fbm displacement.
fn grassHills(dir : vec3f) -> f32 {
  return (fbm3(dir * 2.6) - 0.5) * HILL_AMP;
}

/// Fine, smooth blade field (two octaves of value noise). Smooth rather than a
/// Voronoi cell grid: continuous and dense, so no visible facets and no gaps.
fn grassBlades(dir : vec3f) -> f32 {
  let n = 0.62 * vnoise(dir * BLADE_F) + 0.38 * vnoise(dir * BLADE_F * 2.3);
  return n * BLADE_H;
}

fn mapBody(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);
  let r = length(p);
  return r - CR - grassHills(dir) - grassBlades(dir);
}

fn bodyBound() -> f32 {
  return CR + HILL_AMP + BLADE_H + 0.1;
}

fn calcNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0012;
  return normalize(
      e.xyy * mapBody(p + e.xyy) + e.yyx * mapBody(p + e.yyx)
    + e.yxy * mapBody(p + e.yxy) + e.xxx * mapBody(p + e.xxx));
}

// ---- shading --------------------------------------------------------------

/// Grass colour: hill-shaded green with a subtle blade texture.
fn grassColour(dir : vec3f, blade : f32) -> vec3f {
  let hills = fbm3(dir * 4.0);
  var col = mix(vec3f(0.06, 0.26, 0.09), vec3f(0.18, 0.47, 0.17), hills);
  col = mix(col, vec3f(0.34, 0.52, 0.24), blade * 0.55);
  return col;
}

/// Cheap grass self-shadowing: darken where the blade field is dense.
fn grassAO(blade : f32) -> f32 {
  return 1.0 - 0.30 * blade;
}

fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let n = calcNormal(p);
  let v = -rd;
  let dir = normalize(p + 1e-6);
  let blade = grassBlades(dir) / BLADE_H;   // normalised 0..1 blade density
  let base = grassColour(dir, blade);

  // Sun key light with a gentle ambient, so the grass stays bright but shaded.
  let diff = max(dot(n, SUN1_DIR), 0.0);
  let ao = grassAO(blade);

  var col = base * (vec3f(0.14) + SUN1_COL * diff * ao);

  // A warm fill from the opposite side, for rounded, natural shading.
  let fill = max(dot(n, vec3f(-0.4, 0.35, -0.2)), 0.0);
  col += base * vec3f(0.35, 0.30, 0.22) * fill * 0.25;

  // A hint of subsurface translucency at grazing angles: blades are thin.
  let grazing = pow(1.0 - max(dot(n, v), 0.0), 2.0);
  col += base * vec3f(0.30, 0.50, 0.20) * grazing * 0.4;

  return col;
}

// ---- tile cull ------------------------------------------------------------

@group(1) @binding(0) var<storage, read_write> tileFlags : array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cull_main(@builtin(global_invocation_id) gid : vec3u) {
  let tiles = vec2u((vec2u(frame.res.xy) + u32(TILE) - 1u) / u32(TILE));
  if (gid.x >= tiles.x || gid.y >= tiles.y) { return; }
  let r = bodyBound() * 1.06;
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

// ---- march ----------------------------------------------------------------

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
    let bs = iSphere(ray.o, ray.d, bodyBound());
    if (bs.y > 0.0) {
      var t = max(bs.x, 0.0);
      let tmax = bs.y;
      var hit = false;
      for (var i = 0; i < MARCH_STEPS; i++) {
        let d = mapBody(ray.o + ray.d * t);
        if (d < MARCH_HIT_EPS * max(t, 0.1)) { hit = true; break; }
        t += d * 0.9;
        if (t > tmax) { break; }
      }
      if (hit) {
        col = shadeBody(ray.o + ray.d * t, ray.d, t);
        depthTag = t;
      }
    }
  }

  textureStore(motionTex, vec2i(gid.xy), vec4f(MOTION_NONE, 0.0, 0.0, 0.0));
  textureStore(outTex, vec2i(gid.xy), vec4f(col, depthTag));
}
