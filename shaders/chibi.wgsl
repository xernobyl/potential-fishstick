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
const BLADE_H  : f32 = 0.018;   // grass blade height (rough)
const BLADE_F  : f32 = 110.0;   // rough blade frequency
const PITCH_L  : f32 = 0.62;    // pitch half-length, along Y (goal to goal)
const PITCH_W  : f32 = 0.34;    // pitch half-width, along Z
const MARGIN   : f32 = 0.14;    // grass margin between pitch and track
const TRACK_W  : f32 = 0.07;    // running-track width
const TRACK_R  : f32 = 0.26;    // track corner radius
const FLAG_H   : f32 = 0.07;    // corner-flag post height
const FLAG_R   : f32 = 0.006;   // corner-flag post radius
const LINE_W   : f32 = 0.012;   // pitch line half-width

fn sdRoundBox(p : vec3f, b : vec3f, r : f32) -> f32 {
  let q = abs(p) - b + vec3f(r);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

/// Rolling hills: a low-frequency fbm displacement.
fn grassHills(dir : vec3f) -> f32 {
  return (fbm3(dir * 2.6) - 0.5) * HILL_AMP;
}

/// Fine, smooth blade field (two octaves of value noise), parameterised by
/// frequency: the pitch mows it at 2x (a golf "green"), the rest is "rough".
fn grassBlades(dir : vec3f, freq : f32) -> f32 {
  let n = 0.62 * vnoise(dir * freq) + 0.38 * vnoise(dir * freq * 2.3);
  return n * BLADE_H;
}

/// The sphere surface's X coordinate at (y, z) — anchors things that sit on it.
fn surfX(y : f32, z : f32) -> f32 {
  return sqrt(max(CR * CR - y * y - z * z, 0.02));
}

/// 2D rounded-rectangle SDF, for the running track's oval ring.
fn sdRoundRect2(p : vec2f, b : vec2f, r : f32) -> f32 {
  let q = abs(p) - b + vec2f(r);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/// Is the surface point (in the tangent plane) on the track ring?
fn onTrack(y : f32, z : f32) -> bool {
  let b = vec2f(PITCH_L + MARGIN + TRACK_W * 0.5, PITCH_W + MARGIN + TRACK_W * 0.5);
  return abs(sdRoundRect2(vec2f(y, z), b, TRACK_R)) < TRACK_W * 0.5;
}

fn mapBody(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);
  let r = length(p);

  // Grass displacement by region: the pitch is mowed fine (2x), the track is
  // smooth, everything else (margin + planet) is rough.
  let inPitch = p.x > 0.0 && abs(p.y) < PITCH_L && abs(p.z) < PITCH_W;
  let inTrack = p.x > 0.0 && onTrack(p.y, p.z);

  var d = r - CR - grassHills(dir);
  if (inPitch) {
    d -= grassBlades(dir, BLADE_F * 2.0);
  } else if (!inTrack) {
    d -= grassBlades(dir, BLADE_F);
  }

  // Corner flags: four thin posts at the pitch corners.
  for (var c = 0; c < 4; c++) {
    let sy = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let cy = sy * PITCH_L;
    let cz = sz * PITCH_W;
    let fx = surfX(cy, cz);
    let post = sdRoundBox(p - vec3f(fx + FLAG_H * 0.5, cy, cz),
                          vec3f(FLAG_H * 0.5, FLAG_R, FLAG_R), FLAG_R * 0.5);
    d = min(d, post);
  }

  return d;
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

/// Football pitch markings, drawn in white on the field region. Returns 1 on a
/// line, 0 off. `y` is the goal-to-goal axis, `z` the sideline axis.
fn pitchLines(y : f32, z : f32) -> f32 {
  let ay = abs(y);
  let az = abs(z);
  // Boundary: the outer rectangle — goal lines + sidelines.
  let edgeY = abs(ay - PITCH_L) < LINE_W && az < PITCH_W;
  let edgeZ = abs(az - PITCH_W) < LINE_W && ay < PITCH_L;
  let boundary = select(0.0, 1.0, edgeY || edgeZ);
  // Halfway line, centre circle, centre spot.
  let halfway = select(0.0, 1.0, ay < LINE_W);
  let circle = select(0.0, 1.0, abs(length(vec2f(y, z)) - 0.09) < LINE_W);
  let spot = select(0.0, 1.0, length(vec2f(y, z)) < LINE_W);
  // Penalty area outlines.
  let penL = PITCH_L - 0.16;
  let penW = PITCH_W * 0.55;
  let inPen = az < penW && ay > penL && ay < PITCH_L;
  let penEdge = select(0.0, 1.0, inPen && (abs(ay - penL) < LINE_W
                         || abs(az - penW) < LINE_W || abs(ay - PITCH_L) < LINE_W));
  // Corner arcs: a quarter circle at each of the four corners.
  let cx = PITCH_L - LINE_W;
  let cz = PITCH_W - LINE_W;
  let cornerDist = length(vec2f(ay, az) - vec2f(cx, cz));
  let inCorner = ay > cx && az > cz;
  let arc = select(0.0, 1.0, abs(cornerDist - 0.07) < LINE_W && inCorner);
  return max(max(boundary, max(halfway, circle)), max(max(spot, penEdge), arc));
}

fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let n = calcNormal(p);
  let v = -rd;
  let dir = normalize(p + 1e-6);
  let blade = grassBlades(dir, BLADE_F) / BLADE_H;   // rough blade density (margin/planet)
  var base = grassColour(dir, blade);

  let inPitch = p.x > 0.0 && abs(p.y) < PITCH_L && abs(p.z) < PITCH_W;
  let inTrack = p.x > 0.0 && onTrack(p.y, p.z);

  // Corner flags: bright yellow posts at the four corners.
  for (var c = 0; c < 4; c++) {
    let sy = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let cy = sy * PITCH_L;
    let cz = sz * PITCH_W;
    let fx = surfX(cy, cz);
    if (abs(p.y - cy) < FLAG_R * 3.0 && abs(p.z - cz) < FLAG_R * 3.0
        && p.x > fx - 0.01 && p.x < fx + FLAG_H + 0.01) {
      base = vec3f(1.0, 0.72, 0.12);
    }
  }

  // Track: cinder red. Pitch: fine "green" grass with white markings.
  if (inTrack) {
    base = vec3f(0.55, 0.28, 0.20);
  } else if (inPitch) {
    let line = pitchLines(p.y, p.z);
    let green = mix(vec3f(0.09, 0.40, 0.15), vec3f(0.15, 0.50, 0.21), fbm3(dir * 18.0));
    base = mix(green, vec3f(0.92, 0.94, 0.96), line);
  }

  // Sun key light with a gentle ambient. Grass gets self-shadowing; the smooth
  // track and pitch do not.
  let diff = max(dot(n, SUN1_DIR), 0.0);
  let ao = select(grassAO(blade), 1.0, inPitch || inTrack);

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
