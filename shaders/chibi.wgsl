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
const BLADE_H  : f32 = 0.018;   // grass blade height
const BLADE_F  : f32 = 110.0;   // blade frequency (2x on the pitch)
const PITCH_L  : f32 = 0.62;    // pitch half-length, along Y (goal to goal)
const PITCH_W  : f32 = 0.34;    // pitch half-width, along Z
const MARGIN   : f32 = 0.14;    // grass margin between pitch and track
const TRACK_W  : f32 = 0.07;    // running-track width
const TRACK_R  : f32 = 0.26;    // track corner radius
const FLAG_H   : f32 = 0.07;    // corner-flag post height
const FLAG_R   : f32 = 0.006;   // corner-flag post radius
const LINE_W   : f32 = 0.012;   // pitch line half-width
const ARC_R    : f32 = 0.12;    // corner-arc radius
const PLAYER_Y : f32 = 0.12;    // player's pitch position (along Y)
const PLAYER_Z : f32 = 0.06;    // player's pitch position (along Z)

fn sdRoundBox(p : vec3f, b : vec3f, r : f32) -> f32 {
  let q = abs(p) - b + vec3f(r);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdSphere(p : vec3f, r : f32) -> f32 { return length(p) - r; }

/// A cylinder along +X (the field normal = "up" for objects on the pitch).
fn sdCylinderX(p : vec3f, r : f32, hh : f32) -> f32 {
  let d = vec2f(length(p.yz) - r, abs(p.x) - hh);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

/// Fine, smooth blade field (two octaves of value noise), parameterised by
/// frequency: the pitch mows it at 2x.
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

/// Is (y, z) inside the pitch — the rectangle plus the four corner arcs?
fn inPitchRegion(y : f32, z : f32) -> bool {
  let ay = abs(y);
  let az = abs(z);
  let inRect = ay < PITCH_L && az < PITCH_W;
  let inArc = ay > PITCH_L && az > PITCH_W
              && length(vec2f(ay - PITCH_L, az - PITCH_W)) < ARC_R;
  return inRect || inArc;
}

/// A Sensi-style player: big head, chunky torso, stubby legs. Local space is
/// anchored at the feet with +X as "up" (the field normal).
fn sdPlayer(p : vec3f) -> f32 {
  let a = vec3f(surfX(PLAYER_Y, PLAYER_Z), PLAYER_Y, PLAYER_Z);
  let q = p - a;

  let headR  = 0.040;
  let torsoH = 0.050;
  let torsoW = 0.027;
  let legLen = 0.042;
  let legR   = 0.008;

  let head  = sdSphere(q - vec3f(headR + torsoH + legLen, 0.0, 0.0), headR);
  let torso = sdRoundBox(q - vec3f(legLen + torsoH * 0.5, 0.0, 0.0),
                         vec3f(torsoH * 0.5, torsoW, torsoW), 0.010);
  let leg1 = sdCylinderX(q - vec3f(legLen * 0.5, 0.010, 0.0), legR, legLen * 0.5);
  let leg2 = sdCylinderX(q - vec3f(legLen * 0.5, -0.010, 0.0), legR, legLen * 0.5);

  return min(min(head, torso), min(leg1, leg2));
}

fn mapBody(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);
  let r = length(p);

  // The planet is smooth sand; only the pitch carries grass.
  var d = r - CR;
  if (inPitchRegion(p.y, p.z) && p.x > 0.0) {
    d -= grassBlades(dir, BLADE_F * 2.0);
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

  // The Sensi-style player.
  d = min(d, sdPlayer(p));

  return d;
}

fn bodyBound() -> f32 {
  // Covers the planet, the player (tallest feature) and the corner flags.
  return CR + 0.3;
}

fn calcNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0012;
  return normalize(
      e.xyy * mapBody(p + e.xyy) + e.yyx * mapBody(p + e.yyx)
    + e.yxy * mapBody(p + e.yxy) + e.xxx * mapBody(p + e.xxx));
}

// ---- shading --------------------------------------------------------------

/// Football pitch markings, drawn in white on the field region. Returns 1 on a
/// line, 0 off. `y` is the goal-to-goal axis, `z` the sideline axis.
fn pitchLines(y : f32, z : f32) -> f32 {
  let ay = abs(y);
  let az = abs(z);
  // Boundary: the outer rectangle — goal lines + sidelines.
  let edgeY = abs(ay - PITCH_L) < LINE_W && az < PITCH_W;
  let edgeZ = abs(az - PITCH_W) < LINE_W && ay < PITCH_L;
  let boundary = select(0.0, 1.0, edgeY || edgeZ);
  // Halfway line, centre circle, centre spot. The halfway line and circle sit
  // at the field's pole, head-on to the camera, so they foreshorten less than
  // the boundary lines — thin them to match the same visual weight.
  let halfway = select(0.0, 1.0, ay < LINE_W * 0.6);
  let circle = select(0.0, 1.0, abs(length(vec2f(y, z)) - 0.09) < LINE_W * 0.6);
  let spot = select(0.0, 1.0, length(vec2f(y, z)) < LINE_W * 0.6);
  // Penalty area outlines.
  let penL = PITCH_L - 0.16;
  let penW = PITCH_W * 0.55;
  let inPen = az < penW && ay > penL && ay < PITCH_L;
  let penEdge = select(0.0, 1.0, inPen && (abs(ay - penL) < LINE_W
                         || abs(az - penW) < LINE_W || abs(ay - PITCH_L) < LINE_W));
  // Corner arcs: a quarter circle at each of the four corners, centred on the
  // corner itself and bulging into the outer quadrant.
  let cornerDist = length(vec2f(ay - PITCH_L, az - PITCH_W));
  let inCorner = ay > PITCH_L && az > PITCH_W;
  let arc = select(0.0, 1.0, abs(cornerDist - ARC_R) < LINE_W && inCorner);
  return max(max(boundary, max(halfway, circle)), max(max(spot, penEdge), arc));
}

fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let n = calcNormal(p);
  let v = -rd;
  let dir = normalize(p + 1e-6);

  // Default: sand.
  var base = vec3f(0.74, 0.65, 0.48);

  let inPitch = p.x > 0.0 && inPitchRegion(p.y, p.z);
  let inTrack = p.x > 0.0 && onTrack(p.y, p.z);

  // Pitch: fine green grass with white markings.
  if (inPitch) {
    let line = pitchLines(p.y, p.z);
    let green = mix(vec3f(0.09, 0.40, 0.15), vec3f(0.15, 0.50, 0.21), fbm3(dir * 18.0));
    base = mix(green, vec3f(0.92, 0.94, 0.96), line);
  }
  // Track: cinder red.
  if (inTrack) {
    base = vec3f(0.55, 0.28, 0.20);
  }

  // The Sensi player: head (skin), torso (shirt), legs (shorts).
  let pa = vec3f(surfX(PLAYER_Y, PLAYER_Z), PLAYER_Y, PLAYER_Z);
  if (length(p - pa) < 0.16) {
    let hx = p.x - pa.x;
    if (hx > 0.12) {
      base = vec3f(0.93, 0.78, 0.60);
    } else if (hx > 0.042) {
      base = vec3f(0.80, 0.18, 0.18);
    } else {
      base = vec3f(0.12, 0.12, 0.18);
    }
  }

  // Corner flags: bright yellow posts.
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

  // Sun key light with a gentle ambient.
  let diff = max(dot(n, SUN1_DIR), 0.0);
  var col = base * (vec3f(0.16) + SUN1_COL * diff);

  // A warm fill from the opposite side, for rounded, natural shading.
  let fill = max(dot(n, vec3f(-0.4, 0.35, -0.2)), 0.0);
  col += base * vec3f(0.35, 0.30, 0.22) * fill * 0.25;

  // A hint of subsurface translucency at grazing angles.
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
