// ---------------------------------------------------------------------------
// Chibi planet: a small round sand world with a football pitch — fine grass and
// white markings, a running track, corner flags and a Sensi-style player.
//
// Geometry: a smooth sand sphere; only the pitch carries a fine value-noise
// blade field for the grass. Everything is drawn in the material (no texture
// assets).
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"

const CR       : f32 = 1.0;     // planet radius
const BLADE_H  : f32 = 0.018;   // grass blade height
const BLADE_F  : f32 = 220.0;   // pitch blade frequency
const PITCH_L  : f32 = 0.62;    // pitch half-length, along Y (goal to goal)
const PITCH_W  : f32 = 0.34;    // pitch half-width, along Z
const MARGIN   : f32 = 0.14;    // grass margin between pitch and track
const TRACK_W  : f32 = 0.07;    // running-track width
const TRACK_R  : f32 = 0.26;    // track corner radius
const FLAG_H   : f32 = 0.07;    // corner-flag post height
const FLAG_R   : f32 = 0.006;   // corner-flag post radius
const LINE_W   : f32 = 0.007;   // pitch line half-width
const ARC_R    : f32 = 0.12;    // corner-arc radius
// Player proportions (scaled to ~5% of the pitch length).
const PLAYER_HEAD_R  : f32 = 0.016;
const PLAYER_TORSO_H : f32 = 0.020;
const PLAYER_TORSO_W : f32 = 0.011;
const PLAYER_LEG_LEN : f32 = 0.017;
const PLAYER_LEG_R   : f32 = 0.004;

// Two five-a-side teams (y, z on the pitch) and a referee. Team A plays up
// (+Y), team B down (-Y).
const TEAM_A : array<vec2f, 5> = array<vec2f, 5>(
  vec2f(0.16, 0.08), vec2f(0.03, -0.10), vec2f(-0.10, 0.12),
  vec2f(0.24, -0.03), vec2f(0.00, 0.22),
);
const TEAM_B : array<vec2f, 5> = array<vec2f, 5>(
  vec2f(-0.16, 0.08), vec2f(-0.03, -0.10), vec2f(0.10, -0.12),
  vec2f(-0.24, -0.03), vec2f(0.00, -0.22),
);
const REFEREE : vec2f = vec2f(0.0, 0.0);
const SHIRT_A : vec3f = vec3f(0.80, 0.18, 0.18);   // red
const SHIRT_B : vec3f = vec3f(0.18, 0.28, 0.80);   // blue
const SHIRT_R : vec3f = vec3f(0.10, 0.10, 0.12);   // black

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

/// Is (y, z) inside the pitch (the playing rectangle)?
fn inPitchRegion(y : f32, z : f32) -> bool {
  return abs(y) < PITCH_L && abs(z) < PITCH_W;
}

/// Is (y, z) in the rough margin between the pitch and the track?
fn inMargin(y : f32, z : f32) -> bool {
  let insideInner = sdRoundRect2(vec2f(y, z), vec2f(PITCH_L + MARGIN, PITCH_W + MARGIN), TRACK_R) < 0.0;
  return insideInner && !inPitchRegion(y, z);
}

/// A Sensi-style player at pitch position (y, z). Local space is anchored at
/// the feet with +X as "up".
fn sdPlayerAt(p : vec3f, pos : vec2f) -> f32 {
  let a = vec3f(surfX(pos.x, pos.y), pos.x, pos.y);
  let q = p - a;

  let head  = sdSphere(q - vec3f(PLAYER_HEAD_R + PLAYER_TORSO_H + PLAYER_LEG_LEN, 0.0, 0.0), PLAYER_HEAD_R);
  let torso = sdRoundBox(q - vec3f(PLAYER_LEG_LEN + PLAYER_TORSO_H * 0.5, 0.0, 0.0),
                         vec3f(PLAYER_TORSO_H * 0.5, PLAYER_TORSO_W, PLAYER_TORSO_W), PLAYER_TORSO_W * 0.45);
  let legOff = PLAYER_LEG_R * 1.3;
  let leg1 = sdCylinderX(q - vec3f(PLAYER_LEG_LEN * 0.5, legOff, 0.0), PLAYER_LEG_R, PLAYER_LEG_LEN * 0.5);
  let leg2 = sdCylinderX(q - vec3f(PLAYER_LEG_LEN * 0.5, -legOff, 0.0), PLAYER_LEG_R, PLAYER_LEG_LEN * 0.5);

  return min(min(head, torso), min(leg1, leg2));
}

/// All players (both teams + the referee).
fn sdPlayers(p : vec3f) -> f32 {
  var d = 1e5;
  for (var i = 0; i < 5; i++) {
    d = min(d, sdPlayerAt(p, TEAM_A[i]));
    d = min(d, sdPlayerAt(p, TEAM_B[i]));
  }
  d = min(d, sdPlayerAt(p, REFEREE));
  return d;
}

/// Shade one player: returns (colour, 1) on the player's body, else (0,0,0,0).
fn playerShade(p : vec3f, pos : vec2f, shirt : vec3f) -> vec4f {
  let a = vec3f(surfX(pos.x, pos.y), pos.x, pos.y);
  let q = p - a;
  let ph = PLAYER_LEG_LEN + PLAYER_TORSO_H + 2.0 * PLAYER_HEAD_R;
  let pw = max(PLAYER_TORSO_W, PLAYER_HEAD_R);
  if (q.x <= -0.005 || q.x >= ph + 0.005 || length(q.yz) >= pw + 0.005) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  var col = vec3f(0.12, 0.12, 0.18);   // shorts
  if (q.x > PLAYER_LEG_LEN + PLAYER_TORSO_H) {
    col = vec3f(0.93, 0.78, 0.60);     // head (skin)
  } else if (q.x > PLAYER_LEG_LEN) {
    col = shirt;                       // torso
  }
  return vec4f(col, 1.0);
}

fn mapBody(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);
  let r = length(p);

  // Grass by region: fine on the pitch, rough in the margin, smooth sand
  // elsewhere (track + outside).
  var d = r - CR;
  if (p.x > 0.0 && inPitchRegion(p.y, p.z)) {
    d -= grassBlades(dir, BLADE_F);
  } else if (p.x > 0.0 && inMargin(p.y, p.z)) {
    d -= grassBlades(dir, BLADE_F * 0.5);
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

  // The players (both teams + referee).
  d = min(d, sdPlayers(p));

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
  // Corner arcs: a quarter circle at each corner, drawn INSIDE the pitch.
  let cornerDist = length(vec2f(ay - PITCH_L, az - PITCH_W));
  let inCorner = ay < PITCH_L && az < PITCH_W;
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
  let inMarginG = p.x > 0.0 && inMargin(p.y, p.z);
  let inTrack = p.x > 0.0 && onTrack(p.y, p.z);

  // Pitch: fine green grass with white markings.
  if (inPitch) {
    let line = pitchLines(p.y, p.z);
    let green = mix(vec3f(0.09, 0.40, 0.15), vec3f(0.15, 0.50, 0.21), fbm3(dir * 18.0));
    base = mix(green, vec3f(0.92, 0.94, 0.96), line);
  }
  // Margin: rough, slightly lighter/longer grass.
  if (inMarginG) {
    base = mix(vec3f(0.12, 0.46, 0.18), vec3f(0.20, 0.56, 0.24), fbm3(dir * 8.0));
  }
  // Track: cinder red.
  if (inTrack) {
    base = vec3f(0.55, 0.28, 0.20);
  }

  // The players: two teams + a referee.
  for (var i = 0; i < 5; i++) {
    let pa = playerShade(p, TEAM_A[i], SHIRT_A);
    if (pa.w > 0.5) { base = pa.xyz; }
    let pb = playerShade(p, TEAM_B[i], SHIRT_B);
    if (pb.w > 0.5) { base = pb.xyz; }
  }
  let pr = playerShade(p, REFEREE, SHIRT_R);
  if (pr.w > 0.5) { base = pr.xyz; }

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
