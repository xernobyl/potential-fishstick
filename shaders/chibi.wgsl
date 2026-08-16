// ---------------------------------------------------------------------------
// Chibi planet: a small round sand world with a football pitch — fine grass and
// white markings, a running track, corner flags, goal frames, four floodlights
// and two teams of Sensi-style players plus a referee and linesmen.
//
// Geometry: a smooth sand sphere; only the pitch and margin carry a value-noise
// blade field. Everything is drawn in the material (no texture assets). Shadows
// are raymarched toward the sun.
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
const GOAL_W   : f32 = 0.187;   // goal mouth half-width
const GOAL_H   : f32 = 0.08;    // goal height
const GOAL_R   : f32 = 0.005;   // goal post radius
const FLOOD_H  : f32 = 0.35;    // floodlight pole height
const FLOOD_R  : f32 = 0.01;    // floodlight pole radius
const FLOOD_HEAD : f32 = 0.05;  // floodlight head scale (panel fits within it)
const FLOOD_POWER : f32 = 3.0;  // floodlight light intensity
const FLOOD_PANEL_T : f32 = 0.018;  // panel half-thickness (along the light axis)
const FLOOD_PANEL_W : f32 = 0.065;  // panel half-width / half-height
const FLOOD_LAMP_OFF : f32 = 0.03;  // lamp centre offset from the panel centre
const FLOOD_LAMP_SZ  : f32 = 0.021; // lamp square half-size
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
const SHIRT_R : vec3f = vec3f(0.10, 0.10, 0.12);   // black (referee)
const SHIRT_L : vec3f = vec3f(0.90, 0.80, 0.10);   // yellow (linesmen)
const LINESMAN_A : vec2f = vec2f(0.28, PITCH_W + 0.05);
const LINESMAN_B : vec2f = vec2f(-0.28, -(PITCH_W + 0.05));

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

/// Transform p into the frame anchored at `a` whose +X axis is `axis` (unit).
/// +X becomes "up", Y/Z are two orthonormal tangents. Objects built in this
/// frame come out radial to the planet.
fn localFrame(p : vec3f, a : vec3f, axis : vec3f) -> vec3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(axis.y) > 0.99);
  let t1 = normalize(cross(up, axis));
  let t2 = cross(axis, t1);
  let q = p - a;
  return vec3f(dot(q, axis), dot(q, t1), dot(q, t2));
}

/// A cylinder along the planet's radial direction at anchor `a`.
fn sdCylinderRadial(p : vec3f, a : vec3f, r : f32, hh : f32) -> f32 {
  let q = localFrame(p, a, normalize(a));
  let d = vec2f(length(q.yz) - r, abs(q.x) - hh);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

/// A capsule (line segment) between `a` and `b`, radius `r`.
fn sdCapsule(p : vec3f, a : vec3f, b : vec3f, r : f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
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

/// A smooth, bounded wander for a player at base position `pos`, driven by time.
/// Two incommensurate sine waves per axis give an organic, non-repeating path.
fn wander(pos : vec2f, t : f32) -> vec2f {
  let ph = hash21(pos * 9.1) * 6.2832;
  return vec2f(
    0.07 * sin(t * 0.8 + ph) + 0.03 * sin(t * 1.7 + ph * 1.7),
    0.05 * cos(t * 0.65 + ph * 1.3) + 0.03 * cos(t * 1.3 + ph * 0.6),
  );
}

/// A Sensi-style player at pitch position (y, z), standing radial to the planet.
fn sdPlayerAt(p : vec3f, pos : vec2f) -> f32 {
  let wp = pos + wander(pos, frame.camPos.w);
  let a = vec3f(surfX(wp.x, wp.y), wp.x, wp.y);
  let q = localFrame(p, a, normalize(a));   // +X = radial (up)

  let head  = sdSphere(q - vec3f(PLAYER_HEAD_R + PLAYER_TORSO_H + PLAYER_LEG_LEN, 0.0, 0.0), PLAYER_HEAD_R);
  let torso = sdRoundBox(q - vec3f(PLAYER_LEG_LEN + PLAYER_TORSO_H * 0.5, 0.0, 0.0),
                         vec3f(PLAYER_TORSO_H * 0.5, PLAYER_TORSO_W, PLAYER_TORSO_W), PLAYER_TORSO_W * 0.45);
  let legOff = PLAYER_LEG_R * 1.3;
  let leg1 = sdCylinderX(q - vec3f(PLAYER_LEG_LEN * 0.5, legOff, 0.0), PLAYER_LEG_R, PLAYER_LEG_LEN * 0.5);
  let leg2 = sdCylinderX(q - vec3f(PLAYER_LEG_LEN * 0.5, -legOff, 0.0), PLAYER_LEG_R, PLAYER_LEG_LEN * 0.5);

  return min(min(head, torso), min(leg1, leg2));
}

/// All players (both teams, the referee and the two linesmen).
fn sdPlayers(p : vec3f) -> f32 {
  var d = 1e5;
  for (var i = 0; i < 5; i++) {
    d = min(d, sdPlayerAt(p, TEAM_A[i]));
    d = min(d, sdPlayerAt(p, TEAM_B[i]));
  }
  d = min(d, sdPlayerAt(p, REFEREE));
  d = min(d, sdPlayerAt(p, LINESMAN_A));
  d = min(d, sdPlayerAt(p, LINESMAN_B));
  return d;
}

/// The goal frames: two posts and a crossbar at each end (y = +-PITCH_L),
/// all standing radial to the planet.
fn sdGoal(p : vec3f) -> f32 {
  var d = 1e5;
  for (var g = 0; g < 2; g++) {
    let s = f32(g) * 2.0 - 1.0;
    for (var pi = 0; pi < 2; pi++) {
      let pz = f32(pi) * 2.0 - 1.0;
      let a = vec3f(surfX(s * PITCH_L, pz * GOAL_W), s * PITCH_L, pz * GOAL_W);
      // A radial cylinder is centred on its anchor, so lift the anchor to the
      // post's midpoint to get the full GOAL_H above the surface.
      let post = sdCylinderRadial(p, a + normalize(a) * GOAL_H * 0.5, GOAL_R, GOAL_H * 0.5);
      d = min(d, post);
    }
    // Crossbar at the posts' top height, spanning the goal mouth (world Z).
    let a = vec3f(surfX(s * PITCH_L, GOAL_W), s * PITCH_L, 0.0);
    let q = localFrame(p, a, normalize(a));
    let bar = sdRoundBox(q - vec3f(GOAL_H, 0.0, 0.0), vec3f(GOAL_R, GOAL_R, GOAL_W), GOAL_R * 0.5);
    d = min(d, bar);
  }
  return d;
}

/// Floodlight corner position (y, z) — on the track's centreline so the poles
/// stay inside the planet's silhouette.
fn floodCorner(c : i32) -> vec2f {
  let sy = f32(c & 1) * 2.0 - 1.0;
  let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
  return vec2f(sy * (PITCH_L + MARGIN + TRACK_W * 0.5), sz * (PITCH_W + MARGIN + TRACK_W * 0.5));
}

/// Four A-frame floodlights: two splayed legs, a cross-brace, a metal head
/// angled toward the pitch and an emissive lamp at its front.
fn sdFloodlight(p : vec3f) -> f32 {
  var d = 1e5;
  for (var c = 0; c < 4; c++) {
    let pos = floodCorner(c);
    let a = vec3f(surfX(pos.x, pos.y), pos.x, pos.y);
    let rad = normalize(a);
    let aTop = a + rad * FLOOD_H;

    // Tangent basis, so the A-frame splays across the planet's surface.
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(rad.y) > 0.99);
    let t1 = normalize(cross(up, rad));

    // A-frame: two legs splayed in t1, meeting at the top, plus a cross-brace.
    let spread = FLOOD_H * 0.30;
    let leg1 = sdCapsule(p, a + t1 * spread, aTop, FLOOD_R * 0.8);
    let leg2 = sdCapsule(p, a - t1 * spread, aTop, FLOOD_R * 0.8);
    let brace = sdCapsule(p, a + t1 * spread * 0.45 + rad * FLOOD_H * 0.55,
                          a - t1 * spread * 0.45 + rad * FLOOD_H * 0.55, FLOOD_R * 0.5);
    d = min(d, min(leg1, min(leg2, brace)));

    // Head: a flat panel — a "window" of four lamp squares — angled toward the
    // pitch. Its back face sits on the pole top; the lamps are on the front.
    let inward = normalize(vec3f(CR, 0.0, 0.0) - aTop);
    let headC = aTop + inward * FLOOD_PANEL_T;
    let q = localFrame(p, headC, inward);
    let panel = sdRoundBox(q, vec3f(FLOOD_PANEL_T, FLOOD_PANEL_W, FLOOD_PANEL_W), 0.004);
    d = min(d, panel);
  }
  return d;
}

/// Floodlight illumination: four spotlights aimed at the pitch centre, emitted
/// from the lamp panel at the front of each head.
fn floodlightLight(p : vec3f, n : vec3f) -> vec3f {
  var acc = vec3f(0.0);
  for (var c = 0; c < 4; c++) {
    let pos = floodCorner(c);
    let a = vec3f(surfX(pos.x, pos.y), pos.x, pos.y);
    let aTop = a + normalize(a) * FLOOD_H;
    let inward = normalize(vec3f(CR, 0.0, 0.0) - aTop);
    let lp = aTop + inward * (2.0 * FLOOD_PANEL_T);
    let L = lp - p;
    let d = length(L);
    let l = L / d;
    let aim = normalize(vec3f(CR, 0.0, 0.0) - lp);
    let spot = smoothstep(0.15, 0.85, dot(l, aim));
    let att = 1.0 / (1.0 + 0.3 * d * d);
    acc += vec3f(1.0, 0.96, 0.86) * (FLOOD_POWER * att * spot * max(dot(n, l), 0.0));
  }
  return acc;
}

/// Shade one player: returns (colour, 1) on the player's body, else (0,0,0,0).
fn playerShade(p : vec3f, pos : vec2f, shirt : vec3f) -> vec4f {
  let wp = pos + wander(pos, frame.camPos.w);
  let a = vec3f(surfX(wp.x, wp.y), wp.x, wp.y);
  let q = localFrame(p, a, normalize(a));   // radial frame, +X = up
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

  // Corner flags: four thin posts at the pitch corners, standing radial.
  for (var c = 0; c < 4; c++) {
    let sy = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let cy = sy * PITCH_L;
    let cz = sz * PITCH_W;
    let fx = surfX(cy, cz);
    let q = localFrame(p, vec3f(fx, cy, cz), normalize(vec3f(fx, cy, cz)));
    let post = sdRoundBox(q - vec3f(FLAG_H * 0.5, 0.0, 0.0),
                          vec3f(FLAG_H * 0.5, FLAG_R, FLAG_R), FLAG_R * 0.5);
    d = min(d, post);
  }

  // The players (both teams + referee + linesmen).
  d = min(d, sdPlayers(p));

  // The goal frames.
  d = min(d, sdGoal(p));

  // The floodlight poles.
  d = min(d, sdFloodlight(p));

  return d;
}

/// Raymarched soft shadow toward the sun. Returns 0 (fully shadowed) .. 1 (lit).
/// Marches mapBody (players, goals, floodlights) with a wide penumbra kernel.
fn softShadow(ro : vec3f, rd : vec3f, mint : f32, maxt : f32) -> f32 {
  var res = 1.0;
  var t = mint;
  for (var i = 0; i < 14; i++) {
    let h = mapBody(ro + rd * t);
    res = min(res, h / (t * 0.18));
    t += max(h, 0.012);
    if (res < 0.005 || t > maxt) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

fn bodyBound() -> f32 {
  // Covers the floodlight heads (the tallest feature), plus margin.
  return CR + FLOOD_H + FLOOD_HEAD + 0.1;
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
  // Set to a bright colour when the point is an emissive lamp; shading is
  // bypassed for it (a lamp does not take sun or floodlight).
  var emissive = vec3f(0.0);

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

  // The players: two teams + a referee + two linesmen.
  for (var i = 0; i < 5; i++) {
    let pa = playerShade(p, TEAM_A[i], SHIRT_A);
    if (pa.w > 0.5) { base = pa.xyz; }
    let pb = playerShade(p, TEAM_B[i], SHIRT_B);
    if (pb.w > 0.5) { base = pb.xyz; }
  }
  let pr = playerShade(p, REFEREE, SHIRT_R);
  if (pr.w > 0.5) { base = pr.xyz; }
  let pl1 = playerShade(p, LINESMAN_A, SHIRT_L);
  if (pl1.w > 0.5) { base = pl1.xyz; }
  let pl2 = playerShade(p, LINESMAN_B, SHIRT_L);
  if (pl2.w > 0.5) { base = pl2.xyz; }

  // Goals: white frame. The marched point is ~0 distance from the goal SDF when
  // it sits on a post or crossbar, so proximity colours exactly the frame.
  if (sdGoal(p) < GOAL_R * 1.5) { base = vec3f(0.95, 0.95, 0.97); }

  // Corner flags: fluorescent yellow posts.
  for (var c = 0; c < 4; c++) {
    let sy = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let cy = sy * PITCH_L;
    let cz = sz * PITCH_W;
    let fx = surfX(cy, cz);
    if (abs(p.y - cy) < FLAG_R * 3.0 && abs(p.z - cz) < FLAG_R * 3.0
        && p.x > fx - 0.01 && p.x < fx + FLAG_H + 0.01) {
      base = vec3f(0.85, 0.95, 0.08);
    }
  }

  // Floodlights: metal A-frame structure, topped by a flat panel whose front
  // face carries four emissive lamp squares (a 2x2 grid). The lamps are
  // self-luminous and bright enough to bloom; the rest reads as grey metal.
  for (var c = 0; c < 4; c++) {
    let pos = floodCorner(c);
    let a = vec3f(surfX(pos.x, pos.y), pos.x, pos.y);
    let rad = normalize(a);
    let aTop = a + rad * FLOOD_H;
    let inward = normalize(vec3f(CR, 0.0, 0.0) - aTop);
    let headC = aTop + inward * FLOOD_PANEL_T;
    let q = localFrame(p, headC, inward);
    let ay = abs(q.y);
    let az = abs(q.z);
    let onLamp = abs(q.x - FLOOD_PANEL_T) < FLOOD_PANEL_T * 0.5
        && abs(ay - FLOOD_LAMP_OFF) < FLOOD_LAMP_SZ
        && abs(az - FLOOD_LAMP_OFF) < FLOOD_LAMP_SZ;
    if (onLamp) {
      emissive = vec3f(4.0, 3.6, 2.6);
    } else if (sdFloodlight(p) < FLOOD_R * 1.2) {
      base = vec3f(0.33, 0.36, 0.40);
    }
  }

  // Sun key light, soft-shadowed by the raymarched players/goals/floodlights.
  // Ambient kept low so the floodlights read as the primary source.
  let sunSh = softShadow(p + n * 0.01, SUN1_DIR, 0.02, 1.2);
  let diff = max(dot(n, SUN1_DIR), 0.0) * sunSh;
  var col = base * (vec3f(0.06) + SUN1_COL * diff);

  // Floodlight illumination.
  col += base * floodlightLight(p, n);

  // A warm fill from the opposite side, for rounded, natural shading.
  let fill = max(dot(n, vec3f(-0.4, 0.35, -0.2)), 0.0);
  col += base * vec3f(0.35, 0.30, 0.22) * fill * 0.25;

  // A hint of subsurface translucency at grazing angles.
  let grazing = pow(1.0 - max(dot(n, v), 0.0), 2.0);
  col += base * vec3f(0.30, 0.50, 0.20) * grazing * 0.4;

  // An emissive lamp is its own light source: return it unshaded.
  if (emissive.r > 0.5) { return emissive; }

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
