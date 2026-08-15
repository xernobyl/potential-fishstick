// ---------------------------------------------------------------------------
// Chibi planet: a grassy world with a football pitch ON its surface, a running
// track, terraced grandstands, floodlights, goal nets and a bouncing football.
//
// The pitch is NOT a separate platform — it is a material region on the sphere
// itself (green with white lines). The planet's surface is grass elsewhere:
// rolling hills (low-frequency fbm) with a Voronoi blade field, adapted from
// David Hoskins' "Rolling hills" (Shadertoy Xsf3zX).
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"

// ---- geometry constants ---------------------------------------------------
//
// The pitch's LONG side runs along Z so it reads horizontal from the default
// camera (which sits on the +X side looking at the planet's north pole).

const CR       : f32 = 1.0;     // planet radius
const PITCH_L  : f32 = 0.33;    // pitch half-LENGTH, along Z (goal to goal)
const PITCH_W  : f32 = 0.21;    // pitch half-WIDTH, along X (sideline to sideline)
const TRACK_W  : f32 = 0.07;    // running-track width
const SEAT_W   : f32 = 0.12;    // grandstand tier depth
const SEAT_H   : f32 = 0.05;    // tier rise
const FB_R     : f32 = 0.05;    // football radius
const POST_H   : f32 = 0.10;    // goal height
const POST_R   : f32 = 0.008;   // goal post thickness
const POLE_H   : f32 = 0.40;    // floodlight pole height
const POLE_R   : f32 = 0.012;   // floodlight pole radius
const HEAD_R   : f32 = 0.035;   // floodlight head radius
const LINE_W   : f32 = 0.008;   // pitch line half-width

// ---- primitives (iq formulations; smin comes from common.wgsl) ------------

fn sdSphere(p : vec3f, r : f32) -> f32 { return length(p) - r; }

fn sdBox(p : vec3f, b : vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdRoundBox(p : vec3f, b : vec3f, r : f32) -> f32 {
  let q = abs(p) - b + vec3f(r);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdCylinder(p : vec3f, r : f32, hh : f32) -> f32 {
  let d = vec2f(length(p.xz) - r, abs(p.y) - hh);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

// ---- grass noise (adapted from David Hoskins' "Rolling hills") -------------

/// 2x2 hash, for the Voronoi cell offset.
fn hash22(p : vec2f) -> vec2f {
  let p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  let q = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

/// 2D Voronoi: (blade density 0..~0.4, random cell id). Blades sit at cell centres.
fn voronoi2(x : vec2f) -> vec2f {
  let p = floor(x);
  let f = fract(x);
  var res = 100.0;
  var id = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let b = vec2f(f32(i), f32(j));
      let r = b - f + hash22(p + b);
      let d = dot(r, r);
      if (d < res) { res = d; id = hash21(p + b); }
    }
  }
  return vec2f(max(0.4 - sqrt(res), 0.0), id);
}

/// Rolling hills: a low-frequency fbm, subtle enough that the pitch stays level.
fn grassHills(dir : vec3f) -> f32 {
  return (fbm3(dir * 3.5) - 0.5) * 0.05;
}

/// Grass colour: hill-shaded green, with lighter Voronoi blades and pale seed tips.
fn grassColour(p : vec3f, dir : vec3f) -> vec3f {
  let uv = dir.xz * 26.0;
  let v = voronoi2(uv);
  let blade = v.x;
  let hills = fbm3(dir * 6.0);
  var col = mix(vec3f(0.07, 0.30, 0.09), vec3f(0.22, 0.52, 0.18), hills);
  col = mix(col, vec3f(0.45, 0.62, 0.30), blade * 1.5);
  col = mix(col, vec3f(0.75, 0.80, 0.55), step(0.38, blade) * step(0.6, v.y));
  return col;
}

// ---- the world ------------------------------------------------------------

fn footballPos() -> vec3f {
  let t = frame.camPos.w;                                   // time, seconds
  let bounce = abs(sin(t * 2.1));                           // 0..1
  return vec3f(0.14 * sin(t * 1.1), CR + FB_R + 0.16 * bounce, 0.10 * cos(t * 0.83));
}

fn mapBody(p : vec3f) -> f32 {
  let dir = normalize(p + 1e-6);

  // The planet itself: a sphere. Grass hills displace the surface everywhere
  // EXCEPT the field region, which stays smooth so the pitch is level.
  var d = length(p) - CR;
  let onField = abs(p.x) < PITCH_W * 1.15 && abs(p.z) < PITCH_L * 1.15;
  if (!onField) {
    d += grassHills(dir);
  }

  // Terraced grandstands on the two SIDELINES (x = +-PITCH_W): a staircase.
  for (var i = 0; i < 4; i++) {
    let fi = f32(i);
    let x = PITCH_W + TRACK_W + SEAT_W * (fi + 0.5);
    let y = CR - 0.03 + SEAT_H * (fi + 0.5);
    d = smin(d, sdBox(p - vec3f(x, y, 0.0), vec3f(SEAT_W * 0.5, SEAT_H * 0.5, PITCH_L + TRACK_W)), 0.02);
    d = smin(d, sdBox(p - vec3f(-x, y, 0.0), vec3f(SEAT_W * 0.5, SEAT_H * 0.5, PITCH_L + TRACK_W)), 0.02);
  }

  // Goal frames (posts + crossbar) and a wire net at each END (z = +-PITCH_L).
  for (var g = 0; g < 2; g++) {
    let s = f32(g) * 2.0 - 1.0;
    let gx = PITCH_W * 0.55;
    let gy = CR - 0.02 + POST_H;
    for (var pi = 0; pi < 2; pi++) {
      let px = f32(pi) * 2.0 - 1.0;
      let post = sdRoundBox(p - vec3f(px * gx, CR - 0.02 + POST_H * 0.5, s * PITCH_L),
                            vec3f(POST_R, POST_H * 0.5, POST_R), POST_R * 0.5);
      d = min(d, post);
    }
    let bar = sdRoundBox(p - vec3f(0.0, gy, s * PITCH_L),
                         vec3f(gx, POST_R, POST_R), POST_R * 0.5);
    d = min(d, bar);

    let netZ = s * (PITCH_L + 0.012);
    for (var vw = -3; vw <= 3; vw++) {
      let wx = f32(vw) / 3.0 * gx;
      d = min(d, sdRoundBox(p - vec3f(wx, CR - 0.02 + POST_H * 0.5, netZ),
                            vec3f(0.004, POST_H * 0.5, 0.004), 0.002));
    }
    for (var hw = 1; hw <= 4; hw++) {
      let wy = CR - 0.02 + POST_H * f32(hw) / 4.0;
      d = min(d, sdRoundBox(p - vec3f(0.0, wy, netZ),
                            vec3f(gx, 0.004, 0.004), 0.002));
    }
  }

  // Four floodlight poles at the pitch corners, with a bright head.
  for (var c = 0; c < 4; c++) {
    let sx = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let base = vec3f(sx * (PITCH_W + TRACK_W + 0.04), CR - 0.04, sz * (PITCH_L + TRACK_W + 0.04));
    let pole = sdCylinder(p - (base + vec3f(0.0, POLE_H * 0.5, 0.0)), POLE_R, POLE_H * 0.5);
    d = min(d, pole);
    let head = sdSphere(p - (base + vec3f(0.0, POLE_H, 0.0)), HEAD_R);
    d = min(d, head);
  }

  // The football.
  d = min(d, sdSphere(p - footballPos(), FB_R));

  return d;
}

fn bodyBound() -> f32 {
  return CR + POLE_H + HEAD_R + 0.2;
}

fn calcNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0015;
  return normalize(
      e.xyy * mapBody(p + e.xyy) + e.yyx * mapBody(p + e.yyx)
    + e.yxy * mapBody(p + e.yxy) + e.xxx * mapBody(p + e.xxx));
}

// ---- shading --------------------------------------------------------------

/// Soft shadow from the football only. Returns 0 (shadowed) .. 1 (lit).
fn footballShadow(p : vec3f, l : vec3f, lightDist : f32) -> f32 {
  let fp = footballPos();
  let L = fp - p;
  let tca = dot(L, l);
  let d2 = dot(L, L) - tca * tca;
  if (tca < 0.0 || tca > lightDist) { return 1.0; }
  return smoothstep(FB_R * FB_R, FB_R * FB_R * 0.25, d2);
}

/// The four corner floodlights, aligned with the poles.
fn cornerSpotlights(p : vec3f, n : vec3f) -> vec3f {
  var acc = vec3f(0.0);
  let centre = vec3f(0.0, CR, 0.0);
  for (var c = 0; c < 4; c++) {
    let sx = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let lightPos = vec3f(sx * (PITCH_W + TRACK_W + 0.04), CR + POLE_H,
                         sz * (PITCH_L + TRACK_W + 0.04));
    let L = lightPos - p;
    let d = length(L);
    let l = L / d;
    let att = 1.0 / (1.0 + 0.30 * d * d);
    let aim = normalize(centre - lightPos);
    let spot = smoothstep(0.30, 0.80, dot(l, aim));
    let diff = max(dot(n, l), 0.0);
    let shadow = footballShadow(p, l, d);
    acc += vec3f(1.0, 0.97, 0.88) * att * spot * diff * shadow;
  }
  return acc;
}

/// A patchy panel pattern for the football (pentagon-ish blobs).
fn soccerPattern(n : vec3f) -> f32 {
  let p = n * 5.0;
  let id = floor(p);
  let f = fract(p) - 0.5;
  let blob = smoothstep(0.45, 0.18, length(f));
  return step(0.62, hash13(id)) * blob;
}

/// The pitch markings, drawn as white lines in the material.
fn pitchLines(x : f32, z : f32) -> f32 {
  let bx = abs(x) - PITCH_W;
  let bz = abs(z) - PITCH_L;
  let boundary = select(0.0, 1.0, max(abs(bx), abs(bz)) < LINE_W);
  let halfway = select(0.0, 1.0, abs(z) < LINE_W);
  let circle = select(0.0, 1.0, abs(length(vec2f(x, z)) - 0.06) < LINE_W);
  let spot = select(0.0, 1.0, length(vec2f(x, z)) < LINE_W);
  let penL = PITCH_L - 0.13;
  let penW = PITCH_W * 0.55;
  let inPen = abs(x) < penW && abs(z) > penL && abs(z) < PITCH_L;
  let penEdge = select(0.0, 1.0, inPen && (abs(abs(z) - penL) < LINE_W
                         || abs(abs(x) - penW) < LINE_W || abs(abs(z) - PITCH_L) < LINE_W));
  return max(boundary, max(max(halfway, circle), max(spot, penEdge)));
}

/// The material, decided by WHERE the hit landed.
fn material(p : vec3f) -> vec3f {
  let fp = footballPos();
  let dir = normalize(p + 1e-6);

  // football
  if (length(p - fp) < FB_R * 1.4) {
    let n = (p - fp) / FB_R;
    return mix(vec3f(0.97), vec3f(0.10, 0.10, 0.12), soccerPattern(n));
  }

  // floodlight heads: emissive warm white
  for (var c = 0; c < 4; c++) {
    let sx = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let head = vec3f(sx * (PITCH_W + TRACK_W + 0.04), CR + POLE_H,
                     sz * (PITCH_L + TRACK_W + 0.04));
    if (length(p - head) < HEAD_R * 1.4) { return vec3f(1.0, 0.97, 0.88); }
  }

  // goal frame + net: white
  let onGoal = abs(abs(p.z) - PITCH_L) < 0.02 && abs(p.x) < PITCH_W * 0.6
               && p.y < CR + POST_H + 0.02 && p.y > CR - 0.06;
  if (onGoal) { return vec3f(0.9, 0.92, 0.95); }

  // floodlight poles: dark grey
  for (var c2 = 0; c2 < 4; c2++) {
    let sx = f32(c2 & 1) * 2.0 - 1.0;
    let sz = f32((c2 >> 1) & 1) * 2.0 - 1.0;
    let base = vec3f(sx * (PITCH_W + TRACK_W + 0.04), CR - 0.04, sz * (PITCH_L + TRACK_W + 0.04));
    if (length((p - base).xz) < POLE_R * 1.5 && p.y > CR - 0.06 && p.y < CR + POLE_H) {
      return vec3f(0.25, 0.26, 0.28);
    }
  }

  // grandstands: pale stone, on the +-X sides beyond the track
  if (abs(p.x) > PITCH_W + TRACK_W + SEAT_W * 0.3 && abs(p.z) < PITCH_L + TRACK_W
      && p.y > CR - 0.06 && p.y < CR + SEAT_H * 4.0) {
    return vec3f(0.62, 0.60, 0.55);
  }

  // The pitch: a region ON the sphere surface, green with white lines.
  if (abs(p.x) < PITCH_W + 0.02 && abs(p.z) < PITCH_L + 0.02) {
    let line = pitchLines(p.x, p.z);
    let base = mix(vec3f(0.10, 0.42, 0.16), vec3f(0.16, 0.52, 0.22), fbm3(dir * 10.0));
    return mix(base, vec3f(0.92, 0.94, 0.96), line);
  }

  // The running track: a rectangular ring around the pitch.
  let inOuter = abs(p.x) < PITCH_W + TRACK_W && abs(p.z) < PITCH_L + TRACK_W;
  let inPitch = abs(p.x) < PITCH_W && abs(p.z) < PITCH_L;
  if (inOuter && !inPitch) {
    return vec3f(0.55, 0.28, 0.20);
  }

  // The rest of the planet: grass.
  return grassColour(p, dir);
}

fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let n = calcNormal(p);
  let v = -rd;
  let base = material(p);

  let sunSh = footballShadow(p, SUN1_DIR, 1e3);
  let diff = max(dot(n, SUN1_DIR), 0.0) * sunSh;

  var col = base * (vec3f(0.16) + SUN1_COL * diff);
  col += base * cornerSpotlights(p, n);

  let h = normalize(SUN1_DIR + v);
  let spec = pow(max(dot(n, h), 0.0), 48.0);
  col += spec * SUN1_COL * 0.6;

  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  col += rim * vec3f(0.55, 0.70, 1.0) * 0.30;

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
