// ---------------------------------------------------------------------------
// Chibi planet: a small raymarched world — a football pitch, a running track,
// tiered grandstands and a bouncing football. Neo-retro cartoon PBR.
//
// Everything is one self-contained field so the scene is a drop-in that needs
// no march changes elsewhere. It reuses the shared sky (`background`) and the
// sun directions/colours from tuning, then layers four corner spotlights with
// soft falloff and a soft sphere shadow from the football.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"

// ---- geometry constants ---------------------------------------------------

const CR       : f32 = 1.0;     // planet radius
const FIELD_L  : f32 = 0.33;    // pitch half-length — 1/3 of the planet's diameter
const FIELD_W  : f32 = 0.21;    // pitch half-width
const TRACK_W  : f32 = 0.08;    // running-track width
const SEAT_W   : f32 = 0.14;    // grandstand depth
const SEAT_H   : f32 = 0.05;    // step height per tier
const FB_R     : f32 = 0.055;   // football radius
const POST_H   : f32 = 0.09;    // goalpost height
const POST_R   : f32 = 0.008;   // goalpost thickness

// ---- primitives (iq formulations) -----------------------------------------

fn sdSphere(p : vec3f, r : f32) -> f32 { return length(p) - r; }

fn sdBox(p : vec3f, b : vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdRoundBox(p : vec3f, b : vec3f, r : f32) -> f32 {
  let q = abs(p) - b + vec3f(r);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn smin(a : f32, b : f32, k : f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

/// A flat rounded-rectangle RING (the running track): the shell between two rounded boxes.
fn sdRingFlat(p : vec3f, halfExtents : vec3f, thickness : f32, r : f32) -> f32 {
  let outer = sdRoundBox(p, halfExtents, r);
  let inner = sdRoundBox(p, halfExtents - vec3f(thickness), r);
  return max(-inner, outer);
}

// ---- the world ------------------------------------------------------------

fn footballPos() -> vec3f {
  let t = frame.camPos.w;                                   // time, seconds
  let bounce = abs(sin(t * 2.1));                           // 0..1
  return vec3f(0.14 * sin(t * 1.1), CR + FB_R + 0.16 * bounce, 0.10 * cos(t * 0.83));
}

fn mapBody(p : vec3f) -> f32 {
  // The planet itself.
  var d = sdSphere(p, CR);

  // The pitch: a flat plateau tangent to the north pole.
  let field = sdBox(p - vec3f(0.0, CR - 0.02, 0.0), vec3f(FIELD_L, 0.02, FIELD_W));
  d = smin(d, field, 0.05);

  // The running track: a flat oval ring around the pitch, at the sphere surface.
  let track = sdRingFlat(p - vec3f(0.0, CR - 0.012, 0.0),
                         vec3f(FIELD_L + TRACK_W * 0.5, 0.012, FIELD_W + TRACK_W * 0.5),
                         TRACK_W, 0.02);
  d = smin(d, track, 0.03);

  // Grandstands: three tiers on each LONG side (traditional terrace, not a bowl).
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    let seat = sdBox(p - vec3f(FIELD_L + TRACK_W + SEAT_W * 0.5, CR - 0.01 + SEAT_H * (fi + 0.5), 0.0),
                     vec3f(SEAT_W * 0.5, SEAT_H * 0.5, FIELD_W + TRACK_W));
    d = smin(d, seat, 0.02);
    let seat2 = sdBox(p - vec3f(-(FIELD_L + TRACK_W + SEAT_W * 0.5), CR - 0.01 + SEAT_H * (fi + 0.5), 0.0),
                      vec3f(SEAT_W * 0.5, SEAT_H * 0.5, FIELD_W + TRACK_W));
    d = smin(d, seat2, 0.02);
  }

  // Goal frames: two posts and a crossbar at each end of the pitch.
  for (var g = 0; g < 2; g++) {
    let s = f32(g) * 2.0 - 1.0;                              // +1 / -1 end
    let gx = FIELD_L * 0.45;                                 // goal mouth half-width
    let gy = CR - 0.02 + POST_H;                             // crossbar height
    for (var p = 0; p < 2; p++) {
      let px = f32(p) * 2.0 - 1.0;
      let post = sdRoundBox(p - vec3f(px * gx, CR - 0.02 + POST_H * 0.5, s * FIELD_W),
                            vec3f(POST_R, POST_H * 0.5, POST_R), POST_R * 0.5);
      d = min(d, post);
    }
    let bar = sdRoundBox(p - vec3f(0.0, gy, s * FIELD_W),
                         vec3f(gx, POST_R, POST_R), POST_R * 0.5);
    d = min(d, bar);
  }

  // The football: a small sphere that bounces on the pitch.
  let fb = sdSphere(p - footballPos(), FB_R);
  d = min(d, fb);

  return d;
}

fn bodyBound() -> f32 {
  // Planet radius plus the tallest feature (a grandstand tier) and the football's apex.
  return CR + SEAT_H * 3.0 + 0.25;
}

fn calcNormal(p : vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0015;
  return normalize(
      e.xyy * mapBody(p + e.xyy) + e.yyx * mapBody(p + e.yyx)
    + e.yxy * mapBody(p + e.yxy) + e.xxx * mapBody(p + e.xxx));
}

// ---- shading --------------------------------------------------------------

/// Soft shadow from the football only — cheap, analytic, and the one moving thing that
/// occludes the lights. Returns 0 (shadowed) .. 1 (lit).
fn footballShadow(p : vec3f, l : vec3f, lightDist : f32) -> f32 {
  let fp = footballPos();
  let L = fp - p;
  let tca = dot(L, l);
  let d2 = dot(L, L) - tca * tca;
  if (tca < 0.0 || tca > lightDist) { return 1.0; }
  return smoothstep(FB_R * FB_R, FB_R * FB_R * 0.25, d2);
}

/// The four corner spotlights: poles above the pitch corners, pointing down at the centre.
fn cornerSpotlights(p : vec3f, n : vec3f) -> vec3f {
  var acc = vec3f(0.0);
  let centre = vec3f(0.0, CR, 0.0);
  for (var c = 0; c < 4; c++) {
    let sx = f32(c & 1) * 2.0 - 1.0;
    let sz = f32((c >> 1) & 1) * 2.0 - 1.0;
    let lightPos = vec3f(sx * (FIELD_L + 0.06), CR + 0.42, sz * (FIELD_W + 0.06));
    let L = lightPos - p;
    let d = length(L);
    let l = L / d;
    let att = 1.0 / (1.0 + 0.35 * d * d);            // inverse-square-ish, tunable
    let aim = normalize(centre - lightPos);           // cone axis points at the pitch centre
    let spot = smoothstep(0.35, 0.85, dot(l, aim));   // soft cone edge
    let diff = max(dot(n, l), 0.0);
    let shadow = footballShadow(p, l, d);
    acc += vec3f(1.0, 0.96, 0.86) * att * spot * diff * shadow;
  }
  return acc;
}

/// The cartoon material colour, decided by WHERE the hit landed (a cheap, deliberate fake).
fn material(p : vec3f) -> vec3f {
  let fp = footballPos();
  if (length(p - fp) < FB_R * 1.4) { return vec3f(0.96, 0.97, 1.0); }   // football: white
  // pitch
  if (p.y > CR - 0.03 && abs(p.x) < FIELD_L + 0.02 && abs(p.z) < FIELD_W + 0.02) {
    // alternating mowed stripes, a cartoon touch
    let stripe = step(0.0, sin(p.x * 40.0));
    return mix(vec3f(0.10, 0.42, 0.16), vec3f(0.16, 0.52, 0.22), stripe);
  }
  // running track
  let trackOuter = max(FIELD_L, FIELD_W) + TRACK_W;
  if (p.y > CR - 0.03 && max(abs(p.x), abs(p.z)) < trackOuter + 0.05
      && !(abs(p.x) < FIELD_L && abs(p.z) < FIELD_W)) {
    return vec3f(0.55, 0.28, 0.20);                                   // cinder red
  }
  // grandstands
  if (p.y > CR - 0.03 && (abs(p.x) > FIELD_L + TRACK_W * 0.5)) {
    return vec3f(0.62, 0.60, 0.55);                                   // pale stone
  }
  // the rest of the planet: grass
  return vec3f(0.22, 0.50, 0.26);
}

fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let n = calcNormal(p);
  let v = -rd;
  let base = material(p);

  // Sun key light, with a soft shadow from the football.
  let sunSh = footballShadow(p, SUN1_DIR, 1e3);
  let diff = max(dot(n, SUN1_DIR), 0.0) * sunSh;

  // Ambient: keep it lifted so the cartoon reads bright, not muddy.
  var col = base * (vec3f(0.16) + SUN1_COL * diff);

  // Corner spotlights.
  col += base * cornerSpotlights(p, n);

  // Specular (Blinn-Phong) — a crisp, slightly retro highlight.
  let h = normalize(SUN1_DIR + v);
  let spec = pow(max(dot(n, h), 0.0), 48.0);
  col += spec * SUN1_COL * 0.6;

  // Rim / outline, for the cartoon silhouette.
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  col += rim * vec3f(0.55, 0.70, 1.0) * 0.35;

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
