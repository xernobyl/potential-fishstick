// ---------------------------------------------------------------------------
// Engine plumes and RCS puffs — as billboards in the ADDITIVE layer.
//
// These used to be accumulated inside the scene march (`col += shipJets(...)` in raymarch.wgsl), and
// that stopped being viable the moment the hull became a rasterised mesh. The temporal resolve picks
// solid-over-marched by REPLACING: `readTap` in taa.wgsl does `t.col = sc.rgb` when the solid layer is
// nearer, so every pixel any solid mesh covers discards the marched colour — and the plume went with
// it. The glow was cut off hard at the hull's silhouette instead of blooming over the nozzle, which
// reads as the exhaust starting somewhere behind the ship.
//
// The contrails never had this problem because they were always an additive pass drawn AFTER the
// resolve, soft-testing against the resolved depth and adding on top. The plumes are the same kind of
// thing — additive glow with soft occlusion — and this puts them in the same place. It is the fix the
// architecture already had, applied to the one element still on the wrong side of it.
//
// A BILLBOARD PER BLOB, rather than a fullscreen pass re-marching the ray. Every term the marched
// version summed was a Gaussian at a known world position with a known radius, which is exactly what a
// camera-facing quad with a Gaussian falloff is. The difference is that the old form integrated the
// Gaussian ALONG the ray and this evaluates it across the quad; for blobs this small the two are
// visually the same, and this one needs no ray reconstruction, no depth-tag arithmetic and no
// assumptions about which grid the additive layer happens to be on.
//
// Everything is still derived from the frame uniform — position, orientation, throttle, and the
// angular acceleration the physics actually produced — so there is no buffer and nothing to upload.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "volumetric.wgsl"

// Binding ONE, not zero: the layout keeps the scene texture where every other additive effect has
// it, and simply omits the storage buffer this one does not use.
@group(1) @binding(1) var sceneTex : texture_2d<f32>;   // alpha = depth tag

const QUAD = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
);

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) uv    : vec2f,
  @location(1) tint  : vec3f,
  @location(2) viewZ : f32,
  @location(3) atmo  : vec3f,
};

fn collapse(out : ptr<function, VOut>) {
  (*out).pos = vec4f(0.0, 0.0, 2.0, 1.0);
  (*out).uv = vec2f(0.0);
  (*out).tint = vec3f(0.0);
  (*out).viewZ = 0.0;
  (*out).atmo = vec3f(1.0);
}

/// How far out the quad reaches, as a multiple of the blob's Gaussian radius. At 2.2 the falloff has
/// dropped to ~7e-4 by the edge, so the quad's own boundary is never visible.
const REACH : f32 = 2.2;

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  var out : VOut;

  let q = frame.shipRot;
  let sp = frame.shipPos.xyz;
  let throttle = frame.shipPos.w;
  let ang = frame.shipJet.xyz;
  let rev = frame.shipJet.w;

  let mains = 2u * u32(SHIP_JET_LEN);

  var local : vec3f;
  var radius : f32;
  var tint : vec3f;

  if (ii < mains) {
    // Two plumes, sampled down their length so each reads as a cone rather than a ball. The near
    // cells are small and white-hot, the far ones broad and dim.
    let side = select(-SHIP_NACELLE_X, SHIP_NACELLE_X, (ii / u32(SHIP_JET_LEN)) == 0u);
    let fk = f32(ii % u32(SHIP_JET_LEN));
    let t = fk / f32(SHIP_JET_LEN);
    let fade = throttle * max(0.0, 1.0 - t);
    if (fade <= 0.001) { collapse(&out); return out; }
    local = vec3f(side, SHIP_NACELLE_Y, SHIP_NOZZLE_Z - fk * 0.085 * (0.35 + throttle));
    radius = (0.030 + fk * 0.019) * (0.5 + 0.8 * throttle);
    tint = mix(vec3f(1.9, 1.5, 1.1), SHIP_JET_COL, t) * fade;
  } else if (ii < mains + 2u) {
    // Reverse: a short pair out of the FRONT, so backing off is visible.
    if (rev <= 0.01) { collapse(&out); return out; }
    let side = select(-0.10, 0.10, (ii - mains) == 0u);
    local = vec3f(side, 0.0, 0.40);
    radius = 0.032;
    tint = vec3f(1.3, 0.85, 0.55) * rev;
  } else {
    // RCS: one puff per axis pair, on the side that would produce the MEASURED torque — so the
    // thrusters fire to arrest a rotation as well as to start one.
    let a = ii - mains - 2u;
    let mag = ang[a];
    if (abs(mag) < 0.05) { collapse(&out); return out; }
    let s = sign(mag);
    // Pitch fires at the tail, yaw at the wingtips, roll at opposite tips.
    if (a == 0u) { local = vec3f(0.0, 0.10 * s, -0.34); }
    else if (a == 1u) { local = vec3f(0.46 * s, 0.0, 0.10); }
    else { local = vec3f(0.46 * s, 0.06, -0.24); }
    radius = 0.028;
    tint = SHIP_RCS_COL * clamp(abs(mag) * SHIP_RCS_GAIN, 0.0, 1.4);
  }

  // Authored in body units, like everything else bolted to this hull.
  let centre = sp + qrotate(q, local * SHIP_SCALE);
  let size = radius * SHIP_SCALE * REACH;

  // Camera-facing billboard, from the view direction alone — no per-blob basis to build.
  let toCam = normalize(frame.camPos.xyz - centre);
  var right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam));
  if (abs(toCam.y) > 0.99) { right = vec3f(1.0, 0.0, 0.0); }
  let up = cross(toCam, right);

  let c = QUAD[vi % 6u];
  let wp = centre + (right * c.x + up * c.y) * size;

  out.pos = frame.viewProj * vec4f(wp, 1.0);
  out.uv = c;
  out.tint = tint;
  out.viewZ = length(wp - frame.camPos.xyz);
  // Extinction only, as every additive layer does it — see volTransmittance.
  let toEye = wp - frame.camPos.xyz;
  let dist = length(toEye);
  out.atmo = volTransmittance(frame.camPos.xyz, toEye / max(dist, 1e-6), dist);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  // The same Gaussian the marched version used, with the quad's own extent folded in: `uv` runs to 1
  // at the edge and the quad is REACH radii across, so `|uv| * REACH` is the distance in radii.
  let d = length(in.uv) * REACH;
  let g = exp(-d * d * 1.5);
  if (g <= 0.002) { discard; }

  // Soft occlusion against the scene, as the contrail and the sparks do it: dissolve rather than clip,
  // so a plume passing behind the hull or a ring fades instead of popping.
  let sceneW = textureLoad(sceneTex, toAccumPx(in.pos.xy), 0).a;
  let vis = smoothstep(0.0, 0.06, tagDepth(sceneW) - in.viewZ);

  let a = g * vis;
  return vec4f(in.tint * a * SHIP_JET_GAIN * in.atmo, a);
}
