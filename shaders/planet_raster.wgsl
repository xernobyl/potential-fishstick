// ---------------------------------------------------------------------------
// Planet rasterization — full shading, motion vectors, 6 debug modes.
//
// Debug modes (cycled via uniform.devmode.x):
//   0 = Lit          (production: shadeBody + volumetric)
//   1 = Motion       (motion-vector RGB: |velUv.x|*20, |velUv.y|*20, |velUv|*50)
//   2 = SDF parity   (green=on surface, red=outside, blue=inside — mapBody check)
//   3 = Grid         (fract worldPos × 1.5 shows vertex density)
//   4 = Normals      (calcNormal at worldPos, remapped to 0..1)
//   5 = Winding      (face normal vs SDF gradient; red=inverted, green=correct)
//
// Motion vectors: camera-relative NDC delta, stored as second attachment.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "fibonacci.wgsl"
//!include "sdf.wgsl"
//!include "shade.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "volumetric.wgsl"
//!include "ring_geom.wgsl"
//!include "explosion.wgsl"

struct DrawUniform {
  cameraPos : vec4f,    // current camera
  prevCam   : vec4f,    // previous camera
  devmode   : vec4f,    // x = debug mode 0..4
  spare     : vec4f,
};

@group(1) @binding(0) var<storage, read> vertexData : array<f32>;
@group(1) @binding(1) var<uniform> du : DrawUniform;

// Injected so it agrees with the mesher in planet_surfacenets.wgsl (see SN_NORMAL in tuning.js).
// The vertex shader only reads the position; the normal sits at +3 (packed index or float32x3)
// and is re-derived analytically in the fragment stage for now.

/// Projection with zero translation (for camera-relative velocity)
fn projNoTrans(vp : mat4x4f) -> mat4x4f {
  var m = vp;
  m[3] = vec4f(0.0, 0.0, 0.0, 1.0);
  return m;
}

struct Varying {
  @builtin(position) clip    : vec4f,
  @location(0)       worldPos : vec3f,
  @location(1)       viewDir  : vec3f,
  @location(2)       viewDist : f32,
  @location(3)       px       : vec2f,
  @location(4)       velUv    : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> Varying {
  let base = vi * SN_VERTEX_STRIDE;
  let rel = vec3f(vertexData[base], vertexData[base + 1u], vertexData[base + 2u]);

  let wp = rel + du.cameraPos.xyz;
  let vd = normalize(du.cameraPos.xyz - wp);
  let cp = frame.viewProj * vec4f(wp, 1.0);
  let dt = length(wp - du.cameraPos.xyz);
  let px = uvToPixel(ndcToUV(cp.xy / cp.w));

  // Motion vector: camera-relative, unjittered projection, no translation
  let vpC = projNoTrans(frame.viewProj);
  let vpP = projNoTrans(frame.prevViewProj);
  let cc = vpC * vec4f(wp - du.cameraPos.xyz, 1.0);
  let cp2 = vpP * vec4f(wp - du.prevCam.xyz, 1.0);
  let nc = cc.xy / max(cc.w, 1e-6);
  let np = cp2.xy / max(cp2.w, 1e-6);
  let velUv = (nc - np) * vec2f(0.5, -0.5);

  var out : Varying;
  out.clip     = cp;
  out.worldPos = wp;
  out.viewDir  = vd;
  out.viewDist = dt;
  out.px       = px;
  out.velUv    = velUv;
  return out;
}

@fragment
fn fs(in : Varying) -> @location(0) vec4f {
  let mode = i32(du.devmode.x);

  if (mode == 1) {
    let s = length(in.velUv);
    return vec4f(abs(in.velUv.x) * 20.0, abs(in.velUv.y) * 20.0, s * 50.0, 1.0);
  }
  if (mode == 2) {
    let wp2 = in.worldPos;
    let d2 = mapBody(wp2);
    let t2 = abs(d2);
    var col = vec3f(0.2, 1.0, 0.2);
    if (d2 > 0.0) { col = mix(col, vec3f(1.0, 0.2, 0.2), clamp(t2 * 0.5, 0.0, 1.0)); }
    else          { col = mix(col, vec3f(0.2, 0.2, 1.0), clamp(t2 * 0.5, 0.0, 1.0)); }
    return vec4f(col, 1.0);
  }
  if (mode == 3) {
    return vec4f(abs(fract(in.worldPos * 1.5)), 1.0);
  }
  if (mode == 4) {
    let d3 = mapBody(in.worldPos);
    let n3 = calcNormal(in.worldPos);
    return vec4f(n3 * 0.5 + vec3f(0.5), 1.0);
  }
  // Mode 5: Winding diagnostic — face normal vs SDF gradient alignment
  if (mode == 5) {
    let faceN = normalize(cross(dpdx(in.worldPos), dpdy(in.worldPos)));
    let an = calcNormal(in.worldPos);
    let align = dot(faceN, an);
    if (align < 0.0) {
      return vec4f(1.0, 0.0, 0.0, 1.0); // RED = inverted/backwards
    }
    return vec4f(0.0, clamp(align, 0.0, 1.0), 0.0, 1.0); // GREEN = correct
  }

  // Mode 0: Production — tinted by face mask
  let wp = in.worldPos;
  let rd = -in.viewDir;
  let t  = in.viewDist;
  var col = shadeBody(wp, rd, t);
  if (!(col.r == col.r)) { col = vec3f(0.5, 0.3, 0.8); }
  col = clamp(col, vec3f(0.0), vec3f(100.0));
  let vol = volumetric(frame.camPos.xyz, rd, t, in.px);
  col = col * vol.transmittance + vol.inScatter;
  // Face mask tint: R=onlyX, G=onlyY, B=onlyZ, white=all
  let fm = u32(du.spare.x);
  if (fm != 0u) {
    let tint = vec3f(f32((fm & 1u) != 0u), f32((fm & 2u) != 0u), f32((fm & 4u) != 0u));
    col = mix(col, col * tint * 2.0, 0.7);
  }
  return vec4f(max(vec3f(0.0), col), t);
}
