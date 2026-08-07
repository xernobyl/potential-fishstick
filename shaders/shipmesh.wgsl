// ---------------------------------------------------------------------------
// The ship's hull: a generated triangle mesh, rasterised into the solid layer.
//
// This replaces a marched SDF. The shape is now data (see scene/ship_sdf.js), dual contoured on the CPU
// at startup, and drawn like any other generated mesh — which buys four things the marched version could
// not have. It costs one draw call instead of a per-pixel sphere-trace with its own step loop and bound
// test. It occludes against hardware depth rather than by threading `tmax` through the march. It gets an
// EXACT motion vector from its own rigid transform, where the marched version derived one per pixel from
// a hit point. And it is culled, both by frustum and by back face, neither of which a marched primitive
// can be.
//
// SHADING IS DELIBERATELY CONSTANT. One albedo, one roughness, one f0 for the whole hull — no panel
// lines, no grime, no per-face variation. That is the brief for now, and it also isolates the geometry:
// anything that looks wrong here is the MESH, not a texture hiding or inventing a feature. The vertex
// front end and the lighting are the ones the rings already use, so this file is the material and
// nothing else.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "mesh_vertex.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "explosion.wgsl"

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp        : vec3f,
  @location(1) wn        : vec3f,
  @location(2) viewZ     : f32,
  @location(3) prevClip  : vec4f,
  @location(4) prevViewZ : f32,
};

struct FOut {
  @location(0) colour : vec4f,   // rgb, alpha = linear view distance
  @location(1) motion : vec4f,   // xy pixel delta, z previous distance, w ownership
};

@vertex
fn vs(v : MeshVertex) -> VOut {
  // The hull is rigid, so its transform is the ship's own pose — position and orientation quaternion,
  // both already in the frame uniform because the marched version needed them too. The PREVIOUS pose is
  // there for the same reason, and is what makes the motion vector exact rather than reconstructed.
  let now = meshRigidFromQuat(frame.shipPos.xyz, frame.shipRot);
  let prev = meshRigidFromQuat(frame.prevShipPos.xyz, frame.prevShipRot);
  let x = meshXform(v, now, prev);

  var out : VOut;
  out.pos = x.clip;
  out.wp = x.wp;
  out.wn = x.wn;
  out.viewZ = x.viewZ;
  out.prevClip = x.prevClip;
  out.prevViewZ = x.prevViewZ;
  return out;
}

@fragment
fn fs(in : VOut) -> FOut {
  // Renormalised: interpolation across a triangle shortens a normal, and a short normal biases every
  // dot product in the BRDF toward grazing.
  let N = normalize(in.wn);
  let V = normalize(frame.camPos.xyz - in.wp);

  let alb = SHIPM_ALBEDO;
  let rough = SHIPM_ROUGH;
  let f0 = mix(vec3f(0.04), SHIPM_ALBEDO, SHIPM_METAL);

  var col = sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0)
          + blastLight(in.wp, N, V, alb, rough, f0);

  // The sky as the environment term, with no marched reflection. The rings march one because they are
  // polished enough that the body dominates what they show; this hull is rough enough that a single
  // mirror ray would be both wrong and noisy, which is the same reasoning that fades the rings' march
  // out as their roughness rises — here it starts faded.
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  col += background(reflect(-V, N)) * fresnelSchlickRough(NoV, f0, rough) * SHIPM_ENV;

  // The planet's core throws real light at this range, and leaving it out makes the hull look pasted on
  // against a bright body. Falls off with distance from the origin, which is where the core is.
  let toCore = -normalize(in.wp);
  col += SHIPM_CORE * max(dot(N, toCore), 0.0) / max(dot(in.wp, in.wp), 1.0);

  var out : FOut;
  out.colour = vec4f(max(col, vec3f(0.0)), max(in.viewZ, 1e-4));

  if (in.prevClip.w > 1e-4) {
    let prevPx = uvToPixel(ndcToUV(in.prevClip.xy / in.prevClip.w));
    // The jitter is added BACK, for the same reason the rings do it: `in.pos.xy` is this fragment's
    // pixel centre, but the surface it covers is the one whose UNJITTERED projection is centre + jitter,
    // and the history is indexed by centres. `prevClip` is already unjittered.
    out.motion = vec4f(prevPx - (in.pos.xy + frame.jitter.xy), in.prevViewZ, in.viewZ);
  } else {
    out.motion = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  }
  return out;
}
