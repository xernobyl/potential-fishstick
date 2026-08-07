// ---------------------------------------------------------------------------
// Reflectance. Cook-Torrance GGX, and nothing that knows what it is shading.
//
// Extracted from shade.wgsl deliberately. These are generic and every surface in
// the scene wants them — the body and the satellites — but while they
// lived beside the gummy's material code, lighting ANY new surface meant including
// the whole signed-distance field to get at them. That is a dependency on the
// wrong thing, and it costs compile time and register pressure in shaders that
// never march anything.
// ---------------------------------------------------------------------------

//!include "common.wgsl"

fn distributionGGX(NoH : f32, rough : f32) -> f32 {
  let a2 = rough * rough * rough * rough;      // alpha^2, alpha = rough^2
  let d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn geometrySmith(NoV : f32, NoL : f32, rough : f32) -> f32 {
  let k = rough * rough * 0.5;                 // Schlick-GGX, direct-light remap
  return (NoV / (NoV * (1.0 - k) + k)) * (NoL / (NoL * (1.0 - k) + k));
}
fn fresnelSchlick(u : f32, f0 : vec3f) -> vec3f {
  return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}
fn fresnelSchlickRough(u : f32, f0 : vec3f, rough : f32) -> vec3f {
  return f0 + (max(vec3f(1.0 - rough), f0) - f0) * pow(1.0 - u, 5.0);
}

/// iq's analytic sphere ambient occlusion.

/// Direct light from one sun, including its own soft shadow.
fn sunLight(N : vec3f, V : vec3f, L : vec3f, lightCol : vec3f,
            alb : vec3f, rough : f32, f0 : vec3f, sha : f32) -> vec3f {
  let NoL = clamp(dot(N, L), 0.0, 1.0);
  if (NoL <= 0.0 || sha <= 0.0) { return vec3f(0.0); }   // back-facing: skip the BRDF
  let H = normalize(L + V);
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  let NoH = clamp(dot(N, H), 0.0, 1.0);
  let VoH = clamp(dot(V, H), 0.0, 1.0);
  let F = fresnelSchlick(VoH, f0);
  let spec = distributionGGX(NoH, rough) * geometrySmith(NoV, NoL, rough) * F
           / (4.0 * NoV * NoL + 1e-4);
  return ((1.0 - F) * alb / PI + spec) * lightCol * NoL * sha;
}
