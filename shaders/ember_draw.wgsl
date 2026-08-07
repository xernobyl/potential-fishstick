// ---------------------------------------------------------------------------
// Ember billboards.
//
// Camera-facing quads built entirely in the vertex shader from the particle
// buffer — no vertex or index buffer bound, and no per-instance upload. The
// instance count comes from the simulation via drawIndirect.
//
// Occlusion without a depth buffer: the scene's linear depth is already in the
// accumulation target's alpha, so each fragment compares its own view depth
// against it and fades. That is a *soft* particle — it dissolves into the
// surface rather than clipping against it — and it costs one texture tap.
//
// These are drawn into their own additive target rather than into the
// accumulation buffer, because they are dynamic: accumulating them would trail
// every mote across the screen.
// ---------------------------------------------------------------------------

//!include "common.wgsl"

struct Ember {
  pos   : vec3f,
  size  : f32,
  tint  : vec3f,
  alpha : f32,
};

@group(1) @binding(0) var<storage, read> embers : array<Ember>;
@group(1) @binding(1) var<storage, read> liveList : array<u32>;
@group(1) @binding(2) var spriteTex : texture_2d<f32>;
@group(1) @binding(3) var spriteSamp : sampler;
@group(1) @binding(4) var sceneTex : texture_2d<f32>;   // alpha = linear depth tag

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) uv    : vec2f,
  @location(1) tint  : vec3f,
  @location(2) alpha : f32,
  @location(3) viewZ : f32,
};

// Two triangles, as a corner lookup. Cheaper than an index buffer and keeps the
// draw call free of any bound geometry.
const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
  vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  let e = embers[liveList[ii]];
  let corner = CORNERS[vi];

  // The quad still expands along the world-space basis — that is what makes it
  // face the camera — but it is PROJECTED by the shared matrix, so the particles
  // land exactly where the raymarch thinks the scene is.
  let world = e.pos + (frame.camRight.xyz * corner.x + frame.camUp.xyz * corner.y) * e.size;
  let rel = world - frame.camPos.xyz;
  let clip = frame.viewProj * vec4f(world, 1.0);

  var out : VOut;
  if (clip.w <= 1e-3) {
    // Behind the camera: collapse it offscreen rather than letting the divide
    // produce a NaN that would smear a triangle across the frame.
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2f(0.0);
    out.tint = vec3f(0.0);
    out.alpha = 0.0;
    out.viewZ = 0.0;
    return out;
  }

  out.pos = clip;
  out.uv = corner * 0.5 + 0.5;
  out.tint = e.tint;
  out.alpha = e.alpha;
  out.viewZ = length(rel);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  let sprite = textureSample(spriteTex, spriteSamp, in.uv);

  // Soft occlusion against the scene's stored depth.
  let px = vec2i(in.pos.xy);
  let sceneW = textureLoad(sceneTex, toAccumPx(vec2f(px)), 0).a;
  let sceneDepth = tagDepth(sceneW);
  // Fade over a short band instead of a hard test, so a mote passing behind a
  // lobe dissolves rather than popping off.
  let vis = smoothstep(0.0, 0.06, sceneDepth - in.viewZ);

  let a = sprite.a * in.alpha * vis;
  // Premultiplied additive: the pipeline blends with ONE, so fold alpha in here.
  return vec4f(sprite.rgb * in.tint * in.alpha * vis, a);
}
