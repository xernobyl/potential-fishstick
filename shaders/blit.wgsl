// ---------------------------------------------------------------------------
// Minimal textured blit. Used to build the ember sprite's mip chain: WebGPU has
// no generateMipmap, and for a one-off 64px texture a tiny blit chain is far
// less machinery than a compute reducer.
// ---------------------------------------------------------------------------

//!include "fullscreen.wgsl"

@group(0) @binding(0) var srcTex  : texture_2d<f32>;
@group(0) @binding(1) var srcSamp : sampler;

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  return textureSampleLevel(srcTex, srcSamp, uv, 0.0);
}
