// ---------------------------------------------------------------------------
// Sky background: fullscreen pass that fills sceneRaw with the starfield,
// nebula, and sharp stars. Writes alpha = -1 (TAG_BG). The planet draw pass
// then loads this and overwrites body pixels with alpha > 0.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "fullscreen.wgsl"

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let px = uv * frame.res.xy;
  let rd = cameraRay(px).d;
  let col = bgNebula(rd) + bgSharp(rd);
  return vec4f(col, -1.0);
}
