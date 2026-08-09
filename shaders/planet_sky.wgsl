// ---------------------------------------------------------------------------
// Sky background: fullscreen pass that fills sceneRaw with the starfield,
// nebula, and sharp stars, attenuated by the planet's atmosphere.
// Writes alpha = -1 (TAG_BG). The planet draw pass then loads this and
// overwrites body pixels with alpha > 0.
//
// The atmosphere integration here matches what the raymarch scene does for
// rays that miss the body: full atmospheric column from camera to infinity.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sky.wgsl"
//!include "volumetric.wgsl"
//!include "fullscreen.wgsl"

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let px = uv * frame.res.xy;
  let rd = cameraRay(px).d;

  // Starfield colour before atmosphere
  let bg = bgNebula(rd) + bgSharp(rd);

  // Full atmospheric column for a sky ray (tMax = 1e10 → camera to infinity).
  // Transmittance dims the starfield; in-scatter adds the limb glow.
  let atmo = volumetric(frame.camPos.xyz, rd, 1e10, px);
  let col = bg * atmo.transmittance + atmo.inScatter;

  return vec4f(col, -1.0);
}
