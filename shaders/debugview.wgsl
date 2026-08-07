// ---------------------------------------------------------------------------
// One intermediate buffer, straight to the screen.
//
// A fullscreen sample with a display MODE, because the buffers this inspects do not mean the same
// kind of thing: `scene` and `ember` are linear HDR, `motion` is a signed pixel delta with two more
// channels packed beside it, and the accumulation's alpha is a distance tag. Showing any of them
// with the same mapping would make three of the four unreadable.
//
// The mode arrives in `balance.w`. It used to ride in `camRight.w`, which was NOT free: tilecull still
// read that slot as the aperture it once was, so opening the buffer viewer changed the tile culling.
// An inspector that alters what it inspects is worse than no inspector.
//
// UV rather than pixel indexing throughout: these buffers are at three different resolutions
// (render, display, and the additive layer's own), and uv is the one addressing that does not care.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "fullscreen.wgsl"

@group(1) @binding(0) var srcTex  : texture_2d<f32>;
@group(1) @binding(1) var srcSamp : sampler;

const MODE_HDR    : i32 = 0;   // linear HDR colour, Reinhard'd so the range is visible at all
const MODE_MOTION : i32 = 1;   // xy pixel delta: red right, green down, dark where none was written
const MODE_TAG    : i32 = 2;   // alpha as a distance tag: background, then near -> far
const MODE_ALPHA  : i32 = 3;   // alpha alone, as grey

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let s = textureSampleLevel(srcTex, srcSamp, uv, 0.0);
  let mode = i32(frame.balance.w + 0.5);
  var c = vec3f(0.0);

  if (mode == MODE_MOTION) {
    // A magnitude ramp plus a direction tint. The sentinel is enormous, so anything that did not
    // write a vector reads as flat dark rather than as saturated garbage.
    if (abs(s.x) < MOTION_NONE * 0.5) {
      let m = length(s.xy);
      c = vec3f(0.08, 0.10, 0.14) + vec3f(max(s.x, 0.0), max(s.y, 0.0), max(-s.x, 0.0)) * 0.25
        + vec3f(m * 0.06);
    } else {
      c = vec3f(0.02, 0.0, 0.03);          // no motion stored
    }
  } else if (mode == MODE_TAG) {
    // Distance, banded so the eye can read depth order rather than guessing at a gradient.
    if (isBackground(s.a)) {
      c = vec3f(0.05, 0.02, 0.10);
    } else {
      let d = s.a / max(frame.march.y * 40.0, 1.0);
      c = mix(vec3f(0.1, 0.9, 0.7), vec3f(0.9, 0.2, 0.5), clamp(d, 0.0, 1.0));
      c *= 0.55 + 0.45 * fract(s.a * 2.0);   // contour bands
    }
  } else if (mode == MODE_ALPHA) {
    c = vec3f(s.a);
  } else {
    // HDR: Reinhard rather than the film grade, deliberately. This is a buffer inspector - it should
    // show what is in the texture, not what the grade would make of it.
    c = s.rgb / (1.0 + max(s.rgb, vec3f(0.0)));
  }

  return vec4f(pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2)), 1.0);
}
