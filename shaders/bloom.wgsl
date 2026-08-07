// ---------------------------------------------------------------------------
// Glow: a dual-filter (Kawase) down/up pyramid.
//
// Bloom is the one thing a single pass genuinely cannot do — halation can tint a
// bright pixel but never spread light onto its neighbours.
//
// Dual filtering (Bjorge et al., ARM) rather than a separable Gaussian: the
// downsample is 5 taps and the upsample 8, both exploiting bilinear filtering,
// and each level halves resolution. That gives a very wide, very smooth kernel
// for a fraction of the bandwidth a wide Gaussian would need — and because the
// glow is reconstructed from several resolutions, it has both a tight core and a
// broad halo instead of one fixed radius.
//
// The upsample accumulates with additive blending supplied by the pipeline, so
// each coarser level lands on the finer one without a separate combine pass.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "fullscreen.wgsl"

struct BloomParams {
  texel  : vec2f,       // 1/size of the SOURCE texture
  // Down/upsample: filter radius multiplier. PREFILTER: the reduction ratio, source texels per
  // destination texel, which sets how many taps the box average needs.
  radius : f32,
  weight : f32,         // contribution weight (upsample only)
};

@group(1) @binding(0) var srcTex  : texture_2d<f32>;
@group(1) @binding(1) var srcSamp : sampler;
@group(1) @binding(2) var<uniform> params : BloomParams;

// Optional second source, used only by the prefilter to fold the particles in
// without a separate combine pass.
@group(1) @binding(3) var addTex : texture_2d<f32>;

fn tap(uv : vec2f) -> vec3f {
  return textureSampleLevel(srcTex, srcSamp, uv, 0.0).rgb;
}

/// The threshold itself, applied to ONE sample. Soft-knee rather than a hard cut, or the
/// threshold boundary flickers as values cross it.
///
/// Separated out because WHERE it is applied matters more than what it does. Thresholding a
/// box-averaged colour destroys isolated highlights: a star bright enough to glow on its own,
/// averaged over a 4x4 footprint of near-black sky, lands below the threshold and contributes
/// nothing at all. Applied per tap instead, each 2x2 sub-region keeps its own contribution and
/// the average that follows conserves it. This is the same reason production bloom chains
/// threshold at the first reduction rather than after it.
fn prefilterOne(c : vec3f) -> vec3f {
  let l = max(max(c.r, c.g), c.b);
  // From the frame uniform, not a const: the threshold is the knob you most want to sweep
  // while looking at the image, and as a const every sweep silently measured the same value.
  let thresh = frame.grade3.z;
  let knee = thresh * 0.5;
  let soft = clamp((l - thresh + knee) / max(2.0 * knee, 1e-4), 0.0, 1.0);
  let contrib = max(l - thresh, l * soft * 0.25);
  return c * (contrib / max(l, 1e-4));
}

/// Prefilter: threshold, and fold in the additive layer (embers) so they glow too.
@fragment
fn fsPrefilter(@location(0) uv : vec2f) -> @location(0) vec4f {
  // BOX-AVERAGE THE SOURCE over this destination texel's whole footprint.
  //
  // A single bilinear tap covers 2x2 source texels, which is exactly right when the reduction is
  // 2x — and this pass is not always reducing by 2x. With temporal upsampling on, the
  // accumulation buffer is DISPLAY resolution while level 0 of the pyramid is half the RENDER
  // resolution, so the step is 4x and one tap was reading 4 of every 16 source texels and
  // ignoring the other 12. A star landing on an ignored texel contributed no glow at all, and
  // popped in and out of the bloom as it drifted across the grid — visible on exactly the
  // smallest, brightest features, which is the worst place to lose energy.
  //
  // ratio/2 taps per axis, each an exact 2x2 average, tile the footprint precisely: for a 4x
  // step the taps land on the boundaries between source texels 4j|4j+1 and 4j+2|4j+3, so the
  // four of them average all sixteen with equal weight. For a 2x step it collapses back to the
  // single centred tap, which was already correct.
  let taps = max(1, i32(params.radius * 0.5 + 0.5));
  let inv = 1.0 / f32(taps * taps);
  var c = vec3f(0.0);
  for (var y = 0; y < taps; y++) {
    for (var x = 0; x < taps; x++) {
      let o = (vec2f(f32(x), f32(y)) * 2.0 - f32(taps - 1)) * params.texel;
      // Scene and additive layer summed BEFORE the threshold, so an ember over a dim background
      // can cross it on their combined brightness, as it always could.
      let src = tap(uv + o) + textureSampleLevel(addTex, srcSamp, uv + o, 0.0).rgb;
      c += prefilterOne(src) * inv;
    }
  }

  // Fade to black in a band at the frame border.
  //
  // The pyramid samples clamp-to-edge, because WebGPU has no clamp-to-border. So
  // a bright texel sitting AT the border is replicated outward at every level and
  // smears into a bright bar down that whole edge — worst exactly when a sun
  // drifts in from off screen, which is when there is most energy there to
  // replicate. Killing it here fixes the entire chain at once: every coarser
  // level then has a dark border of its own to replicate instead.
  let e = smoothstep(vec2f(0.0), vec2f(BLOOM_EDGE), uv)
        * smoothstep(vec2f(0.0), vec2f(BLOOM_EDGE), 1.0 - uv);
  c *= e.x * e.y;

  return vec4f(max(c, vec3f(0.0)), 1.0);
}

/// Dual-filter downsample: centre weighted 4, four diagonal taps at half a
/// source texel. Total 5 samples for an effective 4x4 footprint.
@fragment
fn fsDown(@location(0) uv : vec2f) -> @location(0) vec4f {
  let h = params.texel * 0.5 * params.radius;
  var sum = tap(uv) * 4.0;
  sum += tap(uv + vec2f(-h.x, -h.y));
  sum += tap(uv + vec2f( h.x, -h.y));
  sum += tap(uv + vec2f(-h.x,  h.y));
  sum += tap(uv + vec2f( h.x,  h.y));
  return vec4f(sum * (1.0 / 8.0), 1.0);
}

/// Dual-filter upsample: an 8-tap tent, the diagonals weighted 2.
///
/// The offsets are in SOURCE texels, so the tent spans two destination texels rather than the
/// half-texel of Bjorge's original. That is a wider kernel, not a wrong one — it stays symmetric
/// and normalised, and it simply makes each level's contribution smoother and broader. GLOW.radius
/// scales it, so the convention is absorbed there; changing it would silently change the look.
@fragment
fn fsUp(@location(0) uv : vec2f) -> @location(0) vec4f {
  let h = params.texel * params.radius;
  var sum = tap(uv + vec2f(-h.x * 2.0, 0.0));
  sum += tap(uv + vec2f(-h.x, h.y)) * 2.0;
  sum += tap(uv + vec2f(0.0, h.y * 2.0));
  sum += tap(uv + vec2f(h.x, h.y)) * 2.0;
  sum += tap(uv + vec2f(h.x * 2.0, 0.0));
  sum += tap(uv + vec2f(h.x, -h.y)) * 2.0;
  sum += tap(uv + vec2f(0.0, -h.y * 2.0));
  sum += tap(uv + vec2f(-h.x, -h.y)) * 2.0;
  return vec4f(sum * (1.0 / 12.0) * params.weight, 1.0);
}
