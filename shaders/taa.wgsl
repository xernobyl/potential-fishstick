// ---------------------------------------------------------------------------
// Temporal accumulation.
//
// The camera drifts slowly, which is the ideal case for reprojection: blending
// one jittered sample into the reprojected history buys anti-aliasing, a smooth
// DOF bokeh and quieter AO for about the cost of a single sample.
//
// The hard part is knowing when NOT to trust the history. This body is
// *animated* — the heartbeat and life cycle move the surface every frame — and
// reprojection fundamentally assumes static geometry. Three gates handle it:
//
//  1. a SLOPE-SCALED depth gate. Its base must stay tighter than the motion the
//     pulse imparts (~0.08 world units), or the gate happily accepts history from
//     a surface that has since moved, which is exactly what smears — but a flat
//     tolerance is wrong on a steep surface, where one pixel of reprojection error
//     is legitimately a large depth change. It rejected history across most of this
//     body's detail, which measured as the single largest cause of the residual
//     shimmer. So the tolerance carries a term proportional to the local depth
//     gradient, exactly as a rasteriser slope-scales its depth bias.
//  2. class matching: a body pixel may only reuse a body pixel, background only
//     background. The dynamic tag is negative too, so a naive `w < 0` test would
//     let a satellite's history stand in for sky and leave a dark wake behind it
//     as it sweeps through the halo.
//  2b. an exact MOTION VECTOR where one exists. The rasterised solids move on their
//     own, which camera reprojection cannot describe — so they carry a per-pixel
//     screen-space delta from their own vertex shader, and that is used in place of
//     reprojection rather than alongside it.
//  3. variance clipping, because even a valid match can be stale at a
//     high-contrast edge (a star or a specular sliding past). This replaced a
//     luma-difference guard, which could not tell stale history from a noisy
//     sample and so traded ghosting for shimmer — see the note at its use.
//
// World-space dynamic geometry (the satellites) gets screen-space history at a
// heavy fresh-sample weight instead: enough to soften their edges, not enough to
// trail.
// ---------------------------------------------------------------------------

//!include "common.wgsl"

@group(1) @binding(0) var rawTex   : texture_2d<f32>;
@group(1) @binding(1) var histTex  : texture_2d<f32>;
@group(1) @binding(2) var outTex   : texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var linSamp   : sampler;
@group(1) @binding(4) var solidTex  : texture_2d<f32>;   // rgb colour, a = view distance
@group(1) @binding(5) var motionTex : texture_2d<f32>;   // xy pixel delta, or sentinel
@group(1) @binding(6) var wPrevTex : texture_2d<f32>;   // accumulated weight, last frame
@group(1) @binding(7) var wOutTex  : texture_storage_2d<r32float, write>;

/// Bilinear fetch of the accumulated weight, by hand.
///
/// r32float is `unfilterable-float`, so no sampler will do this — and it should be filtered
/// the same way the colour it belongs to is, or the ratio they form is evaluated from two
/// different places and the normalisation is wrong by exactly the interpolation error.
fn sampleWeight(pos : vec2f) -> f32 {
  let hi = vec2i(frame.accumRes.xy) - 1;
  let base = floor(pos - 0.5);
  let f = pos - 0.5 - base;
  let b = vec2i(base);
  let w00 = textureLoad(wPrevTex, clamp(b,                vec2i(0), hi), 0).r;
  let w10 = textureLoad(wPrevTex, clamp(b + vec2i(1, 0), vec2i(0), hi), 0).r;
  let w01 = textureLoad(wPrevTex, clamp(b + vec2i(0, 1), vec2i(0), hi), 0).r;
  let w11 = textureLoad(wPrevTex, clamp(b + vec2i(1, 1), vec2i(0), hi), 0).r;
  return mix(mix(w00, w10, f.x), mix(w01, w11, f.x), f.y);
}

/// Fresh-sample weight for world-space DYNAMIC geometry (the satellites), which gets
/// screen-space history only. High on purpose: enough to soften their edges, not enough to
/// trail. Named because two write paths derive from it.

const CROSS = array<vec2i, 4>(
  vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1),
);

// YCoCg, because the neighbourhood box has to be built in a space whose axes mean
// something. In RGB the three axes are correlated, so a box tight enough to catch a
// luminance change is needlessly tight across hue — it clamps colour the surface never
// actually had, and it throws away history to do it. YCoCg separates luma from two
// chroma axes, which is the space this clamp wants to live in. The transform is exact
// and costs a handful of adds.
fn rgbToYCoCg(c : vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
               0.5 * c.r - 0.5 * c.b,
              -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn yCoCgToRgb(c : vec3f) -> vec3f {
  let t = c.x - c.z;
  return vec3f(t + c.y, c.x + c.z, t - c.y);
}

/// Catmull-Rom resample of the history, in 9 bilinear taps.
///
/// Bilinear is the obvious choice and it is the wrong one, for a reason specific to
/// temporal accumulation: the history is resampled EVERY frame, so its reconstruction
/// filter is applied over and over on its own output. Bilinear is a low-pass, and a
/// low-pass compounded across fifty frames is a blur. That is the softness TAA is famous
/// for, and it comes from the resample rather than from the blend — no blend value fixes
/// it, because the detail is already gone by the time the blend happens.
///
/// Catmull-Rom is an interpolating cubic with negative lobes: it passes through its
/// samples and sharpens as it resamples, so repeated application does not accumulate loss.
///
/// 9 taps rather than the naive 16: the middle two weights per axis are folded into ONE
/// bilinear fetch, placed at an offset chosen so the hardware's own lerp reproduces their
/// weighted sum exactly (Pettineo). Dropping the four corners would make it 5 (Jimenez,
/// for Call of Duty) — worth doing only if this pass ever shows up in a profile, and at
/// 0.33 ms of a 27 ms frame it does not.
///
/// COLOUR ONLY, deliberately. The history's alpha is a depth TAG, not a channel of a
/// picture. A filter with negative lobes applied to a tag returns a distance no surface
/// ever had — and worse, a tag halfway between "background" and a real distance decodes
/// as neither. The alpha keeps its own bilinear fetch.
fn sampleHistoryCatmullRom(pos : vec2f) -> vec3f {
  let texSize = frame.accumRes.xy;
  // The texel whose centre is nearest below `pos`, in the +0.5 centre convention.
  let tc = floor(pos - 0.5) + 0.5;
  let f = pos - tc;
  let f2 = f * f;

  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f2 * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f2 * (-0.5 + 0.5 * f);

  // w1 and w2 are adjacent, so one bilinear fetch at this offset returns their sum.
  let w12 = w1 + w2;
  let offset12 = w2 / w12;

  let tc0 = (tc - 1.0) * frame.accumRes.zw;
  let tc3 = (tc + 2.0) * frame.accumRes.zw;
  let tc12 = (tc + offset12) * frame.accumRes.zw;

  var c = vec3f(0.0);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc0.x,  tc0.y),  0.0).rgb * (w0.x  * w0.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc12.x, tc0.y),  0.0).rgb * (w12.x * w0.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc3.x,  tc0.y),  0.0).rgb * (w3.x  * w0.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc0.x,  tc12.y), 0.0).rgb * (w0.x  * w12.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc3.x,  tc12.y), 0.0).rgb * (w3.x  * w12.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc0.x,  tc3.y),  0.0).rgb * (w0.x  * w3.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc12.x, tc3.y),  0.0).rgb * (w12.x * w3.y);
  c += textureSampleLevel(histTex, linSamp, vec2f(tc3.x,  tc3.y),  0.0).rgb * (w3.x  * w3.y);

  // Those negative lobes undershoot below zero across a high-contrast edge, and a
  // negative radiance poisons everything downstream at once — the YCoCg transform, the
  // clip box, the bloom prefilter. Clamp here rather than hunting the NaN later.
  return max(c, vec3f(0.0));
}

/// Pull `hist` toward the box centre only until it lands ON the box, rather than
/// clamping each channel independently.
///
/// Per-channel clamping moves the colour sideways: clip red and leave green, and the hue
/// shifts to something no sample ever had. Clipping the SEGMENT keeps the direction from
/// the box centre and only shortens it, so the history stays a plausible colour — and it
/// is strictly less aggressive, which means more history survives and the accumulation
/// actually accumulates.
fn clipToBox(hist : vec3f, mn : vec3f, mx : vec3f) -> vec3f {
  let centre = 0.5 * (mx + mn);
  let extent = max(0.5 * (mx - mn), vec3f(1e-5));
  let d = hist - centre;
  let unit = abs(d / extent);
  let worst = max(unit.x, max(unit.y, unit.z));
  // Inside the box already: leave it entirely alone.
  return select(hist, centre + d / worst, worst > 1.0);
}

/// One resolved input sample: a render-grid pixel's colour, its depth TAG, and the distance
/// that tag decodes to.
///
/// Resolving solid-over-body here rather than at three separate call sites is what keeps
/// the upsampling gather honest — every tap must be resolved identically, because the
/// reconstruction, the neighbourhood box and the depth gradient all have to describe the
/// surface that is actually VISIBLE at that tap.
///
/// Motion is deliberately NOT read here. Only the winning tap's vector is ever used, so
/// folding the fetch into this function would cost three or four wasted loads per output
/// pixel — and on the 1:1 path it would load motion for four neighbours that never want it,
/// quietly making the baseline more expensive than it is and biasing its own A/B.
struct Tap { col : vec3f, tag : f32, dist : f32 };

fn readTap(q : vec2i) -> Tap {
  var t : Tap;
  let raw = textureLoad(rawTex, q, 0);
  t.col = raw.rgb;
  t.tag = raw.a;
  t.dist = tagDepth(raw.a);
  let sc = textureLoad(solidTex, q, 0);
  // Tagged with a positive distance exactly like the body: from here on the solid layer
  // IS the surface, and every gate downstream applies to it unchanged.
  if (sc.a > 0.0 && sc.a < t.dist) {
    t.col = sc.rgb;
    t.tag = sc.a;
    t.dist = sc.a;
  }
  return t;
}

/// Reconstruction kernel for the upsampling gather: a Catmull-Rom cubic,
/// in OUTPUT pixels.
///
/// `d` is the offset from the output pixel's centre to where the input sample
/// landed. Catmull-Rom has negative lobes (the second ring at 1..2 output px
/// sharpens by pulling weight away from the centre), but they are mild enough
/// that the variance clip handles them — the same cubic already drives the
/// history resample below. Support is 2 output pixels, which fits inside the
/// 3x3 gather window at any render scale.
fn taauKernel(d : vec2f) -> f32 {
  let r = length(d);
  if (r >= 2.0) { return 0.0; }
  if (r < 1.0) {
    return 1.5 * r*r*r - 2.5 * r*r + 1.0;
  }
  return -0.5 * r*r*r + 2.5 * r*r - 4.0 * r + 2.0;
}

/// A 3x3 of input taps around an output pixel, centred on the nearest input pixel.
///
/// Serves BOTH jobs at once, which is why it is 3x3 rather than the 2x2 the reconstruction
/// alone would need:
///
///  - the RECONSTRUCTION weights each tap by the kernel, so taps beyond its reach contribute
///    nothing and cost only their fetch;
///  - the neighbourhood BOX for the variance clip is built from the same taps, and that is the
///    part that actually needed widening. Built from a 2x2 it spanned only about two output
///    pixels and shifted membership with the jitter, so it described a far narrower range than
///    the pixel could legitimately take and strangled the accumulator: 9.01% residual on the
///    noise-free signal against 1.16% once widened. FSR2 builds its YCoCg box from a 3x3 of
///    the input for the same reason.
///
/// Centring on the nearest input pixel rather than anchoring on the jittered grid also retires
/// the old ratio floor: 3x3 spans +-1 input pixel and the jitter moves samples by at most one
/// output pixel, so the nearest samples stay inside the window down to a render scale of ~0.33
/// rather than only 0.5.
const NINE = array<vec2i, 9>(
  vec2i(-1, -1), vec2i(0, -1), vec2i(1, -1),
  vec2i(-1,  0), vec2i(0,  0), vec2i(1,  0),
  vec2i(-1,  1), vec2i(0,  1), vec2i(1,  1),
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  // TWO grids from here on, and keeping them apart is the whole difficulty of this pass.
  // `out` is the ACCUMULATION grid — display resolution when upsampling — and `in` is the
  // render grid that every upstream pass wrote into.
  let outRes = frame.accumRes.xy;
  let inRes = frame.res.xy;
  if (gid.x >= u32(outRes.x) || gid.y >= u32(outRes.y)) { return; }
  let op = vec2i(gid.xy);
  let opc = vec2f(op) + 0.5;                  // output pixel centre, output grid
  let toIn = inRes * frame.accumRes.zw;       // output grid -> render grid (0.5 at 2x)
  let upsampling = toIn.x < 0.99;
  let inHi = vec2i(inRes) - 1;

  var s : vec4f;
  var rawAcc = vec3f(0.0);              // sum w_i * c_i for this frame's samples
  // Zero, not one. Both branches below assign it, so the initial value is dead today — but a
  // third path added later would silently contribute a full unit of weight and skew the
  // accumulator, and that failure would look like a subtle exposure error rather than a bug.
  var rawW = 0.0;
  var motRaw = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  /// How well this output pixel was actually sampled THIS frame. 1 at 1:1.
  var confidence = 1.0;
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var mN = 0.0;
  var nd = array<f32, 4>(1e9, 1e9, 1e9, 1e9);
  var d0 = 0.0;

  if (upsampling) {
    // GATHER — the mechanism.
    //
    // Each frame supplies one input sample per (1/toIn)^2 output pixels, so an output
    // pixel cannot simply read "its" sample; there isn't one. It is reconstructed from
    // whichever samples landed near it, weighted by HOW near. Frame after frame the jitter
    // walks those landing points across the output grid, and that is what turns low-res
    // samples into genuine full-resolution detail instead of a blurry upscale.
    //
    // The window is anchored on the JITTERED grid, not the pixel grid. That matters: the
    // jitter shifts every sample by the same sub-pixel offset, so anchoring on the naive
    // nearest input pixel would let the actually-nearest samples fall outside the window
    // whenever the jitter pushes them across a cell boundary — which is most frames.
    let baseQ = vec2i(round(opc * toIn - 0.5));
    var acc = vec3f(0.0);
    var wsum = 0.0;
    var bestW = -1.0;
    var bestY = vec3f(0.0);
    var bestDist = 1e30;
    var bestQ = baseQ;
    var tag = TAG_BG;

    for (var k = 0; k < 9; k++) {
      let o = NINE[k];
      let q = baseQ + o;
      // Out-of-range taps are SKIPPED, not clamped, and the distinction matters here in a way
      // it does not on the 1:1 path. There a clamped duplicate only narrows a variance box,
      // which is harmless. Here a clamped tap also contributes KERNEL WEIGHT, computed from the
      // position it was clamped TO rather than the one it stands for — so several offsets
      // collapse onto the same pixel at a frame edge and inflate the denominator, biasing the
      // reconstruction across a two-output-pixel border. `baseQ` itself is always in range, so
      // the centre tap always survives and `wsum` can never reach zero.
      if (any(q < vec2i(0)) || any(q > inHi)) { continue; }
      let tap = readTap(q);

      // Where this sample ACTUALLY is, in output pixels.
      let sPos = (vec2f(q) + 0.5 + frame.jitter.xy) / toIn;
      let w = taauKernel(sPos - opc);
      acc += tap.col * w;
      wsum += w;

      let y = rgbToYCoCg(tap.col);
      m1 += y;
      m2 += y * y;
      mN += 1.0;

      // DILATE the tag and the motion, never filter them. Both describe ONE surface, and
      // an interpolation of two of them describes neither — a tag halfway between
      // background and a hit decodes as nothing real. Nearest wins, which is also what
      // stops a thin foreground feature being swallowed by what is behind it.
      if (tap.dist < bestDist) {
        bestDist = tap.dist;
        tag = tap.tag;
        bestQ = q;
      }
      // The tap that DOMINATES the reconstruction takes the role the centre sample had at
      // 1:1: excluded from the box below, so a firefly cannot vote on its own ceiling.
      if (w > bestW) { bestW = w; bestY = y; }

      // A proper symmetric cross for the depth gradient, now that the window is centred:
      // both sides per axis, which is what the max-of-both-sides estimator wants and what the
      // 2x2 could not supply.
      if (o.y == 0 && o.x == -1) { nd[0] = tap.dist; }
      if (o.y == 0 && o.x ==  1) { nd[1] = tap.dist; }
      if (o.x == 0 && o.y == -1) { nd[2] = tap.dist; }
      if (o.x == 0 && o.y ==  1) { nd[3] = tap.dist; }
    }

    // NOT normalised here. The accumulator below needs the raw sum w_i*c_i, because
    // normalising per frame is precisely the defect this replaces: `acc / wsum` is an estimate
    // located at the CENTROID of this frame's samples, and that centroid slides with the
    // jitter. `s.rgb` therefore holds the reconstruction only for the paths that still want a
    // single value (the firefly clamp, and the fallback when history is rejected).
    rawAcc = acc;
    rawW = wsum;
    s = vec4f(acc / max(wsum, 1e-6), tag);
    // One motion fetch, for the surface that won — see the note on readTap.
    motRaw = textureLoad(motionTex, bestQ, 0);
    // The kernel peaks at 1, so a wsum near 1 means a sample landed essentially on this
    // pixel's centre and the reconstruction is trustworthy. Near 0 means this pixel got
    // nothing useful this frame and should lean on its history instead.
    confidence = clamp(wsum, 0.0, 1.0);
    // The gradient below must be measured against the depth the GATE will compare, which is
    // the dilated (nearest) one — not the quad's corner tap. Using the corner was a real
    // defect: where that tap is sky but the pixel resolves to a surface via another tap, the
    // difference against a 1e9 sky depth is enormous, the slack term saturates at its cap,
    // and the gate opens to maximum exactly at silhouettes — which is the case the cap exists
    // to bound. The 1:1 path always used its own visible depth; this makes the two agree.
    d0 = bestDist;
    m1 -= bestY;
    m2 -= bestY * bestY;
    mN -= 1.0;
  } else {
    // 1:1. Kept as its own path rather than falling out of the gather, because at ratio 1
    // the gather does NOT degenerate to a single tap: the jitter puts a neighbour at the
    // same distance as the centre, so it would blur across two input pixels and quietly
    // make the non-upsampling baseline worse than it is.
    let tap = readTap(op);
    s = vec4f(tap.col, tap.tag);
    rawAcc = tap.col;
    rawW = 1.0;
    d0 = tap.dist;
    motRaw = textureLoad(motionTex, op, 0);
    for (var k = 0; k < 4; k++) {
      let q = clamp(op + CROSS[k], vec2i(0), inHi);
      let n = readTap(q);
      nd[k] = n.dist;
      let y = rgbToYCoCg(n.col);
      m1 += y;
      m2 += y * y;
      mN += 1.0;
    }
  }

  // Motion is only usable if it is ABOUT the surface that won this pixel.
  //
  // The producers cannot see each other: the rings rasterise against their own depth
  // buffer and know nothing of the marched body or the ship, so a ring behind either still
  // writes its motion there. Taking that on trust reprojects the visible surface along a
  // completely unrelated vector — not subtle temporal noise but a history fetch from tens
  // of pixels away, every frame, wherever a ring passes behind something. Hence the
  // ownership check: w carries the writer's own distance, and it has to agree with what is
  // actually visible.
  var motionPx = vec2f(0.0);
  var motionDist = 0.0;
  var haveMotion = false;
  let visDist = tagDepth(s.a);
  if (abs(motRaw.x) < MOTION_NONE * 0.5
      && abs(motRaw.w - visDist) < 1e-3 * max(visDist, 1.0)) {
    // Stored as a delta in RENDER pixels, because that is the grid its producer rasterised
    // in. The history it will index lives in the output grid, so it scales up.
    motionPx = motRaw.xy / toIn;
    motionDist = motRaw.z;
    haveMotion = true;
  }

  // Averaged over however many taps the path above contributed: four neighbours at 1:1,
  // three of the reconstruction quad when upsampling (the dominant tap is excluded).
  let mInv = 1.0 / max(mN, 1.0);
  let mean = m1 * mInv;
  let sigma = sqrt(max(m2 * mInv - mean * mean, vec3f(0.0)));
  // Floor the box so it cannot collapse in smooth regions and pin the history to the
  // current frame. abs() because the two chroma axes are signed.
  let sd = max(sigma, abs(mean) * frame.taa.z + vec3f(1e-4));

  // How much the view distance legitimately changes across ONE pixel here — the same
  // |ddx| + |ddy| that `fwidth` would give in a fragment shader, which is the bound on
  // the change over a one-pixel step in any direction.
  //
  // Per axis this takes the LARGER of the two sides, not the smaller, and that was a
  // measured correction: reprojection can land the fetch in any direction, so the
  // worst-case side is the relevant one. Taking the min instead — the intuitive
  // silhouette-safe choice — systematically under-reports on a surface this noisy, since
  // it picks the smaller of two roughly independent differences at every pixel. It
  // recovered only 9.5% of the residual where the max recovers most of what is available.
  //
  // Background neighbours are EXCLUDED rather than min'd away: tagDepth hands back 1e9
  // for sky, and one such tap would otherwise report an essentially infinite gradient.
  // Body-occluding-body silhouettes still read large and legitimately so; the cap on the
  // slack term is what keeps those bounded, and it is the right place for that guard —
  // a magnitude limit, not a silently biased estimator.
  var gx = 0.0;
  var gy = 0.0;
  for (var k = 0; k < 4; k++) {
    // Sky on either side carries no surface-gradient information. Also skipped when the
    // CENTRE is sky, since then the gate takes its background branch and never reads this.
    if (nd[k] > 1e8 || d0 > 1e8) { continue; }
    let dd = abs(nd[k] - d0);
    if (k < 2) { gx = max(gx, dd); } else { gy = max(gy, dd); }
  }
  // Per OUTPUT pixel, which is the grid the reprojection error is measured in. The taps
  // are one INPUT pixel apart, so the estimate scales by the grid ratio — at 2x upsampling
  // a step of one output pixel crosses half as much surface, and a gate that forgot this
  // would hand out twice the slack it should.
  let gradPx = (gx + gy) * toIn.x;

  // Firefly clamp: an ABSOLUTE ceiling, and the reason is a measured negative result.
  //
  // A relative clamp looks obviously better on paper — a firefly is a pixel far above
  // its neighbours, so bound it by the neighbourhood rather than by a fixed number. It
  // was tried and it is worse: clamping the sample toward its neighbours' mean made the
  // bright-band frame-to-frame residual 46% WORSE, because in bright high-contrast
  // detail the bound bites on legitimate structure, and it bites by a different amount
  // every frame as the neighbourhood shifts. That ADDS variance.
  //
  // Which also settles what the bright-band residual actually is: not outliers, but
  // genuine high-frequency sampling variance in bright detail. Clamping cannot fix that;
  // more samples or a cheaper-to-sample material can. The absolute ceiling stays as what
  // it always was — a guard against a true NaN-adjacent spike poisoning the history —
  // and it is a uniform now, so it can at least be tuned without a recompile.
  let sl = max(max(s.r, s.g), s.b);
  if (sl > frame.flags.z) {
    let k = frame.flags.z / sl;
    s = vec4f(s.rgb * k, s.a);
    rawAcc *= k;                        // the numerator carries the same clamp, or the
  }                                     // accumulator would re-introduce the spike


  var hist = s.rgb;
  var blend = 1.0;                       // 1.0 = discard history entirely
  /// Weight of the retained history, in the same units as `rawW`. Zero means the gates
  /// rejected it and this pixel starts over from what it sampled this frame.
  var histW = 0.0;

  let historyValid = frame.flags.x > 0.5;

  if (historyValid) {
    // The exact ray this sample used, jitter and lens included, so a hit's world
    // point is the point that was actually shaded.
    // The ray through this OUTPUT pixel. Not the ray any single sample used — with
    // upsampling there is no such ray, since the colour came from several — so the world
    // point is reconstructed from the output pixel's own direction at the dilated depth.
    // The jitter still goes in: it is the offset the samples were actually taken with, and
    // dropping it here would reproject from a point half a pixel from where it was shaded.
    let px = opc + frame.jitter.xy / toIn;
    let ray = cameraRayAt(px, outRes);

    // A hit reprojects as a POINT (w=1); the background is at infinity so it
    // reprojects as a DIRECTION (w=0), from the pinhole centre ray. Feeding the
    // per-frame jitter into that direction would re-apply the AA/bokeh offset a
    // second time and smear it.
    // The history stores each pixel's distance from the camera that WROTE it, so the
    // gate below has to compare against the distance from THAT camera, not this one.
    // Measured against the current position instead, a camera translating 0.077 units
    // per frame blows straight through a gate that allows 0.024 — history is rejected
    // almost everywhere and the image shows raw per-frame noise. That looked like
    // "trembling", and it was latent all along: the slow orbital drift simply never
    // moved far enough in one frame to trip it.
    var gateDist = s.a;

    var rp : vec3f;
    if (haveMotion) {
      gateDist = motionDist;
      // An EXACT motion vector, not a reprojection: the solid layer's motion is
      // analytic, so its vertex shader evaluated where this surface point actually
      // was a frame ago. Camera-only reprojection cannot describe that, which is why
      // this geometry had to stay out of the accumulation buffer until now.
      rp = vec3f(opc + motionPx, 1.0);
    } else if (s.a > 0.0) {
      let hitP = ray.o + ray.d * s.a;
      gateDist = length(hitP - frame.prevCamPos.xyz);
      // JITTER-FREE, and this correction is load-bearing.
      //
      // `hitP` lies on the JITTERED ray, so its reprojection carries that sub-pixel offset
      // with it. But the history is indexed by pixel CENTRE — it holds the estimate for the
      // pixel, not for wherever this frame's sample happened to land. Reprojecting without
      // removing the jitter fetches the history from a slightly different place every frame,
      // so the accumulated image is re-filtered along a path that wanders with the jitter
      // sequence and can never settle. That is the "jittery when stopped".
      //
      // Verified in closed form on the CPU rather than by eye: with prevViewProj
      // bit-identical to viewProj — a provably static camera, where the correct drift is
      // exactly zero — the raw reprojection landed 0.2389 output pixels from the pixel it was
      // writing, against a jitter/toIn of 0.2361. It was tracking the jitter.
      //
      // This is also the quantity the depth gate's slope-scaled slack was absorbing: the
      // history it compared against sat ~a pixel away, so the depth mismatch scaled with the
      // local depth gradient. That gate change measured a real 22% win precisely because it
      // compensated for this, one layer downstream of the cause.
      //
      // The background ray is deliberately constructed without jitter, so it needs none of this.
      //
      // The MOTION-VECTOR path was assumed correct here and was not: it subtracted the bare pixel
      // centre from a reprojection of a jittered hit, leaving exactly this error in the only two
      // things that used it then - the ship and the satellites, which were the two that visibly
      // crawled. Fixed at the PRODUCER, so `rp = opc + motionPx` above needs no correction: the delta
      // carries the -jitter itself. Every producer is a rasterised mesh now, and each adds the jitter
      // back to its own fragment centre - see the fragment stages of rings/shipmesh/satmesh.wgsl.
      rp = reprojectPrevAt(hitP, 1.0, outRes);
      rp = vec3f(rp.xy - frame.jitter.xy / toIn, rp.z);
    } else {
      let centre = cameraRayAt(opc, outRes);
      rp = reprojectPrevAt(centre.d, 0.0, outRes);
    }

    if (rp.z > 0.0 && all(rp.xy > vec2f(0.5)) && all(rp.xy < outRes - 0.5)) {
      let h = textureSampleLevel(histTex, linSamp, rp.xy * frame.accumRes.zw, 0.0);
      var ok = false;
      if (s.a > 0.0) {
        // SLOPE-SCALED tolerance: a fixed fraction of distance plus whatever the local
        // depth gradient says a pixel or two of reprojection slide is worth. See
        // TEMPORAL.depthGradSlack for the measurement that motivates it — a flat gate
        // was rejecting valid history over most of the body's detail, and that was the
        // dominant source of the residual shimmer. Capped, because a silhouette sliver's
        // gradient is unbounded and the gate must not open there.
        let tol = frame.taa.w * gateDist
                + min(frame.taa2.x * gradPx, frame.taa2.y * gateDist);
        ok = h.a > 0.0 && abs(h.a - gateDist) < tol;
        // Velocity-based disocclusion: a tight, non-slope-scaled depth check
        // that catches what the permissive slope-scaled gate lets through on
        // fast pans. When the reprojected pixel moved more than 8 output pixels
        // and its depth disagrees by more than 1% of the distance, the surface
        // at that location is not the same one — it was disoccluded.
        let mvLen = length(rp.xy - opc);
        if (ok && mvLen > 8.0) {
          ok = abs(h.a - gateDist) < 0.01 * max(gateDist, 1.0);
        }
      } else {
        ok = isBackground(h.a);
      }
      if (ok) {
        // Variance clipping (Salvi / Karis), NOT a luma-difference guard.
        // Moments were computed above, since the firefly clamp needs the same ones.
        //
        // A guard that throws history away whenever it disagrees with the new
        // sample cannot tell STALE history from a NOISY sample. This shader's
        // samples are noisy by construction — few-tap AO, SSS, transmission, and
        // a lens offset that moves the ray origin every frame — so any guard
        // tight enough to stop ghosting also stops accumulation, and an image
        // that never accumulates *shimmers*. That is the same dilemma from both
        // ends, and no threshold solves it.
        //
        // Clipping solves it instead of trading it: pull the history into the
        // range the current neighbourhood says is plausible. Stale history is far
        // outside that range and gets pulled in at once, so it cannot trail;
        // sample noise is *inside* it, so it still averages away. A converged
        // pixel already sits at the mean and is left untouched.
        // Clipped in YCoCg and along the SEGMENT, not per channel in RGB — see the
        // notes on rgbToYCoCg and clipToBox. Less aggressive and hue-preserving, so more
        // history survives to actually average.
        // Resample with the sharper filter only HERE, where the history is actually
        // going to be used. Nine taps are wasted on every pixel whose history the gate
        // above just rejected, and on this body that was most of them until recently.
        var hcol = h.rgb;
        if (frame.taa2.z > 0.5) { hcol = sampleHistoryCatmullRom(rp.xy); }

        // The upsampling box is smaller and jitter-dependent, so it needs its own width —
        // see TEMPORAL.clipGammaUpsample for the measurement that forced this apart.
        let g = select(frame.taa.y, frame.taa3.z, upsampling);
        hist = yCoCgToRgb(clipToBox(rgbToYCoCg(hcol), mean - g * sd, mean + g * sd));
        // How much past this pixel is allowed to keep. Capping the accumulated weight is what
        // stands in for a blend factor: the effective weight of the new samples becomes
        // rawW / (histW + rawW), so a pixel that got a sample on its centre this frame moves
        // further than one that did not — confidence-adaptive with no separate term.
        let wcap = select(frame.taa3.x, frame.taa3.y, s.a < 0.0);
        histW = min(sampleWeight(rp.xy), wcap);
        // CONFIDENCE-WEIGHTED, and this is what makes upsampling sharpen rather than
        // blur. At 2x, each frame supplies one input sample per four output pixels; a fixed
        // blend would drag all four toward a reconstruction that only one of them was
        // actually sampled for. Scaling by how well this pixel was sampled means the pixel
        // that got a real sample this frame updates, and the ones that did not keep their
        // history until the jitter walks a sample onto them. At 1:1 confidence is 1 and
        // this reduces exactly to what it was.
        blend = select(frame.taa.x, ACCUM_BLEND_BG, s.a < 0.0) * confidence;
      }
    }
  }

  // Dragging moves the camera far per frame, so cap how much past any pixel may hold.
  if (frame.flags.y > 0.5) {
    blend = max(blend, DRAG_BLEND);
    histW = min(histW, rawW * (1.0 / DRAG_BLEND - 1.0));
  }

  // A true running WEIGHTED AVERAGE, not an exponential blend of a per-frame reconstruction.
  //
  //     colour = (hist * histW + sum w_i c_i) / (histW + sum w_i)
  //
  // The samples enter with their own kernel weights and at their own positions, so as the
  // jitter cycles the estimate's effective location converges on the pixel centre instead of
  // chasing the centroid of whichever samples happened to land nearby this frame. When the
  // gates reject the history, histW is 0 and this reduces exactly to the reconstruction.
  //
  // The 1:1 path keeps its exponential blend rather than routing through here: that baseline
  // is measured and validated at 1.70% residual, and the two are equivalent at unit weight, so
  // there is nothing to gain by disturbing it and a regression to risk.
  var outCol : vec3f;
  var outW : f32;
  if (upsampling) {
    let denom = histW + rawW;
    outCol = (hist * histW + rawAcc) / max(denom, 1e-6);
    outW = denom;
  } else {
    outCol = mix(hist, s.rgb, blend);
    outW = 1.0;
  }
  textureStore(outTex, op, vec4f(outCol, s.a));
  textureStore(wOutTex, op, vec4f(outW, 0.0, 0.0, 0.0));
}
