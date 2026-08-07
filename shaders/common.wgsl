// ---------------------------------------------------------------------------
// Shared frame state and small math helpers.
//
// The Frame struct mirrors src/core/uniforms.js field for field. The camera is
// matrices, so the field of view is data rather than a compiled-in constant and
// every consumer projects through the same numbers. Everything else is vec4 on
// purpose: WGSL's uniform address space aligns vec3 to 16 bytes and rounds
// mat3x3 columns, which silently corrupts hand-packed data. vec4-only makes the
// layout impossible to get wrong.
// ---------------------------------------------------------------------------

const PI  : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;
const PHI : f32 = 1.61803398874989;

struct Frame {
  viewProj     : mat4x4f,  // world -> clip, this frame
  invViewProj  : mat4x4f,  // clip  -> world, this frame
  prevViewProj : mat4x4f,  // world -> clip, previous frame
  camRight  : vec4f,   // xyz right,    w aperture
  camUp     : vec4f,   // xyz up,       w focusDist
  camFwd    : vec4f,   // xyz forward, w focal length (half-diagonal units)
  camPos    : vec4f,   // xyz position, w time
  res       : vec4f,   // xy size,      zw 1/size
  screen    : vec4f,   // x 1/diagPx, y diagPx, zw sensor half-extents
  misc      : vec4f,   // x beat, y life, z frameIndex, w dt
  jitter    : vec4f,   // xy pixel jitter, zw lens jitter
  flags     : vec4f,   // x historyValid, y dragging, z fireflyMax, w exposure
  sun       : vec4f,   // xy sunA screen pos, zw sunB screen pos
  shipPos   : vec4f,   // xyz world position, w throttle
  shipRot   : vec4f,   // orientation quaternion, world <- body
  shipJet   : vec4f,   // xyz angular acceleration, w reverse throttle
  prevCamPos : vec4f,  // xyz previous camera position
  prevShipPos : vec4f, // xyz the ship's position last frame
  prevShipRot : vec4f, // the ship's orientation last frame
  taa       : vec4f,   // x blend, y clipGamma, z clipFloor, w depthGate
  taa2      : vec4f,   // x depthGradSlack (px), y depthGradMax, z historyFilter, w taauSigma
  // Step scales for the body march. Uniforms rather than injected consts for the same
  // reason the TAA knobs are: as consts they are baked at pipeline creation, so every
  // console A/B against them silently measures the same shader twice. The cost is one
  // scalar compare per step, against a field evaluation that walks nine lattice
  // candidates — unmeasurable.
  march     : vec4f,   // x spare, y far, z near, w nearBand
  // Runtime A/B toggles, with NO art meaning. They exist so an optimisation that changes
  // the structure of a shader can still be measured against what it replaced, in one
  // session, on the same machine — the alternative is comparing two page loads minutes
  // apart, which on a contended machine cannot resolve anything under about 5%.
  probe     : vec4f,   // x latticeTable, y showFieldEvals, z spare, w testPattern
  // Resolution of the ACCUMULATION buffer, which is not necessarily frame.res.
  //
  // With temporal upsampling the history lives at display resolution while the passes
  // that feed it stay at render resolution, so "the resolution" stops being one number.
  // Every consumer of the accumulation buffer therefore asks for the buffer's own size
  // rather than assuming frame.res — which also means the same shader serves both the
  // upsampling and the 1:1 case, and the two can be compared without recompiling.
  accumRes  : vec4f,   // xy size, zw 1/size
  taa3      : vec4f,   // x weightMax, y weightMaxBg, z clipGammaUpsample, w spare
  // The GRADE, live. These were injected WGSL consts, which meant every one of them was
  // baked at pipeline creation and no slider could reach them — the exposure control in
  // particular silently moved only half of what it names, because the same tuning value fed
  // both a uniform gain and a const inside the curve. They are uniforms now, which is what
  // makes the debug panel a control surface rather than a display.
  grade     : vec4f,   // x filmExposure, y white, z halation, w saturation
  grade2    : vec4f,   // x vignette, y grain, z blackLift, w bloomStrength
  grade3    : vec4f,   // x contrast, y flareStrength, z bloomThreshold, w spare
  // The aurora's SHADING knobs, for the same reason. Their flow counterparts live in JS and
  // are read per step, so they were always live; these four were consts, which would have made
  // four panel sliders that quietly did nothing.
  aurora    : vec4f,   // x gain, y rays, z grazeFade, w emission phase 0..1
  // Size of the ADDITIVE target, which is a third grid alongside `res` and `accumRes`. The
  // passes that draw into it read the resolved scene depth out of the accumulation buffer, so
  // they need the ratio between those two specifically — and it is not `accumRes / res` unless
  // the additive layer happens to be at render resolution. See QUALITY.additiveDisplayRes.
  addRes    : vec4f,   // xy size, zw 1/size
  // The atmosphere's three LEVELS. Uniforms for the same reason the grade's are: these are the
  // ones worth sweeping while looking at the image, and as injected constants a slider bound to
  // one would move nothing. The rest of VOLUME is geometry and stays a constant.
  volume    : vec4f,   // x sigma, y ringOpacity, z g, w spare
  // Per-channel white-balance gains, derived from FILM.temperature on the CPU. Last, because the
  // order of this struct IS the uniform layout - a field inserted anywhere else silently reinterprets
  // everything after it.
  balance   : vec4f,   // xyz gains, w buffer-viewer display mode
  /// The model viewer: x turntable angle, y the same one frame ago, z spare, w 1 when it is active.
  /// Zero in every other scene, which is what lets `w` gate the studio backdrop and the model layout.
  model     : vec4f,
};

@group(0) @binding(0) var<uniform> frame : Frame;

// ---- quaternions -------------------------------------------------------
//
// Here rather than in ship.wgsl and railgun.wgsl, which each carried an identical copy — and neither of
// which the mesh vertex front end could include, since it has no business depending on either. One
// definition, in the header everything already includes.
//
// The inverse rotation used to live here too. Its only caller was the marched hull, which rotated the
// RAY into the ship's local frame rather than rotating the field; a mesh transforms the other way, so
// when the hull became geometry the inverse had nothing left to do.

/// Rotate a vector by a unit quaternion. The two-cross-product form: no matrix, no trig.
fn qrotate(q : vec4f, v : vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}


fn beatPhase() -> f32 { return frame.misc.x; }
fn lifePhase() -> f32 { return frame.misc.y; }
fn frameIndex() -> f32 { return frame.misc.z; }

// ---- the pulse ---------------------------------------------------------
//
// Here rather than in sdf.wgsl, beside the phase accessor it reads: the beat drives
// the body's geometry, the core's brightness and the grade — three consumers, only
// one of which marches anything.

/// Compact quartic bump: 1 at the centre, exactly 0 at +-w, smooth at both ends.
/// This stands in for a Gaussian, and the substitution is worth explaining: the
/// heartbeat is evaluated once per SPHERE CANDIDATE, so at 6 layers x 9
/// candidates it ran ~100 exp() calls per field evaluation, times ~100 march
/// steps, times every pixel. It was among the hottest instructions in the frame.
/// A quartic matched at half-height is visually indistinguishable here and costs
/// three multiplies.
fn bump(x : f32, w : f32) -> f32 {
  let u = clamp(x / w, -1.0, 1.0);
  let v = 1.0 - u * u;
  return v * v;
}

/// Peak of the raw two-bump sum, so `heartbeat` can be normalised to [0, 1].
///
/// Not cosmetic: the marching bound has to know the largest a sphere can swell,
/// and a beat whose peak is some incidental number that falls out of two bump
/// widths makes that bound a hand-computed magic constant which silently rots the
/// moment either width is touched. Pinning the peak at 1 makes PULSE_R mean
/// exactly what it says and makes the bound derivable.
const HEARTBEAT_PEAK : f32 = 1.18994;

/// Two thumps per cycle (the lub-dub of systole/diastole) rather than a sine,
/// which is what makes it read as a pulse instead of a wobble.
/// Returns [0, 1].
fn heartbeat(t : f32) -> f32 {
  let x = fract(t);
  // The onset at the wrap is deliberate and matches the original: a beat should
  // arrive, not fade in.
  return (bump(x - 0.12, 0.23) + 0.55 * bump(x - 0.32, 0.28)) * (1.0 / HEARTBEAT_PEAK);
}


// ---- small utilities ----------------------------------------------------

fn sqr(x : f32) -> f32 { return x * x; }

fn luma(c : vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

/// Polynomial smooth minimum (iq). Blends the sphere octaves together.
fn smin(a : f32, b : f32, k : f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
fn smax(a : f32, b : f32, k : f32) -> f32 { return -smin(-a, -b, k); }

/// Ray vs sphere centred at the origin. Returns (near, far), or (-1,-1) on miss.
fn iSphere(ro : vec3f, rd : vec3f, ra : f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - ra * ra;
  let h = b * b - c;
  if (h < 0.0) { return vec2f(-1.0); }
  let s = sqrt(h);
  return vec2f(-b - s, -b + s);
}

/// Beer-Lambert extinction through `thick` of medium with per-channel sigma_t.
fn beerLambert(thick : f32, sigmaT : vec3f) -> vec3f { return exp(-thick * sigmaT); }

/// Henyey-Greenstein phase function. g>0 scatters forward, so the medium lights
/// up when looking toward a light through a thin part of it.
fn phaseHG(cosT : f32, g : f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosT, 1.5));
}

/// Cornette-Shanks: Henyey-Greenstein with the (1 + cos^2) factor it is missing.
///
/// Same forward-lobe parameter `g`, and the extra factor is what restores the BACKSCATTER a real
/// aerosol has — HG's lobe falls monotonically to a minimum at 180 degrees, where Mie scattering
/// actually turns back up. It is the standard correction for atmospheric aerosol and costs two
/// multiplies, so the atmosphere uses it while the body's interior keeps plain HG: inside a dense
/// medium the light has forgotten which way it came from and the difference is unobservable.
fn phaseCS(cosT : f32, g0 : f32) -> f32 {
  // NORMALISED, and checked rather than assumed: integrating this over the sphere gives 1.000000 at
  // g = 0, 0.3, 0.68 and 0.9 (two million samples). That matters because the constant in front is easy
  // to get wrong in a way nothing catches — an unnormalised phase function still looks like an
  // atmosphere, it just scatters the wrong TOTAL amount of light, so it reads as the sigma being
  // mistuned and gets "fixed" there instead. The check lives here rather than in dev/ because the
  // function is WGSL and a JavaScript copy of it would only ever test the copy.
  //
  // Clamped, because `g` is a UNIFORM now: the denominator vanishes as g approaches 1 with the
  // forward lobe, and a value typed into the console should not be able to divide by zero.
  let g = clamp(g0, 0.0, 0.95);
  let g2 = g * g;
  return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cosT * cosT))
       / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// ---- camera ------------------------------------------------------------
//
// One screen space, shared by everything: origin centred, y UP, and normalised
// so that the half DIAGONAL is 1. Normalising by the diagonal rather than by an
// axis is what makes the framing aspect-independent — the axes only divide up a
// fixed diagonal, so a 21:9 window and a 9:21 window frame the subject alike,
// and a radial effect like the vignette reaches the corners identically at any
// aspect. `frame.screen.zw` holds the resulting half-extents (hypot == 1).
//
// Pixel coordinates are texture-space throughout (origin top-left, y down), so
// the y flip lives here, in one place, and every pass agrees with the
// rasteriser's clip space without anyone having to think about it.

/// Pixel coordinate -> shared screen space, for an arbitrary pixel grid.
///
/// Screen space is normalised by the half DIAGONAL, which makes it genuinely
/// resolution-independent: the same fractional position in a 1280x720 grid and a 2560x1440
/// one maps to the same screen coordinate. Temporal upsampling needs exactly that, since it
/// reasons about output pixels and input pixels in the same breath.
fn screenUVAt(px : vec2f, res : vec2f) -> vec2f {
  return (2.0 * px - res) * (1.0 / length(res)) * vec2f(1.0, -1.0);
}

/// Pixel coordinate -> shared screen space, in the RENDER grid.
fn screenUV(px : vec2f) -> vec2f {
  return (2.0 * px - frame.res.xy) * frame.screen.x * vec2f(1.0, -1.0);
}

/// Shared screen space -> pixel coordinate. Exact inverse of screenUV.
fn uvToPixel(uv : vec2f) -> vec2f {
  return 0.5 * (uv * vec2f(1.0, -1.0) * frame.screen.y + frame.res.xy);
}

/// The same inverse, for an arbitrary pixel grid. Exact inverse of screenUVAt.
fn uvToPixelAt(uv : vec2f, res : vec2f) -> vec2f {
  return 0.5 * (uv * vec2f(1.0, -1.0) * length(res) + res);
}

/// Apply the frame's sub-pixel jitter to a clip-space position.
///
/// The compute passes jitter the RAY — `px + 0.5 + frame.jitter` — so a RASTER pass has to move its
/// geometry by the same amount or it lands on fixed pixel centres every frame, the temporal resolve
/// has nothing to average for it, and it gets no antialiasing at all. That was true of the rings,
/// which are the only geometry here that is rasterised rather than marched, and it went unnoticed
/// because the jitter lives in ray generation where a raster pass never looks.
///
/// The shift is NEGATIVE. The march stores, at pixel `px`, the scene sampled at `px + jitter`, so a
/// surface whose unjittered projection is `px + jitter` has to be drawn at `px`. Same reasoning,
/// and the same sign, as the `rp -= jitter` correction in the resolve.
///
/// y flips because pixel y runs down while NDC y runs up — see uvToPixel.
fn jitterClip(clip : vec4f) -> vec4f {
  let j = frame.jitter.xy * 2.0 * frame.res.zw;
  return vec4f(clip.x - j.x * clip.w, clip.y + j.y * clip.w, clip.z, clip.w);
}

/// Clip-space NDC (from a projection matrix) -> shared screen space.
fn ndcToUV(ndc : vec2f) -> vec2f { return ndc * frame.screen.zw; }

// The shared MOTION-VECTOR helper that used to sit here is gone, and where it went is the point.
//
// `motionFor` existed because the marched ship and the marched satellites each needed the same five
// steps — project through the previous view-projection, convert to previous-frame pixels, measure
// against the PREVIOUS camera, keep the jitter inside the delta, tag the owner — and doing that twice
// slightly differently is how one of them ends up crawling. Both are rasterised meshes now, and they
// get the same five steps from `meshXform` in mesh_vertex.wgsl instead, which is a better place for
// them: a vertex stage knows the previous transform exactly, where a marched sample could only
// reconstruct it from a hit point. When the last caller went, so did the helper.

/// A pixel coordinate in the RENDER grid, expressed in the ACCUMULATION grid.
///
/// Identity when the two are the same size. It exists because the additive passes
/// (embers, contrail, railgun) rasterise at render resolution but read the resolved scene
/// DEPTH out of the accumulation buffer to occlude themselves — and with temporal
/// upsampling that buffer is display resolution, so indexing it with a render-grid integer
/// silently reads the top-left quadrant of the frame. That failure looks like the trails
/// being occluded by geometry from somewhere else entirely.
fn toAccumPx(px : vec2f) -> vec2i {
  return vec2i(px * frame.accumRes.xy * frame.addRes.zw);
}

struct Ray { o : vec3f, d : vec3f };

fn cameraRayUV(uv : vec2f) -> Ray {
  // Unproject the point at INFINITY, which under this reverse-Z projection is NDC z = 0.
  //
  // The result comes back with w = 0 — a homogeneous DIRECTION, not a position — so the world
  // -space ray direction falls straight out with no division and no subtraction. That is
  // strictly better than what this did with a finite projection: it unprojected the far plane
  // and then subtracted the camera position, and differencing two large world coordinates to
  // recover a unit vector is exactly the catastrophic cancellation that code was working
  // around. Here there is nothing to cancel.
  let ndc = uv / frame.screen.zw;
  let dirH = frame.invViewProj * vec4f(ndc, 0.0, 1.0);
  let fwd = normalize(dirH.xyz);

  // PINHOLE. There used to be a thin lens here: one sample per frame on the aperture disk, with
  // temporal accumulation asked to turn it into a smooth bokeh. It was wrong in two ways at once and
  // both are structural, so it is gone rather than tuned.
  //
  // It could not converge. The resolve's variance clip rebuilds its box from each frame's
  // neighbourhood and pulls the history back toward the newest sample, so a per-frame lens offset
  // never settles - measured on a stopped scene, it moved 0.457% of pixels by more than 4 of 255
  // levels against 0.107% for the pixel jitter alone, and more history did not help (weightMax 6 to
  // 200, blend 0.15 to 0.04, both no change).
  //
  // And it only ever applied HERE. Ray generation is a compute-pass idea; the rings are rasterised
  // through `jitterClip(frame.viewProj * ...)`, which carries the pixel jitter and nothing else. So
  // the rings had no defocus while everything marched did - they were the only stable thing on
  // screen at a wide aperture, which is what gave this away.
  //
  // The right shape for it is a deterministic post-process GATHER: circle of confusion from the
  // thin-lens relation, a fixed low-discrepancy disk of taps, applied once to the composited image
  // where the rings are already present. Same input, same output, so it is stable by construction
  // instead of depending on accumulation. See CAMERA in tuning.js.
  return Ray(frame.camPos.xyz, fwd);
}

/// The primary ray, including the off-centre framing and the thin-lens aperture.
/// Shading and reprojection must agree on this exactly, or the history is
/// reprojected from a subtly wrong world position and smears — so both go
/// through the same matrix rather than through two copies of the same algebra.
fn cameraRay(px : vec2f) -> Ray {
  return cameraRayUV(screenUV(px));
}

/// The same ray, from a pixel in an arbitrary grid — the upsampling resolve reasons in
/// OUTPUT pixels while the samples it gathers were taken in the render grid.
fn cameraRayAt(px : vec2f, res : vec2f) -> Ray {
  return cameraRayUV(screenUVAt(px, res));
}


/// Project into the PREVIOUS frame's view, as a pixel coordinate.
/// Returns z<0 when it falls behind that camera.
///
/// `w` selects what is being reprojected, and collapses two cases that used to
/// be two functions that had to agree: 1 for a world POINT, 0 for a DIRECTION.
/// At w=0 the matrix's translation column drops out, which is exactly the
/// rotation-only projection that something at infinity wants — the background
/// must reproject that way, because treating the sky as screen-static drags
/// comet trails off every star.
fn reprojectPrev(p : vec3f, w : f32) -> vec3f {
  let clip = frame.prevViewProj * vec4f(p, w);
  if (clip.w <= 1e-4) { return vec3f(0.0, 0.0, -1.0); }
  return vec3f(uvToPixel(ndcToUV(clip.xy / clip.w)), 1.0);
}
/// Reprojection into an arbitrary pixel grid. The temporal upsampling resolve indexes a
/// history at DISPLAY resolution while `frame.res` is the render resolution, so it cannot
/// use the render-grid version — it would land at half the intended coordinate.
fn reprojectPrevAt(p : vec3f, w : f32, res : vec2f) -> vec3f {
  let clip = frame.prevViewProj * vec4f(p, w);
  if (clip.w <= 1e-4) { return vec3f(0.0, 0.0, -1.0); }
  return vec3f(uvToPixelAt(ndcToUV(clip.xy / clip.w), res), 1.0);
}


// ---- depth tagging -----------------------------------------------------
//
// The scene buffer's alpha channel carries *what* was hit as well as how far:
//   > 0      the body, at that distance
//   -1       background
//   < -50    world-space DYNAMIC geometry, at distance (-w - 100)
//
// The third class exists because reprojection assumes static geometry. Anything
// that moves through the world on its own — the satellites — would trail a
// double-edged ghost behind it, so it is tagged and the accumulator leans on
// fresh samples exactly there. The class is named for that PROPERTY rather than
// for whatever currently has it: it was introduced for the orbiting rocks, which
// are gone, and the satellites inherit it for precisely the same reason.

// NEGATIVE MEANS BACKGROUND, and that is now the only negative there is. A second encoding used to
// live down here - a distance biased into the negatives to mark geometry as "dynamic", which the
// resolve answered by reading history from the same screen pixel with no reprojection at all. Its
// last user was the satellites, and they now carry an exact motion vector instead (their orbits are
// analytic, so the previous transform is just the same evaluation one frame back). With no writers
// the encoding, its predicate and the resolve branch it selected were all unreachable, so they are
// gone rather than left to be maintained.
const TAG_BG : f32 = -1.0;

fn isBackground(w : f32) -> bool { return w < 0.0; }
fn tagDepth(w : f32) -> f32 {
  if (w > 0.0) { return w; }
  return 1e9;
}
