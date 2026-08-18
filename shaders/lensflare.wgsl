// ---------------------------------------------------------------------------
// Lens flares.
//
// Two mechanisms, because they model different things:
//
//  1. GHOSTS + HALO, screen-space, from the already-blurred bloom pyramid. Any
//     bright region throws internal reflections back through the lens, so the
//     source is sampled point-mirrored about the optical centre at a few scales,
//     with a small per-channel offset for the chromatic fringing real coatings
//     produce. This costs a handful of taps and reacts to *everything* bright.
//
//  2. A SUN-ANCHORED streak and starburst. The suns' screen positions come in
//     through the frame uniform, so these can be placed properly rather than
//     inferred, which is what makes them swing correctly as the camera drifts.
//
// Runs at quarter resolution: flares are, by construction, the blurriest thing
// on screen, and nothing here has an edge worth resolving.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "fullscreen.wgsl"

@group(1) @binding(0) var srcTex  : texture_2d<f32>;   // blurred, thresholded scene
@group(1) @binding(1) var srcSamp : sampler;

fn tap(uv : vec2f) -> vec3f {
  // Outside the frame there is nothing to reflect. Clamping would smear the
  // border texel into a bar — but rejecting outright just trades that for a hard
  // straight cut wherever a ghost crosses the frame edge, which is worse: the
  // edge of the SCREEN is not a feature of the lens, so it must not be visible
  // as one. Fade across a band instead, which has neither failure.
  let w = smoothstep(vec2f(0.0), vec2f(FLARE_EDGE), uv)
        * smoothstep(vec2f(0.0), vec2f(FLARE_EDGE), 1.0 - uv);
  let f = w.x * w.y;
  if (f <= 0.0) { return vec3f(0.0); }
  return textureSampleLevel(srcTex, srcSamp, uv, 0.0).rgb * f;
}

/// Chromatic triple-tap: the three channels take slightly different paths.
fn tapChroma(uv : vec2f, centre : vec2f, amount : f32) -> vec3f {
  let dir = normalize(uv - centre + 1e-6);
  let o = dir * amount;
  return vec3f(tap(uv + o).r, tap(uv).g, tap(uv - o).b);
}

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  // The OPTICAL AXIS, not the middle of the frame — and they are not the same point here.
  //
  // Internal reflections mirror through the lens's axis, so that is what a ghost is placed
  // against. This camera composes off-centre (CAMERA.frameOffset shears the projection, which
  // is what keeps the subject's placement stable across aspect ratios), so its axis lands at
  // uv (0.4805, 0.4796) rather than (0.5, 0.5). Mirroring about 0.5 therefore put every ghost
  // and the whole halo ring TWICE that error off — about 100 device pixels sideways at 2560
  // wide, in a fixed direction, on every bright thing in the frame.
  //
  // Derived rather than injected, because the offset is in shared screen space and the
  // conversion to uv depends on the window's shape: inverting `p` below gives
  // uv = 0.5 + p * (1, -1) / (2 * screen.zw).
  let centre = vec2f(0.5) + LENS_AXIS * vec2f(1.0, -1.0) / (2.0 * frame.screen.zw);
  var col = vec3f(0.0);

  // ---- ghosts: point-mirrored copies at increasing scale ----
  let mirrored = centre - (uv - centre);
  for (var i = 0; i < FLARE_GHOSTS; i++) {
    let fi = f32(i);
    let scale = 1.0 + (fi - f32(FLARE_GHOSTS) * 0.5) * FLARE_SPACING;
    let guv = centre + (mirrored - centre) * scale;
    // Fade toward the frame edge: a ghost sliding off screen must not pop.
    let vign = 1.0 - smoothstep(0.35, 0.72, length(guv - centre));
    let tint = 0.6 + 0.4 * cos(vec3f(0.0, 1.6, 3.2) + fi * 1.4);
    col += tapChroma(guv, centre, FLARE_CHROMA * (1.0 + fi)) * vign * tint
         * (FLARE_GHOST_GAIN / f32(FLARE_GHOSTS));
  }

  // ---- halo: a ring pulled from a fixed radial offset ----
  {
    let d = uv - centre;
    let l = length(d) + 1e-6;
    let huv = centre + (d / l) * (l - FLARE_HALO_W * 0.5) * 1.0;
    let ring = exp(-sqr(l - FLARE_HALO_W * 0.5) * 34.0);
    col += tapChroma(huv, centre, FLARE_CHROMA * 2.0) * ring * FLARE_HALO;
  }

  // ---- sun-anchored streak and starburst ----
  // frame.sun holds each sun's position in the shared screen space (half
  // diagonal 1, y up), so convert this fragment to that space to compare. Going
  // through screen.zw rather than through the aspect ratio is what keeps a
  // streak anchored to its sun when the window changes shape.
  //
  // About 0.5 here, deliberately, unlike `centre` above: this is the uv -> screen-space
  // conversion and screen space has its origin at the frame centre by definition. Both sides
  // of the comparison are in that space, so it is consistent — the optical axis simply is not
  // where this particular origin sits.
  let p = (uv - 0.5) * 2.0 * frame.screen.zw * vec2f(1.0, -1.0);
  for (var s = 0; s < 2; s++) {
    var sp = frame.sun.xy;
    var tint = vec3f(1.0, 0.86, 0.62);
    var occl = frame.jitter.z;
    if (s == 1) { sp = frame.sun.zw; tint = vec3f(0.80, 0.55, 1.0); occl = frame.jitter.w; }
    if (sp.x > 900.0) { continue; }            // sentinel: sun is behind us
    // A sun behind the body is occluded: its streak and burst must not draw
    // through the world in front of it.
    if (occl > 0.5) { continue; }

    let d = p - sp;
    // The anamorphic streak. A cylindrical front element smears the highlight
    // along one axis only, so this is deliberately extreme in aspect: wide in x,
    // very tight in y.
    let wide = exp(-abs(d.x) * FLARE_STREAK_FALL);
    let thin = exp(-sqr(d.y) * FLARE_STREAK_TIGHT);
    // Squared falloff keeps the core hot while the tails go properly dark. That
    // contrast is what separates a lens flare from a wash of glow.
    let streak = wide * wide * thin;
    // Starburst from the aperture blades, squared for the same reason and kept
    // well under the streak — on a real anamorphic rig it is a minor feature.
    let ang = atan2(d.y, d.x);
    let blades = 0.5 + 0.5 * cos(ang * 6.0);
    let radial = exp(-length(d) * 7.0) * blades * blades;
    // The streak takes the coating's blue bias; the burst keeps the sun's colour.
    col += FLARE_STREAK_COL * (streak * FLARE_STREAK) + tint * (radial * 0.22);
  }

  // Nothing in a flare may reach the frame border. A streak or ghost that runs
  // off the edge at finite brightness reads as a hard-cut bar across the screen,
  // which is the most artificial thing a flare can do — and unlike the mirrored
  // taps above, the analytic streak and starburst are not bounded by any texture,
  // so they need their own limit. Real lenses vignette their own flares hard, so
  // this is also the physical answer.
  let vg = smoothstep(vec2f(0.0), vec2f(FLARE_VIGNETTE), uv)
         * smoothstep(vec2f(0.0), vec2f(FLARE_VIGNETTE), 1.0 - uv);
  col *= vg.x * vg.y;

  return vec4f(max(col * frame.grade3.y, vec3f(0.0)), 1.0);
}
