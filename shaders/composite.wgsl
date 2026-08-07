// ---------------------------------------------------------------------------
// Composite: assemble the layers, then the film response.
//
// Everything upstream is linear HDR. This is the only pass that knows about
// display encoding, and the last one that may add anything camera-relative —
// the limb glow and the grain both live here for that reason.
//
// The grade is an analytic Kodak-ish film response rather than a LUT: highlights
// that desaturate toward warm white instead of clipping, a red layer that
// scatters (halation), a filmic S-curve with a long shoulder, a cool toe against
// warm highlights, and a black that never quite reaches zero.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "fullscreen.wgsl"
//!include "sky.wgsl"
//!include "explosion.wgsl"

@group(1) @binding(0) var sceneTex : texture_2d<f32>;
@group(1) @binding(1) var emberTex : texture_2d<f32>;
@group(1) @binding(2) var bloomTex : texture_2d<f32>;
@group(1) @binding(3) var flareTex : texture_2d<f32>;
@group(1) @binding(4) var linSamp  : sampler;

/// Filmic S-curve (Hable): long shoulder, gentle toe.
fn hableCurve(x : vec3f) -> vec3f {
  let A = 0.15; let B = 0.50; let C = 0.10;
  let D = 0.20; let E = 0.02; let F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn filmGrade(c0 : vec3f) -> vec3f {
  var c = c0 * vec3f(1.03, 1.00, 0.96);            // warm daylight stock
  let hi = luma(c);

  // Bright film desaturates toward a warm white rather than staying saturated.
  c = mix(c, vec3f(hi) * vec3f(1.05, 1.00, 0.92), smoothstep(1.0, 5.0, hi) * 0.60);
  // Halation: the emulsion's red layer scatters, so highlights bleed warm.
  let h = smoothstep(0.5, 1.4, hi);
  c += h * h * vec3f(0.12, 0.035, 0.005) * frame.grade.z;

  // WHITE BALANCE, before the curve because it is a property of the SENSOR rather than of the print.
  // Derived from FILM.temperature on the CPU - see whiteBalanceGains - and delivered as a uniform so
  // the slider moves it live. The print's cool-toe/warm-highlight split below is a separate stage and
  // deliberately not where this lives.
  c *= frame.balance.rgb;

  c = hableCurve(c * frame.grade.x) / hableCurve(vec3f(frame.grade.y));
  c = pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));   // display transfer

  // Print: cool toe, warm highlights — where the teal/orange look comes from.
  let l = luma(c);
  c *= mix(vec3f(0.90, 0.98, 1.12), vec3f(1.07, 1.01, 0.90), smoothstep(0.05, 0.85, l));

  // CONTRAST, about a mid pivot and in the display-referred domain — after the transfer, which
  // is where a contrast control belongs. Doing it in linear light is the same operation as
  // moving exposure and the white point together, and it crushes the shadows to nothing while
  // barely touching the highlights. Here it pivots, so it opens the gap between the sky and
  // everything in front of it without moving the midtones.
  c = (c - FILM_PIVOT) * frame.grade3.x + FILM_PIVOT;

  // Film's lifted black. This is the single biggest lever on how dark the SKY reads: the
  // background is nearly zero everywhere, so it lands almost exactly on this floor and nothing
  // else in the grade moves it much.
  return mix(FILM_BLACK, max(c, vec3f(0.0)), 1.0 - frame.grade2.z);
}

@fragment
fn fs(@builtin(position) fragPos : vec4f, @location(0) uv : vec2f) -> @location(0) vec4f {
  // This is the one pass that runs at DISPLAY resolution while its inputs are at
  // render resolution, so `fragPos` must not be used to index them. `uv` is
  // resolution-independent; everything scene-side derives from it.
  // TWO pixel coordinates, deliberately named apart, because one variable used to do both
  // jobs and that broke the moment the two grids stopped being the same size:
  //
  //   apx — indexes the ACCUMULATION buffer, which is display resolution under temporal
  //         upsampling and render resolution without it.
  //   rpx — feeds `cameraRay` and `screenUV`, which are defined in RENDER pixels against
  //         `frame.res`. Handing them an accum-grid coordinate makes screen space run
  //         -1..+3 instead of -1..+1, so the sky ray leaves the sensor and the frame fades
  //         to black from left to right.
  let apx = min(uv * frame.accumRes.xy, frame.accumRes.xy - 0.5);
  let rpx = uv * frame.res.xy;

  // The colour is filtered on the way up — a free bilinear upscale — but the
  // alpha channel is a depth TAG, and interpolating between a background -1 and
  // a surface at +4.7 yields a value that means neither. So: filter the colour,
  // point-sample the tag.
  var col = textureSampleLevel(sceneTex, linSamp, uv, 0.0).rgb;
  let sceneW = textureLoad(sceneTex, vec2i(apx), 0).a;
  var depth = tagDepth(sceneW);
  var onSky = isBackground(sceneW);


  col += textureSampleLevel(emberTex, linSamp, uv, 0.0).rgb;
  let halo = textureSampleLevel(bloomTex, linSamp, uv, 0.0).rgb;
  col += halo * frame.grade2.w;
  col += textureSampleLevel(flareTex, linSamp, uv, 0.0).rgb;

  // Spherochromatism. A fast lens's residual spherical aberration varies with
  // wavelength, so the three channels come to focus at slightly different
  // distances and out-of-focus HIGHLIGHTS pick up a colour rim that flips sign
  // across the focal plane: magenta in front, green behind. That flip is the whole
  // tell — lateral chromatic aberration fringes radially outward everywhere and
  // never reverses, which is why it reads as a defect and this reads as glass.
  //
  // Driven off the bloom pyramid rather than a per-wavelength trace, which would
  // mean three marches. The pyramid is already "the bright parts, blurred", and
  // that is precisely the energy this aberration is visible in, so the term is
  // applied exactly where it belongs and costs no extra sample. An approximation,
  // but not an arbitrary one.
  let focus = frame.camUp.w;
  // Signed, normalised defocus. Bounded because the background's 1e9 depth tag
  // must saturate the term, not explode it.
  let coc = clamp((depth - focus) / max(depth, 1e-3), -1.0, 1.0);
  col += halo * SPHERO_TINT * (coc * SPHERO_STRENGTH * frame.grade2.w);

  // Fireballs: fast-changing, so they belong here rather than in the accumulation buffer — a
  // detonation flash is over in a few frames and would smear into a streak.
  //
  // The atmosphere used to be here too, as `limbGlow`, for a reason that no longer holds: a
  // screen-space halo has no world position, so it could not be reprojected and left ghost wedges
  // drifting across the sky. It is a real scattering integral now, computed upstream where every
  // sample IS a world position — see volumetric.wgsl.
  let centre = cameraRay(rpx);
  // `depth` already carries the scene distance (1e9 for sky), so a blast on the
  // far side of the body is occluded by it for free.
  col += blastGlow(centre.o, centre.d, depth);

  // The heartbeat reaches the light itself, not just the geometry.
  col *= frame.flags.w * (1.0 + 0.10 * heartbeat(beatPhase()));

  // NATURAL VIGNETTING — the cosine-fourth-power law, and both of those words were wrong here
  // before: it was neither cos^4 nor applied where vignetting happens.
  //
  // An ideal thin lens darkens off-axis as cos^4 of the field angle, and the four powers are four
  // separate geometric effects: the entrance pupil foreshortens (one power), the image-plane
  // element tilts (one), and the off-axis image point sits further from the pupil (two, by the
  // inverse-square law). This is not a lens defect — it happens to a PERFECT lens, which is why
  // it is the right physical basis for a film look, distinct from the mechanical vignetting a
  // barrel adds. What was here, `1 - k * r^2`, is the first two terms of that law's expansion:
  // right at the centre, and flat where the real curve has already started falling.
  //
  // The field angle is exactly known, so no trigonometry is needed. `suv` is in half-diagonal
  // units and `camFwd.w` is the focal length in the same units, so tan(theta) = r / focal and
  //     cos^2(theta) = 1 / (1 + tan^2),   cos^4 = (cos^2)^2
  // which also means the term now RESPONDS TO THE FIELD OF VIEW, as a real lens does — a wider
  // lens vignettes harder, and that was invisible to the old form.
  //
  // `r` is measured from the OPTICAL AXIS, not from the middle of the frame. The field angle is
  // an angle off the lens's axis, and this camera composes off-centre — `CAMERA.frameOffset`
  // shears the projection — so the axis sits at LENS_AXIS in screen space, not at the origin.
  // Measuring from the origin displaces the whole falloff by about 58 device pixels, which puts
  // ~15% more darkening in one corner than the one opposite. Exactly the same mistake the lens
  // flare was making with its ghosts, in a second place; both read it from the same constant now.
  //
  // Applied in LINEAR LIGHT, before the tone curve, because that is where it physically happens:
  // it is an exposure falloff at the sensor, not a darkening of the print. Applied after the
  // display transfer — which is what this did — the corners get scaled in a space the curve has
  // already compressed, so they never roll through the toe and the falloff reads flatter still.
  let suv = screenUV(rpx) - LENS_AXIS;
  let tan2 = dot(suv, suv) / max(frame.camFwd.w * frame.camFwd.w, 1e-6);
  let cos2 = 1.0 / (1.0 + tan2);
  col *= mix(1.0, cos2 * cos2, frame.grade2.x);

  col = mix(vec3f(luma(col)), col, frame.grade.w);
  col = filmGrade(col);

  // Film grain, last, and stepped at 24fps so it reads as film rather than as
  // video noise — heavier in the shadows, the way real grain is.
  let g = hash13(vec3f(fragPos.xy, floor(frame.camPos.w * 24.0))) - 0.5;
  let l = clamp(luma(col), 0.0, 1.0);
  col += g * frame.grade2.y * (0.35 + 0.65 * (1.0 - l));

  return vec4f(max(col, vec3f(0.0)), 1.0);
}
