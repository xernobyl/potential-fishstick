// ---------------------------------------------------------------------------
// Concentric metal rings, instanced. No vertex or index buffer, no storage buffer:
// every ring's geometry AND its motion are functions of the instance index and the
// clock, the same stateless pattern the satellites and detonations use.
//
// Cross-section is a RECTANGLE, not a circle. A torus has no edges, and edges are
// what make a machined hoop read as metal: flat faces to catch a specular sweep, a
// hard rim to catch a highlight, and a defined direction for the scratches to run
// in. Four sides, so a ring costs majorSegments*4 quads.
//
// The motion is precession, not spin about the view axis. A smooth ring spinning
// about its own axis is invisible; what you see is the axis itself tumbling. The
// spin is still there and it IS visible, but only through the surface detail, which
// lives in the ring's local frame and therefore travels with it.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "ring_geom.wgsl"
//!include "volumetric.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "explosion.wgsl"
//!include "reflect.wgsl"

/// Rectangular cross-section, as (radial, axial) offsets scaled by half-width and
/// half-height. Two corners per side, plus that side's normal in the same basis.
const RC0 = array<vec2f, 4>(vec2f( 1.0,-1.0), vec2f( 1.0, 1.0), vec2f(-1.0, 1.0), vec2f(-1.0,-1.0));
const RC1 = array<vec2f, 4>(vec2f( 1.0, 1.0), vec2f(-1.0, 1.0), vec2f(-1.0,-1.0), vec2f( 1.0,-1.0));
const RN  = array<vec2f, 4>(vec2f( 1.0, 0.0), vec2f( 0.0, 1.0), vec2f(-1.0, 0.0), vec2f( 0.0,-1.0));

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
);

/// One ring's rigid frame plus its dimensions.

/// The ring's world position for a given vertex, at an arbitrary time. Factored out
/// so the vertex shader can evaluate it TWICE — now and one frame ago — which is what
/// makes an exact motion vector possible for rigid motion. No approximation needed:
/// the motion is analytic, so the previous position is too.
fn ringVertex(ii : u32, vi : u32, t : f32) -> vec3f {
  let r = ringDefAt(ii, t);
  let quad = vi / 6u;
  let corner = QUAD[vi % 6u];
  let seg = quad / 4u;
  let side = quad % 4u;
  let au = ((f32(seg) + corner.x) / f32(RING_SEGMENTS)) * TAU;
  let cs = mix(RC0[side], RC1[side], corner.y) * vec2f(r.halfW, r.halfH);
  let radial = r.ax * cos(au) + r.ay * sin(au);
  return radial * (r.radius + cs.x) + r.az * cs.y;
}

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp    : vec3f,
  @location(1) wn    : vec3f,
  @location(2) uv    : vec2f,   // x around the ring [0,1), y across the section
  @location(3) side  : f32,
  @location(4) viewZ : f32,
  @location(5) inst  : f32,
  @location(6) prevClip : vec4f,
  @location(7) prevViewZ : f32,
};

struct FOut {
  @location(0) colour : vec4f,
  /// xy: screen-space pixel delta to where this point was last frame.
  /// z:  its distance from the PREVIOUS camera, for TAA's depth gate.
  /// w:  its distance from the CURRENT camera — an ownership claim, since this pass
  ///     cannot see the body and may be writing over a pixel the body actually wins.
  @location(1) motion : vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  let r = ringDefAt(ii, frame.camPos.w);

  let quad = vi / 6u;
  let corner = QUAD[vi % 6u];
  let seg = quad / 4u;             // which segment around the ring
  let side = quad % 4u;            // which of the four faces

  // Major angle, spanning one segment.
  let u = (f32(seg) + corner.x) / f32(RING_SEGMENTS);
  let au = u * TAU;
  let cs = mix(RC0[side], RC1[side], corner.y) * vec2f(r.halfW, r.halfH);

  // Radial direction in the ring's plane, then out into world space.
  let radial = r.ax * cos(au) + r.ay * sin(au);
  let wp = radial * (r.radius + cs.x) + r.az * cs.y;
  let n2 = RN[side];
  let wn = radial * n2.x + r.az * n2.y;

  var out : VOut;
  out.pos = frame.viewProj * vec4f(wp, 1.0);
  out.wp = wp;
  out.wn = wn;
  out.uv = vec2f(u, corner.y);
  out.side = f32(side);
  out.viewZ = length(wp - frame.camPos.xyz);
  out.inst = f32(ii);
  // Where this same vertex was one frame ago, through the matrix the history was
  // rendered with. Interpolated as a vec4 and divided in the fragment shader, which
  // is what keeps it correct under perspective.
  let prevWp = ringVertex(ii, vi, frame.camPos.w - frame.misc.w);
  out.prevClip = frame.prevViewProj * vec4f(prevWp, 1.0);
  out.prevViewZ = length(prevWp - frame.prevCamPos.xyz);
  return out;
}

/// Fade a procedural octave out as its cells approach the pixel footprint.
///
/// Procedural detail has no mip chain. Nothing band-limits it, so any octave finer
/// than about two pixels aliases — and here it aliases with nothing to hide it,
/// because the rings are rasterised downstream of TAA and get none of the temporal
/// averaging the body leans on. So this does by hand what a texture sampler would
/// do for free: measure the footprint, drop the octaves that no longer fit in it.
///
/// `cells` is the octave's frequency in uv units; `footprint` is uv-per-pixel from
/// screen-space derivatives. Nyquist says an octave is only resolvable below 0.5
/// cells per pixel, so the fade is placed just under that.
fn octaveFade(cells : vec2f, footprint : vec2f) -> f32 {
  let cellsPerPixel = max(cells.x * footprint.x, cells.y * footprint.y);
  return 1.0 - smoothstep(0.22, 0.50, cellsPerPixel);
}

@fragment
fn fs(in : VOut) -> FOut {
  let V = normalize(frame.camPos.xyz - in.wp);
  var N = normalize(in.wn);
  let seed = in.inst * 13.7;

  // ---- surface, in the ring's own parameterisation -----------------------
  // The footprint of one pixel in uv. Safe to take derivatives of `uv` here: u runs
  // 0..1 around the hoop but the seam falls BETWEEN triangles, never across one, so
  // there is no discontinuity for fwidth to trip over.
  let foot = fwidth(in.uv);

  // Scratches run AROUND the ring, because that is the direction a hoop is turned
  // and polished in: low frequency along u so each mark is a long streak, higher
  // across v so the marks sit close together.
  //
  // Absolute numbers matter as much as the ratio. These were once ~768x higher along
  // u, which put them at up to 13 cells per PIXEL — 27x past Nyquist — and no ratio
  // saves detail that fine. They are now sized to the ring's actual screen extent
  // and faded out per octave when the camera pulls back.
  let fScratch = vec2f(34.0, 9.0);
  let fFine = vec2f(96.0, 22.0);
  let wS = octaveFade(fScratch, foot);
  let wF = octaveFade(fFine, foot);
  let scratch = vnoise(vec3f(in.uv.x * fScratch.x,
                             in.uv.y * fScratch.y + in.side * 7.0, seed));
  let fine = vnoise(vec3f(in.uv.x * fFine.x,
                          in.uv.y * fFine.y + in.side * 13.0, seed + 5.0));
  // Signed, so a scratch both polishes its floor and burrs its edges.
  let mark = (scratch - 0.5) * 1.4 * wS + (fine - 0.5) * 0.6 * wF;

  // Grunge lives in the ring's own parameterisation, not world space: the dirt is ON
  // the ring, so it has to travel with it as it turns.
  let fGrime = vec2f(31.0, 5.0);
  let wG = octaveFade(fGrime, foot);
  let grime = vnoise(vec3f(in.uv.x * 9.0, in.uv.y * 2.0, seed * 2.0)) * 0.65
            + vnoise(vec3f(in.uv.x * fGrime.x, in.uv.y * fGrime.y, seed * 3.0)) * 0.35 * wG;

  var f0 = RING_STEEL * (0.78 + 0.34 * grime);
  var rough = clamp(RING_ROUGH + 0.30 * grime - 0.22 * max(mark, 0.0), 0.05, 1.0);
  var alb = vec3f(0.0);                    // conductor: no diffuse lobe

  // Oxide and grime in the low spots: dielectric, dark, rough. Same contrast that
  // makes the corrosion on a real machined part obvious.
  let dirt = smoothstep(0.62, 0.86, grime);
  alb = mix(alb, RING_GRIME * (0.5 + 0.5 * grime), dirt);
  f0 = mix(f0, vec3f(0.04), dirt * 0.85);
  rough = mix(rough, 0.9, dirt * 0.8);

  // Scratches perturb the normal across their length only — a groove has no slope
  // along its run — so the perturbation goes into the ACROSS direction.
  let acrossN = normalize(cross(N, normalize(in.wp - dot(in.wp, N) * N) + vec3f(1e-5)));
  N = normalize(N + acrossN * mark * RING_SCRATCH);

  // Rims are rubbed clean and bright: the edges of a hoop are what contacts things.
  let rim = smoothstep(0.72, 1.0, abs(in.uv.y * 2.0 - 1.0));
  f0 = mix(f0, RING_STEEL * 1.15, rim * 0.55 * (1.0 - dirt));
  rough = mix(rough, 0.12, rim * 0.5 * (1.0 - dirt));

  // ---- lighting ----------------------------------------------------------
  var col = sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0)
          + blastLight(in.wp, N, V, alb, rough, f0);

  // The environment is most of what polished metal shows — and the body is the
  // biggest thing in it, so the reflection is marched rather than faked from the sky
  // alone.
  //
  // But ONE ray can only represent a MIRROR. On a rough or dirty patch the true lobe
  // is wide, and a single sample of it is both physically wrong and noisy: adjacent
  // pixels send their one ray down visibly different paths. So the sharp marched
  // reflection is faded toward the smooth sky term as roughness rises, which is the
  // cheap stand-in for prefiltering the environment — and the smooth term uses the
  // UNPERTURBED normal, so the scratches cannot jitter it either.
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  let flat = normalize(in.wn);
  var env = background(reflect(-V, flat));
  let sharpness = 1.0 - smoothstep(0.16, 0.52, rough);
  // Skipping the march once it is nearly faded out is a real saving, not just tidy:
  // the dirty, rough pixels are the majority.
  if (sharpness > 0.02) {
    let R = reflect(-V, N);
    env = mix(env, traceReflection(in.wp + N * 0.01, R), sharpness);
  }
  col += env * fresnelSchlickRough(NoV, f0, rough) * RING_ENV;

  // The core is a lamp inside the rings; inward faces catch it and it beats.
  let toCore = -normalize(in.wp);
  let d2 = max(dot(in.wp, in.wp), 1e-3);
  col += RING_CORE_COL * RING_CORE_GAIN * max(dot(N, toCore), 0.0) / d2
       * (0.55 + 0.45 * heartbeat(beatPhase())) * (alb + f0 * 0.5);

  col += (alb + f0 * 0.35) * RING_AMBIENT;

  // ---- the atmosphere in front of this hoop ----
  //
  // The rings orbit at 1.75-2.39 and the shell reaches 2.75, so there is genuinely air in front of
  // every one of them. Without this they would read CLEARER than the body behind them, which is
  // backwards. Integrated to this fragment's own depth, because each pass owns its own surface —
  // see the note on where volumetrics run in volumetric.wgsl.
  let vol = volumetric(frame.camPos.xyz, normalize(in.wp - frame.camPos.xyz), in.viewZ, in.pos.xy);
  col = col * vol.transmittance + vol.inScatter;

  var out : FOut;
  // Alpha carries linear view distance, which is how the merge pass resolves this
  // against the marched body. Strictly positive, since 0 means "nothing here".
  out.colour = vec4f(max(col, vec3f(0.0)), max(in.viewZ, 1e-4));

  // Motion, as a pixel delta. Behind the previous camera there is no valid history,
  // so emit the sentinel rather than a wild vector.
  if (in.prevClip.w > 1e-4) {
    let prevPx = uvToPixel(ndcToUV(in.prevClip.xy / in.prevClip.w));
    out.motion = vec4f(prevPx - in.pos.xy, in.prevViewZ, in.viewZ);
  } else {
    out.motion = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  }
  return out;
}
