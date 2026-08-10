import { int, wgslFloat as f } from '../core/wgsl.js';
import { layerRotations, wgslMat3, LAYER_SLOTS, SCHEDULE } from './lattice.js';

/**
 * Every art-direction number in one place.
 *
 * Split by concern, and deliberately the single source of truth: the values that
 * the shaders need are injected as WGSL `const`s (see `wgslDefines` below), so
 * there is never a copy of a constant that can drift out of sync with the JS.
 */

import { SHIP_MESH, SHIP_HARDPOINTS } from './ship_sdf.js';

export const QUALITY = {
  /** Internal render scale. Temporal accumulation carries the anti-aliasing, so
   *  half res is a genuine win rather than a visible downgrade. */
  renderScale: 0.5,
  maxWidth: 2560,
  /** Bloom mip levels for the dual-Kawase chain. 6 reaches a very wide glow. */
  bloomLevels: 6,
  /** Raymarch tile size, in pixels, for the indirect cull. Must match the
   *  workgroup size declared in raymarch.wgsl. */
  tile: 8,
  /**
   * Temporal UPSAMPLING: accumulate at display resolution instead of render resolution.
   *
   * Without it the pipeline renders at `renderScale`, accumulates at `renderScale`, and
   * upscales at the very end — so detail the jittered samples could have resolved at
   * display resolution is discarded before the upscale ever runs. With it the history
   * lives at display resolution and each frame's low-res samples are placed into it
   * according to where they actually landed, so the jitter walks them across the display
   * sub-pixels and the history converges on a genuine full-resolution image.
   *
   * Toggling this REALLOCATES targets, so it is not a per-frame switch — but it is a
   * switch, which is what makes the two comparable at all.
   *
   * ON, and the reasoning is measured rather than assumed. Two defects had to be fixed to get
   * here, and neither was where it first appeared to be.
   *
   * 1. The resolve exponentially blended a per-frame reconstruction, `acc / wsum`, which is an
   *    estimate located at the CENTROID of whichever samples landed nearby — and that centroid
   *    slides with the jitter. It now carries an accumulated WEIGHT and forms a true running
   *    weighted average, so samples accumulate at their own positions. Verified: capping that
   *    weight at 0 / 6 / 60 gave 20.0% / 9.1% / 9.4% residual, so it genuinely retains.
   *
   * 2. That alone did NOT fix it — 9.07% before, 9.07% after. The real limiter was the variance
   *    clip's neighbourhood BOX, built from a 2x2 reconstruction quad spanning about two output
   *    pixels and shifting membership with the jitter. Rebuilding it from a 3x3 of input taps
   *    (as FSR2 does) took the noise-free signal from 9.01% to 0.71% with the clip at FULL
   *    strength. Widening the clip instead had "worked" at 1.16% — a compensation, not a fix,
   *    the same mistake the depth-gate slack made earlier in this project's history.
   *
   * Where it lands, measured side by side on a common display-resolution grid:
   *
   *                        upsampling   1:1 + upscale
   *     pattern residual   0.696%       1.842%
   *     scene residual     1.293%       1.736%
   *     lag                0.51x        4.05x
   *     sharpness pattern  1.094        0.144
   *     sharpness scene    0.091        0.148
   *     taa pass           3.28 ms      0.46 ms
   *
   * On the noise-free signal — where detail and noise ARE separable — it resolves 7.6x more
   * detail at a third of the residual. That is the case upsampling exists for and it is
   * unambiguous. Lag is 0.51x, below the noise floor, i.e. no detectable trailing at all.
   *
   * ONE OPEN QUESTION, stated because it is the reason this is a judgement call and not a
   * proof: on the real scene the sharpness metric reads 39% LOWER for upsampling. That metric
   * cannot separate detail from noise, and upsampling also has 25% less residual, so the most
   * likely reading is "less noise" rather than "less detail" — but it is not established. A
   * controlled visual comparison would settle it; an attempt at one produced two frames that
   * turned out not to share a camera state, so it proved nothing and is not counted here.
   *
   * Cost is +2.8 ms on the TAA pass, roughly 10% of frame. If the softer scene reading turns
   * out to be real, this is one line to flip back.
   */
  taau: true,
  /**
   * Draw the ADDITIVE layer at display resolution rather than render resolution.
   *
   * ON, and on evidence rather than on principle — it took three measurements, and the first two
   * were not enough on their own.
   *
   * The layer gets no temporal antialiasing: it is drawn after the resolve, because reprojection
   * assumes static geometry and would ghost every mote. So at render resolution its aliasing gets
   * upscaled along with it. `beep.additive()` measured the difference that makes — 51% of the
   * layer's own high-frequency band wrong, 60% of its energy, 17% of its mean brightness. Large,
   * but it does not say which is BETTER: lower resolution is also blurrier, and blur suppresses
   * crawl rather than causing it.
   *
   * `beep.subpixel()` settled it, and disagreed with that caution: sliding the camera across one
   * pixel, the render-resolution layer is 2.2x worse on the largest jump between adjacent
   * sub-pixel offsets (0.171% against 0.076%) and 2.5x worse on mean-brightness wobble. It steps;
   * the display-resolution one drifts. So it is a genuine crawl source.
   *
   * Cost: none measurable. Six interleaved throughput runs put wall time at 31.7-32.8 ms in every
   * one, with the ember pass's own spread WITHIN a configuration wider than the gap between them —
   * this frame is bound by the march, not by additive fill. On much weaker hardware that will stop
   * being true, and this is one toggle in the panel.
   */
  additiveDisplayRes: true,
  /**
   * DYNAMIC RESOLUTION: hold a GPU-time target by moving `renderScale`.
   *
   * Off by default, because it trades a fixed known quality for a variable one and that should be
   * a choice. It is only worth having at all because of temporal upsampling — see dynres.js — and
   * it is unusually effective here because the frame is march-bound and the march is per-pixel.
   */
  dynamicRes: false,
  /** GPU milliseconds to aim for. GPU, not wall: wall includes present pacing, so a vsync-limited
   *  frame reads as exactly on budget however much headroom there is, and nothing would ever
   *  climb. 14 ms is about a 70 fps budget with room for the parts that do not scale. */
  dynamicTargetMs: 14,
  /** The rungs, as STEPS. Few and decisive rather than continuous: every change costs a
   *  reallocation of the render-resolution targets and a discontinuity in sample density, so fine
   *  tracking buys nothing and jitters the image. See dynres.js. */
  dynamicLadder: [
    { scale: 1.0, additive: true },
    { scale: 0.85, additive: true },
    { scale: 0.7, additive: true },
    { scale: 0.6, additive: true },
    { scale: 0.5, additive: true },
    { scale: 0.42, additive: true },
    { scale: 0.35, additive: true },
    // Last resort: keep the render scale and give up the additive layer's resolution instead.
    // Measured at 2.2x worse sub-pixel stability, which is why it is the bottom rung and not a
    // step on the way down.
    { scale: 0.35, additive: false },
  ],
  /** Frames of GPU time to median over. Frame times here are spiky — a detonation or a ring
   *  sweeping into view moves the march by milliseconds — and a single sample oscillates. */
  dynamicWindow: 15,
  /** Frames to hold still after a change, so a reallocation cannot happen every frame. */
  dynamicCooldown: 30,
  /** Asymmetric thresholds, as fractions of the target. Drop as soon as it is over; climb only
   *  with real headroom. Being briefly too slow is worse than being briefly too soft, and
   *  climbing into a spike then falling straight back out is what makes this feel worse than a
   *  fixed low setting. */
  dynamicDropAt: 1.05,
  dynamicRaiseAt: 0.82,
};

export const BODY = {
  R: 2.0,
  octaves: 3,
  n0: 20.0,
  nGrow: 4.0,
  rho0: 0.92,
  rhoFall: 0.5,
  jitter: 0.2,
  detail: 0.06,
  /** Lattice-window size for the SUBTRACTED hole layers. 5 (centre + edges) rather
   *  than the 9 the additive layers need: see the note on SFWIN in fibonacci.wgsl. */
  holeCandidates: 5,
  holeOctaves: 3,
  holeDen: 0.5,
  holeScale: 0.95,
  holeDenGrow: 1.6,
  holeScaleFall: 0.8,
  holeOffset: 0.35,
};

/**
 * How the body's field is traced. The raymarch is ~89% of the frame, so this block is
 * the renderer's performance budget in one place.
 *
 * The central fact, and the reason the step scale cannot be a single number: this field
 * is 1-Lipschitz in the far region and NOT 1-Lipschitz near the surface. `mapBody` mixes
 * the detail noise in only within 0.20 of the surface, and that noise adds roughly 0.3 to
 * |grad d| — so inside the band the reported distance is not a lower bound on the true
 * distance and the ray must under-step or it creases the surface. Outside it, the field is
 * a plain smooth-min of exact sphere distances, properly 1-Lipschitz, and can be marched
 * at full length or beyond.
 */
export const MARCH = {
  /** Hard step cap. */
  steps: 160,
  /** Step scale in the far region. A safety factor rather than 1.0 because the smin
   *  itself is only *approximately* distance-preserving.
   *
   *  `omega` used to sit beside this, driving safe over-relaxation. Removed — it moved the
   *  measured evaluation count from 12.36 to 12.41 per pixel, i.e. nothing. */
  far: 0.92,
  /** Step scale inside the noise band. Not over-relaxable at any factor: the premise of
   *  the safety check is that the reported distance bounds a sphere containing no
   *  surface, and in here it does not. */
  near: 0.65,
  /** Where the noise band starts. Must stay at or above the 0.20 fade-in width in
   *  `mapBody`, or steps taken at the far scale reach into un-bounded field. */
  nearBand: 0.48,
  /** Hit threshold, relative to distance travelled — a pixel-footprint proxy. */
  hitEps: 0.0004,
};

/**
 * MEASURED NEGATIVE RESULTS, kept because they redirect the next attempt at this.
 *
 * The march is ~89% of the frame, so the obvious target is the number of steps, and the
 * standard tool for that is a coarse world-space distance volume: evaluate a conservative
 * lower bound on a grid once per frame, then let every ray stride through empty space with
 * a texture fetch instead of a field evaluation. It was built (64^3 r32float, penalised by
 * sqrt(3)*h for trilinear error and by DETAIL*0.5 for the grain it does not model) and it
 * was verified CORRECT: converged images with the volume on and off differed by less than
 * the sampling noise between two frames of the same config, which is what a genuinely
 * conservative bound must do.
 *
 * It did not pay. Raymarch 22.9-23.3 ms with it against 23.1 ms without — inside the
 * noise — plus 0.8 ms to build. Removed.
 *
 * Two reasons, and the second is the one that matters:
 *
 *  1. An over-relaxed field step is LONGER than the volume's bound. The field steps
 *     omega*d; the volume offers d - 0.145. For the volume to stride further it would need
 *     d - 0.145 > omega*d, which is never true. Its only advantage is cost per step, not
 *     distance per step.
 *  2. The march is not step-count-bound at all. Sweeping `nearBand` — which directly
 *     controls how much of each ray is eligible for long far-field steps — moves the
 *     raymarch by under 3%, and moves it the SAME direction whether the band is widened to
 *     0.50 or narrowed to 0.10, which is only possible if the differences are measurement
 *     noise. The step scales here are therefore close to irrelevant to cost.
 *
 * Two more attempts followed, and both also measured as noise: safe over-relaxation
 * (Keinert et al. 2014), and precomputing the layer rotations into a lookup table instead of
 * building them with sin/cos per call. Over-relaxation gave 12.41 evals/pixel against 12.36
 * without it. The table won 3 of 5 paired runs, median 0.83%.
 *
 * At that point the cost model had failed three times, so it got MEASURED instead
 * (PROBE.showFieldEvals counts field evaluations in the shader; the mean comes back over
 * the wire). The answer:
 *
 *   mean field evaluations per pixel   12.4        <- not the ~60 that was assumed
 *   pixels at 1280x720 render res      921,600
 *   field evaluations per frame        11.4M
 *   raymarch                           23.9 ms
 *   => cost per field evaluation       2.1 ns
 *
 * There is no step-count headroom at all: the march already converges in about twelve
 * evaluations, and every knob that could change that moves it by well under 1%. All three
 * optimisations were aimed at a number that was already minimal.
 *
 * The cost is entirely INSIDE the evaluation. Each one walks 3 octaves x 9 lattice
 * candidates plus 3 hole layers x 5 = 42 candidates, so a frame performs ~480M candidate
 * evaluations — and `sfCandidate` computes each candidate's world direction with cos, sin
 * and sqrt BEFORE `layerDist` gets to apply its cheap rejection test. All 480M pay for
 * transcendentals; most are then discarded.
 *
 * THE lever, therefore: reject a candidate before computing its azimuth. The polar term
 * costs nothing (ict = m - 2i/n), and |dir - cw|^2 = 2 - 2*dot(dir, cw) can be bounded from
 * below using only it, because |lp.x*cos + lp.y*sin| <= hypot(lp.x, lp.y) — which sfCell
 * already holds as sqrt(s2). A candidate rejected on that bound never needs its cos/sin.
 * Anything aimed at step count instead should be expected to fail.
 */

/**
 * Spherochromatism — the residual spherical aberration of a fast lens varying with
 * wavelength, so the channels come to focus at slightly different distances.
 *
 * The visible signature is a colour rim on OUT-OF-FOCUS HIGHLIGHTS that flips as
 * you cross the focal plane: magenta-ish in front, green-ish behind. That sign
 * flip is what distinguishes it from plain lateral chromatic aberration, which
 * just fringes radially outward everywhere.
 */
export const SPHERO = {
  strength: 0.55,
  /** Per-channel focus bias, and the SIGN matters. The shader adds
   *  `tint * coc`, and coc is negative in front of the focal plane, so a negative
   *  red/blue with a positive green is what puts magenta in front and green
   *  behind — the usual look of a fast, partially-corrected prime. Flip all three
   *  to swap which side is which. */
  tint: [-1.0, 0.85, -0.75],
};

/**
 * Three concentric metal rings on precessing pseudo-random axes.
 * Stateless — see shaders/rings.wgsl: geometry and motion are both functions of the
 * instance index and the clock.
 */
/**
 * Raymarched reflections on metal. Costs are per REFLECTING pixel whose reflection
 * ray actually enters the body's bounding sphere, which is a small share of a small
 * share of the frame.
 */
export const REFLECT = {
  /** March budget. Short on purpose — a reflection in scratched, curved metal does
   *  not survive the detail that more steps would buy. */
  steps: 28,
  /** Octaves of the body's field to march. 1 = the structural layer only. */
  octaves: 1,
};

export const RINGS = {
  count: 3,
  /** Segments around the hoop. 128 keeps the silhouette smooth at full screen while
   *  costing 3 x 128 x 4 quads, which is nothing. */
  segments: 128,
  /** Innermost radius, then a fixed step outward so they nest concentrically.
   *  r0 must clear the body's bound (~1.5) and the outermost must stay inside the
   *  camera's closest approach (CAMERA.distance - CAMERA.zoom = 3.85). */
  radius0: 3.5,
  gap: 0.64,
  width: 0.11,        // radial half-thickness
  height: 0.22,        // axial half-height
  /**
   * Per-ring taper. Widths shrink outward, which reads as a hierarchy.
   *
   * Here rather than as literals inside `ringDefAt` because the CPU mesh generator needs the same
   * numbers: the geometry is baked per ring, so a drift between the shader's idea of a ring's size
   * and the generator's would show up as the wrong-sized hoop, or as a motion vector describing a
   * ring that is not the one on screen. Injected into WGSL and read by `ringDims` below.
   */
  widthTaper: 0.18,
  heightTaper: 0.12,
  precess: 0.045,      // axis tumble rate, rad/s
  spin: 0.17,          // rotation about its own axis; visible via surface detail
  roughness: 0.3,
  steel: [0.55, 0.56, 0.575],
  grime: [0.13, 0.105, 0.085],
  scratch: 0.22,       // normal perturbation from tool marks
  env: 0.55,
  coreColour: [1.1, 0.34, 0.1],
  coreGain: 0.4,
  ambient: 0.03,
};

/**
 * The rasterised-opaque-geometry layer itself, as distinct from anything currently
 * drawn into it. Lives here rather than under RINGS because it governs how ANY solid
 * composites — a second one should not have to look under the first one's name.
 *
 * It used to carry a `glowThrough` fudge that attenuated the body's halo behind
 * occluding metal. That is gone, and the reason is worth recording: once TAA resolved
 * the solid layer BEFORE accumulating, the solids landed in the buffer the bloom
 * pyramid prefilters from, so they occlude the glow for real. The knob had nothing
 * left to do.
 */
export const SOLID = {
  /** Sentinel written to the motion target where no valid history exists. Chosen far
   *  outside any real pixel delta but well inside f16's range. */
  motionNone: 10000.0,
};

/**
 * The player ship: a 2D arcade game on the surface of a sphere. Two DOF of position,
 * one of heading, no roll or pitch freedom — the orientation is DERIVED from the
 * surface normal and the heading, so the hull cannot tumble. See src/scene/ship.js.
 */
export const SHIP = {
  /** Shell radius. Outside the rings (max 2.39), inside the satellites (min 2.46). */
  orbit: 4.84,

  /** Turn authority about the local normal, rad/s^2, and the damping that caps the
   *  rate. Torque-driven rather than rate-driven, so the ship winds into a turn and
   *  coasts out of it instead of snapping. */
  turnAccel: 4.2,
  turnDamp: 2.6,

  /** Forward and reverse authority, and the drag that sets terminal speed.
   *  Reverse is deliberately weaker — it is for backing off, not for flying. */
  thrust: 2.2,
  reverseThrust: 1.1,
  linDamp: 0.85,
  throttleRate: 6.0,

  /** Cosmetic lean, radians per rad/s of turn rate, and how fast it follows. Never
   *  feeds back into the motion. */
  bankPerTurn: 0.34,
  /** Barrel-roll rate while cruising, rad/s. About five seconds a revolution — slow enough to
   *  read as deliberate rather than as a tumble, and it puts the contrails through a helix. */
  rollRate: 1.1,
  /** Seconds between automatic shots while cruising. The rail gun triggers on a rising EDGE, so
   *  the renderer pulses it for one frame per period; see there for why holding it would not
   *  work. */
  autoFireEvery: 2.4,
  bankRate: 5.0,
  /** Scales turn acceleration into what the RCS puffs show. */
  rcsFromAccel: 0.5,
  /**
   * Plume brightness, now that the plumes are billboards in the additive layer rather than glow
   * summed inside the march.
   *
   * The two are not the same integral. The marched form accumulated its Gaussian along the ray and
   * was attenuated by everything the march applies afterwards — the atmosphere's in-scattering, the
   * resolve; a billboard is added straight onto the resolved image and keeps all of it. Same shape,
   * more of it, so at 1.0 the nozzles read as two white spheres instead of two flames.
   */
  jetGain: 0.42,

  /**
   * Overall size: body units per world unit.
   *
   * OWNED BY `ship_sdf.js`, not written here. The hull is authored at scale 1 and the mesher applies
   * this once; the plumes below are authored in the same units and divide it out the same way. Two
   * copies of this number is two ships — the hull one size and the exhaust another — and the drift
   * only shows up as plumes that no longer start at the nozzles. So there is one.
   *
   * The bounding sphere and step count that used to sit here went with the marched hull: a mesh is
   * bounded by the sphere `concatMeshes` measures from its own vertices, which cannot be wrong.
   */
  scale: SHIP_MESH.scale,
  panel: 9.0,
  hull: [0.30, 0.32, 0.36],
  glass: [0.03, 0.05, 0.09],
  trim: [0.62, 0.14, 0.10],
  env: 0.45,
  ambient: 0.05,
  coreColour: [0.9, 0.32, 0.12],
  /** Engine flame. Hot core, so it reads as fire rather than as a blue glow. */
  jetColour: [1.5, 0.72, 0.28],
  jetLength: 5,
  rcsColour: [0.9, 0.95, 1.1],
  rcsGain: 0.6,
};

/**
 * The ship's contrail. The only stateful thing in the scene — see src/scene/contrail.js
 * for why a trail cannot be a closed-form function of time the way everything else here
 * is.
 */
export const CONTRAIL = {
  /** Ribbon samples. Longer trail, linearly more geometry — but it is a few dozen
   *  quads, so length is a look decision rather than a cost one. */
  samples: 56,
  /** Seconds between samples. Fixed, so spacing is proportional to SPEED: a fast ship
   *  draws a long trail and a stopped one lets its trail collapse and fade. */
  interval: 0.045,
  /**
   * Emitted this far behind the hull, and this far out to each side.
   *
   * DERIVED FROM THE HULL, not typed in beside it. These are the nacelle hardpoints in
   * `ship_sdf.js`, scaled to world units — the same numbers the nozzles are modelled at and the
   * plumes fire from. They used to be literals in three files and had already drifted apart.
   */
  get offset() { return -SHIP_HARDPOINTS.nozzleZ * SHIP_MESH.scale; },
  get spread() { return SHIP_HARDPOINTS.nacelle[0] * SHIP_MESH.scale; },
  /** Vertical offset of the nacelle from the spine — the third component the emit used to drop. */
  get rise() { return SHIP_HARDPOINTS.nacelle[1] * SHIP_MESH.scale; },
  /** Half-width at emission, and how much it disperses over the trail's life. */
  width: 0.012,
  widthGrow: 0.055,
  /** Fade exponent. Above 1 keeps the trail bright near the ship and drops it away
   *  quickly, which reads as vapour rather than as a painted line. */
  falloff: 2.1,
  gain: 1.5,
  hot: [1.05, 0.72, 0.42],
  cold: [0.30, 0.42, 0.72],
};

/**
 * Auroras: ribbons on a curl-noise flow, arriving and dissolving.
 *
 * Shares the contrail's geometry entirely — see shaders/aurora.wgsl. Everything here is either
 * about the FLOW that generates the path or the EMISSION that colours it.
 */
export const AURORA = {
  ribbons: 9,
  /** Samples per ribbon. Long, because these are curtains stretched across the sky rather than
   *  a wake behind something — and because the ribbon is seeded whole, length costs geometry
   *  but no warm-up time. 160 quads x 5 ribbons is still nothing. */
  samples: 160,
  /** Seconds between samples. Fixed, so spacing follows flow speed exactly as the contrail's
   *  follows ship speed. See `speed` — their product is the sample spacing, and it is the
   *  product that was held constant when speed came down. */
  interval: 0.22,

  // ---- the flow ----
  /** World-space frequency of the noise the curl is taken from. Lower is broader, lazier
   *  swirls; higher braids tightly and starts to look like turbulence rather than aurora. */
  curlScale: 0.17,
  /** How fast the field itself evolves. Non-zero is what stops the ribbons settling into fixed
   *  streamlines — with a static field a walker traces the same curve forever. */
  curlDrift: 0.08,
  /** Central-difference epsilon for the curl. Small enough to be a derivative, large enough
   *  that f64 cancellation does not dominate: at 1e-4 the two lattice samples differ in their
   *  last few bits and the curl comes out as noise. */
  curlEps: 0.01,
  /**
   * World units per second along the flow.
   *
   * Set together with `interval`, because their PRODUCT is the sample spacing and therefore
   * the ribbon's total length: samples * speed * interval. At 0.32 and 0.055 a ribbon measured
   * 1.08 units end to end while being up to 0.37 wide — a blob, not a curtain. These give
   * ~9 units of path for a ~0.2 half-width, which is the proportion that reads as a sheet.
   *
   * Speed also sets the path's MINIMUM TURN RADIUS, at speed / maxTurn, and that is what pinned
   * it here. At 0.5 the radius was 0.63 against a shell only 0.7 thick — a walker could barely
   * turn inside its own container, so it overshot the faces and the hard clamp creased the path
   * on every approach. Trading speed for interval fixes it at no cost to anything visible: the
   * SPACING is the product and is unchanged, so the ribbon is the same length and the same
   * smoothness, it just drifts more slowly. Which, for an aurora, is the better read anyway.
   */
  speed: 0.5,

  // ---- where they live ----
  /**
   * Radial shell they are held within — outside the outermost ring (2.75), so the ribbons weave
   * around the whole assembly rather than through the hoops.
   *
   * Deliberately NOT sized to keep them in front of the camera. That was the original reasoning
   * and it does not survive contact with the chase camera, which sits 3.4 units behind a ship
   * that goes wherever it is flown: there is no radius that is reliably in front of it. A near
   * fade in the shader is the honest fix and covers both cameras — see `nearFade`.
   */
  shellMin: 5.8,
  shellMax: 7.2,
  /**
   * How far from each face the vector potential is ramped down to zero, in world units.
   *
   * This is the containment, and it is worth being clear about how little else there is: the
   * flow is the curl of the potential, so a potential that vanishes at a face leaves a velocity
   * with NO radial component there (Bridson 2007 s.3 — see `#step`). Walkers do not need to be
   * pushed back onto the shell because they were never able to advect off it.
   *
   * Two of these have to fit inside the shell's thickness with room to spare, or alpha never
   * reaches 1 and the flow is tangential everywhere — ribbons flattened onto a sphere. At 0.7
   * thickness and 0.25 the middle 0.2 is untouched curl noise.
   *
   * What came before, for the record. A soft radial pull bounds nothing: over 20 s the walkers
   * reached 2.08 and 5.55 against this shell, because at 0.5 u/s the flow carries them further
   * in one time constant than the pull recovers — 2.08 is inside the outermost ring and 5.55 is
   * behind the camera. A hard clamp did bound it, and put a visible kink in the path at every
   * contact, because moving a sample sideways IS a corner. Fixing the field beats correcting
   * the walker.
   */
  shellRamp: 0.5,
  /**
   * Radial steering, as the weight of the radial term against the unit flow direction, per unit
   * of `shellRamp` past the point where alpha starts falling. At 3.0 the desired heading is
   * most of the way to radial by the time the walker reaches a face.
   *
   * The field does the containment; this covers the two ways the WALKER differs from the flow.
   * Its heading trails the flow by up to one turn-rate step, and the tangential term vanishes
   * wherever the potential lines up with the radius, leaving a momentarily null flow that the
   * walker coasts through. Both are small, and both were enough to put the hard clamp — and its
   * 35 degree corner — on every approach to a face.
   */
  shellSteer: 3.0,
  /** Where the steering saturates, in units of `shellRamp` past the ramp's start. Past this
   *  there is nothing more to gain: the heading is already as inward as it is going to get. */
  shellSteerMax: 2.0,
  /** The derivative half of the same controller: how hard an OUTWARD heading inside the ramp
   *  region is steered against, regardless of position. This is what bounds the overshoot —
   *  position alone reacts a turn radius too late, so the walker sailed a third of a unit past
   *  the face on every approach. */
  shellDamp: 6.0,
  /** Hard bound, as slack past the shell — and the ramp length for the recovery steering. A
   *  backstop, not a mechanism: it fires only if both the field and the steering have failed,
   *  and it is kept only so a pathological flow cannot put a curtain behind the camera. */
  shellSlack: 0.7,
  /**
   * Maximum turn rate along the path, in RADIANS PER SECOND. The single parameter that fixed
   * the sharp-turn artefacts, and it fixes them by construction rather than by handling them.
   *
   * The walker used to take its heading straight from the flow (`p += normalize(curl) * v * dt`),
   * which leaves the path's curvature unbounded — and worst exactly where the field is weakest,
   * since normalising a curl passing through zero turns a tiny noisy vector into a wildly
   * swinging direction. A ribbon cannot be expanded through a near-180-degree turn by any
   * method; the miter limit only decides how it fails.
   *
   * Bounding the turn rate bounds the path's MINIMUM RADIUS at speed / maxTurn = 0.31 units,
   * against a maximum half-width of width + widthGrow = 0.225. Above the half-width, so the
   * inside of a turn never folds through itself, and far enough below the shell's 0.7 thickness
   * that a walker can turn around inside its own container — which is what the containment
   * needs and what a larger radius was quietly breaking. The measured joint angle stays under
   * 10 degrees, i.e. a 1.004x miter, against a limit that only engages at 139.
   */
  maxTurn: 0.8,
  /** Seconds for one appear-and-dissolve cycle. Each ribbon is offset within it, so something
   *  is always arriving while something else is fading. */
  lifetime: 26.0,

  // ---- emission ----
  /**
   * Half-width at the head, and how much it opens out along the ribbon.
   *
   * Wider than the contrail because a curtain is not a filament — but only a few times wider,
   * not an order of magnitude. At 0.05/0.30 they rendered as solid blades sweeping across the
   * frame: the shell sits between the camera and the body, so these are FOREGROUND objects and
   * a width that looks reasonable in world units is enormous on screen.
   */
  width: 0.05,
  widthGrow: 0.4,
  /** Fade exponent along the ribbon. Gentler than the contrail's 2.1 — an aurora's brightness
   *  varies slowly along the curtain instead of dropping off behind a nozzle. */
  falloff: 1.9,
  gain: 0.7,
  /**
   * The palette. Nine hues, and no longer the auroral emission spectrum.
   *
   * It used to be the real lines — 557.7 nm oxygen green, 630 nm oxygen red, 427.8 nm ionised
   * nitrogen — and that was the right call while the point was for these to read as auroras. The
   * point now is that there are nine of them and they should be distinguishable, so the constraint
   * is spacing rather than physics.
   *
   * Generated rather than picked, because picking by eye gets the brightness wrong. Luma is 0.71
   * green and 0.07 blue, so a saturated blue at the same nominal value is a tenth as bright as a
   * saturated green — a hand-chosen rainbow has shouting greens and invisible violets. These are
   * evenly spaced in hue and PARTIALLY equalised in luma: full equalisation would need a blue nine
   * times brighter than the green, which is neither achievable nor pleasant, so the correction is
   * raised to 0.55 and the spread lands at 0.38..0.71 instead of 0.13..0.89.
   *
   * Values above 1 are deliberate. This layer is additive HDR and goes through a tone curve that
   * desaturates highlights, so a saturated input is what survives to look saturated.
   *
   * Nine entries for nine ribbons, so `ii % N` gives every ribbon its own hue with none repeated.
   */
  palette: [
    [1.30, 0.29, 0.16],
    [0.87, 0.70, 0.10],
    [0.50, 0.84, 0.10],
    [0.11, 0.90, 0.20],
    [0.10, 0.87, 0.71],
    [0.13, 0.66, 1.09],
    [0.41, 0.22, 1.83],
    [1.15, 0.17, 1.41],
    [1.39, 0.17, 0.83],
  ],
  /**
   * How far the curtain's dim edges shift in hue, 0..1.
   *
   * An AMOUNT now, not a colour. Real auroras shift hue across their width because different
   * species emit at different altitudes, and a fixed fringe colour reproduced that — but with a
   * nine-hue palette it would also tint every ribbon the same and erase the variety the palette
   * exists for. The shift is derived from each ribbon's own tint instead: a channel rotation IS a
   * 120-degree hue rotation, so it is always a different hue and never a muddy one.
   */
  fringe: 0.55,
  /** Depth of the ray striation, 0..1. Real auroras are bundles of field-aligned rays and that
   *  vertical structure is most of what identifies them — but it modulates, it does not shred,
   *  so this stays well below 1. */
  rays: 0.45,
  /** Rays per sample along the ribbon. Below 1 so a ray spans several samples and reads as a
   *  soft band rather than as per-sample flicker. */
  rayFreq: 0.35,
  /** How fast the ray pattern drifts along the curtain, in bands per second. Auroras are never
   *  still; a static striation reads as a texture painted on a solid. */
  rayDrift: 0.55,
  /**
   * Miter limit, as the minimum cos(theta/2) the joint may be divided by.
   *
   * The bisector offset is scaled by 1/cos(theta/2) to keep the ribbon's width constant through
   * a turn, and that factor runs to infinity as the turn approaches 180 degrees. 0.35 caps the
   * widening at ~2.9x, past which the joint creases instead of launching a spike across the
   * frame — the standard trade, and the reason every line renderer has this parameter.
   */
  miterMin: 0.35,
  /**
   * Near fade, as the view distances over which a ribbon goes from invisible to full.
   *
   * These are foreground objects on a shell the camera can get close to — and in chase mode can
   * pass straight through, since that camera follows the ship rather than orbiting the body. A
   * curtain a few centimetres from the lens is a full-screen additive blade, and no amount of
   * shell tuning prevents it, because in chase mode there is no radius that is reliably in
   * front of the camera. Fading by distance handles every case, including the one where the
   * camera flies through a ribbon, and it costs a smoothstep.
   */
  nearFade: [1.1, 2.8],
  /**
   * Where the ribbon fades out as it turns toward the viewer, in sin(angle between the path and
   * the view ray). Below this the alpha ramps linearly to zero.
   *
   * Two jobs, and it is the same number for both, which is the argument for doing it this way.
   *
   * It is the CORRECT RADIOMETRY: a segment pointing at the viewer projects to almost nothing,
   * so without compensation the same emitted energy lands in a vanishing area and the ribbon
   * blows out to a sliding bright blob. Projected area scales with that sine, so alpha scaling
   * with it holds the energy constant. Linear, not smoothstepped — linear IS the correction,
   * and a smoothstep goes as the square near zero and over-darkens.
   *
   * And it is the fix for the camera-facing DEGENERACY, since it reaches zero exactly where
   * `cross(tangent, toCam)` stops having a direction. That is what makes the arbitrary fallback
   * in `ribbonSide` unobservable rather than a pop.
   */
  grazeFade: 0.35,
};

/**
 * Rail gun. Instant beams that fade, alternating between the wing tips.
 * Fired on the KEY EDGE, never while held — see src/scene/railgun.js.
 */
export const RAIL = {
  /** Shots in flight at once. Small: they last under a second, and a deep pool would
   *  only matter if the key could be held, which it deliberately cannot. */
  pool: 8,
  segments: 40,
  /** Seconds a beam takes to fade out. */
  life: 0.55,
  length: 9.0,
  /**
   * Muzzle in the hull's OWN axes, so it tracks the ship exactly.
   *
   * DERIVED FROM THE HULL. This was the literal 0.348, taken from a marched hull whose wings reached
   * x = 0.61 authored; the SDF hull that replaced it reached only 0.40, so the muzzles had been firing
   * from 0.178 units past the wingtip ever since — visibly detached, and nothing pointed at it because
   * both numbers looked deliberate. There are gun pods at the tips now and the beams leave their noses.
   */
  get spread() { return SHIP_HARDPOINTS.wingTip * SHIP_MESH.scale; },
  up: -0.005 * 0.6,
  get forward() { return (SHIP_HARDPOINTS.podZ + 0.16) * SHIP_MESH.scale; },
  /** The helix: how many turns over the beam's length, its radius, and how fast the
   *  strands keep rotating as the shot decays. The spin is what reads as "spun up a
   *  rail" rather than "drawn as a line". */
  turns: 7.0,
  radius: 0.055,
  spin: 5.5,
  width: 0.012,
  falloff: 1.7,
  gain: 2.6,
  hot: [1.1, 0.85, 1.35],
  cold: [0.32, 0.55, 1.15],

  // ---- the POWER SHOT ----
  //
  // Hold the trigger, let go, and something much bigger leaves the gun. The charge is integrated on
  // the renderer's own clock rather than a wall clock, so freezing or pausing the scene freezes the
  // charge with it — one clock, and the instruments keep working.

  /** Seconds of holding to reach a full charge. Long enough to feel like a commitment. */
  chargeTime: 1.6,
  /**
   * How far in you have to be before a release counts as a power shot rather than a tap.
   *
   * Not zero, and not close to it: every ordinary shot ends with a release too, so this is the whole
   * boundary between "I fired" and "I charged". A quarter of the way in is past any accidental hold.
   */
  chargeMin: 0.25,
  /** Multipliers at FULL charge. A power shot is the same beam, drawn much harder. */
  powerLength: 2.6,
  powerRadius: 9.0,
  powerWidth: 2.4,
  /**
   * Brightness multiplier at full charge.
   *
   * MODEST, because the bloom does the drama. Driven at 2.6 the core clipped to white, the pyramid
   * spread it across a third of the frame, and the spirals — the thing that makes the shot read as a
   * power shot rather than a bright line — disappeared inside their own glow. The beam should be the
   * brightest thing on screen without being the only thing on screen.
   */
  powerGain: 1.45,
  powerLife: 2.1,
  /** Strands in the helix. Two for a normal shot; the rest spiral in with charge. */
  strands: 6,
  /**
   * Turns, relative to an ordinary shot's.
   *
   * BELOW ONE, and that is the opposite of the first guess. A power shot is nearly three times longer,
   * so keeping the same turns-per-unit wound thirteen turns into it and the spiral aliased into a
   * bright tube. Fewer, slower turns over a longer beam is what actually reads as a spiral.
   */
  powerTurns: 0.42,

  // ---- sparks ----
  /** Spark billboards per shot. They cost one collapsed vertex each when unused. */
  sparks: 28,
  sparkLife: 0.9,
  sparkSpeed: 3.4,
  /**
   * Billboard radius, world units.
   *
   * SMALL. The first pass had these at 0.055 growing to 0.107 on a power shot, which is a third of the
   * hull's radius — they read as floating orbs rather than as sparks, and a spark that is not small is
   * just a light. A tenth of the hull is about right: individually tiny, collectively a burst.
   */
  sparkSize: 0.018,
  /** Sideways spread of the burst, as a fraction of the forward speed. */
  sparkSpread: 0.85,
  sparkGain: 3.6,
  /** Sparks drawn while CHARGING, spiralling inward to the muzzle. */
  chargeSparks: 20,
  chargeRadius: 0.45,
  chargeGain: 2.2,

  /**
   * Peak angular kick at full charge, radians.
   *
   * 0.13 is about 7.5 degrees, which sounds enormous written down and is roughly right: the first
   * value here was 0.035, two degrees, and it measured correctly while being completely invisible on
   * screen. A kick has to be a decent fraction of the frame to read as an impact rather than as a
   * dropped frame, and it is gone in under half a second either way.
   *
   * Only the SIZE of the kick lives here, because that is a property of the weapon. How the camera
   * responds to being kicked — how fast it settles, how quickly it wobbles — is the camera's, and sits
   * in CAMERA beside the rest of its behaviour.
   */
  shake: 0.13,
};

export const MEDIUM = {
  depth: 1.6,        // local thickness -> optical depth
  density: 4.5,      // how hard the medium absorbs its complement
  g: 0.72,           // Henyey-Greenstein anisotropy
  ior: 1.44,
  disperse: 0.022,
  refrGain: 1.15,
};

export const CORE = {
  radius: 1.56,
  attenuation: 1.6,
  colour: [1.0, 0.2, 0.045],
  intensity: 5.4,
};

export const PULSE = {
  bpm: 42.0,
  radius: 0.112,     // radius swell per beat (heartbeat peaks at exactly 1)
  offset: 0.06,      // radial breathing per beat
  lag: 0.4,          // spread of per-sphere phase offsets
  lifeRate: 0.035,   // sphere birth/death cycles per second
};

/**
 * Surface detonations. Stateless — see explosion.wgsl: each slot's firing index is
 * floor(time/period), which seeds both where it happens and whether it happens.
 */
export const BLASTS = {
  /** Slots, each firing at most once per period. More slots = busier sky. */
  count: 4,
  /** Seconds between a slot's opportunities to fire. */
  period: 5.2,
  /** Fraction of opportunities that actually detonate. Below 1 so the rhythm is
   *  sporadic rather than a metronome — with 4 slots at 5.2 s and 0.55, that
   *  averages a bang every ~2.4 s. */
  chance: 0.55,
  /** Life as a fraction of the period. The slot is dormant for the remainder. */
  life: 0.2,
  /** Radius the fireball expands to, world units (the body's radius is ~1). */
  radius: 0.42,
  /** Where they sit. Just outside the nominal surface, so the fireball bulges out
   *  of the lobes rather than being buried inside them. */
  surface: 1.16,
  power: 6.5,
};

export const SUNS = {
  // Pre-normalised so the shader never has to.
  a: { dir: [0.7657, 0.1214, 0.6316], colour: [1.42, 1.38, 1.24], intensity: 3.0 },
  b: { dir: [-0.5623, 0.3578, -0.7473], colour: [0.8, 0.34, 1.3], intensity: 1.5 },
  /**
   * A third, dimmer light from below — a cool fill that separates the silhouette
   * from the background and puts a rim under the rings.
   *
   * Deliberately given NO shadow ray on the body: a soft shadow is 16 march steps,
   * and at this intensity, from this angle, into a medium that is mostly
   * transmission anyway, nobody could tell. That saved cost is why a third light is
   * affordable at all in a pass that is already 90% of the frame.
   *
   * Also not a flare anchor: `frame.sun` carries two screen positions and adding a
   * third would grow the uniform layout for a light whose disc is dim enough that
   * the streak would barely register. It still gets a disc in the sky.
   */
  c: { dir: [-0.1826, -0.8039, 0.5661], colour: [0.42, 0.68, 0.95], intensity: 1.1 },
};

export const CAMERA = {
  /**
   * How far IN FRONT of the subject the lens focuses, in world units.
   *
   * There is no depth of field any more - see the note below - and this is not a leftover. The
   * composite's spherochromatism needs a focal plane to know which side of focus a highlight is on:
   * the colour rim flips magenta-in-front to green-behind, and that flip is the whole effect. With
   * no focus distance the term degenerates to a constant green rim everywhere, which is precisely
   * the lateral-chromatic-aberration look its own comment calls a defect.
   *
   * Removing the thin lens took this with it, and nothing noticed for several commits because the
   * failure is a plausible-looking image rather than an error.
   */
  focusPull: 1.5,

  distance: 9.4,
  /**
   * Field of view across the screen DIAGONAL, in degrees.
   *
   * Defined on the diagonal rather than on an axis so the subject is framed the
   * same at any aspect ratio: wide, tall and square windows all divide up one
   * fixed diagonal, and a 21:9 window matches its own 9:21 transpose exactly.
   * The camera turns this into a focal length once (see `Camera.focal`).
   */
  diagonalFov: 100,
  /** Subject sits off dead-centre, as a fraction of the HALF DIAGONAL, so the
   *  composition holds its proportions when the window changes shape. */
  frameOffset: [0.034, -0.02],
  /** Thin-lens radius in world units; 0 = pinhole. This is the amplitude of the
   *  per-frame ray-origin offset, so it is also the amplitude of any residual
   *  parallax the accumulation fails to average — i.e. of apparent camera shake.
   *  Kept modest on purpose; see LENS_DISK in renderer.js for the other half of
   *  that fix, which is the sample ORDER rather than its size. */
  /*
   * NO DEPTH OF FIELD, deliberately, and this is where it would go back.
   *
   * `aperture` and `focusPull` used to live here, driving a thin lens sampled ONE point per frame in
   * ray generation and amortised over the temporal history. Two structural faults, not tuning:
   *
   *   IT COULD NOT CONVERGE. The resolve's variance clip rebuilds its box from each frame's
   *   neighbourhood and pulls the history back toward the newest sample, so a per-frame lens offset
   *   never settles. Measured on a scene stopped dead, share of pixels moving by more than 4 of 255
   *   levels between adjacent frames: 0.457% with the lens against 0.107% for the pixel jitter
   *   alone, and 0.000% - bit-exact - with both off. More history did nothing (weightMax 6 to 200,
   *   blend 0.15 to 0.04), because the clip is the limiter and not the window.
   *
   *   IT MISSED HALF THE SCENE. A lens offset moves a RAY ORIGIN, which only exists in the compute
   *   passes. The rings are rasterised through `jitterClip(frame.viewProj * ...)`, which carries the
   *   pixel jitter and nothing else, so they had no defocus at all - at a wide aperture they were
   *   the only stable thing on screen, which is what exposed the whole problem.
   *
   * THE RIGHT SHAPE, for whoever brings it back: a deterministic post-process GATHER, not stochastic
   * sampling. Circle of confusion from the thin-lens relation, CoC ~ A*f*|z-zf| / (z*(zf-f)), read
   * off the depth already sitting in the accumulation buffer's alpha; a FIXED low-discrepancy disk of
   * taps (a golden-angle spiral, 16-32 of them) scaled by that radius; and a reach test per tap - a
   * neighbour contributes only if its own CoC covers this pixel - so a sharp foreground cannot bleed
   * onto a blurred background. Same input, same output, so it is stable by construction rather than
   * dependent on accumulation, and it runs once over the composited image where the rings are
   * already present. Jimenez's "Next Generation Post Processing" (SIGGRAPH 2014) and Unreal's
   * diaphragm DOF are the references; both separate near and far fields, which is the next
   * refinement rather than the starting point.
   *
   * It needs its own target and pass, because a gather has to read a texture and the composite
   * builds its colour in a local. That is why this is a note and not a patch.
   */
  atmosphereR: 5.5,

  /**
   * Slow dolly in and out, layered over the rotation. Two incommensurate periods
   * so it never settles into a visible loop.
   *
   * Slow on purpose, and there is a hard limit on how fast it may be: radial
   * motion changes EVERY pixel's depth at once, and TAA's depth gate
   * (TEMPORAL.depthGate, 1.2% of distance) starts rejecting history when the
   * per-frame change approaches it — the image would soften exactly while moving.
   * At this amplitude and period the per-frame change is ~16x under the gate.
   */
  zoom: 1.7,
  zoomPeriod: 52.0,

  /** Chase camera, active once the player has touched a flight key. `lag` is the
   *  per-frame easing fraction — low, because a rigid follow on a curved surface
   *  swings the frame on every turn. `lead` looks ahead of the ship. */
  chaseBack: 3.4,
  /** Height above the ship. Larger than it looks like it should be on purpose: the
   *  camera lags along a CURVING path, so it settles lower than the instantaneous
   *  target — measured, 2.9 here lands about 1.6 in practice. */
  chaseUp: 2.9,
  chaseLead: 1.4,
  /** Time constant in SECONDS, not a per-frame fraction — see camera.js. Larger is
   *  lazier; this is the lag that makes a follow camera readable rather than rigid. */
  chaseTau: 0.28,
  /** How fast a shake settles. A time constant in seconds, never a per-frame fraction. */
  shakeDecay: 0.19,
  /** Shake frequency, radians per second. Fast enough to read as an impact rather than a wobble. */
  shakeFreq: 47.0,

  /** Slight roll, so the frame is never quite level — handheld, not bolted down. */
  roll: 0.16,
  rollPeriod: 63.0,
};

/**
 * Runtime A/B toggles. NOT art direction — nothing here should change what the frame looks
 * like, only how it is computed.
 *
 * This block exists because of a repeated, expensive lesson: an optimisation that changes
 * the STRUCTURE of a shader cannot be A/B tested by editing the shader, because that means
 * comparing two page loads minutes apart, and on a contended machine that cannot resolve
 * anything under about 5%. Several "obvious" wins in this project turned out to be inside
 * the noise, and one turned out to be a loss. A toggle costs one uniform compare.
 */
/**
 * A ring's SIZE, which depends only on its index — the half of `ringDefAt` that does not involve
 * time. The other half, the precessing basis, stays in WGSL because it is evaluated per vertex.
 *
 * This exists so the CPU mesh generator and the shader cannot disagree about how big a ring is. The
 * expression is mirrored in `ringDefAt`; the CONSTANTS are shared rather than copied, and
 * `dev/meshgen.mjs` pins the two together.
 */
export function ringDims(i) {
  return {
    radius: RINGS.radius0 + i * RINGS.gap,
    halfW: RINGS.width * (1.0 - RINGS.widthTaper * i),
    halfH: RINGS.height * (1.0 - RINGS.heightTaper * i),
  };
}

/**
 * Per-channel linear gains that adapt a `sceneK` white point to a `targetK` one.
 *
 * Von Kries adaptation in the working RGB primaries: put both temperatures on the Planckian locus,
 * convert to linear sRGB, take the ratio. Normalised on green, so moving the slider changes colour and
 * not exposure - which matters because exposure is a separate control and a temperature slider that
 * quietly dims the image is a slider nobody can reason about.
 *
 * The locus approximation is Kim et al. (1999), the cubic used everywhere for this; it is valid from
 * 1667K to 25000K and the input is clamped to that, since outside it the polynomial diverges rather
 * than degrading.
 */
export function whiteBalanceGains(targetK, sceneK) {
  const xy = (K) => {
    const T = Math.min(25000, Math.max(1667, K));
    const t = 1000 / T;
    const t2 = t * t;
    const t3 = t2 * t;
    const x = T < 4000
      ? -0.2661239 * t3 - 0.2343589 * t2 + 0.8776956 * t + 0.179910
      : -3.0258469 * t3 + 2.1070379 * t2 + 0.2226347 * t + 0.240390;
    const x2 = x * x;
    const x3 = x2 * x;
    const y = T < 2222
      ? -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
      : T < 4000
        ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
        : 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    return [x, y];
  };
  // xy -> linear sRGB at Y = 1.
  const rgb = ([x, y]) => {
    const X = x / y;
    const Z = (1 - x - y) / y;
    return [
      3.2404542 * X - 1.5371385 - 0.4985314 * Z,
      -0.9692660 * X + 1.8760108 + 0.0415560 * Z,
      0.0556434 * X - 0.2040259 + 1.0572252 * Z,
    ];
  };
  const wt = rgb(xy(targetK));
  const ws = rgb(xy(sceneK));
  const g = [ws[0] / wt[0], ws[1] / wt[1], ws[2] / wt[2]];
  // Green is the reference: it carries most of the luminance, so normalising there is what keeps the
  // exposure fixed as the slider moves.
  return [g[0] / g[1], 1.0, g[2] / g[1]];
}

export const PROBE = {
  /**
   * 1 = read the layer rotations from the injected table (lattice.js);
   * 0 = recompute them per call with sin/cos, as the shader used to.
   *
   * Kept switchable rather than deleted because the table is not obviously faster. It
   * replaces four transcendentals and a 3x3 multiply with two dynamic indexes into ~288
   * bytes of constant data — and a dynamic index into a const array can lower to scratch
   * memory, while this hardware's transcendentals are cheap. That is an empirical question.
   */
  latticeTable: 1,
  /** 1 = replace the body's radiance with its field-evaluation COUNT, so the mean can be
   *  read back over the wire. Diagnostic only; makes the frame look wrong on purpose. */
  showFieldEvals: 0,
  /** 1 = replace the scene's colour with a noise-free synthetic zone plate. The residual it
   *  leaves is the resolve's own geometric error, with the scene's sampling noise removed. */
  testPattern: 0,
  /**
   * 1 = hold the pixel jitter at zero.
   *
   * A blunt but decisive probe: with no jitter there is no sub-pixel sample movement, so a
   * frozen scene should be bit-identical frame to frame. Any residual that SURVIVES this is
   * not about sampling; any residual that vanishes with it is somewhere in the jitter's
   * handling. Costs no uniform slot — the jitter is computed on the CPU.
   */
  zeroJitter: 0,
  /**
   * 1 = withhold the satellites' motion vector, leaving them on camera reprojection alone.
   *
   * Exists so the fix that gave them one can be A/B'd at a FIXED resolution and on the same frame.
   * The stability metric moves with image sharpness and content, so a before-and-after taken across
   * a window resize or a scene change says nothing - which is a mistake this project has already
   * made once, in the first version of the shake probe.
   */
  noSatMotion: 0,
  /** 1 = band-limit the shading's fine detail by the pixel footprint. See the note in shadeBody. */
  bandLimit: 1,
};

/** Zone-plate chirp rate. Chosen so the corners land near the sampling Nyquist while the
 *  centre — which is what the readback patch measures — stays comfortably band-limited. */
export const TEST_PATTERN_K = 1200.0;

/** Divisor for the eval-count probe. 1/100 keeps a 160-step worst case inside a
 *  comfortable f16 range while the patch mean stays directly readable as a count. */
export const PROBE_EVAL_DIV = 100.0;

export const TEMPORAL = {
  /** Weight of the NEW sample where history is valid. The body is *animated*,
   *  so history goes stale by design; this has to decay fast enough that a
   *  stale value never reads as a trail. */
  blend: 0.15,
  blendBackground: 0.06,
  /** Depth gate BASE, as a fraction of distance. Must stay TIGHTER than the motion
   *  the pulse imparts to the surface (~0.08 world units) or reprojection
   *  happily accepts history from a surface that has since moved. */
  depthGate: 0.012,
  /**
   * Slope-scaled slack on that gate, in PIXELS of tolerated reprojection error.
   *
   * A flat fractional gate is wrong on a steep surface: reprojection lands the history fetch
   * within about a pixel, and on this body — bumpy fBM, grazing angles near the silhouette —
   * one pixel of depth change routinely exceeds 1.2% of distance. So the tolerance carries a
   * term proportional to the local depth gradient, the same trick a rasteriser uses for
   * `depthBiasSlopeScale`: allow what the geometry says a pixel of slide is worth, no more.
   *
   * RETUNED from 2.0 to 1.0, and the history is worth knowing because it corrects an earlier
   * conclusion. This term was introduced when the frozen residual was 4.20%, and widening it
   * measured a 22% win — which was real, but it was earning that win by absorbing a bug
   * rather than by describing geometry. The reprojection was not removing the jitter, so the
   * history being compared sat up to a pixel away and the depth mismatch scaled with exactly
   * this gradient. With that fixed (see the note in taa.wgsl) 2.0 is simply too permissive:
   * it accepts stale history and shows up as lag.
   *
   * Measured after the fix, sweeping this alone:
   *
   *     slack   stability   lag        sharpness
   *     2.0     1.718%      4.77x      0.3442
   *     1.0     1.724%      3.63x      0.3454     <- here
   *     0.0     2.791%      3.02x      0.3661
   *
   * 1.0 gives up nothing on stability against 2.0 and cuts lag by 24%. Going to zero does
   * cut lag further, but costs 62% more residual — so the term is still doing real work, just
   * half as much as it appeared to when it was covering for something else.
   */
  depthGradSlack: 1.0,
  /** Ceiling on that slack, as a fraction of distance. At a silhouette a one-pixel
   *  sliver has a near-infinite gradient, and without a cap the gate would open
   *  completely at precisely the edges where a wrong history fetch is most visible.
   *  0.06 sits just inside the saturation point measured above. */
  depthGradMax: 0.06,
  /**
   * History reconstruction filter: 0 bilinear, 1 Catmull-Rom (9 bilinear taps).
   *
   * A uniform rather than a compile-time switch so the two can be compared in one
   * session — the previous generation of TAA knobs were injected WGSL consts, baked at
   * pipeline creation, and every console A/B against them silently measured the same
   * shader twice.
   */
  historyFilter: 1,
  /**
   * Width of the temporal-upsampling reconstruction kernel: the k in exp(-k*d^2), with d
   * measured in OUTPUT pixels.
   *
   * This is the sharpness/stability dial for TAAU and there is no safe default to derive.
   * At 2x the input samples sit 2 output pixels apart, so a large k weights only the sample
   * nearest each output pixel — sharp, but few pixels get a usable sample per frame, so
   * convergence slows and noise rises. A small k spreads every sample over its neighbours —
   * stable, but that is a blur, which defeats the point. 1.0 puts a sample 1 output pixel
   * away at weight 0.37 and one at 1.41 (the diagonal) at 0.14.
   */
  taauSigma: 0.65,
  /**
   * Cap on accumulated weight, which is what the upsampling path uses in place of `blend`.
   *
   * The accumulator forms a true weighted average: colour = (C_prev*W + sum w_i c_i) / (W +
   * sum w_i). Left uncapped that never forgets, so an animated surface could never update;
   * capping W bounds how much past any pixel may hold and therefore sets the effective blend
   * to `sum w_i / (W_max + sum w_i)`. Which is automatically confidence-adaptive for free — a
   * pixel that got a sample right on its centre this frame moves further than one that did
   * not, with no separate term.
   *
   * 6 is chosen to MATCH the 1:1 path's exponential behaviour rather than to be a new look:
   * blend 0.15 is a retention of 0.85, i.e. an equivalent weight of 0.85/0.15 = 5.7.
   */
  /** The render scale these caps were tuned at. The weight is a count of accumulated SAMPLES, and
   *  a lower scale delivers fewer per output pixel per frame — so under dynamic resolution the cap
   *  has to follow the density, or a pixel that banked confidence at high resolution keeps leaning
   *  on stale history once the scale drops, which is ghosting. Applied as (scale/ref)^2 in the
   *  uniform writer. */
  weightRefScale: 0.5,
  weightMax: 6.0,
  /** The same cap for background pixels, matched to `blendBackground` 0.06 => 1/0.06 - 1. */
  weightMaxBg: 16.0,
  /** Variance-clipping width, in standard deviations of the 3x3 neighbourhood.
   *  ~1.25 is the usual compromise: tight enough that stale history cannot
   *  trail, wide enough that honest sample noise is still free to average. */
  clipGamma: 1.25,
  /**
   * Clip width for the UPSAMPLING path. Now the SAME as the 1:1 path, and the history of this
   * value is worth keeping because it corrects a wrong conclusion.
   *
   * It was briefly 4.0. The upsampling residual on the noise-free signal sat at 9.01% and
   * widening this alone brought it to 1.16%, so the clip looked like the problem. It was not:
   * the BOX was. Built from a 2x2 reconstruction quad it spanned about two output pixels and
   * shifted membership with the jitter, so it described a far narrower range than the pixel
   * could legitimately take — and widening gamma was compensating for a bad estimator rather
   * than fixing it, exactly the mistake the depth-gate slack made earlier.
   *
   * Rebuilding the box from a 3x3 of INPUT taps (what FSR2 does, for this reason) settled it:
   *
   *     box    gamma 1.25   gamma 2.5   gamma 4.0
   *     2x2    9.010%       -           1.157%
   *     3x3    0.710%       0.670%      -
   *
   * At 1.25 with a 3x3 box the residual is 0.71%, against 1.86% for the 1:1 path — better by
   * 2.6x with the clip at FULL strength, no loosened guard against stale history. The two
   * gammas are kept as separate knobs because the two paths have genuinely different
   * neighbourhoods, but they should not need different values, and if they ever do again that
   * is a sign the box is wrong rather than the width.
   */
  clipGammaUpsample: 1.25,
  /** Minimum clip-box half-width, as a fraction of local brightness. Without a floor
   *  the box collapses in smooth regions and the history is pinned to the current
   *  frame — accumulation stops and no blend value can restore it. */
  clipFloor: 0.14,
  /** Absolute firefly ceiling, in linear HDR. A guard against a true freak spike
   *  poisoning the history, not a denoiser — a neighbourhood-relative version was tried
   *  and measured 46% worse in the bright band (see taa.wgsl). Rarely fires by design:
   *  the observed peak is ~4. */
  fireflyMax: 12.0,
  dragBlend: 0.5,               // dragging moves the camera far per frame
};

export const GLOW = {
  threshold: 1.1,
  strength: 0.17,
  /** Per-level weights, coarse..fine, applied during the upsample chain.
   *  Weighting the coarse levels up is what gives the wide, soft halo. */
  levelWeight: 1.0,
  // Doubled when the pyramid moved from render- to display-resolution sizing: the offsets are in
  // SOURCE texels, so a level-0 that is twice as fine spans half as much of the screen. Same look,
  // and now it no longer changes when `renderScale` does.
  radius: 2.0,                  // texel-space filter radius multiplier
  /** Border fade width, in uv, applied at the prefilter. Wide enough that the
   *  coarse pyramid levels have a dark border to replicate instead of a bright
   *  one; narrow enough not to visibly clip glow near the frame edge. */
  edge: 0.025,
};

export const FLARE = {
  strength: 0.45,
  /** Round ghosts are the spherical-element artefact. A real anamorphic rig
   *  shows few of them and they stay dim — the streak is the signature. */
  ghosts: 3,
  ghostSpacing: 0.42,
  ghostChroma: 0.009,
  ghostGain: 0.2,
  haloWidth: 0.42,
  halo: 0.2,
  /** The anamorphic streak. A cylindrical front element smears a highlight along
   *  ONE axis, so what sells it is the extreme aspect: very wide, very thin.
   *  `tight` is the vertical concentration — the bigger it is, the more this
   *  reads as a lens flare rather than as a glow. */
  streak: 1.35,
  streakFall: 3.0,
  streakTight: 3000.0,
  /** Anamorphic streaks are famously blue-biased, from the coatings. */
  streakColour: [0.42, 0.66, 1.0],
  /** Border fade for the mirrored ghost/halo taps, in uv. */
  edge: 0.05,
  /** Border fade for the WHOLE flare layer, in uv. Wider, because the analytic
   *  streak carries real brightness a long way and must be gone before it can
   *  reach the frame edge and read as a bar. */
  vignette: 0.16,
};

/**
 * Small satellites: a central cube for the bus plus two flat cubes for the solar
 * array, on a boom. Sizes are world units, and the body has radius ~1, so these
 * are deliberately tiny — they read as hardware only if they stay small.
 */
/**
 * The model viewer: a turntable for inspecting the generated meshes on their own.
 *
 * FAR FROM THE ORIGIN, and that is the one number here with a reason behind it rather than a taste. The
 * hull's material adds a core-light term that falls off as 1/|p|^2 from the world origin, where the
 * planet is; a model sitting at the origin would be lit by a planet that the viewer deliberately does
 * not draw, and a vertex landing exactly on it would normalise a zero vector. Two hundred units out, the
 * term is ~2.5e-5 and the studio is lit only by the three suns, which is what you want when judging a
 * shape.
 */
export const MODELVIEW = {
  /** Where the model sits. See above — this is a correctness constraint, not a preference. */
  origin: [0, 0, 200],
  /** Turntable rate, radians per second. Slow enough to read the silhouette at every angle. */
  spinRate: 0.55,
  /** Camera stand-off as a multiple of the model's bounding radius, so both models frame the same. */
  distance: 3.4,
  /** Elevation above the model's equator, radians. ~35 degrees is the "quarter top-down" view. */
  elevation: 0.62,
};

export const SATELLITES = {
  count: 5,
  /**
   * Mean orbital radius, varied x0.88..1.18 per satellite. Upper bound is the
   * camera's closest approach (CAMERA.distance - CAMERA.zoom = 3.85): beyond that
   * a satellite sweeps past the lens.
   */
  orbit: 5.7,
  rate: 0.085,          // radians per second, scaled per satellite
  bus: 0.055,           // half-extent of the central cube
  /**
   * Gap from the bus FACE to the inner edge of a panel — not the panel's distance from the centre.
   *
   * The geometry read it as an absolute offset and started the panels at 0.03, which is inside a bus
   * whose half-extent is 0.055 and well inside the boom stubs at 0.078: the arrays grew out of the
   * middle of the body instead of standing off it. Widened as well as corrected, so there is visible
   * daylight between the stub and the array rather than the two just failing to overlap.
   */
  boom: 0.06,
  panelLen: 0.20,       // half-length along the boom, over BOTH panels of a wing
  panelWide: 0.085,     // half-width — wide enough to read as a wing, not a rod
  panelThick: 0.005,    // half-thickness — thin enough to flash edge-on
  /** Cell silicon under cover glass: dark, and blue is nearly all of what little
   *  it reflects diffusely. The sheen carries the rest of the colour. */
  panelColour: [0.035, 0.085, 0.24],
  panelSheen: [0.07, 0.13, 0.30],
  /** Metallised film. Aluminised on some faces, gold-tinted kapton on others —
   *  both are conductors, so these are F0 values, not albedos. */
  filmAluminium: [0.87, 0.89, 0.91],
  filmGold: [0.92, 0.75, 0.44],
  cellScale: 46.0,      // solar-cell grid frequency, in panel space
  /** Scaled to the box they sit on: the bus is ~0.055 across, so these need to be
   *  high or a face gets two cells and the detail never reads. */
  greebleScale: 70.0,
  wrinkleScale: 180.0,
  wrinkle: 0.45,        // crinkle strength, as a tangent-space normal offset
  ambient: 0.05,
};

export const EMBERS = {
  count: 1400,
  rate: 0.42,                   // outward cycles per heartbeat
  spawnR: 3.0,                  // launch from just outside the body
  travel: 1.8,                  // reach to the outer ring
  size: 0.02,                  // world-space billboard radius
  colour: [1.0, 0.44, 0.15],
  intensity: 4.6,
  spriteSize: 64,
  turb: 0.2,                    // curl noise turbulence strength (0 = off)
};

/**
 * The film grade. Every value here is a UNIFORM, read fresh each frame — see
 * `FrameUniforms.write` — so all of it can be moved live from the debug panel.
 *
 * `gain` and `exposure` used to be one number, which was a trap: the same value was written to
 * the uniform as a pre-grade linear gain AND injected as the const inside the tone curve, so
 * every "exposure" change was applied twice and a slider bound to it was violently
 * over-sensitive. They are separate because they do different things — gain is how much light
 * reaches the film, exposure is where the curve's shoulder sits relative to it — and their
 * defaults reproduce the old look exactly (2.0 * 0.42 = 0.84).
 */
/**
 * The atmosphere, as a real scattering medium.
 *
 * This replaced `limbGlow`, which was one smoothstep on the ray's perpendicular distance from the
 * body — a halo painted around the silhouette, added in the composite because a screen-space halo
 * has no world position to reproject. Everything here is a physical parameter of the integral that
 * stood in for; see shaders/volumetric.wgsl for the method and for why it is not a froxel grid.
 */
export const VOLUME = {
  /**
   * Samples along the ray, inside the shell.
   *
   * The step count now carries the quality on its own, because the sample offset is frame-STATIC.
   * It used to advance every frame and lean on the accumulation to converge it, which was wrong:
   * feeding per-frame noise to a variance-clipped accumulator widens its clip box and admits stale
   * history across the whole image, and measured, that tripled the marched image's high-frequency
   * wobble. See volStepOffset. So these steps buy smoothness directly rather than borrowing it.
   */
  steps: 20,
  /** Samples for the EXTINCTION-only integral the additive layer uses. Fewer, because there are no
   *  shadows in it to resolve and it runs per vertex — see `volTransmittance`. */
  trSteps: 6,
  /** Extinction per unit of density-length. With `albedo` below this also sets how much the
   *  atmosphere hides what is behind it, so it is the knob that reads as "thickness". */
  // Eased back from 0.55 when the world doubled. This is a coefficient PER UNIT LENGTH and the
  // shell it integrates through is now twice as deep, so the same number is twice the atmosphere.
  // Measured: it had pushed the whole frame blue-dominant — mean RGB 0.177/0.079/0.191, blue past
  // red — where the grade is built around a warm balance. Reducing the coefficient rather than the
  // albedo keeps the physics of the scattering intact and only asks for less of it.
  sigma: 0.38,
  /**
   * Scattering albedo — what fraction of extinguished light is scattered rather than absorbed, per
   * channel. Blue-weighted because short wavelengths scatter more, which is the whole reason a sky
   * is blue; near 1 because a gas absorbs very little.
   */
  // Less STEEPLY blue-weighted than the first pass. At 0.42/0.62/1.0 the scattering was blue
  // enough to tilt the whole frame off the warm balance the rest of the grade is built around, and
  // a thin aerosol's real ratio is gentler than the pure-Rayleigh lambda^-4 that inspired it.
  albedo: [0.55, 0.70, 1.0],
  /** How fast density falls with altitude above the body, per world unit. The inverse of a scale
   *  height: larger is a thinner, more sharply bounded shell. */
  falloff: 1.3,
  /**
   * Forward-scattering anisotropy for Cornette-Shanks, 0..1. Aerosol scatters strongly forward,
   * which is what makes an atmosphere flare when you look toward a sun and stay dim looking away.
   * Past ~0.85 the lobe gets narrow enough to alias against the step count.
   */
  g: 0.68,
  /** A constant floor standing in for multiple scattering, which single-scattering by definition
   *  misses. Without it a shadowed pocket goes pure black, and real shadowed air does not — it is
   *  lit by the rest of the sky. Cheap, and it is the difference between "shadow" and "hole". */
  ambient: [0.020, 0.026, 0.042],
  /** Softness of the body's terminator, as a fraction of its mean radius. The occluder is a lumpy
   *  distance field approximated by a sphere, so this is both the penumbra and the honest width of
   *  the approximation. */
  bodySoft: 0.10,
  /** Radius the body's shadow is cast from, as a multiple of R. R is the field's MEAN radius and
   *  its lobes reach about half again as far, so a terminator at R leaks light through them. This
   *  sits between the two — the best a single sphere can do against a silhouette that is not one. */
  bodyR: 1.25,
  /** How much of a sun a ring blocks. Below 1 because a machined hoop seen edge-on is thin and the
   *  shell is deep — a fully opaque bar would read as a painted stripe. */
  ringOpacity: 0.85,
  /** Ring penumbra: a base width in world units, plus how much it widens per unit of distance from
   *  the occluder. The spread is what a real penumbra does, and it is also what keeps a sweeping
   *  analytic edge from crawling through a 12-step march. */
  ringSoft: 0.035,
  ringSpread: 0.10,
};

/**
 * Graded for EASTMAN COLOR NEGATIVE 50T 5251 - the 1962 stock, EI 50, tungsten-balanced at 3200K,
 * superseded by 5254 in 1968. Every value below is reasoned from a published characteristic of that
 * emulsion rather than picked by eye, so each one can be argued with:
 *
 *   white 13.4       A negative has a long shoulder and enormous highlight latitude. Raising the
 *                    white point stretches the shoulder so the suns and the rail gun roll off
 *                    instead of clipping - the single most recognisable negative behaviour.
 *   contrast 1.01    Camera negative gamma is around 0.55-0.65; the print stock supplies the rest
 *                    of the system gamma. A 1960s stock is flatter still than a modern one, so the
 *                    S-curve here backs off to nearly straight.
 *   pivot 0.38       Lower pivot with the flatter curve, which keeps the midtones where the old
 *                    long straight-line portion put them rather than lifting the whole frame.
 *   blackLift 0.021  The toe is long and soft. Film has no true black: the base plus fog sets a
 *                    floor, and this is that floor.
 *   black triplet    Warm and very slightly green, because the 1962 dye set has a weak cyan layer
 *                    and high unwanted absorptions. That is what makes period footage read amber in
 *                    the shadows rather than neutral.
 *   saturation 0.94  Same cause, other end: dye crosstalk means the primaries are not clean, so
 *                    the palette is muted next to a modern stock. Pastel, not punchy.
 *   halation 1.42    Pronounced, and red-weighted. Anti-halation backing on this generation was
 *                    far less effective than today's, so bright sources bleed a red-orange corona.
 *   grain 0.026      EI 50 is a slow, fine-grain stock - but a 1962 EI 50 is roughly a modern
 *                    EI 200 in grain terms, so this comes down only slightly rather than to zero.
 *
 * WHAT THIS IS NOT. A real match needs the densitometry, not a grade: per-channel D-logE curves for
 * the three layers, a 3x3 crosstalk matrix for the unwanted absorptions, and the print stock's own
 * curve on top. That is a shader change and it is the honest way to do this; these numbers are the
 * closest the existing analytic grade can get without one. The tungsten balance in particular is
 * NOT applied here - 3200K stock under a daylight-ish sun wants an 85B correction, and faking the
 * cast with the black point would be a different look, not this one.
 */
export const FILM = {
  /** Pre-grade linear gain, applied to the scene before the curve sees it. */
  gain: 0.84,
  /** Input scale into the tone curve. Together with `white` this places the shoulder. */
  exposure: 2.0,
  white: 13.4,
  halation: 1.42,
  grain: 0.026,
  /**
   * How much of the physical vignette to apply, 0 = none, 1 = the full cosine-fourth law.
   *
   * An AMOUNT now, not a coefficient: the shape is fixed by physics (see composite.wgsl) and
   * this only says how far toward it to go. At this field of view the law puts the corners at
   * 17% of centre, which is severe but real — a 100-degree diagonal lens genuinely vignettes
   * that hard, which is why wide lenses ship with correction profiles. Backed off slightly
   * because it is applied in linear light now, where the tone curve's toe lifts the corners
   * back up, so a high amount lands near where the old post-gamma term did.
   */
  vignette: 0.85,
  saturation: 0.94,
  /**
   * How much the black floor is LIFTED toward `black`, 0..1. The dominant control over how
   * dark the sky reads, and by a long way: the background is near zero almost everywhere, so
   * it lands essentially on this floor, and neither exposure nor contrast moves it much.
   * Print film never reaches zero, so some lift is the honest look — but 0.035 was enough to
   * put a visible grey wash behind the stars.
   */
  blackLift: 0.021,
  /** The colour that floor is lifted toward. A look choice, not a level — stays a const. */
  black: [0.026, 0.024, 0.030],
  /**
   * WHITE BALANCE, in kelvin: the temperature the film treats as neutral.
   *
   * Reads like a camera's white-balance setting, and behaves like one - raise it and the image warms,
   * lower it and the image cools - because that is exactly what it is. The scene is rendered
   * D65-referred, so the gains adapt D65 into this white point; at 6500 they are (1,1,1) and nothing
   * happens.
   *
   * DERIVED, not dialled. `whiteBalanceGains` puts the temperature on the Planckian locus (the Kim
   * cubic, valid 1667-25000K), converts both white points to linear sRGB and takes the per-channel
   * ratio, normalised on green so exposure does not move with the slider. That is von Kries
   * adaptation, and doing it properly is what makes the slider behave predictably across its whole
   * range instead of only near where it was tuned.
   *
   * WHY 4750 AND NOT 3200. Eastman 50T is balanced for 3200K, and 3200 here is physically honest: it
   * gives gains of (0.55, 1.00, 2.80) and reads far colder than this project has ever looked. The
   * hand-picked triple this replaces was (0.87, 1.00, 1.42) - a much milder adaptation - and the
   * temperature that reproduces its blue-to-red ratio is 4740. So the default preserves the look that
   * was there, and the physically-correct 50T value is one drag away if you want it.
   *
   * A green-magenta tint axis is the usual companion to this in a grading tool, and is deliberately
   * absent: there is nothing in this scene that needs it, and a knob with no subject is a knob that
   * gets set wrong.
   */
  /**
   * Display transform: 0 Hable, 1 AgX.
   *
   * AgX by default. Hable maps each channel independently, so a bright saturated colour clips one
   * channel first and shifts hue on the way to white - and this scene is mostly bright saturated
   * colour. AgX mixes the channels before the curve and un-mixes the CURVED result, which desaturates
   * toward white instead. Hable is kept because being able to flip between them is how you tell a
   * difference from a preference. See the note in composite.wgsl.
   */
  toneMap: 1,
  /**
   * AgX's own exposure placement, in stops, relative to the Hable path.
   *
   * AgX puts 18% grey higher than `hableCurve(x)/hableCurve(white)` followed by a 2.2 gamma does, so
   * the same scene through it comes out 64% brighter in mean luma - measured, 43.3 to 71.1. That is a
   * placement difference, not a look choice, and leaving it in would mean the exposure slider means
   * something different depending on which transform is selected.
   *
   * -1.45 stops brings the two to the same mean luma, solved from a sweep (46.9 at -1.25, 38.9 at
   * -1.75, target 43.3). So flipping the dropdown changes the CHARACTER of the highlights and nothing
   * else, which is the only way to compare them honestly.
   */
  agxEv: -1.45,
  temperature: 4750,
  /** What the scene's own illuminant is taken to be. D65 because the sky and suns are authored that
   *  way; it exists so the adaptation has a stated reference rather than an implied one. */
  sceneTemperature: 6500,
  /**
   * Contrast about a mid pivot, applied AFTER the display transfer.
   *
   * Display-referred on purpose. Contrast in linear light is just exposure and white point
   * moving together: it crushes the shadows to nothing and barely touches the highlights.
   * Pivoting after the transfer opens the gap between the sky and everything in front of it
   * while leaving the midtones where they were, which is what "more contrast" actually means.
   */
  contrast: 1.01,
  pivot: 0.38,
};

/**
 * The subset of the above that the shaders need, as WGSL constants.
 * Anything a shader reads *per frame* belongs in the uniform buffer instead —
 * this is only for values that are fixed for the lifetime of a pipeline.
 */
export function wgslDefines() {
  const rot = layerRotations();
  const matArray = (ms) =>
    `array<mat3x3f, ${LAYER_SLOTS}>(\n  ${ms.map((m) => wgslMat3(m, f)).join(',\n  ')}\n)`;
  return {
    // The spherical-Fibonacci layer rotations, as a table. Eight fixed matrices that the
    // GPU was recomputing on the order of 360 times per pixel — see lattice.js.
    LAYER_ROT: matArray(rot.fwd),
    LAYER_IROT: matArray(rot.inv),
    // The same schedule the table was built from, for the shader's computed fallback.
    PROBE_EVAL_SCALE: 1 / PROBE_EVAL_DIV,
    TEST_PATTERN_K: TEST_PATTERN_K,
    LAYER_YAW_STEP: SCHEDULE.yawStep,
    LAYER_YAW_BIAS: SCHEDULE.yawBias,
    LAYER_PITCH_STEP: SCHEDULE.pitchStep,
    LAYER_PITCH_BIAS: SCHEDULE.pitchBias,
    // body
    R: BODY.R,
    OCTAVES: int(BODY.octaves),
    N0: BODY.n0,
    NGROW: BODY.nGrow,
    RHO0: BODY.rho0,
    RHOFALL: BODY.rhoFall,
    JITTER: BODY.jitter,
    DETAIL: BODY.detail,
    // March. Only the loop bound and the hit threshold are consts; the step SCALES are
    // uniforms so they can be swept at runtime (see the note in common.wgsl).
    MARCH_STEPS: int(MARCH.steps),
    MARCH_HIT_EPS: MARCH.hitEps,
    HOLE_OCT: int(BODY.holeOctaves),
    SFN_HOLE: int(BODY.holeCandidates),
    HOLEDEN: BODY.holeDen,
    HOLESC: BODY.holeScale,
    HOLEDGROW: BODY.holeDenGrow,
    HOLESFALL: BODY.holeScaleFall,
    HOLEOFF: BODY.holeOffset,
    // medium
    GUM_DEPTH: MEDIUM.depth,
    GUM_DENS: MEDIUM.density,
    GUM_G: MEDIUM.g,
    GUM_IOR: MEDIUM.ior,
    DISPERSE: MEDIUM.disperse,
    REFR_GAIN: MEDIUM.refrGain,
    // core
    CORE_R: CORE.radius,
    CORE_ATT: CORE.attenuation,
    CORE_COL: `vec3f(${CORE.colour.map(f).join(', ')}) * ${f(CORE.intensity)}`,
    // pulse
    PULSE_R: PULSE.radius,
    PULSE_OFF: PULSE.offset,
    PULSE_LAG: PULSE.lag,
    // detonations
    BLAST_COUNT: int(BLASTS.count),
    BLAST_PERIOD: BLASTS.period,
    BLAST_CHANCE: BLASTS.chance,
    BLAST_LIFE: BLASTS.life,
    BLAST_R: BLASTS.radius,
    BLAST_SURF: BLASTS.surface,
    BLAST_POWER: BLASTS.power,
    // suns
    SUN1_DIR: `vec3f(${SUNS.a.dir.map(f).join(', ')})`,
    SUN1_COL: `vec3f(${SUNS.a.colour.map(f).join(', ')}) * ${f(SUNS.a.intensity)}`,
    SUN2_DIR: `vec3f(${SUNS.b.dir.map(f).join(', ')})`,
    SUN2_COL: `vec3f(${SUNS.b.colour.map(f).join(', ')}) * ${f(SUNS.b.intensity)}`,
    SUN3_DIR: `vec3f(${SUNS.c.dir.map(f).join(', ')})`,
    SUN3_COL: `vec3f(${SUNS.c.colour.map(f).join(', ')}) * ${f(SUNS.c.intensity)}`,
    // camera / framing. The projection itself is NOT here: it is a matrix in the
    // frame uniforms, so ray generation, reprojection and the ember billboards
    // cannot end up disagreeing about it.
    ATMO_R: CAMERA.atmosphereR,
    // volumetrics
    VOL_STEPS: int(VOLUME.steps),
    VOL_TR_STEPS: int(VOLUME.trSteps),
    VOL_ALBEDO: `vec3f(${VOLUME.albedo.map(f).join(', ')})`,
    VOL_FALLOFF: VOLUME.falloff,
    VOL_AMBIENT: `vec3f(${VOLUME.ambient.map(f).join(', ')})`,
    VOL_BODY_SOFT: VOLUME.bodySoft,
    VOL_BODY_R: VOLUME.bodyR,
    VOL_RING_SOFT: VOLUME.ringSoft,
    VOL_RING_SPREAD: VOLUME.ringSpread,
    // temporal — blend, gate and clip live in the frame UNIFORM now rather than here,
    // so they can be A/B tested without a recompile. Only the background weight is
    // still compile-time.
    ACCUM_BLEND_BG: TEMPORAL.blendBackground,
    DRAG_BLEND: TEMPORAL.dragBlend,
    // glow / flare
    BLOOM_EDGE: GLOW.edge,
    // The optical axis in shared screen space. The projection is sheared by `frameOffset`, so
    // the camera's forward direction lands HERE and not at the origin — which is what the lens
    // flare has to mirror its ghosts through. Verified against the actual projection: with
    // frameOffset [0.034, -0.02] the forward axis projects to uv (0.4805, 0.4796).
    LENS_AXIS: `vec2f(${(-CAMERA.frameOffset[0]).toFixed(6)}, ${(-CAMERA.frameOffset[1]).toFixed(6)})`,
    FLARE_EDGE: FLARE.edge,
    FLARE_VIGNETTE: FLARE.vignette,
    FLARE_GHOSTS: int(FLARE.ghosts),
    FLARE_SPACING: FLARE.ghostSpacing,
    FLARE_CHROMA: FLARE.ghostChroma,
    FLARE_GHOST_GAIN: FLARE.ghostGain,
    FLARE_HALO_W: FLARE.haloWidth,
    FLARE_HALO: FLARE.halo,
    FLARE_STREAK: FLARE.streak,
    FLARE_STREAK_FALL: FLARE.streakFall,
    FLARE_STREAK_TIGHT: FLARE.streakTight,
    FLARE_STREAK_COL: `vec3f(${FLARE.streakColour.map(f).join(', ')})`,
    // satellites
    SAT_COUNT: int(SATELLITES.count),
    SAT_ORB: SATELLITES.orbit,
    SAT_RATE: SATELLITES.rate,
    SAT_BUS: SATELLITES.bus,
    SAT_BOOM: SATELLITES.boom,
    SAT_PANEL_LEN: SATELLITES.panelLen,
    SAT_PANEL_WIDE: SATELLITES.panelWide,
    SAT_PANEL_THICK: SATELLITES.panelThick,
    SAT_PANEL_COL: `vec3f(${SATELLITES.panelColour.map(f).join(', ')})`,
    SAT_PANEL_SHEEN: `vec3f(${SATELLITES.panelSheen.map(f).join(', ')})`,
    SAT_FILM_AL: `vec3f(${SATELLITES.filmAluminium.map(f).join(', ')})`,
    SAT_FILM_AU: `vec3f(${SATELLITES.filmGold.map(f).join(', ')})`,
    SAT_CELL_SCALE: SATELLITES.cellScale,
    SAT_GREEBLE_SCALE: SATELLITES.greebleScale,
    SAT_WRINKLE_SCALE: SATELLITES.wrinkleScale,
    SAT_WRINKLE: SATELLITES.wrinkle,
    SAT_AMBIENT: SATELLITES.ambient,
    // embers
    EMBER_COUNT: int(EMBERS.count),
    EMBER_RATE: EMBERS.rate,
    EMBER_R0: EMBERS.spawnR,
    EMBER_TRAVEL: EMBERS.travel,
    EMBER_SIZE: EMBERS.size,
    EMBER_COL: `vec3f(${EMBERS.colour.map(f).join(', ')}) * ${f(EMBERS.intensity)}`,
    EMBER_TURB: f(EMBERS.turb),
    // raymarched reflections
    REFL_STEPS: int(REFLECT.steps),
    REFL_OCT: int(REFLECT.octaves),
    // railgun
    RAIL_SEGMENTS: int(RAIL.segments),
    RAIL_LIFE: RAIL.life,
    RAIL_LENGTH: RAIL.length,
    RAIL_TURNS: RAIL.turns,
    RAIL_RADIUS: RAIL.radius,
    RAIL_SPIN: RAIL.spin,
    RAIL_WIDTH: RAIL.width,
    RAIL_FALLOFF: RAIL.falloff,
    RAIL_GAIN: RAIL.gain,
    RAIL_HOT: `vec3f(${RAIL.hot.map(f).join(', ')})`,
    RAIL_COLD: `vec3f(${RAIL.cold.map(f).join(', ')})`,
    // The muzzle, needed by the charge swarm — it gathers at the wing tips the shots leave from, and
    // reading them from the same constants is what keeps the two in the same place.
    RAIL_SPREAD: RAIL.spread,
    RAIL_UP: RAIL.up,
    RAIL_FORWARD: RAIL.forward,
    RAIL_POOL: int(RAIL.pool),
    // the power shot
    RAIL_STRANDS: f(RAIL.strands),
    RAIL_POWER_LENGTH: RAIL.powerLength,
    RAIL_POWER_RADIUS: RAIL.powerRadius,
    RAIL_POWER_WIDTH: RAIL.powerWidth,
    RAIL_POWER_GAIN: RAIL.powerGain,
    RAIL_POWER_LIFE: RAIL.powerLife,
    RAIL_POWER_TURNS: RAIL.powerTurns,
    // sparks
    RAIL_SPARKS: f(RAIL.sparks),
    RAIL_SPARK_LIFE: RAIL.sparkLife,
    RAIL_SPARK_SPEED: RAIL.sparkSpeed,
    RAIL_SPARK_SIZE: RAIL.sparkSize,
    RAIL_SPARK_SPREAD: RAIL.sparkSpread,
    RAIL_SPARK_GAIN: RAIL.sparkGain,
    RAIL_CHARGE_SPARKS: f(RAIL.chargeSparks),
    RAIL_CHARGE_RADIUS: RAIL.chargeRadius,
    // contrail
    TRAIL_COUNT: int(CONTRAIL.samples),
    TRAIL_W0: CONTRAIL.width,
    TRAIL_WGROW: CONTRAIL.widthGrow,
    TRAIL_FALLOFF: CONTRAIL.falloff,
    TRAIL_GAIN: CONTRAIL.gain,
    TRAIL_HOT: `vec3f(${CONTRAIL.hot.map(f).join(', ')})`,
    TRAIL_COLD: `vec3f(${CONTRAIL.cold.map(f).join(', ')})`,
    // aurora
    AURORA_COUNT: int(AURORA.samples),
    AURORA_W0: AURORA.width,
    AURORA_WGROW: AURORA.widthGrow,
    AURORA_FALLOFF: AURORA.falloff,
    AURORA_RAY_FREQ: AURORA.rayFreq,
    AURORA_RAY_DRIFT: AURORA.rayDrift,
    AURORA_MITER_MIN: AURORA.miterMin,
    AURORA_NEAR0: AURORA.nearFade[0],
    AURORA_NEAR1: AURORA.nearFade[1],
    AURORA_FRINGE: AURORA.fringe,
    AURORA_PALETTE_N: int(AURORA.palette.length),
    AURORA_PALETTE: `array<vec3f, ${AURORA.palette.length}>(`
      + AURORA.palette.map((c) => `vec3f(${c.map(f).join(', ')})`).join(', ') + ')',
    // ship
    SHIP_SCALE: SHIP.scale,
    SHIP_PANEL: SHIP.panel,
    SHIP_HULL: `vec3f(${SHIP.hull.map(f).join(', ')})`,
    SHIP_GLASS: `vec3f(${SHIP.glass.map(f).join(', ')})`,
    SHIP_TRIM: `vec3f(${SHIP.trim.map(f).join(', ')})`,
    SHIP_ENV: SHIP.env,
    SHIP_AMBIENT: SHIP.ambient,
    SHIP_CORE_COL: `vec3f(${SHIP.coreColour.map(f).join(', ')})`,
    SHIP_JET_COL: `vec3f(${SHIP.jetColour.map(f).join(', ')})`,
    SHIP_RCS_COL: `vec3f(${SHIP.rcsColour.map(f).join(', ')})`,
    SHIP_RCS_GAIN: SHIP.rcsGain,
    SHIP_JET_GAIN: SHIP.jetGain,
    SHIP_JET_LEN: int(SHIP.jetLength),
    // rings
    RING_SEGMENTS: RINGS.segments,
    // The ring COUNT is a shader constant now, not only a draw-call argument: the volumetric
    // shadow loop iterates the hoops analytically, so it has to know how many there are.
    RING_COUNT: int(RINGS.count),
    RING_R0: RINGS.radius0,
    RING_GAP: RINGS.gap,
    RING_W: RINGS.width,
    RING_WTAPER: RINGS.widthTaper,
    RING_HTAPER: RINGS.heightTaper,
    RING_H: RINGS.height,
    RING_PRECESS: RINGS.precess,
    RING_SPIN: RINGS.spin,
    RING_ROUGH: RINGS.roughness,
    RING_STEEL: `vec3f(${RINGS.steel.map(f).join(', ')})`,
    RING_GRIME: `vec3f(${RINGS.grime.map(f).join(', ')})`,
    RING_SCRATCH: RINGS.scratch,
    RING_ENV: RINGS.env,
    RING_CORE_COL: `vec3f(${RINGS.coreColour.map(f).join(', ')})`,
    RING_CORE_GAIN: RINGS.coreGain,
    RING_AMBIENT: RINGS.ambient,
    MOTION_NONE: SOLID.motionNone,
    // spherochromatism
    SPHERO_STRENGTH: SPHERO.strength,
    SPHERO_TINT: `vec3f(${SPHERO.tint.map(f).join(', ')})`,
    // film
    // Only the look CONSTANTS remain compile-time; every level is a uniform now.
    // 2^agxEv, folded in at the shader so the branch stays one multiply.
    AGX_GAIN: f(Math.pow(2, FILM.agxEv)),
    // The meshed hull. It is PAINTED, not bare metal: the palette below is the same SHIP_HULL /
    // SHIP_GLASS / SHIP_TRIM the marched hull used, reused rather than re-picked, so the mesh reads as
    // the same ship rather than as a different one that happens to be the same shape.
    //
    // The flat constant material that briefly lived here was a deliberate scaffold — it isolated the
    // geometry while the mesher was being trusted, on the grounds that anything odd had to be the mesh
    // rather than a texture. That job is done and the scaffold comes out.
    //
    SHIPM_ENV: f(0.55),
    SHIPM_CORE: `vec3f(${[0.36, 0.16, 0.30].map(f).join(', ')})`,
    FILM_BLACK: `vec3f(${FILM.black.map(f).join(', ')})`,
    FILM_PIVOT: FILM.pivot,
    // Nacelle hardpoint, so the engine plumes in shipjets.wgsl start at the nozzles the hull actually has
    // rather than at a literal that has to be kept in step by hand.
    SHIP_NACELLE_X: f(SHIP_HARDPOINTS.nacelle[0]),
    SHIP_NACELLE_Y: f(SHIP_HARDPOINTS.nacelle[1]),
    SHIP_NOZZLE_Z: f(SHIP_HARDPOINTS.nozzleZ),
    // The model viewer's studio location, needed by the satellite display layout in WGSL.
    MODEL_ORIGIN: `vec3f(${MODELVIEW.origin.map(f).join(', ')})`,
    // tiling
    TILE: int(QUALITY.tile),
  };
}

