// ---------------------------------------------------------------------------
// The scene pass: one jittered radiance sample per pixel, into a linear HDR
// target. Alpha carries the depth tag (see common.wgsl).
//
// A compute pass rather than a fullscreen fragment pass, for two reasons:
//  - the tile flag is shared by the whole workgroup, so the skip is coherent;
//  - it writes a storage texture directly, with no attachment/blend state.
//
// The grade, glow, flares and grain all live downstream, because accumulating
// and blooming have to happen in linear light.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "volumetric.wgsl"
//!include "hash.wgsl"
//!include "sdf.wgsl"
//!include "sky.wgsl"
//!include "shade.wgsl"
//!include "satellite.wgsl"
//!include "ship.wgsl"

@group(1) @binding(0) var outTex : texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<storage, read> tileFlags : array<u32>;
@group(1) @binding(2) var motionTex : texture_storage_2d<rgba16float, write>;

/// A synthetic, NOISE-FREE test signal: a greyscale zone plate, whose spatial frequency
/// rises with radius so a single image spans well below Nyquist at the centre to right at it
/// in the corners.
///
/// Why a synthetic pattern is worth having: the real scene's own sampling noise — few-tap
/// AO, SSS, transmission, a lens offset that moves every frame — is the same order as the
/// artefacts being hunted, so a frozen-scene residual cannot separate "the sampler is noisy"
/// from "the resolve is geometrically wrong". This signal has no noise at all; it is an exact
/// function of screen position. On a correct pipeline the frozen residual over the low
/// frequency centre must therefore fall to almost nothing, and whatever is left is the
/// resolve's own error rather than the scene's.
///
/// Defined on the shared screen space, so it is identical at any resolution or aspect and a
/// render-resolution change cannot be mistaken for a change in the signal.
fn testPattern(uv : vec2f) -> vec3f {
  return vec3f(0.5 + 0.5 * cos(TEST_PATTERN_K * dot(uv, uv)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u,
        @builtin(workgroup_id) wid : vec3u) {
  let size = vec2u(frame.res.xy);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let tiles = (size + u32(TILE) - 1u) / u32(TILE);
  let mayHitBody = tileFlags[wid.y * tiles.x + wid.x] != 0u;

  // The jitter is what buys anti-aliasing over time; the lens offset is what
  // turns one sample per frame into a real bokeh once accumulated.
  let px = vec2f(gid.xy) + 0.5 + frame.jitter.xy;
  let ray = cameraRay(px, frame.jitter.zw);

  var col = background(ray.d);
  var depthTag = TAG_BG;

  // ---- the body ----
  if (mayHitBody) {
    let bound = bodyBound();
    let bs = iSphere(ray.o, ray.d, bound);
    if (bs.y > 0.0) {
      var t = max(bs.x, 0.0);
      let tmax = bs.y;
      var hit = false;
      // TWO step scales, because the field's conditioning is not uniform.
      //
      // Near the surface the DETAIL noise adds ~0.3 to |grad d| (its amplitude times its
      // ~10/unit gradient) and the capH placement term adds more, so steps above ~0.7 of
      // the reported distance overshoot and crease it. But mapBody only mixes that noise
      // in within 0.20 of the surface — outside that band the field is a plain smin of
      // exact sphere distances, which is properly 1-Lipschitz and can be marched almost
      // fully.
      //
      // Safe OVER-relaxation (Keinert et al. 2014) was implemented here and REMOVED. It
      // was correct — the unbounding-sphere overlap test, applied only in the Lipschitz
      // region where the premise holds — and it was pointless: the counter below put the
      // mean at 12.36 field evaluations per pixel without it and 12.41 with it. There is
      // no step-count headroom in this march to win. See the note in tuning.js.
      //
      // Counted, not assumed. Three attempts to make this march cheaper all measured as
      // noise, because the cost model behind them was wrong; this counter is what
      // established that 12 evaluations, not ~60, is the real figure.
      var evals = 0;
      for (var i = 0; i < MARCH_STEPS; i++) {
        let d = mapBody(ray.o + ray.d * t);
        evals += 1;
        if (d < MARCH_HIT_EPS * t) { hit = true; break; }
        t += d * select(frame.march.y, frame.march.z, d < frame.march.w);
        if (t > tmax) { break; }
      }
      if (hit) {
        col = shadeBody(ray.o + ray.d * t, ray.d, t);
        depthTag = t;
      }
      // Cost probe: report the field-evaluation count as radiance, scaled so the patch
      // mean read back over the wire is directly the mean count. Everything downstream
      // still runs, so this measures the real march rather than a stripped copy of it.
      if (frame.probe.y > 0.5) {
        col = vec3f(f32(evals) * PROBE_EVAL_SCALE);
      }
    }
  }

  // ---- satellites, composited by depth ----
  // Tested against the body's hit distance so they occlude and are occluded
  // correctly without ever entering the marched field. They live outside the
  // body's tiles, so this runs everywhere.
  var nearT = 1e4;
  if (depthTag > 0.0) { nearT = depthTag; }

  let sat = hitSatellites(ray.o, ray.d, nearT);
  if (sat.hit) {
    col = shadeSatellite(sat, ray.o, ray.d);
    // A POSITIVE tag, as the ship writes: the motion vector below is exact for these too, so they
    // take the reprojecting path instead of the dynamic tag's same-pixel history. That fallback is
    // why they crawled while the rings - which do reproject along their own motion - sat still.
    depthTag = sat.t;
    nearT = sat.t;
  }

  let ship = hitShip(ray.o, ray.d, nearT);
  if (ship.hit) {
    col = shadeShip(ship, ray.o, ray.d);
    // A POSITIVE depth tag, not the dynamic one. The ship writes an exact motion
    // vector below, so TAA can follow it properly instead of falling back to
    // fresh-sample-heavy screen-space history.
    depthTag = ship.t;
    nearT = ship.t;
  }

  // Plumes and thruster puffs, occluded by whatever ended up in front.
  col += shipJets(ray.o, ray.d, nearT);

  // ---- the atmosphere ----
  //
  // LAST, and after `nearT` is final. It integrates only as far as the nearest thing this pass
  // resolved, so it has to run once every candidate for that has been considered — the body, the
  // satellites and the ship all set it, and putting this before them would have drawn air through
  // whichever of them was in front.
  //
  // Here rather than in the composite so the temporal resolve averages it: every sample is a world
  // position and the step offset moves every frame, so 12 steps converge into a smooth integral
  // rather than 12 visible shells. See volumetric.wgsl for why this is not a froxel grid.
  //
  // Skipped under the field-evaluation probe, which reports work as radiance and must not have
  // anything added to it.
  if (frame.probe.y <= 0.5) {
    let vol = volumetric(ray.o, ray.d, nearT, vec2f(gid.xy));
    col = col * vol.transmittance + vol.inScatter;
  }

  // Motion. The sentinel goes everywhere by default — the body and the sky are handled
  // by camera reprojection, which is exact for them and needs nothing stored. Only the
  // ship overrides it, and its motion is EXACT rather than estimated: `lp` is the hit
  // point in the hull's own space, so the same piece of hull can be found in last
  // frame's transform directly.
  var motion = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  // The PIXEL CENTRE, shared by both writers below. `uvToPixel` returns centre-relative
  // coordinates, so pairing it with the integer corner would put a fixed half-pixel bias into every
  // motion vector - which is indistinguishable from the crawl these are here to remove.
  let motionPx = vec2f(gid.xy) + 0.5;
  if (ship.hit) {
    // Exact: `lp` is the hit point in the hull's own space, so the same piece of hull is found
    // directly in last frame's transform.
    motion = motionFor(frame.prevShipPos.xyz + qrotate(frame.prevShipRot, ship.lp),
                       motionPx, ship.t);
  } else if (sat.hit && frame.probe.z <= 0.5) {
    // Also exact, and for the same reason by a different route: the orbits are analytic, so the
    // previous transform is the same evaluation one frame back. `misc.w` is dt.
    motion = motionFor(satPrevWorld(sat, frame.camPos.w - frame.misc.w), motionPx, sat.t);
  }
  textureStore(motionTex, vec2i(gid.xy), motion);

  // Replace only the COLOUR: the depth tags, motion vectors and everything the temporal
  // gates key off stay exactly as the real scene produced them, so this measures the resolve
  // under production conditions rather than a simplified stand-in for it.
  if (frame.probe.w > 0.5) {
    col = testPattern(screenUV(px));
  }

  textureStore(outTex, vec2i(gid.xy), vec4f(col, depthTag));
}
