/**
 * Dynamic resolution: hold a frame-time target by moving the render scale.
 *
 * This is only worth having because of temporal upsampling, and the reason is structural rather
 * than incidental. Without it the accumulation buffer IS the render target, so every scale change
 * either resamples the history or throws it away — a visible pop to a noisy image every time the
 * controller moves. With it the history lives at output resolution and only the input sample
 * density changes, so the accumulated detail survives the transition. See `Targets.own` versus
 * `ownOut` for the allocation split that makes that true.
 *
 * The lever is unusually effective here: the frame is march-bound, and the march is per-pixel
 * compute, so its cost tracks the scale almost linearly.
 *
 * Design notes, all of them about not making things worse:
 *
 *   GPU TIME, and nothing else. Wall time includes present pacing, so a vsync-limited frame reads
 *   as exactly on budget however much headroom there is — and worse, at 60 Hz it reads as 16.7 ms
 *   forever, so against a 14 ms target the controller would find itself over budget every single
 *   frame and pin itself to the bottom rung. There is no honest fallback, so without timestamp
 *   queries this REFUSES TO RUN rather than driving off a number that cannot mean what it needs to.
 *
 *   RAW samples, PUSHED. `Profiler.timings` is exponentially smoothed so the HUD stays readable,
 *   and a median of an already-smoothed series is a lag on top of a lag. Worse, readbacks land
 *   every few frames while a frame hook fires every frame, so POLLING it filled the window with
 *   duplicates — and the median of one repeated number is that number. The profiler reports the
 *   raw total once per resolved frame and this consumes those.
 *
 *   A MEDIAN over a window, never a single frame. Frame times here are spiky — a detonation or a
 *   ring sweeping across the view moves the march by milliseconds — and reacting to one sample
 *   would oscillate.
 *
 *   A LADDER of STEPS, and a step is more than a render scale. Every change costs a reallocation of
 *   the render-resolution targets and a discontinuity in sample density, so few decisive rungs beat
 *   fine tracking. The LAST rung gives up the additive layer's resolution rather than the render
 *   scale — measured at 2.2x worse sub-pixel stability, so it is genuinely a last resort, which is
 *   what the bottom of a ladder is for.
 *
 *   THE ROAD RUNS OUT, and it stops here on purpose. What does not scale even after that rung — the
 *   resolve above all, then the glow and the composite — is a few milliseconds of the frame, and
 *   reducing it means lowering the OUTPUT resolution. That is leaving temporal upsampling's premise
 *   rather than tuning it, so the ladder does not go there.
 *
 *   ASYMMETRY. Drop quickly when over budget, recover slowly. Being briefly too slow is worse than
 *   being briefly too soft, and climbing eagerly into a spike then falling straight back out is
 *   the failure mode that makes dynamic resolution feel worse than a fixed low setting.
 */

import { QUALITY } from './tuning.js';

export class DynamicRes {
  constructor() {
    this.samples = [];
    this.cooldown = 0;
    this.index = 0;
    this.lastChange = 0;
  }

  /** Nearest rung to the live settings, so switching the feature on does not jump the image. */
  #snap() {
    const l = QUALITY.dynamicLadder;
    let best = 0;
    let bestCost = Infinity;
    for (let i = 0; i < l.length; i++) {
      // Scale dominates; the additive flag only breaks ties between rungs of equal scale.
      const cost = Math.abs(l[i].scale - QUALITY.renderScale)
                 + (l[i].additive === QUALITY.additiveDisplayRes ? 0 : 0.01);
      if (cost < bestCost) { bestCost = cost; best = i; }
    }
    return best;
  }

  /**
   * One genuine GPU-time sample. Pushed by the profiler when a readback resolves — see the note on
   * raw samples above.
   */
  sample(ms) {
    if (!(ms > 0) || !Number.isFinite(ms)) return;
    this.samples.push(ms);
    if (this.samples.length > QUALITY.dynamicWindow) this.samples.shift();
  }

  /**
   * Decide, from the samples pushed since the last decision.
   *
   * @returns {{scale: number, additive: boolean}|null} a rung to apply, or null to hold
   */
  update() {
    if (this.cooldown > 0) { this.cooldown--; return null; }
    if (this.samples.length < QUALITY.dynamicWindow) return null;

    const sorted = this.samples.slice().sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    const target = QUALITY.dynamicTargetMs;
    const ladder = QUALITY.dynamicLadder;

    // Re-derived from the live settings, so a slider or a preset moving them underneath the
    // controller is picked up rather than fought.
    this.index = this.#snap();

    let next = this.index;
    if (median > target * QUALITY.dynamicDropAt && this.index < ladder.length - 1) {
      // Over budget: step down. More than one rung if badly over, because crawling down through a
      // ladder while every frame misses is the same as not adapting.
      const over = median / target;
      next = Math.min(ladder.length - 1, this.index + (over > 1.5 ? 2 : 1));
    } else if (median < target * QUALITY.dynamicRaiseAt && this.index > 0) {
      next = this.index - 1;
    }
    if (next === this.index) return null;

    this.index = next;
    // The window is cleared, not kept: the samples in it describe a resolution that no longer
    // applies, and averaging across a change is how a controller convinces itself to oscillate.
    this.samples.length = 0;
    this.cooldown = QUALITY.dynamicCooldown;
    return ladder[next];
  }

  /** Called when something else takes over the scale, so the controller does not fight it. */
  reset() {
    this.samples.length = 0;
    this.cooldown = QUALITY.dynamicCooldown;
  }
}
