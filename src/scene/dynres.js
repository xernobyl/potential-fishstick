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
 * compute, so its cost tracks the scale almost linearly. What caps the win is everything that does
 * NOT scale — the resolve, the additive layer, the glow, the composite — which is about 10 ms of a
 * 35 ms frame. Below roughly that the controller runs out of road, and no ladder step will help.
 *
 * Design notes, all of them about not making things worse:
 *
 *   GPU TIME, not wall time. Wall includes present pacing, so a vsync-limited frame reads as
 *   "exactly on budget" no matter how much headroom there is, and the controller would never
 *   climb. The profiler's timestamp queries give the real figure; wall is only the fallback when
 *   they are unavailable.
 *
 *   A MEDIAN over a window, never a single frame. Frame times here are spiky — a detonation or a
 *   ring sweeping across the view moves the march by milliseconds — and reacting to one sample
 *   would oscillate.
 *
 *   A LADDER, not a continuous scale. Every change costs a reallocation of the render-resolution
 *   targets and a discontinuity in sample density, so the useful thing is few, decisive steps
 *   rather than fine tracking.
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

  /** Nearest ladder rung to a scale, so switching the feature on does not jump the image. */
  #snap(scale) {
    const l = QUALITY.dynamicLadder;
    let best = 0;
    for (let i = 1; i < l.length; i++) {
      if (Math.abs(l[i] - scale) < Math.abs(l[best] - scale)) best = i;
    }
    return best;
  }

  /**
   * @param {number} gpuMs  measured GPU time for the frame, or 0 if unavailable
   * @param {number} dt     seconds, the wall-clock fallback
   * @returns {number|null} a new render scale to apply, or null to leave it alone
   */
  update(gpuMs, dt) {
    const ms = gpuMs > 0 ? gpuMs : dt * 1000;
    if (!(ms > 0) || !Number.isFinite(ms)) return null;

    this.samples.push(ms);
    if (this.samples.length > QUALITY.dynamicWindow) this.samples.shift();
    if (this.cooldown > 0) { this.cooldown--; return null; }
    if (this.samples.length < QUALITY.dynamicWindow) return null;

    const sorted = this.samples.slice().sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    const target = QUALITY.dynamicTargetMs;
    const ladder = QUALITY.dynamicLadder;

    // Re-derive the rung from the live value, so a slider or a preset moving `renderScale`
    // underneath the controller is picked up rather than fought.
    this.index = this.#snap(QUALITY.renderScale);

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
