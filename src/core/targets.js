/**
 * Every render target, allocated in one place.
 *
 * Centralised because the sizes are interdependent — the bloom pyramid derives
 * from the scene resolution, the tile buffer from the scene resolution and the
 * tile size — and because a resize has to rebuild them together or a pass will
 * quietly sample a stale texture of the wrong size.
 *
 * `generation` bumps on every rebuild; passes compare it to know when their
 * cached bind groups are invalid. Bind groups hold concrete texture views, so
 * they cannot outlive a reallocation.
 */

import { QUALITY } from '../scene/tuning.js';

/** Linear HDR everywhere. rgba16float is the sweet spot: enough range for the
 *  emissive core, half the bandwidth of rgba32float, and filterable. */
const HDR = 'rgba16float';

function tex(device, label, width, height, usage, format = HDR) {
  return device.createTexture({
    label,
    size: { width: Math.max(1, width), height: Math.max(1, height) },
    format,
    usage,
  });
}

const U = GPUTextureUsage;

export class Targets {
  constructor(device) {
    this.device = device;
    this.generation = 0;
    this.width = 0;
    this.height = 0;
    this._owned = [];
    this._ownedOut = new Map();

    // Samplers never change, so build them once. Clamping matters: the bloom and
    // flare passes reach outside the frame, and repeating would wrap a bright
    // corner round to the opposite edge.
    this.linear = device.createSampler({
      label: 'linear-clamp',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.nearest = device.createSampler({
      label: 'nearest-clamp',
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  /**
   * @param {number} displayW  swapchain width
   * @param {number} displayH  swapchain height
   */
  resize(displayW, displayH) {
    const scale = QUALITY.renderScale;
    const w = Math.max(4, Math.round(displayW * scale));
    const h = Math.max(4, Math.round(displayH * scale));
    // The TAAU flag changes the accumulation size without changing the render size, so it
    // has to participate in this test or toggling it would silently keep the old buffers.
    const wantAccumW = QUALITY.taau ? displayW : w;
    // DISPLAY width, not the accumulation width. Tying it to the latter made the flag a no-op
    // whenever upsampling was off — the composite still runs at display resolution and still
    // upscales this layer, so there is a real difference to be had at renderScale 0.5 with
    // `taau` false, and the flag was quietly declining to take it.
    const wantAddW = QUALITY.additiveDisplayRes ? displayW : w;
    const renderSame = w === this.width && h === this.height;
    const outSame = wantAccumW === this.accumWidth && wantAddW === this.addWidth
                 && displayW === this.displayWidth && displayH === this.displayHeight;
    if (renderSame && outSame) return false;

    // ONLY WHAT CHANGED. A render-scale change leaves the output group — the accumulation buffers
    // above all — untouched, so the accumulated image survives it. That is the whole point: see
    // the note on `own` versus `ownOut`. The return value says whether the HISTORY was lost, so
    // the caller knows whether it has to reset the accumulator, and a scale change no longer
    // reports that it does.
    this.destroyOwned();
    if (!outSame) this.destroyOwnedOut();
    this.width = w;
    this.height = h;
    this.displayWidth = displayW;
    this.displayHeight = displayH;

    const d = this.device;

    // Scene: written by the raymarch compute pass, read by TAA.
    this.sceneRaw = this.own(tex(d, 'scene-raw', w, h,
      U.STORAGE_BINDING | U.TEXTURE_BINDING));

    // Accumulation ping-pong. Both need STORAGE (written) and TEXTURE (read as
    // history, and read by everything downstream).
    // COPY_SRC so the stability benchmark can read the buffer back and diff it. That
    // is the only honest way to measure residual jitter: scraping the canvas is not
    // synchronised with presentation and returns numbers that disagree run to run.
    //
    // At DISPLAY resolution when temporal upsampling is on, which is the whole mechanism:
    // the history is the thing that has to be full-resolution, because that is where the
    // jittered low-res samples accumulate into real detail. Everything upstream of it
    // stays at render resolution. Nothing may assume these two are the same size — the
    // `accumRes` uniform exists so consumers ask instead of assuming.
    this.accumWidth = QUALITY.taau ? displayW : w;
    this.accumHeight = QUALITY.taau ? displayH : h;
    const aw = this.accumWidth;
    const ah = this.accumHeight;
    this.accum = [
      this.ownOut('accum-0', () => tex(d, 'accum-0', aw, ah,
        U.STORAGE_BINDING | U.TEXTURE_BINDING | U.COPY_SRC)),
      this.ownOut('accum-1', () => tex(d, 'accum-1', aw, ah,
        U.STORAGE_BINDING | U.TEXTURE_BINDING | U.COPY_SRC)),
    ];
    this.accumIndex = 0;

    // Accumulated WEIGHT per output pixel, ping-ponged alongside the colour.
    //
    // Temporal upsampling needs a true running weighted average, not an exponential blend of
    // a per-frame reconstruction: at 2x each frame supplies one sample per four output pixels,
    // so normalising per frame produces an estimate sitting at the CENTROID of whichever
    // samples landed nearby — and that centroid slides with the jitter, which wobbles no
    // matter how the kernel is shaped. Carrying the denominator instead lets the samples
    // accumulate at their true positions.
    //
    // r32float, because r16float is not a core WebGPU storage format — the same trap that
    // shaped the motion target. It is `unfilterable-float` when sampled, so the resolve does
    // its own bilinear from four loads rather than pretending a sampler will do it.
    //
    // A separate single-channel target rather than stealing the colour's alpha: that alpha is
    // the depth TAG, which is signed, class-encoded, and must never be interpolated. Moving
    // the tag out to make room would touch five consumers instead of one.
    // Initial contents are ZERO, and that is a WebGPU guarantee rather than an assumption:
    // the spec requires resources be zero-initialised on first use. Zero is also exactly the
    // right value — no accumulated weight means the first frame after allocation falls back to
    // its own reconstruction, which is what `resetHistory` arranges anyway.
    this.accumWeight = [
      this.ownOut('accum-w-0', () => tex(d, 'accum-w-0', aw, ah, U.STORAGE_BINDING | U.TEXTURE_BINDING, 'r32float')),
      this.ownOut('accum-w-1', () => tex(d, 'accum-w-1', aw, ah, U.STORAGE_BINDING | U.TEXTURE_BINDING, 'r32float')),
    ];

    // Particles land in their own additive target: they are dynamic, so folding
    // them into the accumulation buffer would trail every mote.
    // The ADDITIVE target, and its resolution is a real decision rather than an accident.
    //
    // Everything drawn here — particles, the contrail, the rail guns, the auroras — is drawn
    // AFTER the temporal resolve, because reprojection assumes static geometry and any of them
    // would ghost. That also means none of them get temporal antialiasing, so at render
    // resolution their aliasing is upscaled along with them. `additiveDisplayRes` trades fill
    // rate for that; measure before choosing, since the ember sprites dominate the fill.
    //
    // COPY_SRC for the same reason the bloom pyramid has it: this layer is the one the temporal
    // filter never sees, so it is the one whose stability has to be MEASURED rather than assumed.
    this.addWidth = QUALITY.additiveDisplayRes ? displayW : w;
    this.addHeight = QUALITY.additiveDisplayRes ? displayH : h;
    this.ember = this.ownOut('ember', () => tex(d, 'ember', this.addWidth, this.addHeight,
      U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC));

    // Rasterised OPAQUE geometry (the rings), with alpha carrying linear view
    // distance so the composite can resolve it against the marched body. Named for
    // what it is rather than for its current occupant: this is the second feature to
    // want exactly this, so another one should add a draw, not another target.
    this.solid = this.own(tex(d, 'solid', w, h,
      U.RENDER_ATTACHMENT | U.TEXTURE_BINDING));
    this.solidDepth = this.own(tex(d, 'solid-depth', w, h,
      U.RENDER_ATTACHMENT, 'depth24plus'));
    // Per-pixel motion, for every surface that can say where it was last frame.
    //
    //   xy  screen-space PIXEL DELTA to the previous position. A delta rather than an
    //       absolute coordinate because deltas are small, and f16 holds integers
    //       exactly only to 2048 — a 4K target would exceed that.
    //   z   the distance from the PREVIOUS camera to that previous point, which is the
    //       only quantity TAA's depth gate can honestly compare its history against.
    //   w   the distance from the CURRENT camera to the surface that wrote this, i.e. a
    //       claim of OWNERSHIP. The producers cannot see each other — the rings have
    //       their own depth buffer and know nothing of the body or the ship — so a ring
    //       behind the body still stamps its motion onto that pixel. Without this
    //       channel the consumer reprojects the body along the ring's motion, which is
    //       tens of pixels wrong every frame.
    //
    // rgba16float rather than rg16float because it must also be a STORAGE texture: the
    // scene pass is compute and writes the ship's motion directly, and rg16float is not
    // a core WebGPU storage format. Cleared to a sentinel so "no stored motion" is
    // distinguishable from "did not move".
    this.motion = this.own(tex(d, 'motion', w, h,
      U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.STORAGE_BINDING));

    // Bloom pyramid. Level 0 is half the RENDER scene, and each level halves again.
    //
    // Deliberately sized off the render resolution rather than the accumulation buffer's,
    // because bloom is a blur and feeding it display-resolution input would cost 4x to
    // produce the same halo. The consequence to know about: with temporal upsampling on, the
    // prefilter reads a display-resolution source into a quarter-resolution target, so it is
    // a 4x downsample in one step and undersamples the very detail upsampling just resolved.
    // It is invisible in a wide soft glow, which is why it is left alone — but if the bloom
    // ever picks up crawling on fine highlights, this is where it comes from, and the fix is
    // a proper mip chain on the way in rather than a wider prefilter.
    //
    // ALWAYS at least one level. The loop used to test before pushing, so a
    // sufficiently small target produced an EMPTY pyramid — and the flare pass
    // then indexed `bloom[min(1, length-1)]` == `bloom[-1]` and threw on undefined.
    // Guaranteeing one level is the honest fix: a renderer that has a bloom stage
    // at all should not have a size at which that stage silently ceases to exist.
    this.bloom = [];
    // From the DISPLAY size, not the render size, and that is a fix rather than a preference.
    // The upsample offsets are in source texels, so a pyramid sized off the render target made the
    // glow's radius IN DISPLAY PIXELS a function of `renderScale` — halve the scale and the glow
    // doubled in width. Under a controller that adapts continuously the bloom would visibly
    // breathe. A screen-space effect should not know what the render scale is.
    let bw = Math.max(4, displayW >> 1);
    let bh = Math.max(4, displayH >> 1);
    for (let i = 0; i < QUALITY.bloomLevels; i++) {
      this.bloom.push({
        // COPY_SRC so the pyramid can be read back and measured. Free on every
        // implementation that matters, and the alternative is reasoning about the
        // glow's alignment instead of measuring it.
        texture: this.ownOut(`bloom-${i}`, () => tex(d, `bloom-${i}`, bw, bh,
          U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC)),
        width: bw,
        height: bh,
      });
      // Stop before a level would stop being smaller than its parent, which would
      // add cost and blur nothing.
      const nw = bw >> 1;
      const nh = bh >> 1;
      if (nw < 4 || nh < 4) break;
      bw = nw;
      bh = nh;
    }

    // Flares: quarter res. They are the blurriest thing on screen by
    // construction, so there is no edge here worth resolving.
    // Display-based, for the same reason as the pyramid above.
    this.flareWidth = Math.max(4, displayW >> 2);
    this.flareHeight = Math.max(4, displayH >> 2);
    this.flare = this.ownOut('flare', () => tex(d, 'flare', this.flareWidth, this.flareHeight,
      U.RENDER_ATTACHMENT | U.TEXTURE_BINDING));

    // One u32 per raymarch tile.
    this.tilesX = Math.ceil(w / QUALITY.tile);
    this.tilesY = Math.ceil(h / QUALITY.tile);
    this.tileFlags?.destroy();
    this.tileFlags = d.createBuffer({
      label: 'tile-flags',
      size: Math.max(4, this.tilesX * this.tilesY * 4),
      usage: GPUBufferUsage.STORAGE,
    });

    this.generation++;
    // TRUE means the history is gone and the caller must reset the accumulator. A render-scale
    // change alone does not lose it.
    return !outSame;
  }

  /** The accumulation target being written this frame, and last frame's. */
  get accumWrite() { return this.accum[this.accumIndex]; }
  get accumRead() { return this.accum[1 - this.accumIndex]; }
  get accumWeightWrite() { return this.accumWeight[this.accumIndex]; }
  get accumWeightRead() { return this.accumWeight[1 - this.accumIndex]; }
  swapAccum() { this.accumIndex = 1 - this.accumIndex; }

  /**
   * Ownership, in TWO GROUPS, and the split is the whole reason dynamic resolution works.
   *
   * `own` is render-resolution: everything whose size follows `renderScale`. `ownOut` is
   * output-resolution: the accumulation buffers, their weights, the additive layer, the bloom
   * pyramid and the flares — everything sized from the display.
   *
   * A render-scale change must reallocate the first group and LEAVE THE SECOND ALONE. That is the
   * property temporal upsampling exists to give: the history lives at output resolution, so
   * changing the input sample density does not invalidate it. Destroying everything on any resize
   * — which is what this did — threw the accumulated image away on every scale change, which is
   * exactly the pop that makes naive dynamic resolution look bad.
   */
  own(t) {
    this._owned.push(t);
    return t;
  }

  /**
   * Keyed and IDEMPOTENT, unlike `own`. A render-scale change re-runs the whole of `resize`, so the
   * output allocations are reached again even though they must not be redone — returning the
   * existing texture is what makes that safe, and it keeps the allocation code in one readable
   * sequence instead of split across two conditional blocks. The factory is a thunk so nothing is
   * created just to be thrown away.
   */
  ownOut(key, make) {
    const have = this._ownedOut.get(key);
    if (have) return have;
    const t = make();
    this._ownedOut.set(key, t);
    return t;
  }

  destroyOwned() {
    for (const t of this._owned) t.destroy();
    this._owned.length = 0;
  }

  destroyOwnedOut() {
    for (const t of this._ownedOut.values()) t.destroy();
    this._ownedOut.clear();
    this.bloom = [];
  }

  destroy() {
    this.destroyOwned();
    this.destroyOwnedOut();
    this.tileFlags?.destroy();
  }
}
