import { whiteBalanceGains } from '../scene/tuning.js';
/**
 * The per-frame uniform block, shared by every pass as bind group 0.
 *
 * Three deliberate choices:
 *
 * 1. The camera is MATRICES. Ray generation, temporal reprojection and the
 *    ember billboards all need the same projection; expressing it once as data
 *    is what stops them disagreeing, and it takes the field of view out of the
 *    shader constants so it can change per frame.
 *
 * 2. Everything else is a vec4. WGSL's uniform address space rounds mat3x3
 *    columns to 16 bytes and aligns vec3 to 16, which is a rich source of
 *    silent, garbled data. Packing by hand as vec4 makes the layout obvious and
 *    impossible to get subtly wrong. The mat4x4s are 16-byte aligned already and
 *    sit first, so no padding is ever needed.
 *
 * 3. One buffer, written once per frame from a single ArrayBuffer. Splitting by
 *    update frequency is the usual advice, but here *everything* changes every
 *    frame, so a single write of the whole block beats several small ones.
 */

// REPURPOSING A SLOT MEANS GREPPING ITS READERS FIRST. Removing depth of field freed `camRight.w`
// (the aperture) and `camUp.w` (the focus distance), and both were quietly still in use: tilecull
// inflated its cull margin by the aperture, and the composite's spherochromatism keyed off the focus
// distance. Neither failed loudly - one silently changed the culling whenever the buffer viewer was
// open, the other degenerated to a constant colour rim on every highlight. A slot is not free just
// because the feature that named it is gone.
export const FRAME_FLOATS = 172;                // 3 mat4x4 + 31 vec4
export const FRAME_BYTES = FRAME_FLOATS * 4;

/** Field offsets, in floats. Kept next to the WGSL struct in common.wgsl. */
const O = {
  viewProj: 0,      // world -> clip, this frame
  invViewProj: 16,  // clip  -> world, this frame (ray generation)
  prevViewProj: 32, // world -> clip, previous frame (reprojection)
  camRight: 48,     // xyz right,      w spare (explicitly zeroed)
  camUp: 52,        // xyz up,         w focus distance (composite's spherochromatism)
  camFwd: 56,       // xyz forward,    w focal length (half-diagonal units)
  camPos: 60,       // xyz position,   w time
  res: 64,          // xy size,        zw 1/size
  screen: 68,       // x 1/diagPx, y diagPx, zw sensor half-extents
  misc: 72,         // x beat, y life, z frameIndex, w dt
  jitter: 76,       // xy pixel jitter, zw spare (was the lens offset)
  flags: 80,        // x historyValid, y dragging, z fireflyMax, w exposure
  sun: 84,          // xy sun-A screen pos, zw sun-B screen pos
  shipPos: 88,      // xyz world position, w throttle
  shipRot: 92,      // orientation quaternion, world <- body
  shipJet: 96,      // xyz angular acceleration (what the RCS reacts to), w reverse
  // The PREVIOUS camera position. Needed because the accumulation buffer stores
  // distances measured from wherever the camera was when it wrote them, so TAA's depth
  // gate cannot compare them against distances from where the camera is now.
  prevCamPos: 100,
  // The ship's transform LAST frame. With these, a hit on the hull can be mapped back
  // to exactly where that piece of hull was, because the local hit point is known —
  // the same exactness the rings get, rather than an estimate.
  prevShipPos: 104,
  prevShipRot: 108,
  // TAA knobs live HERE rather than as injected WGSL consts, so they can be changed at
  // runtime and therefore actually A/B tested. As constants they were baked at pipeline
  // creation, and console tweaks silently did nothing.
  taa: 112,         // x blend, y clipGamma, z clipFloor, w depthGate
  // The depth gate's SLOPE-SCALED term. Same idea as a rasteriser's
  // depthBiasSlopeScale: a fixed tolerance is wrong on a steep surface, because there
  // one pixel of reprojection error is legitimately a large depth change.
  taa2: 116,        // x depthGradSlack (pixels), y depthGradMax, z historyFilter, w taauSigma
  march: 120,       // x bandLimit (probe), y far, z near, w nearBand
  probe: 124,       // x latticeTable, y showFieldEvals, z noSatMotion, w testPattern
  // Size of the ACCUMULATION buffer, which with temporal upsampling is the DISPLAY
  // resolution while frame.res stays the render resolution. Every accum consumer reads
  // this rather than assuming the two are equal.
  accumRes: 128,    // xy size, zw 1/size
  taa3: 132,        // x weightMax, y weightMaxBg, z clipGammaUpsample, w spare
  // The grade. Uniforms rather than injected consts, for the same reason the TAA knobs are:
  // a const is baked at pipeline creation, so a slider bound to one moves nothing.
  grade: 136,       // x filmExposure, y white, z halation, w saturation
  grade2: 140,      // x vignette, y grain, z blackLift, w bloomStrength
  grade3: 144,      // x contrast, y flareStrength, z bloomThreshold, w toneMap (0 Hable, 1 AgX)
  aurora: 148,      // x gain, y rays, z grazeFade, w emission phase 0..1
  addRes: 152,      // xy additive-target size, zw 1/size
  volume: 156,      // x sigma, y ringOpacity, z g, w spare
  // White-balance gains, derived from FILM.temperature. A uniform and not an injected const for the
  // same reason the grade is: a const is baked at pipeline creation, so a slider bound to one moves
  // nothing.
  //
  // APPENDED, and the block grew from 160 floats to 164 to hold it. There was no free slot: `volume`
  // was already at 156, and the first version of this put `balance` there too - which does not
  // collide loudly, it silently overwrites the atmosphere's sigma with a red gain. The WGSL struct
  // below is what caught it, because a member has to exist in the struct at that offset and the
  // order there is the layout. Anything added here goes at the END and grows FRAME_FLOATS.
  balance: 160,     // xyz linear per-channel gains, w buffer-viewer display mode
  // The MODEL VIEWER's state, and zero in every other scene — which is what makes `w` a usable flag.
  // x: turntable angle now, y: the same one frame ago (so the spin gets an exact motion vector),
  // z: unused, w: 1 while the model-viewer scene is active. See scenes/modelview.js.
  model: 164,
  // The trigger's live state, for the muzzle glow the charge draws. x: charge 0..1, y: the hue the
  // shot will come out as, so the glow previews it, z/w spare.
  weapon: 168,
};

export class FrameUniforms {
  constructor(device) {
    this.device = device;
    this.cpu = new Float32Array(FRAME_FLOATS);
    this.buffer = device.createBuffer({
      label: 'frame-uniforms',
      size: FRAME_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * @param {object} s        frame state
   * @param {import('../scene/camera.js').Camera} s.camera
   */
  write(s) {
    const a = this.cpu;
    const camera = s.camera;
    const cam = camera.current;

    a.set(camera.viewProj, O.viewProj);
    a.set(camera.invViewProj, O.invViewProj);
    a.set(camera.prevViewProj, O.prevViewProj);

    // The basis survives the move to matrices because two things genuinely want
    // it in world space: the thin-lens offset (which shifts the ray ORIGIN, not
    // its projection) and the ember billboards' quad expansion.
    a[O.camRight] = cam.right[0]; a[O.camRight + 1] = cam.right[1]; a[O.camRight + 2] = cam.right[2];
    // Explicitly zeroed rather than left alone: a spare slot still holding whatever a removed
    // feature put there is how that feature keeps half-working.
    a[O.camRight + 3] = 0.0;

    a[O.camUp] = cam.up[0]; a[O.camUp + 1] = cam.up[1]; a[O.camUp + 2] = cam.up[2];
    a[O.camUp + 3] = s.focusDist;

    a[O.camFwd] = cam.fwd[0]; a[O.camFwd + 1] = cam.fwd[1]; a[O.camFwd + 2] = cam.fwd[2];
    // The focal length, in the same half-diagonal units as the shared screen space — so
    // tan(field angle) is |screenUV| / focal, with no other conversion. The vignette needs it,
    // and needs it as a uniform rather than a const because the field of view is now a slider.
    a[O.camFwd + 3] = camera.focal;

    a[O.camPos] = cam.pos[0]; a[O.camPos + 1] = cam.pos[1]; a[O.camPos + 2] = cam.pos[2];
    a[O.camPos + 3] = s.time;

    a[O.res] = s.width; a[O.res + 1] = s.height;
    a[O.res + 2] = 1 / s.width; a[O.res + 3] = 1 / s.height;

    a.set(camera.screen, O.screen);

    a[O.misc] = s.beat; a[O.misc + 1] = s.life;
    a[O.misc + 2] = s.frameIndex; a[O.misc + 3] = s.dt;

    a[O.jitter] = s.jitter[0]; a[O.jitter + 1] = s.jitter[1];

    a[O.flags] = s.historyValid ? 1 : 0;
    a[O.flags + 1] = s.dragging ? 1 : 0;
    // Absolute firefly ceiling. Lives in what was the spare slot rather than growing the
    // block for one float, and being a uniform it is tunable without a recompile.
    a[O.flags + 2] = s.taa.fireflyMax;
    a[O.flags + 3] = s.exposure;

    a[O.sun] = s.sunA[0]; a[O.sun + 1] = s.sunA[1];
    a[O.sun + 2] = s.sunB[0]; a[O.sun + 3] = s.sunB[1];

    const sh = s.ship;
    a[O.shipPos] = sh.pos[0]; a[O.shipPos + 1] = sh.pos[1]; a[O.shipPos + 2] = sh.pos[2];
    a[O.shipPos + 3] = sh.throttle;
    a.set(sh.rot, O.shipRot);
    a[O.shipJet] = sh.angAccel[0]; a[O.shipJet + 1] = sh.angAccel[1];
    a[O.shipJet + 2] = sh.angAccel[2];
    a[O.shipJet + 3] = sh.reverse;

    const pv = camera.previous.pos;
    a[O.prevCamPos] = pv[0]; a[O.prevCamPos + 1] = pv[1]; a[O.prevCamPos + 2] = pv[2];

    a[O.prevShipPos] = sh.prevPos[0]; a[O.prevShipPos + 1] = sh.prevPos[1];
    a[O.prevShipPos + 2] = sh.prevPos[2];
    a.set(sh.prevRot, O.prevShipRot);

    const tw = s.taa;
    a[O.taa] = tw.blend; a[O.taa + 1] = tw.clipGamma;
    a[O.taa + 2] = tw.clipFloor; a[O.taa + 3] = tw.depthGate;
    a[O.taa2] = tw.depthGradSlack; a[O.taa2 + 1] = tw.depthGradMax;
    a[O.taa2 + 2] = tw.historyFilter; a[O.taa2 + 3] = tw.taauSigma;

    const mr = s.march;
    a[O.march] = s.probe.bandLimit; a[O.march + 1] = mr.far;
    a[O.march + 2] = mr.near; a[O.march + 3] = mr.nearBand;

    a[O.probe] = s.probe.latticeTable;
    a[O.probe + 1] = s.probe.showFieldEvals;
    a[O.probe + 2] = s.probe.noSatMotion;
    a[O.probe + 3] = s.probe.testPattern;

    a[O.accumRes] = s.accumWidth; a[O.accumRes + 1] = s.accumHeight;
    a[O.accumRes + 2] = 1 / s.accumWidth; a[O.accumRes + 3] = 1 / s.accumHeight;

    // Sample density relative to what the caps were tuned at — see TEMPORAL.weightRefScale.
    const dens = (s.renderScale / tw.weightRefScale) ** 2;
    a[O.taa3] = tw.weightMax * dens; a[O.taa3 + 1] = tw.weightMaxBg * dens;
    a[O.taa3 + 2] = tw.clipGammaUpsample;

    const gr = s.grade;
    a[O.grade] = gr.exposure; a[O.grade + 1] = gr.white;
    a[O.grade + 2] = gr.halation; a[O.grade + 3] = gr.saturation;
    a[O.grade2] = gr.vignette; a[O.grade2 + 1] = gr.grain;
    a[O.grade2 + 2] = gr.blackLift; a[O.grade2 + 3] = s.glow.strength;
    // MEMOISED on the temperature: the derivation walks two polynomials and a matrix, which is
    // nothing against a frame but is also pointless to redo when the slider has not moved.
    if (gr.temperature !== this._wbK || gr.sceneTemperature !== this._wbScene) {
      this._wbK = gr.temperature;
      this._wbScene = gr.sceneTemperature;
      this._wb = whiteBalanceGains(gr.temperature, gr.sceneTemperature);
    }
    a[O.balance] = this._wb[0]; a[O.balance + 1] = this._wb[1]; a[O.balance + 2] = this._wb[2];
    a[O.balance + 3] = s.viewMode;
    a[O.model] = s.modelSpin ?? 0;
    a[O.model + 1] = s.modelPrevSpin ?? 0;
    a[O.model + 2] = 0;
    a[O.model + 3] = s.modelView ? 1 : 0;
    a[O.weapon] = s.charge ?? 0;
    a[O.weapon + 1] = s.chargeHue ?? 0;
    a[O.weapon + 2] = 0;
    a[O.weapon + 3] = 0;

    a[O.grade3] = gr.contrast; a[O.grade3 + 1] = s.flareStrength;
    a[O.grade3 + 2] = s.glow.threshold;
    a[O.grade3 + 3] = gr.toneMap;

    a[O.addRes] = s.addWidth; a[O.addRes + 1] = s.addHeight;
    a[O.addRes + 2] = 1 / s.addWidth; a[O.addRes + 3] = 1 / s.addHeight;

    const vol = s.volume;
    a[O.volume] = vol.sigma; a[O.volume + 1] = vol.ringOpacity;
    a[O.volume + 2] = vol.g;

    const au = s.aurora;
    a[O.aurora] = au.gain; a[O.aurora + 1] = au.rays;
    // Not a tuning value: how far through the current emission interval the simulation is. The
    // shader adds it to every sample's age so the ring buffer's one-sample shift stops being
    // visible. See Aurora.emitPhase.
    a[O.aurora + 2] = au.grazeFade; a[O.aurora + 3] = s.auroraPhase;

    this.device.queue.writeBuffer(this.buffer, 0, a);
  }

  /** Layout entry shared by every pass, so bind group 0 is interchangeable. */
  static bindGroupLayout(device) {
    return device.createBindGroupLayout({
      label: 'frame-bgl',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', minBindingSize: FRAME_BYTES },
      }],
    });
  }

  bindGroup(layout) {
    return this.device.createBindGroup({
      label: 'frame-bg',
      layout,
      entries: [{ binding: 0, resource: { buffer: this.buffer } }],
    });
  }
}
