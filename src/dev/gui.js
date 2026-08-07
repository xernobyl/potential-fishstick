/**
 * The tuning panel.
 *
 * Every control here binds STRAIGHT to the live tuning object — no shadow state, no apply
 * button, no serialisation. That is only possible because the values it touches are read fresh
 * every frame: the grade and the glow are uniforms (see FrameUniforms.write), the camera's
 * projection is rebuilt when its inputs change, and the rest are plain objects the renderer
 * dereferences per frame. A control bound to something baked at pipeline creation would be a
 * slider that does nothing, which is worse than no slider, so anything in that category either
 * got moved onto the uniform first or is not exposed here.
 *
 * Loaded LAZILY, on the first press of `g`. lil-gui is 59 KB and this is a debug surface, so
 * it has no business being in the startup path of a renderer whose whole point is frame time.
 *
 * lil-gui (MIT) is vendored rather than fetched from a CDN: this project has no build step and
 * no package manager, and it should keep working with the network off.
 */

import GUI from '../../vendor/lil-gui.esm.js';
import { FILM, GLOW, FLARE, CAMERA, AURORA, TEMPORAL, QUALITY } from '../scene/tuning.js';

/**
 * @param {import('../renderer.js').Renderer} renderer
 * @returns {GUI}
 */
export function buildGui(renderer) {
  const gui = new GUI({ title: 'beep beep beep', width: 300 });

  // ---- grade ----
  // First, because it is the one set of controls you reach for while LOOKING at the image
  // rather than while reasoning about it.
  const film = gui.addFolder('Film');
  film.add(FILM, 'gain', 0.05, 3.0, 0.01).name('gain (pre-grade)');
  film.add(FILM, 'exposure', 0.1, 6.0, 0.01).name('exposure (curve)');
  film.add(FILM, 'white', 2.0, 24.0, 0.1).name('white point');
  film.add(FILM, 'contrast', 0.5, 2.0, 0.01);
  film.add(FILM, 'blackLift', 0.0, 0.12, 0.001).name('black lift');
  film.add(FILM, 'saturation', 0.0, 2.0, 0.01);
  film.add(FILM, 'halation', 0.0, 3.0, 0.01);
  film.add(FILM, 'vignette', 0.0, 1.0, 0.01).name('vignette (cos^4)');
  film.add(FILM, 'grain', 0.0, 0.12, 0.001);

  // ---- glow ----
  const glow = gui.addFolder('Glow').close();
  glow.add(GLOW, 'threshold', 0.0, 6.0, 0.01);
  glow.add(GLOW, 'strength', 0.0, 0.8, 0.005);
  // Both of these are baked into the per-level parameter buffers, which are only written on
  // resize — so they need an explicit rebuild rather than just a new value.
  glow.add(GLOW, 'radius', 0.25, 3.0, 0.01)
    .onChange(() => renderer.passes.bloom.invalidate());
  glow.add(GLOW, 'levelWeight', 0.2, 1.6, 0.01).name('level weight')
    .onChange(() => renderer.passes.bloom.invalidate());
  glow.add(FLARE, 'strength', 0.0, 2.0, 0.01).name('flare');

  // ---- camera ----
  const cam = gui.addFolder('Camera').close();
  cam.add(CAMERA, 'diagonalFov', 40, 140, 0.5).name('fov (diagonal)');
  cam.add(CAMERA, 'distance', 2.0, 12.0, 0.05);
  cam.add(CAMERA, 'zoom', 0.0, 3.0, 0.01).name('dolly amplitude');
  cam.add(CAMERA, 'aperture', 0.0, 0.08, 0.001);
  cam.add(CAMERA, 'focusPull', -2.0, 3.0, 0.01).name('focus pull');
  cam.add(CAMERA, 'roll', 0.0, 0.6, 0.01);

  // ---- auroras ----
  // The flow parameters are read per step, so they take effect on the next frame — but the
  // ribbons already in the buffer were integrated under the OLD ones, so a big change takes a
  // ribbon lifetime to fully show. `reseed` skips that wait.
  const aur = gui.addFolder('Auroras').close();
  aur.add(AURORA, 'speed', 0.05, 1.0, 0.01);
  aur.add(AURORA, 'maxTurn', 0.1, 4.0, 0.05).name('max turn (rad/s)');
  aur.add(AURORA, 'curlScale', 0.05, 1.2, 0.01).name('curl scale');
  aur.add(AURORA, 'curlDrift', 0.0, 0.5, 0.005).name('curl drift');
  aur.add(AURORA, 'gain', 0.0, 3.0, 0.01);
  aur.add(AURORA, 'grazeFade', 0.05, 1.0, 0.01).name('graze fade');
  aur.add(AURORA, 'rays', 0.0, 1.0, 0.01).name('ray depth');
  aur.add({ reseed: () => renderer.aurora.reseed() }, 'reseed');

  // ---- temporal ----
  const taa = gui.addFolder('Temporal').close();
  taa.add(TEMPORAL, 'blend', 0.02, 1.0, 0.01);
  taa.add(TEMPORAL, 'clipGamma', 0.5, 4.0, 0.05).name('clip gamma');
  taa.add(TEMPORAL, 'clipGammaUpsample', 0.5, 4.0, 0.05).name('clip gamma (TAAU)');
  taa.add(TEMPORAL, 'depthGate', 0.001, 0.06, 0.001).name('depth gate');
  taa.add(TEMPORAL, 'weightMax', 1, 32, 1).name('weight max');
  taa.add(TEMPORAL, 'taauSigma', 0.4, 2.0, 0.01).name('TAAU sigma');
  // Reallocates the accumulation buffer, so it goes through the renderer rather than
  // being poked directly.
  taa.add(QUALITY, 'taau').name('upsampling').onChange(() => renderer.resize());
  taa.add(QUALITY, 'renderScale', 0.25, 1.0, 0.05).name('render scale')
    .onChange(() => renderer.resize());
  // Reallocates the additive target. Off by default; `beep.additive()` measures both sides of
  // the trade, and on a contended machine the cost side of it does not resolve.
  taa.add(QUALITY, 'additiveDisplayRes').name('additive @ display res')
    .onChange(() => renderer.resize());

  return gui;
}
