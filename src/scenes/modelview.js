/**
 * The model viewer: one generated mesh on a turntable, lit and nothing else.
 *
 * A quarter top-down view of the ship or a satellite, spinning, against the sky. It exists because the
 * meshes are GENERATED — there is no file to open in a modelling tool, so the only way to look at what
 * the contourer actually produced used to be to find the object in the scene and chase it with the
 * camera. Now it is a dropdown.
 *
 * IT REUSES THE MODELS' OWN SHADERS, unchanged. The ship draws through `shipmesh.wgsl` and the satellite
 * through `satmesh.wgsl` — the same passes, the same materials, the same vertex front end the real scene
 * uses. That is the property worth protecting: a viewer with its own simplified shader shows you a
 * model that nothing in the game renders, and the discrepancies it hides are exactly the ones you opened
 * it to find. Combined with the wireframe view on `B`, this inspects the real thing.
 *
 * THE SPIN IS THE MODEL, NOT THE CAMERA. Rotating the camera keeps the lighting fixed to the model and
 * shows you the same shading from every angle; rotating the MODEL sweeps it through fixed light, which
 * is how you see a form. Both models therefore take their turntable angle from `frame.model`, and both
 * take the PREVIOUS angle from it too — so the spin gets an exact motion vector and the temporal resolve
 * treats it exactly as it treats the real scene.
 *
 * WHY IT STILL RUNS THE MARCH PASS. It needs a backdrop, a depth tag of "background" everywhere the
 * model is not, and a motion sentinel; the scene pass already produces all three. A `studio` flag in
 * `raymarch.wgsl` drops the body, the plumes and the atmosphere and leaves the sky. Skipping the pass
 * outright would leave the previous scene's samples in the buffer for the resolve to blend against.
 */

import { Scene } from './scene.js';
import { ScenePass } from '../passes/scene.js';
import { SolidMeshPass } from '../passes/solidmesh.js';
import { Mesh } from '../core/mesh.js';
import { buildShipMesh, buildSatelliteMesh } from './planetoid.js';
import { MODELVIEW, SATELLITES, wgslDefines } from '../scene/tuning.js';

/**
 * The models on offer. `key` selects the pass, `label` names it in the dropdown.
 *
 * Framing is NOT listed here: the camera stands off a multiple of the model's bounding radius, and
 * that radius comes from the mesh itself where there is one and from the satellite's boom reach where
 * the geometry is instanced and has no single sphere. See `#radius`. A number copied into this table
 * would be a second source of truth for something the mesh already knows.
 */
export const MODELS = [
  { key: 'ship', label: 'ship' },
  { key: 'satellite', label: 'satellite' },
];

export class ModelViewScene extends Scene {
  static label = 'model viewer';

  constructor(gpu, targets, shaders) {
    super();
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    /** Index into MODELS. Driven by the GUI. */
    this.model = 0;
    this.spin = 0;
    this.prevSpin = 0;
    // A stand-in for the real ship, filled each frame. The plumes are gated off in studio mode, so the
    // throttle and thruster fields stay zero and nothing downstream has to know they are not a ship.
    this._ship = {
      pos: MODELVIEW.origin, prevPos: MODELVIEW.origin,
      rot: [0, 0, 0, 1], prevRot: [0, 0, 0, 1],
      throttle: 0, angAccel: [0, 0, 0], reverse: 0,
    };

    this.passes = {
      // The backdrop, in studio mode: sky only. See the header.
      scene: new ScenePass(gpu, targets, shaders),

      // Both models are built here and only one is recorded per frame. They are small and building
      // them lazily on the first switch would stall the frame that switched, which is the one frame
      // where a stall is most obviously the viewer's fault.
      ship: new SolidMeshPass(gpu, targets, shaders, {
        label: 'model-ship',
        shader: 'shipmesh.wgsl',
        mesh: () => buildShipMesh().map((data, i) => {
          const m = new Mesh(gpu.device, data, `model-ship-lod${i}`);
          m.errorWorld = data.errorWorld;
          return m;
        }),
        // The hull has an LOD chain, so it needs a sphere for the level to be chosen from a real
        // distance. It is not used for culling here — a single object dead centre is always
        // visible — but the two questions share the one fact about where the object is.
        worldSphere: (_id, r) => [
          MODELVIEW.origin[0], MODELVIEW.origin[1], MODELVIEW.origin[2], r.radius,
        ],
      }),
      satellite: new SolidMeshPass(gpu, targets, shaders, {
        label: 'model-satellite',
        shader: 'satmesh.wgsl',
        mesh: () => new Mesh(gpu.device, buildSatelliteMesh(), 'model-satellite'),
        // ONE satellite, not the whole swarm: one bus and its two array wings.
        instances: (range) => (range.id === 0 ? 1 : 2),
      }),
    };
  }

  /**
   * Only the pass being shown.
   *
   * The renderer applies LOD and the wireframe view to whatever this lists, so listing the hidden model
   * too would have it selecting a level and compiling a wireframe pipeline for geometry nobody is
   * looking at.
   */
  get solidPasses() {
    return [this.#active()];
  }

  #active() {
    return MODELS[this.model].key === 'ship' ? this.passes.ship : this.passes.satellite;
  }

  /** The model's bounding radius in world units, for framing. */
  #radius() {
    if (MODELS[this.model].key === 'ship') {
      return this.passes.ship.meshes?.[0]?.ranges[0].radius ?? 1;
    }
    // The satellite is instanced from a unit cube, so no single mesh sphere describes it. Its reach is
    // the boom plus a panel half-length, which is the same number the orbital version bounds itself by.
    return SATELLITES.boom + SATELLITES.panelLen * 2;
  }

  async init(rc) {
    await Promise.all([
      this.passes.scene.init(rc.frameBGL),
      this.passes.ship.init(rc.frameBGL, wgslDefines()),
      this.passes.satellite.init(rc.frameBGL, wgslDefines()),
    ]);
  }

  update(rc) {
    const { dt, camera } = rc;
    this.prevSpin = this.spin;
    this.spin += dt * MODELVIEW.spinRate;

    // A QUARTER TOP-DOWN VIEW: elevated, looking down and in. Placed rather than integrated, so
    // switching models re-frames immediately instead of easing across the studio.
    const o = MODELVIEW.origin;
    const dist = this.#radius() * MODELVIEW.distance;
    const el = MODELVIEW.elevation;
    // Offset in the model's own frame: back along -Z, up by the elevation. The model spins, so a fixed
    // camera sees every side without the lighting following it around.
    const pos = [
      o[0],
      o[1] + Math.sin(el) * dist,
      o[2] - Math.cos(el) * dist,
    ];
    const fwd = [o[0] - pos[0], o[1] - pos[1], o[2] - pos[2]];
    const len = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
    for (let i = 0; i < 3; i++) fwd[i] /= len;
    camera.lookAt(pos, fwd, dist);
  }

  writeState(st) {
    st.modelView = true;
    // No weapon in the studio; the charge swarm reads this and would otherwise draw the planetoid's.
    st.charge = 0;
    st.modelSpin = this.spin;
    st.modelPrevSpin = this.prevSpin;

    // The hull rides on the ship transform the real scene uses, so `shipmesh.wgsl` needs no branch at
    // all: put the model at the studio origin and hand it the turntable angle as an orientation. The
    // PREVIOUS pose is the previous angle, which is what gives the spin an exact motion vector.
    const o = MODELVIEW.origin;
    //
    // Every field the uniform writer reads, and no more — it indexes them positionally, so a missing
    // one is a NaN in the block rather than an error at the call site.
    this._ship.pos = o;
    this._ship.prevPos = o;
    this._ship.rot = quatY(this.spin);
    this._ship.prevRot = quatY(this.prevSpin);
    st.ship = this._ship;
  }

  recordWorld(encoder, frameBG, p, rc) {
    this.passes.scene.record(encoder, frameBG, p);
    // No frustum planes: one object, dead centre, always visible. Passing them would only spend a
    // sphere test to reach the same conclusion.
    this.#active().record(encoder, frameBG, p, null);
  }

  destroy() {
    // Both models, not just the visible one — the pass built its buffers at init either way.
    this.passes.ship.destroy();
    this.passes.satellite.destroy();
  }

  recordAdditive(encoder) {
    // NOTHING TO ADD, BUT THE LAYER STILL HAS TO BE CLEARED. In the real scene the ember pass clears it
    // on its way through; with no ember pass here, skipping this entirely would leave the planetoid's
    // last frame of embers and contrails hanging over the model after a scene switch. An empty
    // clearing pass is the cheapest correct answer and costs one attachment write.
    const pass = encoder.beginRenderPass({
      label: 'model-additive-clear',
      colorAttachments: [{
        view: this.targets.ember.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.end();
  }
}

/** A quaternion for a rotation of `a` about +Y. */
function quatY(a) {
  return [0, Math.sin(a * 0.5), 0, Math.cos(a * 0.5)];
}
