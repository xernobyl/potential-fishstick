/**
 * The chibi grass planet: a small raymarched world covered in grass — rolling
 * hills and a field of blades, adapted from David Hoskins' "Rolling hills".
 *
 * A SELF-CONTAINED MARCH. The whole field, material and lighting live in
 * `chibi.wgsl`, so this scene is a drop-in that touches nothing else in the
 * march. It still reuses the entire shared post chain — TAA, bloom, flare,
 * grade — for free.
 */

import { Scene } from './scene.js';
import { SolidMeshPass } from '../passes/solidmesh.js';
import { AdditivePass } from '../passes/additive.js';
import { Mesh } from '../core/mesh.js';
import { Aurora } from '../scene/aurora.js';
import { buildSatelliteMesh } from './planetoid.js';
import { SATELLITES, AURORA, wgslDefines } from '../scene/tuning.js';

/** A stand-in for the ship, filled once. The uniform writer reads these positionally. */
const DUMMY_SHIP = () => ({
  pos: [0, 0, 0], prevPos: [0, 0, 0],
  rot: [0, 0, 0, 1], prevRot: [0, 0, 0, 1],
  throttle: 0, angAccel: [0, 0, 0], reverse: 0,
});

export class ChibiScene extends Scene {
  static label = 'chibi planet';

  constructor(gpu, targets, shaders) {
    super();
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
    this._ship = DUMMY_SHIP();
    // Arcball state, so a drag re-frames and a release keeps the view.
    this._yaw = 0;
    this._pitch = 0.35;
    this._dist = 3.2;   // camera stand-off, dollied with W/S and wheel

    // The shared world dressing, reused from the planetoid: a curl-noise aurora
    // and the satellite swarm. Both are generic passes, so they drop in with no
    // changes to the march or the post chain.
    this.aurora = new Aurora(gpu.device);
    this.satellites = new SolidMeshPass(gpu, targets, shaders, {
      label: 'satellites',
      shader: 'satmesh.wgsl',
      clear: true,   // the first (and only) solid pass here, so it clears the layer
      mesh: () => new Mesh(gpu.device, buildSatelliteMesh(), 'satellites'),
      instances: (range) => (range.id === 0 ? SATELLITES.count : SATELLITES.count * 2),
    });
    this.auroraPass = new AdditivePass(gpu, targets, shaders, {
      label: 'aurora',
      shader: 'aurora.wgsl',
      vertices: (AURORA.samples - 1) * 6,
      instances: AURORA.ribbons,
      source: () => this.aurora.buffer,
    });
  }

  get solidPasses() { return [this.satellites]; }

  async init(rc) {
    const d = this.gpu.device;
    const defines = wgslDefines();

    this.cullBGL = d.createBindGroupLayout({
      label: 'chibi-cull-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.marchBGL = d.createBindGroupLayout({
      label: 'chibi-march-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    const mod = await this.shaders.module('chibi.wgsl', defines);

    // Both pipelines bind the frame uniform at group 0 and the tile flags at group 1;
    // the march adds its output textures at group 2.
    this.cullPipe = await d.createComputePipelineAsync({
      label: 'chibi-cull',
      layout: d.createPipelineLayout({ bindGroupLayouts: [rc.frameBGL, this.cullBGL] }),
      compute: { module: mod, entryPoint: 'cull_main' },
    });
    this.marchPipe = await d.createComputePipelineAsync({
      label: 'chibi-march',
      layout: d.createPipelineLayout({
        bindGroupLayouts: [rc.frameBGL, this.cullBGL, this.marchBGL],
      }),
      compute: { module: mod, entryPoint: 'march_main' },
    });

    // The shared dressing: the satellite mesh pass and the aurora additive pass.
    await this.satellites.init(rc.frameBGL, defines);
    await this.auroraPass.init(rc.frameBGL, defines);
  }

  #sync() {
    const t = this.targets;
    if (this.generation === t.generation) return;
    this.generation = t.generation;
    const d = this.gpu.device;
    this.cullBG = d.createBindGroup({
      label: 'chibi-cull-bg',
      layout: this.cullBGL,
      entries: [{ binding: 0, resource: { buffer: t.tileFlags } }],
    });
    this.marchBG = d.createBindGroup({
      label: 'chibi-march-bg',
      layout: this.marchBGL,
      entries: [
        { binding: 0, resource: t.sceneRaw.createView() },
        { binding: 1, resource: t.motion.createView() },
      ],
    });
  }

  update(rc) {
    const { input, camera } = rc;

    // Arcball around the planet, idle holds a gentle orbit.
    if (input.everUsed) {
      this._yaw = -(input.x / Math.max(1, input.width) - 0.5) * Math.PI * 2 * 1.6;
      this._pitch = Math.min(1.2, Math.max(-1.2, (input.y / Math.max(1, input.height) - 0.5) * 2.2));
    } else {
      this._yaw += rc.dt * 0.10;
    }

    // Dolly: W / Up move in, S / Down move out (cmd.pitch is +1 on W/Up).
    const dolly = (input.cmd?.pitch ?? 0) * 1.6;
    this._dist = Math.min(8.0, Math.max(1.5, this._dist - dolly * rc.dt));
    // Wheel / trackpad pinch zoom.
    const zoom = Math.exp(Math.min(1.8, Math.max(-2.3, input.zoom ?? 0)));

    const dist = this._dist * zoom;
    const cp = Math.cos(this._pitch);
    const pos = [
      dist * cp * Math.cos(this._yaw),
      dist * Math.sin(this._pitch),
      dist * cp * Math.sin(this._yaw),
    ];
    const fwd = [-pos[0], -pos[1], -pos[2]];
    const len = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
    for (let i = 0; i < 3; i++) fwd[i] /= len;
    camera.lookAt(pos, fwd, dist);

    // Advance the aurora ribbons.
    this.aurora.update(rc.dt, rc.time);
  }

  writeState(st) {
    st.modelView = false;
    st.charge = 0;
    st.ship = this._ship;
    st.auroraPhase = this.aurora.emitPhase;
  }

  recordWorld(encoder, frameBG, profiler, rc) {
    this.#sync();
    const t = this.targets;

    {
      const pass = encoder.beginComputePass({
        label: 'chibi-cull', ...profiler.scope('chibi-cull'),
      });
      pass.setPipeline(this.cullPipe);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this.cullBG);
      pass.dispatchWorkgroups(Math.ceil(t.tilesX / 8), Math.ceil(t.tilesY / 8));
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({
        label: 'chibi-march', ...profiler.scope('chibi-march'),
      });
      pass.setPipeline(this.marchPipe);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this.cullBG);
      pass.setBindGroup(2, this.marchBG);
      pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
      pass.end();
    }

    // The satellite swarm, after the march so the motion sentinel is in place.
    // It clears the solid layer itself (first solid pass), so the previous
    // scene's ship/satellite cannot linger over the planet.
    this.satellites.record(encoder, frameBG, profiler, rc.frustum);
  }

  recordAdditive(encoder, frameBG, profiler) {
    // Clear the shared additive target; the planetoid's embers/contrails must not linger here.
    const pass = encoder.beginRenderPass({
      label: 'chibi-additive-clear',
      colorAttachments: [{
        view: this.targets.ember.createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.end();

    // The aurora, accumulated into the same target.
    this.auroraPass.record(encoder, frameBG, profiler);
  }

  destroy() {
    this.aurora.destroy();
    this.satellites.destroy();
  }
}
