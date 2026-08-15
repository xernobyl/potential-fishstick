/**
 * The chibi football planet: a small raymarched world — a pitch, a running track,
 * grandstands and a bouncing football — with cartoon PBR and corner spotlights.
 *
 * A SELF-CONTAINED MARCH. Unlike the planetoid it does not share `sdf.wgsl` /
 * `shade.wgsl`: the whole field, material and lighting live in `chibi.wgsl`, so
 * this scene is a drop-in that touches nothing else in the march. It still
 * reuses the entire shared post chain — TAA, bloom, flare, grade — for free,
 * because it produces the same depth-tagged scene output every scene does.
 */

import { Scene } from './scene.js';
import { wgslDefines } from '../scene/tuning.js';

/** A stand-in for the ship, filled once. The uniform writer reads these positionally. */
const DUMMY_SHIP = () => ({
  pos: [0, 0, 0], prevPos: [0, 0, 0],
  rot: [0, 0, 0, 1], prevRot: [0, 0, 0, 1],
  throttle: 0, angAccel: [0, 0, 0], reverse: 0,
});

export class ChibiScene extends Scene {
  static label = 'chibi football';

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
  }

  get solidPasses() { return []; }

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

    const dist = 3.2;
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
  }

  writeState(st) {
    st.modelView = false;
    st.charge = 0;
    st.ship = this._ship;
  }

  recordWorld(encoder, frameBG, profiler) {
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
  }

  recordAdditive(encoder) {
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
  }
}
