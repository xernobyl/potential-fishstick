/**
 * The water planet: an ocean world, raymarched with a realistic PBR water
 * surface. Reuses the planetoid's Fibonacci octaves-of-spheres geometry (the
 * same spherical-Fibonacci machinery) but shades it as water: GGX sun glints on
 * a calm normal, analytic sphere-integral AO, a real refracted interior, and a
 * slow animated ripple. Same sky, same post chain.
 *
 * A SELF-CONTAINED MARCH, like the chibi scene: the field, material and lighting
 * live in `water.wgsl`, so this scene drops in and touches nothing else in the
 * march. It still reuses the entire shared post chain — TAA, bloom, flare,
 * grade — for free.
 */

import { Scene } from './scene.js';
import { wgslDefines, SUNS, BODY } from '../scene/tuning.js';

const WATER_BOUND = BODY.R * 1.4;   // water sphere radius (matches water.wgsl bound)

/** Ray-sphere: does a ray from `o` toward unit `d` hit the origin sphere of `r`? */
function occludesSphere(o, d, r) {
  const b = o[0] * d[0] + o[1] * d[1] + o[2] * d[2];
  const c = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - r * r;
  const h = b * b - c;
  if (h < 0) return false;       // misses entirely
  const t = -b - Math.sqrt(h);
  return t > 0;                   // hits in front of the ray origin
}

export class WaterScene extends Scene {
  static label = 'water planet';

  constructor(gpu, targets, shaders) {
    super();
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;
    this.generation = -1;
    // Orbit state for an idle gentle tumble around the world.
    this._yaw = 0.4;
    this._pitch = 0.12;
    this._dist = 5.0;
    this._camPos = [5.0, 0, 0];   // matches the initial orbit position
  }

  get solidPasses() { return []; }

  async init(rc) {
    const d = this.gpu.device;
    const defines = wgslDefines();

    this.cullBGL = d.createBindGroupLayout({
      label: 'water-cull-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.marchBGL = d.createBindGroupLayout({
      label: 'water-march-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    const mod = await this.shaders.module('water.wgsl', defines);

    this.cullPipe = await d.createComputePipelineAsync({
      label: 'water-cull',
      layout: d.createPipelineLayout({ bindGroupLayouts: [rc.frameBGL, this.cullBGL] }),
      compute: { module: mod, entryPoint: 'cull_main' },
    });
    this.marchPipe = await d.createComputePipelineAsync({
      label: 'water-march',
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
      label: 'water-cull-bg',
      layout: this.cullBGL,
      entries: [{ binding: 0, resource: { buffer: t.tileFlags } }],
    });
    this.marchBG = d.createBindGroup({
      label: 'water-march-bg',
      layout: this.marchBGL,
      entries: [
        { binding: 0, resource: t.sceneRaw.createView() },
        { binding: 1, resource: t.motion.createView() },
      ],
    });
  }

  update(rc) {
    const { input, camera } = rc;

    // Orbit around the world; a drag re-frames, otherwise an idle drift.
    if (input.everUsed) {
      this._yaw = -(input.x / Math.max(1, input.width) - 0.5) * Math.PI * 2 * 1.6;
      this._pitch = Math.min(1.2, Math.max(-1.2, (input.y / Math.max(1, input.height) - 0.5) * 2.2));
    } else {
      this._yaw += rc.dt * 0.06;
    }

    const dolly = (input.cmd?.pitch ?? 0) * 1.6;
    this._dist = Math.min(12.0, Math.max(3.0, this._dist - dolly * rc.dt));
    const zoom = Math.exp(Math.min(1.8, Math.max(-2.3, input.zoom ?? 0)));

    const dist = this._dist * zoom;
    const cp = Math.cos(this._pitch);
    const pos = [
      dist * cp * Math.cos(this._yaw),
      dist * Math.sin(this._pitch),
      dist * cp * Math.sin(this._yaw),
    ];
    this._camPos = pos;
    const fwd = [-pos[0], -pos[1], -pos[2]];
    const len = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
    for (let i = 0; i < 3; i++) fwd[i] /= len;
    camera.lookAt(pos, fwd, dist);
  }

  writeState(st) {
    st.modelView = false;
    st.charge = 0;
    // Sun occlusion: is each sun's direction blocked by the water sphere? The
    // lens flare's streak must not draw through the body. A ray from the camera
    // toward the sun intersects the origin-centred sphere of radius
    // WATER_BOUND iff the sun is behind the planet.
    st.sunOccA = occludesSphere(this._camPos, SUNS.a.dir, WATER_BOUND) ? 1 : 0;
    st.sunOccB = occludesSphere(this._camPos, SUNS.b.dir, WATER_BOUND) ? 1 : 0;
  }

  recordWorld(encoder, frameBG, profiler) {
    this.#sync();
    const t = this.targets;

    // No solid meshes: clear the solid colour + depth layer so the previous
    // scene's meshes cannot linger over the water.
    {
      const pass = encoder.beginRenderPass({
        label: 'water-solid-clear',
        colorAttachments: [{
          view: t.solid.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear', storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: t.solidDepth.createView(),
          depthClearValue: 0.0,
          depthLoadOp: 'clear', depthStoreOp: 'store',
        },
      });
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({
        label: 'water-cull', ...profiler.scope('water-cull'),
      });
      pass.setPipeline(this.cullPipe);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this.cullBG);
      pass.dispatchWorkgroups(Math.ceil(t.tilesX / 8), Math.ceil(t.tilesY / 8));
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({
        label: 'water-march', ...profiler.scope('water-march'),
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
      label: 'water-additive-clear',
      colorAttachments: [{
        view: this.targets.ember.createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.end();
  }
}
