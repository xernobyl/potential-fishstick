/**
 * Scene1Rasterized — GPU-driven rasterized planet matching the raymarched body.
 *
 * UV sphere mesh with mapBody surface extraction. Fragment shader calls
 * shadeBody + volumetric for identical shading. Shared state with PlanetoidScene.
 *
 * 1=planetoid  2=rasterized  3=modelview
 * B: debug views (0=Lit 1=Wire 2=Points 3=Normals 4=SDF 5=Winding)
 */

import { Scene } from './scene.js';
import { Ship } from '../scene/ship.js';
import { Contrail } from '../scene/contrail.js';
import { Railgun } from '../scene/railgun.js';
import { Aurora } from '../scene/aurora.js';
import { AdditivePass } from '../passes/additive.js';
import { SolidMeshPass } from '../passes/solidmesh.js';
import { Mesh } from '../core/mesh.js';
import { PlanetMeshManager } from '../passes/PlanetMeshManager.js';
import { buildRingMesh, buildShipMesh, buildSatelliteMesh } from './planetoid.js';
import { RINGS, SATELLITES, SHIP, CONTRAIL, RAIL, AURORA as AURORA_TUNE,
         wgslDefines } from '../scene/tuning.js';
import { extractFrustum } from '../core/frustum.js';

const DEBUG_NAMES = ['Lit', 'Wire', 'Points', 'Normals', 'SDF', 'Winding'];

export class Scene1Rasterized extends Scene {
  static label = 'planet rasterized';

  constructor(gpu, targets, shaders, planetoid) {
    super();
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;

    // Share simulation state with PlanetoidScene for seamless switching
    if (planetoid) {
      this.ship = planetoid.ship;
      this.contrail = planetoid.contrail;
      this.railgun = planetoid.railgun;
      this.aurora = planetoid.aurora;
      this._fireSlot = planetoid._fireSlot;
    } else {
      this.ship = new Ship(); this.contrail = new Contrail(gpu.device);
      this.railgun = new Railgun(gpu.device); this.aurora = new Aurora(gpu.device);
      this._fireSlot = -1;
    }

    this.meshManager = new PlanetMeshManager(gpu.device);
    this._frustum = new Float32Array(20);
    this.debugMode = 0;
    this._prevCam = new Float32Array(3);
    this._cpuUni = new Float32Array(16);
    this._hudEl = document.getElementById('view');

    this.passes = {
      rings: new SolidMeshPass(gpu, targets, shaders, {
        label: 'rings', shader: 'rings.wgsl', depthClear: false,
        mesh: () => new Mesh(gpu.device, buildRingMesh(), 'rings'),
        worldSphere: (_id, r) => [0, 0, 0, r.radius],
      }),
      shipmesh: new SolidMeshPass(gpu, targets, shaders, {
        label: 'ship-mesh', shader: 'shipmesh.wgsl', clear: false,
        mesh: () => buildShipMesh().map((data, i) => {
          const m = new Mesh(gpu.device, data, `ship-lod${i}`);
          m.errorWorld = data.errorWorld;
          return m;
        }),
        worldSphere: (_id, r) =>
          [this.ship.pos[0], this.ship.pos[1], this.ship.pos[2], r.radius],
      }),
      satellites: new SolidMeshPass(gpu, targets, shaders, {
        label: 'satellites', shader: 'satmesh.wgsl', clear: false,
        mesh: () => new Mesh(gpu.device, buildSatelliteMesh(), 'satellites'),
        instances: (range) => (range.id === 0 ? SATELLITES.count : SATELLITES.count * 2),
      }),
    };
    this._solid = [this.passes.rings, this.passes.shipmesh, this.passes.satellites];
  }

  get solidPasses() { return this._solid; }

  cycleDebug() {
    this.debugMode = (this.debugMode + 1) % DEBUG_NAMES.length;
    if (this._hudEl) {
      this._hudEl.style.display = 'block';
      this._hudEl.textContent = `debug ${this.debugMode}: ${DEBUG_NAMES[this.debugMode]}`;
    }
  }

  async init(rc) {
    const d = this.gpu.device, fbgl = rc.frameBGL, defines = wgslDefines();

    this._drawUniformBuf = d.createBuffer({
      label: 'planet-draw-uniform', size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const meshBGL = this.meshManager.mesherBGL(d);
    const drawBGL = this.meshManager.drawBGL(d);

    const [meshMod, skyMod, drawMod] = await Promise.all([
      this.shaders.module('planet_mesher.wgsl', defines),
      this.shaders.module('planet_sky.wgsl', defines),
      this.shaders.module('planet_raster.wgsl', defines),
    ]);

    const meshLayout = d.createPipelineLayout({
      label: 'planet-mesh-layout', bindGroupLayouts: [fbgl, meshBGL],
    });

    this._meshPipe = await d.createComputePipelineAsync({
      label: 'planet-mesh', layout: meshLayout,
      compute: { module: meshMod, entryPoint: 'main' },
    });
    this._meshBG = this.meshManager.buildMesherBG(d, meshBGL);

    this._skyPipe = await d.createRenderPipelineAsync({
      label: 'planet-sky',
      layout: d.createPipelineLayout({ label: 'sky-layout', bindGroupLayouts: [fbgl] }),
      vertex: { module: skyMod, entryPoint: 'vs' },
      fragment: { module: skyMod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });

    const depthStencil = {
      format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'greater-equal',
    };

    this._drawPipe = await d.createRenderPipelineAsync({
      label: 'planet-draw',
      layout: d.createPipelineLayout({ label: 'draw-layout', bindGroupLayouts: [fbgl, drawBGL] }),
      vertex: { module: drawMod, entryPoint: 'vs' },
      fragment: { module: drawMod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil,
    });

    this._wirePipe = await d.createRenderPipelineAsync({
      label: 'planet-wire',
      layout: d.createPipelineLayout({ label: 'wire-layout', bindGroupLayouts: [fbgl, drawBGL] }),
      vertex: { module: drawMod, entryPoint: 'vs' },
      fragment: { module: drawMod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'line-list', cullMode: 'none' },
      depthStencil,
    });

    this._pointPipe = await d.createRenderPipelineAsync({
      label: 'planet-points',
      layout: d.createPipelineLayout({ label: 'point-layout', bindGroupLayouts: [fbgl, drawBGL] }),
      vertex: { module: drawMod, entryPoint: 'vs' },
      fragment: { module: drawMod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'point-list', cullMode: 'none' },
      depthStencil,
    });

    this._dBG = this.meshManager.buildDrawBG(d, drawBGL, this._drawUniformBuf);

    // ---- additive passes ----
    const add = this.additive = {
      embers: null,
      contrail: new AdditivePass(this.gpu, this.targets, this.shaders, {
        label: 'contrail', shader: 'contrail.wgsl', vertices: (CONTRAIL.samples - 1) * 6, instances: 2,
        source: () => this.contrail.buffer,
      }),
      railgun: new AdditivePass(this.gpu, this.targets, this.shaders, {
        label: 'railgun', shader: 'railgun.wgsl', vertices: RAIL.segments * 6,
        instances: RAIL.pool * RAIL.strands, source: () => this.railgun.buffer,
      }),
      sparks: new AdditivePass(this.gpu, this.targets, this.shaders, {
        label: 'sparks', shader: 'sparks.wgsl', vertices: 6, instances: (RAIL.pool + 1) * RAIL.sparks,
        source: () => this.railgun.buffer,
      }),
      shipjets: new AdditivePass(this.gpu, this.targets, this.shaders, {
        label: 'ship-jets', shader: 'shipjets.wgsl', vertices: 6, instances: SHIP.jetLength * 2 + 2 + 3,
      }),
      aurora: new AdditivePass(this.gpu, this.targets, this.shaders, {
        label: 'aurora', shader: 'aurora.wgsl',
        vertices: (AURORA_TUNE.samples - 1) * 6, instances: AURORA_TUNE.ribbons,
        source: () => this.aurora.buffer,
      }),
    };

    const { EmberPass } = await import('../passes/embers.js');
    add.embers = new EmberPass(this.gpu, this.targets, this.shaders);
    await add.embers.init(fbgl);

    await Promise.all([
      ...this._solid.map((m) => m.init(fbgl, defines)),
      add.contrail.init(fbgl, defines), add.railgun.init(fbgl, defines),
      add.sparks.init(fbgl, defines), add.aurora.init(fbgl, defines),
      add.shipjets.init(fbgl, defines),
    ]);
  }

  update(rc) {
    const { dt, time, input, camera } = rc;
    this.ship.update(dt, input.cmd);
    this.contrail.update(dt, this.ship);
    let fire = false;
    if (!this.ship.flown) {
      const slot = Math.floor(time / SHIP.autoFireEvery);
      if (slot !== this._fireSlot) { this._fireSlot = slot; fire = true; }
    }
    this.railgun.update(time, dt, this.ship, !!input.cmd.trigger, fire);
    const kick = this.railgun.takeKick();
    if (kick > 0) camera.kick(kick * RAIL.shake);
    this.aurora.update(dt, time);
    camera.update(time, dt, input, this.ship);
  }

  writeState(st) {
    st.ship = this.ship;
    st.auroraPhase = this.aurora.emitPhase;
    st.modelView = false;
    st.charge = this.railgun.chargeFrac;
    st.chargeHue = this.railgun.chargeHue;
  }

  recordWorld(encoder, frameBG, profiler, rc) {
    const t = this.targets, cam = rc.camera;
    const grid = this.meshManager.resolveGrid();
    this.meshManager.writeUniform(cam, grid);

    const cp = cam.current.pos, du = this._cpuUni;
    du[0] = cp[0]; du[1] = cp[1]; du[2] = cp[2]; du[3] = 0;
    du[4] = this._prevCam[0]; du[5] = this._prevCam[1]; du[6] = this._prevCam[2]; du[7] = 0;
    du[8] = this.debugMode; du[9] = 0; du[10] = 0; du[11] = 0;
    this.gpu.device.queue.writeBuffer(this._drawUniformBuf, 0, du);
    this._prevCam.set(cp);

    // Compute
    encoder.pushDebugGroup('planet-mesh');
    {
      const pass = encoder.beginComputePass({ label: 'planet-mesh' });
      pass.setPipeline(this._meshPipe);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this._meshBG);
      pass.dispatchWorkgroups(Math.ceil(grid.nx * grid.ny / 64));
      pass.end();
    }
    encoder.popDebugGroup();

    // Sky
    {
      const pass = encoder.beginRenderPass({
        label: 'planet-sky',
        colorAttachments: [{ view: t.sceneRaw.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: -1 }, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(this._skyPipe);
      pass.setBindGroup(0, frameBG);
      pass.draw(3);
      pass.end();
    }

    // Planet
    {
      const wire = this.debugMode === 1, points = this.debugMode === 2;
      const pipe = points ? this._pointPipe : (wire ? this._wirePipe : this._drawPipe);
      const pass = encoder.beginRenderPass({
        label: 'planet-body',
        colorAttachments: [{ view: t.sceneRaw.createView(), loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
          view: t.solidDepth.createView(),
          depthClearValue: 0.0, depthLoadOp: 'clear', depthStoreOp: 'store',
        },
        ...profiler.scope('planet'),
      });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, frameBG);
      pass.setBindGroup(1, this._dBG);
      if (points) {
        pass.draw(this.meshManager.vertexBuffer.size / 24);
      } else {
        pass.setIndexBuffer(this.meshManager.ensureIndexBuffer(grid), 'uint32');
        pass.drawIndexed(this.meshManager.indexCount(grid));
      }
      pass.end();
    }

    // Solid meshes
    extractFrustum(cam.viewProj, this._frustum);
    for (const m of this._solid) m.record(encoder, frameBG, profiler, this._frustum);
  }

  recordAdditive(encoder, frameBG, profiler) {
    const a = this.additive;
    a.embers.simulate(encoder, frameBG, profiler);
    a.embers.record(encoder, frameBG, profiler);
    a.contrail.record(encoder, frameBG, profiler);
    a.railgun.record(encoder, frameBG, profiler);
    a.sparks.record(encoder, frameBG, profiler);
    a.aurora.record(encoder, frameBG, profiler);
    a.shipjets.record(encoder, frameBG, profiler);
  }

  destroy() {
    this.contrail?.destroy(); this.railgun?.destroy(); this.aurora?.destroy();
    this.additive?.embers?.destroy?.();
    for (const m of this._solid) m.destroy();
    this.meshManager.destroy();
    this._drawUniformBuf?.destroy();
  }
}
