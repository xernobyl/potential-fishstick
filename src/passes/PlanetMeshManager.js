/**
 * PlanetMeshManager — UV sphere mesh with GPU SDF displacement.
 * Precomputed index buffers on CPU. Zero render-loop allocations.
 */

import { BODY } from '../scene/tuning.js';

export const PLANET_MESH = { ny: 128, depthExponent: 2.2, tanWarp: 0.7,
  rMin: BODY.R * 0.55, rMax: BODY.R * 2.65, maxTerrainAmp: BODY.R * 0.55 };

const MAX_NX = 384, MAX_NY = 256, STRIDE = 6;
const GRID_FLOATS = 28, GRID_BYTES = GRID_FLOATS * 4;

function buildIndices(Nx, Ny) {
  const o = [];
  for (let iy = 0; iy < Ny - 1; iy++) for (let ix = 0; ix < Nx; ix++) {
    const nx = (ix + 1) % Nx;
    const a = iy * Nx + ix, b = iy * Nx + nx;
    const c = (iy + 1) * Nx + nx, d = (iy + 1) * Nx + ix;
    o.push(a, b, c, a, c, d);
  }
  return new Uint32Array(o);
}

function buildEdges(Nx, Ny) {
  const seen = new Set(), o = [];
  const k = (a, b) => Math.min(a, b) * Nx * Ny + Math.max(a, b);
  for (let iy = 0; iy < Ny - 1; iy++) for (let ix = 0; ix < Nx; ix++) {
    const nx = (ix + 1) % Nx;
    const a = iy * Nx + ix, b = iy * Nx + nx;
    const c = (iy + 1) * Nx + nx, d = (iy + 1) * Nx + ix;
    for (const [x, y] of [[a,b],[b,c],[c,a],[a,c],[c,d],[d,a]]) {
      const key = k(x, y); if (seen.has(key)) continue; seen.add(key); o.push(x, y);
    }
  }
  return new Uint32Array(o);
}

export class PlanetMeshManager {
  constructor(device) {
    this.device = device;
    this._idxGrid = null;

    this.uniformBuffer = device.createBuffer({
      label: 'planet-grid-uniform', size: GRID_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._cpuUni = new Float32Array(GRID_FLOATS);

    const MC = MAX_NX * MAX_NY;
    this.vertexBuffer = device.createBuffer({
      label: 'planet-vertex-buffer', size: MC * STRIDE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });

    this._triBuf = null;
    this._edgeBuf = null;
  }

  resolveGrid() {
    const ny = Math.max(3, Math.min(MAX_NY, PLANET_MESH.ny));
    return { nx: ny, ny };
  }

  indexCount(g) { return g.nx * (g.ny - 1) * 6; }
  edgeCount(g) { return buildEdges(g.nx, g.ny).length; }
  gridChanged(g) { return !this._idxGrid || g.nx !== this._idxGrid.nx || g.ny !== this._idxGrid.ny; }

  ensureIndexBuffer(g) {
    if (!this.gridChanged(g)) return this._triBuf;
    this._triBuf?.destroy(); this._edgeBuf?.destroy();
    const t = buildIndices(g.nx, g.ny), e = buildEdges(g.nx, g.ny);
    this._triBuf = this.device.createBuffer({
      label: `planet-tris-${g.nx}x${g.ny}`, size: t.byteLength,
      usage: GPUBufferUsage.INDEX, mappedAtCreation: true,
    });
    new Uint32Array(this._triBuf.getMappedRange()).set(t); this._triBuf.unmap();
    this._edgeBuf = this.device.createBuffer({
      label: `planet-edges-${g.nx}x${g.ny}`, size: e.byteLength,
      usage: GPUBufferUsage.INDEX, mappedAtCreation: true,
    });
    new Uint32Array(this._edgeBuf.getMappedRange()).set(e); this._edgeBuf.unmap();
    this._idxGrid = g;
    return this._triBuf;
  }

  ensureEdgeBuffer(g) { this.ensureIndexBuffer(g); return this._edgeBuf; }

  writeUniform(camera, grid) {
    const c = camera.current.pos, u = this._cpuUni;
    u[0] = c[0]; u[1] = c[1]; u[2] = c[2]; u[3] = 0;
    u[4] = grid.nx; u[5] = grid.ny; u[6] = 1; u[7] = 0;
    u[8] = PLANET_MESH.rMin; u[9] = PLANET_MESH.rMax;
    u[10] = PLANET_MESH.depthExponent; u[11] = PLANET_MESH.tanWarp;
    u[12] = PLANET_MESH.maxTerrainAmp; u[13] = 0; u[14] = 0; u[15] = 0;
    u[16] = 0; u[17] = 0; u[18] = 0; u[19] = 0;
    u[20] = 1 / grid.nx; u[21] = 1 / (grid.ny - 1); u[22] = 0; u[23] = 0;
    u[24] = BODY.R; u[25] = 0; u[26] = 0; u[27] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
  }

  mesherBGL(device) { return device.createBindGroupLayout({ label: 'planet-mesher-bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: GRID_BYTES } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]}); }

  buildMesherBG(device, layout) { return device.createBindGroup({ label: 'planet-mesher-bg', layout, entries: [
    { binding: 0, resource: { buffer: this.uniformBuffer } },
    { binding: 1, resource: { buffer: this.vertexBuffer } },
  ]}); }

  drawBGL(device) { return device.createBindGroupLayout({ label: 'planet-draw-bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: 'uniform', minBindingSize: 64 } },
  ]}); }

  buildDrawBG(device, layout, uniformBuf) { return device.createBindGroup({
    label: 'planet-draw-bg', layout, entries: [
      { binding: 0, resource: { buffer: this.vertexBuffer } },
      { binding: 1, resource: { buffer: uniformBuf } },
    ],
  }); }

  destroy() {
    this.uniformBuffer?.destroy();
    this.vertexBuffer?.destroy();
    this._triBuf?.destroy();
    this._edgeBuf?.destroy();
  }
}
