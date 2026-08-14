/**
 * SurfaceNetsManager — GPU Surface Nets / Dual Contouring isosurface extraction.
 *
 * 3-pass cached-corner pipeline:
 *   0. clear_buffers  — reset indirect draw args + cell vertex IDs
 *   1. eval_corners   — one SDF sample per grid corner → cornerBuffer
 *   2. compute_vertices — one vertex per surface cell → vertexBuffer (sparse)
 *   3. generate_faces — quads as triangle pairs → indexBuffer
 *
 * Vertex placement toggled via SURFACE_NETS.useDC:
 *   false = Surface Nets (centroid of edge crossings, fast, smooth fields)
 *   true  = Dual Contouring (QEF minimise over crossing planes, preserves edges)
 *
 * Grid resolution:
 *   lateralRes = cells across the shorter screen axis (height). Width
 *                auto-adjusts to match the display aspect ratio, so cells
 *                project to roughly square pixels.
 *   depthRes   = cells along the view depth axis. Independent of lateralRes
 *                so you can trade depth detail against XY detail.
 *
 * All buffers allocated once at init or on resolution change. Zero render-loop
 * allocations — only writeBuffer for uniforms and dispatches.
 */

import { BODY, SN_NORMAL } from '../scene/tuning.js';

/** Live tuning knobs. Read fresh each frame — sliders just mutate these. */
export const SURFACE_NETS = {
  /** Cells across the shorter screen axis (height). Width = lateralRes * aspect. */
  lateralRes: 48,
  /** Cells along the depth axis. */
  depthRes: 32,
  /** false = Surface Nets (centroid), true = Dual Contouring (QEF solve). */
  useDC: false,
  /** 'world' = axis-aligned cube around body, 'frustum' = unproject from NDC. */
  gridMode: 'world',
  /** Frustum mode: world-space depth range the grid covers in front of the camera. */
  nearDist: 0.5,
  farDist: 10.0,
  /** World-space half-extent of the grid cube, as a multiple of body radius. */
  extentMul: 1.5,
  /** Margin in cells so the surface is never clipped by the grid boundary. */
  pad: 2,
};

// ---- buffer sizing ---------------------------------------------------------

const GRID_UNIFORM_FLOATS = 40;   // GridUniform: 2×vec4f + 3×f32 + u32 + 2×vec4u + u32 + 2×f32 + mat4x4f = 160 bytes
const GRID_UNIFORM_BYTES = GRID_UNIFORM_FLOATS * 4;
const INDIRECT_BYTES = 24;        // 5 draw args + vertexCount
const WIRE_INDIRECT_BYTES = 20;  // 5 draw args (no atomics needed)
// pos.xyz + normal, in the build's chosen format. Must agree with the SN_VERTEX_STRIDE that
// wgslDefines() injects into planet_surfacenets.wgsl and planet_raster.wgsl — see SN_NORMAL in
// tuning.js. 4 floats = position + one packed Fibonacci index, 6 = position + float32x3 normal.
const VERTEX_STRIDE_FLOATS = SN_NORMAL.bits > 0 ? 4 : 6;
const VERTEX_STRIDE_BYTES = VERTEX_STRIDE_FLOATS * 4;

function maxCells(nx, ny, nz) { return nx * ny * nz; }
function maxCorners(nx, ny, nz) { return (nx + 1) * (ny + 1) * (nz + 1); }
function maxIndices(nx, ny, nz) {
  return maxCells(nx, ny, nz) * 3 * 6;  // up to 3 quads per cell
}

function ceilDiv(n, d) { return Math.ceil(n / d); }

export class SurfaceNetsManager {
  constructor(device) {
    this.device = device;
    this._res = null;
    this._aspect = 1.0;
    this._gen = 0;       // bumped on buffer resize; both BG caches check it

    // ---- uniform buffer (re-uploaded every frame) ----
    this.gridUniform = device.createBuffer({
      label: 'sn-grid-uniform',
      size: GRID_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Dual-view buffer for writing mixed f32/u32 fields.
    this._cpuBuf = new ArrayBuffer(GRID_UNIFORM_BYTES);
    this._cpuF32 = new Float32Array(this._cpuBuf);
    this._cpuU32 = new Uint32Array(this._cpuBuf);

    // ---- indirect draw args + vertex counter ----
    this.indirectBuffer = device.createBuffer({
      label: 'sn-indirect',
      size: INDIRECT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });

    // Large buffers allocated lazily on first resolve
    this.cornerBuffer = null;
    this.cellVertBuffer = null;
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.wireIndexBuffer = null;
    this.wireIndirectBuf = null;
  }

  // ---- resolution management ----------------------------------------------

  /**
   * Current grid resolution, with aspect-ratio adjustment.
   * `aspect` = display width / display height.
   *   ny = lateralRes  (shorter axis)
   *   nx = lateralRes * aspect  (wider axis, rounded)
   *   nz = depthRes
   */
  resolve(aspect = this._aspect) {
    this._aspect = aspect;
    const ny = Math.max(8, Math.min(128, SURFACE_NETS.lateralRes));
    const nx = Math.max(8, Math.min(256, Math.round(ny * aspect)));
    const nz = Math.max(4, Math.min(128, SURFACE_NETS.depthRes));
    return { nx, ny, nz, total: nx * ny * nz };
  }

  /** True when the resolution has changed since the last ensureBuffers. */
  changed(res) {
    return !this._res || res.nx !== this._res.nx || res.ny !== this._res.ny || res.nz !== this._res.nz;
  }

  /**
   * (Re)allocate storage buffers when the resolution changes.
   * Idempotent — call every frame, it only does work on a change.
   */
  ensureBuffers(res) {
    if (!this.changed(res)) return;
    this._res = { nx: res.nx, ny: res.ny, nz: res.nz, total: res.total };

    const d = this.device;
    const nc = maxCells(res.nx, res.ny, res.nz);
    const nco = maxCorners(res.nx, res.ny, res.nz);
    const ni = maxIndices(res.nx, res.ny, res.nz);

    this.cornerBuffer?.destroy();
    this.cellVertBuffer?.destroy();
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.wireIndexBuffer?.destroy();
    this.wireIndirectBuf?.destroy();

    this.cornerBuffer = d.createBuffer({
      label: `sn-corners-${res.nx}x${res.ny}x${res.nz}`,
      size: nco * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.cellVertBuffer = d.createBuffer({
      label: `sn-cellverts-${res.nx}x${res.ny}x${res.nz}`,
      size: nc * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.vertexBuffer = d.createBuffer({
      label: `sn-vertices-${res.nx}x${res.ny}x${res.nz}`,
      size: nc * VERTEX_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });
    this.indexBuffer = d.createBuffer({
      label: `sn-indices-${res.nx}x${res.ny}x${res.nz}`,
      size: Math.max(ni * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
    });

    this.wireIndexBuffer = d.createBuffer({
      label: `sn-wire-${res.nx}x${res.ny}x${res.nz}`,
      size: Math.max(ni * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
    });
    this.wireIndirectBuf = d.createBuffer({
      label: 'sn-wire-indirect',
      size: WIRE_INDIRECT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });

    this._gen++;  // bump generation so both BG caches invalidate
  }

  // ---- per-frame uniform upload -------------------------------------------

  /**
   * Position the grid to cover the body as seen from the camera.
   *
   * The grid is a world-space cube centered on the body origin. Cell sizes
   * may differ per axis (lateral vs depth), but are uniform within each axis.
   */
  writeUniform(camPos, invViewProj) {
    const R = BODY.R;
    const extent = R * SURFACE_NETS.extentMul;
    const res = this.resolve();

    const worldExtent = extent;
    const cellX = (2 * worldExtent) / res.nx;
    const cellY = (2 * worldExtent) / res.ny;
    const cellZ = (2 * worldExtent) / res.nz;

    const frustum = SURFACE_NETS.gridMode === 'frustum';
    const NEAR = 0.05;
    const zNearNDC = NEAR / Math.max(SURFACE_NETS.nearDist, NEAR);
    const zFarNDC  = NEAR / Math.max(SURFACE_NETS.farDist, NEAR);

    const uf = this._cpuF32;
    const ui = this._cpuU32;

    uf[0] = -worldExtent; uf[1] = -worldExtent; uf[2] = -worldExtent; uf[3] = 0;
    uf[4] = camPos[0]; uf[5] = camPos[1]; uf[6] = camPos[2]; uf[7] = 0;
    uf[8] = cellX; uf[9] = cellY; uf[10] = cellZ;
    ui[11] = SURFACE_NETS.useDC ? 1 : 0;
    ui[12] = res.nx; ui[13] = res.ny; ui[14] = res.nz; ui[15] = 0;
    ui[16] = res.nx + 1; ui[17] = res.ny + 1; ui[18] = res.nz + 1; ui[19] = 0;
    ui[20] = frustum ? 1 : 0;
    uf[21] = zNearNDC; uf[22] = zFarNDC;
    // invViewProj at uf[24..39]
    if (invViewProj) { uf.set(invViewProj, 24); } else { uf.fill(0, 24, 40); }

    this.device.queue.writeBuffer(this.gridUniform, 0, this._cpuF32);
  }

  // ---- bind group layouts -------------------------------------------------

  mesherBGL(device) {
    return device.createBindGroupLayout({
      label: 'sn-mesher-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', minBindingSize: GRID_UNIFORM_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
      ],
    });
  }

  buildMesherBG(device, layout) {
    if (this._mesherBG && this._mesherGen === this._gen) return this._mesherBG;
    this._mesherBG = device.createBindGroup({
      label: 'sn-mesher-bg',
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.gridUniform } },
        { binding: 1, resource: { buffer: this.cornerBuffer } },
        { binding: 2, resource: { buffer: this.cellVertBuffer } },
        { binding: 3, resource: { buffer: this.vertexBuffer } },
        { binding: 4, resource: { buffer: this.indexBuffer } },
        { binding: 5, resource: { buffer: this.indirectBuffer } },
        { binding: 6, resource: { buffer: this.wireIndexBuffer } },
        { binding: 7, resource: { buffer: this.wireIndirectBuf } },
      ],
    });
    this._mesherGen = this._gen;
    return this._mesherBG;
  }

  // ---- draw bind group layout ---------------------------------------------

  drawBGL(device) {
    return device.createBindGroupLayout({
      label: 'sn-draw-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', minBindingSize: 64 } },
      ],
    });
  }

  buildDrawBG(device, layout, drawUniformBuf) {
    if (this._drawBG && this._drawGen === this._gen) return this._drawBG;
    this._drawBG = device.createBindGroup({
      label: 'sn-draw-bg',
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.vertexBuffer } },
        { binding: 1, resource: { buffer: drawUniformBuf } },
      ],
    });
    this._drawGen = this._gen;
    return this._drawBG;
  }

  // ---- dispatch counts ----------------------------------------------------

  clearDispatch(res) {
    return { x: ceilDiv(Math.max(res.total, 1), 64), y: 1, z: 1 };
  }

  cornerDispatch(res) {
    return {
      x: ceilDiv(res.nx + 1, 4),
      y: ceilDiv(res.ny + 1, 4),
      z: ceilDiv(res.nz + 1, 4),
    };
  }

  cellDispatch(res) {
    return {
      x: ceilDiv(res.nx, 4),
      y: ceilDiv(res.ny, 4),
      z: ceilDiv(res.nz, 4),
    };
  }

  /** Wireframe dispatch: ceil(maxIndices ÷ 3 ÷ 64) = ceil(triangles / 64). */
  wireDispatch(res) {
    const maxTri = maxIndices(res.nx, res.ny, res.nz) / 3;
    return { x: ceilDiv(Math.max(maxTri, 1), 64), y: 1, z: 1 };
  }

  // ---- teardown -----------------------------------------------------------

  destroy() {
    this.gridUniform?.destroy();
    this.indirectBuffer?.destroy();
    this.cornerBuffer?.destroy();
    this.cellVertBuffer?.destroy();
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.wireIndexBuffer?.destroy();
    this.wireIndirectBuf?.destroy();
  }
}
