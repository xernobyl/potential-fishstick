/**
 * A generated mesh on the GPU: one interleaved vertex buffer, one index buffer, one layout.
 *
 * Deliberately thin. It owns buffers and nothing else — no transform, no material, no draw order —
 * because the transform is analytic and lives in the vertex shader, the material is the pass's, and
 * the order is the frame graph's. A mesh here is a shape, not an object.
 *
 * ONE LAYOUT FOR EVERY SHAPE, which is the property worth protecting. Position and normal are what a
 * mesh always has; the object id is what lets several shapes share a buffer and a draw call while
 * each fetches its own transform; and `extra` is four floats whose meaning belongs to the shape,
 * which is what stops the next generated mesh from needing its own layout, its own pipeline and its
 * own pass. See `meshgen.js` for the generation side and what it puts in `extra`.
 *
 * Indices are 32-bit. A rectangular tube at 128 segments is already 1032 vertices and future shapes
 * will be denser; the 16-bit ceiling is close enough that hitting it would be a silent corruption
 * rather than an error, and the bandwidth difference is nothing at this scale.
 */

import { interleave, MESH_FIELDS, MESH_STRIDE_FLOATS } from '../scene/meshgen.js';

const BYTES_PER_FLOAT = 4;
const STRIDE = MESH_STRIDE_FLOATS * BYTES_PER_FLOAT;
/** WebGPU's name for a run of N floats. */
const FORMAT = ['', 'float32', 'float32x2', 'float32x3', 'float32x4'];

/**
 * The vertex layout every generated mesh uses. Passed straight to a pipeline's `vertex.buffers`.
 *
 * Locations match the `MeshVertex` struct in `mesh_vertex.wgsl` — the two have to agree, and the
 * shader-side struct names each field so a mismatch is a compile error rather than a garbled shape.
 */
export const MESH_VERTEX_LAYOUT = {
  arrayStride: STRIDE,
  // DERIVED from the field table rather than written out: the offsets and the stride have to agree,
  // and hand-numbering them is how they stop agreeing. Shader locations are the table's order, which
  // is the order `MeshVertex` declares in mesh_vertex.wgsl.
  attributes: MESH_FIELDS.map((f, i) => ({
    shaderLocation: i,
    offset: f.offset * BYTES_PER_FLOAT,
    format: FORMAT[f.size],
  })),
};

export class Mesh {
  /**
   * @param {GPUDevice} device
   * @param {object} data   the output of `concatMeshes`, or of one generator plus ids
   * @param {string} label
   */
  constructor(device, data, label) {
    const verts = interleave(data);
    this.label = label;
    this.indexCount = data.indices.length;
    // One entry per object in the buffer, each with its index range and object-space bounding sphere.
    // A single-object mesh gets one covering the whole buffer, so consumers never special-case.
    this.ranges = data.ranges ?? [{
      id: 0, start: 0, count: data.indices.length,
      centre: [0, 0, 0], radius: Infinity,
    }];
    this.vertexCount = data.vertexCount ?? data.positions.length / 3;
    // Kept on the CPU for the wireframe view, which needs to walk them to find the edges.
    this.indices = data.indices;

    // Written once at build time, never again: the geometry is static in object space and its
    // animation is a transform. `mappedAtCreation` avoids a staging copy for a one-off upload.
    this.vertexBuffer = device.createBuffer({
      label: `${label}-verts`,
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(verts);
    this.vertexBuffer.unmap();

    this.indexBuffer = device.createBuffer({
      label: `${label}-indices`,
      size: data.indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(data.indices);
    this.indexBuffer.unmap();
  }

  /**
   * The same geometry as a LINE LIST, for the wireframe debug view. Built on first use and kept.
   *
   * Deduplicated: an interior edge is shared by two triangles, so the naive three-lines-per-triangle
   * expansion draws every interior edge twice. On a closed mesh that is very nearly 2x the lines for
   * exactly the same picture — and worse, double-drawn edges look brighter than boundary ones, which
   * would make a watertight mesh look like it had seams. Euler says a closed triangle mesh has 3F/2
   * edges, so the dedup is not a micro-optimisation; it is half.
   *
   * Lazy because this is a debug view: a session that never presses the key never pays for it.
   */
  lineIndices(device) {
    if (this.lineBuffer) return this.lineBuffer;
    const seen = new Set();
    const out = [];
    const idx = this.indices;
    for (let t = 0; t < idx.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = idx[t + e];
        const b = idx[t + (e + 1) % 3];
        // Order-independent key, so (a,b) and (b,a) are the same edge.
        const key = a < b ? a * this.vertexCount + b : b * this.vertexCount + a;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(a, b);
      }
    }
    this.lineCount = out.length;
    this.lineBuffer = device.createBuffer({
      label: `${this.label}-lines`,
      size: out.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.lineBuffer.getMappedRange()).set(out);
    this.lineBuffer.unmap();
    return this.lineBuffer;
  }

  /** Bind the vertex buffer and the LINE index buffer, then draw every edge once. */
  drawWireframe(pass, device, instances = 1) {
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.lineIndices(device), 'uint32');
    pass.drawIndexed(this.lineCount, instances);
  }

  /** Bind the buffers. Separate from drawing now that a caller may issue several ranges. */
  bind(pass) {
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
  }

  /** Bind and draw everything, optionally instanced. */
  draw(pass, instances = 1) {
    this.bind(pass);
    pass.drawIndexed(this.indexCount, instances);
  }

  /** One object's index range. Assumes `bind` has been called. */
  drawRange(pass, range, instances = 1) {
    pass.drawIndexed(range.count, instances, range.start);
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.lineBuffer?.destroy();
  }
}
