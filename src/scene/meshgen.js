/**
 * Generated triangle meshes, on the CPU, in each object's own local space.
 *
 * Everything solid in this scene has so far been synthesised in a vertex shader from
 * `vertex_index` — no buffers, no indices, the topology implied by arithmetic. That is compact and it
 * was right while there was one such shape, but it does not generalise: every new shape means new
 * index arithmetic in a new shader, the topology cannot be inspected or tested, and nothing can share
 * a pipeline. This module is the other half of the trade — the shapes become data.
 *
 * THREE PROPERTIES THE DESIGN DEPENDS ON, all of them load-bearing:
 *
 *   LOCAL SPACE, not world. A ring's radius and cross-section depend only on its index; only its
 *   basis precesses (see `ringDefAt`). So the geometry is RIGID and its animation is a rotation,
 *   which means the mesh can be baked once and transformed per frame — and, critically, that the
 *   previous frame's position is the same vertex under the previous frame's basis. Exact motion
 *   vectors for free, which is what admits this geometry to the accumulation buffer at all.
 *
 *   PER-VERTEX OBJECT ID. Several generated meshes are concatenated into one buffer, each vertex
 *   carrying the index of the object it belongs to, so one draw call covers all of them and the
 *   vertex shader fetches each object's transform. Baked positions cannot be instanced when the
 *   objects differ in size, and three draw calls to say one thing is worse than one attribute.
 *
 *   HARD EDGES BY DUPLICATION. A rectangular tube has four flat faces meeting at four creases. Shared
 *   vertices would average the normals across a crease and round it off, so each face gets its own
 *   vertices. That is why the vertex count is 4x what a watertight hull would need, and it is not
 *   waste — it is the shape.
 *
 * Pure functions over typed arrays, no GPU types, so the topology can be checked without a device.
 * See `dev/meshgen.mjs`, which validates it against the arithmetic it replaces.
 */

/**
 * Revolve a 2D profile around the local Z axis.
 *
 * The profile is a list of EDGES rather than points, each with its own normal, which is what makes
 * hard creases expressible: two edges meeting at a corner do not share vertices. Coordinates are
 * (radial, axial) offsets from the sweep circle, so `radius` places the circle and the profile rides
 * on it.
 *
 * @param {object} spec
 * @param {number} spec.segments  divisions around the sweep; the density knob
 * @param {number} spec.radius    sweep radius in local units
 * @param {{from:[number,number], to:[number,number], normal:[number,number]}[]} spec.profile
 * @returns {{positions: Float32Array, normals: Float32Array, indices: Uint32Array}}
 */
export function revolveProfile({ segments, radius, profile }) {
  if (!(segments >= 3)) throw new Error(`segments must be >= 3, got ${segments}`);
  if (!profile?.length) throw new Error('profile must have at least one edge');

  // Closed sweep: segment `segments` is segment 0, so the ring joins itself exactly rather than
  // leaving a seam whose two sides disagree by a rounding error.
  const rings = segments;                 // distinct major angles
  const perEdge = rings * 2;              // two vertices per edge per angle
  const vertCount = profile.length * perEdge;
  const triCount = profile.length * segments * 2;

  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const indices = new Uint32Array(triCount * 3);

  let vi = 0;
  let ii = 0;
  for (let e = 0; e < profile.length; e++) {
    const { from, to, normal } = profile[e];
    const base = vi;
    for (let s = 0; s < rings; s++) {
      const a = (s / segments) * Math.PI * 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      // The two profile ends at this angle. Radial offset scales the sweep circle; axial is along Z.
      for (const p of [from, to]) {
        const r = radius + p[0];
        positions[vi * 3] = ca * r;
        positions[vi * 3 + 1] = sa * r;
        positions[vi * 3 + 2] = p[1];
        // The normal's radial component rotates with the sweep; its axial component does not.
        normals[vi * 3] = ca * normal[0];
        normals[vi * 3 + 1] = sa * normal[0];
        normals[vi * 3 + 2] = normal[1];
        vi++;
      }
    }
    // Quads between consecutive angles, wrapping the last back to the first.
    for (let s = 0; s < segments; s++) {
      const a0 = base + s * 2;
      const a1 = base + ((s + 1) % rings) * 2;
      // Winding chosen so the face points along `normal`. The order matters and the obvious one is
      // wrong: sweeping anticlockwise, (a0, a0+1, a1+1) winds AGAINST the outward normal, which the
      // headless check caught on every one of 2304 triangles. Nothing culls today, so this would have
      // been invisible until the first pass that did.
      indices[ii++] = a0; indices[ii++] = a1 + 1; indices[ii++] = a0 + 1;
      indices[ii++] = a0; indices[ii++] = a1; indices[ii++] = a1 + 1;
    }
  }
  return { positions, normals, indices };
}

/**
 * A rectangular-section tube: the ring shape, as four flat faces.
 *
 * The profile reproduces the cross-section the ring's vertex shader described with its RC0/RC1/RN
 * tables — corners walked in order with an outward normal per face — so the generated mesh is the
 * same surface, not a similar one. `dev/meshgen.mjs` asserts that.
 */
export function rectTube({ segments, radius, halfW, halfH }) {
  const w = halfW;
  const h = halfH;
  const profile = [
    { from: [w, -h], to: [w, h], normal: [1, 0] },     // outer
    { from: [w, h], to: [-w, h], normal: [0, 1] },     // top
    { from: [-w, h], to: [-w, -h], normal: [-1, 0] },  // inner
    { from: [-w, -h], to: [w, -h], normal: [0, -1] },  // bottom
  ];
  return revolveProfile({ segments, radius, profile });
}

/**
 * Concatenate meshes into one buffer set, tagging each vertex with the index of the object it came
 * from. One draw call, one pipeline, N analytic transforms.
 *
 * @param {{positions: Float32Array, normals: Float32Array, indices: Uint32Array}[]} parts
 * @returns {{positions: Float32Array, normals: Float32Array, ids: Float32Array,
 *            indices: Uint32Array, vertexCount: number, indexCount: number}}
 */
export function concatMeshes(parts) {
  let nv = 0;
  let ni = 0;
  for (const p of parts) { nv += p.positions.length / 3; ni += p.indices.length; }

  const positions = new Float32Array(nv * 3);
  const normals = new Float32Array(nv * 3);
  // A float rather than a u32: it rides in the same interleaved vertex buffer as the rest, and
  // mixing integer and float attributes in one stride buys nothing at three objects.
  const ids = new Float32Array(nv);
  const indices = new Uint32Array(ni);

  let vOff = 0;
  let iOff = 0;
  for (let k = 0; k < parts.length; k++) {
    const p = parts[k];
    const count = p.positions.length / 3;
    positions.set(p.positions, vOff * 3);
    normals.set(p.normals, vOff * 3);
    ids.fill(k, vOff, vOff + count);
    // Indices are per-part, so they shift by the running vertex offset.
    for (let i = 0; i < p.indices.length; i++) indices[iOff + i] = p.indices[i] + vOff;
    vOff += count;
    iOff += p.indices.length;
  }
  return { positions, normals, ids, indices, vertexCount: nv, indexCount: ni };
}

/** Interleave into one buffer: position, normal, object id. 7 floats, 28 bytes. */
export function interleave({ positions, normals, ids, vertexCount }) {
  const STRIDE = 7;
  const out = new Float32Array(vertexCount * STRIDE);
  for (let v = 0; v < vertexCount; v++) {
    const o = v * STRIDE;
    out[o] = positions[v * 3];
    out[o + 1] = positions[v * 3 + 1];
    out[o + 2] = positions[v * 3 + 2];
    out[o + 3] = normals[v * 3];
    out[o + 4] = normals[v * 3 + 1];
    out[o + 5] = normals[v * 3 + 2];
    out[o + 6] = ids[v];
  }
  return out;
}

/** Floats per interleaved vertex, shared by the generator and the pipeline's vertex layout. */
export const MESH_STRIDE_FLOATS = 7;
