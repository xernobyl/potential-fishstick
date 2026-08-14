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

  // THE SEAM RING IS DUPLICATED, and this is not laziness about wrapping.
  //
  // Wrapping the last quad back to vertex 0 gives a watertight sweep with fewer vertices, and it is
  // wrong here: the shading reads `u` around the sweep, so the closing quad would interpolate from
  // (segments-1)/segments back DOWN to 0 instead of up to 1, mirroring the detail across one segment.
  // A hairline seam, in the one place a reader would assume is fine because the positions match.
  //
  // So there are `segments + 1` rings of vertices and `u` runs 0..1 continuously. The position uses
  // the angle modulo the sweep, so the first and last rings are BIT-IDENTICAL rather than merely
  // close - cos(TAU) and cos(0) are not guaranteed to agree, and a crack of one ulp is still a crack.
  const rings = segments + 1;             // distinct major angles, last coincident with the first
  const perEdge = rings * 2;              // two vertices per edge per angle
  const vertCount = profile.length * perEdge;
  const triCount = profile.length * segments * 2;

  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  // Per-vertex data whose meaning belongs to the SHAPE, not to the mesh system: (u along the sweep,
  // v across the profile edge, which edge, spare). Any future generator uses these four however it
  // likes, which is what keeps one vertex layout serving every generated mesh.
  const extra = new Float32Array(vertCount * 4);
  const indices = new Uint32Array(triCount * 3);

  let vi = 0;
  let ii = 0;
  for (let e = 0; e < profile.length; e++) {
    const { from, to, normal } = profile[e];
    const base = vi;
    for (let s = 0; s < rings; s++) {
      // Position from the angle MODULO the sweep, so ring `segments` is bit-identical to ring 0.
      const a = ((s % segments) / segments) * Math.PI * 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      const u = s / segments;             // but `u` runs past it, to 1.0
      // The two profile ends at this angle. Radial offset scales the sweep circle; axial is along Z.
      for (let k = 0; k < 2; k++) {
        const p = k === 0 ? from : to;
        const r = radius + p[0];
        positions[vi * 3] = ca * r;
        positions[vi * 3 + 1] = sa * r;
        positions[vi * 3 + 2] = p[1];
        // The normal's radial component rotates with the sweep; its axial component does not.
        normals[vi * 3] = ca * normal[0];
        normals[vi * 3 + 1] = sa * normal[0];
        normals[vi * 3 + 2] = normal[1];
        extra[vi * 4] = u;
        extra[vi * 4 + 1] = k;            // 0 at `from`, 1 at `to`
        extra[vi * 4 + 2] = e;            // which profile edge, for per-face shading
        vi++;
      }
    }
    // Quads between consecutive angles, wrapping the last back to the first.
    for (let s = 0; s < segments; s++) {
      const a0 = base + s * 2;
      // No modulo: the seam ring exists, so the last quad reaches it rather than folding back.
      const a1 = base + (s + 1) * 2;
      // Winding chosen so the face points along `normal`. The order matters and the obvious one is
      // wrong: sweeping anticlockwise, (a0, a0+1, a1+1) winds AGAINST the outward normal, which the
      // headless check caught on every one of 2304 triangles. Nothing culls today, so this would have
      // been invisible until the first pass that did.
      indices[ii++] = a0; indices[ii++] = a1 + 1; indices[ii++] = a0 + 1;
      indices[ii++] = a0; indices[ii++] = a1; indices[ii++] = a1 + 1;
    }
  }
  return { positions, normals, extra, indices };
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
 * A closed box with FLAT faces: 24 vertices, 12 triangles.
 *
 * TWENTY-FOUR VERTICES, NOT EIGHT, and that is the whole reason this is a generator rather than a
 * literal. A cube corner has three different normals and a vertex carries one; sharing the eight corners
 * would average the three into a diagonal and shade the box like a rounded blob. Hard-surface geometry
 * loses its edges exactly here, and it loses them silently — the silhouette stays right, so it reads as
 * "the lighting looks off" rather than as a geometry bug.
 *
 * UNIT BY DEFAULT so one buffer can serve differently-shaped boxes: an instanced draw scales the unit
 * cube per instance, which is what the satellites do. Because both the box and the scale are
 * axis-aligned, a face normal along an axis stays along that axis under the scale, so the instance
 * scaling needs no inverse-transpose — the one case where the shortcut is exact rather than close.
 *
 * @param {number[]} [halfExtents] half-size on each axis
 */
export function box(halfExtents = [1, 1, 1]) {
  const positions = new Float32Array(24 * 3);
  const normals = new Float32Array(24 * 3);
  const extra = new Float32Array(24 * 4);
  const indices = new Uint32Array(36);

  let v = 0;
  let i = 0;
  for (let axis = 0; axis < 3; axis++) {
    for (const sign of [1, -1]) {
      // (u, v, n) right-handed, so the quad below comes out counter-clockwise seen from OUTSIDE —
      // which is what `frontFace: 'ccw'` plus back-face culling requires. Swapping u and v for the
      // negative face is what keeps the handedness rather than mirroring it.
      const n = [0, 0, 0];
      n[axis] = sign;
      const uAxis = sign > 0 ? (axis + 1) % 3 : (axis + 2) % 3;
      const vAxis = sign > 0 ? (axis + 2) % 3 : (axis + 1) % 3;

      const base = v;
      for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const p = [0, 0, 0];
        p[axis] = sign * halfExtents[axis];
        p[uAxis] = su * halfExtents[uAxis];
        p[vAxis] = sv * halfExtents[vAxis];
        positions.set(p, v * 3);
        normals.set(n, v * 3);
        v++;
      }
      indices.set([base, base + 1, base + 2, base, base + 2, base + 3], i);
      i += 6;
    }
  }
  return { positions, normals, extra, indices, vertexCount: 24, triangleCount: 12 };
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
  const extra = new Float32Array(nv * 4);
  // A float rather than a u32: it rides in the same interleaved vertex buffer as the rest, and
  // mixing integer and float attributes in one stride buys nothing at three objects.
  const ids = new Float32Array(nv);
  const indices = new Uint32Array(ni);

  // PER-OBJECT INDEX RANGES and an object-space bounding sphere each.
  //
  // One buffer and one draw is right while everything in it is always visible; the moment an object can
  // be culled, or wants a different level of detail, it needs to be drawable on its own. Recording the
  // ranges at build time costs nothing and is what makes both possible — see core/frustum.js.
  const ranges = [];

  let vOff = 0;
  let iOff = 0;
  for (let k = 0; k < parts.length; k++) {
    const p = parts[k];
    const count = p.positions.length / 3;
    positions.set(p.positions, vOff * 3);
    normals.set(p.normals, vOff * 3);
    extra.set(p.extra, vOff * 4);
    ids.fill(k, vOff, vOff + count);
    // Indices are per-part, so they shift by the running vertex offset.
    for (let i = 0; i < p.indices.length; i++) indices[iOff + i] = p.indices[i] + vOff;
    // Bounding sphere in OBJECT space: centroid of the extremes, then the true farthest vertex from it.
    // A sphere rather than a box because everything cullable here rotates, and a sphere is
    // rotation-invariant — so this radius stays correct for every orientation the object will ever have.
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < count; v++) {
      for (let a = 0; a < 3; a++) {
        const val = p.positions[v * 3 + a];
        if (val < lo[a]) lo[a] = val;
        if (val > hi[a]) hi[a] = val;
      }
    }
    const centre = [0, 1, 2].map((a) => (lo[a] + hi[a]) * 0.5);
    let r2 = 0;
    for (let v = 0; v < count; v++) {
      const dx = p.positions[v * 3] - centre[0];
      const dy = p.positions[v * 3 + 1] - centre[1];
      const dz = p.positions[v * 3 + 2] - centre[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) r2 = d2;
    }
    ranges.push({ id: k, start: iOff, count: p.indices.length, centre, radius: Math.sqrt(r2) });

    vOff += count;
    iOff += p.indices.length;
  }
  return { positions, normals, extra, ids, indices, ranges, vertexCount: nv, indexCount: ni };
}

/**
 * THE VERTEX FORMAT, declared once.
 *
 * Everything downstream is derived from this table: the interleaving below, the GPU vertex layout in
 * `core/mesh.js`, and by correspondence the `MeshVertex` struct in `mesh_vertex.wgsl`. The offsets
 * used to be written out by hand alongside a separately-declared stride, which is the same shape of
 * hazard as a uniform block with hand-numbered slots - add a field and the stride moves while the
 * offsets do not, and the result is not an error but a garbled mesh.
 *
 * `extra` is what keeps ONE format serving every generated shape: four floats whose meaning belongs
 * to the generator. Adding a field here is a deliberate act that changes every mesh's memory; giving
 * `extra` a new meaning is not.
 */
export const MESH_FIELDS = (() => {
  const fields = [
    { name: 'position', key: 'positions', size: 3 },
    { name: 'normal', key: 'normals', size: 3 },
    { name: 'extra', key: 'extra', size: 4 },
    { name: 'objectId', key: 'ids', size: 1 },
  ];
  let offset = 0;
  for (const f of fields) { f.offset = offset; offset += f.size; }
  return fields;
})();

/** Floats per interleaved vertex. Derived, so it cannot drift from the field table. */
export const MESH_STRIDE_FLOATS = MESH_FIELDS.reduce((n, f) => n + f.size, 0);

/**
 * Interleave into one buffer, driven by the field table.
 *
 * A generic loop rather than an unrolled one: this runs once per mesh at build time, never per frame,
 * so the only thing worth optimising for is that it cannot disagree with the layout.
 */
export function interleave(data) {
  const out = new Float32Array(data.vertexCount * MESH_STRIDE_FLOATS);
  for (const f of MESH_FIELDS) {
    const src = data[f.key];
    if (!src) throw new Error(`mesh data is missing ${f.key}`);
    for (let v = 0; v < data.vertexCount; v++) {
      const o = v * MESH_STRIDE_FLOATS + f.offset;
      for (let k = 0; k < f.size; k++) out[o + k] = src[v * f.size + k];
    }
  }
  return out;
}

/**
 * Split shared vertices where faces meet at a hard angle, so creases shade FLAT.
 *
 * Dual contouring shares one vertex across a crease, so the single normal it carries smooths the
 * lighting across the edge and the crease reads rounded — the geometry is sharp, the shading is not.
 * This is the "hard edges by duplication" trick `box` and `revolveProfile` do by hand, generalised to
 * any mesh: group each vertex's triangles by their geometric face normal, and where two faces meet at
 * more than `angleDeg`, duplicate the vertex so each face keeps its own normal.
 *
 * Vertices whose triangles all face within `angleDeg` of each other are left untouched, so smooth
 * blends keep their smooth normals — only genuine creases split.
 *
 * Zero-area triangles (which dual contouring keeps for watertightness) carry no orientation, so they
 * join the first group at their vertex rather than forcing a split.
 *
 * Positions are copied verbatim (only normals change and vertices duplicate), so bounding spheres stay
 * correct and the weave stays watertight.
 *
 * @param {{positions:Float32Array, normals:Float32Array, indices:Uint32Array, vertexCount?:number}} mesh
 * @param {number} [angleDeg]  crease angle; faces meeting steeper than this split
 * @returns {{positions:Float32Array, normals:Float32Array, indices:Uint32Array,
 *            vertexCount:number, triangleCount:number}}
 */
export function splitHardEdges(mesh, angleDeg = 40) {
  const { positions, normals, indices } = mesh;
  const vertexCount = mesh.vertexCount ?? positions.length / 3;
  const triCount = indices.length / 3;
  const cosT = Math.cos(angleDeg * Math.PI / 180);

  // Geometric face normal per triangle; (0,0,0) for zero-area triangles.
  const faceN = new Float32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const ux = positions[b * 3] - positions[a * 3];
    const uy = positions[b * 3 + 1] - positions[a * 3 + 1];
    const uz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const vx = positions[c * 3] - positions[a * 3];
    const vy = positions[c * 3 + 1] - positions[a * 3 + 1];
    const vz = positions[c * 3 + 2] - positions[a * 3 + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz);
    if (l > 1e-12) { faceN[t * 3] = nx / l; faceN[t * 3 + 1] = ny / l; faceN[t * 3 + 2] = nz / l; }
  }

  // Which triangles use each vertex.
  const vertTris = Array.from({ length: vertexCount }, () => []);
  for (let t = 0; t < triCount; t++) {
    vertTris[indices[t * 3]].push(t);
    vertTris[indices[t * 3 + 1]].push(t);
    vertTris[indices[t * 3 + 2]].push(t);
  }

  const newPos = [];
  const newNor = [];
  const cornerMap = new Int32Array(triCount * 3);   // each triangle corner -> new vertex index

  const assignCorner = (t, corner, ni) => { cornerMap[t * 3 + corner] = ni; };

  for (let v = 0; v < vertexCount; v++) {
    const tris = vertTris[v];
    // Greedy groups of NON-degenerate triangles, by face-normal angle.
    const groups = [];                 // { tris:[], nx, ny, nz }
    const degenerate = [];
    for (const t of tris) {
      const fx = faceN[t * 3], fy = faceN[t * 3 + 1], fz = faceN[t * 3 + 2];
      if (fx === 0 && fy === 0 && fz === 0) { degenerate.push(t); continue; }
      let placed = false;
      for (const g of groups) {
        if (fx * g.nx + fy * g.ny + fz * g.nz >= cosT) {
          g.tris.push(t); g.nx += fx; g.ny += fy; g.nz += fz; placed = true; break;
        }
      }
      if (!placed) groups.push({ tris: [t], nx: fx, ny: fy, nz: fz });
    }

    if (groups.length <= 1) {
      // Smooth (or all-degenerate): keep the original vertex and its normal.
      const ni = newPos.length / 3;
      newPos.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
      newNor.push(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]);
      for (const t of tris) {
        if (indices[t * 3] === v) assignCorner(t, 0, ni);
        if (indices[t * 3 + 1] === v) assignCorner(t, 1, ni);
        if (indices[t * 3 + 2] === v) assignCorner(t, 2, ni);
      }
    } else {
      // Hard edge: one copy per group, each shaded by its group's face-normal average.
      const groupVertex = [];
      for (const g of groups) {
        const l = Math.hypot(g.nx, g.ny, g.nz) || 1;
        const ni = newPos.length / 3;
        newPos.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
        newNor.push(g.nx / l, g.ny / l, g.nz / l);
        groupVertex.push(ni);
        for (const t of g.tris) {
          if (indices[t * 3] === v) assignCorner(t, 0, ni);
          if (indices[t * 3 + 1] === v) assignCorner(t, 1, ni);
          if (indices[t * 3 + 2] === v) assignCorner(t, 2, ni);
        }
      }
      // Zero-area triangles carry no orientation; they join the first group.
      for (const t of degenerate) {
        if (indices[t * 3] === v) assignCorner(t, 0, groupVertex[0]);
        if (indices[t * 3 + 1] === v) assignCorner(t, 1, groupVertex[0]);
        if (indices[t * 3 + 2] === v) assignCorner(t, 2, groupVertex[0]);
      }
    }
  }

  const newIndices = new Uint32Array(indices.length);
  for (let t = 0; t < triCount; t++) {
    newIndices[t * 3] = cornerMap[t * 3];
    newIndices[t * 3 + 1] = cornerMap[t * 3 + 1];
    newIndices[t * 3 + 2] = cornerMap[t * 3 + 2];
  }

  return {
    positions: new Float32Array(newPos),
    normals: new Float32Array(newNor),
    indices: newIndices,
    vertexCount: newPos.length / 3,
    triangleCount: triCount,
  };
}
