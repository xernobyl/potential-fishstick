// Does the GENERATED ring mesh describe the same surface as the vertex-shader arithmetic it replaces?
//
// The rings were synthesised from `vertex_index`: 6 vertices per quad, 4 faces per segment, the
// topology implied by integer division. Replacing that with a real indexed mesh is only safe if the
// two agree exactly — a port that is a little bit different is a look change nobody asked for, and one
// that is different only at the seam is a hairline crack that no image metric would attribute to this.
//
// So this reimplements the old shader's formula (rings.wgsl, RC0/RC1/RN and the QUAD table) and
// asserts the generated mesh hits the same set of positions and normals, plus the properties the old
// arithmetic got for free and a buffer could get wrong: unit normals, no degenerate triangles, a
// closed sweep with no seam, and consistent winding.

import { rectTube, revolveProfile, box, concatMeshes, interleave, MESH_FIELDS, MESH_STRIDE_FLOATS }
  from '../src/scene/meshgen.js';

const TAU = Math.PI * 2;
let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

// ---- the arithmetic being replaced, transcribed from rings.wgsl ----
const RC0 = [[1, -1], [1, 1], [-1, 1], [-1, -1]];
const RC1 = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
const RN = [[1, 0], [0, 1], [-1, 0], [0, -1]];

/** The old vertex shader's position/normal, in ring-local space (ax=X, ay=Y, az=Z). */
function oldVertex(seg, side, cx, cy, segments, radius, halfW, halfH) {
  const au = ((seg + cx) / segments) * TAU;
  const csx = (RC0[side][0] + (RC1[side][0] - RC0[side][0]) * cy) * halfW;
  const csy = (RC0[side][1] + (RC1[side][1] - RC0[side][1]) * cy) * halfH;
  const r = radius + csx;
  const n = RN[side];
  return {
    p: [Math.cos(au) * r, Math.sin(au) * r, csy],
    n: [Math.cos(au) * n[0], Math.sin(au) * n[0], n[1]],
  };
}

const SEGMENTS = 96;
const RADIUS = 3.4;
const HW = 0.42;
const HH = 0.06;
const mesh = rectTube({ segments: SEGMENTS, radius: RADIUS, halfW: HW, halfH: HH });

// ---- counts ----
const nv = mesh.positions.length / 3;
check(nv === 4 * (SEGMENTS + 1) * 2, 'vertex count is 4 faces x (segments+1) x 2, seam ring included',
      `${nv} vs ${4 * (SEGMENTS + 1) * 2}`);
check(mesh.indices.length === 4 * SEGMENTS * 2 * 3, 'index count is 4 faces x segments x 2 tris',
      `${mesh.indices.length}`);

// ---- the surface matches the old formula, as a SET ----
const key = (v) => v.map((x) => (Math.abs(x) < 1e-9 ? 0 : x).toFixed(6)).join(',');
const want = new Set();
for (let seg = 0; seg < SEGMENTS; seg++) {
  for (let side = 0; side < 4; side++) {
    for (const [cx, cy] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      const v = oldVertex(seg, side, cx, cy, SEGMENTS, RADIUS, HW, HH);
      want.add(`${key(v.p)}|${key(v.n)}`);
    }
  }
}
const got = new Set();
for (let v = 0; v < nv; v++) {
  got.add(`${key([mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]])}`
    + `|${key([mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]])}`);
}
let missing = 0;
for (const g of got) if (!want.has(g)) missing++;
check(missing === 0, 'every generated vertex lies on the old surface', `${missing} stray`);
check(got.size === want.size, 'and the two describe the same set of corners',
      `generated ${got.size}, expected ${want.size}`);

// ---- unit normals ----
let worstN = 0;
for (let v = 0; v < nv; v++) {
  const l = Math.hypot(mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]);
  worstN = Math.max(worstN, Math.abs(l - 1));
}
check(worstN < 1e-6, 'normals are unit length', `worst |len-1| ${worstN.toExponential(2)}`);

// ---- no degenerate triangles ----
let degenerate = 0;
let minArea = Infinity;
for (let t = 0; t < mesh.indices.length; t += 3) {
  const [a, b, c] = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
  const P = (i) => [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]];
  const [p0, p1, p2] = [P(a), P(b), P(c)];
  const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const w = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
  const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  const area = 0.5 * Math.hypot(cr[0], cr[1], cr[2]);
  if (!(area > 1e-12)) degenerate++;
  minArea = Math.min(minArea, area);
}
check(degenerate === 0, 'no degenerate triangles', `min area ${minArea.toExponential(2)}`);

// ---- the sweep closes: the last segment must reference the first ring's vertices ----
const maxIndex = mesh.indices.reduce((m, i) => Math.max(m, i), 0);
check(maxIndex === nv - 1, 'indices span exactly the vertices, so the sweep wraps rather than seams',
      `max ${maxIndex} of ${nv - 1}`);

// ---- winding faces the profile normal ----
let backwards = 0;
for (let t = 0; t < mesh.indices.length; t += 3) {
  const [a, b, c] = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
  const P = (i) => [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]];
  const [p0, p1, p2] = [P(a), P(b), P(c)];
  const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const w = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
  const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  const n = [mesh.normals[a * 3], mesh.normals[a * 3 + 1], mesh.normals[a * 3 + 2]];
  if (cr[0] * n[0] + cr[1] * n[1] + cr[2] * n[2] <= 0) backwards++;
}
check(backwards === 0, 'winding agrees with the profile normal on every triangle', `${backwards} flipped`);

// ---- the seam carries u = 1, not a wrap back to 0 ----
//
// This is what the duplicated seam ring buys, and getting it wrong would mirror the shading detail
// across one segment - a hairline artefact in the one place a reader assumes is safe because the
// positions match. Checked on positions AND on u.
{
  const perEdgeVerts = (SEGMENTS + 1) * 2;
  const first = 0;                       // edge 0, ring 0, `from`
  const seam = SEGMENTS * 2;             // edge 0, ring `segments`, `from`
  const same = [0, 1, 2].every((k) =>
    mesh.positions[first * 3 + k] === mesh.positions[seam * 3 + k]);
  check(same, 'the seam ring is BIT-identical in position to the first ring');
  check(mesh.extra[first * 4] === 0 && mesh.extra[seam * 4] === 1,
        'and u runs 0 -> 1 across the sweep rather than wrapping',
        `u ${mesh.extra[first * 4]} -> ${mesh.extra[seam * 4]}`);
  check(perEdgeVerts * 4 === nv, 'four profile edges, each with its own seam ring');
  // v and the edge index are what the fragment shader reads as `side`.
  check(mesh.extra[1] === 0 && mesh.extra[5] === 1, 'v is 0 at `from` and 1 at `to`');
  const edges = new Set();
  for (let v = 0; v < nv; v++) edges.add(mesh.extra[v * 4 + 2]);
  check(edges.size === 4 && [...edges].sort().join() === '0,1,2,3',
        'every vertex carries its profile edge index', `${[...edges].sort().join()}`);
}

// ---- concatenation tags and offsets ----
const three = concatMeshes([
  rectTube({ segments: 16, radius: 1, halfW: 0.2, halfH: 0.05 }),
  rectTube({ segments: 16, radius: 2, halfW: 0.2, halfH: 0.05 }),
  rectTube({ segments: 16, radius: 3, halfW: 0.2, halfH: 0.05 }),
]);
const perRing = 4 * (16 + 1) * 2;
check(three.vertexCount === perRing * 3, 'concat sums the vertices', `${three.vertexCount}`);
check(three.ids[0] === 0 && three.ids[perRing] === 1 && three.ids[perRing * 2] === 2,
      'each vertex carries its object index');
const maxI = three.indices.reduce((m, i) => Math.max(m, i), 0);
check(maxI === three.vertexCount - 1, 'concat rebases indices onto the merged buffer', `max ${maxI}`);
// A rebased index that forgot its offset would still be IN RANGE, so check the second object's
// triangles actually reference the second object's vertices.
// Indices per object, not vertices per object — the first version of this line used the vertex
// stride and landed in the THIRD object, which read as a failure in the code rather than in the test.
const indicesPerRing = 4 * 16 * 2 * 3;
const secondTri = three.indices.slice(indicesPerRing, indicesPerRing + 3);
check(secondTri.every((i) => i >= perRing && i < perRing * 2),
      'the second object\'s triangles reference the second object\'s vertices', `${secondTri}`);

// ---- per-object ranges and bounding spheres ----
{
  const r = three.ranges;
  check(r.length === 3, 'concat records one range per object', `${r.length}`);
  check(r[0].start === 0 && r[1].start === indicesPerRing && r[2].start === indicesPerRing * 2,
        'ranges start at the right index offsets', r.map((x) => x.start).join());
  check(r.every((x) => x.count === indicesPerRing), 'and cover their own indices');
  // A ring of radius 2 with a 0.2 half-width: the farthest vertex is at about 2.2 from the centre.
  check(Math.abs(r[1].radius - 2.2) < 0.05, 'the bounding sphere fits the object',
        `radius ${r[1].radius.toFixed(3)}`);
  check(r.every((x) => Math.hypot(...x.centre) < 1e-6),
        'and a hoop is centred on its own origin, so rotation cannot move it');
}

const inter = interleave(three);
check(inter.length === three.vertexCount * MESH_STRIDE_FLOATS,
      `interleave packs ${MESH_STRIDE_FLOATS} floats per vertex`);

// ---- the field table is the single source, so nothing may disagree with it ----
//
// The offsets and the stride used to be written out by hand in two files. They agreed, until they
// would not have: adding a field moves the stride and leaves the offsets, which garbles the mesh
// rather than erroring. These assertions are cheap and they are what makes the table authoritative.
{
  let running = 0;
  let ok = true;
  for (const f of MESH_FIELDS) { if (f.offset !== running) ok = false; running += f.size; }
  check(ok, 'field offsets are cumulative with no gaps or overlaps');
  check(running === MESH_STRIDE_FLOATS, 'the stride is the sum of the field sizes',
        `${running} vs ${MESH_STRIDE_FLOATS}`);
  // Every field must name a real array in the generator's output, or interleave silently writes zeros.
  const one = rectTube({ segments: 8, radius: 1, halfW: 0.1, halfH: 0.05 });
  const merged = concatMeshes([one]);
  const missing = MESH_FIELDS.filter((f) => !merged[f.key]);
  check(missing.length === 0, 'every field maps to an array the generator produces',
        missing.map((f) => f.key).join());
  // And the interleaved data must actually round-trip: read a vertex back through the offsets.
  const packed = interleave(merged);
  const v = 5;
  const pos = MESH_FIELDS[0];
  const same = [0, 1, 2].every((k) =>
    packed[v * MESH_STRIDE_FLOATS + pos.offset + k] === merged.positions[v * 3 + k]);
  check(same, 'a vertex read back through the table matches the source arrays');
}
check(inter[10] === 0 && inter[perRing * 11 + 10] === 1, 'the id survives interleaving');

// ---- density is a knob, and the profile primitive generalises ----
const dense = rectTube({ segments: 512, radius: 3.4, halfW: HW, halfH: HH });
check(dense.positions.length / 3 === 4 * (512 + 1) * 2, 'density scales with `segments`');
const tri = revolveProfile({
  segments: 8,
  radius: 1,
  profile: [
    { from: [0.1, -0.1], to: [0.1, 0.1], normal: [1, 0] },
    { from: [0.1, 0.1], to: [-0.1, 0], normal: [0.4, 0.9] },
    { from: [-0.1, 0], to: [0.1, -0.1], normal: [0.4, -0.9] },
  ],
});
check(tri.positions.length / 3 === 3 * (8 + 1) * 2, 'revolveProfile takes an arbitrary profile',
      `${tri.positions.length / 3} vertices for a 3-edge profile`);

// ---- box: flat faces, outward winding ----
//
// The winding is the whole test. Back-face culling makes a reversed box INVISIBLE rather than wrong,
// which is the failure mode that survives a screenshot review — and the ring generator shipped exactly
// that bug once, on all 2304 of its triangles, hidden by `cullMode: 'none'`.
{
  const b = box([2, 3, 4]);
  check(b.positions.length / 3 === 24, 'a box has 24 vertices, not 8 — three normals meet at a corner',
        `${b.positions.length / 3} vertices`);

  let mismatched = 0;
  let area = 0;
  for (let t = 0; t < b.indices.length / 3; t++) {
    const [i0, i1, i2] = [0, 1, 2].map((k) => b.indices[t * 3 + k]);
    const at = (j) => [0, 1, 2].map((k) => b.positions[j * 3 + k]);
    const [p0, p1, p2] = [at(i0), at(i1), at(i2)];
    const e1 = p1.map((x, k) => x - p0[k]);
    const e2 = p2.map((x, k) => x - p0[k]);
    // Geometric normal from the winding: it must agree with the vertex normal, or the triangle faces in.
    const g = [e1[1] * e2[2] - e1[2] * e2[1],
               e1[2] * e2[0] - e1[0] * e2[2],
               e1[0] * e2[1] - e1[1] * e2[0]];
    const len = Math.hypot(...g);
    area += len / 2;
    const nv = [0, 1, 2].map((k) => b.normals[i0 * 3 + k]);
    if (g.reduce((acc, x, k) => acc + (x / len) * nv[k], 0) < 0.999) mismatched++;
  }
  check(mismatched === 0, 'every box triangle is wound outward, matching its own vertex normal',
        `${mismatched} facing the wrong way`);
  // 2*(4*6 + 4*8 + 6*8) for half-extents 2,3,4 — exact, so a scaled or duplicated face shows up here.
  check(Math.abs(area - 208) < 1e-9, 'and the surface area is exactly right',
        `${area.toFixed(4)} vs 208`);
}

process.exit(failed === 0 ? 0 : 1);
