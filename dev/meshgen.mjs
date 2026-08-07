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

import { rectTube, revolveProfile, concatMeshes, interleave } from '../src/scene/meshgen.js';

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
check(nv === 4 * SEGMENTS * 2, 'vertex count is 4 faces x segments x 2',
      `${nv} vs ${4 * SEGMENTS * 2}`);
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

// ---- concatenation tags and offsets ----
const three = concatMeshes([
  rectTube({ segments: 16, radius: 1, halfW: 0.2, halfH: 0.05 }),
  rectTube({ segments: 16, radius: 2, halfW: 0.2, halfH: 0.05 }),
  rectTube({ segments: 16, radius: 3, halfW: 0.2, halfH: 0.05 }),
]);
const perRing = 4 * 16 * 2;
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

const inter = interleave(three);
check(inter.length === three.vertexCount * 7, 'interleave packs 7 floats per vertex');
check(inter[6] === 0 && inter[perRing * 7 + 6] === 1, 'the id survives interleaving');

// ---- density is a knob, and the profile primitive generalises ----
const dense = rectTube({ segments: 512, radius: 3.4, halfW: HW, halfH: HH });
check(dense.positions.length / 3 === 4 * 512 * 2, 'density scales with `segments`');
const tri = revolveProfile({
  segments: 8,
  radius: 1,
  profile: [
    { from: [0.1, -0.1], to: [0.1, 0.1], normal: [1, 0] },
    { from: [0.1, 0.1], to: [-0.1, 0], normal: [0.4, 0.9] },
    { from: [-0.1, 0], to: [0.1, -0.1], normal: [0.4, -0.9] },
  ],
});
check(tri.positions.length / 3 === 3 * 8 * 2, 'revolveProfile takes an arbitrary profile',
      `${tri.positions.length / 3} vertices for a 3-edge profile`);

process.exit(failed === 0 ? 0 : 1);
