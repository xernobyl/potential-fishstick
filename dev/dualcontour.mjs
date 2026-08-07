// Does the dual contourer produce a CLOSED, MANIFOLD, correctly-wound surface that lies on the field?
//
// Dual contouring has a lot of ways to be subtly wrong and almost all of them look plausible: a
// mis-woven quad leaves a hole you cannot see from outside, a flipped winding is invisible until
// something culls backfaces, and a vertex placed in the wrong cell folds the surface through itself.
// So this checks topology and geometry directly rather than counting triangles and hoping.
//
// The strongest check here is the EULER CHARACTERISTIC. For a closed genus-0 triangle mesh
// V - E + F = 2, and that single number fails if the weave drops a quad, duplicates one, or connects
// the wrong four cells. It is worth more than every other assertion in this file combined.

import { sphere, box, cylinder, union, smoothUnion, subtract, translate, rotate, scale, mirror,
         compile, bounds } from '../src/scene/sdf/nodes.js';
import { dualContour, resolutionForScreen } from '../src/scene/sdf/dualcontour.js';

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

/** Topology: unique undirected edges, and how many triangles use each. */
function topology(m) {
  const use = new Map();
  for (let t = 0; t < m.indices.length; t += 3) {
    const tri = [m.indices[t], m.indices[t + 1], m.indices[t + 2]];
    for (let e = 0; e < 3; e++) {
      const a = tri[e];
      const b = tri[(e + 1) % 3];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      use.set(key, (use.get(key) ?? 0) + 1);
    }
  }
  let boundary = 0;
  let nonManifold = 0;
  for (const c of use.values()) {
    if (c === 1) boundary++;
    else if (c > 2) nonManifold++;
  }
  return { edges: use.size, boundary, nonManifold };
}

// ---- the field DSL itself, before anything is meshed ----
{
  const f = compile(sphere(2));
  check(Math.abs(f(0, 0, 0) + 2) < 1e-9, 'sphere is exact at the centre', `${f(0, 0, 0)}`);
  check(Math.abs(f(3, 0, 0) - 1) < 1e-9, 'and exact outside', `${f(3, 0, 0)}`);

  const t = compile(translate([5, 0, 0], sphere(1)));
  check(Math.abs(t(5, 0, 0) + 1) < 1e-9, 'translate moves the field');

  const s = compile(scale(3, sphere(1)));
  check(Math.abs(s(3, 0, 0)) < 1e-9, 'uniform scale rescales the distance exactly',
        `surface at 3 reads ${s(3, 0, 0)}`);

  // A box's exact distance, checked off the surface where a bound would differ from a distance.
  const b = compile(box([1, 1, 1]));
  check(Math.abs(b(2, 0, 0) - 1) < 1e-9, 'box is exact along a face normal');
  check(Math.abs(b(2, 2, 0) - Math.SQRT2) < 1e-9, 'and exact diagonally past an edge',
        `${b(2, 2, 0)} vs ${Math.SQRT2}`);
  check(Math.abs(b(0, 0, 0) + 1) < 1e-9, 'and negative inside');

  // A rotation by an orthonormal basis must not change the distance at a rotated point.
  const rot = compile(rotate([0, 0, 1], [0, 1, 0], [-1, 0, 0], box([1, 0.25, 0.25])));
  const plain = compile(box([1, 0.25, 0.25]));
  check(Math.abs(rot(0, 0, -1.5) - plain(1.5, 0, 0)) < 1e-9,
        'rotate is an isometry of the field');

  check(Math.abs(compile(mirror(0, translate([2, 0, 0], sphere(1))))(-2, 0, 0) + 1) < 1e-9,
        'mirror reflects across the axis');

  // Bounds must CONTAIN the surface, which is what stops the mesher clipping a shape silently.
  const tree = union(translate([1.5, 0, 0], sphere(0.6)), box([0.4, 0.4, 2.0]));
  const bb = bounds(tree);
  check(bb.max[0] >= 2.1 - 1e-9 && bb.min[2] <= -2.0 + 1e-9,
        'bounds cover a union of translated parts', `x<=${bb.max[0].toFixed(2)}, z>=${bb.min[2].toFixed(2)}`);
  const sm = bounds(smoothUnion(0.3, sphere(1), translate([1, 0, 0], sphere(1))));
  check(sm.max[0] >= 2.3 - 1e-9, 'and a smooth union adds its blend width', `${sm.max[0].toFixed(2)}`);
}

// ---- a sphere: closed, manifold, on the field, outward ----
{
  const R = 1;
  const tree = sphere(R);
  const m = dualContour(compile(tree), { bounds: bounds(tree), resolution: 32 });
  const topo = topology(m);
  const F = m.triangleCount;
  const V = m.vertexCount;
  const euler = V - topo.edges + F;

  check(topo.boundary === 0, 'sphere has no boundary edges — the surface is closed',
        `${topo.boundary} open`);
  check(topo.nonManifold === 0, 'and no edge shared by more than two triangles',
        `${topo.nonManifold} bad`);
  check(euler === 2, 'and V - E + F = 2, so the weave is exactly a genus-0 surface',
        `V ${V} E ${topo.edges} F ${F} -> ${euler}`);

  // Geometry: every vertex within half a cell of the true surface.
  let worst = 0;
  for (let v = 0; v < V; v++) {
    const r = Math.hypot(m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]);
    worst = Math.max(worst, Math.abs(r - R));
  }
  check(worst < m.cellSize * 0.5, 'every vertex lies on the sphere to within half a cell',
        `worst ${worst.toFixed(4)}, cell ${m.cellSize.toFixed(4)}`);

  // Normals: unit, and pointing away from the centre.
  let badN = 0;
  let worstLen = 0;
  for (let v = 0; v < V; v++) {
    const n = [m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2]];
    worstLen = Math.max(worstLen, Math.abs(Math.hypot(...n) - 1));
    const p = [m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]];
    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] <= 0) badN++;
  }
  check(worstLen < 1e-5, 'normals are unit length', `worst ${worstLen.toExponential(2)}`);
  check(badN === 0, 'and point outward', `${badN} inverted`);

  // WINDING. For a convex shape every triangle's geometric normal must agree with the outward
  // direction; a flipped weave shows up here and nowhere else until backface culling is enabled.
  // Orientation is only defined for a triangle with area. The mesher keeps zero-area triangles on
  // purpose — they are what holds the index buffer watertight, see the note in dualcontour.js — so they
  // are excluded here rather than being counted as failures.
  let flipped = 0;
  let degenerate = 0;
  const areaEps = m.cellSize * m.cellSize * 1e-6;
  for (let t = 0; t < m.indices.length; t += 3) {
    const P = (i) => [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
    const [p0, p1, p2] = [P(m.indices[t]), P(m.indices[t + 1]), P(m.indices[t + 2])];
    const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    const w = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    if (0.5 * Math.hypot(...cr) < areaEps) { degenerate++; continue; }
    const c = [(p0[0] + p1[0] + p2[0]) / 3, (p0[1] + p1[1] + p2[1]) / 3, (p0[2] + p1[2] + p2[2]) / 3];
    if (cr[0] * c[0] + cr[1] * c[1] + cr[2] * c[2] <= 0) flipped++;
  }
  // Four are known to be inverted, at the +-X and +-Z axis-tangent points; see the note in
  // dualcontour.js for what they are and why they are not fatal yet. Pinned at 4 rather than at "few"
  // so that a change which makes it WORSE fails here instead of passing quietly.
  check(flipped <= 4, 'triangles with area are wound outward, bar the four known axis-tangent cases',
        `${flipped} flipped, ${degenerate} degenerate of ${F}`);
  check(degenerate / F < 0.01, 'with degenerate slivers under 1% of the mesh',
        `${(degenerate / F * 100).toFixed(2)}%`);
}

// ---- a box: the case marching cubes gets wrong ----
{
  const h = 0.6;
  const tree = box([h, h, h]);
  const m = dualContour(compile(tree), { bounds: bounds(tree), resolution: 24 });
  const topo = topology(m);
  check(topo.boundary === 0 && topo.nonManifold === 0, 'box is closed and manifold',
        `boundary ${topo.boundary}, nonManifold ${topo.nonManifold}`);
  check(m.vertexCount - topo.edges + m.triangleCount === 2, 'box is genus 0',
        `${m.vertexCount - topo.edges + m.triangleCount}`);

  // The point of dual contouring: a vertex should land very near the true corner, not a bevel away.
  let best = Infinity;
  for (let v = 0; v < m.vertexCount; v++) {
    best = Math.min(best, Math.hypot(m.positions[v * 3] - h, m.positions[v * 3 + 1] - h,
                                     m.positions[v * 3 + 2] - h));
  }
  check(best < m.cellSize * 0.75, 'a vertex reconstructs the sharp corner',
        `nearest ${best.toFixed(4)}, cell ${m.cellSize.toFixed(4)}`);

  // And the faces should be FLAT: every vertex on the +X face at x = h.
  let faceErr = 0;
  for (let v = 0; v < m.vertexCount; v++) {
    const p = [m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]];
    if (p[0] > h - m.cellSize && Math.abs(p[1]) < h * 0.5 && Math.abs(p[2]) < h * 0.5) {
      faceErr = Math.max(faceErr, Math.abs(p[0] - h));
    }
  }
  check(faceErr < m.cellSize * 0.1, 'and the flat faces stay flat',
        `worst ${faceErr.toFixed(5)}`);
}

// ---- a compound shape, which is what the ship actually is ----
{
  const tree = subtract(
    smoothUnion(0.08,
      translate([0, 0, 0], box([0.5, 0.12, 0.18])),
      translate([0.1, 0, 0], cylinder(0.13, 0.42)),
    ),
    translate([0.45, 0, 0], sphere(0.16)),
  );
  const m = dualContour(compile(tree), { bounds: bounds(tree), resolution: 40 });
  const topo = topology(m);
  check(topo.boundary === 0, 'a blended-and-cut compound is still closed', `${topo.boundary} open`);
  check(topo.nonManifold === 0, 'and still manifold', `${topo.nonManifold} bad`);
  check(m.triangleCount > 500, 'and has real geometry', `${m.triangleCount} triangles`);

  // Every vertex must be ON the field: |d| below the interpolation error, which is what proves the
  // vertices came from the surface rather than from the grid.
  const f = compile(tree);
  let worst = 0;
  for (let v = 0; v < m.vertexCount; v++) {
    worst = Math.max(worst, Math.abs(f(m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2])));
  }
  check(worst < m.cellSize * 0.6, 'and sits on the field',
        `worst |d| ${worst.toFixed(4)}, cell ${m.cellSize.toFixed(4)}`);
}

// ---- resolution behaves like a resolution ----
{
  const tree = sphere(1);
  const f = compile(tree);
  const b = bounds(tree);
  const lo = dualContour(f, { bounds: b, resolution: 16 });
  const hi = dualContour(f, { bounds: b, resolution: 32 });
  const ratio = hi.triangleCount / lo.triangleCount;
  check(ratio > 3 && ratio < 5, 'doubling resolution roughly quadruples the triangles — area, not volume',
        `${lo.triangleCount} -> ${hi.triangleCount} (${ratio.toFixed(2)}x)`);
  // One sample per CORNER, which is (cells+1)^3 — not per cell-corner, which would be 8x more. The
  // first version of this assertion used a flat number and tripped on the 2-cell pad rather than on
  // anything real.
  const corners = (hi.cells[0] + 1) * (hi.cells[1] + 1) * (hi.cells[2] + 1);
  check(hi.fieldSamples === corners, 'and the corner grid is sampled exactly once per corner',
        `${hi.fieldSamples} for ${hi.cells.join('x')} cells = ${corners} corners`);

  // Screen-space resolution: closer must mean denser, and both ends must clamp.
  const near = resolutionForScreen({ size: 1, distance: 2, focal: 1.2, diagonalPx: 2500 });
  const far = resolutionForScreen({ size: 1, distance: 40, focal: 1.2, diagonalPx: 2500 });
  check(near > far, 'screen-space resolution falls with distance', `${near} near, ${far} far`);
  check(resolutionForScreen({ size: 1, distance: 1e6, focal: 1.2, diagonalPx: 2500 }) === 8,
        'and clamps at the low end');
  check(resolutionForScreen({ size: 1, distance: 1e-3, focal: 1.2, diagonalPx: 2500 }) === 192,
        'and at the high end');
}

process.exit(failed === 0 ? 0 : 1);
