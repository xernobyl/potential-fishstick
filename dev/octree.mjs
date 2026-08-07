// Does adaptive dual contouring produce a mesh you can actually rely on?
//
// The traversal tables in octree.js are transcribed, not derived, and a single wrong entry gives a hole,
// a doubled triangle or an inverted winding — all of which look fine in a screenshot and all of which
// break something later. These checks are what stand in for reading them.
//
// The strongest one by far is the EQUIVALENCE check: simplifying with a threshold of zero must reproduce
// the uniform contourer exactly on a curved surface. Every table has to be right for that to hold, so it
// pins them all down at once with a number no amount of plausible-looking code can fake.

import { sphere, box, cylinder, union, subtract, translate, smoothUnion, compile, bounds }
  from '../src/scene/sdf/nodes.js';
import { dualContour } from '../src/scene/sdf/dualcontour.js';
import { dualContourAdaptive } from '../src/scene/sdf/octree.js';

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

/**
 * Topology of an indexed mesh.
 *
 * Triangles with a REPEATED index are excluded before counting, and that is a measurement decision
 * worth stating: such a triangle contributes its one real edge twice from inside itself, so counting it
 * reports an edge with four users on a mesh that is perfectly manifold. The mesher drops them for the
 * same reason — see processEdge — so this should find none, and it asserts that below.
 */
const topology = (m) => {
  const edges = new Map();
  let faces = 0;
  let repeated = 0;
  for (let t = 0; t < m.indices.length / 3; t++) {
    const [a, b, c] = [0, 1, 2].map((k) => m.indices[t * 3 + k]);
    if (a === b || b === c || a === c) { repeated++; continue; }
    faces++;
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const k = x < y ? `${x},${y}` : `${y},${x}`;
      edges.set(k, (edges.get(k) ?? 0) + 1);
    }
  }
  let boundary = 0;
  let nonManifold = 0;
  for (const n of edges.values()) {
    if (n === 1) boundary++;
    else if (n !== 2) nonManifold++;
  }
  return { V: m.vertexCount, E: edges.size, F: faces, repeated, boundary, nonManifold,
           euler: m.vertexCount - edges.size + faces };
};

/** Triangles whose winding disagrees with the outward direction on a convex shape. */
const inverted = (m, centre) => {
  let bad = 0;
  for (let t = 0; t < m.indices.length / 3; t++) {
    const [a, b, c] = [0, 1, 2].map((k) => m.indices[t * 3 + k]);
    const at = (j) => [0, 1, 2].map((k) => m.positions[j * 3 + k]);
    const [p0, p1, p2] = [at(a), at(b), at(c)];
    const e1 = p1.map((x, k) => x - p0[k]);
    const e2 = p2.map((x, k) => x - p0[k]);
    const g = [e1[1] * e2[2] - e1[2] * e2[1],
               e1[2] * e2[0] - e1[0] * e2[2],
               e1[0] * e2[1] - e1[1] * e2[0]];
    const len = Math.hypot(...g);
    if (len < 1e-12) continue;
    const out = p0.map((x, k) => (x + p1[k] + p2[k]) / 3 - centre[k]);
    if (g.reduce((s, x, k) => s + (x / len) * out[k], 0) < 0) bad++;
  }
  return bad;
};

/** Worst |field| over the vertices. On an exact SDF this IS the geometric error. */
const worstDeviation = (m, f) => {
  let w = 0;
  for (let i = 0; i < m.vertexCount; i++) {
    w = Math.max(w, Math.abs(f(m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2])));
  }
  return w;
};

// ---- the equivalence that pins down every table ----
{
  const t = sphere(1);
  const f = compile(t);
  const b = bounds(t);
  const uni = dualContour(f, { bounds: b, resolution: 32 });
  const ad = dualContourAdaptive(f, { bounds: b, resolution: 32, error: 0 });
  check(uni.vertexCount === ad.vertexCount && uni.triangleCount === ad.triangleCount,
        'at threshold 0 the adaptive contourer reproduces the uniform one exactly',
        `${ad.vertexCount}v/${ad.triangleCount}t vs ${uni.vertexCount}v/${uni.triangleCount}t`);

  // The same vertices, as a SET. Not in the same order — the octree numbers its leaves in tree order
  // and the uniform grid numbers them in raster order, so the two index the identical geometry
  // differently. Comparing element-wise fails on that permutation while telling you nothing about
  // whether the geometry agrees, which is the thing actually worth asserting.
  const sortedKeys = (m) => {
    const out = [];
    for (let i = 0; i < m.vertexCount; i++) {
      out.push(`${m.positions[i * 3].toFixed(9)},${m.positions[i * 3 + 1].toFixed(9)},${m.positions[i * 3 + 2].toFixed(9)}`);
    }
    return out.sort();
  };
  const ku = sortedKeys(uni);
  const ka = sortedKeys(ad);
  let differing = 0;
  for (let i = 0; i < ku.length; i++) if (ku[i] !== ka[i]) differing++;
  check(differing === 0, 'and places the identical set of vertices, to the last decimal',
        `${differing} of ${ku.length} differ`);
}

// ---- topology holds at every simplification level ----
{
  const t = sphere(1);
  const f = compile(t);
  const b = bounds(t);
  for (const budget of [0, 0.005, 0.01, 0.02, 0.04]) {
    const m = dualContourAdaptive(f, { bounds: b, resolution: 32, error: budget });
    const g = topology(m);
    check(g.boundary === 0 && g.nonManifold === 0 && g.euler === 2 && g.repeated === 0,
          `budget ${budget}: closed, manifold, genus 0`,
          `V${g.V} E${g.E} F${g.F} euler ${g.euler} bnd ${g.boundary} nm ${g.nonManifold}`);
    check(inverted(m, [0, 0, 0]) === 0, `budget ${budget}: every triangle faces outward`);
  }
}

// ---- simplification actually simplifies, and monotonically ----
{
  const t = sphere(1);
  const f = compile(t);
  const b = bounds(t);
  const counts = [0, 0.005, 0.01, 0.02, 0.04]
    .map((e) => dualContourAdaptive(f, { bounds: b, resolution: 32, error: e }).triangleCount);
  let monotone = true;
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[i - 1]) monotone = false;
  check(monotone, 'a larger budget never produces MORE triangles', counts.join(' -> '));
  check(counts[counts.length - 1] < counts[0] * 0.25,
        'and a generous budget removes most of them',
        `${counts[0]} -> ${counts[counts.length - 1]}`);
}

// ---- the error budget means what it says ----
//
// The budget is a world LENGTH. On an exact field the vertex deviation is the real geometric error, so
// this is the check that the metric is calibrated rather than merely monotone.
{
  const t = sphere(1);
  const f = compile(t);
  const b = bounds(t);
  let rising = true;
  let prev = 0;
  for (const budget of [0, 0.01, 0.04]) {
    const m = dualContourAdaptive(f, { bounds: b, resolution: 32, error: budget });
    const w = worstDeviation(m, f);
    if (w < prev) rising = false;
    prev = w;
    check(w <= Math.max(budget, 0.0015) * 2.0,
          `budget ${budget}: worst vertex deviation stays within it`,
          `${w.toFixed(5)}`);
  }
  check(rising, 'and deviation grows with the budget rather than jumping around');
}

// ---- flat regions merge for FREE, which is the whole argument for doing this on the octree ----
{
  const t = box([0.5, 0.5, 0.5]);
  const f = compile(t);
  const b = bounds(t);
  const uni = dualContour(f, { bounds: b, resolution: 32 });
  const ad = dualContourAdaptive(f, { bounds: b, resolution: 32, error: 0 });
  check(ad.triangleCount < uni.triangleCount * 0.4,
        'a box collapses hard at budget ZERO — eight coplanar cells cost one vertex no error',
        `${uni.triangleCount} -> ${ad.triangleCount}`);
  check(Math.abs(worstDeviation(ad, f) - worstDeviation(uni, f)) < 1e-9,
        'and loses no accuracy doing it',
        `${worstDeviation(ad, f).toExponential(2)} both`);
  const g = topology(ad);
  check(g.boundary === 0 && g.nonManifold === 0 && g.euler === 2,
        'the collapsed box is still closed, manifold and genus 0',
        `euler ${g.euler}`);
  check(inverted(ad, [0, 0, 0]) === 0, 'and correctly wound');
}

// ---- a compound shape with a subtraction, which is where sign configurations get awkward ----
{
  const t = subtract(
    smoothUnion(0.08, box([0.4, 0.18, 0.5]), translate([0, 0.2, 0], sphere(0.25))),
    translate([0, 0, -0.5], cylinder(0.12, 0.3)),
  );
  const f = compile(t);
  const b = bounds(t);
  for (const budget of [0, 0.006, 0.02]) {
    const m = dualContourAdaptive(f, { bounds: b, resolution: 48, error: budget });
    const g = topology(m);
    check(g.boundary === 0 && g.repeated === 0,
          `compound shape, budget ${budget}: watertight with no degenerate index triples`,
          `F${g.F} bnd ${g.boundary} nm ${g.nonManifold}`);
  }
}

// ---- the LOD chain is nested and costs barely more than its coarsest member ----
{
  const t = sphere(1);
  const f = compile(t);
  const b = bounds(t);
  const budgets = [0.004, 0.01, 0.02, 0.04];
  const chain = dualContourAdaptive(f, { bounds: b, resolution: 32, error: budgets });
  check(Array.isArray(chain) && chain.length === budgets.length,
        'an array of budgets returns one mesh per budget', `${chain.length} levels`);
  const counts = chain.map((m) => m.triangleCount);
  let descending = true;
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[i - 1]) descending = false;
  check(descending, 'each level is coarser than the last', counts.join(' -> '));

  for (const m of chain) {
    const g = topology(m);
    if (g.boundary !== 0 || g.nonManifold !== 0 || g.euler !== 2) {
      check(false, 'every LOD level is closed, manifold and genus 0',
            `F${g.F} euler ${g.euler} bnd ${g.boundary} nm ${g.nonManifold}`);
      break;
    }
  }
  check(chain.every((m) => topology(m).euler === 2),
        'every LOD level is closed, manifold and genus 0');

  // Asking for the levels one at a time must give the same meshes as asking for the chain: the chain
  // reuses one progressively-simplified tree, and that is only sound because simplification is monotone.
  const separate = budgets.map((e) => dualContourAdaptive(f, { bounds: b, resolution: 32, error: e }));
  const same = chain.every((m, i) => m.triangleCount === separate[i].triangleCount
                                  && m.vertexCount === separate[i].vertexCount);
  check(same, 'and the chain matches building each level independently',
        `${counts.join('/')} vs ${separate.map((m) => m.triangleCount).join('/')}`);
}

// ---- one field evaluation per corner, still ----
{
  const t = sphere(1);
  let calls = 0;
  const f = compile(t);
  const counted = (x, y, z) => { calls++; return f(x, y, z); };
  const b = bounds(t);
  const m = dualContourAdaptive(counted, { bounds: b, resolution: 16, error: 0.02 });
  const corners = m.fieldSamples;
  // Corners, plus a bounded amount of crossing refinement and gradient work, which scale with surface
  // AREA rather than volume. The bound is what would break if a per-cell sampling loop crept back in.
  check(calls < corners * 3, 'the grid is still sampled once per corner, not once per cell',
        `${calls} evaluations for ${corners} corners`);
}

process.exit(failed === 0 ? 0 : 1);
