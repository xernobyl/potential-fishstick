/**
 * ADAPTIVE dual contouring: one octree, simplified by real error, woven without cracks.
 *
 * The uniform contourer in dualcontour.js spends the same number of triangles on a flat panel as on a
 * nozzle rim, because a uniform grid has no way to know the difference. This one does. It builds the
 * same finest-level cells, then merges any group of eight whose surface a single vertex can represent to
 * within a stated error — so triangles end up where the shape actually curves.
 *
 * WHY THIS AND NOT DECIMATION AFTERWARDS. The obvious alternative is to contour uniformly and then run
 * an edge-collapse decimator with a quadric error metric. But dual contouring has ALREADY computed that
 * metric: the QEF each cell solves for its vertex is a quadric, and it is built from the field's own
 * gradients rather than reconstructed from the triangles those gradients produced. Decimating afterwards
 * means throwing that away, rebuilding an approximation of it from the mesh, and paying for a full-
 * resolution mesh first. Simplifying the octree uses the exact quantity, and never builds the triangles
 * it is going to delete.
 *
 * WHY THE ERROR IS TRUSTWORTHY. A QEF in normal-equation form is ADDITIVE (see qef.js): a parent's error
 * is the exact sum of its children's, not a re-estimate. So the threshold means the same thing at every
 * level of the tree, which is what lets one number control the whole simplification.
 *
 * CRACKS ARE THE HARD PART, and they are the reason this is a recursion over node groups rather than a
 * loop over cells. Where a large merged cell meets small ones, the surface must still be stitched into a
 * single closed sheet. The `cellProc` / `faceProc` / `edgeProc` descent below is Ju et al.'s solution
 * (Dual Contouring of Hermite Data, SIGGRAPH 2002): it enumerates every minimal edge exactly once, no
 * matter how the tree is refined around it, and emits one quad per edge between the four cells that
 * share it. That is what makes the output watertight rather than nearly watertight.
 *
 * THE TABLES BELOW ARE LOAD-BEARING AND ARE NOT DERIVED HERE. They encode which children of which nodes
 * share a given face or edge, and a single wrong entry produces a hole, a doubled triangle, or an
 * inverted winding — all of which are silent in a screenshot. They are checked rather than trusted: the
 * headless suite asserts the result is closed, manifold, genus-0 by Euler characteristic, and has zero
 * inverted triangles, and separately that simplifying with a threshold of zero reproduces the uniform
 * contourer's output exactly. Those two together pin the tables down.
 */

import { QefPool } from './qef.js';
import { sampleGrid, findCrossings, CELL_EDGES, FloatList, IntList } from './grid.js';

/**
 * Child/corner ordering: index i is the corner at offset ((i>>2)&1, (i>>1)&1, i&1).
 *
 * Every table in this file is written against this ordering. It is the ordering Ju's tables assume, and
 * changing it means regenerating all of them, not renumbering one.
 */
const CHILD_OFFSETS = [
  [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
];

/** The 12 edges as corner pairs, grouped by axis: x = 0..3, y = 4..7, z = 8..11. */
const EDGE_VMAP = [
  [0, 4], [1, 5], [2, 6], [3, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 1], [2, 3], [4, 5], [6, 7],
];

// ---- Ju et al.'s traversal tables. See the header: verified by assertion, not by inspection. ----

const CELL_FACE_MASK = [
  [0, 4, 0], [1, 5, 0], [2, 6, 0], [3, 7, 0],
  [0, 2, 1], [4, 6, 1], [1, 3, 1], [5, 7, 1],
  [0, 1, 2], [2, 3, 2], [4, 5, 2], [6, 7, 2],
];

const CELL_EDGE_MASK = [
  [0, 1, 2, 3, 0], [4, 5, 6, 7, 0],
  [0, 4, 1, 5, 1], [2, 6, 3, 7, 1],
  [0, 2, 4, 6, 2], [1, 3, 5, 7, 2],
];

const FACE_FACE_MASK = [
  [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
  [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
  [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]],
];

const FACE_EDGE_MASK = [
  [[1, 4, 0, 5, 1, 1], [1, 6, 2, 7, 3, 1], [0, 4, 6, 0, 2, 2], [0, 5, 7, 1, 3, 2]],
  [[0, 2, 3, 0, 1, 0], [0, 6, 7, 4, 5, 0], [1, 2, 0, 6, 4, 2], [1, 3, 1, 7, 5, 2]],
  [[1, 1, 0, 3, 2, 0], [1, 5, 4, 7, 6, 0], [0, 1, 5, 0, 4, 1], [0, 3, 7, 2, 6, 1]],
];

const EDGE_EDGE_MASK = [
  [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
  [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
  [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]],
];

const PROCESS_EDGE_MASK = [[3, 2, 1, 0], [7, 5, 6, 4], [11, 10, 9, 8]];

/** Which of the two nodes of a face pair supplies each of the four edge slots. */
const FACE_EDGE_ORDERS = [[0, 0, 1, 1], [0, 1, 0, 1]];

const LEAF = 0;
const INTERNAL = 1;

/**
 * For each of the 256 corner-sign configurations: can ONE vertex represent this cell's surface?
 *
 * This is the test that makes the output manifold, and without it the whole scheme quietly produces
 * meshes that are watertight but not manifold — edges shared by four triangles instead of two.
 *
 * A cube cell's surface separates the inside corners from the outside ones. If both sets are connected
 * under cube-edge adjacency, that separating surface is a single sheet and a single vertex is the right
 * model for it. If either set is disconnected — two opposite corners inside, say — the cell contains two
 * separate sheets, and collapsing eight children into one vertex welds them into a non-manifold pinch.
 *
 * Merging is what creates these configurations. The leaves come straight from the field, where the grid
 * is fine enough that a cell rarely straddles two sheets; a merged cell is eight times larger, so it
 * straddles them routinely. Ju's original formulation checks only the error and stops here, which is
 * exactly where the non-manifold vertices in adaptive dual contouring come from — the problem
 * Manifold Dual Contouring (Schaefer, Ju et al. 2007) exists to solve. Refusing the collapse is the
 * conservative half of that solution: it costs a few triangles on the cells where it triggers and buys
 * a mesh that every downstream consumer can rely on.
 */
const SINGLE_SHEET = (() => {
  // Corner index bits are (x<<2)|(y<<1)|z, so the three cube-edge neighbours of a corner are the
  // indices differing in exactly one bit.
  const connected = (mask) => {
    let start = -1;
    for (let c = 0; c < 8; c++) if (mask & (1 << c)) { start = c; break; }
    if (start < 0) return false;
    let seen = 1 << start;
    const stack = [start];
    while (stack.length) {
      const c = stack.pop();
      for (const bit of [1, 2, 4]) {
        const n = c ^ bit;
        if ((mask & (1 << n)) && !(seen & (1 << n))) { seen |= (1 << n); stack.push(n); }
      }
    }
    return seen === mask;
  };
  const table = new Uint8Array(256);
  for (let m = 0; m < 256; m++) {
    table[m] = (connected(m) && connected((~m) & 255)) ? 1 : 0;
  }
  return table;
})();

/**
 * Node storage as a flat pool.
 *
 * Struct-of-arrays rather than an object per node: the ship's hull has a few thousand surface cells and
 * roughly a third as many interior nodes again, all walked repeatedly during simplification and
 * contouring. Objects would put every one of those walks through pointer chasing and the build through
 * the garbage collector.
 */
class NodePool {
  constructor(capacity = 4096) {
    this.type = new Uint8Array(capacity);
    this.children = new Int32Array(capacity * 8).fill(-1);
    this.corners = new Uint8Array(capacity);     // bitmask: bit i set = corner i is inside
    this.qef = new Int32Array(capacity).fill(-1);
    this.size = new Int32Array(capacity);        // in finest-cell units
    this.minI = new Int32Array(capacity);
    this.minJ = new Int32Array(capacity);
    this.minK = new Int32Array(capacity);
    this.px = new Float64Array(capacity);
    this.py = new Float64Array(capacity);
    this.pz = new Float64Array(capacity);
    this.nx = new Float64Array(capacity);
    this.ny = new Float64Array(capacity);
    this.nz = new Float64Array(capacity);
    this.vertexIndex = new Int32Array(capacity).fill(-1);
    this.length = 0;
  }

  #grow() {
    const cap = this.type.length * 2;
    const copy = (arr, Type, per = 1) => {
      const g = new Type(cap * per);
      g.set(arr);
      return g;
    };
    const oldChildren = this.children;
    this.type = copy(this.type, Uint8Array);
    this.children = new Int32Array(cap * 8).fill(-1);
    this.children.set(oldChildren);
    this.corners = copy(this.corners, Uint8Array);
    const oldQef = this.qef;
    this.qef = new Int32Array(cap).fill(-1);
    this.qef.set(oldQef);
    this.size = copy(this.size, Int32Array);
    this.minI = copy(this.minI, Int32Array);
    this.minJ = copy(this.minJ, Int32Array);
    this.minK = copy(this.minK, Int32Array);
    this.px = copy(this.px, Float64Array);
    this.py = copy(this.py, Float64Array);
    this.pz = copy(this.pz, Float64Array);
    this.nx = copy(this.nx, Float64Array);
    this.ny = copy(this.ny, Float64Array);
    this.nz = copy(this.nz, Float64Array);
    const oldVi = this.vertexIndex;
    this.vertexIndex = new Int32Array(cap).fill(-1);
    this.vertexIndex.set(oldVi);
  }

  alloc(type, i, j, k, size) {
    if (this.length + 1 > this.type.length) this.#grow();
    const n = this.length++;
    this.type[n] = type;
    this.minI[n] = i;
    this.minJ[n] = j;
    this.minK[n] = k;
    this.size[n] = size;
    this.corners[n] = 0;
    this.qef[n] = -1;
    this.vertexIndex[n] = -1;
    for (let c = 0; c < 8; c++) this.children[n * 8 + c] = -1;
    return n;
  }
}

/**
 * Mesh a compiled SDF adaptively.
 *
 * @param {(x:number,y:number,z:number)=>number} f  compiled field, from `compile()`
 * @param {object} opts
 * @param {{min:number[],max:number[]}} opts.bounds  world box to grid
 * @param {number} opts.resolution                   finest cells along the LONGEST axis
 * @param {number|number[]} [opts.error]  simplification budget in WORLD LENGTH. One value returns one
 *        mesh; an array returns one mesh per value — an LOD chain, see below.
 * @param {number} [opts.bias]  QEF regularisation, 0..1
 * @param {number} [opts.pad]   cells of margin, so the surface is never clipped
 * @returns {object|object[]}  a mesh, or one per error value
 */
export function dualContourAdaptive(f, { bounds, resolution, error = 0, bias = 0.1, pad = 2 }) {
  const grid = sampleGrid(f, { bounds, resolution, pad });
  const { d, lo, cell, nxC, nyC, nzC, cornerIdx } = grid;
  const { edgeMap, exs, ens, grad } = findCrossings(f, grid);

  const qefs = new QefPool();
  const nodes = new NodePool();
  const vout = [0, 0, 0];
  const nrm = [0, 0, 0];

  // ---- 1. leaves: one per finest cell the surface passes through ----
  //
  // Keyed by cell coordinate in a Map rather than a dense array: the surface is a two-dimensional set
  // inside a three-dimensional grid, so the populated cells are a small fraction of the volume and a
  // dense array would be mostly -1.
  let level = new Map();
  const key = (i, j, k) => (k * nyC + j) * nxC + i;

  for (let k = 0; k < nzC; k++) {
    for (let j = 0; j < nyC; j++) {
      for (let i = 0; i < nxC; i++) {
        let count = 0;
        const h = qefs.alloc();
        for (let e = 0; e < 12; e++) {
          const [oi, oj, ok, axis] = CELL_EDGES[e];
          const id = edgeMap[(cornerIdx(i + oi, j + oj, k + ok) * 3) + axis];
          if (id < 0) continue;
          qefs.addPlane(h,
            exs.a[id * 3], exs.a[id * 3 + 1], exs.a[id * 3 + 2],
            ens.a[id * 3], ens.a[id * 3 + 1], ens.a[id * 3 + 2]);
          count++;
        }
        if (count === 0) continue;

        const n = nodes.alloc(LEAF, i, j, k, 1);
        nodes.qef[n] = h;
        // Corner signs, in the ordering every table in this file assumes.
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const [ci, cj, ck] = CHILD_OFFSETS[c];
          if (!(d[cornerIdx(i + ci, j + cj, k + ck)] > 0)) mask |= (1 << c);
        }
        nodes.corners[n] = mask;
        qefs.solve(h, lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell, cell, bias, vout);
        nodes.px[n] = vout[0]; nodes.py[n] = vout[1]; nodes.pz[n] = vout[2];
        grad(vout[0], vout[1], vout[2], nrm);
        nodes.nx[n] = nrm[0]; nodes.ny[n] = nrm[1]; nodes.nz[n] = nrm[2];
        level.set(key(i, j, k), n);
      }
    }
  }

  // ---- 2. build parents bottom-up until one node covers everything ----
  //
  // The index space is the smallest power-of-two cube that contains the grid. Cells outside the grid
  // simply have no node, which costs nothing — the tree is sparse, so a cubic index space over a
  // non-cubic grid wastes index range rather than memory or field evaluations.
  let span = 1;
  while (span < Math.max(nxC, nyC, nzC)) span *= 2;

  let root = -1;
  for (let size = 1; size < span; size *= 2) {
    const parents = new Map();
    for (const [, child] of level) {
      const pi = Math.floor(nodes.minI[child] / (size * 2)) * (size * 2);
      const pj = Math.floor(nodes.minJ[child] / (size * 2)) * (size * 2);
      const pk = Math.floor(nodes.minK[child] / (size * 2)) * (size * 2);
      const pk3 = `${pi},${pj},${pk}`;
      let p = parents.get(pk3);
      if (p === undefined) {
        p = nodes.alloc(INTERNAL, pi, pj, pk, size * 2);
        parents.set(pk3, p);
      }
      // Which of the eight slots this child occupies, in CHILD_OFFSETS order.
      const ci = (nodes.minI[child] - pi) / size;
      const cj = (nodes.minJ[child] - pj) / size;
      const ck = (nodes.minK[child] - pk) / size;
      nodes.children[p * 8 + ((ci << 2) | (cj << 1) | ck)] = child;
    }
    level = parents;
  }
  // One entry left is the root; a surface that fits in a single cell leaves the leaf itself.
  for (const [, n] of level) root = n;
  if (root === -1) {
    return emptyResult(grid, error);
  }

  // ---- 3. simplify, then contour, once per error budget ----
  //
  // The budgets are applied in INCREASING order against the same tree. Simplification is monotone — a
  // node collapsible at a small threshold is collapsible at a larger one — so each level starts from
  // the previous level's tree instead of rebuilding, and the whole LOD chain costs barely more than its
  // coarsest member. That is the real payoff of octree simplification over decimation: the levels are
  // nested by construction, so they cannot pop in a way that reveals different topology.
  const budgets = Array.isArray(error) ? [...error].sort((a, b) => a - b) : [error];
  const meshes = budgets.map((budget) => {
    // Squared, once, here: the QEF residual is in squared world units, so a budget expressed as a
    // LENGTH — which is the only unit a caller can reason about — has to be squared to compare.
    simplify(nodes, qefs, root, budget * budget, lo, cell, bias, grad, vout, nrm, d, cornerIdx);
    return contour(nodes, root, grid);
  });

  return Array.isArray(error) ? meshes : meshes[0];
}

function emptyResult(grid, error) {
  const one = {
    positions: new Float32Array(0),
    normals: new Float32Array(0),
    indices: new Uint32Array(0),
    vertexCount: 0,
    triangleCount: 0,
    cells: [grid.nxC, grid.nyC, grid.nzC],
    cellSize: grid.cell,
    fieldSamples: grid.fieldSamples,
  };
  return Array.isArray(error) ? error.map(() => one) : one;
}

/**
 * Collapse any node whose eight children one vertex can stand in for.
 *
 * Depth first, so a node is only considered once its children have been. A node with a surviving
 * INTERNAL child is not collapsible at all — collapsing across two levels at once would discard the
 * topology of the level in between.
 *
 * The sign mask of a collapsed node is rebuilt from its children rather than resampled from the field.
 * That matters: the contouring tables read corner signs to decide whether an edge is crossed and which
 * way the quad faces, and the child that owns a corner is the authority on it. Resampling would be
 * subtly different at exactly the cells where a thin feature was merged away — the field says the
 * corner is outside, the mesh that survived says it is inside — and the quads would face the wrong way
 * there. Corners no child claims inherit the cell centre's sign, which every child agrees on.
 */
function simplify(nodes, qefs, node, threshold, lo, cell, bias, grad, vout, nrm, d, cornerIdx) {
  if (node < 0 || nodes.type[node] !== INTERNAL) return node;

  const signs = [-1, -1, -1, -1, -1, -1, -1, -1];
  let collapsible = true;

  for (let i = 0; i < 8; i++) {
    const c = simplify(nodes, qefs, nodes.children[node * 8 + i], threshold,
                       lo, cell, bias, grad, vout, nrm, d, cornerIdx);
    nodes.children[node * 8 + i] = c;
    if (c < 0) continue;
    if (nodes.type[c] === INTERNAL) { collapsible = false; continue; }
    signs[i] = (nodes.corners[c] >> i) & 1;
  }
  if (!collapsible) return node;

  // The merged QEF: the exact sum of the children's, which is the whole reason for the normal-equation
  // form. Accumulated into a shared SCRATCH slot first, because most candidates are rejected and a
  // fresh allocation per attempt would grow the pool by every rejection at every LOD level. The
  // children are never touched either way, so a larger budget can try again from the same state.
  const trial = qefs.scratch;
  qefs.clear(trial);
  for (let i = 0; i < 8; i++) {
    const c = nodes.children[node * 8 + i];
    if (c >= 0) qefs.addQef(trial, nodes.qef[c]);
  }
  if (qefs.count(trial) === 0) return node;

  const size = nodes.size[node];
  const ox = lo[0] + nodes.minI[node] * cell;
  const oy = lo[1] + nodes.minJ[node] * cell;
  const oz = lo[2] + nodes.minK[node] * cell;
  qefs.solve(trial, ox, oy, oz, cell * size, bias, vout);
  if (qefs.error(trial, vout[0], vout[1], vout[2]) > threshold) return node;

  // Corners a surviving child claims come from that child — it is the authority on the surface that
  // actually remains there, and resampling the field would disagree with it exactly where a thin
  // feature was merged away. Corners no child claims lie in a region the surface never entered, so the
  // field's own sign is both available and exact; the usual trick of copying the cell centre's sign is
  // a guess where a fact is on hand.
  let mask = 0;
  for (let i = 0; i < 8; i++) {
    let bit = signs[i];
    if (bit === -1) {
      const [ci, cj, ck] = CHILD_OFFSETS[i];
      const gi = nodes.minI[node] + ci * nodes.size[node];
      const gj = nodes.minJ[node] + cj * nodes.size[node];
      const gk = nodes.minK[node] + ck * nodes.size[node];
      bit = d[cornerIdx(gi, gj, gk)] > 0 ? 0 : 1;
    }
    mask |= (bit << i);
  }
  // Refuse a collapse that would weld two surface sheets into one vertex. See SINGLE_SHEET.
  if (!SINGLE_SHEET[mask]) return node;

  // Accepted: give it a permanent slot. Re-accumulating is eight adds of fourteen floats, which is
  // cheaper than having carried a per-attempt allocation for every rejection to get here.
  const merged = qefs.alloc();
  for (let i = 0; i < 8; i++) {
    const c = nodes.children[node * 8 + i];
    if (c >= 0) qefs.addQef(merged, nodes.qef[c]);
  }

  nodes.type[node] = LEAF;
  nodes.corners[node] = mask;
  nodes.qef[node] = merged;
  nodes.px[node] = vout[0]; nodes.py[node] = vout[1]; nodes.pz[node] = vout[2];
  // The FIELD's gradient at the merged vertex, not the average of the children's normals. The children's
  // normals belong to points up to a cell away and averaging them rounds off exactly the feature the
  // merge just kept; six evaluations here is a rounding error against the corner grid.
  grad(vout[0], vout[1], vout[2], nrm);
  nodes.nx[node] = nrm[0]; nodes.ny[node] = nrm[1]; nodes.nz[node] = nrm[2];
  for (let i = 0; i < 8; i++) nodes.children[node * 8 + i] = -1;
  return node;
}

/** Walk the tree, number the surviving vertices, and weave the quads. */
function contour(nodes, root, grid) {
  const positions = new FloatList();
  const normals = new FloatList();
  const indices = new IntList();
  let vertexCount = 0;

  // Number every leaf that survived, in tree order. Done as its own pass so an index is available
  // before any quad that references it is emitted.
  const number = (n) => {
    if (n < 0) return;
    if (nodes.type[n] === LEAF) {
      nodes.vertexIndex[n] = vertexCount++;
      positions.push3(nodes.px[n], nodes.py[n], nodes.pz[n]);
      normals.push3(nodes.nx[n], nodes.ny[n], nodes.nz[n]);
      return;
    }
    for (let i = 0; i < 8; i++) number(nodes.children[n * 8 + i]);
  };
  number(root);

  const ctx = { nodes, indices, triangles: 0 };
  cellProc(ctx, root);

  return {
    positions: positions.trimmed(),
    normals: normals.trimmed(),
    indices: indices.trimmed(),
    vertexCount,
    triangleCount: ctx.triangles,
    cells: [grid.nxC, grid.nyC, grid.nzC],
    cellSize: grid.cell,
    fieldSamples: grid.fieldSamples,
  };
}

/**
 * Emit the quad for one minimal edge, shared by four cells.
 *
 * THE SMALLEST OF THE FOUR CELLS OWNS THE DECISION. Where cells of different sizes meet, only the
 * smallest resolves the edge finely enough for its corner signs to be authoritative; a larger
 * neighbour's corners straddle the feature and can disagree. Taking the sign — and therefore the
 * winding — from the smallest is what makes the seam between two levels of detail consistent instead of
 * a row of flipped triangles.
 */
function processEdge(ctx, n0, n1, n2, n3, dir) {
  const { nodes } = ctx;
  const quad = [n0, n1, n2, n3];
  let minSize = Infinity;
  let minIndex = 0;
  let flip = false;
  const signChange = [false, false, false, false];

  for (let i = 0; i < 4; i++) {
    const edge = PROCESS_EDGE_MASK[dir][i];
    const c0 = EDGE_VMAP[edge][0];
    const c1 = EDGE_VMAP[edge][1];
    const m0 = (nodes.corners[quad[i]] >> c0) & 1;
    const m1 = (nodes.corners[quad[i]] >> c1) & 1;
    if (nodes.size[quad[i]] < minSize) {
      minSize = nodes.size[quad[i]];
      minIndex = i;
      flip = m1 !== 1;
    }
    signChange[i] = m0 !== m1;
  }
  if (!signChange[minIndex]) return;

  const i0 = nodes.vertexIndex[quad[0]];
  const i1 = nodes.vertexIndex[quad[1]];
  const i2 = nodes.vertexIndex[quad[2]];
  const i3 = nodes.vertexIndex[quad[3]];
  if (i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0) return;

  // A TRIANGLE WITH A REPEATED INDEX IS DROPPED. Not the same thing as the uniform contourer's zero-area
  // triangles, which it deliberately keeps — and the difference is the whole reason this is safe here and
  // not there.
  //
  // There, a degenerate triangle has three DISTINCT vertices that happen to coincide in space, and it is
  // still the second user of a real edge its neighbour owns; dropping it opens a boundary. Here, adaptive
  // refinement puts the SAME node on two sides of a minimal edge, so the quad is genuinely a triangle and
  // the extra one has a repeated index. Its only edges are a self-loop and a doubling of an edge already
  // inside it — it is the second user of nothing. Keeping it is what makes an otherwise clean mesh report
  // edges with four users.
  //
  // Measured on a 32-cell sphere at a 0.01 budget: 96 of 2364, and dropping them takes the edge-use
  // histogram to entirely twos and the Euler characteristic from 50 to 2. Watertightness is unaffected.
  const emit = (a, b, c) => {
    if (a === b || b === c || a === c) return;
    ctx.indices.push3(a, b, c);
    ctx.triangles++;
  };
  if (!flip) {
    emit(i0, i1, i3);
    emit(i0, i3, i2);
  } else {
    emit(i0, i3, i1);
    emit(i0, i2, i3);
  }
}

function edgeProc(ctx, n0, n1, n2, n3, dir) {
  if (n0 < 0 || n1 < 0 || n2 < 0 || n3 < 0) return;
  const { nodes } = ctx;
  const quad = [n0, n1, n2, n3];
  if (nodes.type[n0] !== INTERNAL && nodes.type[n1] !== INTERNAL
      && nodes.type[n2] !== INTERNAL && nodes.type[n3] !== INTERNAL) {
    processEdge(ctx, n0, n1, n2, n3, dir);
    return;
  }
  for (let i = 0; i < 2; i++) {
    const m = EDGE_EDGE_MASK[dir][i];
    const sub = [0, 0, 0, 0];
    for (let j = 0; j < 4; j++) {
      sub[j] = nodes.type[quad[j]] !== INTERNAL ? quad[j] : nodes.children[quad[j] * 8 + m[j]];
    }
    edgeProc(ctx, sub[0], sub[1], sub[2], sub[3], m[4]);
  }
}

function faceProc(ctx, n0, n1, dir) {
  if (n0 < 0 || n1 < 0) return;
  const { nodes } = ctx;
  if (nodes.type[n0] !== INTERNAL && nodes.type[n1] !== INTERNAL) return;

  const pair = [n0, n1];
  for (let i = 0; i < 4; i++) {
    const m = FACE_FACE_MASK[dir][i];
    const a = nodes.type[n0] !== INTERNAL ? n0 : nodes.children[n0 * 8 + m[0]];
    const b = nodes.type[n1] !== INTERNAL ? n1 : nodes.children[n1 * 8 + m[1]];
    faceProc(ctx, a, b, m[2]);
  }
  for (let i = 0; i < 4; i++) {
    const m = FACE_EDGE_MASK[dir][i];
    const order = FACE_EDGE_ORDERS[m[0]];
    const sub = [0, 0, 0, 0];
    for (let j = 0; j < 4; j++) {
      const src = pair[order[j]];
      sub[j] = nodes.type[src] !== INTERNAL ? src : nodes.children[src * 8 + m[1 + j]];
    }
    edgeProc(ctx, sub[0], sub[1], sub[2], sub[3], m[5]);
  }
}

function cellProc(ctx, node) {
  if (node < 0 || ctx.nodes.type[node] !== INTERNAL) return;
  const { nodes } = ctx;
  for (let i = 0; i < 8; i++) cellProc(ctx, nodes.children[node * 8 + i]);
  for (let i = 0; i < 12; i++) {
    const m = CELL_FACE_MASK[i];
    faceProc(ctx, nodes.children[node * 8 + m[0]], nodes.children[node * 8 + m[1]], m[2]);
  }
  for (let i = 0; i < 6; i++) {
    const m = CELL_EDGE_MASK[i];
    edgeProc(ctx,
      nodes.children[node * 8 + m[0]], nodes.children[node * 8 + m[1]],
      nodes.children[node * 8 + m[2]], nodes.children[node * 8 + m[3]], m[4]);
  }
}
