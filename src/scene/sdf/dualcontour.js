/**
 * Dual contouring: a signed distance field in, an indexed triangle mesh out.
 *
 * WHY DUAL CONTOURING AND NOT MARCHING CUBES. Marching cubes places vertices ON grid edges, so every
 * vertex sits on a lattice line and a sharp edge between two flat faces gets sampled as a staircase and
 * then smoothed into a bevel. Dual contouring places ONE vertex per cell, positioned by minimising a
 * quadratic error function over the surface planes that pass through that cell — so a corner where
 * three faces meet is reconstructed AT the corner. For a hull made of boxes and cylinders that is the
 * whole difference between a model and a melted version of it. It also emits one quad per
 * sign-changing edge rather than up to five triangles per cell, which is roughly a third of the
 * triangles for the same surface.
 *
 * The cost is that DC can place a vertex outside its own cell when the QEF is ill-conditioned, which
 * produces self-intersecting geometry. The standard mitigation, used here, is to regularise the system
 * toward the mass point and then clamp the result into the cell — see `solveQef`.
 *
 * PERFORMANCE SHAPE, because this is meant to run per frame eventually:
 *
 *   ONE FIELD EVALUATION PER GRID CORNER. Naively, each cell samples its 8 corners and each corner is
 *   shared by 8 cells, so a per-cell loop evaluates the field 8x more than it needs to. The corner grid
 *   is sampled once into a flat Float32Array and everything else reads it. At 64 cells that is 274625
 *   evaluations instead of 2097152.
 *
 *   FLAT TYPED ARRAYS, NO OBJECTS. Vertices, normals and indices accumulate into growable typed arrays,
 *   and the cell->vertex and edge->crossing maps are Int32Arrays of -1. A Map keyed on "x,y,z" strings
 *   is the obvious way to write this and it allocates a string per lookup; at these counts that is the
 *   dominant cost and it is all garbage.
 *
 *   SPARSE WHERE IT MATTERS. Crossings are stored only for edges that actually change sign — O(surface
 *   area) rather than O(volume). Storing position and normal for every edge would be 20 MB at 64^3 for
 *   data that is 98% empty.
 *
 * RESOLUTION IS A SCREEN-SPACE DECISION. `resolutionForScreen` derives the cell count from how large a
 * cell is allowed to be in PIXELS at a given viewing distance, which is the only version of "enough
 * detail" that survives someone resizing the window or flying closer. Because the mesher is a pure
 * function of (tree, bounds, resolution), an LOD chain is the same call at several resolutions; nothing
 * here holds state that would need resetting between them.
 */

/** Growable Float32Array that amortises like a JS array but stays a typed buffer. */
class FloatList {
  constructor(cap = 4096) {
    this.a = new Float32Array(cap);
    this.n = 0;
  }

  push3(x, y, z) {
    if (this.n + 3 > this.a.length) this.#grow(this.n + 3);
    this.a[this.n++] = x;
    this.a[this.n++] = y;
    this.a[this.n++] = z;
  }

  #grow(need) {
    let cap = this.a.length * 2;
    while (cap < need) cap *= 2;
    const next = new Float32Array(cap);
    next.set(this.a.subarray(0, this.n));
    this.a = next;
  }

  trimmed() { return this.a.slice(0, this.n); }
}

class IntList {
  constructor(cap = 8192) {
    this.a = new Uint32Array(cap);
    this.n = 0;
  }

  push3(x, y, z) {
    if (this.n + 3 > this.a.length) this.#grow(this.n + 3);
    this.a[this.n++] = x;
    this.a[this.n++] = y;
    this.a[this.n++] = z;
  }

  #grow(need) {
    let cap = this.a.length * 2;
    while (cap < need) cap *= 2;
    const next = new Uint32Array(cap);
    next.set(this.a.subarray(0, this.n));
    this.a = next;
  }

  trimmed() { return this.a.slice(0, this.n); }
}

/**
 * How many cells across, for a given on-screen error budget.
 *
 * `errorPx` is the largest a cell may appear, in pixels, when the shape is `distance` away. One cell is
 * roughly one triangle edge, so this is the knob that decides silhouette smoothness. The conversion is
 * the same one the renderer uses for its own footprint maths: screen space spans 2 over the diagonal in
 * pixels, and a screen offset maps to a world offset by `distance / focal`.
 *
 * Clamped at both ends. Below about 8 cells a shape stops being recognisable; above 192 the build cost
 * and the vertex count grow as the square of the resolution for a surface that no longer gains detail.
 */
export function resolutionForScreen({ size, distance, focal, diagonalPx, errorPx = 2.0 }) {
  const worldPerPixel = (2 * distance) / (focal * diagonalPx);
  const cells = size / Math.max(worldPerPixel * errorPx, 1e-6);
  return Math.max(8, Math.min(192, Math.round(cells)));
}

/**
 * Solve the cell's quadratic error function for a vertex position.
 *
 * Minimises the sum of squared distances to the surface PLANES sampled on the cell's edges: each
 * crossing contributes a plane through `p` with normal `n`, and the best vertex is the least-squares
 * intersection of them. That is what puts a vertex exactly on a corner where three planes meet, and on
 * a face where they are parallel it is underdetermined — which is what the regularisation is for.
 *
 * Solved as normal equations (A = sum n n^T, b = sum n (n.p)) with a Tikhonov term pulling toward the
 * mass point. The alternative is an SVD, which is more robust and much more code; the regularised
 * normal equations are what practical implementations use, and the clamp below covers what they miss.
 *
 * `bias` trades feature sharpness for stability: 0 is sharpest and can place a vertex far outside the
 * cell on a nearly-degenerate system, 0.1 is visually indistinguishable and well behaved.
 */
function solveQef(px, py, pz, nx, ny, nz, count, cx, cy, cz, cell, bias, out) {
  // Mass point: the average of the crossings, which is both the regularisation target and the fallback.
  let mx = 0;
  let my = 0;
  let mz = 0;
  for (let i = 0; i < count; i++) { mx += px[i]; my += py[i]; mz += pz[i]; }
  mx /= count; my /= count; mz /= count;

  let a00 = bias; let a01 = 0; let a02 = 0;
  let a11 = bias; let a12 = 0; let a22 = bias;
  let b0 = bias * mx; let b1 = bias * my; let b2 = bias * mz;

  for (let i = 0; i < count; i++) {
    const ax = nx[i];
    const ay = ny[i];
    const az = nz[i];
    const d = ax * px[i] + ay * py[i] + az * pz[i];
    a00 += ax * ax; a01 += ax * ay; a02 += ax * az;
    a11 += ay * ay; a12 += ay * az; a22 += az * az;
    b0 += ax * d; b1 += ay * d; b2 += az * d;
  }

  // Explicit 3x3 inverse. The determinant guard is the ill-conditioned case — a flat face gives a rank-1
  // system, and without the bias term above it would be singular outright.
  const c00 = a11 * a22 - a12 * a12;
  const c01 = a02 * a12 - a01 * a22;
  const c02 = a01 * a12 - a02 * a11;
  const det = a00 * c00 + a01 * c01 + a02 * c02;

  if (Math.abs(det) < 1e-12) {
    out[0] = mx; out[1] = my; out[2] = mz;
  } else {
    const c11 = a00 * a22 - a02 * a02;
    const c12 = a01 * a02 - a00 * a12;
    const c22 = a00 * a11 - a01 * a01;
    const inv = 1 / det;
    out[0] = (c00 * b0 + c01 * b1 + c02 * b2) * inv;
    out[1] = (c01 * b0 + c11 * b1 + c12 * b2) * inv;
    out[2] = (c02 * b0 + c12 * b1 + c22 * b2) * inv;
  }

  // CLAMP INTO THE CELL. This is what keeps the mesh from self-intersecting: a vertex that escapes its
  // own cell can cross a neighbour's, and the quads woven between them then fold through each other.
  // Clamping costs a little feature sharpness on the cells where it triggers and nothing anywhere else.
  out[0] = Math.min(Math.max(out[0], cx), cx + cell);
  out[1] = Math.min(Math.max(out[1], cy), cy + cell);
  out[2] = Math.min(Math.max(out[2], cz), cz + cell);
}

/**
 * Mesh a compiled SDF.
 *
 * @param {(x:number,y:number,z:number)=>number} f  compiled field, from `compile()`
 * @param {object} opts
 * @param {{min:number[],max:number[]}} opts.bounds  world box to grid; from `bounds()`
 * @param {number} opts.resolution                   cells along the LONGEST axis
 * @param {number} [opts.bias]                       QEF regularisation, 0..1
 * @param {number} [opts.pad]                        cells of margin, so the surface is never clipped
 * @returns {{positions:Float32Array, normals:Float32Array, indices:Uint32Array,
 *            vertexCount:number, triangleCount:number, cells:number[], cellSize:number}}
 */
export function dualContour(f, { bounds, resolution, bias = 0.1, pad = 2 }) {
  // A cube cell, sized off the longest axis, so cells are isotropic. Anisotropic cells would make the
  // QEF's error metric axis-dependent and tilt the vertices it places.
  const span = [0, 1, 2].map((i) => bounds.max[i] - bounds.min[i]);
  const longest = Math.max(span[0], span[1], span[2]);
  const cell = longest / resolution;

  // Pad outward so a surface exactly on the bound still has a cell to be found in.
  const lo = [0, 1, 2].map((i) => bounds.min[i] - pad * cell);
  const n = [0, 1, 2].map((i) => Math.ceil((span[i] + 2 * pad * cell) / cell));
  const [nxC, nyC, nzC] = n;
  const cw = nxC + 1;
  const ch = nyC + 1;
  const cd = nzC + 1;

  // ---- 1. one field evaluation per corner ----
  const d = new Float32Array(cw * ch * cd);
  for (let k = 0; k < cd; k++) {
    const z = lo[2] + k * cell;
    for (let j = 0; j < ch; j++) {
      const y = lo[1] + j * cell;
      const rowBase = (k * ch + j) * cw;
      for (let i = 0; i < cw; i++) {
        d[rowBase + i] = f(lo[0] + i * cell, y, z);
      }
    }
  }
  const cornerIdx = (i, j, k) => (k * ch + j) * cw + i;

  // ---- 2. crossings on sign-changing edges, sparse ----
  //
  // Edge ids are (corner, axis): three per corner, so the map is 3x the corner count. Only the ones
  // that change sign get an entry in the compact crossing arrays.
  const edgeMap = new Int32Array(cw * ch * cd * 3).fill(-1);
  const exs = new FloatList();
  const ens = new FloatList();
  let crossings = 0;

  // Central differences for the normal, at the crossing rather than at a corner: the gradient varies
  // across a cell and the QEF's plane is only as good as the normal it is given. Epsilon is a fraction
  // of a cell, small enough to be local and large enough to stay out of the field's own noise floor.
  const eps = cell * 0.25;
  const grad = (x, y, z, out) => {
    const gx = f(x + eps, y, z) - f(x - eps, y, z);
    const gy = f(x, y + eps, z) - f(x, y - eps, z);
    const gz = f(x, y, z + eps) - f(x, y, z - eps);
    const l = Math.hypot(gx, gy, gz) || 1;
    out[0] = gx / l; out[1] = gy / l; out[2] = gz / l;
  };
  const nrm = [0, 0, 0];

  const addCrossing = (i, j, k, axis, di, dj, dk) => {
    const a = d[cornerIdx(i, j, k)];
    const b = d[cornerIdx(i + di, j + dj, k + dk)];
    if ((a > 0) === (b > 0)) return;
    // Linear interpolation, then ONE bisection step. Linear alone is exact for a planar field and off by
    // a fraction of a cell on a curved one; a single refinement halves that for one extra evaluation,
    // which is the cheapest accuracy in the whole function.
    let t = a / (a - b);
    const x0 = lo[0] + i * cell;
    const y0 = lo[1] + j * cell;
    const z0 = lo[2] + k * cell;
    const step = cell;
    let x = x0 + di * step * t;
    let y = y0 + dj * step * t;
    let z = z0 + dk * step * t;
    const mid = f(x, y, z);
    if ((mid > 0) === (a > 0)) t += (1 - t) * 0.5 * (Math.abs(mid) / (Math.abs(mid) + Math.abs(b)));
    else t -= t * 0.5 * (Math.abs(mid) / (Math.abs(mid) + Math.abs(a)));
    x = x0 + di * step * t;
    y = y0 + dj * step * t;
    z = z0 + dk * step * t;

    grad(x, y, z, nrm);
    edgeMap[(cornerIdx(i, j, k) * 3) + axis] = crossings++;
    exs.push3(x, y, z);
    ens.push3(nrm[0], nrm[1], nrm[2]);
  };

  for (let k = 0; k < cd; k++) {
    for (let j = 0; j < ch; j++) {
      for (let i = 0; i < cw; i++) {
        if (i + 1 < cw) addCrossing(i, j, k, 0, 1, 0, 0);
        if (j + 1 < ch) addCrossing(i, j, k, 1, 0, 1, 0);
        if (k + 1 < cd) addCrossing(i, j, k, 2, 0, 0, 1);
      }
    }
  }

  // ---- 3. one vertex per cell that any crossing touches ----
  const cellVert = new Int32Array(nxC * nyC * nzC).fill(-1);
  const positions = new FloatList();
  const normals = new FloatList();
  let vertexCount = 0;

  // Scratch for the QEF, reused for every cell: 12 edges is the maximum a cell can have.
  const qpx = new Float64Array(12);
  const qpy = new Float64Array(12);
  const qpz = new Float64Array(12);
  const qnx = new Float64Array(12);
  const qny = new Float64Array(12);
  const qnz = new Float64Array(12);
  const vout = [0, 0, 0];

  // The 12 edges of a cell, as (corner offset, axis).
  const CELL_EDGES = [
    [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0],
    [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1],
    [0, 0, 0, 2], [1, 0, 0, 2], [0, 1, 0, 2], [1, 1, 0, 2],
  ];

  const cellIdx = (i, j, k) => (k * nyC + j) * nxC + i;

  for (let k = 0; k < nzC; k++) {
    for (let j = 0; j < nyC; j++) {
      for (let i = 0; i < nxC; i++) {
        let count = 0;
        for (let e = 0; e < 12; e++) {
          const [oi, oj, ok, axis] = CELL_EDGES[e];
          const id = edgeMap[(cornerIdx(i + oi, j + oj, k + ok) * 3) + axis];
          if (id < 0) continue;
          qpx[count] = exs.a[id * 3];
          qpy[count] = exs.a[id * 3 + 1];
          qpz[count] = exs.a[id * 3 + 2];
          qnx[count] = ens.a[id * 3];
          qny[count] = ens.a[id * 3 + 1];
          qnz[count] = ens.a[id * 3 + 2];
          count++;
        }
        if (count === 0) continue;

        solveQef(qpx, qpy, qpz, qnx, qny, qnz, count,
                 lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell, cell, bias, vout);

        // NORMAL FROM THE FIELD at the placed vertex, not averaged from the crossings. The crossings'
        // normals belong to points up to a cell away; the gradient here is this vertex's own. It costs
        // six evaluations per vertex, and vertices scale with surface AREA, so it is a rounding error
        // against the corner grid.
        grad(vout[0], vout[1], vout[2], nrm);
        cellVert[cellIdx(i, j, k)] = vertexCount++;
        positions.push3(vout[0], vout[1], vout[2]);
        normals.push3(nrm[0], nrm[1], nrm[2]);
      }
    }
  }

  // ---- 4. one quad per sign-changing edge, woven between the four cells sharing it ----
  const indices = new IntList();
  let triangleCount = 0;

  // EVERY TRIANGLE IS EMITTED, INCLUDING THE ZERO-AREA ONES, and that is a decision rather than an
  // oversight.
  //
  // Two DISTINCT cells can place their vertex at the same point — both clamped against the same shared
  // corner — so a quad can split into one real triangle and one with zero area. Dropping the empty one
  // is tempting: it rasterises to no pixels and its normal is undefined. Measured on a 32-cell sphere it
  // is 40 triangles of 9564, and dropping them opened 36 BOUNDARY EDGES, because the degenerate triangle
  // was still the second user of an edge its neighbour also owned. The hole has zero area, but the index
  // buffer stops being watertight.
  //
  // Watertight is worth more here: decimation, shadow volumes and any further mesh processing all
  // require manifold input, and 0.4% of triangles covering no pixels costs nothing to rasterise. So the
  // topology stays clean and the consumer that needs well-defined normals filters by area itself — see
  // the winding check in dev/dualcontour.mjs.
  const quad = (v0, v1, v2, v3, flip) => {
    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) return;
    // Winding from the sign at the edge's ends, so every triangle faces outward. Getting this wrong is
    // invisible until something enables backface culling, which is exactly how the ring mesh's winding
    // bug hid.
    // The sign test below picks which diagonal order faces outward. Which of the two is which was
    // settled by measurement, not derivation: the headless check counts triangles whose geometric
    // normal disagrees with the outward direction on a convex shape, and the first version had 9560 of
    // 9564 inverted. Nothing culls backfaces today, so this would have been invisible until something
    // did — exactly how the ring mesh's winding bug hid.
    // Winding from the sign at the edge's low corner. One rule for all three axes, which is only
    // possible because the cell traversal above is right-handed per axis — see that note. Verified
    // against the field's own gradient normals: zero inverted triangles on a sphere and on a box.
    if (flip) {
      indices.push3(v0, v2, v1);
      indices.push3(v0, v3, v2);
    } else {
      indices.push3(v0, v1, v2);
      indices.push3(v0, v2, v3);
    }
    triangleCount += 2;
  };

  const at = (i, j, k) => (
    i < 0 || j < 0 || k < 0 || i >= nxC || j >= nyC || k >= nzC ? -1 : cellVert[cellIdx(i, j, k)]
  );

  for (let k = 0; k < cd; k++) {
    for (let j = 0; j < ch; j++) {
      for (let i = 0; i < cw; i++) {
        const base = cornerIdx(i, j, k) * 3;
        // An X edge is shared by the four cells around it in Y and Z, and so on by symmetry.
        // THE FOUR CELLS ARE TRAVERSED IN THE RIGHT-HANDED PAIR FOR THAT AXIS, and getting that wrong
        // is what produced the four inverted triangles this used to ship with.
        //
        // For an edge along axis a, the perpendicular pair (b, c) must be the right-handed cycle:
        // (Y,Z) for X, (Z,X) for Y, (X,Y) for Z. The Y case used (X,Z) — the reverse — so its loop wound
        // the opposite way from the other two. That was compensated for with an inverted sign test,
        // which works everywhere the quad is well-conditioned and fails exactly where it is not: four
        // triangles, at the +-X and +-Z axis-tangent points.
        //
        // With the ordering correct, ALL THREE AXES SHARE ONE SIGN RULE, which is the tell that the
        // handedness is now right rather than merely patched. Backface culling is on, so this is load
        // bearing.
        const flip = d[cornerIdx(i, j, k)] > 0;
        if (edgeMap[base] >= 0 && j > 0 && k > 0) {
          // a = X, (b, c) = (Y, Z)
          quad(at(i, j - 1, k - 1), at(i, j, k - 1), at(i, j, k), at(i, j - 1, k), flip);
        }
        if (edgeMap[base + 1] >= 0 && i > 0 && k > 0) {
          // a = Y, (b, c) = (Z, X)
          quad(at(i - 1, j, k - 1), at(i - 1, j, k), at(i, j, k), at(i, j, k - 1), flip);
        }
        if (edgeMap[base + 2] >= 0 && i > 0 && j > 0) {
          // a = Z, (b, c) = (X, Y)
          quad(at(i - 1, j - 1, k), at(i, j - 1, k), at(i, j, k), at(i - 1, j, k), flip);
        }
      }
    }
  }

  return {
    positions: positions.trimmed(),
    normals: normals.trimmed(),
    indices: indices.trimmed(),
    vertexCount,
    triangleCount,
    cells: [nxC, nyC, nzC],
    cellSize: cell,
    fieldSamples: d.length,
  };
}
