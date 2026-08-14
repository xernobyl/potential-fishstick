/**
 * The sampling stage every contourer shares: a distance grid, and the surface crossings on it.
 *
 * This is the expensive half of meshing and it is identical whether the result is woven at one uniform
 * resolution or adaptively across an octree. Extracting it means the two weavers differ only in the part
 * that is genuinely different, and — more usefully — that the adaptive weaver cannot accidentally sample
 * the field differently from the reference implementation it is checked against.
 *
 * ONE FIELD EVALUATION PER GRID CORNER. Not per cell: eight cells share every corner, so evaluating per
 * cell would be an eightfold waste of the single most expensive operation in the build. The crossings
 * then cost a handful more evaluations each, but only on edges that actually change sign, which is a
 * surface-area-sized set rather than a volume-sized one.
 */

/** A growable Float64 list. Avoids the per-push allocation an array of arrays would do. */
export class FloatList {
  constructor(capacity = 4096) {
    this.a = new Float64Array(capacity);
    this.n = 0;
  }

  push3(x, y, z) {
    if (this.n + 3 > this.a.length) {
      const grown = new Float64Array(this.a.length * 2);
      grown.set(this.a);
      this.a = grown;
    }
    this.a[this.n++] = x;
    this.a[this.n++] = y;
    this.a[this.n++] = z;
  }

  trimmed() { return new Float32Array(this.a.subarray(0, this.n)); }
}

/** The same, for indices. */
export class IntList {
  constructor(capacity = 4096) {
    this.a = new Uint32Array(capacity);
    this.n = 0;
  }

  push3(x, y, z) {
    if (this.n + 3 > this.a.length) {
      const grown = new Uint32Array(this.a.length * 2);
      grown.set(this.a);
      this.a = grown;
    }
    this.a[this.n++] = x;
    this.a[this.n++] = y;
    this.a[this.n++] = z;
  }

  trimmed() { return new Uint32Array(this.a.subarray(0, this.n)); }
}

/**
 * The 12 edges of a cell, as (corner offset i, j, k, axis).
 *
 * Shared because the leaf cells of the octree are the cells of the uniform grid — the adaptive weaver
 * builds its finest level from exactly this, so a difference here would be a difference in the thing
 * being compared.
 */
export const CELL_EDGES = [
  [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0],
  [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1],
  [0, 0, 0, 2], [1, 0, 0, 2], [0, 1, 0, 2], [1, 1, 0, 2],
];

/**
 * Sample the field on a padded, isotropic grid.
 *
 * Cells are cubes sized off the LONGEST axis. Anisotropic cells would make the QEF's error metric
 * axis-dependent and tilt every vertex it places, which shows up as a shape that is subtly sheared
 * rather than as anything that looks like a bug.
 *
 * @param {(x:number,y:number,z:number)=>number} f  compiled field
 * @param {{bounds:{min:number[],max:number[]}, resolution:number, pad?:number}} opts
 */
export function sampleGrid(f, { bounds, resolution, pad = 2 }) {
  const span = [0, 1, 2].map((i) => bounds.max[i] - bounds.min[i]);
  const longest = Math.max(span[0], span[1], span[2]);
  const cell = longest / resolution;

  // Pad outward so a surface exactly on the bound still has a cell to be found in.
  const lo = [0, 1, 2].map((i) => bounds.min[i] - pad * cell);
  const nCells = [0, 1, 2].map((i) => Math.ceil((span[i] + 2 * pad * cell) / cell));
  const [nxC, nyC, nzC] = nCells;
  const cw = nxC + 1;
  const ch = nyC + 1;
  const cd = nzC + 1;

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

  return {
    d, lo, cell, cw, ch, cd, nxC, nyC, nzC,
    cornerIdx: (i, j, k) => (k * ch + j) * cw + i,
    cellIdx: (i, j, k) => (k * nyC + j) * nxC + i,
    fieldSamples: d.length,
  };
}

/**
 * Build a gradient estimator on this grid: central differences at a fraction of a cell.
 *
 * At the crossing rather than at a corner, because the gradient varies across a cell and the QEF's plane
 * is only as good as the normal it is given. The epsilon is small enough to be local and large enough to
 * stay out of the field's own noise floor.
 */
function gradientAt(f, cell) {
  const eps = cell * 0.25;
  return (x, y, z, out) => {
    const gx = f(x + eps, y, z) - f(x - eps, y, z);
    const gy = f(x, y + eps, z) - f(x, y - eps, z);
    const gz = f(x, y, z + eps) - f(x, y, z - eps);
    const l = Math.hypot(gx, gy, gz) || 1;
    out[0] = gx / l; out[1] = gy / l; out[2] = gz / l;
  };
}

/**
 * Find the surface crossing on every sign-changing grid edge.
 *
 * Sparse: edge ids are (corner, axis), three per corner, and only the ones that change sign get an entry
 * in the compact position/normal lists. On a closed surface that is a few percent of the edges.
 *
 * @param {(x:number,y:number,z:number)=>number} f  compiled field
 * @param {object} grid  from `sampleGrid`
 * @param {(x:number,y:number,z:number,out:number[])=>boolean} [analytic]  analytic normal, or null
 * @returns {{edgeMap:Int32Array, exs:FloatList, ens:FloatList, crossings:number}}
 */
export function findCrossings(f, grid, analytic = null) {
  const { d, lo, cell, cw, ch, cd, cornerIdx } = grid;
  const edgeMap = new Int32Array(cw * ch * cd * 3).fill(-1);
  const exs = new FloatList();
  const ens = new FloatList();
  const grad = gradientAt(f, cell);
  const nrm = [0, 0, 0];
  let crossings = 0;

  const addCrossing = (i, j, k, axis, di, dj, dk) => {
    const a = d[cornerIdx(i, j, k)];
    const b = d[cornerIdx(i + di, j + dj, k + dk)];
    if ((a > 0) === (b > 0)) return;
    // Linear interpolation, then ONE bisection step. Linear alone is exact for a planar field and off by
    // a fraction of a cell on a curved one; a single refinement halves that for one extra evaluation,
    // which is the cheapest accuracy in the whole build.
    let t = a / (a - b);
    const x0 = lo[0] + i * cell;
    const y0 = lo[1] + j * cell;
    const z0 = lo[2] + k * cell;
    let x = x0 + di * cell * t;
    let y = y0 + dj * cell * t;
    let z = z0 + dk * cell * t;
    const mid = f(x, y, z);
    if ((mid > 0) === (a > 0)) t += (1 - t) * 0.5 * (Math.abs(mid) / (Math.abs(mid) + Math.abs(b)));
    else t -= t * 0.5 * (Math.abs(mid) / (Math.abs(mid) + Math.abs(a)));
    x = x0 + di * cell * t;
    y = y0 + dj * cell * t;
    z = z0 + dk * cell * t;

    // Analytic normal where one exists, else central differences. The analytic path keeps a sharp
    // crease sharp; the fallback covers smooth blends and rounded shapes, whose blended normal is only
    // correct as the field's own gradient.
    if (!(analytic && analytic(x, y, z, nrm))) grad(x, y, z, nrm);
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

  return { edgeMap, exs, ens, crossings, grad };
}
