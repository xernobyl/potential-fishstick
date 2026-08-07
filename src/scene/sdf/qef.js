/**
 * The quadratic error function, in the one form that can be ADDED.
 *
 * A QEF measures how far a candidate vertex is from the surface planes sampled around it:
 *
 *     E(x) = sum_i ( n_i . (x - p_i) )^2
 *
 * Expanded, that is `x^T A x - 2 x^T b + c` with `A = sum n n^T`, `b = sum n (n.p)` and
 * `c = sum (n.p)^2`. Storing those three accumulators instead of the planes themselves is what makes
 * the whole adaptive scheme possible: **the QEF of a merged cell is the sum of its children's QEFs**,
 * exactly, with no reference to the original planes. A parent octree node can therefore ask "what would
 * my error be if I replaced all eight children with one vertex?" for the cost of adding ten numbers,
 * and the answer is not an estimate.
 *
 * Keeping the planes instead would mean a parent re-solving over every plane beneath it — quadratic in
 * the tree, and the reason naive implementations of this are slow.
 *
 * `c` (btb) is carried only for the residual. Solving needs A and b alone, but `E(x)` at the solution is
 * the number the simplification threshold compares against, and without `c` the residual is off by a
 * constant that varies per cell — which would make one global threshold meaningless.
 *
 * STORED AS A FLAT POOL rather than as objects. There is one of these per surface cell and the ship's
 * hull has thousands; a pool of Float64Arrays keeps them contiguous and allocates once, where an object
 * per cell would put the build in the garbage collector's path for no benefit.
 *
 * Doubles, not floats. `A` is a sum of outer products of unit vectors and is routinely
 * ill-conditioned — a flat face makes it rank one — and the residual is a difference of similar
 * magnitudes. Single precision loses the sign of the residual on nearly-flat cells, which reads as
 * random over-simplification.
 */

/** Floats per QEF: ata(6) + atb(3) + btb(1) + massSum(3) + count(1). */
const STRIDE = 14;

const ATA = 0;      // a00 a01 a02 a11 a12 a22
const ATB = 6;      // b0 b1 b2
const BTB = 9;
const MASS = 10;    // sum of the crossing points
const COUNT = 13;

export class QefPool {
  /** @param {number} capacity  number of QEFs to reserve; grows by doubling */
  constructor(capacity = 1024) {
    this.data = new Float64Array(capacity * STRIDE);
    this.length = 0;
    /**
     * A permanent slot for trial accumulations — see `clear`. Reserved here rather than by the caller so
     * its handle is stable for the pool's whole life and cannot be confused with a real QEF.
     */
    this.scratch = this.alloc();
  }

  /** Reserve one zeroed QEF and return its handle. */
  alloc() {
    const need = (this.length + 1) * STRIDE;
    if (need > this.data.length) {
      const grown = new Float64Array(Math.max(need, this.data.length * 2));
      grown.set(this.data);
      this.data = grown;
    }
    const h = this.length++;
    this.data.fill(0, h * STRIDE, h * STRIDE + STRIDE);
    return h;
  }

  /** Add one surface plane: the plane through `p` with unit normal `n`. */
  addPlane(h, px, py, pz, nx, ny, nz) {
    const a = this.data;
    const o = h * STRIDE;
    const d = nx * px + ny * py + nz * pz;
    a[o + ATA] += nx * nx;
    a[o + ATA + 1] += nx * ny;
    a[o + ATA + 2] += nx * nz;
    a[o + ATA + 3] += ny * ny;
    a[o + ATA + 4] += ny * nz;
    a[o + ATA + 5] += nz * nz;
    a[o + ATB] += nx * d;
    a[o + ATB + 1] += ny * d;
    a[o + ATB + 2] += nz * d;
    a[o + BTB] += d * d;
    a[o + MASS] += px;
    a[o + MASS + 1] += py;
    a[o + MASS + 2] += pz;
    a[o + COUNT] += 1;
  }

  /** Accumulate another QEF into this one. The operation the whole octree simplification rests on. */
  addQef(h, other) {
    const a = this.data;
    const o = h * STRIDE;
    const s = other * STRIDE;
    for (let i = 0; i < STRIDE; i++) a[o + i] += a[s + i];
  }

  /**
   * Zero an existing QEF so it can be reused.
   *
   * Exists for the simplifier's TRIAL accumulations. Most candidate collapses are rejected, and
   * allocating a fresh QEF for each attempt would grow the pool by every rejection across every LOD
   * level — thousands of 112-byte slots that are dead the moment the error test fails. One scratch slot,
   * cleared and refilled, costs nothing and never grows.
   */
  clear(h) {
    this.data.fill(0, h * STRIDE, h * STRIDE + STRIDE);
  }

  /** Number of planes that went in. Zero means the cell never saw the surface. */
  count(h) { return this.data[h * STRIDE + COUNT]; }

  /**
   * Minimise the QEF, with Tikhonov regularisation toward the mass point and a clamp into the cell.
   *
   * `bias` trades feature sharpness for stability: 0 is sharpest and can throw a vertex far outside the
   * cell on a nearly-degenerate system, 0.1 is visually indistinguishable and well behaved. The clamp
   * is what stops a vertex crossing into a neighbour's cell, where the quads woven between them would
   * fold through each other.
   */
  solve(h, cx, cy, cz, cellSize, bias, out) {
    const a = this.data;
    const o = h * STRIDE;
    const n = a[o + COUNT] || 1;
    const mx = a[o + MASS] / n;
    const my = a[o + MASS + 1] / n;
    const mz = a[o + MASS + 2] / n;

    const a00 = a[o + ATA] + bias;
    const a01 = a[o + ATA + 1];
    const a02 = a[o + ATA + 2];
    const a11 = a[o + ATA + 3] + bias;
    const a12 = a[o + ATA + 4];
    const a22 = a[o + ATA + 5] + bias;
    const b0 = a[o + ATB] + bias * mx;
    const b1 = a[o + ATB + 1] + bias * my;
    const b2 = a[o + ATB + 2] + bias * mz;

    // Explicit 3x3 inverse. The determinant guard is the ill-conditioned case — a flat face gives a
    // rank-one system, and without the bias above it would be singular outright.
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

    out[0] = Math.min(Math.max(out[0], cx), cx + cellSize);
    out[1] = Math.min(Math.max(out[1], cy), cy + cellSize);
    out[2] = Math.min(Math.max(out[2], cz), cz + cellSize);
  }

  /**
   * The residual `E(x)` — how badly one vertex at `x` represents every plane that went in.
   *
   * This is the simplification metric. It is in squared world units, so the threshold a caller passes
   * has a physical meaning: `error` is roughly (distance the surface would move)^2 summed over the
   * planes being merged, which is why the LOD budget can be expressed as a length and squared once.
   *
   * The unregularised A and b, deliberately: the bias exists to keep the SOLVE well behaved, and
   * including it here would report the pull toward the mass point as if it were surface error.
   */
  error(h, x, y, z) {
    const a = this.data;
    const o = h * STRIDE;
    const a00 = a[o + ATA];
    const a01 = a[o + ATA + 1];
    const a02 = a[o + ATA + 2];
    const a11 = a[o + ATA + 3];
    const a12 = a[o + ATA + 4];
    const a22 = a[o + ATA + 5];
    // x^T A x
    const ax = a00 * x + a01 * y + a02 * z;
    const ay = a01 * x + a11 * y + a12 * z;
    const az = a02 * x + a12 * y + a22 * z;
    const xax = x * ax + y * ay + z * az;
    // -2 x^T b + c
    const xb = x * a[o + ATB] + y * a[o + ATB + 1] + z * a[o + ATB + 2];
    // Clamped at zero: the true minimum is non-negative, and rounding can push a near-zero residual
    // slightly under, which would read as a perfect fit and collapse a cell that should not.
    return Math.max(0, xax - 2 * xb + a[o + BTB]);
  }
}
