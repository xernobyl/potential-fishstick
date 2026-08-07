/**
 * Signed distance fields as DATA, and a compiler that turns them into a closure.
 *
 * A shape is a tree of plain objects — `sphere(0.3)`, `smoothUnion(0.1, a, b)`, `translate([x,y,z], c)`
 * — with no behaviour attached. That separation is what makes the rest possible: the same tree can be
 * meshed on the CPU (see `dualcontour.js`), inspected, serialised, diffed between versions, or later
 * emitted as a flat instruction buffer for a compute shader. A tree of closures could not be any of
 * those things.
 *
 * COMPILED, NOT INTERPRETED. `compile(node)` walks the tree ONCE and returns `(x, y, z) => distance`
 * built from nested closures. A tree-walking evaluator would re-dispatch on `node.op` at every sample,
 * and a mesher takes hundreds of thousands of samples — at 64^3 the corner grid alone is 274625 of
 * them, each touching every node in the tree. Specialising away the dispatch is the difference between
 * a mesh that builds in milliseconds and one that stalls startup.
 *
 * WHY NOT STRING CODEGEN. Emitting JS source and `new Function`-ing it is faster still, and it is what
 * this should become if the field ever gets big. It is not here yet: closure composition already puts
 * the ship's build under a frame, and generated source cannot be stepped through in a debugger or run
 * under a CSP that forbids eval — which this page, being a static file on Pages, would rather keep.
 *
 * DISTANCES ARE EXACT WHERE THEY CAN BE. Every primitive here is a true Euclidean distance (Quilez's
 * formulations for the box and the capped cylinder, which are exact rather than bounds) because dual
 * contouring interpolates along edges and differentiates for normals: a field that only bounds the
 * distance places vertices slightly wrong and tilts every normal. The one exception is documented on
 * `scale`, and the smooth operators are exact-ish by construction rather than by proof.
 */

// ---- primitives ------------------------------------------------------------------------------
//
// Each returns a node whose `f` is a plain function of a point. Keeping `f` on the node rather than in
// a switch in the compiler means adding a primitive is adding one object, and the compiler never
// grows.

/** @param {number} r radius */
export const sphere = (r) => ({ op: 'sphere', r, f: (x, y, z) => Math.hypot(x, y, z) - r });

/**
 * Axis-aligned box of HALF-extents `h`. Quilez's exact formulation: the outside distance is the length
 * of the positive part of |p| - h, and the inside distance is the largest (least negative) component,
 * which the max with 0 and the min with 0 select between without a branch.
 */
export const box = (h) => ({
  op: 'box',
  h,
  f: (x, y, z) => {
    const qx = Math.abs(x) - h[0];
    const qy = Math.abs(y) - h[1];
    const qz = Math.abs(z) - h[2];
    const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
    const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
    return outside + inside;
  },
});

/** Box with rounded edges of radius `r`. A box inflated by r, which is exactly what subtracting r does. */
export const roundBox = (h, r) => {
  const b = box([h[0] - r, h[1] - r, h[2] - r]);
  return { op: 'roundBox', h, r, f: (x, y, z) => b.f(x, y, z) - r };
};

/** Capped cylinder along Y: radius `r`, half-height `hh`. Exact, including the rim. */
export const cylinder = (r, hh) => ({
  op: 'cylinder',
  r,
  hh,
  f: (x, y, z) => {
    const dx = Math.hypot(x, z) - r;
    const dy = Math.abs(y) - hh;
    const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
    return Math.min(Math.max(dx, dy), 0) + outside;
  },
});

/** Infinite plane through the origin with unit normal `n`. Useful as a cutter. */
export const plane = (n) => ({ op: 'plane', n, f: (x, y, z) => x * n[0] + y * n[1] + z * n[2] });

/**
 * Cone-ish tapered cylinder along Y, from radius `r0` at -hh to `r1` at +hh.
 *
 * A cone's exact distance needs the segment-to-cone-surface case analysis; this is the standard
 * conservative form, exact on the lateral surface and on the caps, and a slight UNDER-estimate near
 * the rim where the two meet. Under-estimating is the safe direction for a mesher: it moves an edge
 * crossing by less than a cell rather than skipping a sign change.
 */
export const cone = (r0, r1, hh) => ({
  op: 'cone',
  r0,
  r1,
  hh,
  f: (x, y, z) => {
    const q = Math.hypot(x, z);
    const t = Math.min(Math.max((y + hh) / (2 * hh), 0), 1);
    const r = r0 + (r1 - r0) * t;
    const dLat = (q - r) * (2 * hh) / Math.hypot(2 * hh, r1 - r0);
    const dCap = Math.abs(y) - hh;
    const outside = Math.hypot(Math.max(dLat, 0), Math.max(dCap, 0));
    return Math.min(Math.max(dLat, dCap), 0) + outside;
  },
});

// ---- combinators -----------------------------------------------------------------------------

export const union = (...kids) => ({ op: 'union', kids });
export const intersect = (...kids) => ({ op: 'intersect', kids });
/** a minus b. */
export const subtract = (a, b) => ({ op: 'subtract', kids: [a, b] });

/**
 * Polynomial smooth minimum (Quilez). `k` is the blend WIDTH in world units, so it means the same
 * thing regardless of the scale of the shapes being joined — which is why it is expressed this way
 * rather than as the exponential variant's rate.
 *
 * Smooth operators are not exact distances: the blend region reports slightly short. That is
 * deliberate and harmless here for the same reason the cone's under-estimate is.
 */
export const smoothUnion = (k, ...kids) => ({ op: 'smoothUnion', k, kids });
export const smoothSubtract = (k, a, b) => ({ op: 'smoothSubtract', k, kids: [a, b] });

// ---- transforms ------------------------------------------------------------------------------

export const translate = (t, child) => ({ op: 'translate', t, kids: [child] });

/**
 * Rotation from an explicit orthonormal basis, given as three world-space axes.
 *
 * A basis rather than Euler angles: the inverse of an orthonormal basis is its transpose, so the
 * point-to-local transform is three dot products with no matrix inversion and no angle-order
 * convention to get wrong. Callers that think in angles can build the basis once.
 */
export const rotate = (ax, ay, az, child) => ({ op: 'rotate', ax, ay, az, kids: [child] });

/**
 * UNIFORM scale only, and that restriction is not laziness.
 *
 * Dividing a distance by a uniform factor rescales it exactly. A non-uniform scale does not: the
 * result is no longer a distance in any metric, so edge interpolation and gradient normals both go
 * wrong, and the error is largest exactly where the shape is most stretched. Non-uniform shapes are
 * built by giving the PRIMITIVE different extents instead.
 */
export const scale = (s, child) => ({ op: 'scale', s, kids: [child] });

/** Mirror across a plane through the origin. `axis` is 0, 1 or 2. */
export const mirror = (axis, child) => ({ op: 'mirror', axis, kids: [child] });

/** Inflate (or with a negative r, deflate) by a radius. Exact for any exact child. */
export const round = (r, child) => ({ op: 'round', r, kids: [child] });

// ---- the compiler ----------------------------------------------------------------------------

const smin = (a, b, k) => {
  // Quilez's polynomial smin. h is how far into the blend we are, 0..1.
  const h = Math.min(Math.max(0.5 + 0.5 * (b - a) / k, 0), 1);
  return b * (1 - h) + a * h - k * h * (1 - h);
};

/**
 * Turn a node tree into `(x, y, z) => distance`.
 *
 * Recursion happens ONCE, here. Every returned closure captures its children's closures directly, so
 * evaluating a sample is a chain of calls with no property lookups and no branching on node type.
 */
export function compile(node) {
  if (!node || typeof node !== 'object') throw new Error(`not an SDF node: ${node}`);
  if (node.f) return node.f;                       // a primitive carries its own function

  const kids = (node.kids ?? []).map(compile);
  switch (node.op) {
    case 'union': {
      if (kids.length === 2) {
        const [a, b] = kids;
        return (x, y, z) => Math.min(a(x, y, z), b(x, y, z));
      }
      return (x, y, z) => {
        let d = Infinity;
        for (let i = 0; i < kids.length; i++) d = Math.min(d, kids[i](x, y, z));
        return d;
      };
    }
    case 'intersect': {
      return (x, y, z) => {
        let d = -Infinity;
        for (let i = 0; i < kids.length; i++) d = Math.max(d, kids[i](x, y, z));
        return d;
      };
    }
    case 'subtract': {
      const [a, b] = kids;
      return (x, y, z) => Math.max(a(x, y, z), -b(x, y, z));
    }
    case 'smoothUnion': {
      const k = node.k;
      if (kids.length === 2) {
        const [a, b] = kids;
        return (x, y, z) => smin(a(x, y, z), b(x, y, z), k);
      }
      return (x, y, z) => {
        let d = kids[0](x, y, z);
        for (let i = 1; i < kids.length; i++) d = smin(d, kids[i](x, y, z), k);
        return d;
      };
    }
    case 'smoothSubtract': {
      const [a, b] = kids;
      const k = node.k;
      // Smooth subtraction is a smooth min against the negated cutter, with the sign flipped so the
      // blend bulges outward rather than pinching.
      return (x, y, z) => -smin(-a(x, y, z), b(x, y, z), k);
    }
    case 'translate': {
      const [c] = kids;
      const [tx, ty, tz] = node.t;
      return (x, y, z) => c(x - tx, y - ty, z - tz);
    }
    case 'rotate': {
      const [c] = kids;
      const { ax, ay, az } = node;
      // World -> local by the transpose, which for an orthonormal basis is the inverse.
      return (x, y, z) => c(
        x * ax[0] + y * ax[1] + z * ax[2],
        x * ay[0] + y * ay[1] + z * ay[2],
        x * az[0] + y * az[1] + z * az[2],
      );
    }
    case 'scale': {
      const [c] = kids;
      const s = node.s;
      const inv = 1 / s;
      return (x, y, z) => c(x * inv, y * inv, z * inv) * s;
    }
    case 'mirror': {
      const [c] = kids;
      const a = node.axis;
      if (a === 0) return (x, y, z) => c(Math.abs(x), y, z);
      if (a === 1) return (x, y, z) => c(x, Math.abs(y), z);
      return (x, y, z) => c(x, y, Math.abs(z));
    }
    case 'round': {
      const [c] = kids;
      const r = node.r;
      return (x, y, z) => c(x, y, z) - r;
    }
    default:
      throw new Error(`unknown SDF op: ${node.op}`);
  }
}

/**
 * A conservative world-space bound for a tree, as {min, max}.
 *
 * The mesher needs a box to grid, and asking the caller for one is how a shape ends up silently
 * clipped. Derived rather than sampled: each primitive knows its own extent and each operator knows
 * what it does to a box. Unions widen, intersections could narrow but are left wide (safe), smooth
 * operators add their blend width, and `round` adds its radius.
 */
export function bounds(node) {
  const grow = (b, r) => ({ min: b.min.map((v) => v - r), max: b.max.map((v) => v + r) });
  const merge = (bs) => ({
    min: [0, 1, 2].map((i) => Math.min(...bs.map((b) => b.min[i]))),
    max: [0, 1, 2].map((i) => Math.max(...bs.map((b) => b.max[i]))),
  });

  switch (node.op) {
    case 'sphere': return { min: [-node.r, -node.r, -node.r], max: [node.r, node.r, node.r] };
    case 'box': return { min: node.h.map((v) => -v), max: node.h.slice() };
    case 'roundBox': return { min: node.h.map((v) => -v), max: node.h.slice() };
    case 'cylinder': return { min: [-node.r, -node.hh, -node.r], max: [node.r, node.hh, node.r] };
    case 'cone': {
      const r = Math.max(node.r0, node.r1);
      return { min: [-r, -node.hh, -r], max: [r, node.hh, r] };
    }
    case 'plane':
      throw new Error('plane is unbounded — use it only inside an intersect/subtract');
    case 'union': return merge(node.kids.map(bounds));
    case 'smoothUnion': return grow(merge(node.kids.map(bounds)), node.k);
    // An intersection is bounded by either operand; the first is enough and is safe.
    case 'intersect': return bounds(node.kids[0]);
    case 'subtract': return bounds(node.kids[0]);
    case 'smoothSubtract': return grow(bounds(node.kids[0]), node.k);
    case 'translate': {
      const b = bounds(node.kids[0]);
      return { min: b.min.map((v, i) => v + node.t[i]), max: b.max.map((v, i) => v + node.t[i]) };
    }
    case 'rotate': {
      // The rotated box's extent along each world axis is the sum of |basis component| times the
      // child's half-extents — the standard OBB-to-AABB widening, done from the basis rows.
      const b = bounds(node.kids[0]);
      const c = [0, 1, 2].map((i) => (b.min[i] + b.max[i]) * 0.5);
      const h = [0, 1, 2].map((i) => (b.max[i] - b.min[i]) * 0.5);
      const rows = [node.ax, node.ay, node.az];
      const wc = [0, 1, 2].map((i) => rows[0][i] * c[0] + rows[1][i] * c[1] + rows[2][i] * c[2]);
      const wh = [0, 1, 2].map((i) =>
        Math.abs(rows[0][i]) * h[0] + Math.abs(rows[1][i]) * h[1] + Math.abs(rows[2][i]) * h[2]);
      return { min: [0, 1, 2].map((i) => wc[i] - wh[i]), max: [0, 1, 2].map((i) => wc[i] + wh[i]) };
    }
    case 'scale': {
      const b = bounds(node.kids[0]);
      return { min: b.min.map((v) => v * node.s), max: b.max.map((v) => v * node.s) };
    }
    case 'mirror': {
      const b = bounds(node.kids[0]);
      const out = { min: b.min.slice(), max: b.max.slice() };
      // Mirroring makes the axis symmetric about 0, so the extent is the larger side either way.
      const e = Math.max(Math.abs(b.min[node.axis]), Math.abs(b.max[node.axis]));
      out.min[node.axis] = -e;
      out.max[node.axis] = e;
      return out;
    }
    case 'round': return grow(bounds(node.kids[0]), node.r);
    default: throw new Error(`unknown SDF op in bounds: ${node.op}`);
  }
}
