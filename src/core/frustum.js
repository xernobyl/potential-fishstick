/**
 * View frustum extraction and sphere culling.
 *
 * Planes come straight out of the view-projection matrix (Gribb & Hartmann): a point is inside the
 * frustum when its clip coordinates satisfy the clip-volume inequalities, and each inequality rearranges
 * into a plane in WORLD space by combining rows of the matrix. No inverse, no corner unprojection, and
 * it works for any projection the matrix can express.
 *
 * FIVE PLANES, NOT SIX, and that is this renderer's projection rather than an approximation. The
 * projection is reverse-Z and INFINITE: clip z runs from 1 at the near plane toward 0 at infinity, so
 * there is no far plane to extract — the inequality it would come from is `z >= 0`, which every point in
 * front of the camera satisfies. Extracting it anyway yields a degenerate plane with a zero normal, and
 * testing against that rejects everything or nothing depending on the sign of a rounding error.
 *
 * WHY SPHERES AND NOT BOXES. Everything cullable here rotates: the rings precess, the ship flies. A
 * sphere is rotation-invariant, so one radius computed at build time stays correct forever, while an AABB
 * has to be re-fitted every frame from the current orientation — and a re-fitted AABB of a rotating
 * object is looser than the sphere it replaced. An AABB earns its place for axis-aligned static geometry,
 * which this scene has none of.
 *
 * THE HONEST PAYOFF, stated because it would be easy to oversell: with four objects that are almost
 * always on screen, this rejects almost nothing today. What it is really for is the per-object machinery
 * — ranges, bounds, one draw per visible object — which is the same machinery LOD selection needs, and
 * the point at which a scene gains enough objects for culling to matter is not the point at which you
 * want to be retrofitting it.
 */

/** Column-major index, matching scene/mat4.js. */
const M = (c, r) => c * 4 + r;

/**
 * Extract world-space frustum planes from a view-projection matrix.
 *
 * Each plane is (a, b, c, d) with a unit normal pointing INWARD, so a sphere at `p` with radius `r` is
 * outside when `dot(n, p) + d < -r`. Normalised, so that comparison is in world units and the same
 * epsilon means the same thing at every distance.
 *
 * @param {Float32Array} vp  column-major 4x4 view-projection matrix
 * @param {Float32Array} out 5 planes x 4 floats, must be length >= 20. Reused across frames.
 * @returns {Float32Array} `out`
 */
export function extractFrustum(vp, out) {
  // Left:   x >= -w  ->  (row3 + row0) . p >= 0
  // Right:  x <=  w  ->  (row3 - row0) . p >= 0
  // Bottom: y >= -w  ->  (row3 + row1) . p >= 0
  // Top:    y <=  w  ->  (row3 - row1) . p >= 0
  // Near:   z <=  w  ->  (row3 - row2) . p >= 0
  //
  // That last one is the reverse-Z near plane: z = w at the near plane and falls toward 0 with distance,
  // so `w - z >= 0` is "at or beyond near". The forward-Z convention's `z >= 0` far plane does not exist
  // here, which is why there are five.
  const rows = [
    [vp[M(0, 3)] + vp[M(0, 0)], vp[M(1, 3)] + vp[M(1, 0)], vp[M(2, 3)] + vp[M(2, 0)], vp[M(3, 3)] + vp[M(3, 0)]],
    [vp[M(0, 3)] - vp[M(0, 0)], vp[M(1, 3)] - vp[M(1, 0)], vp[M(2, 3)] - vp[M(2, 0)], vp[M(3, 3)] - vp[M(3, 0)]],
    [vp[M(0, 3)] + vp[M(0, 1)], vp[M(1, 3)] + vp[M(1, 1)], vp[M(2, 3)] + vp[M(2, 1)], vp[M(3, 3)] + vp[M(3, 1)]],
    [vp[M(0, 3)] - vp[M(0, 1)], vp[M(1, 3)] - vp[M(1, 1)], vp[M(2, 3)] - vp[M(2, 1)], vp[M(3, 3)] - vp[M(3, 1)]],
    [vp[M(0, 3)] - vp[M(0, 2)], vp[M(1, 3)] - vp[M(1, 2)], vp[M(2, 3)] - vp[M(2, 2)], vp[M(3, 3)] - vp[M(3, 2)]],
  ];

  for (let i = 0; i < 5; i++) {
    const [a, b, c, d] = rows[i];
    // Normalising is what makes `d` a distance. An unnormalised plane still classifies points correctly
    // by sign, but the magnitude is scaled by the row's length, so a radius could not be compared
    // against it — which is the whole point of a sphere test.
    const len = Math.hypot(a, b, c) || 1;
    out[i * 4] = a / len;
    out[i * 4 + 1] = b / len;
    out[i * 4 + 2] = c / len;
    out[i * 4 + 3] = d / len;
  }
  return out;
}

/**
 * Is a world-space sphere at least partly inside?
 *
 * Conservative in the safe direction: a sphere straddling a plane counts as visible. Rejects only when
 * the sphere is entirely on the outside of some single plane, which is the standard test and can keep a
 * sphere that is outside the frustum but outside no single plane — a corner case that costs a draw call
 * and never drops geometry.
 */
export function sphereVisible(planes, cx, cy, cz, radius) {
  for (let i = 0; i < 5; i++) {
    const d = planes[i * 4] * cx + planes[i * 4 + 1] * cy + planes[i * 4 + 2] * cz + planes[i * 4 + 3];
    if (d < -radius) return false;
  }
  return true;
}
