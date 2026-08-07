/**
 * Level-of-detail selection, in the only unit that survives a window resize: PIXELS.
 *
 * A mesh built by the adaptive contourer carries the world-space error it was simplified to (see
 * octree.js). How much that error matters depends entirely on how large the object is on screen — the
 * same 3 mm of surface deviation is invisible on a distant ship and a visible facet on a close one. So
 * selection converts the level's world error into the pixels it would occupy at the object's CURRENT
 * distance, and takes the coarsest level whose error is still under a stated pixel budget.
 *
 * WHY THIS RATHER THAN DISTANCE THRESHOLDS. Hand-tuned switch distances are the usual approach and they
 * are wrong on every screen but the one they were tuned on: double the resolution and every level pops
 * at twice the apparent detail; change the field of view and they all shift. Expressing the budget in
 * pixels means the tuning number is a perceptual claim — "no one can see a facet smaller than a
 * pixel" — which stays true everywhere.
 *
 * The conversion is the same one `resolutionForScreen` uses in reverse, and deliberately so: one derives
 * a mesh resolution from a pixel budget, the other picks among meshes using the same budget. Two
 * different conversions would mean the finest level could be finer than anything the selector would ever
 * ask for.
 */

/**
 * World units per pixel at a given distance from the camera.
 *
 * Screen space spans 2 over the diagonal, and a screen offset maps to a world offset by
 * `distance / focal`.
 */
export function worldPerPixel(distance, focal, diagonalPx) {
  return (2 * distance) / (focal * Math.max(diagonalPx, 1));
}

/**
 * Choose a level: the coarsest whose error stays under `errorPx` on screen.
 *
 * @param {{errorWorld:number}[]} levels  finest first, error ascending
 * @param {number} distance  from the camera to the object, world units
 * @param {number} focal
 * @param {number} diagonalPx
 * @param {number} errorPx   how large a deviation may appear before it is worth more triangles
 * @returns {number} index into `levels`
 */
export function selectLod(levels, distance, focal, diagonalPx, errorPx) {
  const perPixel = worldPerPixel(distance, focal, diagonalPx);
  const budget = perPixel * errorPx;
  // Coarsest first, so the first acceptable level found is the cheapest acceptable level. Falling
  // through to 0 is the right failure mode: if nothing qualifies, the object is close enough that the
  // finest mesh is what it deserves.
  for (let i = levels.length - 1; i > 0; i--) {
    if (levels[i].errorWorld <= budget) return i;
  }
  return 0;
}
