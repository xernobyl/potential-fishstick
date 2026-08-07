/**
 * The ship, as a signed distance field.
 *
 * Authored in the hull's own body space, matching the convention the rest of the ship code uses: +Z is
 * the nose, +Y is up, +X is right, and the origin is the centre of mass the camera and the contrail
 * both hang off. Every number here is in those units, and `SHIP_MESH.scale` at the bottom is the only
 * place a world size appears — so re-sizing the ship is one number and does not disturb the shape.
 *
 * WHY A TREE AND NOT A WGSL FUNCTION. The old hull was an SDF marched in the scene shader, which meant
 * the shape could only ever be evaluated on the GPU, one ray at a time. As data it can be MESHED — and
 * a mesh is what lets the ship be rasterised with the rings, occluded by hardware depth, culled, and
 * given an exact motion vector from its own rigid transform rather than from a per-pixel hit point.
 *
 * SHAPE NOTES, since the geometry is the interesting part:
 *
 *   The fuselage is a rounded box rather than a capsule, so the top and bottom read as flat panels with
 *   a defined edge — which is exactly the feature dual contouring reconstructs and marching cubes would
 *   have bevelled away. It is the reason to prefer one over the other, made visible.
 *
 *   Wings are MIRRORED rather than authored twice. `mirror` folds the field about x = 0, so the
 *   starboard wing is the port wing by construction and cannot drift from it.
 *
 *   The blends are `smoothUnion` with a width in world units, so the fillet where a wing meets the
 *   fuselage is the same size regardless of how big either part is. That is the whole reason the
 *   operator takes a width rather than a rate.
 *
 *   The nozzle is a SUBTRACTION, which is what makes it a hole rather than a dark texture. Dual
 *   contouring resolves the rim as a sharp circle because the subtraction is exact there.
 */

import {
  sphere, box, roundBox, cylinder, cone,
  union, subtract, smoothUnion, translate, rotate, mirror, scale,
} from './sdf/nodes.js';

/**
 * A cylinder along +Z rather than +Y.
 *
 * `cylinder` and `cone` are authored along Y because that is the axis their closed forms are simplest
 * on. This basis maps the primitive's local Y onto world Z: `rotate` takes the rows of the world->local
 * matrix, so `ay = [0,0,1]` reads "local y is world z". Right-handed, so the shape is rotated rather
 * than reflected — which matters for nothing symmetric and matters a great deal the first time it is
 * not.
 */
const alongZ = (child) => rotate([1, 0, 0], [0, 0, 1], [0, -1, 0], child);

/** The hull, in body units: roughly 1.0 long, 0.55 across the wings. */
export function shipTree() {
  const fuselage = roundBox([0.085, 0.075, 0.34], 0.045);

  // A blunt cone for the nose, blended in so there is no crease where it meets the body.
  const nose = translate([0, 0, 0.36], alongZ(cone(0.10, 0.012, 0.13)));

  // The canopy: a bulge on top, forward of centre. Sphere rather than a box because this is the one
  // part that should read as curved.
  const canopy = translate([0, 0.065, 0.10], scale(1.0, sphere(0.075)));

  // Swept wings, mirrored. The rotation is a shallow sweep about Y, and the box is thin in Y so the
  // wing has a defined leading and trailing edge rather than a rounded tube.
  const sweep = Math.cos(0.42);
  const sweepS = Math.sin(0.42);
  const wing = translate([0.20, -0.005, -0.06],
    rotate([sweep, 0, -sweepS], [0, 1, 0], [sweepS, 0, sweep],
           roundBox([0.19, 0.014, 0.10], 0.012)));

  // A vertical fin, and a pair of small ventral strakes that give the silhouette something under it.
  const fin = translate([0, 0.13, -0.26], roundBox([0.012, 0.10, 0.075], 0.010));
  const strake = translate([0.055, -0.085, -0.20], roundBox([0.012, 0.055, 0.10], 0.010));

  // Engine housing, and the nozzle cut out of it.
  const housing = translate([0, 0, -0.36], alongZ(cylinder(0.072, 0.075)));
  const nozzle = translate([0, 0, -0.40], alongZ(cone(0.030, 0.062, 0.070)));

  const hull = smoothUnion(0.05,
    fuselage,
    nose,
    canopy,
    mirror(0, wing),
    fin,
    mirror(0, strake),
    housing,
  );

  // Subtracted last, so the blends above cannot fill the hole back in. Hard subtraction rather than
  // smooth: the nozzle rim should be a crisp circle, which is the case dual contouring is good at.
  return subtract(hull, nozzle);
}

/**
 * Mesh parameters, kept beside the shape they describe.
 *
 * `scale` converts body units to world units and is applied to the MESH once, not to the field — a
 * uniform scale of a distance field is exact, but scaling the tree would also scale every blend width
 * and fillet, which are authored in body units on purpose.
 */
export const SHIP_MESH = {
  /** Body units -> world units. Matches the marched hull it replaces. */
  scale: 0.60,
  /**
   * Largest a cell may appear on screen, in pixels, at the distance the ship is usually seen from.
   *
   * CALIBRATED, not guessed. Measured on this tree, at the resolutions the derivation can land on:
   *
   *     res 24    876 tris   26 ms    21 KB
   *     res 32   2124 tris   33 ms    49 KB
   *     res 40   3540 tris   43 ms    82 KB
   *     res 48   4936 tris   38 ms   115 KB
   *     res 64   8836 tris  101 ms   207 KB
   *
   * 6 px lands the derivation around 40-48 at the chase distance, which is a few thousand triangles for
   * a hull that is usually 40 px across, and a build that costs less than a shader compile. The first
   * version asked for 2 px, derived 192, capped at 96, and spent 221 ms making 20044 triangles for a
   * silhouette nobody can see the facets of anyway — the derivation was right and the budget was wrong.
   *
   * The resolution is still DERIVED rather than fixed, so the mesh suits the window rather than the
   * machine it was authored on, and the same call at other resolutions is the LOD chain when that is
   * wanted.
   */
  errorPx: 6.0,
  /** Typical viewing distance in world units, for the resolution estimate. The chase camera's stand-off. */
  viewDistance: 3.2,
  /** Ceiling, so a huge window cannot ask for a mesh that takes visible time to build at startup. */
  maxResolution: 48,
};
