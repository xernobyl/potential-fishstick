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
   * FINER THAN IT USED TO BE (6.0 -> 3.0), and that is the adaptive contourer paying for itself. A
   * uniform grid spends the same triangles on a flat wing panel as on the nozzle rim, so asking for
   * detail meant buying it everywhere; the budget had to be loose to keep the count sane. Simplification
   * removes the waste where the surface is flat, so the resolution can be set by what the SHARP features
   * need and the flat ones cost nothing. Measured on this tree, four LOD levels built from one octree:
   *
   *     res 48    74 ms    4672 / 4096 / 3110 / 2474 tris
   *     res 64   144 ms    7564 / 5302 / 3550 / 2544 tris
   *     res 80   167 ms   10364 / 6132 / 3564 / 2068 tris
   *
   * res 64 is the pick: its finest level has the fidelity of a 64-cell grid at roughly the triangle
   * count the old 48-cell UNIFORM mesh cost (4936), and the whole chain builds in about the time one
   * uniform mesh used to.
   */
  errorPx: 3.0,
  /** Typical viewing distance in world units, for the resolution estimate. The chase camera's stand-off. */
  viewDistance: 3.2,
  /** Ceiling, so a huge window cannot ask for a mesh that takes visible time to build at startup. */
  maxResolution: 64,

  /**
   * The LOD chain, as simplification budgets in BODY units — the same units this file authors in.
   *
   * Finest first. They are geometric ERROR, not triangle targets: a level is "the mesh you get when no
   * vertex is allowed to be more than this far from the true surface", which is what makes the selector
   * able to reason about them in pixels (see core/lod.js). Roughly geometric spacing, so each level is a
   * meaningful step rather than a rounding of the last.
   *
   * The whole chain comes out of ONE octree, simplified progressively, so the levels are nested by
   * construction — a vertex in a coarse level exists in every finer one. That is why switching between
   * them does not pop in the way independently-built levels do.
   *
   * The finest is not zero: at zero the contourer still merges genuinely coplanar cells for no error at
   * all, and a hair above zero also collects the cells that are flat to within a rounding error.
   */
  lodErrors: [0.0015, 0.004, 0.010, 0.025],

  /**
   * How large a geometric error may APPEAR before it is worth stepping to a finer level, in pixels.
   *
   * Sub-pixel is the honest threshold for "cannot be seen" and it is what this is set near. Raising it
   * saves triangles on a ship that is usually small on screen; lowering it below about half a pixel buys
   * nothing, because the resolve and the film grade are both operating at that scale already.
   */
  lodErrorPx: 0.9,
};
