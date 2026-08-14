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
  sphere, box, roundBox, cylinder, cone, plane,
  union, intersect, subtract, smoothUnion, translate, rotate, mirror, repeat,
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

/**
 * WHERE THINGS ATTACH TO THE HULL, in authored body units.
 *
 * Exported because three other files need these and were each carrying their own copy. The rail guns
 * fired from `RAIL.spread`, the contrails emitted from `CONTRAIL.spread`, and the engine plumes from a
 * literal in the plume shader — all describing points on a hull defined here, none derived from it.
 * They had already drifted: the muzzles were sitting 0.178 units past the wingtip, left behind when the
 * marched hull (whose wings reached 0.61) was replaced by this one.
 *
 * A hardpoint is a property of the SHAPE. Anything bolted to it reads the number from here.
 */
export const SHIP_HARDPOINTS = {
  /** Engine nacelle centreline. Two of them, mirrored: the nozzles, plumes and contrails all sit here. */
  nacelle: [0.175, -0.038, -0.30],
  /** Where a nozzle's mouth is, so the plumes start at the hole rather than inside it. */
  nozzleZ: -0.46,
  /** Wing tip, where the gun pods are. The rail guns fire from here. */
  wingTip: 0.55,
  /** Gun pod centre along Z, so the muzzle is at its nose rather than its middle. */
  podZ: -0.02,
};

/** The hull, in body units: roughly 1.15 long, 1.10 across the wings. */
export function shipTree() {
  const [nx, ny, nz] = SHIP_HARDPOINTS.nacelle;

  // ---- the core body ----
  //
  // A rounded box CHINED along its lower flanks: intersecting with two mirrored angled planes shaves
  // the belly into a keel, which is the single cheapest way to stop a fuselage reading as a loaf. The
  // chine is a plane, so dual contouring reconstructs it as a dead-straight crease — the case this
  // mesher was chosen for.
  // A PLANE KEEPS WHERE `dot(p - t, n) <= 0`, so the origin has to come out NEGATIVE or the intersect
  // removes the body instead of shaving it. The first version had the normal the wrong way round and
  // produced an empty fuselage — which, blended into a smoothUnion with everything else, quietly
  // deformed the whole hull rather than obviously vanishing.
  const chine = mirror(0, translate([0.11, -0.075, 0], plane([0.67, -0.74, 0])));
  const fuselage = intersect(roundBox([0.10, 0.082, 0.36], 0.035), chine);

  // A blunt cone for the nose, blended in so there is no crease where it meets the body.
  const nose = translate([0, -0.005, 0.40], alongZ(cone(0.095, 0.014, 0.14)));

  // A DOME FOR THE PILOT — a real bubble, not the flattened blister this used to be.
  //
  // The sphere is cut off BELOW its equator rather than through it: a hemisphere sitting on a hull
  // reads as a bump, where a sphere cut low reads as a bubble you could sit inside, because you can
  // see the glass turn back under itself at the sill. The box is what does the cutting, and its top
  // is above the sphere's top so only the bottom is trimmed.
  const domeR = 0.090;
  const dome = translate([0, 0.062, 0.145],
    intersect(sphere(domeR), translate([0, 0.030, 0], box([domeR, domeR, domeR]))));
  // A frame ring around the sill, so the glass meets the hull at something rather than just ending.
  const domeRing = translate([0, 0.040, 0.145], alongZ(cylinder(0.082, 0.010)));
  // And the coaming: a raised lip in front of the glass, which is the detail that makes a canopy
  // read as a cockpit rather than as a glass ball stuck on a fuselage.
  const coaming = translate([0, 0.052, 0.225], roundBox([0.062, 0.020, 0.030], 0.010));

  // ---- wings ----
  //
  // Swept, thin in Y so they have a defined leading and trailing edge, and reaching all the way to the
  // gun pods. The sweep is a rotation about Y rather than a modelled taper.
  //
  // A BOX, not a roundBox. A fillet here is sub-cell (0.012 vs a 0.016 cell) and reads as a jagged
  // staircase rather than a rounded edge; the crisp 90-degree edge is exactly what the hard-edge
  // splitting + analytic normals reconstruct as a clean straight line.
  const sweep = Math.cos(0.40);
  const sweepS = Math.sin(0.40);
  const wing = translate([0.30, -0.008, -0.05],
    rotate([sweep, 0, -sweepS], [0, 1, 0], [sweepS, 0, sweep],
           box([0.27, 0.015, 0.115])));

  // Gun pods at the tips — so the rail guns come out of something instead of out of thin air.
  const pod = translate([SHIP_HARDPOINTS.wingTip, -0.005, SHIP_HARDPOINTS.podZ],
    alongZ(cylinder(0.030, 0.115)));
  const podNose = translate([SHIP_HARDPOINTS.wingTip, -0.005, SHIP_HARDPOINTS.podZ + 0.115],
    alongZ(cone(0.030, 0.008, 0.045)));

  // A pylon joining each pod to the wing underside, and a fence part way out.
  const pylon = translate([0.42, -0.035, -0.03], roundBox([0.075, 0.028, 0.055], 0.010));
  const fence = translate([0.36, 0.022, -0.02], roundBox([0.024, 0.032, 0.070], 0.010));

  // ---- twin nacelles ----
  //
  // TWO of them, which is the whole point of this revision: there are two contrails and two plumes, and
  // there was one nozzle for them to come out of. They sit on the hardpoint the trails read.
  const nacelle = translate([nx, ny, nz], alongZ(cylinder(0.062, 0.16)));
  // A wider collar at the front, so the intake reads as a lip rather than a cut cylinder.
  const collar = translate([nx, ny, nz + 0.15], alongZ(cylinder(0.078, 0.022)));
  // The bell, flaring out to the mouth.
  //
  // `cone(r0, r1, hh)` puts r0 at NEGATIVE z once `alongZ` has turned it, so the wide end is written
  // first. Getting this backwards is not a cosmetic error: the first version tapered the bell inward
  // toward the mouth and put the NOZZLE's wide end deep inside a nacelle narrower than it, so the cut
  // ate through the wall from within and left it negative-thickness. That is where 36 non-manifold
  // edges came from, and nothing about the silhouette gave it away.
  const bell = translate([nx, ny, nz - 0.15], alongZ(cone(0.098, 0.062, 0.045)));

  // ---- vertical surfaces ----
  //
  // TWIN FINS, canted outward off the nacelles rather than one fin on the spine. Symmetric, and it ties
  // the engines into the silhouette instead of leaving them as two tubes stuck on the back.
  const fin = translate([nx + 0.010, 0.105, nz - 0.06],
    rotate([Math.cos(0.22), Math.sin(0.22), 0], [-Math.sin(0.22), Math.cos(0.22), 0], [0, 0, 1],
           roundBox([0.011, 0.095, 0.085], 0.009)));

  // A dorsal spine running back from the canopy, and ventral strakes under the belly.
  const spine = translate([0, 0.080, -0.12], roundBox([0.028, 0.030, 0.185], 0.012));
  const strake = translate([0.070, -0.090, -0.16], roundBox([0.011, 0.045, 0.105], 0.009));

  // ---- greeble ----
  //
  // Rows of hard-edged blocks, unioned WITHOUT a blend so they keep their corners.
  //
  // EVERY FEATURE AND EVERY GAP IS AT LEAST THREE GRID CELLS, and that is a hard constraint rather than
  // a preference. A cell that spans a gap sees the surface enter and leave twice, and dual contouring
  // gives it ONE vertex — so the two sheets get welded and the mesh stops being manifold. The first
  // pass here had gaps of 1.8 cells and produced 74 non-manifold edges; nothing looked obviously
  // broken, it just quietly stopped being a surface you could rely on.
  //
  // At the resolution this builds at (cell ~0.018) that means blocks and gaps both around 0.055. Which
  // is also why the greeble is CHUNKY: fine detail is not available at any price short of a resolution
  // that costs seconds to build, and pretending otherwise produces noise rather than detail.
  const flankBlocks = translate([0.100, -0.020, -0.09],
    repeat(2, 0.115, 4, roundBox([0.014, 0.026, 0.030], 0.005)));
  const spineBlocks = translate([0, 0.104, -0.13],
    repeat(2, 0.110, 3, roundBox([0.022, 0.011, 0.028], 0.005)));
  const nacelleRibs = translate([nx, ny + 0.056, nz],
    repeat(2, 0.110, 3, roundBox([0.032, 0.014, 0.028], 0.006)));

  const hull = smoothUnion(0.045,
    fuselage,
    nose,
    dome,
    domeRing,
    coaming,
    mirror(0, wing),
    mirror(0, pylon),
    mirror(0, nacelle),
    mirror(0, collar),
    mirror(0, bell),
    mirror(0, fin),
    spine,
    mirror(0, strake),
  );

  // Hard union for the detail: a blend would round the corners off the very thing that makes greeble
  // read as machinery.
  const detailed = union(
    hull,
    mirror(0, pod),
    mirror(0, podNose),
    mirror(0, fence),
    mirror(0, flankBlocks),
    spineBlocks,
    mirror(0, nacelleRibs),
  );

  // ---- what gets cut out ----
  //
  // Subtracted LAST, so the blends above cannot fill the holes back in. Hard subtraction rather than
  // smooth: a nozzle rim and a vent slot should both be crisp, which is the case dual contouring is
  // good at.
  // Wide at the mouth, narrowing forward — a bell, not a spike. The wall it leaves is the bell's outer
  // radius minus this one's: 0.098 - 0.050 = 0.048, which is three grid cells, the same floor every
  // other feature here is held to.
  const nozzle = translate([nx, ny, SHIP_HARDPOINTS.nozzleZ],
    alongZ(cone(0.050, 0.014, 0.070)));
  // Cooling slots along the spine, and an intake notch in each collar.
  const vents = translate([0, 0.110, -0.185],
    repeat(2, 0.110, 3, box([0.032, 0.022, 0.026])));
  // Same floor: the collar is 0.078, so a 0.030 bore leaves a 0.048 lip.
  const intake = translate([nx, ny, nz + 0.163], alongZ(cylinder(0.030, 0.030)));

  return subtract(detailed, union(mirror(0, nozzle), vents, mirror(0, intake)));
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
  /**
   * FLOOR on the derived resolution, and it is a correctness constraint rather than a quality one.
   *
   * Measured on this hull: at 64 cells the mesh comes out with 20 non-manifold edges, at 80 and above
   * with none. The greeble, the nozzle bells and the nacelle-to-wing junction all have features around
   * three cells wide at 80; drop below that and a cell starts spanning two surface sheets, which dual
   * contouring answers with one vertex and a pinch. A small window must not be allowed to produce a
   * mesh that is no longer a surface.
   */
  minResolution: 80,
  /** Typical viewing distance in world units, for the resolution estimate. The chase camera's stand-off. */
  viewDistance: 3.2,
  /** Ceiling, so a huge window cannot ask for a mesh that takes visible time to build at startup. */
  /**
   * Ceiling. 80 is also the floor, so this hull always meshes at exactly 80 — the detail needs that
   * much to come out manifold, and 96 costs 524 ms against 344 ms for triangles nobody can see on a
   * hull that is usually 40 px across.
   */
  maxResolution: 80,

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
