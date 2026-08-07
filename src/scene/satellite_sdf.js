/**
 * The satellite, as signed distance fields — a bus and an array wing, meshed separately.
 *
 * TWO TREES, NOT ONE, and that is forced by the articulation rather than by taste. The arrays rotate
 * about their boom to track the sun while the bus stays nadir-pointing, so a single rigid mesh could
 * not express the thing that makes these read as satellites instead of as debris. Two shapes, each in
 * its own local space, each placed by its own frame — see `satPart` in satellite.wgsl.
 *
 * These were three analytic boxes: one unit cube instanced fifteen times with per-instance half-extents.
 * That was exactly right while a satellite WAS three boxes. It stopped being right the moment the answer
 * to "what does it look like up close" became "a cube", which is what the model viewer made obvious.
 *
 * SIZES COME FROM `SATELLITES` in tuning.js. The orbital mechanics, the shadowing reach and the
 * material all read those numbers, so the geometry has to be built from them rather than beside them —
 * the ship taught that lesson twice over, with a muzzle constant that outlived the hull it was measured
 * from.
 */

import {
  box, roundBox, cylinder, cone, sphere,
  union, subtract, smoothUnion, translate, rotate, mirror, repeat,
} from './sdf/nodes.js';
import { SATELLITES } from './tuning.js';

/** A Y-axis primitive turned to lie along Z. Same helper the ship uses, same reasoning. */
const alongZ = (child) => rotate([1, 0, 0], [0, 0, 1], [0, -1, 0], child);

/**
 * The bus: the box in the middle, with everything a spacecraft has bolted to it.
 *
 * Built in the same axes the analytic version used, so the material's greeble and wrinkle coordinates
 * land where they always did: x along track, y the boom, z zenith.
 */
export function satelliteBusTree() {
  const r = SATELLITES.bus;

  // The core, chamfered rather than square. A hard cube reads as a placeholder; a chamfer catches a
  // highlight on every edge and is the cheapest thing that makes a box look manufactured.
  const core = roundBox([r, r, r], r * 0.16);

  // Multi-layer insulation, as slightly proud panels on the four side faces. Proud rather than flush
  // so there is a real step for the light to catch — the material already draws the crinkle.
  const mliZ = translate([0, 0, r * 0.96], box([r * 0.80, r * 0.80, r * 0.10]));
  const mliX = translate([r * 0.96, 0, 0], box([r * 0.10, r * 0.80, r * 0.80]));

  // The high-gain dish, on a short mast off the zenith face.
  //
  // SOLID, not a hollowed shell. The obvious way to get a concave face is to subtract a sphere from a
  // shallow cone, and it does not survive contact with a grid: measured, the bowl removed everything
  // but a 3.8 mm annulus at the rim — one cell wide, so it vanished at one resolution and resolved as
  // a razor-thin DISCONNECTED ring at the next. A dish a few dozen pixels across does not need a real
  // bowl; it needs a rim, a taper and a feed horn, all of which have thickness.
  // Every joint OVERLAPS rather than abuts. Parts that merely touch come out as separate closed
  // components — watertight and manifold, but three objects where one was intended, and an antenna
  // that reads as floating a hair off its own mast.
  const mast = translate([0, 0, r * 1.40], alongZ(cylinder(r * 0.09, r * 0.55)));
  const dishFace = translate([0, 0, r * 2.05], alongZ(cone(r * 0.30, r * 0.62, r * 0.15)));
  const dishRim = translate([0, 0, r * 2.20], alongZ(cylinder(r * 0.62, r * 0.05)));
  // The feed, on a stub out in front of the face — the bit that says "antenna" rather than "plate".
  const feedArm = translate([0, 0, r * 2.50], alongZ(cylinder(r * 0.055, r * 0.32)));
  const feedHorn = translate([0, 0, r * 2.90], alongZ(cone(r * 0.10, r * 0.19, r * 0.10)));
  const dish = union(dishFace, dishRim, feedArm, feedHorn);

  // Star tracker and a radiator fin on the anti-sun side.
  const tracker = translate([r * 0.5, -r * 0.75, r * 0.62], alongZ(cylinder(r * 0.20, r * 0.30)));
  const radiator = translate([-r * 1.08, 0, -r * 0.15], box([r * 0.14, r * 0.62, r * 0.55]));

  // Boom stubs where the arrays attach, so the wings grow out of something.
  //
  // ALONG the boom, which is Y — and `cylinder` is already Y-native, so it needs no rotation at all.
  // Wrapping it in `alongX` laid each stub ACROSS the boom instead of along it, leaving a 0.09r gap to
  // the core: two extra closed components, an antenna mast that looked fine and two little barrels
  // floating beside the bus.
  const boomStub = translate([0, r * 1.25, 0], cylinder(r * 0.16, r * 0.45));

  // Thruster nozzles at two corners of the nadir face, and a fuel tank bulge.
  const thruster = translate([r * 0.62, r * 0.62, -r * 1.15], alongZ(cone(r * 0.20, r * 0.09, r * 0.22)));
  const tank = translate([0, -r * 0.30, -r * 1.05], sphere(r * 0.42));

  // Greeble: a row of avionics boxes along one flank and a ladder of ribs along another. Sized well
  // clear of the grid — the ship's lesson was that anything under about three cells is not detail, it
  // is noise the mesher cannot resolve.
  const avionics = translate([0, r * 0.55, r * 1.02],
    repeat(0, r * 0.62, 3, box([r * 0.18, r * 0.22, r * 0.12])));
  const ribs = translate([-r * 1.05, 0, r * 0.30],
    repeat(1, r * 0.66, 3, box([r * 0.10, r * 0.16, r * 0.30])));

  const body = smoothUnion(r * 0.10,
    core,
    mast,
    mirror(1, boomStub),
    tank,
  );

  // Hard union for everything that should keep its corners.
  return union(
    body,
    mliZ, mirror(2, mliZ),
    mliX, mirror(0, mliX),
    dish,
    tracker,
    radiator,
    mirror(0, mirror(1, thruster)),
    avionics,
    ribs,
  );
}

/**
 * One array wing: a framed panel on a short boom arm.
 *
 * Panel space matches the analytic version exactly — x thin (the face normal), y along the boom, z
 * across the width — because the cell grid, the busbars and the sheen in `shadeSatSurface` are all
 * authored against it and none of that should have to move.
 */
export function satellitePanelTree() {
  const t = SATELLITES.panelThick;
  const L = SATELLITES.panelLen;
  const W = SATELLITES.panelWide;
  const boom = SATELLITES.boom;

  // ONE-SIDED, growing outward from y = 0 where the bus is.
  //
  // This mesh used to be MIRRORED about its own origin, so it extended both ways from wherever it was
  // placed — and placed at the joint, its inboard half reached past the bus to y = -0.085. Both wings
  // did that, their inboard halves overlapped across the centreline and merged, and what should have
  // been two panels rendered as three: outboard, merged-middle, outboard.
  //
  // A wing extends AWAY from the spacecraft. Building it one-sided is what makes that true by
  // construction rather than by choosing a placement that happens to hide the other half.
  const mid = boom + L;

  // The substrate. Thicker than the analytic slab was: three grid cells is the floor for meshing it at
  // all, and at this size on screen the difference is invisible where a broken surface would not be.
  const sheet = translate([0, mid, 0], box([t * 1.6, L * 0.97, W * 0.97]));

  // A frame around the rim, standing slightly proud of the cells.
  const railLong = translate([0, mid, W * 0.965], box([t * 2.2, L, W * 0.035]));
  const railOuter = translate([0, mid + L * 0.965, 0], box([t * 2.2, L * 0.035, W]));
  const railInner = translate([0, mid - L * 0.965, 0], box([t * 2.2, L * 0.035, W]));

  // A spine down the middle and cross ribs, which is what an array's back actually looks like.
  const spine = translate([0, mid, 0], box([t * 2.4, L, W * 0.05]));
  const crossRibs = translate([0, mid, 0], repeat(1, L * 0.62, 3, box([t * 2.0, L * 0.04, W * 0.9])));

  // The arm from the bus out to the panel's inner edge.
  const arm = translate([0, boom * 0.5, 0], cylinder(t * 3.0, boom * 0.6));

  return union(sheet, railLong, mirror(2, railLong), railOuter, railInner,
               spine, crossRibs, arm);
}

/**
 * Mesh parameters. Two shapes, two budgets.
 *
 * The bus is small and dense with detail, so it wants a fine grid on a small box. The panel is large
 * and nearly flat, so it wants a coarse one — and the adaptive contourer collapses its flat faces to
 * almost nothing, which is exactly the case that argument was made for.
 */
export const SAT_MESH = {
  busResolution: 56,
  panelResolution: 44,
  /** Simplification budgets, in the units these trees are authored in (world units already). */
  error: [0.0008, 0.0025],
};
