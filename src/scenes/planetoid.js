/**
 * The planetoid: the scene this renderer was built for.
 *
 * A raymarched SDF body lit as a scattering medium, three precessing metal rings, a flyable ship, five
 * satellites, GPU embers, curl-noise auroras and a volumetric atmosphere. Everything here used to live
 * directly in `renderer.js`; it moved wholesale when the model viewer arrived, and the move was almost
 * entirely mechanical because the frame graph already had a seam in exactly this place — see scene.js.
 *
 * WHAT THIS OWNS: the world's state (ship physics, contrail, rail guns, auroras), the passes that draw
 * it, and where the camera goes. What it does NOT own is everything after the sample: the temporal
 * resolve, bloom, the flare, the grade and the buffer viewer are the renderer's, shared with every
 * other scene.
 *
 * The ordering rules that used to be comments in `frame()` are preserved with them, because they are
 * properties of THIS scene rather than of rendering in general: the solids draw before the resolve so
 * TAA can accumulate their exact motion vectors, and the first solid pass clears the shared layer while
 * the rest load it.
 */

import { Scene } from './scene.js';
import { ScenePass } from '../passes/scene.js';
import { SolidMeshPass } from '../passes/solidmesh.js';
import { AdditivePass } from '../passes/additive.js';
import { EmberPass } from '../passes/embers.js';
import { Mesh } from '../core/mesh.js';
import { Ship } from '../scene/ship.js';
import { Contrail } from '../scene/contrail.js';
import { Railgun } from '../scene/railgun.js';
import { Aurora } from '../scene/aurora.js';
import { rectTube, concatMeshes } from '../scene/meshgen.js';
import { shipTree, SHIP_MESH } from '../scene/ship_sdf.js';
import { satelliteBusTree, satellitePanelTree, SAT_MESH } from '../scene/satellite_sdf.js';
import { compile, bounds } from '../scene/sdf/nodes.js';
import { resolutionForScreen } from '../scene/sdf/dualcontour.js';
import { dualContourAdaptive } from '../scene/sdf/octree.js';
import { RINGS, SATELLITES, SHIP, CONTRAIL, RAIL, AURORA, ringDims, wgslDefines }
  from '../scene/tuning.js';

/**
 * The ship's hull, contoured from its SDF into an LOD chain.
 *
 * At module scope rather than inside the scene: it is a pure function of the tree and the screen, holds
 * no device state, and doing it here keeps the scene's constructor free of a hundred-millisecond call.
 *
 * The mesh is scaled AFTER contouring rather than by scaling the tree: a uniform scale of a distance
 * field is exact, but scaling the tree would also scale every blend width and fillet, which are authored
 * in body units on purpose.
 */
let shipMeshCache = null;

export function buildShipMesh() {
  // MEMOISED, because two scenes want it. The model viewer draws the same hull as the planetoid, and
  // contouring it twice would double the startup cost (~135 ms) to produce two identical results. The
  // GPU buffers are still separate — a pass owns its meshes — but the CPU work happens once.
  if (shipMeshCache) return shipMeshCache;

  const tree = shipTree();
  const b = bounds(tree);
  const size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) * SHIP_MESH.scale;
  const res = Math.max(
    SHIP_MESH.minResolution,
    Math.min(
      SHIP_MESH.maxResolution,
      resolutionForScreen({
        size,
        distance: SHIP_MESH.viewDistance,
        focal: 1.2,
        diagonalPx: 2500,
        errorPx: SHIP_MESH.errorPx,
      }),
    ),
  );

  // THE WHOLE LOD CHAIN FROM ONE OCTREE. Passing the array of budgets simplifies progressively rather
  // than meshing four times, so four levels cost barely more than the finest one — and the levels are
  // nested, which is what stops a switch from popping.
  const levels = dualContourAdaptive(compile(tree), {
    bounds: b,
    resolution: res,
    error: SHIP_MESH.lodErrors,
  });

  const out = levels.map((mesh, i) => {
    // Body units -> world units. A uniform scale leaves normals unchanged, which is the whole reason
    // `scale` is uniform-only.
    const positions = new Float32Array(mesh.positions.length);
    for (let j = 0; j < positions.length; j++) positions[j] = mesh.positions[j] * SHIP_MESH.scale;

    // Through `concatMeshes` even though there is one object, rather than hand-rolling the ids and the
    // empty `extra`. It is the same three lines either way until it is not: the first version wrote them
    // by hand and silently skipped the per-object RANGE, so the ship fell back to a bounding sphere of
    // Infinity and was never culled. One code path produces the ids, the extra data and the bounds
    // together, so a mesh cannot arrive half-described.
    const level = concatMeshes([{
      positions,
      normals: mesh.normals,
      extra: new Float32Array(mesh.vertexCount * 4),
      indices: mesh.indices,
    }]);
    // The budget this level was simplified to, in WORLD units — what the selector compares against.
    level.errorWorld = SHIP_MESH.lodErrors[i] * SHIP_MESH.scale;
    return level;
  });

  shipMeshCache = out;
  return out;
}

/**
 * The satellite: a bus and one array wing, contoured and concatenated into one buffer.
 *
 * TWO RANGES, NOT TWO MESHES, so the whole swarm is one buffer and one bind. The vertex tag
 * `concatMeshes` writes is what tells the shader which of the two a vertex belongs to, and the pass
 * draws each range with its own instance count — one bus per satellite, two wings.
 */
let satMeshCache = null;

export function buildSatelliteMesh() {
  if (satMeshCache) return satMeshCache;
  const parts = [
    [satelliteBusTree(), SAT_MESH.busResolution],
    [satellitePanelTree(), SAT_MESH.panelResolution],
  ].map(([tree, resolution]) => {
    const b = bounds(tree);
    // THE FINE BUDGET, and the coarse one was a mistake worth naming. Simplification error is a
    // LENGTH, and it has to be small against the thinnest thing being simplified — not against the
    // object's size. At 0.0025 it was 16% of the panel's 0.016 thickness, which is enough to collapse
    // the sheet's two faces into each other: 24 boundary edges, a holed array, at any resolution.
    const m = dualContourAdaptive(compile(tree), {
      bounds: b, resolution, error: SAT_MESH.error[0],
    });
    return {
      positions: m.positions,
      normals: m.normals,
      extra: new Float32Array(m.vertexCount * 4),
      indices: m.indices,
    };
  });
  satMeshCache = concatMeshes(parts);
  return satMeshCache;
}

/** The three hoops, concatenated into one buffer with a per-vertex ring index. */
export function buildRingMesh() {
  return concatMeshes(Array.from({ length: RINGS.count }, (_, i) => rectTube({
    segments: RINGS.segments, ...ringDims(i),
  })));
}

export class PlanetoidScene extends Scene {
  static label = 'planetoid';

  constructor(gpu, targets, shaders) {
    super();
    this.gpu = gpu;
    this.targets = targets;
    this.shaders = shaders;

    this.ship = new Ship();
    this.contrail = new Contrail(gpu.device);
    this.railgun = new Railgun(gpu.device);
    this.aurora = new Aurora(gpu.device);
    this._fireSlot = -1;

    this.passes = {
      scene: new ScenePass(gpu, targets, shaders),
      embers: new EmberPass(gpu, targets, shaders),

      // THE SOLID MESHES, IN DRAW ORDER. The first CLEARS the shared solid layer and the rest LOAD it,
      // so this order is load-bearing rather than cosmetic — see SolidMeshPass#record.
      rings: new SolidMeshPass(gpu, targets, shaders, {
        label: 'rings',
        shader: 'rings.wgsl',
        // Baked in ring-local space, each vertex tagged with its ring index so one draw covers all
        // three and the vertex shader fetches each ring's precessing basis. That is what keeps the
        // motion vectors exact — see meshgen.js.
        mesh: () => new Mesh(gpu.device, buildRingMesh(), 'rings'),
        // A hoop's sphere is centred on the world origin and its radius does not depend on the
        // precessing basis, so this needs nothing the CPU does not already have.
        worldSphere: (_id, r) => [0, 0, 0, r.radius],
      }),
      // The hull, meshed from its SDF. Appends rather than clears, which is what the solid layer was
      // always meant to support.
      shipmesh: new SolidMeshPass(gpu, targets, shaders, {
        label: 'ship-mesh',
        shader: 'shipmesh.wgsl',
        clear: false,
        mesh: () => buildShipMesh().map((data, i) => {
          const m = new Mesh(gpu.device, data, `ship-lod${i}`);
          m.errorWorld = data.errorWorld;
          return m;
        }),
        // The hull's sphere rides with the ship, centred on its position — an orientation cannot change
        // a sphere, which is the whole reason this needs nothing but a position.
        //
        // The radius is used AS IS. `buildShipMesh` scales the vertices into world units before handing
        // them to `concatMeshes`, so the radius measured from them is already world-space; multiplying
        // by the mesh scale again shrank the sphere to 60% of the hull and would have culled the ship
        // while part of it was still on screen.
        worldSphere: (_id, r) => [
          this.ship.pos[0], this.ship.pos[1], this.ship.pos[2], r.radius,
        ],
      }),
      // Five satellites, each a contoured bus and two contoured array wings — see satellite_sdf.js.
      // One buffer, two ranges, each instanced by the orbital frame its instance index selects.
      //
      // No `worldSphere`, so nothing is culled: the orbits are evaluated in WGSL and never reach the
      // CPU, and a second copy of them in JavaScript to reject 180 triangles would cost more than it
      // saves.
      satellites: new SolidMeshPass(gpu, targets, shaders, {
        label: 'satellites',
        shader: 'satmesh.wgsl',
        clear: false,
        mesh: () => new Mesh(gpu.device, buildSatelliteMesh(), 'satellites'),
        // Range 0 is the bus, one per satellite; range 1 is an array wing, two per satellite.
        instances: (range) => (range.id === 0 ? SATELLITES.count : SATELLITES.count * 2),
      }),

      // Three instanced additive draws that differ only in their data — see AdditivePass. The `source`
      // thunks are called when bind groups are built rather than captured now, so an owner is free to
      // reallocate its buffer.
      contrail: new AdditivePass(gpu, targets, shaders, {
        label: 'contrail',
        shader: 'contrail.wgsl',
        vertices: (CONTRAIL.samples - 1) * 6,
        instances: 2,                         // one per nacelle, selected by instance index
        source: () => this.contrail.buffer,
      }),
      railgun: new AdditivePass(gpu, targets, shaders, {
        label: 'railgun',
        shader: 'railgun.wgsl',
        vertices: RAIL.segments * 6,
        // Always the maximum strand count. A normal shot runs two and collapses the rest, which costs
        // one degenerate vertex each and keeps this a single instanced draw whatever is in flight.
        instances: RAIL.pool * RAIL.strands,
        source: () => this.railgun.buffer,
      }),
      // The burst a shot throws off, and the swarm a charge pulls in — see sparks.wgsl for why those
      // are one pass. ONE SLOT PAST THE POOL: the extra shot's worth of instances is the charge
      // swarm, which is drawn from the live trigger rather than from a stored record.
      sparks: new AdditivePass(gpu, targets, shaders, {
        label: 'sparks',
        shader: 'sparks.wgsl',
        vertices: 6,
        instances: (RAIL.pool + 1) * RAIL.sparks,
        source: () => this.railgun.buffer,
      }),
      aurora: new AdditivePass(gpu, targets, shaders, {
        label: 'aurora',
        shader: 'aurora.wgsl',
        vertices: (AURORA.samples - 1) * 6,
        instances: AURORA.ribbons,
        source: () => this.aurora.buffer,
      }),
    };

    this._solid = [this.passes.rings, this.passes.shipmesh, this.passes.satellites];
  }

  get solidPasses() { return this._solid; }

  async init(rc) {
    await Promise.all([
      this.passes.scene.init(rc.frameBGL),
      this.passes.embers.init(rc.frameBGL),
      ...this._solid.map((m) => m.init(rc.frameBGL, wgslDefines())),
      this.passes.contrail.init(rc.frameBGL, wgslDefines()),
      this.passes.railgun.init(rc.frameBGL, wgslDefines()),
      this.passes.sparks.init(rc.frameBGL, wgslDefines()),
      this.passes.aurora.init(rc.frameBGL, wgslDefines()),
    ]);
  }

  update(rc) {
    const { dt, time, input, camera } = rc;
    this.ship.update(dt, input.cmd);
    this.contrail.update(dt, this.ship);
    // AUTO-FIRE while the ship is still cruising, so there is something to look at before anyone
    // touches a key. A one-frame PULSE per period.
    //
    // THE PLAYER'S KEY IS NOT IN HERE, and putting it in was a real bug: the trigger already fires on
    // RELEASE, so feeding the keydown edge in as well fired twice per tap — and because the wing
    // alternates between shots, the pair came out of both wings at once. One press, one shot, and the
    // press that produces it is the one that ends.
    let fire = false;
    if (!this.ship.flown) {
      const slot = Math.floor(time / SHIP.autoFireEvery);
      if (slot !== this._fireSlot) { this._fireSlot = slot; fire = true; }
    }
    this.railgun.update(time, dt, this.ship, !!input.cmd.trigger, fire);
    // Only the KICK is the scene's; decaying it is the renderer's, because a camera that keeps a
    // decaying offset must decay it in every scene — see Renderer.frame.
    const kick = this.railgun.takeKick();
    if (kick > 0) camera.kick(kick * RAIL.shake);
    this.aurora.update(dt, time);
    camera.update(time, dt, input, this.ship);
  }

  writeState(st) {
    st.ship = this.ship;
    st.auroraPhase = this.aurora.emitPhase;
    st.modelView = false;
    // The trigger's live state, for the charge swarm at the muzzles.
    st.charge = this.railgun.chargeFrac;
    st.chargeHue = this.railgun.chargeHue;
  }

  recordWorld(encoder, frameBG, p, rc) {
    this.passes.scene.record(encoder, frameBG, p);
    // Solids BEFORE the resolve: they carry exact motion vectors, so TAA can accumulate them, which is
    // what anti-aliases their silhouettes and puts them in the bloom.
    for (const m of this._solid) m.record(encoder, frameBG, p, rc.frustum);
  }

  recordAdditive(encoder, frameBG, p) {
    this.passes.embers.simulate(encoder, frameBG, p);
    this.passes.embers.record(encoder, frameBG, p);
    // Into the same additive target, right after the particles.
    this.passes.contrail.record(encoder, frameBG, p);
    this.passes.railgun.record(encoder, frameBG, p);
    this.passes.sparks.record(encoder, frameBG, p);
    this.passes.aurora.record(encoder, frameBG, p);
  }

  destroy() {
    this.contrail.destroy();
    this.railgun.destroy();
    this.aurora.destroy();
    this.passes.embers.destroy?.();
    for (const m of this._solid) m.destroy();
  }
}
