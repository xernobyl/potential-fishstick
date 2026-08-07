/**
 * What a SCENE is, and what it deliberately is not.
 *
 * A scene owns the WORLD: what exists, how it moves, where the camera goes, and which passes draw it.
 * It does not own the renderer's machinery — targets, the uniform block, the temporal resolve, bloom,
 * the flare, the film grade and the buffer viewer are all shared, and every scene gets them for free.
 *
 * THE SPLIT IS THE POINT, and it falls where the frame graph already had a seam. The renderer's job was
 * always "sample a frame, resolve it temporally, grade it"; the planetoid was simply the only thing
 * being sampled. Naming that boundary costs one indirection and buys a second scene that reuses the
 * entire post chain — the same TAA, the same AgX, the same instruments — without a flag anywhere in it.
 *
 * WHY NOT JUST A FLAG. The obvious cheaper move is `if (modelView)` sprinkled through `frame()`. It
 * works for two scenes and collapses at three, and worse, it puts the model viewer's concerns inside
 * the code that has to stay correct for the real scene. A scene that cannot reach into the frame graph
 * cannot break it.
 *
 * THE CONTRACT, in the order the renderer calls it:
 *
 *   init(rc)            build passes and pipelines. Awaited once, before the first frame.
 *   update(rc)          advance world state and position the camera. `rc.dt` may be exactly zero.
 *   writeState(st, rc)  contribute scene-specific fields to the frame uniform.
 *   recordWorld(...)    draw into the scene and solid layers, before the temporal resolve.
 *   recordAdditive(...) draw into the shared additive layer, after it.
 *   destroy()           release anything the scene allocated.
 *
 * `solidPasses` lists the rasterised mesh passes so the renderer can apply LOD selection and the
 * wireframe view uniformly — those are properties of how the renderer draws meshes, not of what any
 * particular scene contains, so a new scene gets both by listing its passes rather than by
 * reimplementing either.
 *
 * `rc` is the RENDER CONTEXT, rebuilt each frame and never retained: `{ renderer, gpu, targets, camera,
 * shaders, frameBGL, time, dt, input, frustum }`. Passing it rather than storing a renderer reference
 * on the scene keeps the dependency one-way and obvious at every call site.
 */

export class Scene {
  /**
   * A short label, for the scene dropdown.
   * @type {string}
   */
  static label = 'scene';

  /** @param {object} rc render context */
  // eslint-disable-next-line no-unused-vars
  async init(rc) {}

  /** Advance the world and place the camera. */
  // eslint-disable-next-line no-unused-vars
  update(rc) {}

  /** Add this scene's own fields to the uniform state object. */
  // eslint-disable-next-line no-unused-vars
  writeState(st, rc) {}

  /**
   * Rasterised mesh passes, for LOD selection and the wireframe view.
   * @returns {import('../passes/solidmesh.js').SolidMeshPass[]}
   */
  get solidPasses() { return []; }

  /** Draw the world: the marched layer and the solid layer, before the resolve. */
  // eslint-disable-next-line no-unused-vars
  recordWorld(encoder, frameBG, profiler, rc) {}

  /** Draw into the shared additive layer, after the resolve. */
  // eslint-disable-next-line no-unused-vars
  recordAdditive(encoder, frameBG, profiler, rc) {}

  destroy() {}
}
