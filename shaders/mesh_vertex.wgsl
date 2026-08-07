// ---------------------------------------------------------------------------
// The shared vertex front end for GENERATED MESHES.
//
// Every generated mesh arrives with the same attributes (see MESH_VERTEX_LAYOUT in core/mesh.js) and
// needs the same four things done to it: transform object space into world by that object's analytic
// basis, jitter the clip position to match the marched geometry it composites against, measure the
// view distance, and transform the SAME vertex by the PREVIOUS frame's basis to get an exact motion
// vector. All four are here so a new shape is a profile and a fragment shader, not a fifth copy of
// this arithmetic.
//
// The object's transform is not here, deliberately: it is whatever the shape's own shader knows how
// to compute - `ringDefAt` for the rings - so this takes a basis rather than an object index. That
// keeps the front end ignorant of what it is drawing, which is the whole point.
// ---------------------------------------------------------------------------

//!include "common.wgsl"

/// Matches MESH_VERTEX_LAYOUT. Named fields rather than a packed blob so a layout change that
/// forgets the shader is a compile error.
struct MeshVertex {
  @location(0) pos   : vec3f,   // object space
  @location(1) nor   : vec3f,   // object space
  @location(2) extra : vec4f,   // the shape's own data; see the generator
  @location(3) objId : f32,
};

/// An orthonormal object-to-world basis. Rigid: generated meshes animate by rotation, which is what
/// makes their previous position the same vertex under the previous basis.
struct MeshBasis {
  ax : vec3f,
  ay : vec3f,
  az : vec3f,
};

fn meshToWorld(b : MeshBasis, v : vec3f) -> vec3f {
  return b.ax * v.x + b.ay * v.y + b.az * v.z;
}

/// Everything the temporal side needs from a mesh vertex, in one place.
///
/// `clip` is jittered and `prevClip` is NOT, matching the convention the rest of the renderer uses:
/// the history is indexed by unjittered pixel centres, so the fragment stage adds the jitter back
/// when it turns prevClip into a pixel delta. See rings.wgsl's fragment stage and jitterClip.
struct MeshXform {
  wp        : vec3f,
  wn        : vec3f,
  clip      : vec4f,
  viewZ     : f32,
  prevClip  : vec4f,
  prevViewZ : f32,
};

fn meshXform(v : MeshVertex, now : MeshBasis, prev : MeshBasis) -> MeshXform {
  var o : MeshXform;
  o.wp = meshToWorld(now, v.pos);
  o.wn = meshToWorld(now, v.nor);
  o.clip = jitterClip(frame.viewProj * vec4f(o.wp, 1.0));
  o.viewZ = length(o.wp - frame.camPos.xyz);
  // The same vertex, one frame ago. Exact rather than estimated, because the only thing that changed
  // is the basis - which is why the geometry is baked in object space in the first place.
  let prevWp = meshToWorld(prev, v.pos);
  o.prevClip = frame.prevViewProj * vec4f(prevWp, 1.0);
  o.prevViewZ = length(prevWp - frame.prevCamPos.xyz);
  return o;
}
