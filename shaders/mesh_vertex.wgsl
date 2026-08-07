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

/// A rigid object-to-world transform: an orthonormal basis and an origin.
///
/// Rigid, not affine, and that is the property the temporal side depends on — a generated mesh animates
/// by rotating and translating, never by deforming, so the previous position of a vertex is the same
/// vertex under the previous transform. The origin was added for the ship; the rings pass zero, being
/// centred on the world origin.
struct MeshRigid {
  ax     : vec3f,
  ay     : vec3f,
  az     : vec3f,
  origin : vec3f,
};

fn meshToWorld(b : MeshRigid, v : vec3f) -> vec3f {
  return b.origin + b.ax * v.x + b.ay * v.y + b.az * v.z;
}

/// The same transform's rotation applied to a DIRECTION, which must not pick up the translation.
fn meshDirToWorld(b : MeshRigid, v : vec3f) -> vec3f {
  return b.ax * v.x + b.ay * v.y + b.az * v.z;
}

/// Build a rigid transform from a position and an orientation QUATERNION.
///
/// The ship carries a quaternion because its orientation is integrated, while the rings carry a basis
/// because theirs is derived analytically. Converting here rather than at either call site keeps one
/// transform type in the vertex path.
fn meshRigidFromQuat(pos : vec3f, q : vec4f) -> MeshRigid {
  var b : MeshRigid;
  b.ax = qrotate(q, vec3f(1.0, 0.0, 0.0));
  b.ay = qrotate(q, vec3f(0.0, 1.0, 0.0));
  b.az = qrotate(q, vec3f(0.0, 0.0, 1.0));
  b.origin = pos;
  return b;
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

fn meshXform(v : MeshVertex, now : MeshRigid, prev : MeshRigid) -> MeshXform {
  var o : MeshXform;
  o.wp = meshToWorld(now, v.pos);
  o.wn = meshDirToWorld(now, v.nor);
  o.clip = jitterClip(frame.viewProj * vec4f(o.wp, 1.0));
  o.viewZ = length(o.wp - frame.camPos.xyz);
  // The same vertex, one frame ago. Exact rather than estimated, because the only thing that changed
  // is the basis - which is why the geometry is baked in object space in the first place.
  let prevWp = meshToWorld(prev, v.pos);
  o.prevClip = frame.prevViewProj * vec4f(prevWp, 1.0);
  o.prevViewZ = length(prevWp - frame.prevCamPos.xyz);
  return o;
}
