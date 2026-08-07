// ---------------------------------------------------------------------------
// Satellites as real geometry: one unit cube, instanced once per box.
//
// These used to be analytic boxes intersected inside the scene march. Slab tests are cheap, but they
// were paid PER PIXEL OF THE WHOLE SCREEN — five satellites, a bounding sphere each and up to three slab
// tests behind it, evaluated by every ray whether or not it went anywhere near one. As triangles the same
// fifteen boxes cost 180 triangles and one instanced draw, and the rasteriser only shades the pixels they
// actually cover.
//
// WHAT THE MOVE BUYS BEYOND THE OBVIOUS. Hardware depth, so satellites occlude the rings and each other
// without threading `tmax` through a march. Back-face culling, which halves their fragments. And the
// scene march loses its satellite branch entirely, which was executing for every ray in the frame.
//
// WHAT IT COSTS, stated plainly: an analytic box has a perfect silhouette at any zoom and a rasterised one
// has twelve triangles' worth. For a box that is the same thing — a box IS its twelve triangles, so the
// silhouette is exact rather than approximated. This is the case where the conversion is lossless.
//
// ONE UNIT CUBE, SCALED PER INSTANCE, rather than fifteen baked boxes in a buffer: the bus and the two
// array wings are wildly different shapes, and baking them would put the same 24 vertices in memory
// fifteen times over so that a matrix could be avoided. The scale is axis-aligned and so are the face
// normals, so a normal along an axis stays along that axis — the one case where skipping the
// inverse-transpose is exact, not an approximation. See `box()` in meshgen.js.
//
// NOT FRUSTUM CULLED, deliberately. A satellite's position comes from `satFrameAt` in WGSL and never
// reaches the CPU, so culling these would mean a second implementation of the orbital mechanics in
// JavaScript, kept in step by hand, to reject at most fifteen boxes of twelve triangles. That trade is
// the wrong way round. The rings and the hull are culled because their transforms are already CPU-side
// (see core/frustum.js); if satellites ever become numerous enough to matter, the answer is to move the
// orbit to a compute pass and cull on the GPU, not to duplicate it.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "mesh_vertex.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "reflect.wgsl"
//!include "satellite.wgsl"

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp        : vec3f,
  @location(1) wn        : vec3f,
  @location(2) viewZ     : f32,
  @location(3) prevClip  : vec4f,
  @location(4) prevViewZ : f32,
  // The vertex in its BOX's own space, in real units. Every surface feature is authored against this,
  // so it travels with the satellite instead of swimming across it as the thing rotates. It is exactly
  // what the intersector used to hand the shading as `local`.
  @location(5) local     : vec3f,
  @location(6) mat       : f32,
  @location(7) seed      : f32,
};

struct FOut {
  @location(0) colour : vec4f,   // rgb, alpha = linear view distance
  @location(1) motion : vec4f,   // xy pixel delta, z previous distance, w ownership
};

/// A `SatPart` is already a rigid frame plus a scale; this is the frame half of it.
fn satRigid(p : SatPart) -> MeshRigid {
  var b : MeshRigid;
  b.ax = p.ax;
  b.ay = p.ay;
  b.az = p.az;
  b.origin = p.centre;
  return b;
}

@vertex
fn vs(v : MeshVertex, @builtin(instance_index) inst : u32) -> VOut {
  // Three boxes per satellite, laid out so the instance number decomposes without any lookup.
  let sat = f32(inst / 3u);
  let part = inst % 3u;

  let now = satPart(sat, part, frame.camPos.w);
  // `misc.w` is dt. The orbits are analytic, so one frame ago is the same evaluation at an earlier
  // time — no stored transforms, no velocity extrapolation, and exact for both the orbital motion and
  // the arrays' sun-tracking rotation.
  let prev = satPart(sat, part, frame.camPos.w - frame.misc.w);

  // Unit cube -> this box's shape. Position scales; the normal does not need to, because both the box
  // and the scale are axis-aligned (see the header).
  var vv = v;
  vv.pos = v.pos * now.rad;
  let x = meshXform(vv, satRigid(now), satRigid(prev));

  var out : VOut;
  out.pos = x.clip;
  out.wp = x.wp;
  out.wn = x.wn;
  out.viewZ = x.viewZ;
  out.prevClip = x.prevClip;
  out.prevViewZ = x.prevViewZ;
  out.local = vv.pos;
  out.mat = f32(now.mat);
  out.seed = sat;
  return out;
}

@fragment
fn fs(in : VOut) -> FOut {
  // Renormalised: interpolation across a triangle shortens a normal, and a short normal biases every
  // dot product in the BRDF toward grazing. Flat across a box face, so this changes little here — but
  // it is the same one line everywhere and the exception is not worth the reader's attention.
  let N = normalize(in.wn);
  let V = normalize(frame.camPos.xyz - in.wp);

  var out : FOut;
  let col = shadeSatSurface(in.local, N, i32(in.mat + 0.5), in.seed, in.wp, V);
  out.colour = vec4f(max(col, vec3f(0.0)), max(in.viewZ, 1e-4));

  if (in.prevClip.w > 1e-4) {
    let prevPx = uvToPixel(ndcToUV(in.prevClip.xy / in.prevClip.w));
    // The jitter is added BACK, as every rasterised layer does: `in.pos.xy` is this fragment's pixel
    // centre, but the surface it covers is the one whose UNJITTERED projection is centre + jitter, and
    // the history is indexed by centres. `prevClip` is already unjittered.
    out.motion = vec4f(prevPx - (in.pos.xy + frame.jitter.xy), in.prevViewZ, in.viewZ);
  } else {
    out.motion = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  }
  return out;
}
