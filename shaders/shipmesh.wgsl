// ---------------------------------------------------------------------------
// The ship's hull: a generated triangle mesh, rasterised into the solid layer.
//
// This replaces a marched SDF. The shape is now data (see scene/ship_sdf.js), dual contoured on the CPU
// at startup, and drawn like any other generated mesh — which buys four things the marched version could
// not have. It costs one draw call instead of a per-pixel sphere-trace with its own step loop and bound
// test. It occludes against hardware depth rather than by threading `tmax` through the march. It gets an
// EXACT motion vector from its own rigid transform, where the marched version derived one per pixel from
// a hit point. And it is culled, both by frustum and by back face, neither of which a marched primitive
// can be.
//
// THE HULL IS PAINTED, in the fragment stage, from the vertex's OBJECT-SPACE position. Panel lines, a
// per-panel tint, a glass canopy and a stripe along the wings — the same paint the marched hull carried,
// from the same palette in tuning.js, so this reads as the same ship rather than as a new one that
// happens to be the same shape. A flat constant material lived here briefly while the mesher was being
// trusted; that was scaffolding, and it went out when the geometry stopped being the thing in question.
//
// IN OBJECT SPACE, WHICH IS THE WHOLE TRICK. The paint is a function of where a point sits on the hull,
// so it travels with the ship exactly — no projection to pick, no seam to hide, and nothing to swim when
// the ship rolls. The mesh's positions are in world units, so the varying divides by SHIP_SCALE once
// and every threshold below is in the units ship_sdf.js authors in.
//
// AND IT IS BETTER PAINT THAN THE MARCHED VERSION HAD, in one specific way. Masking the panel seams needs
// to exclude the axis a face points along, or that face seams all at once instead of being divided; the
// marched shader approximated that with the normalised POSITION, because a per-pixel field normal was
// expensive there. A mesh carries its object-space normal for free, so the mask is the real thing —
// the same `1 - |n|` idiom the satellite bus uses.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "mesh_vertex.wgsl"
//!include "sky.wgsl"
//!include "brdf.wgsl"
//!include "explosion.wgsl"

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp        : vec3f,
  @location(1) wn        : vec3f,
  @location(2) viewZ     : f32,
  @location(3) prevClip  : vec4f,
  @location(4) prevViewZ : f32,
  /// The vertex in AUTHORED body units — object space divided by the build scale. Every feature
  /// threshold in the fragment stage is in these, matching ship_sdf.js one for one.
  @location(5) local     : vec3f,
  /// The object-space normal, for the seam mask. Not the world normal: the mask must not rotate.
  @location(6) lnor      : vec3f,
};

struct FOut {
  @location(0) colour : vec4f,   // rgb, alpha = linear view distance
  @location(1) motion : vec4f,   // xy pixel delta, z previous distance, w ownership
};

@vertex
fn vs(v : MeshVertex) -> VOut {
  // The hull is rigid, so its transform is the ship's own pose — position and orientation quaternion,
  // both already in the frame uniform because the marched version needed them too. The PREVIOUS pose is
  // there for the same reason, and is what makes the motion vector exact rather than reconstructed.
  let now = meshRigidFromQuat(frame.shipPos.xyz, frame.shipRot);
  let prev = meshRigidFromQuat(frame.prevShipPos.xyz, frame.prevShipRot);
  let x = meshXform(v, now, prev);

  var out : VOut;
  out.pos = x.clip;
  out.wp = x.wp;
  out.wn = x.wn;
  out.viewZ = x.viewZ;
  out.prevClip = x.prevClip;
  out.prevViewZ = x.prevViewZ;
  out.local = v.pos / SHIP_SCALE;
  out.lnor = v.nor;
  return out;
}

@fragment
fn fs(in : VOut) -> FOut {
  // Renormalised: interpolation across a triangle shortens a normal, and a short normal biases every
  // dot product in the BRDF toward grazing.
  let N = normalize(in.wn);
  let V = normalize(frame.camPos.xyz - in.wp);

  // ---- paint, all of it in authored body units ----
  let al = in.local;

  // Panel lines. `floor` of the scaled position gives a cell index, and hashing it gives each panel its
  // own slightly different paint batch — which is most of what stops a hull reading as one moulded piece.
  let g = al * SHIP_PANEL;
  let ph = hash13(floor(g));

  // The seam between panels. Weighting by (1 - |n|) drops the axis this face points along: that
  // coordinate barely varies across the face, so leaving it in would flash the whole face as one seam
  // instead of ruling it. Object-space normal, so the ruling does not rotate with the ship.
  let w = 1.0 - abs(normalize(in.lnor));
  let e = abs(fract(g) - 0.5) * w;
  let seam = smoothstep(0.40, 0.50, max(max(e.x, e.y), e.z));

  // The canopy is glass. Tested against the blister's own sphere from ship_sdf.js rather than against a
  // slab, so the region follows the shape it belongs to: inside the sphere AND on its upper dome, which
  // leaves the blended skirt where it meets the spine as hull.
  let toCanopy = al - vec3f(0.0, 0.065, 0.10);
  let isGlass = smoothstep(0.082, 0.068, length(toCanopy)) * smoothstep(0.052, 0.072, al.y);

  // A painted stripe out along each wing, because a readable silhouette wants a graphic. The wings run
  // from |x| ~ 0.01 to ~ 0.39 and are thin in y, so the gate keeps it off the fuselage and off the fin.
  let stripe = smoothstep(0.032, 0.0, abs(abs(al.x) - 0.27))
             * step(0.14, abs(al.x)) * step(abs(al.y), 0.05);

  var alb = mix(SHIP_HULL * (0.85 + 0.30 * ph), SHIP_GLASS, isGlass);
  alb = mix(alb, alb * 0.45, seam);
  alb = mix(alb, SHIP_TRIM, stripe * (1.0 - isGlass));

  // Painted panels over metal: a dielectric f0, not a conductor's. The glass is smoother and slightly
  // brighter-reflecting, which is nearly all of what makes it read as glass at this size.
  let f0 = mix(vec3f(0.06), vec3f(0.09), isGlass);
  let rough = mix(clamp(0.34 + 0.22 * ph + 0.25 * seam, 0.06, 1.0), 0.08, isGlass);

  var col = sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, 1.0)
          + sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0)
          + blastLight(in.wp, N, V, alb, rough, f0);

  // The sky as the environment term, with no marched reflection. The rings march one because they are
  // polished enough that the body dominates what they show; this hull is rough enough that a single
  // mirror ray would be both wrong and noisy, which is the same reasoning that fades the rings' march
  // out as their roughness rises — here it starts faded.
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  col += background(reflect(-V, N)) * fresnelSchlickRough(NoV, f0, rough) * SHIPM_ENV;

  // The planet's core throws real light at this range, and leaving it out makes the hull look pasted on
  // against a bright body. Falls off with distance from the origin, which is where the core is.
  let toCore = -normalize(in.wp);
  col += SHIPM_CORE * max(dot(N, toCore), 0.0) / max(dot(in.wp, in.wp), 1.0);

  var out : FOut;
  out.colour = vec4f(max(col, vec3f(0.0)), max(in.viewZ, 1e-4));

  if (in.prevClip.w > 1e-4) {
    let prevPx = uvToPixel(ndcToUV(in.prevClip.xy / in.prevClip.w));
    // The jitter is added BACK, for the same reason the rings do it: `in.pos.xy` is this fragment's
    // pixel centre, but the surface it covers is the one whose UNJITTERED projection is centre + jitter,
    // and the history is indexed by centres. `prevClip` is already unjittered.
    out.motion = vec4f(prevPx - (in.pos.xy + frame.jitter.xy), in.prevViewZ, in.viewZ);
  } else {
    out.motion = vec4f(MOTION_NONE, 0.0, 0.0, 0.0);
  }
  return out;
}
