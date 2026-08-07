// ---------------------------------------------------------------------------
// Tile classification.
//
// One thread per raymarch tile. Decides whether any ray in the tile could
// possibly reach the body's bounding sphere, and writes a flag the raymarch
// reads to skip the expensive march wholesale.
//
// Why per-tile FLAGS rather than an indirect dispatch over a compacted list:
// the sky and the satellites still have to be shaded everywhere, so the
// pass cannot simply not run for empty tiles. What actually costs is the march,
// not the workgroup launch — and because a whole 8x8 workgroup shares the flag,
// every invocation in it takes the same branch, so there is no divergence to pay
// for either. An indirect dispatch would add a compaction pass and a second
// full-screen pass for the sky to buy the same saving.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "sdf.wgsl"

@group(1) @binding(0) var<storage, read_write> tileFlags : array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let tiles = vec2u((vec2u(frame.res.xy) + u32(TILE) - 1u) / u32(TILE));
  if (gid.x >= tiles.x || gid.y >= tiles.y) { return; }

  let bound = bodyBound();
  // Inflate a little: a tile's own angular extent means a corner ray can graze what a centre ray
  // misses. There used to be an aperture term here as well, and it outlived the aperture - see the
  // note in uniforms.js on repurposing slots.
  let margin = bound * 0.06;
  let r = bound + margin;

  var visible = false;
  // Four corners plus the centre. With 8px tiles against a body covering a good
  // fraction of the screen, that is conservative enough in practice, and the
  // margin above absorbs the rest.
  for (var j = 0u; j < 5u; j++) {
    var off = vec2f(0.0);
    if (j == 0u) { off = vec2f(0.0, 0.0); }
    else if (j == 1u) { off = vec2f(f32(TILE), 0.0); }
    else if (j == 2u) { off = vec2f(0.0, f32(TILE)); }
    else if (j == 3u) { off = vec2f(f32(TILE), f32(TILE)); }
    else { off = vec2f(f32(TILE) * 0.5, f32(TILE) * 0.5); }

    let px = vec2f(gid.xy) * f32(TILE) + off;
    let ray = cameraRay(px);
    if (iSphere(ray.o, ray.d, r).y > 0.0) { visible = true; break; }
  }

  tileFlags[gid.y * tiles.x + gid.x] = select(0u, 1u, visible);
}
