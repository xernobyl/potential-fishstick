// ---------------------------------------------------------------------------
// Planet mesh: UV sphere with mapBody surface extraction.
// ---------------------------------------------------------------------------

//!include "sdf.wgsl"

struct GridUniform {
  cameraPos : vec4f, gridRes : vec4f, rMinMax : vec4f,
  band : vec4f, bodyCentre : vec4f, invRes : vec4f, bodyR : vec4f,
};

@group(1) @binding(0) var<uniform> u : GridUniform;
@group(1) @binding(1) var<storage, read_write> vertices : array<f32>;
const STRIDE = 6u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let id = gid.x; let Nx = u32(u.gridRes.x); let Ny = u32(u.gridRes.y);
  if (id >= Nx * Ny) { return; }
  let col = id % Nx; let row = id / Nx;
  let phi = f32(col) * TAU * u.invRes.x; let theta = f32(row) * PI * u.invRes.y;
  let st = sin(theta); let ct = cos(theta); let sp = sin(phi); let cp = cos(phi);
  let dir = vec3f(st * cp, ct, st * sp);

  var r = u.bodyR.x;
  for (var i = 0; i < 8; i = i + 1) {
    let d = mapBody(dir * r);
    if (abs(d) < 0.0004) { break; }
    r = r - clamp(d, -0.6, 0.6);
    r = max(r, 0.15);
  }
  let wp = dir * r;
  let nor = calcNormal(wp);
  let rel = wp - u.cameraPos.xyz;
  let vi = id * STRIDE;
  vertices[vi] = rel.x; vertices[vi + 1u] = rel.y; vertices[vi + 2u] = rel.z;
  vertices[vi + 3u] = nor.x; vertices[vi + 4u] = nor.y; vertices[vi + 5u] = nor.z;
}
