// ---------------------------------------------------------------------------
// Planet Surface Nets / Dual Contouring — 3-pass cached-corner GPU isosurface.
//
// Pass 0 (clear_buffers):        reset indirect draw args + cell vertex IDs
// Pass 1 (eval_corners):         one SDF evaluation per grid corner → cornerBuffer
// Pass 2 (compute_vertices):     one vertex per surface cell → vertexBuffer
// Pass 3 (generate_faces):       quads as triangle pairs → indexBuffer
//
// Toggle `useDC` in the grid uniform to switch between:
//   Surface Nets (useDC == 0):  vertex = centroid of edge crossing midpoints
//   Dual Contouring (useDC != 0): vertex = QEF minimiser over crossing planes
//
// The cached-corner architecture means each grid corner is evaluated exactly
// once. At lateralRes=48, depthRes=32 that is ~80k evaluations instead of
// ~600k — an ~8× reduction that keeps the driver from timing out.
// ---------------------------------------------------------------------------

//!include "sdf.wgsl"

// ---- uniforms & buffers ----------------------------------------------------

struct GridUniform {
  /// World-space position of corner (0,0,0)
  origin     : vec4f,
  /// Camera world position, so vertices can be stored camera-relative
  cameraPos  : vec4f,
  /// World-space cell size along each axis (may differ: lateral vs depth)
  cellSizeX  : f32,
  cellSizeY  : f32,
  cellSizeZ  : f32,
  /// 0 = Surface Nets (centroid), 1 = Dual Contouring (QEF solve)
  useDC      : u32,
  /// Nx, Ny, Nz cell counts
  res        : vec4u,
  /// Nx+1, Ny+1, Nz+1 corner counts
  padRes     : vec4u,
  /// 0 = world-space grid (origin + cellSize), 1 = frustum (unproject via invViewProj)
  mode       : u32,
  /// NDC z at grid k=0 (far end of depth range, ~0 for reverse-Z infinite)
  zNearNDC   : f32,
  /// NDC z at grid k=Nz (near end of depth range)
  zFarNDC    : f32,
  /// Inverse view-projection for frustum mode (unused in world mode)
  invViewProj : mat4x4f,
};

/// Indirect draw arguments for indexed drawing, plus the vertex counter.
///
/// The first five fields are the standard D3D12/Vulkan/WebGPU indirect indexed
/// draw layout. `drawIndexedIndirect` reads exactly 20 bytes from the buffer
/// start, so `vertexCount` at offset 20 is invisible to the draw call.
struct DrawIndirect {
  indexCount    : atomic<u32>,  // written by Pass 3
  instanceCount : u32,          // always 1
  firstIndex    : u32,          // always 0
  baseVertex    : i32,          // always 0
  firstInstance : u32,          // always 0
  vertexCount   : atomic<u32>,  // written by Pass 2
};

/// Wireframe indirect draw args (line-list). No atomics — written once by pass 4.
struct WireIndirect {
  indexCount    : u32,
  instanceCount : u32,
  firstIndex    : u32,
  baseVertex    : i32,
  firstInstance : u32,
};

/// Each output vertex: camera-relative position (vec3) and world-space normal (vec3).
/// Camera-relative matches the convention in planet_raster.wgsl, which does
/// `wp = rel + du.cameraPos.xyz` in the vertex stage.
const VERTEX_STRIDE = 6u;

@group(1) @binding(0) var<uniform> grid : GridUniform;
@group(1) @binding(1) var<storage, read_write> corners    : array<f32>;
@group(1) @binding(2) var<storage, read_write> cellVerts  : array<i32>;
@group(1) @binding(3) var<storage, read_write> vertices   : array<f32>;
@group(1) @binding(4) var<storage, read_write> indices    : array<u32>;
@group(1) @binding(5) var<storage, read_write> drawArgs   : DrawIndirect;
@group(1) @binding(6) var<storage, read_write> wireIndices : array<u32>;
@group(1) @binding(7) var<storage, read_write> wireArgs    : WireIndirect;


// ---- helpers --------------------------------------------------------------

fn cornerIdx(i: u32, j: u32, k: u32) -> u32 {
  return (k * grid.padRes.y + j) * grid.padRes.x + i;
}

fn cellIdx(ci: u32, cj: u32, ck: u32) -> u32 {
  return (ck * grid.res.y + cj) * grid.res.x + ci;
}

fn cellSize() -> vec3f {
  return vec3f(grid.cellSizeX, grid.cellSizeY, grid.cellSizeZ);
}

fn cornerPos(i: u32, j: u32, k: u32) -> vec3f {
  if (grid.mode == 0u) {
    // World-space: axis-aligned cube
    return grid.origin.xyz + vec3f(f32(i), f32(j), f32(k)) * cellSize();
  }
  // Frustum: unproject grid corner from NDC to world space
  let ndcX = (f32(i) / f32(grid.res.x)) * 2.0 - 1.0;  // -1..1
  let ndcY = (f32(j) / f32(grid.res.y)) * 2.0 - 1.0;  // -1..1
  let ndcZ = grid.zFarNDC + (grid.zNearNDC - grid.zFarNDC) * f32(k) / f32(grid.res.z);
  let clip = grid.invViewProj * vec4f(ndcX, ndcY, ndcZ, 1.0);
  return clip.xyz / clip.w;
}

/// Safe corner read: out-of-bounds returns a large positive value (empty
/// space) so the surface stays inside the grid rather than wrapping around.
fn readCorner(i: u32, j: u32, k: u32) -> f32 {
  if (i >= grid.padRes.x || j >= grid.padRes.y || k >= grid.padRes.z) { return 1e6; }
  return corners[cornerIdx(i, j, k)];
}


// ============================================================================
// PASS 0 — CLEAR
// ============================================================================

/// 1D dispatch: ceil(totalCells / 64) workgroups.
/// Thread 0 resets indirect draw args; every thread clears one cell vertex slot.
@compute @workgroup_size(64)
fn clear_buffers(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u) {
    atomicStore(&drawArgs.indexCount, 0u);
    drawArgs.instanceCount = 1u;
    drawArgs.firstIndex    = 0u;
    drawArgs.baseVertex    = 0;
    drawArgs.firstInstance = 0u;
    atomicStore(&drawArgs.vertexCount, 0u);
  }
  let totalCells = grid.res.x * grid.res.y * grid.res.z;
  if (gid.x < totalCells) {
    cellVerts[gid.x] = -1;
  }
}


// ============================================================================
// PASS 1 — EVALUATE CORNERS
// ============================================================================

/// 3D dispatch: ceil(padRes / 4) each axis.
/// Each thread evaluates mapBody at exactly one grid corner.
@compute @workgroup_size(4, 4, 4)
fn eval_corners(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; let j = gid.y; let k = gid.z;
  if (i >= grid.padRes.x || j >= grid.padRes.y || k >= grid.padRes.z) { return; }
  corners[cornerIdx(i, j, k)] = mapBody(cornerPos(i, j, k));
}


// ============================================================================
// PASS 2 — COMPUTE DUAL VERTICES
// ============================================================================

/// 12 edges of a cell as (corner0, corner1) pairs.
/// Corner ordering: i = (c>>2)&1, j = (c>>1)&1, k = c&1.
const EDGE_PAIRS = array<vec2u, 12>(
  vec2u(0u, 4u), vec2u(1u, 5u), vec2u(2u, 6u), vec2u(3u, 7u),  // X
  vec2u(0u, 2u), vec2u(1u, 3u), vec2u(4u, 6u), vec2u(5u, 7u),  // Y
  vec2u(0u, 1u), vec2u(2u, 3u), vec2u(4u, 5u), vec2u(6u, 7u),  // Z
);

/// Solve the cell's quadratic error function for the dual vertex.
///
/// Minimises Σ (nᵢ·(x - pᵢ))² over the edge crossing planes in this cell.
/// Each crossing contributes its position `p` and the field's gradient `n`
/// (computed via calcNormal at p). The QEF is solved as normal equations
/// (A = Σ nnᵀ, b = Σ n(n·p)) with Tikhonov regularisation toward the mass
/// point. The result is clamped into the cell to prevent self-intersections.
///
/// Ported from src/scene/sdf/qef.js.
fn solveQef(
  count: u32,
  mass: vec3f,     // average of crossing positions (pre-computed)
  crossings: array<vec3f, 12>,
  cellMin: vec3f,
  cellSz: vec3f,
  bias: f32,
) -> vec3f {
  // Normal equations, with bias pulling toward the mass point.
  var a00 = bias; var a01 = 0.0; var a02 = 0.0;
  var a11 = bias; var a12 = 0.0; var a22 = bias;
  var b0 = bias * mass.x; var b1 = bias * mass.y; var b2 = bias * mass.z;

  for (var i = 0u; i < count; i++) {
    let p = crossings[i];
    // Normal from the field's gradient at the crossing point.
    // This costs 4 SDF evaluations per crossing — the QEF's main expense.
    let n = calcNormal(p);
    let d = dot(n, p);
    a00 += n.x * n.x; a01 += n.x * n.y; a02 += n.x * n.z;
    a11 += n.y * n.y; a12 += n.y * n.z; a22 += n.z * n.z;
    b0 += n.x * d; b1 += n.y * d; b2 += n.z * d;
  }

  // Explicit 3×3 inverse via cofactors. The determinant guard catches
  // the ill-conditioned case (nearly-flat face → rank-deficient system).
  // The bias term ensures the matrix is never exactly singular.
  let c00 = a11 * a22 - a12 * a12;
  let c01 = a02 * a12 - a01 * a22;
  let c02 = a01 * a12 - a02 * a11;
  let det = a00 * c00 + a01 * c01 + a02 * c02;

  var result: vec3f;
  if (abs(det) < 1e-12) {
    result = mass;  // singular — fall back to mass point
  } else {
    let c11 = a00 * a22 - a02 * a02;
    let c12 = a01 * a02 - a00 * a12;
    let c22 = a00 * a11 - a01 * a01;
    let inv = 1.0 / det;
    result.x = (c00 * b0 + c01 * b1 + c02 * b2) * inv;
    result.y = (c01 * b0 + c11 * b1 + c12 * b2) * inv;
    result.z = (c02 * b0 + c12 * b1 + c22 * b2) * inv;
  }

  // Clamp into the cell: a vertex that escapes its own cell can cross a
  // neighbour's, and the quads woven between them fold through each other.
  let cellMax = cellMin + cellSz;
  result = clamp(result, cellMin, cellMax);
  return result;
}

@compute @workgroup_size(4, 4, 4)
fn compute_vertices(@builtin(global_invocation_id) gid: vec3u) {
  let ci = gid.x; let cj = gid.y; let ck = gid.z;
  if (ci >= grid.res.x || cj >= grid.res.y || ck >= grid.res.z) { return; }

  // Read the 8 corner SDF values
  var cv: array<f32, 8>;
  for (var c = 0u; c < 8u; c++) {
    let di = (c >> 2u) & 1u; let dj = (c >> 1u) & 1u; let dk = c & 1u;
    cv[c] = readCorner(ci + di, cj + dj, ck + dk);
  }

  // Accumulate edge crossing midpoints (always computed — needed for both modes)
  var sumPos = vec3f(0.0);
  var crossingCount: u32 = 0u;

  // For Dual Contouring: collect crossing positions for the QEF solver
  var crossings: array<vec3f, 12>;

  let csz = cellSize();

  for (var e = 0u; e < 12u; e++) {
    let c0 = EDGE_PAIRS[e].x;
    let c1 = EDGE_PAIRS[e].y;
    let v0 = cv[c0];
    let v1 = cv[c1];

    if ((v0 > 0.0) == (v1 > 0.0)) { continue; }

    // Linear interpolation along the edge
    let denom = v0 - v1;
    let t = select(0.5, v0 / denom, abs(denom) > 1e-12);

    let di0 = (c0 >> 2u) & 1u; let dj0 = (c0 >> 1u) & 1u; let dk0 = c0 & 1u;
    let di1 = (c1 >> 2u) & 1u; let dj1 = (c1 >> 1u) & 1u; let dk1 = c1 & 1u;
    let p0 = cornerPos(ci + di0, cj + dj0, ck + dk0);
    let p1 = cornerPos(ci + di1, cj + dj1, ck + dk1);

    let cp = mix(p0, p1, t);
    sumPos += cp;

    // Store crossing position for the QEF path. The normal is computed
    // inside solveQef via calcNormal at each crossing.
    crossings[crossingCount] = cp;
    crossingCount += 1u;
  }

  if (crossingCount == 0u) { return; }

  let inv = 1.0 / f32(crossingCount);
  let mass = sumPos * inv;  // centroid of crossings — Surface Nets vertex

  var vertexPos: vec3f;
  if (grid.useDC != 0u) {
    // Dual Contouring: solve QEF to find the minimiser
    let cellMin = cornerPos(ci, cj, ck);
    vertexPos = solveQef(crossingCount, mass, crossings, cellMin, csz, 0.1);
  } else {
    // Surface Nets: vertex at the centroid of edge crossing midpoints
    vertexPos = mass;
  }

  // Normal from the field's gradient at the solved position
  let nor = calcNormal(vertexPos);

  // Atomically reserve a compact vertex slot
  let slot = atomicAdd(&drawArgs.vertexCount, 1u);
  cellVerts[cellIdx(ci, cj, ck)] = i32(slot);

  // Write vertex data — camera-relative position (matching planet_raster.wgsl convention)
  let base = slot * VERTEX_STRIDE;
  vertices[base]     = vertexPos.x - grid.cameraPos.x;
  vertices[base + 1u] = vertexPos.y - grid.cameraPos.y;
  vertices[base + 2u] = vertexPos.z - grid.cameraPos.z;
  vertices[base + 3u] = nor.x;
  vertices[base + 4u] = nor.y;
  vertices[base + 5u] = nor.z;
}


// ============================================================================
// PASS 3 — GENERATE FACES
// ============================================================================

/// Look up the vertex index for cell (ci, cj, ck). Returns -1 for out-of-bounds
/// or empty cells.
fn cellVertId(ci: i32, cj: i32, ck: i32) -> i32 {
  if (ci < 0 || cj < 0 || ck < 0) { return -1; }
  if (ci >= i32(grid.res.x) || cj >= i32(grid.res.y) || ck >= i32(grid.res.z)) { return -1; }
  return cellVerts[cellIdx(u32(ci), u32(cj), u32(ck))];
}

/// Emit two triangles forming a quad (v0,v1,v2,v3) into the index buffer.
/// `flip` selects winding direction: when true the diagonal goes the other way
/// so the normal faces the opposite direction.
fn emitQuad(v0: i32, v1: i32, v2: i32, v3: i32, flip: bool) {
  if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) { return; }

  let base = atomicAdd(&drawArgs.indexCount, 6u);
  let u0 = u32(v0); let u1 = u32(v1); let u2 = u32(v2); let u3 = u32(v3);

  if (!flip) {
    indices[base]     = u0; indices[base + 1u] = u1; indices[base + 2u] = u2;
    indices[base + 3u] = u0; indices[base + 4u] = u2; indices[base + 5u] = u3;
  } else {
    indices[base]     = u0; indices[base + 1u] = u2; indices[base + 2u] = u1;
    indices[base + 3u] = u0; indices[base + 4u] = u3; indices[base + 5u] = u2;
  }
}

@compute @workgroup_size(4, 4, 4)
fn generate_faces(@builtin(global_invocation_id) gid: vec3u) {
  let ci = gid.x; let cj = gid.y; let ck = gid.z;
  if (ci >= grid.res.x || cj >= grid.res.y || ck >= grid.res.z) { return; }

  let ici = i32(ci); let icj = i32(cj); let ick = i32(ck);

  // Read the 8 corner SDF values; corners 0/4, 0/2, 0/1 used for edge tests
  var cv: array<f32, 8>;
  for (var c = 0u; c < 8u; c++) {
    let di = (c >> 2u) & 1u; let dj = (c >> 1u) & 1u; let dk = c & 1u;
    cv[c] = readCorner(ci + di, cj + dj, ck + dk);
  }

  // ---- X-edge (anchor corner 0 → corner 4, along +X) ----
  // Perpendicular pair (Y,Z) — right-handed cycle.
  // CPU reference order: (x, y-1, z-1), (x, y, z-1), (x, y, z), (x, y-1, z)
  if ((cv[0u] > 0.0) != (cv[4u] > 0.0)) {
    let v0 = cellVertId(ici,     icj - 1, ick - 1);
    let v1 = cellVertId(ici,     icj,     ick - 1);
    let v2 = cellVertId(ici,     icj,     ick    );
    let v3 = cellVertId(ici,     icj - 1, ick    );
    emitQuad(v0, v1, v2, v3, cv[0u] > 0.0);
  }

  // ---- Y-edge (anchor corner 0 → corner 2, along +Y) ----
  // Perpendicular pair (Z,X) — right-handed cycle.
  // CPU reference order: (x-1, y, z-1), (x-1, y, z), (x, y, z), (x, y, z-1)
  if ((cv[0u] > 0.0) != (cv[2u] > 0.0)) {
    let v0 = cellVertId(ici - 1, icj,     ick - 1);
    let v1 = cellVertId(ici - 1, icj,     ick    );
    let v2 = cellVertId(ici,     icj,     ick    );
    let v3 = cellVertId(ici,     icj,     ick - 1);
    emitQuad(v0, v1, v2, v3, cv[0u] > 0.0);
  }

  // ---- Z-edge (anchor corner 0 → corner 1, along +Z) ----
  // Perpendicular pair (X,Y) — right-handed cycle.
  // CPU reference order: (x-1, y-1, z), (x, y-1, z), (x, y, z), (x-1, y, z)
  if ((cv[0u] > 0.0) != (cv[1u] > 0.0)) {
    let v0 = cellVertId(ici - 1, icj - 1, ick    );
    let v1 = cellVertId(ici,     icj - 1, ick    );
    let v2 = cellVertId(ici,     icj,     ick    );
    let v3 = cellVertId(ici - 1, icj,     ick    );
    emitQuad(v0, v1, v2, v3, cv[0u] > 0.0);
  }
}


// ============================================================================
// PASS 4 — GENERATE WIREFRAME (debug view)
// ============================================================================

/// 1D dispatch: ceil(triangleCount / 64) workgroups.
/// Reads the triangle index buffer and emits line-list indices: each triangle's
/// three edges become three line pairs (6 indices per triangle, 2× expansion).
///
/// Runs after pass 3 so `drawArgs.indexCount` is fully written; `atomicLoad`
/// reads the final triangle index count.
@compute @workgroup_size(64)
fn generate_wireframe(@builtin(global_invocation_id) gid: vec3u) {
  let triCount = atomicLoad(&drawArgs.indexCount) / 3u;
  if (gid.x >= triCount) { return; }

  let triBase = gid.x * 3u;
  let a = indices[triBase];
  let b = indices[triBase + 1u];
  let c = indices[triBase + 2u];

  let outBase = gid.x * 6u;
  wireIndices[outBase]     = a;
  wireIndices[outBase + 1u] = b;
  wireIndices[outBase + 2u] = b;
  wireIndices[outBase + 3u] = c;
  wireIndices[outBase + 4u] = c;
  wireIndices[outBase + 5u] = a;

  // First thread writes the indirect draw args for the line-list draw
  if (gid.x == 0u) {
    wireArgs.indexCount    = triCount * 6u;
    wireArgs.instanceCount = 1u;
    wireArgs.firstIndex    = 0u;
    wireArgs.baseVertex    = 0;
    wireArgs.firstInstance = 0u;
  }
}
