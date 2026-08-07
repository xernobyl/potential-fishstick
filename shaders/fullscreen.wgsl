// ---------------------------------------------------------------------------
// Fullscreen triangle.
//
// Three vertices, no vertex buffer, no index buffer: the positions are derived
// from the vertex index. Cheaper to set up than a quad and it avoids the
// diagonal-seam quantisation a two-triangle quad can introduce.
// ---------------------------------------------------------------------------

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> FSOut {
  // (0,0) (2,0) (0,2) in clip space -> covers [-1,1] with one triangle
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out : FSOut;
  out.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);      // texture space: y down
  return out;
}
