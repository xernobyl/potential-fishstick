/**
 * 4x4 matrix math, column-major to match WGSL's `mat4x4f` memory layout
 * (element [col][row] lives at index col*4 + row).
 *
 * Only what the camera needs. Both inverses here are CLOSED FORM rather than a
 * general cofactor inverse: the view and projection matrices have known
 * structure, so inverting them by hand is exact, cheaper, and — the real reason —
 * has no near-singular case to get wrong.
 *
 * Depth is REVERSE-Z and INFINITE: NDC z is 1 at the near plane and approaches 0 at
 * infinity. WebGPU's clip volume is 0 <= z_clip <= w_clip, and this projection makes
 * z_clip the constant NEAR while w_clip is the view-space z — so the left test is always
 * satisfied and the right one reduces to `z_view >= NEAR`. That is exactly near-plane
 * clipping with no far-plane clipping at all, which is the point: there is no far plane to
 * choose, and therefore no far plane to get wrong.
 *
 * Reverse-Z is the standard pairing for a FLOATING-POINT depth buffer, where the float's
 * dense precision near 0 cancels the projection's hyperbolic crowding near the eye. This
 * renderer gets no measurable benefit from that today — its only hardware depth buffer is
 * the rings' `depth24plus`, which is fixed point over a range of about 1.75 to 3 units and is
 * discarded the moment the pass ends, and every other depth comparison in the pipeline works
 * on LINEAR view distance carried in an alpha channel rather than on NDC z. It is adopted for
 * the infinite far plane and because it is correct by default if a float depth target ever
 * appears; the precision argument is noted so nobody later assumes it was the reason.
 */

export const NEAR = 0.05;

export const create = () => new Float32Array(16);

/** out = a * b (both column-major). `out` may alias neither input. */
export function mul(out, a, b) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b0 + a[4 + r] * b1 + a[8 + r] * b2 + a[12 + r] * b3;
    }
  }
  return out;
}

/**
 * World -> view. The rows are the camera basis, so this is a rotation transpose
 * plus a translation, which is why no inverse-transpose is needed anywhere.
 */
export function view(out, b) {
  const { right: r, up: u, fwd: f, pos: p } = b;
  out[0] = r[0]; out[4] = r[1]; out[8] = r[2]; out[12] = -dot(r, p);
  out[1] = u[0]; out[5] = u[1]; out[9] = u[2]; out[13] = -dot(u, p);
  out[2] = f[0]; out[6] = f[1]; out[10] = f[2]; out[14] = -dot(f, p);
  out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
  return out;
}

/** View -> world: the basis as columns, translation restored. */
export function viewInverse(out, b) {
  const { right: r, up: u, fwd: f, pos: p } = b;
  out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = 0;
  out[4] = u[0]; out[5] = u[1]; out[6] = u[2]; out[7] = 0;
  out[8] = f[0]; out[9] = f[1]; out[10] = f[2]; out[11] = 0;
  out[12] = p[0]; out[13] = p[1]; out[14] = p[2]; out[15] = 1;
  return out;
}

/**
 * Perspective projection with a shifted principal point.
 *
 * `sx`/`sy` are the sensor half-extents in the shared screen space (see
 * `Camera.screen`), so the field of view is set once against the DIAGONAL and
 * the aspect ratio only decides how that fixed diagonal is divided between the
 * axes. A 21:9 window and a 9:21 window therefore frame the subject identically.
 *
 * `offset` shifts the framing in the same units — a fraction of the half
 * diagonal — so the off-centre composition survives a change of aspect too.
 */
export function projection(out, focal, offset, sx, sy) {
  out.fill(0);
  out[0] = focal / sx;
  out[5] = focal / sy;
  out[8] = -offset[0] / sx;
  out[9] = -offset[1] / sy;
  // The whole depth mapping, and it is this short only because the far plane is at infinity:
  //   z_clip = NEAR   (constant, row 2 is (0, 0, 0, NEAR))
  //   w_clip = z_view (row 3 is (0, 0, 1, 0))
  //   => NDC z = NEAR / z_view, which is 1 at the near plane and tends to 0 with distance.
  // Still a genuine invertible mapping despite the constant row: rows 2 and 3 are
  // independent, so `projectionInverse` below is exact rather than a pseudo-inverse.
  out[10] = 0;
  out[11] = 1;
  out[14] = NEAR;
  return out;
}

/** Exact inverse of `projection`, for turning a pixel back into a world ray. */
export function projectionInverse(out, focal, offset, sx, sy) {
  // Solved by hand from the forward matrix above, given clip (cx, cy, cz, cw):
  //   z_view = cw                        (row 3 of the forward matrix)
  //   w_out  = cz / NEAR                 (row 2)
  //   x_view = cx * sx/focal + cw * offset.x/focal
  //   y_view = cy * sy/focal + cw * offset.y/focal
  // Note w_out comes from cz, so unprojecting at NDC z = 0 yields w = 0 — a DIRECTION rather
  // than a point, which is precisely what ray generation wants at infinity.
  out.fill(0);
  out[0] = sx / focal;
  out[5] = sy / focal;
  out[11] = 1 / NEAR;
  out[12] = offset[0] / focal;
  out[13] = offset[1] / focal;
  out[14] = 1;
  return out;
}

/**
 * Transform (v, w) and divide through. Returns null when the result is behind
 * the camera, which callers must handle — a silent divide by a negative w puts
 * things on screen mirrored through the centre.
 */
export function project(m, v, w, out) {
  const cw = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * w;
  if (cw <= 1e-6) return null;
  const cx = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * w;
  const cy = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * w;
  out[0] = cx / cw;
  out[1] = cy / cw;
  return out;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
