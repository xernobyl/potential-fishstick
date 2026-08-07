/**
 * The per-layer rotations that decorrelate the spherical-Fibonacci lattices.
 *
 * These are eight fixed matrices, and computing them on the GPU was costing more than
 * everything they are used for. `sfCell` needs one, the field calls `sfCell` six times per
 * evaluation (three sphere layers, three hole layers), and the march evaluates the field
 * ~60 times per pixel — so a pair of sin/cos and a 3x3 multiply that could have been a
 * table lookup was running on the order of 360 times per pixel to produce the same eight
 * answers every frame.
 *
 * So they are computed here, once, and injected as WGSL constants. The generating
 * constants moved here with them: this file is now the single definition, rather than JS
 * holding a copy of numbers that live in the shader.
 *
 * Slots: 0..2 the sphere octaves, 4..6 their matching hole layers, 7 the starfield.
 * (3 is unused — the hole layers are offset by 4 so the two families cannot collide.)
 */

/** Layers to generate. One more than the highest slot any shader asks for. */
export const LAYER_SLOTS = 8;

/**
 * Rotation schedule. Irrational-ish multipliers so no two layers line up.
 * Exported because the shader keeps a computed fallback path (PROBE.latticeTable) and the
 * two must agree exactly — a table that disagreed with its fallback would make the A/B
 * between them compare two different scenes.
 */
export const SCHEDULE = {
  yawStep: 2.39996,
  yawBias: 1.7,
  pitchStep: 1.732,
  pitchBias: 0.6,
};
const { yawStep: YAW_STEP, yawBias: YAW_BIAS,
        pitchStep: PITCH_STEP, pitchBias: PITCH_BIAS } = SCHEDULE;

// Matrices are flat COLUMN-major, matching WGSL's mat3x3f(col0, col1, col2).
const rotY = (a) => {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, -s, 0, 1, 0, s, 0, c];
};
const rotX = (a) => {
  const c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, c, -s, 0, s, c];
};

/** A*B, column-major: column j of the result is A times column j of B. */
function mul(A, B) {
  const out = new Array(9);
  for (let j = 0; j < 3; j++) {
    const bx = B[j * 3], by = B[j * 3 + 1], bz = B[j * 3 + 2];
    for (let i = 0; i < 3; i++) {
      out[j * 3 + i] = A[i] * bx + A[3 + i] * by + A[6 + i] * bz;
    }
  }
  return out;
}

const transpose = (M) => {
  const out = new Array(9);
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) out[j * 3 + i] = M[i * 3 + j];
  return out;
};

function layerRot(i) {
  return mul(rotY(i * YAW_STEP + YAW_BIAS), rotX(i * PITCH_STEP + PITCH_BIAS));
}

/**
 * Both families, forward and inverse.
 *
 * The inverse is the TRANSPOSE, which is only true for an orthonormal matrix — so that
 * property is asserted rather than assumed. A silent indexing slip in `mul` would produce
 * a matrix that still looks plausible and would shear the whole lattice, which reads as
 * the field being subtly wrong everywhere rather than as an obvious failure.
 */
export function layerRotations() {
  const fwd = [];
  const inv = [];
  for (let i = 0; i < LAYER_SLOTS; i++) {
    const M = layerRot(i);
    assertOrthonormal(M, i);
    fwd.push(M);
    inv.push(transpose(M));
  }
  return { fwd, inv };
}

function assertOrthonormal(M, i) {
  const col = (j) => [M[j * 3], M[j * 3 + 1], M[j * 3 + 2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const c0 = col(0), c1 = col(1), c2 = col(2);
  const err = Math.max(
    Math.abs(dot(c0, c0) - 1), Math.abs(dot(c1, c1) - 1), Math.abs(dot(c2, c2) - 1),
    Math.abs(dot(c0, c1)), Math.abs(dot(c0, c2)), Math.abs(dot(c1, c2)),
  );
  // det, for handedness: a reflection would mirror the lattice rather than rotate it.
  const det = c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
            - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
            + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
  if (err > 1e-12 || Math.abs(det - 1) > 1e-12) {
    throw new Error(`layer rotation ${i} is not a rotation (orthonormality ${err}, det ${det})`);
  }
}

/** One matrix as a WGSL mat3x3f literal. */
export function wgslMat3(M, f) {
  const c = (j) => `vec3f(${f(M[j * 3])}, ${f(M[j * 3 + 1])}, ${f(M[j * 3 + 2])})`;
  return `mat3x3f(${c(0)}, ${c(1)}, ${c(2)})`;
}
