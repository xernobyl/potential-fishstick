// Does the Fibonacci-spiral normal encoding round-trip, and is its error as small as claimed?
//
// The SurfaceNets planet mesh can store its per-vertex normal as ONE index into an n-point
// spherical-Fibonacci spiral (see SN_NORMAL in src/scene/tuning.js and `fibNearestIndex` /
// `fibDecodeNormal` in shaders/fibonacci.wgsl). This transcribes those two WGSL functions and
// checks the properties the encoding is chosen FOR: the inverse is exact (it finds the true
// nearest spiral point), decoding is unit-length, and 16 bits is enough precision that no normal
// is ever more than a fraction of a degree off.
//
// The encode/decode live in WGSL and cannot be imported here, so this mirrors them — the same
// transcription discipline dev/meshgen.mjs uses for rings.wgsl. Keep the two in step.

const TAU = Math.PI * 2;
const PHI = (1 + Math.sqrt(5)) / 2;
const PI = Math.PI;

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

// ---- transcribed from shaders/fibonacci.wgsl ------------------------------

const SFWIN = [
  [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
  [1, 1], [1, -1], [-1, 1], [-1, -1],
];

const fract = (x) => x - Math.floor(x);

/** sfPointFwd — index -> unit direction. */
function decode(i, n) {
  const m = 1.0 - 1.0 / n;
  const ph = TAU * fract(i * PHI);
  const ct = m - 2.0 * i / n;
  const st = Math.sqrt(Math.max(1.0 - ct * ct, 0.0));
  return [Math.cos(ph) * st, Math.sin(ph) * st, ct];
}

/** fibNearestIndex — unit-ish direction -> nearest spiral index. */
function encode(d, n) {
  const ct = Math.max(-1.0, Math.min(1.0, d[2]));
  const s2 = Math.max(1.0 - ct * ct, 1e-7);
  const m = 1.0 - 1.0 / n;
  const k = Math.max(2.0, Math.floor(Math.log(n * PI * Math.sqrt(5.0) * s2) / Math.log(PHI + 1.0)));
  const Fk = Math.pow(PHI, k) / Math.sqrt(5.0);
  const F0 = Math.floor(Fk + 0.5);
  const F1 = Math.floor(Fk * PHI + 0.5);
  const ka = [2.0 * F0 / n, 2.0 * F1 / n];
  const kb = [
    TAU * (fract((F0 + 1.0) * PHI) - (PHI - 1.0)),
    TAU * (fract((F1 + 1.0) * PHI) - (PHI - 1.0)),
  ];
  const det = ka[1] * kb[0] - ka[0] * kb[1];
  // iB = [[ka.y, kb.y], [-ka.x, -kb.x]] / det, then cc = floor(iB * (az, ct - m)).
  const az = Math.atan2(d[1], d[0]);
  const cc0 = Math.floor((ka[1] / det) * az + (kb[1] / det) * (ct - m));
  const cc1 = Math.floor((-ka[0] / det) * az + (-kb[0] / det) * (ct - m));

  let best = 0;
  let bestDot = -2.0;
  for (const [w0, w1] of SFWIN) {
    const i = F0 * (cc0 + w0) + F1 * (cc1 + w1);
    if (i < 0.0 || i > n - 0.5) continue;
    const dd = decode(i, n);
    const q = dd[0] * d[0] + dd[1] * d[1] + dd[2] * d[2];
    if (q > bestDot) { bestDot = q; best = i; }
  }
  return best;
}

// ---- helpers ---------------------------------------------------------------

function randomDir() {
  const z = Math.random() * 2 - 1;
  const th = Math.random() * TAU;
  const r = Math.sqrt(1 - z * z);
  return [r * Math.cos(th), r * Math.sin(th), z];
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(a, b))));

// ---- the inverse finds the true nearest point ------------------------------
for (const n of [16, 64, 256]) {
  const pts = Array.from({ length: n }, (_, i) => decode(i, n));
  let wrong = 0;
  const SAMPLES = 20000;
  for (let s = 0; s < SAMPLES; s++) {
    const d = randomDir();
    const e = encode(d, n);
    let bi = 0;
    let ba = Infinity;
    for (let i = 0; i < n; i++) {
      const a = angle(pts[i], d);
      if (a < ba) { ba = a; bi = i; }
    }
    if (e !== bi) wrong++;
  }
  check(wrong === 0, `inverse is exact (matches brute-force nearest) at n=${n}`,
        `${wrong}/${SAMPLES} mismatches`);
}

// ---- decode is unit length --------------------------------------------------
{
  const n = 1 << 16;
  let worst = 0;
  for (let i = 0; i < 100000; i++) {
    const d = decode(Math.floor(Math.random() * n), n);
    worst = Math.max(worst, Math.abs(Math.hypot(d[0], d[1], d[2]) - 1));
  }
  check(worst < 1e-6, 'every decoded direction is unit length', `worst |len-1| ${worst.toExponential(2)}`);
}

// ---- 16-bit precision -------------------------------------------------------
{
  const n = 1 << 16;
  let maxErr = 0;
  let sum = 0;
  const SAMPLES = 200000;
  for (let s = 0; s < SAMPLES; s++) {
    const d = randomDir();
    const dd = decode(encode(d, n), n);
    const a = angle(d, dd);
    sum += a;
    if (a > maxErr) maxErr = a;
  }
  const deg = (r) => (r * 180) / Math.PI;
  check(maxErr < 0.6 * (Math.PI / 180), '16-bit max error is under 0.6 deg',
        `max ${deg(maxErr).toFixed(4)} deg, mean ${deg(sum / SAMPLES).toFixed(4)} deg`);
}

// ---- index round-trip: decode(i) -> encode -> i -----------------------------
{
  const n = 1 << 16;
  let mismatch = 0;
  for (let s = 0; s < 100000; s++) {
    const i = Math.floor(Math.random() * n);
    if (encode(decode(i, n), n) !== i) mismatch++;
  }
  check(mismatch === 0, 'decode(encode) is the identity on indices', `${mismatch}/100000 mismatches`);
}

process.exit(failed ? 1 : 0);
