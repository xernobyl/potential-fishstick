// Is the polar rejection bound STRICT?
//
// `sfDotMax` claims that lct*ict + lst*ist is an upper bound on dot(lp, lcw) over every azimuth,
// so rejecting on it can never discard a candidate that mattered. If that claim is ever wrong,
// the field loses a sphere it should have had — which the marcher renders as a hard seam, and
// which no image metric would attribute to this. So it is checked directly, against a
// reimplementation of the same arithmetic, over the parameter ranges the shader actually uses.
//
// This mirrors shaders/fibonacci.wgsl: sfCell / sfPolar / sfDotMax / sfDirLocal.

const PHI = 1.61803398874989;
const TAU = 6.28318530718;

// A deterministic PRNG, so a failure is reproducible.
let seed = 0x2545f491;
const rnd = () => {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >>> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed / 4294967296;
};

function cell(dir, n) {
  // No layer rotation: the bound is a statement about one frame, and `lp` is already the
  // rotated query, so rotating here would only test the rotation.
  const ct = Math.min(1, Math.max(-1, dir[2]));
  const s2 = Math.max(1 - ct * ct, 1e-7);
  const m = 1 - 1 / n;
  const k = Math.max(2, Math.floor(Math.log(n * Math.PI * Math.sqrt(5) * s2) / Math.log(PHI + 1)));
  const Fk = Math.pow(PHI, k) / Math.sqrt(5);
  const F = [Math.floor(Fk + 0.5), Math.floor(Fk * PHI + 0.5)];
  const ka = [2 * F[0] / n, 2 * F[1] / n];
  const fr = (x) => x - Math.floor(x);
  const kb = [TAU * (fr((F[0] + 1) * PHI) - (PHI - 1)), TAU * (fr((F[1] + 1) * PHI) - (PHI - 1))];
  const det = ka[1] * kb[0] - ka[0] * kb[1];
  // mat2x2f(vec2f(ka.y,-ka.x), vec2f(kb.y,-kb.x)) * (1/det), column-major, times (atan2, ct-m)
  const a = Math.atan2(dir[1], dir[0]), b = ct - m;
  const iB = [[ka[1] / det, -ka[0] / det], [kb[1] / det, -kb[0] / det]];
  const cc = [Math.floor(iB[0][0] * a + iB[1][0] * b), Math.floor(iB[0][1] * a + iB[1][1] * b)];
  return { F, cc, m, n, lxy: Math.hypot(dir[0], dir[1]), lp: dir };
}

const WIN = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];

function polar(c, j) {
  const i = c.F[0] * (WIN[j][0] + c.cc[0]) + c.F[1] * (WIN[j][1] + c.cc[1]);
  if (i < 0 || i > c.n - 0.5) return null;
  const ict = c.m - 2 * i / c.n;
  return { i, ict, ist: Math.sqrt(Math.max(1 - ict * ict, 0)) };
}

const dotMax = (c, p) => c.lp[2] * p.ict + c.lxy * p.ist;

function dirLocal(p) {
  const ip = TAU * (p.i * PHI - Math.floor(p.i * PHI));
  return [Math.cos(ip) * p.ist, Math.sin(ip) * p.ist, p.ict];
}

// The layer densities the field actually uses: N0 * NGROW^oct for the sphere layers, and the
// denser hole layers on top, plus the starfield's 3800.
const DENSITIES = [34, 34 * 3.1, 34 * 3.1 * 3.1, 190, 190 * 3.4, 3800];

let worst = Infinity, checked = 0, skipped = 0, fails = 0, firstFail = null;
for (let t = 0; t < 400000; t++) {
  // Uniform on the sphere, plus deliberate near-pole cases — that is where s2 is floored and
  // where any bound is most likely to break.
  let z;
  if (t % 8 === 0) z = (1 - 1e-6 * rnd()) * (rnd() < 0.5 ? -1 : 1);   // hug the poles
  else z = 2 * rnd() - 1;
  const ph = TAU * rnd();
  const st = Math.sqrt(Math.max(0, 1 - z * z));
  const dir = [Math.cos(ph) * st, Math.sin(ph) * st, z];
  // `rot * dir` in f32 leaves |lp| a few ulps off unity, and the whole point of using
  // length(lp.xy) rather than sqrt(1 - lp.z^2) is that the bound must survive that. Perturb it
  // far harder than f32 ever would — if the inequality is an identity plus Cauchy-Schwarz, it
  // does not care about the magnitude at all.
  if (t % 3 === 0) {
    const e = 1 + (rnd() - 0.5) * 1e-4;
    dir[0] *= e; dir[1] *= e; dir[2] *= e;
  }

  const n = DENSITIES[t % DENSITIES.length];
  const c = cell(dir, n);
  for (let j = 0; j < 9; j++) {
    const p = polar(c, j);
    if (!p) { skipped++; continue; }
    const lcw = dirLocal(p);
    const real = c.lp[0] * lcw[0] + c.lp[1] * lcw[1] + c.lp[2] * lcw[2];
    const bound = dotMax(c, p);
    const slack = bound - real;          // must be >= 0
    checked++;
    if (slack < worst) worst = slack;
    if (slack < -1e-6) {
      fails++;
      if (!firstFail) firstFail = { dir, n, j, real, bound, slack };
    }
  }
}

console.log(`candidates checked   ${checked}   (out of range, not applicable: ${skipped})`);
console.log(`worst slack          ${worst.toExponential(3)}   must be >= 0`);
console.log(`                     negative slack = a candidate the shader may skip wrongly`);
console.log(`violations           ${fails}`);
if (firstFail) console.log('first violation      ' + JSON.stringify(firstFail));
console.log(fails === 0 ? 'PASS  the polar bound is strict on every case tried'
                        : 'FAIL  the bound is not an upper bound');
process.exit(fails === 0 ? 0 : 1);
