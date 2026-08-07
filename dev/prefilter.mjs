// Does the bloom prefilter see a point highlight the same wherever it lands?
//
// The symptom was star glow popping in and out as the camera drifted. With a 4x reduction, one
// bilinear tap reads 4 of every 16 source texels — so a star's contribution depended entirely on
// which texel it happened to occupy. This slides a single bright texel through all 16 sub-
// positions of one destination footprint and reports the spread of the prefilter's output.
//
// Position-invariant output (spread 0) is the property that makes the glow stop popping.
// Mirrors shaders/bloom.wgsl: prefilterOne + fsPrefilter.

const THRESH = 1.1;                 // GLOW.threshold
const STAR = 24.0;                  // one bright accum texel, roughly a star's peak
const SKY = 0.004;                  // the near-black background it sits on

const prefilterOne = (l) => {
  const knee = THRESH * 0.5;
  const soft = Math.min(1, Math.max(0, (l - THRESH + knee) / Math.max(2 * knee, 1e-4)));
  const contrib = Math.max(l - THRESH, l * soft * 0.25);
  return l * (contrib / Math.max(l, 1e-4));   // == contrib, kept in the shader's shape
};

/** A 4x4 source footprint with the star at (sx, sy). */
const footprint = (sx, sy) => {
  const f = new Float64Array(16).fill(SKY);
  f[sy * 4 + sx] = STAR;
  return f;
};

/** A bilinear tap at source coordinate (x, y) in texel units, on a 4x4 tile. */
const bilinear = (f, x, y) => {
  const x0 = Math.floor(x - 0.5), y0 = Math.floor(y - 0.5);
  const fx = x - 0.5 - x0, fy = y - 0.5 - y0;
  const at = (i, j) => f[Math.min(3, Math.max(0, j)) * 4 + Math.min(3, Math.max(0, i))];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy)
       + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
};

// OLD: one bilinear tap at the footprint centre (2.0, 2.0), threshold after.
const oldWay = (f) => prefilterOne(bilinear(f, 2.0, 2.0));

// NEW: 2x2 bilinear taps at 0.5 / 2.5, each thresholded, then averaged.
const newWay = (f) => {
  let acc = 0;
  for (const y of [1.0, 3.0]) for (const x of [1.0, 3.0]) acc += prefilterOne(bilinear(f, x, y));
  return acc / 4;
};

const stats = (fn) => {
  const v = [];
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) v.push(fn(footprint(x, y)));
  const mean = v.reduce((a, b) => a + b) / v.length;
  const lo = Math.min(...v), hi = Math.max(...v);
  const zeros = v.filter((x) => x < 1e-6).length;
  return { mean, lo, hi, spread: hi - lo, rel: mean > 1e-9 ? (hi - lo) / mean : 0, zeros, v };
};

const o = stats(oldWay), n = stats(newWay);
const f = (x) => x.toFixed(4);
console.log('star peak %s on sky %s, threshold %s', STAR, SKY, THRESH);
console.log('');
console.log('               mean      min       max       spread    positions giving zero');
console.log(`one tap        ${f(o.mean)}    ${f(o.lo)}    ${f(o.hi)}    ${f(o.spread)}    ${o.zeros}/16`);
console.log(`per-tap box    ${f(n.mean)}    ${f(n.lo)}    ${f(n.hi)}    ${f(n.spread)}    ${n.zeros}/16`);
console.log('');
console.log(`one tap varies by ${(o.rel * 100).toFixed(1)}% of its own mean as the star slides`);
console.log(`per-tap box varies by ${(n.rel * 100).toFixed(1)}%`);
// Energy: the true contribution of the footprint, thresholded per source texel.
const truth = footprint(0, 0).reduce((a, _, i, arr) => a + prefilterOne(arr[i]), 0) / 16;
console.log(`exact per-texel answer ${f(truth)}  -> box error ${f(Math.abs(n.mean - truth))},`
          + ` one-tap error ${f(Math.abs(o.mean - truth))}`);

const ok = n.spread < 1e-9 && n.zeros === 0;
console.log('');
console.log(ok ? 'PASS  the box is position-invariant and never drops the highlight'
               : 'FAIL  still position-dependent');
process.exit(ok ? 0 : 1);
