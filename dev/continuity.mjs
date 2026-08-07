// Does the ribbon's PARAMETERISATION move continuously?
//
// The geometry never jumped — a ring-buffer shift moves indices, not world points. What jumped
// was everything derived from the index: age, and therefore width, fade and the ray phase. This
// follows a fixed world point across emissions and measures the frame-to-frame step in the
// `ageSamples` the vertex shader computes for it. Continuous means every step is dt/interval.
import { Aurora } from '../src/scene/aurora.js';
import { AURORA } from '../src/scene/tuning.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
const device = { createBuffer: ({ size }) => ({ size, destroy() {} }), queue: { writeBuffer() {} } };

const a = new Aurora(device);
const dt = 1 / 60;
const n = a.count;
const last = n - 1;
const expected = dt / AURORA.interval;      // the age step a continuous scheme must produce

// Warm up past the initial seeding.
for (let f = 0; f < 300; f++) a.update(dt, f * dt);

const key = (r, i) => {
  const b = r * n * 4 + i * 4;
  return `${a.cpu[b].toFixed(6)},${a.cpu[b + 1].toFixed(6)},${a.cpu[b + 2].toFixed(6)}`;
};
const ageOf = (i) => (last - i) + a.emitPhase;   // exactly what the vertex shader computes

let prev = new Map();
let worstAge = 0, worstHead = 0, headSteps = [], commits = 0, tracked = 0;
for (let i = 0; i < n; i++) prev.set(key(0, i), ageOf(i));
let prevHead = [a.cpu[(n - 1) * 4], a.cpu[(n - 1) * 4 + 1], a.cpu[(n - 1) * 4 + 2]];

for (let f = 300; f < 300 + 60 * 20; f++) {
  const before = a.emitPhase;
  a.update(dt, f * dt);
  if (a.emitPhase < before) commits++;

  const now = new Map();
  for (let i = 0; i < n; i++) {
    const k = key(0, i);
    const age = ageOf(i);
    now.set(k, age);
    const was = prev.get(k);
    // Skip the head: it is a moving point, so it has no previous incarnation to compare to.
    if (was !== undefined && i !== last) {
      worstAge = Math.max(worstAge, Math.abs(age - was - expected));
      tracked++;
    }
  }
  prev = now;

  const h = [a.cpu[(n - 1) * 4], a.cpu[(n - 1) * 4 + 1], a.cpu[(n - 1) * 4 + 2]];
  const step = Math.hypot(h[0] - prevHead[0], h[1] - prevHead[1], h[2] - prevHead[2]);
  headSteps.push(step);
  worstHead = Math.max(worstHead, step);
  prevHead = h;
}

headSteps.sort((x, y) => x - y);
const med = headSteps[headSteps.length >> 1];
const spacing = AURORA.speed * AURORA.interval;
const smooth = AURORA.speed * dt;

console.log(`emissions      ${commits} over 20 s   samples compared ${tracked}`);
console.log(`age step       expected ${expected.toFixed(5)} per frame`);
console.log(`               worst deviation ${worstAge.toExponential(2)} samples`
          + `   (a shift would show ${(1).toFixed(2)})`);
console.log(`head motion    median ${med.toFixed(5)} u/frame   worst ${worstHead.toFixed(5)}`);
console.log(`               smooth = ${smooth.toFixed(5)} (speed*dt);`
          + ` a per-emission head would show 0 then ${spacing.toFixed(4)}`);
const ok = worstAge < 1e-3 && worstHead < smooth * 1.5;
console.log(ok ? 'PASS  parameterisation and head are both continuous'
                : 'FAIL  something still steps');
process.exit(ok ? 0 : 1);
