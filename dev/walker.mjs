// Headless walker test: run the aurora simulation with no GPU and measure the two things the
// ribbon renderer actually cares about — how far the walkers stray from the shell, and how
// sharply the emitted path turns (which is what the miter join has to survive).
import { Aurora } from '../src/scene/aurora.js';
import { AURORA } from '../src/scene/tuning.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };

const device = {
  createBuffer: ({ size }) => ({ size, destroy() {} }),
  queue: { writeBuffer() {} },
};

const a = new Aurora(device);
const dt = 1 / 60;
const SECONDS = Number(process.argv[2] ?? 120);

let rMin = Infinity, rMax = -Infinity;
for (let f = 0; f < SECONDS * 60; f++) {
  a.update(dt, f * dt);
  for (let r = 0; r < a.ribbons; r++) {
    const o = r * 3;
    const d = Math.hypot(a.pos[o], a.pos[o + 1], a.pos[o + 2]);
    rMin = Math.min(rMin, d); rMax = Math.max(rMax, d);
  }
}

// Turn angles along the EMITTED samples — the geometry the shader expands, not the sim steps.
const angles = [];
const seg = [];
for (let r = 0; r < a.ribbons; r++) {
  const base = r * a.count * 4;
  const dirs = [];
  for (let i = 1; i < a.count; i++) {
    const dx = a.cpu[base + i * 4] - a.cpu[base + (i - 1) * 4];
    const dy = a.cpu[base + i * 4 + 1] - a.cpu[base + (i - 1) * 4 + 1];
    const dz = a.cpu[base + i * 4 + 2] - a.cpu[base + (i - 1) * 4 + 2];
    const l = Math.hypot(dx, dy, dz);
    if (l < 1e-9) continue;
    seg.push(l);
    dirs.push([dx / l, dy / l, dz / l]);
  }
  for (let i = 1; i < dirs.length; i++) {
    const d = Math.max(-1, Math.min(1, dirs[i][0] * dirs[i - 1][0] + dirs[i][1] * dirs[i - 1][1]
                                     + dirs[i][2] * dirs[i - 1][2]));
    angles.push(Math.acos(d));
  }
}
angles.sort((x, y) => x - y);
const deg = (x) => (x * 180 / Math.PI).toFixed(2);
const q = (p) => angles[Math.min(angles.length - 1, Math.floor(p * angles.length))];
const worst = angles[angles.length - 1];

// Ribbon extent end to end, and total path length.
let ext = 0, plen = 0;
for (let r = 0; r < a.ribbons; r++) {
  const b = r * a.count * 4, e = b + (a.count - 1) * 4;
  ext += Math.hypot(a.cpu[e] - a.cpu[b], a.cpu[e + 1] - a.cpu[b + 1], a.cpu[e + 2] - a.cpu[b + 2]);
}
for (const l of seg) plen += l;

// The miter widens the joint by 1/cos(theta/2); the limit is cos(theta/2) < miterMin.
const widen = (t) => 1 / Math.cos(t / 2);
const limitAngle = 2 * Math.acos(AURORA.miterMin);

console.log(`radius       ${rMin.toFixed(3)} .. ${rMax.toFixed(3)}   shell ${AURORA.shellMin}..${AURORA.shellMax}`
          + `   hard bound ${(AURORA.shellMin - AURORA.shellSlack).toFixed(2)}..${(AURORA.shellMax + AURORA.shellSlack).toFixed(2)}`);
console.log(`turn angle   median ${deg(q(0.5))}   p99 ${deg(q(0.99))}   worst ${deg(worst)}  (n=${angles.length})`);
console.log(`miter widen  worst ${widen(worst).toFixed(3)}x   limit fires at ${deg(limitAngle)} (${(1 / AURORA.miterMin).toFixed(2)}x)`);
console.log(`             predicted bound ${deg(AURORA.maxTurn * AURORA.interval)} = maxTurn * interval`);
console.log(`ribbon       ${(plen / a.ribbons).toFixed(2)} u path   ${(ext / a.ribbons).toFixed(2)} u end-to-end`
          + `   half-width up to ${(AURORA.width + AURORA.widthGrow).toFixed(3)}`);
console.log(`min radius   ${(AURORA.speed / AURORA.maxTurn).toFixed(3)} u = speed / maxTurn`
          + `   (${((AURORA.speed / AURORA.maxTurn) / (AURORA.width + AURORA.widthGrow)).toFixed(1)}x the half-width)`);
