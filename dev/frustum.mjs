// Do the extracted frustum planes actually bound what the camera can see?
//
// Culling is the one optimisation whose failure mode is invisible in the direction that matters: too
// LOOSE and you lose nothing but the saving, too TIGHT and geometry vanishes at the screen edge, usually
// on someone else's machine at a different aspect ratio. So the planes are checked against points whose
// visibility is known by construction.
//
// The awkward part of this projection is that there are FIVE planes, not six: it is reverse-Z and
// infinite, so the far plane's inequality is satisfied by everything in front of the camera and
// extracting it yields a degenerate zero-normal plane. That is exactly the kind of thing that works on
// the machine it was written on and rejects the whole scene on a driver that rounds the other way.

import * as m4 from '../src/scene/mat4.js';
import { extractFrustum, sphereVisible } from '../src/core/frustum.js';

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

// A camera at +Z looking back at the origin, matching the project's basis convention: fwd is the view
// direction, right and up complete a right-handed set.
const cam = { pos: [0, 0, 10], fwd: [0, 0, -1], right: [1, 0, 0], up: [0, 1, 0] };
const focal = 1.2;
const aspect = 16 / 9;
const view = m4.view(m4.create(), cam);
const proj = m4.projection(m4.create(), focal, [0, 0], aspect / Math.hypot(aspect, 1),
                           1 / Math.hypot(aspect, 1));
const vp = m4.mul(m4.create(), proj, view);
const planes = extractFrustum(vp, new Float32Array(20));

// ---- the planes themselves ----
{
  let worst = 0;
  for (let i = 0; i < 5; i++) {
    const len = Math.hypot(planes[i * 4], planes[i * 4 + 1], planes[i * 4 + 2]);
    worst = Math.max(worst, Math.abs(len - 1));
  }
  check(worst < 1e-6, 'every plane normal is unit length, so d is a world-space distance',
        `worst |len-1| ${worst.toExponential(2)}`);
  check(planes.length === 20, 'five planes, not six — the far plane does not exist in this projection',
        `${planes.length / 4} planes`);
}

// ---- points whose visibility is known ----
{
  check(sphereVisible(planes, 0, 0, 0, 0.1), 'a point at the origin, dead ahead, is visible');
  check(!sphereVisible(planes, 0, 0, 20, 0.1), 'a point BEHIND the camera is not',
        'z = 20 with the camera at z = 10 looking to -z');
  check(sphereVisible(planes, 0, 0, -1000, 1), 'and a point very far ahead still is — no far plane');
  check(!sphereVisible(planes, 100, 0, 0, 0.1), 'a point far off to the side is not');
  check(!sphereVisible(planes, 0, 100, 0, 0.1), 'nor one far above');

  // A big sphere straddling the near plane must be KEPT: conservative in the safe direction.
  check(sphereVisible(planes, 0, 0, 11, 3), 'a sphere straddling the near plane is kept, not dropped');
  // The same sphere with a tiny radius is genuinely behind and must go.
  check(!sphereVisible(planes, 0, 0, 11, 0.05), 'while a small one at the same place is dropped');
}

// ---- the aspect ratio is respected, which is what a hand-written frustum usually gets wrong ----
{
  // At 10 units ahead, the horizontal half-extent is wider than the vertical for a 16:9 window. A point
  // just inside the horizontal edge but outside the vertical one distinguishes the two.
  const z = 0;                    // 10 units in front of the camera
  let hx = 0;
  while (sphereVisible(planes, hx, 0, z, 0) && hx < 100) hx += 0.05;
  let hy = 0;
  while (sphereVisible(planes, 0, hy, z, 0) && hy < 100) hy += 0.05;
  check(hx > hy, 'the frustum is wider than it is tall on a 16:9 window',
        `half-extents x ${hx.toFixed(2)}, y ${hy.toFixed(2)}`);
  check(Math.abs(hx / hy - aspect) < 0.1, 'and the ratio matches the aspect',
        `${(hx / hy).toFixed(3)} vs ${aspect.toFixed(3)}`);
}

// ---- rotating the camera must move the frustum with it ----
{
  const side = { pos: [0, 0, 10], fwd: [1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] };
  const vp2 = m4.mul(m4.create(), proj, m4.view(m4.create(), side));
  const p2 = extractFrustum(vp2, new Float32Array(20));
  check(!sphereVisible(p2, 0, 0, 0, 0.1), 'looking away, the origin leaves the frustum');
  check(sphereVisible(p2, 50, 0, 10, 0.1), 'and what is now ahead enters it');
}

process.exit(failed === 0 ? 0 : 1);
