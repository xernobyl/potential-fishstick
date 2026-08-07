// Does LOD selection pick the level a viewer could not tell from the finest one?
//
// The failure modes here are quiet in both directions. Too eager and the ship visibly loses its nose at
// mid distance; too shy and the whole mechanism costs build time and memory for nothing. Neither shows
// up as an error, so both are pinned to numbers.

import { selectLod, worldPerPixel } from '../src/core/lod.js';
import { resolutionForScreen } from '../src/scene/sdf/dualcontour.js';
import { SHIP_MESH } from '../src/scene/ship_sdf.js';

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

// The ship's real chain, in world units, exactly as the renderer builds it.
const levels = SHIP_MESH.lodErrors.map((e) => ({ errorWorld: e * SHIP_MESH.scale }));
const FOCAL = 1.2;
const DIAG = 2500;
const BUDGET = SHIP_MESH.lodErrorPx;

// ---- the basics ----
{
  check(levels.length >= 2, 'the ship has an LOD chain at all', `${levels.length} levels`);
  let ascending = true;
  for (let i = 1; i < levels.length; i++) {
    if (levels[i].errorWorld <= levels[i - 1].errorWorld) ascending = false;
  }
  check(ascending, 'levels are ordered finest first',
        levels.map((l) => l.errorWorld.toFixed(4)).join(' < '));
}

// ---- selection is monotone in distance ----
{
  let prev = -1;
  let monotone = true;
  const picks = [];
  for (const dist of [0.5, 1, 2, 4, 8, 16, 32, 64]) {
    const l = selectLod(levels, dist, FOCAL, DIAG, BUDGET);
    picks.push(`${dist}m:L${l}`);
    if (l < prev) monotone = false;
    prev = l;
  }
  check(monotone, 'a further object never selects a FINER level', picks.join(' '));
}

// ---- the near and far ends behave ----
{
  check(selectLod(levels, 0.2, FOCAL, DIAG, BUDGET) === 0,
        'up close it picks the finest level');
  check(selectLod(levels, 5000, FOCAL, DIAG, BUDGET) === levels.length - 1,
        'far away it picks the coarsest');
}

// ---- the chosen level's error really is sub-pixel ----
//
// This is the claim the whole scheme rests on: whatever it picks, the geometric error it accepted is
// under the pixel budget at that distance. If this holds, switching levels cannot be visible.
{
  let worstPx = 0;
  let violations = 0;
  for (let dist = 0.5; dist < 200; dist *= 1.15) {
    const l = selectLod(levels, dist, FOCAL, DIAG, BUDGET);
    const px = levels[l].errorWorld / worldPerPixel(dist, FOCAL, DIAG);
    worstPx = Math.max(worstPx, px);
    // Level 0 is the floor — there is nothing finer to escalate to, so it is allowed to exceed.
    if (l > 0 && px > BUDGET + 1e-9) violations++;
  }
  check(violations === 0,
        'every selected level above the floor keeps its error under the pixel budget',
        `worst ${worstPx.toFixed(2)} px, budget ${BUDGET} px`);
}

// ---- a finer budget must never select a coarser level ----
{
  let ok = true;
  for (let dist = 0.5; dist < 200; dist *= 1.3) {
    const strict = selectLod(levels, dist, FOCAL, DIAG, 0.4);
    const loose = selectLod(levels, dist, FOCAL, DIAG, 2.0);
    if (strict > loose) ok = false;
  }
  check(ok, 'a stricter pixel budget never picks a coarser level than a loose one');
}

// ---- resolution and selection agree about what a pixel is ----
//
// `resolutionForScreen` derives the FINEST mesh from a pixel budget and `selectLod` chooses among the
// levels using the same conversion. If they disagreed, the finest level could be finer than anything the
// selector would ever ask for — paying for detail that is unreachable by construction.
{
  const size = 1.0;
  const dist = SHIP_MESH.viewDistance;
  const res = resolutionForScreen({ size, distance: dist, focal: FOCAL, diagonalPx: DIAG, errorPx: 4 });
  const cellWorld = size / res;
  const cellPx = cellWorld / worldPerPixel(dist, FOCAL, DIAG);
  check(Math.abs(cellPx - 4) < 0.05,
        'a cell sized for N pixels measures N pixels through the selector\'s own conversion',
        `${cellPx.toFixed(3)} px for a 4 px request`);
}

// ---- the chain spans a useful range ----
{
  const near = worldPerPixel(1.5, FOCAL, DIAG) * BUDGET;
  const far = worldPerPixel(40, FOCAL, DIAG) * BUDGET;
  check(levels[0].errorWorld <= near,
        'the finest level is fine enough to be chosen at close range',
        `${levels[0].errorWorld.toFixed(5)} <= ${near.toFixed(5)}`);
  check(levels[levels.length - 1].errorWorld <= far,
        'and the coarsest is still within budget at long range',
        `${levels[levels.length - 1].errorWorld.toFixed(5)} <= ${far.toFixed(5)}`);
}

process.exit(failed === 0 ? 0 : 1);
