// Do the scenes honour the contract the renderer calls them through?
//
// The renderer calls six methods on whatever scene is active and reads one getter. A scene missing any
// of them fails at the first frame after a switch — in the middle of a render loop, where the stack
// trace points at the renderer rather than at the scene that is actually incomplete. Checking the shape
// here means a new scene is caught before it is ever selected.
//
// No GPU: these are structural checks and the constants behind them. Anything needing a device belongs
// in the browser instruments.

import { Scene } from '../src/scenes/scene.js';
import { PlanetoidScene } from '../src/scenes/planetoid.js';
import { ModelViewScene, MODELS } from '../src/scenes/modelview.js';
import { MODELVIEW, SHIP } from '../src/scene/tuning.js';

let failed = 0;
const check = (ok, what, extra = '') => {
  process.stdout.write(`  ${ok ? 'PASS' : 'FAIL'}  ${what}${extra ? `   ${extra}` : ''}\n`);
  if (!ok) failed++;
};

const SCENES = [PlanetoidScene, ModelViewScene];
const CONTRACT = ['init', 'update', 'writeState', 'recordWorld', 'recordAdditive', 'destroy'];

// ---- the contract ----
{
  for (const S of SCENES) {
    check(Object.getPrototypeOf(S.prototype) === Scene.prototype
          || S.prototype instanceof Scene,
          `${S.name} extends Scene`);
    check(typeof S.label === 'string' && S.label.length > 0,
          `${S.name} has a label for the dropdown`, S.label);
    const missing = CONTRACT.filter((m) => typeof S.prototype[m] !== 'function');
    check(missing.length === 0, `${S.name} implements every method the renderer calls`,
          missing.length ? `missing ${missing.join(', ')}` : CONTRACT.length + ' methods');
    check(Object.getOwnPropertyDescriptor(S.prototype, 'solidPasses')?.get
          || typeof Scene.prototype.solidPasses !== 'undefined',
          `${S.name} exposes solidPasses for LOD and the wireframe view`);
  }
  const labels = SCENES.map((S) => S.label);
  check(new Set(labels).size === labels.length, 'scene labels are distinct', labels.join(', '));
}

// ---- the model list ----
{
  check(MODELS.length >= 2, 'the viewer offers more than one model', `${MODELS.length}`);
  const keys = MODELS.map((m) => m.key);
  check(new Set(keys).size === keys.length, 'model keys are distinct', keys.join(', '));
  check(MODELS.every((m) => m.key && m.label), 'every model has a key and a label');
}

// ---- the studio distance is a correctness constraint, not a preference ----
//
// The hull's material adds a core-light term that falls off as 1/|p|^2 from the world origin. A model
// sitting near the origin would be lit by a planet the viewer deliberately does not draw, and a vertex
// landing exactly on it would normalise a zero vector. This is the number that keeps that from
// happening, so it is asserted rather than left as a comment nobody re-derives.
{
  const d2 = MODELVIEW.origin.reduce((s, v) => s + v * v, 0);
  const d = Math.sqrt(d2);
  check(d > 50, 'the studio is far enough from the origin that the core light is negligible',
        `|origin| ${d.toFixed(0)}, falloff 1/${Math.round(d2)}`);
  check(1 / d2 < 1e-3, 'and the term it contributes is under a thousandth',
        `${(1 / d2).toExponential(1)}`);
  // Far outside the planet and its atmosphere, so nothing of the real scene intrudes even before the
  // studio flag drops the body.
  check(d > 20, 'and well outside the body and its atmosphere');
}

// ---- framing ----
{
  check(MODELVIEW.distance > 1.5,
        'the camera stands off far enough to frame a model rather than sit inside it',
        `${MODELVIEW.distance}x the bounding radius`);
  check(MODELVIEW.elevation > 0.2 && MODELVIEW.elevation < Math.PI / 2 - 0.15,
        'the elevation is a quarter view: above the equator, not straight down',
        `${(MODELVIEW.elevation * 180 / Math.PI).toFixed(0)} degrees`);
  check(MODELVIEW.spinRate > 0 && MODELVIEW.spinRate < 3,
        'the turntable turns, and slowly enough to read',
        `${(2 * Math.PI / MODELVIEW.spinRate).toFixed(1)} s per revolution`);
}

// ---- the planetoid still owns what it always did ----
//
// The extraction was meant to be mechanical. If a constant the scene depends on stopped resolving, the
// scene would fail at construction, which needs a GPU — so this is the cheap version of that check.
{
  check(typeof SHIP.autoFireEvery === 'number' && SHIP.autoFireEvery > 0,
        'the planetoid can still find its auto-fire period', `${SHIP.autoFireEvery}s`);
}

process.exit(failed === 0 ? 0 : 1);
