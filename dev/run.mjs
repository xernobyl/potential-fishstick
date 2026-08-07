#!/usr/bin/env node
/**
 * Every headless check, in one command:  node dev/run.mjs
 *
 * These exist because the interesting properties of this renderer are not things you can see.
 * A lattice bound that is a few ulps too tight drops a sphere and seams the surface; a ribbon
 * parameterised by buffer index shimmers four times a second; a bloom prefilter that undersamples
 * loses a star's glow entirely. Each of those was found by measuring, and each stays fixed only
 * as long as something keeps measuring it.
 *
 * No GPU and no dependencies: each check either reimplements the shader arithmetic it is
 * validating, or drives the real CPU-side simulation against a stub device.
 */
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const checks = [
  ['polarbound', 'the march\'s polar rejection bound is a strict upper bound'],
  ['prefilter', 'the bloom prefilter sees a point highlight the same wherever it lands'],
  ['walker', 'aurora walkers stay in their shell and never turn sharply'],
  ['continuity', 'the aurora ribbon\'s parameterisation does not step on emission'],
  ['meshgen', 'the generated ring mesh is the same surface as the arithmetic it replaces'],
];

let failed = 0;
for (const [name, what] of checks) {
  process.stdout.write(`\n=== ${name} — ${what}\n`);
  const r = spawnSync(process.execPath, [join(here, `${name}.mjs`)], { stdio: 'inherit' });
  if (r.status !== 0) failed++;
}
process.stdout.write(`\n${failed === 0 ? 'ALL PASS' : `${failed} FAILED`}\n`);
process.exit(failed === 0 ? 0 : 1);
