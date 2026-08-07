# beep beep beep

**[Live →](https://xernobyl.github.io/potential-fishstick/)**  · WebGPU · needs Chrome, Edge, Safari 18+ or a recent Firefox

A translucent planetoid, raymarched. The body is a signed-distance field built from octaves of
spheres on a spherical-Fibonacci lattice, lit as a scattering medium rather than a surface, wrapped
in a volumetric atmosphere that the rings cast real shadows through. Around it: three precessing
metal hoops, a flyable ship trailing contrails and firing helical rail guns, orbiting satellites,
GPU particle embers, nine curl-noise auroras, marched reflections, temporal upsampling from half
resolution, and an analytic film response on the way out.

Everything is procedural. No textures, no meshes, no assets — every shape is a distance field or a
handful of instanced boxes. The one dependency is a vendored copy of `lil-gui` for the tuning panel,
loaded on demand and outside the render path.

Press **`g`** for the controls and the tuning panel. **Drag** to orbit, **arrows/WASD** to fly,
**space** to shoot.

**[ARCHITECTURE.md](ARCHITECTURE.md)** is the shorter read: the layers, the frame graph, and the
temporal contract that most of this renderer's bugs have lived in. Start there if you are going to
change something rather than just look at it.

---

### About the rest of this file

It is long, and it is a log rather than a manual. The interesting parts of this renderer are not the
features but the things that turned out to be wrong about them: a temporal filter that tracked its
own jitter, a lattice bound a few ulps too tight, a bloom prefilter that read four of every sixteen
texels, a vignette measured from the wrong point, a benchmark that read a patch from the wrong
quarter of the frame and invalidated its own conclusion. Each of those was found by measuring, and
each entry below records the measurement, including the ones that came out negative and got
reverted.

If you only want to run it, the next section is all you need. If you want to change something, start
at [Tuning it](#tuning-it) and `src/scene/tuning.js`.

## Running it

**It must be served over HTTP.** Opening `index.html` from the file system will not
work: the shader loader uses `fetch()` and the code is ES modules, and the `file://`
origin blocks both. You will get a blank page.

From this directory:

```bash
python3 serve.py
```

Then open <http://localhost:8080>.

`serve.py` is `http.server` with `Cache-Control: no-store` on every response, and that one
header is the whole reason it exists. Plain `python3 -m http.server` sends no cache header at
all, so the browser applies heuristic freshness and serves a stale `.wgsl` or `.js` after an
edit — which surfaces as the shader compiler reporting an error against source that is no
longer on disk. Any static server will do as long as it does that; `npx serve .` will not.

**Needs WebGPU**, which means a recent Chrome or Edge, Safari 18+, or a recent
Firefox. If it is missing or the adapter cannot be acquired, the page says so
explicitly rather than failing silently.

### Hosting it

It is a static site with no build step, so any static host works — GitHub Pages from the repository
root needs no configuration. Two things make that true, and both are deliberate:

- **Nothing is located by an absolute path.** The entry script is `./src/main.js`, the shader root
  is `new URL('../../shaders/', import.meta.url)`, and lil-gui is imported relative to its
  importer. So the whole thing runs unchanged from a subdirectory like `/<repo>/`, with no base
  path to configure and nothing to rewrite at deploy time.
- **The shader loader only revalidates in local development.** `no-cache` is there because a stale
  `.wgsl` fails as a compile error against source that no longer exists on disk. A deployed shader
  cannot change underneath you, so in production that would buy nothing and cost two dozen
  conditional round trips before the first frame — see `DEV` in `src/core/wgsl.js`.

WebGPU requires a secure context, which Pages satisfies. `.nojekyll` is present so the tree is
served exactly as committed. `serve.py` and `dev/` get published too and are harmless; they are
just never fetched.

The one thing to know is that this is a genuine renderer, not a lightweight demo: the march is the
whole frame budget, and `QUALITY.renderScale` at 0.5 with `maxWidth` at 2560 is tuned for an Apple
silicon laptop. On weaker hardware it will be slow, and the honest fix is the render-scale slider
behind `g` rather than anything automatic — see the dynamic-resolution idea below.

## Controls

Press **`g`** to put this list on screen; `g` again for the tuning panel, again for neither.

- **drag** — orbit the camera. Releasing keeps your angle; the slow dolly and roll
  continue underneath.
- **arrows / A,D** — turn the ship. **up / W** thrusts, **down / S** reverses.
- **space** — fire the rail guns. One shot per press, alternating wings; holding does
  nothing, so hammer it.
- **C** — toggle between the chase camera and the free orbit camera.
- **G** — cycle: the control list → the tuning panel → nothing. The panel is loaded on first use,
  so it costs nothing until you want it.
- **T** — freeze: holds the clock, the input *and* `dt`, so not one thing moves. Discards the
  accumulated image, so what you see rebuilds from scratch under the new regime.
- **P** — pause: the same hold, but it **keeps** the accumulated image. Use this to look at the
  picture as it already was; use `T` when you want to watch how it converges.
- **U** — temporal upsampling on/off. Off renders 1:1 instead of reconstructing from half
  resolution. Worth knowing: this is *not* where jitter comes from, and turning it off makes a
  frozen frame slightly less stable, because native resolution is sharper and carries four times
  the independent samples.
- **J** — the pixel jitter on/off. Off means no antialiasing; with it off and the scene frozen the
  frame is bit-exact, which is the baseline every stability figure here is measured against.
- **F** — fullscreen.
- **R** — record 15 s of 1080p at a constant 30 fps to an MP4, then download it. The render is
  OFFLINE, so it takes about 50 s of wall time and the result is smooth regardless — see below.

Click the canvas first so it has keyboard focus.

## Tuning it

Press `g`. Every control binds straight to the live tuning object, which is only possible
because the values behind them are read fresh each frame — the grade and the glow are frame
UNIFORMS rather than injected WGSL constants, specifically so a slider can reach them. That
distinction is the whole design of the panel: a control bound to something baked at pipeline
creation is a slider that does nothing, which is worse than no slider, so anything in that
category was either moved onto the uniform first or is not exposed.

Two of them are not simply values and say so in code: `GLOW.radius` and `levelWeight` live in
per-level parameter buffers written on resize, so they trigger an explicit rebuild; `taau`,
`renderScale` and `additiveDisplayRes` reallocate targets and go through `renderer.resize()`.

Three folders are not sliders. **Monitor** shows which grid each stage is actually running at,
which is the state you otherwise have to reconstruct from three tuning values. **Measure** runs
the instruments below on demand and leaves each headline in one field — the full report still goes
to the console, because a single number is a summary and the reports exist to stop it being read
alone. **Presets** saves and loads named looks to `localStorage`, and captures the ART STATE only:
snapshotting every folder meant a preset stored the panel's own UI, and since the preset picker
was itself a saved control, loading one re-entered its own handler and never completed.

The one dependency this project has is `lil-gui` (MIT), vendored as a single file under
`vendor/` rather than fetched from a CDN — there is no build step and no package manager here,
and it should keep working with the network off.

For the SKY specifically, the control that matters is **black lift**, not exposure: the
background is near zero almost everywhere, so it lands essentially on the grade's black floor
and nothing else moves it much. **Contrast** is second, and is applied about a mid pivot after
the display transfer — in linear light it would just be exposure and white point moving
together, crushing the shadows while barely touching the highlights.

## Measuring it

The on-screen HUD is a convenience and its numbers are not trustworthy — browsers
throttle `requestAnimationFrame` for a hidden canvas, and the readings are smoothed
and contended. For real numbers:

```
http://localhost:8080/?bench
http://localhost:8080/?bench&frames=200
http://localhost:8080/?bench=stability
```

This drives the renderer from a plain loop on a synthetic clock — no rAF, identical
camera path every run — and reports per-pass medians to the console. It also
self-checks: GPU passes run serially, so if the per-pass times sum to more than a
frame, or if two passes report one shared interval, the breakdown says so instead of
publishing an impossible number.

There are nine instruments, and eight of them exist because timing alone cannot answer the
question being asked of it:

- **`beep.bench({stability:true})`** — is the image STEADY? Frozen clock, diff consecutive
  accumulation buffers.
- **`beep.lag()`** — is the image RIGHT? Accumulate over motion to instant T, then freeze at
  T and converge from a reset history; the difference is temporal lag. Reports the noise
  floor beside it, because the lag figure means nothing alone. Also reports **sharpness**
  (relative gradient energy), which is how a history reconstruction filter is judged.
  Validated by a control: with the history disabled there is no lag by construction, and it
  correctly reports 0.63x noise.
- **`beep.compare(mutate, restore)`** — do two configs render the SAME IMAGE? Converge,
  switch, converge, diff. This is what an accelerator needs: a bound that skips the surface
  entirely is both wrong and extremely fast, so timing cannot validate it.
- **`beep.sharp(configs)`** — which config retains more high-frequency ENERGY? Measures several
  configurations on one display-resolution grid, resampling each from its own accumulation
  buffer size. Necessary because sharpness is per-pixel: comparing a 720p buffer against a
  1440p one directly makes any upsampler look 2x softer than it is. Read it with `beep.detail()`,
  never alone — energy is not the same as detail, and this instrument cannot tell them apart.
- **`beep.detail()`** — is that detail REAL? Sharpness cannot tell recovered detail from added
  grain, because both live in the same frequency band. This renders a native-resolution ground
  truth and projects each candidate's high-frequency band onto the reference's, splitting the
  magnitude a sharpness metric reports into `signal` (true detail) and `noise` (everything else).
  Slow — it converges four configurations, one at full resolution. See below for what it found.
- **`beep.subpixel()`** — does the image FLICKER when the camera moves less than a pixel? Slides
  the camera across one pixel and asks whether translation-invariant quantities stay invariant.
  The only instrument that sees crawl, because crawl only happens when a feature slides across the
  sampling grid. Two traps it avoids: it compares invariants rather than shifting-and-diffing,
  since resampling by a fractional offset blurs by an amount that depends on the fraction and is
  then indistinguishable from the aliasing; and it makes the TAA jitter REPEATABLE rather than
  disabling it, because the jitter is the antialiasing and switching it off measures the renderer
  with its AA removed. The first version disabled it and got the opposite answer.
- **`beep.additive()`** — does drawing the un-filtered additive layer small change what you see?
  Converges it at display and render resolution and compares their high-frequency bands, with the
  frame cost of each beside it.
- **`beep.evals()`** — how much WORK does the march do? Counts field evaluations in the
  shader. It exists because three cost models in a row were wrong; see `MARCH` in `tuning.js`.
- **`beep.bench()`** — how LONG does it take?

Steadiness and correctness are different axes. A badly ghosting accumulator is beautifully
steady, so `stability` alone will happily bless it — which is exactly why the depth gate
could not be safely loosened until `lag` existed.

Which of the timing modes to trust depends on what you are asking:

- **`?bench=stability`** freezes the clock and diffs the accumulation buffer between
  frames, so it measures what the shader COMPUTED. It is unaffected by machine load and
  repeats to well under a percent — the number to A/B a sampling or TAA change against.
- **`?bench`** measures TIME, so it is only as good as the machine is quiet. Compare
  shares of the frame rather than milliseconds, and do not compare a run taken while the
  machine is busy against one taken idle.

`window.beep` is a live handle: `beep.bench()`, `beep.tuning`, `beep.renderer`.
Most tuning values are read per frame, so you can change them from the console and
see it immediately — including the TAA knobs, which live in the uniform block for
exactly this reason. As injected WGSL constants they were baked at pipeline creation, so
console tweaks silently did nothing and an A/B that "showed no difference" was measuring
the same shader twice.

`beep.bench({serial: true})` drains the queue every frame to isolate one frame from
queue depth. It stalls, so its frame time is not a throughput figure.

`beep.shot()` saves a PNG of the current frame, and it exists because you cannot screenshot a
WebGPU canvas any other way: `drawImage` returns transparent black, since the swapchain texture is
released the moment it is presented. The frame has to be copied out during the frame that drew it,
which needs the renderer's cooperation. Pass a scale to downsize — `beep.shot(0.5)`.

### Recording it

Press `r`. Fifteen seconds, 1920x1080, constant 30 fps, downloaded as an MP4.

It is an OFFLINE render, which is the whole point. Frames are driven on a synthetic clock at exactly
1/fps and each is stamped with the time it REPRESENTS, so a frame that takes 90 ms to march still
occupies 33.3 ms of video. Constant frame rate by construction, however slow the renderer is — about
50 s of wall time for 15 s of footage.

That is cheap to build only because `renderer.frame(time, input)` already takes its clock as an
argument, the same property every instrument here relies on. The recorder is that loop with an
encoder attached, and it borrows the instruments' neutral input rather than defining a second one.

Two things it deliberately does not do. It does not use `MediaRecorder` with `captureStream()`,
which is the obvious route and the wrong one: that is a realtime pipeline that timestamps by wall
clock, so a 25 fps render gives dropped and duplicated frames and a variable rate, and
`requestFrame()` does not fix the timestamps. WebCodecs hands you the timestamp, which is the
control this needs. And it does not build the `VideoFrame` from the canvas, because a WebGPU canvas
cannot be read as an image source — the swapchain texture is released at present time, so
`drawImage` and anything like it return transparent black. The frame is copied out of the swapchain
during the frame that drew it and handed over as a raw buffer, which also lets the surface's native
BGRA pass through with no channel swizzle.

`mp4-muxer` (MIT) is vendored beside lil-gui: WebCodecs emits raw encoded chunks, not a container.

### The headless checks

```bash
node dev/run.mjs
```

No GPU, no dependencies, under a second. Each check either reimplements the shader arithmetic it
is validating or drives the real CPU-side simulation against a stub device. They exist because the
properties that matter most here are ones you cannot see:

- **`polarbound`** — the march's polar rejection bound really is an upper bound, over 3.1M
  candidates at the densities the field uses, including near-pole cases and deliberately non-unit
  queries. A few ulps the wrong way drops a sphere and seams the surface, and no image metric
  would attribute that to the bound.
- **`prefilter`** — the bloom prefilter sees a point highlight the same wherever it lands in a
  destination footprint. Slides one bright texel through all sixteen sub-positions.
- **`walker`** — aurora walkers stay inside their shell without the hard clamp firing, and the
  emitted path's joint angle stays far under the miter limit.
- **`continuity`** — the ribbon's age and head both advance continuously across an emission,
  rather than stepping a whole sample as the ring buffer shifts.

### Does temporal upsampling actually resolve detail?

Yes, and `beep.detail()` is the measurement that says so — the earlier answer ("39% less sharp on
the real scene") was an instrument bug, not a result.

The difficulty is that sharpness is the MAGNITUDE of an image's high-frequency band and noise lives
in that same band, so a sharper number can mean recovered detail or added grain and nothing about
the number distinguishes them. The fix is a ground truth and a projection instead of a magnitude:
render the same instant at native display resolution and let the temporal filter converge — with
the clock frozen, the jitter sequence supersamples the frame — then decompose any candidate's band
against the reference's.

    signal = <H(X), H(R)> / <H(R), H(R)>              how much of the TRUE detail is present
    noise  = ||H(X) - signal * H(R)|| / ||H(R)||      band energy that is not detail

These are orthogonal, so `magnitude^2 = signal^2 + noise^2` — the same total a sharpness metric
reports, split into the two halves it was confusing. A second, independent convergence of the
REFERENCE's own configuration runs alongside as the control: its true signal is 1 and its true
noise 0, so whatever it reports is the method's floor. Every figure is read against it, exactly as
lag is read against stability.

The answer, stable across 44 and 140 settle frames:

```
                 retained detail    noise    psnr
reference             100%           —        —
control (native)      100%          0.50     36.1     <- the floor
taau  (0.5x -> 1x)     57-62%       0.37     34.7
lowres (0.5x upscale)  45%          0.37     32.4
```

So upsampling retains **51% of native detail against 40% for a plain upscale** — +11 points — for
+2.8 points of extra noise and 0.73 dB of PSNR. It does not reach native, which is exactly what
should be expected of it.

These figures were re-measured after the thin lens was removed, and they moved: the earlier run
read 60% against 45% at no measurable noise penalty and about +2 dB. Same method, different
renderer — the lens used to add per-frame randomness to every configuration including the control,
which flattered the comparison.

The control still does not reach 1.0, and the explanation that used to sit here was **wrong**. It
blamed the aperture offset moving the ray origin every frame. The aperture is gone now and the
control still reads 0.865 with a 0.49 noise floor, so that was not the cause — or not the only one.
What is left is the pixel jitter, which is the antialiasing and cannot be removed without removing
the thing being measured, plus the shading's own per-frame variation. The residual is a property of
the method, not of a knob that can be turned off.

## Ideas not yet done

- **Graphs, rather than numbers, in the panel.** The Monitor folder shows current values and the
  Measure folder runs each instrument on demand, but neither plots anything over time. A frame-time
  or residual sparkline would show a regression that a single number hides.

- **An art pass on the new world scale.** The planet is twice the size relative to everything else
  now, and while every coefficient that needed it was rescaled, the LOOK moved: it reads cooler and
  flatter than it did. The atmosphere's blue in-scattering integrates over twice the depth, and the
  suns want re-balancing against it. All of it is on sliders — Film, Atmosphere — so this is taste,
  not engineering.

## Where things are

```
index.html
serve.py             static dev server; exists only to send Cache-Control: no-store
src/
  main.js            boot, input wiring, the rAF loop
  renderer.js        the frame graph — pass ORDER and nothing else
  core/
    device.js        adapter/device acquisition, optional features, resize
    wgsl.js          WGSL include resolver + typed constant injection
    uniforms.js      the per-frame uniform block (camera matrices + vec4s)
    targets.js       every render target, allocated together
    profiler.js      timestamp queries
  scene/
    tuning.js        >>> every art-direction number lives here <<<
    camera.js        drift / arcball, and the view-projection matrices
    ship.js          the player ship: polar state, DERIVED orientation
    contrail.js      twin trails — one of two STATEFUL things in the scene
    aurora.js        curl-noise ribbons; the other stateful one, for the same reason
    railgun.js       a ring of shots, each an event with a birth time
    mat4.js          column-major 4x4, closed-form inverses
    input.js         mouse, touch, and the flight keys
  passes/            one file per pass, each records itself
    additive.js      ONE pass for the contrail, auroras and rail guns — see below
  dev/
    benchmark.js     the deterministic benchmark and the GPU instruments
    gui.js           the tuning panel (press g); loaded on demand
dev/                 headless checks, no GPU: node dev/run.mjs
vendor/
  lil-gui.esm.js     the one dependency (MIT), vendored, no build step
shaders/
  common.wgsl        frame uniforms, shared screen space, camera, pulse, depth tags
  brdf.wgsl          Cook-Torrance GGX; knows nothing about what it shades
  fibonacci.wgsl     spherical-Fibonacci lattices, forward and inverse
  sdf.wgsl           the body's distance field
  shade.wgsl         the translucent material
  reflect.wgsl       cheap marched reflections, for any metal surface
  rings.wgsl         the three metal hoops
  ship.wgsl          the ship's hull, and its engine and RCS plumes
  contrail.wgsl      camera-facing ribbons through past positions
  aurora.wgsl        field-aligned ribbon, curl-noise paths, emission palette
  railgun.wgsl       double-helix beams, everything derived from age
  explosion.wgsl     surface detonations
  satellite.wgsl     analytic box satellites
  sky.wgsl           stars, nebula, suns, limb glow
  ...                taa, bloom, lensflare, composite, embers, tilecull
```

The shader dependency graph is kept shallow on purpose. `brdf.wgsl` and
`fibonacci.wgsl` exist because they were once buried inside `shade.wgsl` and
`sdf.wgsl` — which meant that lighting any new surface, or drawing a single star,
pulled in the entire marched body. Anything that only wants reflectance or a lattice
should be able to say so.

**Start with `src/scene/tuning.js`.** It is the single source of truth for every
art-direction number. Values the shaders need reach them one of two ways, and the split is
deliberate: anything that should be TUNABLE WHILE LOOKING at the image — the grade, the glow,
the TAA knobs — goes into the frame uniform, and everything else is injected as a WGSL
constant. Either way there is no second copy that can drift out of sync, but only the first
kind can be moved without a pipeline rebuild, which is why the panel can bind to it directly.

## Notes for anyone reading the code

A few decisions that look odd until you know why. The comments in each file carry
the full reasoning; these are the ones worth knowing up front:

- **The field of view is set on the screen DIAGONAL**, not an axis, so a wide window
  and its own transpose frame the subject identically.
- **The projection is REVERSE-Z and INFINITE.** NDC z is 1 at the near plane and tends to 0
  with distance, which falls out of a strikingly short matrix: row 2 is `(0, 0, 0, NEAR)` and
  row 3 is `(0, 0, 1, 0)`, so `NDC z = NEAR / z_view`. WebGPU clips to `0 <= z_clip <= w_clip`
  and here `z_clip` is the constant NEAR, so the left test always passes and the right reduces
  to `z_view >= NEAR` — near clipping, no far clipping, and no far plane to choose. `NEAR` is
  now the only depth parameter in the renderer.
  Two things move with it and cannot be forgotten: the rings' depth state flips to
  `depthCompare: 'greater'` with a `0.0` clear, and ray generation unprojects **NDC z = 0**,
  which returns `w = 0` — a homogeneous direction. That last part is a genuine improvement:
  the finite version unprojected the far plane and then subtracted the camera position, and
  differencing two large world coordinates to recover a unit vector is exactly the
  cancellation that code was working around. Now there is nothing to cancel.
  Adopted for the infinite far plane, NOT for precision: reverse-Z earns its reputation by
  cancelling a FLOAT depth buffer's precision curve, and the only hardware depth buffer here
  is the rings' fixed-point `depth24plus` over a ~1.75-3 unit range that is discarded when the
  pass ends. Every other depth comparison in the pipeline works on linear view distance in an
  alpha channel. Verified as a pure reparameterisation: residual 1.72% to 1.70%, sharpness
  0.345 to 0.3453, evals 12.32 to 12.34 — all inside the noise.
- **The camera is uploaded as matrices.** Ray generation, temporal reprojection and
  the ember billboards all need the same projection, and three hand-rolled copies of
  that algebra is exactly how they stop agreeing — which was a real bug here.
- **Reprojection takes a `w` argument**: 1 for a world point, 0 for a direction. At
  `w=0` the translation column drops out, which is the rotation-only projection the
  sky needs. One code path instead of two that must match.
- **Anything fast-changing or camera-relative stays OUT of the accumulation
  buffer** — reprojection assumes static geometry. That is why embers, flares, the
  limb glow and the explosion fireballs all live downstream of TAA, while the light
  an explosion casts on the body does not.
- **TAA uses variance clipping, not a contrast guard.** A guard that discards history
  whenever it disagrees with the new sample cannot tell stale history from a noisy
  sample, and this renderer's samples are noisy by construction — so any threshold
  tight enough to stop ghosting also stops accumulation, which shimmers.
- **The ship's orientation is DERIVED, not integrated.** Its whole state is an angle
  along its orbit and that angle's rate; world position and the orientation quaternion
  are pure functions of it. A free quaternion was tried first and was worse: it let the
  ship tumble, and a tumbling ship stops reading as a 2D game and starts reading as
  debris. Deriving it means the hull is tangent to the sphere by construction rather
  than by a corrective torque, the quaternion is exactly unit with nothing to
  renormalise, and tumbling is not representable. Bank and pitch trim are cosmetic and
  never feed back into the motion — that separation is what lets a rail feel like
  flying.
- **Anything that can say where it was last frame is welcome in the accumulation
  buffer.** The rings move, but their motion is analytic, so their vertex shader
  evaluates its own position at `t` and `t - dt` and hands TAA an exact motion vector
  rather than asking it to guess. That is what anti-aliases their thin silhouettes, and
  it also put them in the bloom — which retired a fudge that used to attenuate the
  body's halo behind them.
- **Almost nothing here stores state, and the exceptions are marked.** Rings, satellites,
  detonations and the ship's hull are closed-form functions of time, so they need no
  memory and cannot desync. A CONTRAIL is a history by definition, and a RAIL SHOT is an
  event with a birth time — those two get real buffers, and the comments say why.
- **A rail gun's muzzle is stored in SHIP-LOCAL space.** Frozen in world space at fire
  time it drifts 2.5 hull-radii behind the ship over one shot's life (measured), which
  reads as beams floating in the void. The shader rebuilds the origin from the ship's live
  transform; only the DIRECTION stays frozen, because that is the part you aimed.
- **Anything reading the accumulation buffer needs one bind group PER PARITY.** The
  ping-pong flips every frame while a bind group cached on `targets.generation` only
  rebuilds on resize — so a cached view points at the wrong half half the time, and
  binding `accumRead` gets last frame's depth besides. `taa`, `composite`, `embers`,
  `contrail` and `railgun` all select by `accumIndex`.
- **Motion vectors carry an ownership claim.** Two passes write them and neither can
  see the other — the rings rasterise against their own depth buffer and know nothing of
  the marched body or the ship — so a ring passing behind either still stamps its motion
  onto that pixel. The `w` channel therefore carries the writer's own distance, and the
  consumer only trusts a vector that matches the surface actually visible. Without that,
  the body gets reprojected along a ring's motion, which is a history fetch tens of
  pixels away rather than subtle noise.
- **Reprojection must remove the jitter, and forgetting to was the actual cause of the
  shimmer.** The sample's world point lies on the JITTERED ray, but the history is indexed by
  pixel CENTRE — so reprojecting without subtracting the jitter fetches the history from a
  slightly different place every frame, and the accumulated image is re-filtered along a path
  that wanders with the jitter sequence and can never settle. Verified in closed form on the
  CPU rather than by eye: with `prevViewProj` bit-identical to `viewProj` — a provably static
  camera, where the correct drift is exactly zero — the raw reprojection landed 0.2389 output
  pixels from the pixel it was writing, against a `jitter/toIn` of 0.2361. It was tracking the
  jitter. Fixing it took the frozen residual from 3.50% to **1.74%**, against 4.20% when the
  investigation started. The other two paths were already right, which is what gave it away:
  motion vectors are built against the unjittered centre, and the background ray deliberately
  omits jitter — only the reprojection carried it.
- **TAA's depth gate is slope-scaled — and it was compensating for the bug above.** Worth
  reading in that light: it measured a real 22% win, and it earned it by absorbing a depth
  mismatch that existed because the history being compared sat about a pixel away. A genuine
  improvement sitting one layer downstream of the cause — and once the cause was fixed its
  width had to come down, from 2.0 pixels to 1.0, which cut lag 24% for no loss of stability.
  Numbers in `TEMPORAL.depthGradSlack`. A compensating fix does not stop being a compensation
  just because it measured well.

  A tolerance expressed purely as a fraction of distance is wrong on a steep surface,
  where a single pixel of reprojection error is legitimately a large depth change. On a
  frozen scene, opening the gate from 1.2% to 5% of distance dropped the frame-to-frame
  residual from 4.2% to 3.3%, and 20% and 100% gave the same — it *saturates*, so the
  gate had been rejecting valid history across most of the body's detail and ~3.2% is the
  real sampling floor. Loosening it flat is not the fix, because the tight base is what
  stops the pulse's own motion being accepted as history; scaling the tolerance by the
  local depth gradient buys the whole win (4.200% to 3.292%) while flat regions keep the
  tight guard, exactly as a rasteriser slope-scales its depth bias. Estimate the gradient
  with the LARGER of the two sides per axis, not the smaller: the intuitive
  silhouette-safe `min` under-reports on a noisy surface and recovered only 9.5%. Sky
  neighbours are excluded outright, and a cap bounds the silhouette case.
- **The history is resampled with Catmull-Rom, not bilinear.** The filter is applied every
  frame to its own output, so a low-pass compounds: fifty frames of bilinear is a blur, and
  that — not the blend — is where TAA's reputation for softness comes from. An
  interpolating cubic does not accumulate the loss. Measured: +9.5% retained detail and lag
  down from 2.07x to 1.71x noise, for +9% residual (3.21% to 3.50%, still well under the
  4.20% this all started from). Nine bilinear taps rather than sixteen, folding each axis's
  middle pair into one hardware lerp. COLOUR ONLY — the alpha is a depth tag, and a filter
  with negative lobes returns a distance no surface ever had.
- **The march is NOT step-count-bound. Do not optimise its step count.** Three attempts
  did — a correct coarse distance volume, safe over-relaxation, a rotation lookup table —
  and all three measured as noise. So the cost model got measured rather than reasoned
  about, by counting field evaluations in the shader: the mean is **12.4 per pixel**, not
  the ~60 assumed, and no knob moves it by as much as 1%. The march already converges in
  about twelve steps. At 921,600 pixels that is 11.4M evaluations in 23.9 ms, i.e. **2.1 ns
  each** — and each walks 42 lattice candidates whose `cos`/`sin`/`sqrt` are paid BEFORE the
  cheap rejection test can discard them, so a frame buys ~480M sets of transcendentals and
  throws most away. That is where the time is. Full numbers and the specific fix in the
  MARCH block of `tuning.js`; read it before touching anything here.
- **Temporal upsampling is ON, and getting there took two fixes in the wrong places first.**
  `QUALITY.taau` accumulates at display resolution and gathers the render-resolution samples by
  where they actually landed (Jimenez, SIGGRAPH 2017). Two defects, neither where it appeared:
  the resolve blended a per-frame reconstruction whose effective sample point was the sliding
  CENTROID of nearby samples — fixed by carrying an accumulated WEIGHT and forming a true
  running weighted average — and then, when that changed nothing, the real limiter turned out to
  be the variance clip's neighbourhood BOX, built from a 2x2 quad spanning two output pixels and
  shifting with the jitter. Rebuilding it from a 3x3 of input taps (as FSR2 does) took the
  noise-free signal from 9.01% to **0.71%** with the clip at full strength. Widening the clip
  instead had "worked" at 1.16% — a compensation, not a fix.
  Result on a common display-resolution grid: pattern residual 0.70% against 1.84%, lag 0.51x
  against 4.05x, and **7.6x more resolved detail** on the noise-free signal. Cost +2.8 ms.
  One open question, kept honest in `QUALITY.taau`: the real scene reads 39% *lower* sharpness,
  on a metric that cannot separate detail from noise while upsampling also carries 25% less
  noise. Likely benign, not established.
- **A measured "win" that compensates rather than fixes will keep looking like a win.** Three
  times in this project a threshold was widened, measured better, and turned out to be masking a
  defect one layer up: the depth gate's slope-scaled slack (absorbing the jitter the reprojection
  failed to remove), the upsampling clip gamma (absorbing a box that was too small), and the
  march's step scales (aimed at a step count that was already minimal). The tell each time was
  that the fix helped without the underlying number ever converging.
- **`PROBE.testPattern` renders a noise-free zone plate, and it is the sharpest instrument
  here.** The scene's own sampling noise is the same order as the artefacts worth hunting, so
  a frozen residual cannot separate "the sampler is noisy" from "the resolve is geometrically
  wrong". The synthetic signal has no noise at all, so on a correct resolve its residual must
  collapse — which is how the upsampling regression above was caught in one measurement.
- **Two coordinate grids, and nothing may assume they are equal.** `frame.res` is the render
  resolution; `frame.accumRes` is the accumulation buffer's, which differs under upsampling.
  Two bugs came out of conflating them, both worth knowing: the composite's `rpx` was doing
  double duty as an accum index AND as the render-pixel input to `cameraRay`/`screenUV`, so
  changing it for one job sent the sky ray off the sensor and faded the frame to black from
  left to right; and the additive passes index the accumulation buffer with their own
  render-grid fragment coordinates, which silently reads the top-left quadrant unless it goes
  through `toAccumPx`.
- **`renderer.resize()` asks Targets every frame, and must.** It used to gate the target
  rebuild on `syncSize` reporting a CANVAS change, which is a different question — so any
  setting that changes target sizes without changing the window (`renderScale`, and now
  `QUALITY.taau`) was silently ignored until something else forced a resize. Targets owns its
  own idempotence; the check is a couple of integer compares.
- **The auroras are the contrail with a different source of positions.** Same ring buffer,
  same additive target — only the path differs: a walker advected through CURL noise instead of
  a ship being recorded. Curl specifically, because the curl of any smooth field is
  divergence-free, so walkers swirl indefinitely instead of piling into the attractors a raw
  noise field has everywhere.
  Two early mistakes worth keeping. The ribbons were seeded COLLAPSED, as a contrail correctly
  is — but filling the buffer at a fixed interval takes most of a life, so they spent it as
  stubs; they are seeded by integrating the path up front. And the life envelope was stored
  per-sample at emission, which meant a seeded ribbon carried w = 0 everywhere and the shader
  multiplied the whole curtain by zero — visible only as a short bright dash at the tip. An
  aurora brightens as one sheet, so the envelope is a whole-ribbon value written across every
  sample each frame. A contrail genuinely wants the per-sample version; this does not.
- **The aurora ribbons went through every failure a polyline can have, and each fix was one
  layer earlier than the last.** Worth reading as a sequence, because the pattern is that
  handling a bad case is always worse than not producing it.
  1. *Barcode striation.* The ray pattern was `hash11(floor(along * freq))` — a square wave with
     random heights. Smooth value noise, two octaves, drifting along the curtain over time.
  2. *Pinched and gapped joints.* One averaged tangent per vertex, as the contrail uses, is only
     right while the path turns gently relative to the ribbon's width. Replaced with a proper
     MITER JOIN — offset along the bisector, divided by cos(theta/2), with a miter limit.
  3. *Spikes and creases through sharp turns.* The miter limit decides how a near-180-degree
     joint FAILS; nothing makes it look right. The walker had no momentum — `p += normalize(curl)
     * speed * dt` adopts the flow direction outright — so curvature was unbounded, and worst
     where the field is weakest, since normalising a curl passing through zero swings wildly.
     The walker now carries a heading and steers it at no more than `AURORA.maxTurn` rad/s,
     which bounds the path's minimum radius at `speed / maxTurn`. Measured headless over 240 s:
     worst joint angle 11.2 degrees against a limit that only engages at 139, i.e. a 1.005x
     miter. The limit is now a guard, not a working part.
  4. *Containment kinks.* A soft radial pull bounds nothing (walkers reached 2.08 and 5.55
     against a 2.9-3.6 shell); a hard clamp bounds it and creases the path, because moving a
     sample sideways IS a corner — it put a 35-degree joint in a path otherwise bounded to 8.
     Fixed in the FIELD instead, per Bridson 2007 s.3: multiply the vector potential by a
     weighting that vanishes at the shell's faces, and since `curl(alpha * psi) = alpha *
     curl(psi) + grad(alpha) x psi`, at a face only the cross-product term survives — which is
     tangential by construction. The walker cannot advect off the shell, and the field is still
     a curl, so it is still divergence-free. A PD radial steering term covers the two ways the
     walker differs from the flow (turn-rate lag, and the tangential term vanishing where psi
     lines up with the radius). Measured: 2.756 - 3.649, hard clamp never fires.
  5. *Collapse when a segment points at the camera.* `cross(tangent, toCam)` goes to zero there,
     so the width axis loses its direction and the quad spins. Snapping to a fixed axis turns a
     degeneracy into a pop, and no camera-facing formula avoids it, because at that
     configuration the ribbon genuinely has no preferred side — the information is not there.
     A blend toward a FIELD-ALIGNED axis was tried first and rejected on both counts it was
     supposed to win on. It needed the camera-facing candidate's SIGN matched against the field
     one, and that match flips whenever the view direction lies in the curtain's plane, which is
     a routine configuration and a fresh discontinuity. And a purely field-aligned curtain seen
     from outside the shell is seen almost exactly edge-on — right for a photograph of a limb,
     wrong for this scene.
     What works is noticing that the quad projects to nothing there anyway, and fading it out
     by `sin(angle between the path and the view ray)` — which is the SAME quantity twice over,
     since the projected area scales with it and scaling alpha by it is the energy correction
     for foreshortening. Without that correction the ribbon blows out to a sliding bright blob
     exactly where its orientation is least defined. Linear, not smoothstepped: linear is the
     correction, and a smoothstep goes as the square near zero and over-darkens.
  6. *Shimmer, from parameterising by BUFFER INDEX.* The worst of the lot and the last found.
     Age was read off the index — `(last - idx) / last` — and the ray phase straight off `idx`.
     But an emission shifts the whole ring buffer, so a fixed point in the WORLD keeps its
     position and loses an index: its age, and therefore its width, its fade and its ray phase,
     all jumped a full sample four and a half times a second while everything else moved
     continuously. The striation was the visible part — its phase stepped back 0.35 of a noise
     cell on every shift while the drift term slid forward, which is a sawtooth.
     Fixed by adding the fraction of the way through the current interval, which runs 0 -> 1 and
     resets exactly as `(last - idx)` gains its 1, so the sum is continuous across the shift.
     Measured by following fixed world points across 91 emissions: the per-frame age step
     deviates by 1.6e-14 samples from the ideal, against the 1.00 a shift produces.
  7. *A stuttering leading edge*, same cause one layer along: the head sample was written only
     on emission, so the tip stood still for a whole interval and then jumped the full sample
     spacing — about ten pixels, four and a half times a second. The head now tracks the walker
     every frame, so the last segment grows from nothing to full length over the interval and
     the shift that follows is exact, because the sample being committed is the one already
     there. Measured: the head moves `speed * dt` every frame, worst equal to median.
- **The additive layer is the one nothing filters, and that is a deliberate trade.** Embers,
  the contrail, the rail guns and the auroras all land in one target drawn AFTER the temporal
  resolve, because reprojection assumes static geometry and any of them would ghost. The cost is
  that they get no temporal antialiasing and, by default, they rasterise at render resolution and
  are bilinearly upscaled.
  `QUALITY.additiveDisplayRes` fixes it and is now ON. That took three measurements, and the
  first two were not enough on their own — worth following, because the second one pointed the
  wrong way.
  `beep.additive()` measured the difference the resolution makes: at render resolution the layer
  gets **51% of its own high-frequency band wrong**, carries only **60% of its high-frequency
  energy**, and is off by **17% of its own mean brightness**. Large — but it does not say which is
  BETTER, and the tempting reading is wrong: lower resolution is also blurrier, and blur suppresses
  crawl rather than causing it.
  `beep.subpixel()` settled it and disagreed with that caution. Sliding the camera across one
  pixel, the render-resolution layer is **2.2x worse** on the largest jump between adjacent
  sub-pixel offsets (0.171% against 0.076%) and **2.5x worse** on mean-brightness wobble. It steps;
  the display-resolution one drifts. The marched image, for comparison, has the highest band
  wobble of the three (0.80%) but the LOWEST adjacent step (0.107%) — smooth-but-soft, which is
  what a temporal filter should look like.
  Cost: none measurable. Six interleaved throughput runs put wall time at 31.7–32.8 ms in every
  one, with the ember pass's own spread WITHIN a configuration wider than the gap between
  configurations. This frame is bound by the march, not by additive fill. On much weaker hardware
  that stops being true, and it is one toggle in the panel.
- **Dynamic resolution works because of temporal upsampling, not alongside it.** Without TAAU the
  accumulation buffer IS the render target, so a scale change resamples the history or throws it
  away — a pop to a noisy image every time the controller moves. With it the history is at output
  resolution and only the input sample density changes.
  Getting that property actually required splitting the allocations, because `Targets.resize`
  destroyed EVERYTHING on any size change, history included. There are two ownership groups now:
  `own` for render-resolution targets and `ownOut` for output-resolution ones, with `ownOut` keyed
  and idempotent so re-running `resize` returns the existing texture instead of leaking a new one.
  The return value changed meaning too — it now says whether the HISTORY was lost, so a scale
  change no longer tells the caller to reset the accumulator. Verified by watching the
  accumulation texture's width stay constant while the render width walked 763 -> 534 -> 1526.
  It took THREE lifetimes, not two, and the third only became necessary once the ladder existed.
  The additive target's size depends on `additiveDisplayRes`, and the ladder's bottom rung flips
  exactly that flag — so with the additive size folded into the accumulation group's test, taking
  that rung destroyed the history, defeating the whole reason for the split at the one moment the
  feature most needs it. Neither change was wrong alone, which is why it survived being written and
  reviewed twice. The additive entry is retired on its own now.
  Two things had to move with it. The bloom pyramid and the flares were sized off the RENDER
  resolution, which made the glow's radius in display pixels a function of `renderScale` — halve
  the scale and the glow doubled in width, so under a controller it would visibly breathe. They are
  display-sized now, and `GLOW.radius` doubled to keep the same look through a pyramid that is
  twice as fine. And the TAA weight cap is a count of accumulated SAMPLES, so it scales with
  density: without that, a pixel that banked confidence at high resolution keeps leaning on stale
  history after the scale drops, which is ghosting.
  The controller itself is deliberately dull — GPU time not wall time (wall includes present
  pacing, so a vsync-limited frame reads as exactly on budget however much headroom there is), a
  median over a window rather than a single frame, a short ladder rather than a continuous scale
  since every change costs a reallocation, and asymmetric thresholds so it drops fast and climbs
  slowly. Measured: with an unreachable target it walks to the bottom rung and HOLDS rather than
  oscillating; with an easy one it climbs a rung per cooldown and settles at the top.
  Three things about the FEED were wrong on the first pass, and all three would have made the
  controller quietly bad rather than obviously broken. `Profiler.timings` is exponentially smoothed
  so the HUD stays readable, and taking a median of an already-smoothed series is a lag on top of a
  lag. Readbacks land every few frames while the frame hook fires every frame, so polling filled the
  window with duplicates — and the median of one repeated number is that number. And with no
  timestamp queries `total()` returns 0, so the wall-time fallback took over: at 60 Hz that reads
  16.7 ms forever, which against a 14 ms target means over budget every frame and a permanent pin to
  the bottom rung. The profiler now reports the RAW total once per resolved frame, the controller
  consumes those pushed samples, and without timestamps the feature refuses to run and says so once.
  Verified against live GPU timing by driving frames with a yield between them so readbacks can
  actually land: from scale 1.0 with an 11 ms target it dropped to 0.7 within 50 frames and HELD
  there for 350 more, with the accumulation texture unchanged the whole way.
  The tail that does not scale is the honest limit, and the ladder's last rung goes after the
  biggest remaining piece of it: keep the render scale, give up the additive layer's resolution.
  Beyond that only the resolve is left, and reducing that means lowering the OUTPUT resolution —
  which is leaving temporal upsampling's premise rather than tuning it, so the ladder stops.
- **Three passes were one pass with three names.** The contrail, the auroras and the rail guns each
  had their own ~100-line class, and the aurora and contrail files were byte-identical modulo
  identifiers and TWO lines — the shader name and where the instance count came from. That is not a
  family, it is one pass with three parameter sets, and the duplication was actively harmful rather
  than untidy: the per-parity bind-group bug was found in the contrail, and then found AGAIN in the
  aurora, because the copy did not come with the reasoning. `AdditivePass` takes a label, a shader,
  a vertex and instance count, and a thunk for the source buffer — data, not behaviour — and the
  blend, the load-not-clear, the absent depth attachment and the parity handling exist once.
  Worth being clear about what this is NOT: an entity-component system. The things in this scene
  that look like entities — rings, satellites, detonations, stars, the sphere lobes — have no CPU
  representation at all. They are closed-form functions of an index and the clock, evaluated in a
  shader, and there is no per-entity state to lay out in component arrays because there is no state.
  The five things that DO keep state already store it as parallel typed arrays, which is the layout
  an ECS exists to produce. A registry over them would add indirection and a scheduler to a problem
  that has neither dynamic composition nor a mix of behaviours to dispatch between.
- **Feeding per-frame noise to a variance-clipped accumulator poisons the WHOLE image.** The
  volumetric march jittered its sample offset every frame and leaned on the resolve to converge the
  integral — the same trade this renderer makes everywhere else, and here it was wrong.
  A variance clip builds its box from the 3x3 neighbourhood of the CURRENT sample. A noisier sample
  is a wider box, and a wider box admits more stale history — everywhere, not only where the noise
  was. The atmosphere covers most of the screen, so one deliberately-noisy term degraded everything.
  Measured with `beep.subpixel()`, which is what found it: the accumulation buffer's high-frequency
  wobble under sub-pixel camera motion read 2.37%, against 0.33% for the additive layer that has no
  antialiasing at all. Seven times worse in the part of the pipeline that is supposed to be filtered.
  Making the dither frame-STATIC — spatial interleaved-gradient noise, no temporal term — took it to
  1.274%, and the step count went 12 to 20 to carry the quality directly instead of borrowing it
  from the accumulation. The residual above the pre-atmosphere 0.80% is the integral's genuine
  reprojection error: a view-ray integral has no motion vector, so some disagreement with the
  history is inherent to computing it upstream of the resolve, which is a deliberate trade.
  It also removed the THIRD independent source of per-frame randomness. There should be one — the
  shared `frame.jitter`, expressed as a ray offset by the compute passes and a clip offset by the
  raster ones. Two expressions of one number is inherent to having both; a third number of its own
  was not, and it was the one causing the damage.
- **CLOSED: the "lag regression" was the metric's denominator, not lag.** `beep.lag()` reports
  `lag / noiseFloor`, and reading that ratio as a regression signal is wrong because the floor moves
  far more than the numerator does. Measured across render scales and aperture settings, absolute
  lag held at 6.0-7.7% while the floor ran 0.65-2.33%, so the ratio swung between 2.6x and 11.3x
  with nothing about the lag changing.
  The proof is the aperture case. Setting `CAMERA.aperture` to 0 removes per-frame randomness and
  can only improve the image — and it produces the WORST ratio of the lot, 11.33x, with lag
  unchanged at 7.31%. It is dividing by the noise it just removed. (`CAMERA.aperture` no longer
  exists — the thin lens was removed for unrelated reasons, see the note in `tuning.js` — so this
  particular experiment is history rather than something to re-run.)
  So nothing regressed: the 3.6x historical figure and the 4.9x that replaced it are the same lag
  against different noise floors. The report now leads with ABSOLUTE lag and spells out that the
  ratio is a visibility test — is the lag that exists visible above the sampling noise — and not
  something to compare between builds.
- **The atmosphere was twice the intended thickness after the world doubled.** `VOLUME.sigma` is a
  coefficient per unit length and the shell it integrates through went from 2.75 to 5.5, so the
  same 0.55 bought twice the atmosphere. Eased to 0.38, with the albedo's blue weighting softened
  from 0.42/0.62/1.0 to 0.55/0.70/1.0 — a thin aerosol's ratio is gentler than the pure-Rayleigh
  lambda^-4 that inspired the first pass.
  This was ALSO an attempt to explain the image reading cooler than it used to, and as an
  explanation it failed: mean chromaticity moved from 0.926 to 0.939 red-over-blue, which is
  nothing. The magenta bias is the body and the sky, which were always that colour. Recorded
  because a measurement that refutes your hypothesis is worth keeping — the perceptual change has
  no measurable baseline behind it, so it stays a taste call on the Film sliders rather than
  something to be chased with numbers.
- **Scaling a world is mostly a hunt for the numbers that are secretly lengths.** Doubling the
  planet was four tuning values and a day's worth of consequences, and the consequences were all
  the same shape: a constant that reads as dimensionless but is not.
  Halved, because they are coefficients PER UNIT LENGTH and now act over twice the distance: the
  medium's optical depth, the core attenuation, the aerial haze. Left alone because they really are
  ratios: the per-sphere jitter, the pulse amplitude, the shadow softness, the march's step scales
  and its hit epsilon (which is relative to `t`).
  And several were absolute lengths hiding in shader source rather than in tuning: the band over
  which surface grain mixes into the field, the shadow ray's origin offset and reach, the step
  clamps inside the shadow march, and the grain's own frequency — which is a length^-1, so it had
  to go the other way. Those are all expressed against `R` now rather than as bare numbers, which
  makes them scale-invariant by construction instead of correct until the next time.
  Symptoms, for the record: the body went opaque and dark, the embers vanished entirely (their
  spawn shell had not moved), and the surface grain became half the size it should be.
- **A shader loop hoisted the wrong thing, and it cost more than the feature.** `ringShadow` called
  `ringDefAt` inside the march — 12 steps times 2 suns times 3 rings is 72 evaluations per pixel of
  a value that depends only on the ring index and the clock, each costing a hash, three sin/cos
  pairs, a cross product and two normalises. Hoisted to 3, and the ring pass's share of the
  atmosphere fell from 0.56 ms to 0.20.
  Making a zero-thickness medium return early is what made that measurable at all: with `sigma` at
  0 genuinely free rather than merely invisible, the slider doubles as an A/B, and the atmosphere's
  cost reads straight off it — 1.94 ms on the march, 0.20 on the rings.
- **The march rejects a lattice candidate before it knows where the candidate IS.** The single
  biggest perf win in the project: **raymarch 30.7 -> 22.9 ms, a 25% cut**, with the image
  unchanged.
  The field walks a 3x3 window of spherical-Fibonacci cells per layer, six layers deep, on every
  evaluation — and then throws most of those candidates away against a distance bound that only
  needs the ANGLE between the query and the candidate. Computing a candidate means a `fract`, a
  `cos`, a `sin` and a 3x3 rotation; the bound was being tested after all of it.
  The candidate's LATITUDE alone bounds that angle. By Cauchy-Schwarz on the equatorial parts,
  `dot(lp, lcw) <= lst*ist + lct*ict`, which is exactly the cosine of the difference in polar
  angle — attained when the azimuths coincide, so it is a true upper bound on the dot product and
  therefore a true lower bound on the distance. A candidate whose BEST case is already too far to
  matter cannot be rescued by any azimuth. So: cheap polar test, then the azimuth, then the same
  test now tight, then the animation. Three tiers.
  Strictness is the whole safety property — a wrongly skipped candidate is a hole in the field,
  which the marcher renders as a hard seam and which no image metric would attribute to this. So
  it is verified directly rather than argued: over 3.1 million candidates across the densities the
  field actually uses, including deliberate near-pole cases, the worst slack is +4.1e-15.
  The same restructure moved the whole loop into the LAYER'S OWN FRAME. A rotation preserves every
  length and angle the loop asks for, so rotating the query once replaces rotating each candidate
  back — nine multiplies and six adds apiece, gone. The starfield got it too, by returning the
  cosine to its nearest point instead of the point itself, which is all any caller wanted.
- **The bloom prefilter was reading 4 of every 16 source texels.** With upsampling on, the
  accumulation buffer is display resolution while level 0 of the pyramid is half the RENDER
  resolution — a 4x reduction in one step, taken with a single bilinear tap, which covers 2x2
  texels. A star landing on one of the twelve ignored texels contributed NO glow, and popped in
  and out as it drifted. Measured by sliding one bright texel through all sixteen sub-positions of
  a footprint: the single tap produced **zero for 12 of 16 positions** and 4.9 for the others, a
  400% swing about its own mean. Four bilinear taps at the sub-quad boundaries tile the footprint
  exactly and make it position-invariant: spread 0.
  The second half of that bug was WHERE the threshold sits. Thresholding a 4x4 average destroys
  isolated highlights — a star bright enough to glow on its own, averaged against near-black sky,
  lands under the threshold and vanishes. It is applied per tap now, at the finest granularity
  bilinear allows, which is why production bloom chains threshold at the first reduction rather
  than after it.
- **The vignette was neither cos^4 nor applied where vignetting happens.** An ideal thin lens
  falls off as the fourth power of the cosine of the field angle — one power from the entrance
  pupil foreshortening, one from the image-plane element tilting, two from the inverse-square
  distance to the off-axis image point. It is not a lens defect; it happens to a PERFECT lens.
  What was here, `1 - k*r^2`, is the first two terms of that law's expansion: right at the centre
  and flat where the real curve has already started to fall.
  No trigonometry is needed, which is the neat part. `screenUV` is in half-diagonal units and the
  focal length is in the same units, so `tan(theta) = |suv| / focal` and `cos^4 = (1/(1+tan^2))^2`.
  The field angle is therefore exact, and the term now RESPONDS to the field of view as a real
  lens does — a wider lens vignettes harder, which the old form could not express.
  It also moved into linear light, before the tone curve. Vignetting is an exposure falloff at the
  sensor, not a darkening of the print; applied after the display transfer the corners were scaled
  in a space the curve had already compressed, so they never rolled through the toe.
- **Three defects the re-review turned up, all of the same shape: a quantity measured from the
  wrong origin, or with an approximation where an identity was needed.**
  The VIGNETTE measured its radius from the middle of the frame rather than from the optical axis
  — the identical mistake the lens flare's ghosts were making, sitting untouched in a second
  place. This camera composes off-centre, so the axis is 0.039 half-diagonals away, which
  displaces the falloff by about 58 device pixels and leaves one corner ~15% darker than the one
  opposite. Both read it from the same `LENS_AXIS` constant now.
  The POLAR BOUND took its equatorial magnitude as `sqrt(1 - lp.z^2)` with `lp.z` clamped. Both
  are approximations of the thing Cauchy-Schwarz actually needs, and `rot * dir` leaves |lp| a few
  ulps off unity — so the "upper" bound could come out a hair low, which is the one direction that
  drops a sphere and seams the surface. It uses `length(lp.xy)` and the raw `lp.z` now, which is
  both provably safe for any input and slightly tighter. The check was extended to feed it
  deliberately non-unit queries; worst slack +3.3e-16.
  And `QUALITY.additiveDisplayRes` tied itself to the ACCUMULATION width rather than the display
  width, which made it a silent no-op whenever upsampling was off — even though the composite
  still runs at display resolution there and still upscales the layer. Verified across all four
  combinations of the two flags.
- **`converge` read its patch from the wrong grid, and it invalidated a published result.** The
  readback copies from the ACCUMULATION buffer but was handed `targets.width` as the texture size,
  and `FrameSampler.record` centres its patch using the size it is given. With upsampling on those
  two differ by 2x, so it read a patch centred at a QUARTER of the frame while the reference read
  the middle — two configurations, two different regions, reported side by side. The
  "temporal upsampling measures 39% less sharp on the real scene" result came from here, and every
  sampler in the file is now sized off the texture it actually copies.
- **The lens flare mirrored its ghosts through the wrong point.** Internal reflections mirror
  through the lens's OPTICAL AXIS; the shader used `vec2f(0.5)`, the middle of the frame. This
  camera composes off-centre — `CAMERA.frameOffset` shears the projection so the subject's
  placement survives an aspect change — so its axis actually lands at uv (0.4805, 0.4796),
  measured by projecting the camera's own forward direction. Every ghost and the whole halo
  ring therefore sat TWICE that error off, about 100 device pixels sideways at 2560 wide, in a
  fixed direction, on every bright thing in frame. The bloom was checked at the same time and
  exonerated: cross-correlating the level-0 pyramid against a box-downsampled, thresholded copy
  of the accumulation buffer peaks at dx = +0.27 texels, i.e. aligned to within half a render
  pixel. Bloom textures carry `COPY_SRC` now so that measurement can be repeated.
- **Smoothing uses time constants, never per-frame fractions.** A fixed per-frame lag is
  frame-rate dependent: 0.08 is a ~200 ms time constant at 60 fps and ~480 ms at 25 fps,
  so the camera lagged a different amount every frame and jittered for reasons nothing
  on screen explained. `1 - exp(-dt/tau)` is the fix.
- **Reflections test a bounding sphere before marching anything.** Most reflection
  rays miss the body, and one quadratic is much cheaper than discovering that over 28
  steps. They then march a COARSE field (one octave, no holes) — detail that would
  not survive being seen in scratched, curved metal is detail not worth marching.

## Performance

The raymarch is ~90% of the frame. Everything else — TAA, the rings, the particles,
the bloom pyramid, the flares, the composite — comes to a couple of milliseconds
between them. If you want it faster, the field is the only place to look.

Numbers from `?bench` at 1280x720 render resolution. Read the SHARE of the frame, not
the milliseconds: absolute figures move 10-20% with machine load, and on a laptop they
move again with the power state, so a run taken while the machine is busy or unplugged
is not comparable to one taken idle.

| pass | share of frame | note |
|---|---|---|
| raymarch | ~89% | includes the tile cull, which is under the counter's resolution |
| rings | ~4% | geometry, reflections and motion vectors together |
| composite | ~3% | grade, grain, flares |
| taa | ~1.2% | |
| everything else | under 1% each | at or below the timestamp resolution |

Whole-frame time landed at 27.1-27.6 ms across runs on the machine this was written on,
against a 26.7 ms baseline before the ship, the rings and the reflections existed.

### Reading the pass breakdown at all

Two instrumentation traps are worth knowing about, because both produced confidently
wrong numbers here for a long time.

**Pass duration is not `end - begin`.** The beginning-of-pass timestamps are not
per-pass on this backend. In one frame's raw counters, four consecutive passes reported
`begin` at the *identical* tick, and `ember-draw` spanned 29 ms while ending at the same
instant `taa` did — for a pass that costs about 0.05 ms. Passes on one queue run
serially, so the cost of a pass is the gap between the previous pass finishing and this
one finishing. Switching to that reconstruction took `rings` from a reported 25-28 ms to
1.2 ms and made the parts sum to the frame instead of 1.5x it. `profiler.js` uses the
begins only when they are self-consistent, and exposes an `onRaw` hook so the counters
can be inspected directly rather than argued about.

**A pass that reports nothing must not be treated as zero.** Some passes come back with
both counters unwritten. Walking back to a zero end measures from the counter's epoch,
which is how `ember-draw` came out at 10 ms; such a pass now reports *unknown* and its
successor absorbs it, because the pair's combined cost is known and the split is not.

`ember-draw` is still not trustworthy — its end timestamp lands past the frame's own
span under pipelined submission. Every other pass is stable to the counter's 65 us
resolution.

That middle row is worth knowing about if you touch `sdf.wgsl`: the additive sphere
layers must use all 9 lattice candidates, because a missed nearest point makes the
min jump as the ray moves and creases the surface with seam lines. The SUBTRACTED
hole layers get away with 5 — the same error shifts a crater rim rather than tearing
a silhouette. `SFWIN` is ordered centre-first so a caller can walk a prefix.

Render resolution is half the display by default (`QUALITY.renderScale`); temporal
accumulation carries the anti-aliasing, so it is a real win rather than a downgrade.
