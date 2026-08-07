// ---------------------------------------------------------------------------
// Sparks: the burst a shot throws off, and the swarm a charge pulls in.
//
// TWO EFFECTS, ONE PASS, because they are the same billboard with a different position function and
// they are never both interesting at once — you charge, then you fire. Splitting them would be two
// passes, two pipelines and two bind groups to draw a few dozen quads.
//
// NOTHING IS SIMULATED. Every spark's whole trajectory is a closed form in (shot seed, spark index,
// age), exactly as the beams are: the CPU uploads a shot record and never touches it again. That is the
// same property the rest of this renderer leans on — a particle that can say where it is at time t can
// also say where it was last frame, which is what keeps the temporal resolve from smearing it.
//
// The charge swarm runs the clock BACKWARDS. Sparks spiral inward and arrive at the muzzle as the
// charge completes, so the wind-up reads as gathering rather than as a glow that happens to brighten.
// It borrows the shot's hue before the shot exists, which is why the hue is chosen on the trigger's
// press rather than its release.
// ---------------------------------------------------------------------------

//!include "common.wgsl"
//!include "hash.wgsl"
//!include "volumetric.wgsl"

struct Shot {
  local : vec4f,    // xyz muzzle in SHIP-LOCAL space, w fire time
  dir   : vec4f,    // xyz world direction frozen at fire time, w which wing
  extra : vec4f,    // x power 0..1, y hue, z seed, w spare
};

@group(1) @binding(0) var<storage, read> shots : array<Shot>;
@group(1) @binding(1) var sceneTex : texture_2d<f32>;   // alpha = depth tag

const QUAD = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
);

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) uv    : vec2f,
  @location(1) tint  : vec3f,
  @location(2) fade  : f32,
  @location(3) viewZ : f32,
  @location(4) atmo  : vec3f,
};

fn shotTint(hue : f32) -> vec3f {
  return 0.55 + 0.45 * cos(TAU * (hue + vec3f(0.0, 0.33, 0.67)));
}

/// A perpendicular basis around an axis. The scatter is radially symmetric, so the choice is free.
fn basisAround(axis : vec3f, e0 : ptr<function, vec3f>, e1 : ptr<function, vec3f>) {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
  *e0 = normalize(cross(up, axis));
  *e1 = cross(axis, *e0);
}

fn collapse(out : ptr<function, VOut>) {
  (*out).pos = vec4f(0.0, 0.0, 2.0, 1.0);
  (*out).uv = vec2f(0.0);
  (*out).tint = vec3f(0.0);
  (*out).fade = 0.0;
  (*out).viewZ = 0.0;
  (*out).atmo = vec3f(1.0);
}

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VOut {
  var out : VOut;

  let perShot = u32(RAIL_SPARKS);
  let shotIdx = ii / perShot;
  let k = f32(ii % perShot);

  // The LAST slot of the pool is the charge swarm rather than a shot's burst. It is drawn from the
  // live trigger state instead of a stored record, so it needs no pool entry of its own — and the pool
  // is a ring, so borrowing a slot's INSTANCES costs nothing as long as no shot is read from it.
  let isCharge = shotIdx >= u32(RAIL_POOL);
  var centre : vec3f;
  var tint : vec3f;
  var fade : f32;
  var size : f32;

  if (isCharge) {
    let charge = frame.weapon.x;
    if (charge <= 0.001 || k >= RAIL_CHARGE_SPARKS) { collapse(&out); return out; }

    // Both muzzles feed the charge, so half the swarm gathers at each wing tip.
    let side = select(-1.0, 1.0, (u32(k) & 1u) == 0u);
    let muzzle = frame.shipPos.xyz
               + qrotate(frame.shipRot, vec3f(RAIL_SPREAD * side, RAIL_UP, RAIL_FORWARD));

    // Each spark has its own approach: a random direction out, and a phase so they do not arrive
    // together. `frac` runs 1 -> 0 as the charge fills, which is the inward spiral.
    let h = hash13(vec3f(k * 1.7, 3.1, 7.7));
    let ang = h * TAU + frame.camPos.w * 2.4 + k;
    let phase = hash11(k * 0.37 + 11.0);
    let frac = clamp(1.0 - charge + phase * 0.35, 0.0, 1.0);

    let axis = qrotate(frame.shipRot, vec3f(0.0, 0.0, 1.0));
    var e0 : vec3f;
    var e1 : vec3f;
    basisAround(axis, &e0, &e1);
    let r = RAIL_CHARGE_RADIUS * frac;
    // A little forward wander too, so the swarm is a volume rather than a disc.
    let alongAxis = (hash11(k * 5.9) - 0.5) * RAIL_CHARGE_RADIUS * frac;
    centre = muzzle + (e0 * cos(ang) + e1 * sin(ang)) * r + axis * alongAxis;

    tint = shotTint(frame.weapon.y);
    // Brightest as it lands, and the whole swarm brightens with the charge.
    fade = charge * (1.0 - frac) * 0.9 + 0.1 * charge;
    size = RAIL_SPARK_SIZE * (0.7 + 1.1 * charge);
  } else {
    let sh = shots[shotIdx];
    let power = sh.extra.x;
    // Ordinary shots throw a handful; a power shot throws the lot.
    let live = mix(RAIL_SPARKS * 0.22, RAIL_SPARKS, power);
    if (k >= live) { collapse(&out); return out; }

    let age = (frame.camPos.w - sh.local.w) / (RAIL_SPARK_LIFE * mix(1.0, RAIL_POWER_LIFE, power));
    if (age < 0.0 || age > 1.0) { collapse(&out); return out; }

    let muzzle = frame.shipPos.xyz + qrotate(frame.shipRot, sh.local.xyz);
    let axis = normalize(sh.dir.xyz);
    var e0 : vec3f;
    var e1 : vec3f;
    basisAround(axis, &e0, &e1);

    // Direction: mostly along the beam, splayed sideways by the spread. Hashed off the shot's own
    // seed so two shots never scatter alike.
    let seed = sh.extra.z;
    let a = hash21(vec2f(k, seed)) * TAU;
    let radial = hash21(vec2f(k + 31.0, seed));
    let speed = RAIL_SPARK_SPEED * (0.35 + 0.65 * hash21(vec2f(k + 61.0, seed)))
              * mix(1.0, 1.8, power);
    let dir = axis * (0.3 + 0.7 * radial)
            + (e0 * cos(a) + e1 * sin(a)) * RAIL_SPARK_SPREAD * radial;

    // Eased so they burst out and coast, rather than travelling at a constant rate.
    let travel = (1.0 - exp(-age * 3.2)) * speed;
    centre = muzzle + dir * travel;

    tint = shotTint(sh.extra.y);
    fade = pow(1.0 - age, 1.6);
    size = RAIL_SPARK_SIZE * mix(0.7, 1.5, power) * (0.35 + 0.85 * (1.0 - age));
  }

  if (fade <= 0.001) { collapse(&out); return out; }

  // Camera-facing billboard. The basis comes from the view matrix's rows, so it needs no per-particle
  // work and stays exactly square on screen.
  let toCam = normalize(frame.camPos.xyz - centre);
  var right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam));
  if (abs(toCam.y) > 0.99) { right = vec3f(1.0, 0.0, 0.0); }
  let up = cross(toCam, right);

  let c = QUAD[vi % 6u];
  let wp = centre + (right * c.x + up * c.y) * size;

  out.pos = frame.viewProj * vec4f(wp, 1.0);
  out.uv = c;
  out.tint = tint;
  out.fade = fade;
  out.viewZ = length(wp - frame.camPos.xyz);
  // Extinction only, as every additive layer does it — see volTransmittance.
  let toEye = wp - frame.camPos.xyz;
  let dist = length(toEye);
  out.atmo = volTransmittance(frame.camPos.xyz, toEye / max(dist, 1e-6), dist);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4f {
  if (in.fade <= 0.0) { discard; }
  // A round, soft spark. Squared falloff rather than a hard disc: these are small and a crisp edge
  // would alias badly against a layer nothing filters.
  let d = dot(in.uv, in.uv);
  if (d > 1.0) { discard; }
  let profile = (1.0 - d) * (1.0 - d);

  // Soft occlusion against the scene, as the beams and the embers do it: dissolve rather than clip.
  let sceneW = textureLoad(sceneTex, toAccumPx(in.pos.xy), 0).a;
  let vis = smoothstep(0.0, 0.06, tagDepth(sceneW) - in.viewZ);

  let a = in.fade * profile * vis;
  // A white-hot core inside the tint, which is what makes a spark read as hot rather than coloured.
  let col = mix(in.tint, vec3f(1.0), profile * profile * 0.7);
  return vec4f(col * a * RAIL_SPARK_GAIN * in.atmo, a);
}
