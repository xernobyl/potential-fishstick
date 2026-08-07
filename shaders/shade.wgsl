// ---------------------------------------------------------------------------
// Material and lighting: a translucent gummy.
//
// A Cook-Torrance GGX candy coat (roughness patched by noise: glossy where wet,
// matte where sugar-frosted) over a scattering interior:
//
//   - ambient occlusion = analytic sphere AO x a cheap SDF AO (iq)
//   - the SAME sphere integral about -N gives LOCAL THICKNESS. Inverting
//     occlusion to get thickness is Barre-Brisebois & Bouchard, "Approximating
//     Translucency" (GDC 2011). Both fall out of one pass over the neighbours.
//   - that thickness drives Beer-Lambert extinction with a per-channel sigma_t
//     (the medium absorbs its own complement) and an HG phase function
//   - real refraction bends the view ray inside and marches, so the core reads
//     as a solid glowing object rather than a surface-only fake
// ---------------------------------------------------------------------------

//!include "brdf.wgsl"
//!include "explosion.wgsl"
//!include "sdf.wgsl"
//!include "sky.wgsl"

// ---- Cook-Torrance terms ------------------------------------------------

fn sphOcclusion(pos : vec3f, nor : vec3f, sph : vec4f) -> f32 {
  let di = sph.xyz - pos;
  let l = length(di);
  let nl = dot(nor, di / l);
  let h = l / sph.w;
  let h2 = h * h;
  let k2 = 1.0 - h2 * nl * nl;
  var res = max(0.0, nl) / h2;
  if (k2 > 0.001) {
    res = nl * acos(-nl * sqrt((h2 - 1.0) / (1.0 - nl * nl))) - sqrt(k2 * (h2 - 1.0));
    res = (res / h2 + atan2(sqrt(k2 / (h2 - 1.0)), 1.0)) / PI;
  }
  return res;
}

/// One pass over the neighbouring spheres, two results — the Fibonacci lookups
/// are the expensive part, so both integrals share them:
///   .x ambient visibility, the occlusion integrated about +nor  (AO)
///   .y local thickness, the SAME integral about -nor
fn sphereAOThickness(pos : vec3f, nor : vec3f) -> vec2f {
  var vis = 1.0;
  var back = 1.0;
  let dir = normalize(pos);
  var surf = R;
  var n = N0;
  var rho = RHO0;

  for (var oct = 0; oct < 2; oct++) {
    let c = sfCell(dir, n, oct);
    var capH = 0.0;
    for (var j = 0; j < SFN; j++) {
      let cand = sfCandidate(c, j);
      if (!cand.ok) { continue; }
      let i = cand.idx;
      let cw = cand.cw;

      let jr = (hash11(i + f32(oct) * 57.1) - 0.5) * JITTER * rho;
      var rr = rho * (0.85 + 0.35 * hash11(i * 1.7 + f32(oct) * 13.3));
      // must match layerDist's pulse exactly, or the AO drifts off the surface
      let ph = hash11(i * 0.77 + f32(oct) * 3.13);
      let amp = 0.65 + 0.70 * hash11(i * 1.31 + f32(oct) * 7.71);
      let beat = heartbeat(beatPhase() - ph * PULSE_LAG);
      rr *= 1.0 + PULSE_R * amp * beat;
      rr *= lifeEnvelope(i, oct);
      let s = vec4f(cw * (surf + jr + PULSE_OFF * rho * amp * (beat - 0.35)), rr);

      // The 3x3 window reaches well past the spheres that matter. sphOcclusion
      // falls off as 1/h^2, so past ~6 radii it is worth under a few percent —
      // skip those instead of paying its acos/atan. The weight fades to exactly
      // 0 at the skip radius: a hard cut would make AO jump as spheres cross it,
      // which reads as faint facets on the big lobes.
      let l2 = dot(pos - s.xyz, pos - s.xyz);
      let fade = 1.0 - smoothstep(rr * rr * 16.0, rr * rr * 36.0, l2);
      if (l2 > rr * rr * 1.0404 && fade > 0.0) {
        vis  *= 1.0 - clamp(sphOcclusion(pos,  nor, s) * fade, 0.0, 1.0);
        back *= 1.0 - clamp(sphOcclusion(pos, -nor, s) * fade, 0.0, 1.0);
      }
      let u = surf * length(dir - cw) / rr;
      capH = max(capH, rr * max(0.0, 1.0 - u * u));
    }
    surf += capH * 0.62;
    n *= NGROW;
    rho *= RHOFALL;
  }
  return vec2f(vis, 1.0 - back);
}

/// Cheap SDF ambient occlusion (overall concavity), on the coarse field.
fn calcAO(p : vec3f, n : vec3f) -> f32 {
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.012 + 0.14 * f32(i) / 4.0;
    occ += (h - mapImpl(p + n * h, 2, false)) * sca;
    sca *= 0.7;
  }
  return clamp(1.0 - 1.6 * occ, 0.0, 1.0);
}


// ---- the gummy ---------------------------------------------------------

/// Shade a hit on the body. `p` is the surface point, `rd` the view ray.
fn shadeBody(p : vec3f, rd : vec3f, t : f32) -> vec3f {
  let nor = calcNormal(p);
  let rn = normalize(p);
  let V = -rd;
  let N = nor;

  // AO *and* local thickness from one sphere-integral pass.
  let aot = sphereAOThickness(p, nor);
  let ao = clamp(aot.x * calcAO(p, nor), 0.0, 1.0);
  // Only the sphere integral is usable for thickness: an SDF AO probed along -N
  // marches *into* the body, pins to 0, and would saturate this to 1 everywhere,
  // flattening the whole effect. Floored a little — nothing on a gummy is truly
  // clear, and a zero blows the thin tips out to white.
  let thick = clamp(aot.y * 1.15, 0.16, 1.0);

  // BAND-LIMIT THE FINE DETAIL BY THE PIXEL FOOTPRINT.
  //
  // `fleck` runs at 34 per unit and the grain at up to 15/R, which at any distance is finer than the
  // sampling grid can carry. Sampling it anyway produces two costs at once: it aliases in space, and
  // because the jitter moves the sample every frame it varies in time, which the resolve then has to
  // admit into its variance clip. Fading each octave out as it approaches Nyquist removes both, and it
  // is the correct filtering rather than a trade - the octave genuinely carries no information at that
  // scale.
  //
  // The footprint is the world size of one pixel at this hit: screen space spans 2 over `diagPx`
  // pixels, and a screen offset maps to a world offset of `t/focal` times that. Grazing angles are
  // deliberately NOT folded in - the 1/dot(N,V) term is unbounded at the silhouette and would fade the
  // grain to nothing along every limb, which is where the eye most expects it.
  //
  // The smooth fbm terms below are left alone: at 2 to 4.5 per unit they sit far below the pixel rate
  // for any view that fills the frame with the body, so filtering them would cost detail and buy
  // nothing. `frame.march.x` switches this, so both sides can be measured.
  let fp = select(0.0, t * 2.0 * frame.screen.x / max(frame.camFwd.w, 1e-4), frame.march.x > 0.5);
  let fleck = vnoiseBL(p * 34.0, 34.0, fp);
  let dn = detailNoiseBL(p, fp);

  // Purple gummy: hue and value both wander on smooth noise. Deliberately NOT
  // keyed to the Fibonacci blob id — a per-blob step puts a hard Voronoi edge
  // across every large lobe, which reads as faceting.
  let hueN = clamp(fbm3(p * 2.2 + 5.0) * 1.5 - 0.25, 0.0, 1.0);
  var candy = mix(vec3f(1.00, 0.12, 0.62), vec3f(0.42, 0.16, 1.00), hueN);
  candy = mix(candy, vec3f(0.72, 0.18, 1.00), 0.30);
  candy *= 0.86 + 0.28 * fbm3(p * 3.7 - 12.0);

  // Sugar-dusted patches: noise-driven roughness, matte where frosted.
  var sugar = smoothstep(0.42, 0.72, fbm3(p * 4.5 + 31.0));
  sugar = clamp(sugar + 0.22 * fleck - 0.08, 0.0, 1.0);

  // The look is mostly transmission, so keep the diffuse albedo low.
  var alb = candy * 0.10 + 0.02;
  alb = mix(alb, candy * 0.26 + 0.11, sugar);
  alb *= 0.86 + 0.28 * dn;

  // Extinction: the medium absorbs the complement of its own hue.
  var sigmaT = (1.0 - candy) * GUM_DENS + 0.35;
  sigmaT *= 1.0 + 1.2 * sugar;

  let f0 = vec3f(0.05);                                  // candy coat, IOR ~1.5
  let rough = clamp(mix(0.13, 0.70, sugar) + 0.10 * (dn - 0.5), 0.05, 1.0);
  // Offsets and reach are body-space lengths, so they are expressed against R. `w` is a softness
  // RATIO (h / (w * t)) and is dimensionless, so it stays put.
  let sha1 = softshadow(p + N * (R * 0.02), SUN1_DIR, R * 0.02, R * 6.0, 0.11);
  let sha2 = softshadow(p + N * (R * 0.02), SUN2_DIR, R * 0.02, R * 6.0, 0.11);

  let direct = sunLight(N, V, SUN1_DIR, SUN1_COL, alb, rough, f0, sha1)
             + sunLight(N, V, SUN2_DIR, SUN2_COL, alb, rough, f0, sha2)
             // Fill light, unshadowed on purpose — see SUNS.c in tuning.js.
             + sunLight(N, V, SUN3_DIR, SUN3_COL, alb, rough, f0, 1.0)
             + blastLight(p, N, V, alb, rough, f0);

  // Transmission: light that travelled *through* the medium. Beer-Lambert over
  // the local thickness, forward-scattered, plus a wrapped term so whole lobes
  // glow rather than just rims.
  let T = beerLambert(thick * GUM_DEPTH, sigmaT);
  let wr1 = max(0.0, (dot(N, SUN1_DIR) + 0.6) / 1.6);
  let wr2 = max(0.0, (dot(N, SUN2_DIR) + 0.6) / 1.6);
  var trans = SUN1_COL * T * (phaseHG(dot(V, -SUN1_DIR), GUM_G) + 0.35 * wr1) * mix(0.35, 1.0, sha1)
            + SUN2_COL * T * (phaseHG(dot(V, -SUN2_DIR), GUM_G) + 0.35 * wr2) * mix(0.35, 1.0, sha2);
  trans *= candy * (1.0 - 0.55 * sugar);

  // Ambient IBL approximation: cool sky dome, violet-tinted bounce from below.
  let sky = vec3f(0.30, 0.40, 0.62);
  let grd = vec3f(0.16, 0.09, 0.22);
  let env = mix(grd, sky, 0.5 + 0.5 * N.y);
  let NoV = clamp(dot(N, V), 1e-4, 1.0);
  let Fenv = fresnelSchlickRough(NoV, f0, rough);
  let ambient = ao * (alb * env * (1.0 - Fenv) * 1.15 + env * Fenv * 0.30);

  // Red-hot core glowing out through the shell. Same medium, but the path length
  // is how far this point sits above the core, so it burns through thin walls and
  // blazes inside the carved holes.
  let beatG = heartbeat(beatPhase());
  let coreRad = CORE_R * (1.0 + 0.10 * beatG);
  let shell = max(length(p) - coreRad, 0.0);
  var coreEm = CORE_COL * (0.75 + 0.55 * beatG) * beerLambert(shell * CORE_ATT, sigmaT * 0.55);
  coreEm *= 0.55 + 0.45 * (1.0 - thick);
  coreEm *= 0.62;                                        // refraction carries the sharp view

  // Real refraction: bend the view ray into the body and march. This is the
  // difference between a surface-only fake and actually seeing into the
  // material. The interior uses the coarse hole-free field — interior detail is
  // invisible through a dense medium, so paying for it would be waste.
  var refr = vec3f(0.0);
  let ri = refract(rd, N, 1.0 / GUM_IOR);
  if (dot(ri, ri) > 0.0) {                               // 0 on total internal reflection
    let q0 = p - N * 0.006;
    var tt = 0.0;
    for (var i = 0; i < 28; i++) {
      let dd = -mapImpl(q0 + ri * tt, 2, false);         // inside: -d is the wall
      if (dd < 0.002) { break; }
      tt += max(dd * 0.8, 0.01);
      if (tt > 6.0) { break; }
    }
    let cs = iSphere(q0, ri, coreRad);
    if (cs.x > 0.0 && cs.x < tt) {
      // terminates in the core: emission attenuated over the path to it
      refr = CORE_COL * (0.75 + 0.55 * beatG) * beerLambert(cs.x, sigmaT * 0.75);
    } else {
      // exits: whatever is behind, tinted by the path travelled. Dispersion is
      // applied per channel on the exit direction only — three sharp-sky taps
      // are nearly free, whereas three interior marches would not be.
      let aC = beerLambert(tt, sigmaT);
      var rr2 = refract(rd, N, 1.0 / (GUM_IOR - DISPERSE));
      var rb = refract(rd, N, 1.0 / (GUM_IOR + DISPERSE));
      if (dot(rr2, rr2) == 0.0) { rr2 = ri; }
      if (dot(rb, rb) == 0.0) { rb = ri; }
      // Nebula once — it is smooth and cannot resolve a 0.02 IOR shift — and
      // only the sharp stars/suns per channel, which is where fringing lives.
      let bg = bgNebula(ri)
             + vec3f(bgSharp(rr2).r, bgSharp(ri).g, bgSharp(rb).b);
      refr = bg * aC + candy * aC * 0.30;
    }
  }
  let Fv = fresnelSchlick(clamp(dot(N, V), 0.0, 1.0), f0).g;

  var col = direct + ambient + trans * 0.55 + coreEm + refr * (1.0 - Fv) * REFR_GAIN;

  // Candy rim: thin edges transmit almost everything.
  let fre = pow(clamp(1.0 + dot(rd, N), 0.0, 1.0), 3.0);
  col += fre * candy * 0.30 * (1.0 - sugar);
  col += fre * ao * vec3f(0.22, 0.24, 0.40) * 0.5;

  // Thin-film interference on the frosted patches: a sugar crust is a stack of
  // microcrystals, so it throws a faint oil-slick tint that shifts with angle.
  let iri = 0.5 + 0.5 * cos(TAU * (fre * 3.0 + dn * 0.8) + vec3f(0.0, 2.1, 4.2));
  col += iri * sugar * pow(fre, 2.0) * 0.30;

  // Aerial haze with view distance across the little world.
  let haze = 1.0 - exp(-0.05 * max(t - 6.0, 0.0));
  col = mix(col, vec3f(0.42, 0.44, 0.58) * 0.5, haze * 0.45);
  return col;
}
