// ---------------------------------------------------------------------------
// The ember sprite, baked once at start-up.
//
// Every mote shares one appearance, so evaluating its profile per-pixel per-
// particle every frame would be pure waste. Baking it into a small mipped
// texture turns the per-particle cost into one bilinear tap, and the mip chain
// means a distant mote reads a pre-filtered level instead of aliasing.
//
// The profile is a tight core plus a wide falloff — a single Gaussian reads
// either as a hard dot or as fog, never as a spark.
// ---------------------------------------------------------------------------

//!include "fullscreen.wgsl"

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let d = (uv - 0.5) * 2.0;
  let r = clamp(length(d), 0.0, 1.0);

  let core = exp(-r * r * 22.0);          // the spark itself
  let halo = exp(-r * r * 3.2) * 0.22;    // the air around it
  // Hard zero at the rim so neighbouring sprites in the atlas cannot bleed and
  // so the quad's own edge is invisible.
  let edge = 1.0 - smoothstep(0.75, 1.0, r);

  let a = (core + halo) * edge;
  // Slightly hotter in the middle: colour is applied per particle, this is only
  // the shape and its internal temperature gradient.
  let tint = mix(vec3f(1.0, 0.72, 0.42), vec3f(1.0), core * 0.75);
  return vec4f(tint * a, a);
}
