/**
 * WGSL module loading.
 *
 * WGSL has no preprocessor and no include mechanism, so every real project ends
 * up with a small resolver like this one. Ours supports:
 *
 *   //!include "path/relative/to/shaders.wgsl"     -- textual, include-once
 *   //!define NAME value                            -- from the JS `defines` map
 *
 * Include-once matters: the shading code pulls in `sdf.wgsl`, which pulls in
 * `hash.wgsl`, and duplicating a definition is a hard compile error in WGSL.
 *
 * Everything is cached by (path, defines) so switching quality settings does not
 * re-fetch, and `GPUShaderModule`s are cached too — creating them is not free.
 */

const SHADER_ROOT = new URL('../../shaders/', import.meta.url);

/** raw text cache: path -> Promise<string> */
const sources = new Map();

function loadSource(path) {
  let p = sources.get(path);
  if (!p) {
    // `no-cache` REVALIDATES rather than disabling the cache: the browser still stores the
    // response and still gets a 304 when nothing changed, it just never serves one without
    // asking. That matters here more than it looks like it should. A static dev server that
    // sends no Cache-Control leaves the browser applying heuristic freshness, and a stale
    // shader does not fail like a stale file — it fails as a compile error against source that
    // no longer exists on disk, naming identifiers you have already deleted. That is a genuinely
    // confusing hour, and it costs one conditional request per shader, once, at start-up.
    p = fetch(new URL(path, SHADER_ROOT), { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error(`shader not found: ${path} (${r.status})`);
        return r.text();
      })
      .catch((e) => {
        sources.delete(path);              // let a later attempt retry
        throw e;
      });
    sources.set(path, p);
  }
  return p;
}

const INCLUDE_RE = /^[ \t]*\/\/!include[ \t]+"([^"]+)"[ \t]*$/gm;

/** Resolve includes depth-first, emitting each file at most once. */
async function resolve(path, seen, defines) {
  if (seen.has(path)) return '';
  seen.add(path);

  const raw = await loadSource(path);

  // Gather includes first so the fetches can overlap.
  const deps = [...raw.matchAll(INCLUDE_RE)].map((m) => m[1]);
  const resolvedDeps = new Map();
  await Promise.all(
    deps.map(async (d) => resolvedDeps.set(d, await resolve(d, seen, defines)))
  );

  const body = raw.replace(INCLUDE_RE, (_, dep) => resolvedDeps.get(dep) ?? '');
  return `// ---- ${path} ----\n${body}\n`;
}

/**
 * Mark a define as an integer.
 *
 * This is not sugar, it is necessary: JavaScript cannot tell `12` from `12.0`,
 * so inferring the WGSL type from the value silently types every round float as
 * i32. WGSL then rejects it at the first arithmetic or call site — and the error
 * appears wherever the constant is *used*, not where it was declared, which is a
 * miserable thing to debug. Types are therefore declared, never guessed.
 */
export function int(v) {
  return { __wgslInt: v | 0 };
}

function defineBlock(defines) {
  const entries = Object.entries(defines);
  if (!entries.length) return '';
  const lines = entries.map(([k, v]) => {
    if (typeof v === 'boolean') return `const ${k} : bool = ${v};`;
    if (v && typeof v === 'object' && '__wgslInt' in v) {
      return `const ${k} : i32 = ${v.__wgslInt};`;
    }
    // Every plain number is f32. Integers are the exception and must say so.
    if (typeof v === 'number') return `const ${k} : f32 = ${wgslFloat(v)};`;
    return `const ${k} = ${v};`;                 // escape hatch: raw WGSL
  });
  return `// ---- injected defines ----\n${lines.join('\n')}\n`;
}

/**
 * WGSL needs a decimal point to read a literal as f32.
 * Exported because tuning.js formats vector literals by hand and had its own
 * copy of this — one rounding convention, one place.
 */
export function wgslFloat(v) {
  return Number.isInteger(v) ? `${v}.0` : String(v);
}

/**
 * Assemble the full source for an entry file.
 * `defines` become real WGSL `const`s, which the driver folds just like a macro
 * while remaining type-checked.
 */
export async function buildSource(path, defines = {}) {
  const code = await resolve(path, new Set(), defines);
  return `${defineBlock(defines)}\n${code}`;
}

export class ShaderCache {
  constructor(device) {
    this.device = device;
    this.modules = new Map();
  }

  /** Cached `GPUShaderModule` for (path, defines). */
  async module(path, defines = {}) {
    const key = `${path}::${JSON.stringify(defines)}`;
    let entry = this.modules.get(key);
    if (entry) return entry;

    entry = (async () => {
      const code = await buildSource(path, defines);
      const mod = this.device.createShaderModule({ label: path, code });

      // Surface diagnostics eagerly: a warning here is usually a real bug, and
      // silently-degraded shaders are miserable to debug later.
      const info = await mod.getCompilationInfo();
      if (info.messages.length) {
        const lines = code.split('\n');
        for (const m of info.messages) {
          const where = m.lineNum ? `${path}:${m.lineNum}:${m.linePos}` : path;
          const src = m.lineNum ? `\n    ${lines[m.lineNum - 1] ?? ''}` : '';
          const text = `[${m.type}] ${where} — ${m.message}${src}`;
          if (m.type === 'error') console.error(text);
          else console.warn(text);
        }
        if (info.messages.some((m) => m.type === 'error')) {
          throw new Error(`WGSL compile failed in ${path} (see console)`);
        }
      }
      return mod;
    })();

    this.modules.set(key, entry);
    return entry;
  }
}
