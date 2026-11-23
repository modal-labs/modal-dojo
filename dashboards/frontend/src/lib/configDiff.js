// Helpers for the Metrics page's config diff: flatten each run's nested
// config to dotted keys and line the runs up key by key.

// `{ "a.b": leaf }` for every leaf; arrays are leaves, compared as a whole.
export function flattenConfig(config, prefix = "", out = {}) {
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    if (prefix) out[prefix] = config;
    return out;
  }
  const keys = Object.keys(config);
  if (!keys.length && prefix) out[prefix] = config;
  for (const key of keys) {
    flattenConfig(config[key], prefix ? `${prefix}.${key}` : key, out);
  }
  return out;
}

// `undefined` marks a key the run doesn't have, so it can differ from `null`.
export function formatConfigValue(value) {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

// One row per key in any run, sorted: `{ key, values, differs }`, with
// `values` in `configs` order. A key missing from a run differs from one set.
export function diffConfigs(configs, search = "") {
  const flat = configs.map((config) => flattenConfig(config ?? {}));
  const keys = new Set(flat.flatMap((entries) => Object.keys(entries)));
  const needle = search.trim().toLowerCase();
  return [...keys]
    .filter((key) => key.toLowerCase().includes(needle))
    .sort()
    .map((key) => {
      const values = flat.map((entries) => (key in entries ? entries[key] : undefined));
      const shown = values.map((value) => (value === undefined ? undefined : JSON.stringify(value)));
      return { key, values, differs: shown.some((value) => value !== shown[0]) };
    });
}
