import assert from "node:assert/strict";
import test from "node:test";

import { diffConfigs, flattenConfig, formatConfigValue } from "./configDiff.js";

test("flattenConfig uses dotted keys and keeps arrays and empty objects as leaves", () => {
  assert.deepEqual(flattenConfig({ lr: 1e-6, recipe: { tp: 2, gpus: [0, 1], extra: {} }, name: null }), {
    lr: 1e-6,
    "recipe.tp": 2,
    "recipe.gpus": [0, 1],
    "recipe.extra": {},
    name: null,
  });
});

test("diffConfigs lines runs up by key and marks the keys that differ", () => {
  const rows = diffConfigs([
    { lr: 1e-6, recipe: { tp: 2, async: true } },
    { lr: 1e-6, recipe: { tp: 4, async: true }, extra: "x" },
  ]);
  assert.deepEqual(rows, [
    { key: "extra", values: [undefined, "x"], differs: true },
    { key: "lr", values: [1e-6, 1e-6], differs: false },
    { key: "recipe.async", values: [true, true], differs: false },
    { key: "recipe.tp", values: [2, 4], differs: true },
  ]);
});

test("diffConfigs compares arrays by value and treats a missing key as different from null", () => {
  const rows = diffConfigs([{ gpus: [0, 1], name: null }, { gpus: [0, 1] }]);
  assert.deepEqual(rows.map(({ key, differs }) => [key, differs]), [
    ["gpus", false],
    ["name", true],
  ]);
});

test("diffConfigs search is a case-insensitive substring match on the key", () => {
  const rows = diffConfigs([{ recipe: { lr: 1 }, model: "a" }], " LR ");
  assert.deepEqual(rows.map((row) => row.key), ["recipe.lr"]);
});

test("formatConfigValue shows missing keys as a dash and strings unquoted", () => {
  assert.equal(formatConfigValue(undefined), "—");
  assert.equal(formatConfigValue("qwen"), "qwen");
  assert.equal(formatConfigValue(null), "null");
  assert.equal(formatConfigValue([1, 2]), "[1,2]");
});
