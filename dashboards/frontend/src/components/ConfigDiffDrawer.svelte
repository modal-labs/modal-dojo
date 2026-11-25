<script>
  import { PanelRightClose } from "lucide-svelte";
  import Drawer from "./Drawer.svelte";
  import { fetchRun } from "../lib/api.js";
  import { diffConfigs, formatConfigValue } from "../lib/configDiff.js";

  // `runs`: `[{ id, label, color }]` in the Metrics page's order. The first
  // run is the baseline other runs' cells are highlighted against.
  let { open = false, runs = [], onclose = () => {} } = $props();

  // run id -> { config } | { error }, kept across opens; a run's config
  // doesn't change once it launches.
  let configs = $state({});
  let onlyDifferences = $state(true);
  let search = $state("");

  $effect(() => {
    if (!open) return;
    for (const run of runs) {
      if (configs[run.id]) continue;
      configs[run.id] = { loading: true };
      fetchRun(run.id)
        .then((detail) => {
          configs[run.id] = detail ? { config: detail.config ?? {} } : { error: "Run not found" };
        })
        .catch((error) => {
          configs[run.id] = { error: error.message || String(error) };
        });
    }
  });

  let loaded = $derived(runs.filter((run) => configs[run.id]?.config));
  let loading = $derived(runs.some((run) => configs[run.id]?.loading));
  let allRows = $derived(diffConfigs(loaded.map((run) => configs[run.id].config), search));
  let differingCount = $derived(allRows.filter((row) => row.differs).length);
  let rows = $derived(onlyDifferences ? allRows.filter((row) => row.differs) : allRows);

  function differsFromBaseline(row, index) {
    return index > 0 && JSON.stringify(row.values[index]) !== JSON.stringify(row.values[0]);
  }
</script>

<Drawer {open} {onclose} width="min(960px, calc(100vw - 24px))">
  <div class="w-full h-full flex flex-col" aria-label="Config diff">
    <div class="drawer-panel-header">
      <div class="min-w-0">
        <div class="drawer-panel-eyebrow">Compare runs</div>
        <h2 class="text-(--text-bright) text-[16px] font-medium leading-[24px] m-0">Config diff</h2>
        <div class="text-(--muted) text-[12px] mt-[2px]">
          {#if loading}
            Loading configs…
          {:else}
            {differingCount} of {allRows.length} {search.trim() ? "matching " : ""}keys differ across {loaded.length} runs
          {/if}
        </div>
      </div>
      <button class="drawer-panel-close ghost-hover" onclick={onclose} aria-label="Close config diff">
        <PanelRightClose size={16} />
      </button>
    </div>

    <div class="flex flex-wrap items-center gap-[10px] p-[12px_20px]">
      <input
        class="flex-1 min-w-[200px] bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
        type="search"
        placeholder="Filter keys (e.g. lr, recipe.)"
        bind:value={search}
        aria-label="Filter config keys"
      />
      <label class="inline-flex items-center gap-[5px] text-[12px] text-(--muted) cursor-pointer select-none">
        <input type="checkbox" bind:checked={onlyDifferences} />
        <span>Only differences</span>
      </label>
    </div>

    {#each runs.filter((run) => configs[run.id]?.error) as run (run.id)}
      <div class="text-(--red) text-[12px] p-[0_20px_8px]">{run.label}: {configs[run.id].error}</div>
    {/each}

    <div class="flex-1 min-h-0 overflow-auto p-[0_20px_20px]">
      {#if !loading && !rows.length}
        <div class="detail-empty">
          {#if search.trim()}No keys match “{search}”.{:else if onlyDifferences}The configs are identical.{:else}No config recorded.{/if}
        </div>
      {:else if rows.length}
        <table class="config-diff-table">
          <thead>
            <tr>
              <th>Key</th>
              {#each loaded as run (run.id)}
                <th title={run.id}>
                  <span class="inline-flex items-center gap-[6px] min-w-0">
                    <span class="inline-block w-[10px] h-[10px] rounded-[3px] shrink-0" style:background={run.color}></span>
                    <span class="truncate">{run.label}</span>
                  </span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each rows as row (row.key)}
              <tr class:config-diff-row={row.differs}>
                <td class="config-diff-key" title={row.key}>{row.key}</td>
                {#each row.values as value, index (loaded[index].id)}
                  <td
                    class:config-diff-changed={differsFromBaseline(row, index)}
                    class:config-diff-missing={value === undefined}
                    title={formatConfigValue(value)}
                  >{formatConfigValue(value)}</td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  </div>
</Drawer>
