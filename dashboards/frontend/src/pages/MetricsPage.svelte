<script>
  // Metrics compare page: pick several training runs and overlay their
  // wandb/trackio scalars W&B-style. Panels are the union of every selected
  // run's metric keys; each panel draws one line per run that logged it.
  import { Eye, EyeOff, FileDiff, Plus, X } from "lucide-svelte";
  import AddRunsModal from "../components/AddRunsModal.svelte";
  import ConfigDiffDrawer from "../components/ConfigDiffDrawer.svelte";
  import RunMetricsPanels from "../components/RunMetricsPanels.svelte";
  import { runColor } from "../lib/metricSeries.js";
  import { toggleInSet } from "../lib/set.js";

  let {
    runIds = [],
    // The training list's rows and filters, shared with the Add-run modal.
    runs = [],
    filterProps = {},
    search: runSearch = $bindable(""),
    matchingRunCount = 0,
    hasMoreRuns = false,
    runsLoading = false,
    getStatus,
    onLoadMore = () => {},
    isRunning = () => false,
    onChangeRunIds = () => {},
  } = $props();

  let results = $state({});
  let search = $state("");
  let configDiffOpen = $state(false);
  let hiddenRuns = $state(new Set());
  let smoothing = $state(false);
  let addOpen = $state(false);

  let runMeta = $derived(
    runIds.map((id, index) => ({ id, label: id, color: runColor(index) })),
  );

  function removeRun(id) {
    hiddenRuns = new Set([...hiddenRuns].filter((runId) => runId !== id));
    onChangeRunIds(runIds.filter((runId) => runId !== id));
  }

  function addRuns(ids) {
    const fresh = ids.filter((id) => !runIds.includes(id));
    if (fresh.length) onChangeRunIds([...runIds, ...fresh]);
  }
</script>

<section class="flex flex-col gap-[16px] p-[0_24px_24px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="compare-runs-bar flex flex-wrap items-center gap-[8px]">
    {#each runMeta as run (run.id)}
      <span class="run-chip" class:run-chip-hidden={hiddenRuns.has(run.id)}>
        <span class="inline-block w-[10px] h-[10px] rounded-[3px] shrink-0" style:background={run.color}></span>
        <a
          class="truncate text-inherit no-underline hover:text-(--text-bright) hover:underline"
          href={`/training/${encodeURIComponent(run.id)}`}
          title={run.id}
          target="_blank"
          rel="noopener">{run.label}</a
        >
        {#if results[run.id]?.error}
          <span class="text-(--red) text-[11px]" title={results[run.id].error}>error</span>
        {/if}
        <button
          type="button"
          title={hiddenRuns.has(run.id) ? "Show run" : "Hide run"}
          aria-label={`${hiddenRuns.has(run.id) ? "Show" : "Hide"} ${run.id}`}
          onclick={() => (hiddenRuns = toggleInSet(hiddenRuns, run.id))}
        >
          {#if hiddenRuns.has(run.id)}<EyeOff size={12} />{:else}<Eye size={12} />{/if}
        </button>
        <button type="button" title="Remove run" aria-label={`Remove ${run.id}`} onclick={() => removeRun(run.id)}>
          <X size={12} />
        </button>
      </span>
    {/each}

    <button type="button" class="log-button inline-flex items-center gap-[4px]" onclick={() => (addOpen = true)}>
      <Plus size={12} strokeWidth={2.1} />
      <span>Add runs</span>
    </button>

    <button
      type="button"
      class="log-button inline-flex items-center gap-[4px] disabled:opacity-50 disabled:cursor-default"
      disabled={runIds.length < 2}
      title={runIds.length < 2 ? "Add another run to compare configs" : "Show where the runs' configs differ"}
      onclick={() => (configDiffOpen = true)}
    >
      <FileDiff size={12} strokeWidth={2.1} />
      <span>Diff config</span>
    </button>

    {#if runIds.length}
      <input
        class="ml-auto w-[280px] max-[640px]:w-full bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
        type="search"
        placeholder="Search panels (e.g. loss, rollout/)"
        bind:value={search}
        aria-label="Search panels by metric name"
      />
    {/if}
  </div>

  <ConfigDiffDrawer open={configDiffOpen} runs={runMeta} onclose={() => (configDiffOpen = false)} />

  <AddRunsModal
    open={addOpen}
    {filterProps}
    bind:search={runSearch}
    {runs}
    {matchingRunCount}
    {hasMoreRuns}
    loading={runsLoading}
    {getStatus}
    existingRunIds={runIds}
    {onLoadMore}
    onAdd={addRuns}
    onclose={() => (addOpen = false)}
  />

  {#if !runIds.length}
    <div class="detail-empty">
      <div class="text-(--text-bright) mb-[6px]">No runs selected.</div>
      <div class="max-w-[64ch] leading-[1.5]">
        Use <strong>Add runs</strong>, or tick runs on the Training runs page and press <strong>Compare metrics</strong>.
      </div>
    </div>
  {:else}
    <div class="flex flex-wrap items-center gap-[10px]">
      <!-- Same trailing window LineChart's Smoothing checkbox uses by default. -->
      <label class="inline-flex items-center gap-[5px] text-[12px] text-(--muted) cursor-pointer select-none" title="Trailing mean over the last 5 steps">
        <input type="checkbox" bind:checked={smoothing} />
        <span>Smoothing</span>
      </label>
    </div>

    <RunMetricsPanels {runIds} hiddenIds={hiddenRuns} {search} {smoothing} {isRunning} bind:results />
  {/if}
</section>
