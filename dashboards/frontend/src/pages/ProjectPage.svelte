<script>
  // Project workspace page (/training/project/<name>): the multi-run metrics
  // overlay for every run in the project above the project's full runs table.
  // Membership is queried server-side (?project=), so unlike the grouped
  // Training runs list it sees every run in the project, not just the loaded
  // page of the global list.
  import { ArrowLeft } from "lucide-svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import RunMetricsPanels from "../components/RunMetricsPanels.svelte";
  import RunsTable from "../components/RunsTable.svelte";
  import { fetchRuns } from "../lib/api.js";

  let {
    project,
    getStatus,
    modelName,
    showFrameworkStatus,
    selectedRunIds = new Set(),
    onToggleRunSelected = () => {},
    onOpenDetail = () => {},
    onBack = () => {},
  } = $props();

  const PAGE_SIZE = 100;
  const POLL_MS = 5000;

  let runs = $state([]);
  let loading = $state(true);
  let error = $state(null);
  let search = $state("");
  let smoothing = $state(false);

  async function loadAll(signal) {
    const out = [];
    let offset = 0;
    for (;;) {
      const page = await fetchRuns({
        signal,
        limit: PAGE_SIZE,
        offset,
        facets: { project: [project] },
      });
      out.push(...page);
      if (page.length < PAGE_SIZE) break;
      offset += PAGE_SIZE;
    }
    return out;
  }

  $effect(() => {
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const page = await loadAll(controller.signal);
        if (controller.signal.aborted) return;
        runs = page;
        error = null;
      } catch (err) {
        if (controller.signal.aborted) return;
        if (!runs.length) error = err instanceof Error ? err.message : String(err);
      } finally {
        if (!controller.signal.aborted) loading = false;
      }
    };
    loading = true;
    void refresh();
    const interval = setInterval(refresh, POLL_MS);
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  });

  let runIds = $derived(runs.map((run) => run.run_id).filter(Boolean));

  function isRunning(id) {
    const run = runs.find((item) => item.run_id === id);
    return (run?.status || "").toLowerCase() === "running";
  }
</script>

<section class="flex flex-col gap-[16px] p-[0_24px_24px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="flex flex-wrap items-center gap-[12px]">
    <button type="button" class="log-button inline-flex items-center gap-[4px]" onclick={onBack}>
      <ArrowLeft size={12} strokeWidth={2.1} />
      <span>Training runs</span>
    </button>
    <h1
      class="text-(--text-bright) text-[16px] font-medium [font-family:var(--font-mono)] m-0 overflow-hidden text-ellipsis whitespace-nowrap"
      title={project}
    >{project}</h1>
    <span class="text-(--muted) text-[12px]">{runs.length} run{runs.length === 1 ? "" : "s"}</span>
    {#if runIds.length}
      <input
        class="ml-auto w-[280px] max-[640px]:w-full bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
        type="search"
        placeholder="Search panels (e.g. loss, rollout/)"
        bind:value={search}
        aria-label="Search panels by metric name"
      />
      <label class="inline-flex items-center gap-[5px] text-[12px] text-(--muted) cursor-pointer select-none" title="Trailing mean over the last 5 steps">
        <input type="checkbox" bind:checked={smoothing} />
        <span>Smoothing</span>
      </label>
    {/if}
  </div>

  {#if loading}
    <div class="table-wrap freeze-header">
      <MinimalTableSkeleton
        class="training-runs-table"
        columns={["Name", "Status", "Stage", "Type", "Model", "Dataset", "Recipe", "Group", "Tags", "Created", "Last updated", ""]}
        rows={6}
      />
    </div>
  {:else if error}
    <div class="page-empty">Failed to load: {error}</div>
  {:else if !runs.length}
    <div class="page-empty">No runs in project “{project}”.</div>
  {:else}
    <RunMetricsPanels {runIds} {search} {smoothing} {isRunning} />
    <RunsTable
      {runs}
      {selectedRunIds}
      {onToggleRunSelected}
      {onOpenDetail}
      {getStatus}
      {modelName}
      {showFrameworkStatus}
    />
  {/if}
</section>
