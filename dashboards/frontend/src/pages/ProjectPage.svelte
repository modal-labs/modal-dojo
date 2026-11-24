<script>
  // Project workspace page (/training/project/<name>): W&B-style layout with
  // the project's runs in a left sidebar beside the multi-run metrics overlay.
  // Membership is queried server-side (?project=), so unlike the grouped
  // Training runs list it sees every run in the project, not just the loaded
  // page of the global list.
  import { ArrowLeft, Eye, EyeOff } from "lucide-svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import RunMetricsPanels from "../components/RunMetricsPanels.svelte";
  import { fetchRuns } from "../lib/api.js";
  import { runColor } from "../lib/metricSeries.js";

  let {
    project,
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
  let hiddenIds = $state(new Set());
  let runSearch = $state("");

  function toggleHidden(id) {
    const next = new Set(hiddenIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    hiddenIds = next;
  }

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
  let sidebarRuns = $derived(
    runs
      .map((run, i) => ({ run, i }))
      .filter(
        ({ run }) =>
          !runSearch || (run.run_id || "").toLowerCase().includes(runSearch.toLowerCase()),
      ),
  );

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
    <div class="grid grid-cols-[260px_minmax(0,1fr)] gap-[16px] max-[900px]:grid-cols-1 items-start">
      <aside class="flex flex-col gap-[6px] min-w-0">
        <div class="text-(--text-bright) text-[13px] font-medium">Runs <span class="text-(--muted) font-normal">{runs.length}</span></div>
        <input
          class="w-full bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
          type="search"
          placeholder="Search runs"
          bind:value={runSearch}
          aria-label="Search runs in this project"
        />
        <div class="flex flex-col gap-[2px]">
          {#each sidebarRuns as { run, i } (run.run_id)}
            {@const hidden = hiddenIds.has(run.run_id)}
            <div
              class="flex items-center gap-[6px] rounded-[5px] p-[4px_6px] min-w-0 hover:bg-(--color-c-gray-08,#1c1c1c)"
            >
              <button
                type="button"
                class="shrink-0 inline-flex items-center bg-transparent border-0 p-0 cursor-pointer text-(--muted) hover:text-(--text)"
                aria-label={hidden ? `Show ${run.run_id} in charts` : `Hide ${run.run_id} from charts`}
                onclick={() => toggleHidden(run.run_id)}
              >
                {#if hidden}<EyeOff size={12} strokeWidth={2} />{:else}<Eye size={12} strokeWidth={2} />{/if}
              </button>
              <span
                class="shrink-0 w-[8px] h-[8px] rounded-full"
                style={`background: ${runColor(i)};${hidden ? " opacity: 0.25;" : ""}`}
              ></span>
              <button
                type="button"
                class="min-w-0 flex-1 text-left overflow-hidden text-ellipsis whitespace-nowrap text-[12px] [font-family:var(--font-mono)] text-(--text) bg-transparent border-0 p-0 cursor-pointer hover:text-(--text-bright)"
                title={run.run_id}
                onclick={() => onOpenDetail(run.run_id)}
              >{run.run_id}</button>
            </div>
          {:else}
            <div class="text-(--muted) text-[12px] p-[4px_6px]">No runs match “{runSearch}”.</div>
          {/each}
        </div>
      </aside>
      <div class="flex flex-col gap-[12px] min-w-0">
        <div class="flex items-center gap-[12px]">
          <input
            class="flex-1 min-w-0 bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
            type="search"
            placeholder="Search panels (e.g. loss, rollout/)"
            bind:value={search}
            aria-label="Search panels by metric name"
          />
          <label class="shrink-0 inline-flex items-center gap-[5px] text-[12px] text-(--muted) cursor-pointer select-none" title="Trailing mean over the last 5 steps">
            <input type="checkbox" bind:checked={smoothing} />
            <span>Smoothing</span>
          </label>
        </div>
        <RunMetricsPanels {runIds} {hiddenIds} {search} {smoothing} {isRunning} />
      </div>
    </div>
  {/if}
</section>
