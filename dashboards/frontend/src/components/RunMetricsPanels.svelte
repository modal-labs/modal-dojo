<script>
  // Multi-run metric panels shared by the Metrics compare page and the
  // project workspace page: fetch each run's scalars, union the metric keys
  // into prefix-grouped panels, one colored line per run.
  import ChartSkeleton from "./ChartSkeleton.svelte";
  import CompareLineChart from "./CompareLineChart.svelte";
  import GroupSection from "./GroupSection.svelte";
  import { fetchRunMetrics } from "../lib/api.js";
  import { formatMetricValue, groupMetricKeys, metricAxisLabel, runColor } from "../lib/metricSeries.js";
  import { toggleInSet } from "../lib/set.js";

  let {
    runIds = [],
    // Runs left out of the panels (hidden chips on the Metrics page).
    hiddenIds = new Set(),
    // Metric-name substring filter.
    search = "",
    smoothing = false,
    isRunning = () => false,
    // runId → { payload, error }; bindable so callers can flag per-run errors.
    results = $bindable({}),
  } = $props();

  const POLL_MS = 5000;

  let collapsed = $state(new Set());
  let stepDomains = $state({});

  let runKey = $derived(runIds.join("\n"));
  let anyLive = $derived(
    runIds.some((id) => isRunning(id) || results[id]?.payload?.stale),
  );

  async function loadOne(id, signal) {
    try {
      const payload = await fetchRunMetrics(id, { signal });
      if (signal.aborted) return;
      results = { ...results, [id]: { payload, error: "" } };
    } catch (err) {
      if (signal.aborted) return;
      const error = err instanceof Error ? err.message : String(err);
      results = { ...results, [id]: { payload: results[id]?.payload ?? null, error } };
    }
  }

  $effect(() => {
    const ids = runKey ? runKey.split("\n") : [];
    const poll = anyLive;
    if (!ids.length) return;
    const controller = new AbortController();
    const loadAll = (only = ids) => Promise.all(only.map((id) => loadOne(id, controller.signal)));
    loadAll();
    const interval = poll
      ? setInterval(
          () => loadAll(ids.filter((id) => isRunning(id) || results[id]?.payload?.stale)),
          POLL_MS,
        )
      : null;
    return () => {
      controller.abort();
      if (interval) clearInterval(interval);
    };
  });

  let runMeta = $derived(
    runIds.map((id, index) => ({ id, label: id, color: runColor(index) })),
  );
  let visibleRuns = $derived(runMeta.filter((run) => !hiddenIds.has(run.id)));
  let loadingCount = $derived(runIds.filter((id) => !results[id]).length);

  // Union of keys across visible runs; the first run that names a key's
  // step axis decides it (runs of one framework agree).
  let union = $derived.by(() => {
    const stepKeys = {};
    const keys = new Set();
    for (const run of visibleRuns) {
      const payload = results[run.id]?.payload;
      if (!payload) continue;
      for (const key of Object.keys(payload.series ?? {})) {
        keys.add(key);
        if (!(key in stepKeys) && payload.step_keys?.[key]) stepKeys[key] = payload.step_keys[key];
      }
    }
    return { keys: [...keys], stepKeys };
  });
  let groups = $derived(groupMetricKeys(union.keys, search));

  function seriesFor(key) {
    const out = [];
    for (const run of visibleRuns) {
      const raw = results[run.id]?.payload?.series?.[key];
      if (!raw?.length) continue;
      const points = raw
        .map(([x, y]) => ({ x: Number(x), y: Number(y) }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x);
      out.push({ ...run, points });
    }
    return out;
  }

  function onDomainChange(axis, domain) {
    stepDomains = {
      ...stepDomains,
      [axis]: domain && domain[1] > domain[0] ? [Math.floor(domain[0]), Math.ceil(domain[1])] : null,
    };
  }
</script>

{#if !runIds.length}
  <div class="detail-empty">No runs selected.</div>
{:else if loadingCount === runIds.length}
  <div class="metric-grid" aria-busy="true" aria-label="Loading metrics">
    {#each [0, 1, 2, 3, 4, 5] as i (i)}
      <div class="metric-panel"><ChartSkeleton variant="line" height={160} showTitle /></div>
    {/each}
  </div>
{:else}
  <div class="flex items-center gap-[10px]">
    {#if loadingCount}
      <span class="text-(--muted) text-[12px]" role="status">Loading {loadingCount} more…</span>
    {/if}
    {#if Object.values(stepDomains).some(Boolean)}
      <button class="log-button" type="button" onclick={() => (stepDomains = {})}>Reset zoom</button>
    {/if}
  </div>
  {#if !union.keys.length}
    <div class="detail-empty">No metrics reported yet for the {visibleRuns.length === runIds.length ? "" : "visible "}selected runs.</div>
  {:else if !groups.length}
    <div class="detail-empty">No metrics match “{search}”.</div>
  {/if}

  {#each groups as group (group.name)}
    <GroupSection
      title={group.name}
      subtitle={`${group.keys.length} ${group.keys.length === 1 ? "panel" : "panels"}`}
      expanded={!collapsed.has(group.name)}
      onToggle={() => (collapsed = toggleInSet(collapsed, group.name))}
    >
      <div class="metric-grid p-[0_12px_12px]">
        {#each group.keys as key (key)}
          {@const axis = union.stepKeys[key] ?? ""}
          <div class="metric-panel">
            <CompareLineChart
              title={key}
              series={seriesFor(key)}
              smoothingWindow={smoothing ? 5 : 1}
              xLabel={metricAxisLabel(union.stepKeys[key])}
              formatY={formatMetricValue}
              xDomain={stepDomains[axis] ?? null}
              onChangeDomainX={(domain) => onDomainChange(axis, domain)}
            />
          </div>
        {/each}
      </div>
    </GroupSection>
  {/each}
{/if}
