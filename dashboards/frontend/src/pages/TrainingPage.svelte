<script>
  import { ChartLine, ExternalLink, Maximize2, PanelRightClose, X } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBar from "../components/FilterBar.svelte";
  import GroupSection from "../components/GroupSection.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import RunsTable from "../components/RunsTable.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { fetchRun } from "../lib/api.js";
  import { toggleInSet } from "../lib/set.js";

  let {
    totalRuns,
    matchingRunCount,
    hasMoreRuns,
    onLoadMore = () => {},
    completedTotal,
    runningTotal,
    stoppedTotal,
    failedTotal,
    recipes,
    recipeCounts,
    activeRecipes,

    statuses,
    statusCounts,
    activeStatuses,
    groups,
    groupCounts,
    activeGroups,
    projects,
    projectCounts,
    activeProjects,
    tagCounts,
    activeTags,
    trainingTypes,
    trainingTypeCounts,
    activeTrainingTypes,
    filteredRuns,
    runGroups,
    groupBy = $bindable(),
    loading,
    error,
    modelName,
    getStatus,
    showFrameworkStatus,
    fmtDuration,
    search = $bindable(),
    drawerRunId = null,
    onOpenDetail = () => {},
    onCloseDrawer = () => {},
    onToggleRecipe,
    onSelectAllRecipes,
    onClearRecipes,
    onToggleStatus,
    onSelectAllStatuses,
    onClearStatuses,
    onToggleGroup,
    onSelectAllGroups,
    onClearGroups,
    onToggleProject,
    onSelectAllProjects,
    onClearProjects,
    onAddTagValue,
    onRemoveTagValue,
    onClearTags,
    onToggleTrainingType,
    onSelectAllTrainingTypes,
    onClearTrainingTypes,
    // Runs ticked for "Compare metrics" (a Set of run ids, owned by the parent).
    selectedRunIds = new Set(),
    onToggleRunSelected = () => {},
    onClearRunSelection = () => {},
    onCompareMetrics = () => {},
    onOpenProject = () => {},
  } = $props();

  // The drawer is now driven by the parent: it holds the run-summary while the
  // full rollouts/logs detail lives on its own page. Clicking a run (or the
  // drawer's Expand button) navigates to that page; the page's Collapse button
  // brings the summary back as this drawer.
  // The list payload carries only the fields the table renders, so the drawer
  // fetches the run's full record (config and all) when it opens.
  let drawerDetail = $state(null);
  // Set only when the server says the run is gone. The list holds one page, so
  // a run being absent from it means nothing — the drawer renders from the
  // detail fetch for runs deep-linked or scrolled past.
  let drawerRunMissing = $state(false);

  $effect(() => {
    const runId = drawerRunId;
    drawerDetail = null;
    drawerRunMissing = false;
    if (!runId) return;
    const controller = new AbortController();
    fetchRun(runId, { signal: controller.signal })
      .then((detail) => {
        if (detail) drawerDetail = detail;
        else drawerRunMissing = true;
      })
      .catch(() => {});
    return () => controller.abort();
  });

  let selectedRun = $derived.by(() => {
    const listRun = filteredRuns.find((run) => run.run_id === drawerRunId) || null;
    const detail = drawerDetail?.run_id === drawerRunId ? drawerDetail : null;
    if (!listRun) return detail;
    if (!detail) return listRun;
    return { ...listRun, ...detail };
  });

  const drawerWidth = "min(420px, calc(100vw - 24px))";

  function projectPath(project) {
    return `/training/project/${encodeURIComponent(project)}`;
  }

  function closeDrawer() {
    onCloseDrawer();
  }

  $effect(() => {
    if (drawerRunId && drawerRunMissing) {
      onCloseDrawer();
    }
  });

  // Inverted vs. evals' expanded set so groups default to expanded, including
  // ones that appear later via auto-refresh.
  let collapsedGroupKeys = $state(new Set());

  function toggleGroupSection(key) {
    collapsedGroupKeys = toggleInSet(collapsedGroupKeys, key);
  }

  $effect(() => {
    void groupBy;
    collapsedGroupKeys = new Set();
  });

  let list = $state(null);

  // A new query restarts paging at the first page. Staying scrolled to the
  // bottom of the old, longer list would immediately ask for page after page
  // again, so send the reader back to the top of the new results.
  $effect(() => {
    void search;
    const root = list;
    if (!root) return;
    for (const wrap of root.querySelectorAll(".table-wrap")) wrap.scrollTop = 0;
  });

  // A run row is ~50 DOM nodes, so a full history is seconds of layout on a
  // phone and a re-render of every row on each refresh. The server hands out
  // one page at a time and scrolling a table near its end asks for the next
  // one. Rows scroll inside `.table-wrap`, and a scroll event is the only
  // signal that means "the reader reached the end of *this* scrollport": a
  // sentinel + IntersectionObserver either never re-fires once it is already
  // visible (a short group's table doesn't scroll at all) or pages the whole
  // history unprompted, which is what this PR exists to stop. Grouped tables
  // may not scroll at all, so the footer's button is what keeps every run
  // reachable; scrolling is only the shortcut.
  function growOnScroll(event) {
    if (!hasMoreRuns) return;
    const el = event.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 600) {
      onLoadMore();
    }
  }
</script>

<section class="summary-sticky grid grid-cols-5 gap-[14px] p-[0_24px] mb-[24px] max-[900px]:grid-cols-2">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{totalRuns}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{completedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending runs</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Stopped runs</span>
    <strong>{stoppedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Failed runs</span>
    <strong>{failedTotal}</strong>
  </article>
</section>

<section class="[border:0] [background:transparent] flex flex-col gap-[24px] p-[0_24px_16px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="m-0">
    <FilterBar
      {recipes}
      {recipeCounts}
      {activeRecipes}
      allRecipesActive={activeRecipes.size === recipes.length}
      {statuses}
      {statusCounts}
      {activeStatuses}
      allStatusesActive={activeStatuses.size === statuses.length}
      {groups}
      {groupCounts}
      {activeGroups}
      allGroupsActive={activeGroups.size === groups.length}
      {projects}
      {projectCounts}
      {activeProjects}
      allProjectsActive={activeProjects.size === projects.length}
      {tagCounts}
      {activeTags}
      {trainingTypes}
      {trainingTypeCounts}
      {activeTrainingTypes}
      allTrainingTypesActive={activeTrainingTypes.size === trainingTypes.length}
      bind:search
      bind:groupBy
      onToggleRecipe={onToggleRecipe}
      onSelectAllRecipes={onSelectAllRecipes}
      onClearRecipes={onClearRecipes}
      onToggleStatus={onToggleStatus}
      onSelectAllStatuses={onSelectAllStatuses}
      onClearStatuses={onClearStatuses}
      onToggleGroup={onToggleGroup}
      onSelectAllGroups={onSelectAllGroups}
      onClearGroups={onClearGroups}
      onToggleProject={onToggleProject}
      onSelectAllProjects={onSelectAllProjects}
      onClearProjects={onClearProjects}
      onAddTagValue={onAddTagValue}
      onRemoveTagValue={onRemoveTagValue}
      onClearTags={onClearTags}
      onToggleTrainingType={onToggleTrainingType}
      onSelectAllTrainingTypes={onSelectAllTrainingTypes}
      onClearTrainingTypes={onClearTrainingTypes}
    />
  </div>

  {#if selectedRunIds.size}
    <div class="compare-bar" role="toolbar" aria-label="Selected runs">
      <span class="text-(--text-bright) text-[13px] font-medium [font-variant-numeric:tabular-nums]">
        {selectedRunIds.size} run{selectedRunIds.size === 1 ? "" : "s"} selected
      </span>
      <button type="button" class="compare-bar-primary" onclick={onCompareMetrics}>
        <ChartLine size={14} strokeWidth={2.1} />
        <span>Compare metrics</span>
      </button>
      <button type="button" class="log-button inline-flex items-center gap-[4px]" onclick={onClearRunSelection}>
        <X size={12} strokeWidth={2.1} />
        <span>Clear</span>
      </button>
    </div>
  {/if}

  <div class="p-0" bind:this={list}>
    {#if loading}
      <div class="table-wrap freeze-header">
        <MinimalTableSkeleton
          class="training-runs-table"
          columns={["Name", "Status", "Stage", "Type", "Model", "Dataset", "Recipe", "Group", "Tags", "Created", "Last updated", ""]}
          rows={8}
        />
      </div>
    {:else if error}
      <div class="page-empty">Failed to load: {error}</div>
    {:else if !totalRuns}
      <div class="page-empty">No training runs found yet.</div>
    {:else if !filteredRuns.length}
      <div class="page-empty">No runs match the current filters.</div>
    {:else}
      {#if groupBy === "none"}
        <RunsTable
          runs={filteredRuns}
          {drawerRunId}
          {selectedRunIds}
          {onToggleRunSelected}
          {onOpenDetail}
          {getStatus}
          {modelName}
          {showFrameworkStatus}
          onscroll={growOnScroll}
        />
      {:else}
        <div class="flex flex-col gap-[24px] p-0">
          {#each runGroups as group (group.key)}
            {@const isProject = groupBy === "project"}
            <GroupSection
              title={group.key}
              subtitle={`${group.runs.length} run${group.runs.length === 1 ? "" : "s"}`}
              expanded={!collapsedGroupKeys.has(group.key)}
              onToggle={() => toggleGroupSection(group.key)}
              href={isProject ? projectPath(group.key) : null}
              onNavigate={isProject ? () => onOpenProject(group.key) : null}
            >
              {#snippet meta()}
                {#if group.latestCreatedAt}
                  <span class="group-meta-pill [font-variant-numeric:tabular-nums]">
                    <TimeAgo timestamp={group.latestCreatedAt} showJustNow falsyRepresentation="—" />
                  </span>
                {/if}
              {/snippet}
              <RunsTable
                runs={group.runs}
                frozenOffset="360px"
                {drawerRunId}
                {selectedRunIds}
                {onToggleRunSelected}
                {onOpenDetail}
                {getStatus}
                {modelName}
                {showFrameworkStatus}
                onscroll={growOnScroll}
              />
            </GroupSection>
          {/each}
        </div>
      {/if}

      <div class="page-empty flex flex-col items-center gap-[10px]">
        <span>Showing {filteredRuns.length} of {matchingRunCount} runs</span>
        {#if hasMoreRuns}
          <button type="button" class="load-more-button" onclick={onLoadMore}>
            Load more
          </button>
        {/if}
      </div>
    {/if}
  </div>
</section>

{#if selectedRun}
  <Drawer open={!!selectedRun} onclose={closeDrawer} width={drawerWidth}>
    <div
      class="w-full h-full max-h-[100vh]"
      aria-label={`Training run ${selectedRun.run_id}`}
    >
      <div class="drawer-panel-header">
        <div class="min-w-0 overflow-hidden">
          <div class="drawer-panel-eyebrow">Training run</div>
          <h2 class="text-(--text-bright) text-[16px] font-medium [font-family:var(--font-mono)] leading-[24px] whitespace-nowrap overflow-hidden text-ellipsis" title={selectedRun.run_id}>{selectedRun.run_id}</h2>
        </div>
        <div class="flex items-center gap-[8px] shrink-0">
          <button
            class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
            onclick={(event) => {
              if (!event.shiftKey) return onOpenDetail(selectedRun.run_id);
              const url = new URL(`/training/${encodeURIComponent(selectedRun.run_id)}`, window.location.href);
              window.open(url.href, "_blank", "noopener,noreferrer");
            }}
            title="Expand to full view"
          >
            <Maximize2 size={12} />
            <span>Expand</span>
          </button>
          {#if selectedRun.modal_app_url}
            <a
              class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] no-underline text-(--muted) text-[12px] font-medium leading-[16px] ghost-hover"
              href={selectedRun.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>View in app</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-panel-close ghost-hover" onclick={closeDrawer} aria-label="Close run drawer">
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <div class="p-[4px_20px_16px]">
        <RunSummary
          run={selectedRun}
          {getStatus}
          {showFrameworkStatus}
          {modelName}
          {fmtDuration}
        />
      </div>
    </div>
  </Drawer>
{/if}
