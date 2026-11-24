<script>
  // The training runs table, shared by the Training runs page (ungrouped and
  // per-group) and a project's workspace page.
  import { ExternalLink, Maximize2 } from "lucide-svelte";
  import FrameworkStageProgress from "./FrameworkStageProgress.svelte";
  import ResizableTable from "./ResizableTable.svelte";
  import StatusPill from "./StatusPill.svelte";
  import TimeAgo from "./TimeAgo.svelte";
  import { formatTagValue, getGroupTags } from "../lib/format.js";
  import { normalizeMetricLinks } from "../lib/metricLinks.js";

  let {
    runs,
    frozenOffset = "",
    drawerRunId = null,
    selectedRunIds = new Set(),
    onToggleRunSelected = () => {},
    onOpenDetail = () => {},
    onscroll = null,
    getStatus,
    modelName,
    showFrameworkStatus,
  } = $props();

  const columns = [
    { key: "name", label: "Name", width: 268, minWidth: 168 },
    { key: "status", label: "Status", width: 116, minWidth: 96 },
    { key: "stage", label: "Stage", width: 190, minWidth: 130 },
    { key: "training_type", label: "Type", width: 116, minWidth: 88 },
    { key: "model", label: "Model", width: 210, minWidth: 120 },
    { key: "dataset", label: "Dataset", width: 180, minWidth: 120 },
    { key: "recipe", label: "Recipe", width: 116, minWidth: 88 },
    { key: "group", label: "Group", width: 280, minWidth: 220 },
    { key: "tags", label: "Tags", width: 240, minWidth: 160 },
    { key: "created", label: "Created", width: 150, minWidth: 130 },
    { key: "updated", label: "Last updated", width: 170, minWidth: 150 },
    { key: "actions", label: "", ariaLabel: "Actions", width: 236, minWidth: 180 },
  ];

  const MAX_VISIBLE_TAGS = 2;

  function tagsTitle(groupTags) {
    return (groupTags?.tags || []).map((tag) => `${tag.key}=${formatTagValue(tag.value)}`).join("\n");
  }

  function trainingRunDetailPath(runId) {
    return `/training/${encodeURIComponent(runId)}`;
  }

  function openRunInNewTab(runId) {
    if (typeof window === "undefined") return;
    const url = new URL(trainingRunDetailPath(runId), window.location.href);
    window.open(url.href, "_blank", "noopener,noreferrer");
  }

  function isPlainLeftClick(event) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey;
  }

  function selectRun(runId, event) {
    if (event?.shiftKey) {
      event.preventDefault();
      openRunInNewTab(runId);
      return;
    }

    if (event && !isPlainLeftClick(event)) return;
    event?.preventDefault();
    onOpenDetail(runId);
  }

  function frameworkProgress(run) {
    const progress = run?.framework_progress;
    if (!progress || typeof progress !== "object") return null;
    const current = Number(progress.current);
    const total = Number(progress.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return null;
    }
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: progress.unit || "step",
    };
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  function resumeBadge(run) {
    const state = run?.resume_state;
    if (!state) return "";
    const parts = [];
    if (state.attempt_count > 1) parts.push(`attempt ${state.attempt_count}`);
    if (state.resumed_from_checkpoint) {
      parts.push(
        state.resume_from_iteration != null
          ? `resumed @ ${state.resume_from_iteration}`
          : "resumed",
      );
    }
    return parts.join(" · ");
  }

  function metricLinksForRun(run) {
    return normalizeMetricLinks(
      run?.metric_links?.length ? run.metric_links : run?.wandb_links,
    );
  }
</script>

<div
  class="table-wrap freeze-header"
  style={frozenOffset ? `--frozen-table-offset: ${frozenOffset};` : ""}
  {onscroll}
>
  <ResizableTable class="training-runs-table" {columns} stickyFirstColumn>
    <tbody>
      {#each runs as run, runIndex (`${run.run_id || "run"}-${run.created_at || 0}-${runIndex}`)}
        {@const runName = run.run_id || "—"}
        {@const status = getStatus(run)}
        {@const stageLabel = run.display_stage}
        {@const progress = frameworkProgress(run)}
        {@const groupTags = getGroupTags(run)}
        <tr class="run-row" class:row-selected={drawerRunId === run.run_id}>
          <td class="min-w-0 row-open-cell">
            <input
              type="checkbox"
              class="run-select-checkbox"
              checked={selectedRunIds.has(run.run_id)}
              aria-label={`Select ${runName} to compare metrics`}
              onclick={(event) => event.stopPropagation()}
              onchange={() => onToggleRunSelected(run.run_id)}
            />
            <a
              href={trainingRunDetailPath(run.run_id)}
              class="cell-open-button pl-[40px]!"
              title={runName}
              aria-label={`Open training run ${runName}`}
              onclick={(event) => selectRun(run.run_id, event)}
            >
              <div class="block text-(--text-bright) [font-family:var(--font-mono)] [font-weight:400] text-[14px] leading-[20px] overflow-hidden text-ellipsis whitespace-nowrap">{runName}</div>
            </a>
          </td>
          <td class="row-open-cell">
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              <span class="flex flex-col items-start gap-[4px] min-w-0 max-w-full">
                <StatusPill status={status} />
                {#if resumeBadge(run)}
                  <span class="[border:1px_solid_color-mix(in_srgb,var(--yellow,#fbbf24)_42%,transparent)] rounded-[999px] bg-[color-mix(in_srgb,var(--yellow,#fbbf24)_10%,transparent)] text-(--yellow,#fbbf24) text-[11px] leading-[14px] p-[1px_6px] whitespace-nowrap">{resumeBadge(run)}</span>
                {/if}
              </span>
            </a>
          </td>
          <td class="min-w-0 row-open-cell">
            <a
              href={trainingRunDetailPath(run.run_id)}
              class="cell-open-button whitespace-normal! leading-[16px]!"
              onclick={(event) => selectRun(run.run_id, event)}
            >
              {#if showFrameworkStatus(run) && stageLabel}
                <FrameworkStageProgress
                  progress={progress}
                  progressLabel={progressLabel(progress)}
                  stageLabel={stageLabel}
                  compact
                  active={status.toLowerCase() === "pending"}
                />
              {:else}
                <span class="text-(--muted)">—</span>
              {/if}
            </a>
          </td>
          <td class="row-open-cell">
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {(run.training_type ?? "rl").toUpperCase()}
            </a>
          </td>
          <td class="min-w-0 row-open-cell" title={modelName(run)}>
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {modelName(run)}
            </a>
          </td>
          <td class="min-w-0 row-open-cell" title={run.dataset || "—"}>
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {run.dataset || "—"}
            </a>
          </td>
          <td class="row-open-cell">
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {run.recipe || "—"}
            </a>
          </td>
          <td class="group-cell row-open-cell" title={groupTags?.group_id || run.group_id || ""}>
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {#if groupTags?.group_id || run.group_id}
                <span class="inline-block max-w-full overflow-hidden text-ellipsis whitespace-nowrap align-bottom p-[2px_8px] rounded-[999px] text-[0.72rem] [font-variant-numeric:tabular-nums] text-(--muted) [border:1px_solid_var(--border,#2f2f2f)] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)]">{groupTags?.group_id || run.group_id}</span>
              {:else}
                <span class="group-empty">—</span>
              {/if}
            </a>
          </td>
          <td class="min-w-0 row-open-cell" title={tagsTitle(groupTags)}>
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              {#if groupTags?.tags.length}
                <span class="flex items-center gap-[4px] min-w-0 max-w-full overflow-hidden">
                  {#each groupTags.tags.slice(0, MAX_VISIBLE_TAGS) as tag (tag.key)}
                    <span class="block shrink-0 max-w-[80px] overflow-hidden text-ellipsis whitespace-nowrap p-[2px_7px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[999px] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)] text-(--text) [font-family:var(--font-mono)] text-[11px] leading-[14px]">
                      <span class="text-(--muted)">{tag.key.split(".").at(-1)}=</span>{formatTagValue(tag.value)}
                    </span>
                  {/each}
                  {#if groupTags.tags.length > MAX_VISIBLE_TAGS}
                    <span class="shrink-0 p-[2px_7px] rounded-[999px] text-(--muted) [font-family:var(--font-mono)] text-[11px] leading-[14px] [border:1px_dashed_var(--border,#2f2f2f)]">+{groupTags.tags.length - MAX_VISIBLE_TAGS}</span>
                  {/if}
                </span>
              {:else}
                <span class="group-empty">—</span>
              {/if}
            </a>
          </td>
          <td class="whitespace-nowrap row-open-cell">
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              <TimeAgo timestamp={run.created_at} showJustNow falsyRepresentation="—" />
            </a>
          </td>
          <td class="whitespace-nowrap row-open-cell">
            <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
              <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
            </a>
          </td>
          <td class="min-w-0 overflow-visible">
            <div class="flex items-center flex-wrap gap-[6px]">
              <button
                class="inline-flex items-center gap-[6px] whitespace-nowrap [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
                title="Open expanded view"
                aria-label={`Open expanded view for training run ${run.run_id}`}
                onclick={(event) => {
                  event.stopPropagation();
                  selectRun(run.run_id, event);
                }}
              >
                <Maximize2 size={12} strokeWidth={2.1} />
                <span class="expand-button-label">Expand</span>
              </button>
              {#if run.modal_app_url}
                <a
                  class="training-open-modal-link"
                  href={run.modal_app_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onclick={(event) => event.stopPropagation()}
                >
                  <span class="open-modal-link-label">Open in Modal</span>
                  <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                </a>
              {:else}
                <span class="training-open-modal-link open-modal-link-disabled">
                  <span class="open-modal-link-label">Open in Modal</span>
                  <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                </span>
              {/if}
              {#each metricLinksForRun(run) as link (link.url)}
                <a
                  class="training-open-modal-link training-open-metric-link"
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onclick={(event) => event.stopPropagation()}
                >
                  <span class="open-modal-link-label">{link.label}</span>
                  <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                </a>
              {/each}
            </div>
          </td>
        </tr>
      {/each}
    </tbody>
  </ResizableTable>
</div>
