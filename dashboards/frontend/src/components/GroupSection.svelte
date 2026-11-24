<script>
  import { ChevronDown } from "lucide-svelte";

  let { title, subtitle, expanded, onToggle, meta, children, href = null, onNavigate = null } = $props();

  // SPA navigation on plain left-click; modified clicks keep the browser's
  // default anchor behavior (new tab etc.).
  function handleNavigate(event) {
    if (!onNavigate) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    event.preventDefault();
    onNavigate();
  }
</script>

<section class="group-section">
  {#if href}
    <div class="group-section-header">
      <a class="min-w-0 no-underline text-inherit" {href} onclick={handleNavigate}>
        <div class="text-(--text-bright) text-[0.86rem] [font-weight:600] overflow-hidden text-ellipsis whitespace-nowrap hover:underline">{title}</div>
        <div class="text-(--muted) text-[0.74rem] overflow-hidden text-ellipsis whitespace-nowrap">{subtitle}</div>
      </a>
      <button
        type="button"
        class="flex items-center gap-[0.35rem] [flex-wrap:wrap] [justify-content:flex-end] [border:0] [background:transparent] [color:inherit] cursor-pointer p-0 [font:inherit]"
        onclick={onToggle}
        aria-label={expanded ? `Collapse ${title}` : `Expand ${title}`}
      >
        {@render meta?.()}
        <ChevronDown
          size={14}
          style={`color: var(--muted); transform: ${expanded ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </button>
    </div>
  {:else}
    <button class="group-section-header" onclick={onToggle}>
      <div class="min-w-0">
        <div class="text-(--text-bright) text-[0.86rem] [font-weight:600] overflow-hidden text-ellipsis whitespace-nowrap">{title}</div>
        <div class="text-(--muted) text-[0.74rem] overflow-hidden text-ellipsis whitespace-nowrap">{subtitle}</div>
      </div>
      <div class="flex items-center gap-[0.35rem] [flex-wrap:wrap] [justify-content:flex-end]">
        {@render meta?.()}
        <ChevronDown
          size={14}
          style={`color: var(--muted); transform: ${expanded ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </div>
    </button>
  {/if}

  {#if expanded}
    {@render children?.()}
  {/if}
</section>
