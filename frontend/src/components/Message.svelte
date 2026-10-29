<script>
  import { marked } from 'marked'
  import DOMPurify from 'dompurify'
  import { formatArgs } from '../lib/chat.svelte.js'
  import Figure from './Figure.svelte'

  let { message } = $props()

  const html = $derived(message.text ? DOMPurify.sanitize(marked.parse(message.text)) : '')
</script>

{#if message.role === 'user'}
  <div class="user">
    <div class="bubble">{message.text}</div>
  </div>
{:else}
  <article class="assistant">
    <div class="label who">Analyst</div>

    {#if message.tools.length}
      <div class="tools">
        {#each message.tools as tool (tool.id)}
          <span class="chip {tool.status}" title={JSON.stringify(tool.arguments)}>
            <span class="indicator"></span>
            {tool.name}{#if formatArgs(tool.arguments)}<span class="args">({formatArgs(tool.arguments)})</span>{/if}
          </span>
        {/each}
      </div>
    {/if}

    {#if message.pending && !message.text}
      <div class="thinking">
        <span></span><span></span><span></span>
        <span class="label">Working</span>
      </div>
    {/if}

    {#if html}
      <div class="markdown">{@html html}</div>
    {/if}

    {#if message.figures.length}
      <div class="figures">
        {#each message.figures as figure, i (i)}
          {#if figure.type === 'chart'}
            <!-- loaded on first use: the charting libraries are most of the bundle -->
            {#await import('./ChartCard.svelte') then { default: ChartCard }}
              <ChartCard item={figure} />
            {/await}
          {:else}
            <Figure {figure} />
          {/if}
        {/each}
      </div>
    {/if}

    {#if message.error}
      <div class="error"><span class="label">Error</span> {message.error}</div>
    {/if}
  </article>
{/if}

<style>
  .user {
    display: flex;
    justify-content: flex-end;
  }
  .bubble {
    max-width: 80%;
    padding: 10px 14px;
    border: 1px solid var(--border-strong);
    border-radius: 10px 10px 2px 10px;
    background: var(--panel-2);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  .assistant {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .who {
    color: var(--accent);
  }

  .tools {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 3px 9px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--panel);
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--muted);
  }
  .args {
    color: var(--faint);
  }
  .indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--yellow);
  }
  .chip.running .indicator {
    animation: pulse 1s ease-in-out infinite;
  }
  .chip.done .indicator {
    background: var(--green);
  }
  .chip.error {
    border-color: var(--red-dim);
  }
  .chip.error .indicator {
    background: var(--red);
  }

  .thinking {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .thinking > span:not(.label) {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  .thinking > span:nth-child(2) {
    animation-delay: 0.2s;
  }
  .thinking > span:nth-child(3) {
    animation-delay: 0.4s;
  }
  .thinking .label {
    margin-left: 6px;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 0.25;
    }
    50% {
      opacity: 1;
    }
  }

  .figures {
    display: grid;
    gap: 12px;
  }

  .error {
    padding: 10px 14px;
    border: 1px solid var(--red-dim);
    border-radius: 6px;
    background: var(--red-dim);
    color: var(--error-text);
    font-size: 14px;
  }
  .error .label {
    color: var(--red);
    margin-right: 6px;
  }

  /* Rendered markdown from the LLM */
  .markdown {
    overflow-wrap: anywhere;
  }
  .markdown :global(h1),
  .markdown :global(h2),
  .markdown :global(h3) {
    margin: 1.2em 0 0.4em;
    font-weight: 600;
    line-height: 1.3;
  }
  .markdown :global(h1) {
    font-size: 1.35em;
  }
  .markdown :global(h2) {
    font-size: 1.15em;
  }
  .markdown :global(h3) {
    font-size: 1em;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
  }
  .markdown :global(:first-child) {
    margin-top: 0;
  }
  .markdown :global(p),
  .markdown :global(ul),
  .markdown :global(ol) {
    margin: 0 0 0.8em;
  }
  .markdown :global(ul),
  .markdown :global(ol) {
    padding-left: 1.3em;
  }
  .markdown :global(li) {
    margin: 0.2em 0;
  }
  .markdown :global(li::marker) {
    color: var(--accent);
  }
  .markdown :global(strong) {
    color: var(--strong);
  }
  .markdown :global(a) {
    color: var(--accent);
  }
  .markdown :global(code) {
    font-family: var(--mono);
    font-size: 0.88em;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--panel-2);
    border: 1px solid var(--border);
  }
  .markdown :global(pre) {
    padding: 12px 14px;
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel);
  }
  .markdown :global(pre code) {
    padding: 0;
    border: 0;
    background: none;
  }
  .markdown :global(blockquote) {
    margin: 0 0 0.8em;
    padding-left: 12px;
    border-left: 2px solid var(--accent);
    color: var(--muted);
  }
  .markdown :global(hr) {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 1.2em 0;
  }
  .markdown :global(table) {
    display: block;
    overflow-x: auto;
    border-collapse: collapse;
    margin: 0 0 1em;
    font-size: 13.5px;
  }
  .markdown :global(th) {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: left;
    background: var(--panel);
  }
  .markdown :global(th),
  .markdown :global(td) {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  .markdown :global(tbody tr:nth-child(odd)) {
    background: var(--stripe);
  }
</style>
