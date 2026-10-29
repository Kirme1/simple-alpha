<script>
  import { tick } from 'svelte'
  import { chat } from '../lib/chat.svelte.js'

  let list

  $effect(() => {
    chat.activity.length
    tick().then(() => list?.scrollTo({ top: list.scrollHeight }))
  })
</script>

<aside class="activity">
  <div class="head">
    <span class="label">Activity stream</span>
    <span class="count label">{chat.activity.length}</span>
  </div>

  <ol bind:this={list}>
    {#each chat.activity as entry (entry.id)}
      <li class={entry.kind}>
        <span class="time">[{entry.time}]</span>
        <span class="text">{entry.text}</span>
      </li>
    {:else}
      <li class="none">No activity yet</li>
    {/each}
  </ol>
</aside>

<style>
  aside {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-left: 1px solid var(--border);
    background: var(--chrome-side);
  }

  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 52px;
    padding: 0 16px;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border);
  }
  .count {
    color: var(--faint);
  }

  ol {
    flex: 1;
    overflow-y: auto;
    margin: 0;
    padding: 12px;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  li {
    display: flex;
    gap: 8px;
    padding: 7px 10px;
    border: 1px solid var(--border);
    border-left: 2px solid var(--faint);
    border-radius: 4px;
    background: var(--panel-2);
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.45;
  }
  .time {
    color: var(--faint);
    flex-shrink: 0;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  li.tool {
    border-left-color: var(--accent);
  }
  li.ok {
    border-left-color: var(--green);
  }
  li.error {
    border-left-color: var(--red);
    color: var(--error-text);
  }
  li.none {
    border-left-color: var(--border);
    color: var(--faint);
  }
</style>
