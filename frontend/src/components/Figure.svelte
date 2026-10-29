<script>
  import { formatArgs } from '../lib/chat.svelte.js'

  let { figure } = $props()

  let open = $state(false)
  const src = $derived(`data:${figure.mime};base64,${figure.data}`)
  const caption = $derived(
    [figure.tool, formatArgs(figure.arguments)].filter(Boolean).join(' · ')
  )

  function onKey(e) {
    if (e.key === 'Escape') open = false
  }
</script>

<svelte:window onkeydown={open ? onKey : undefined} />

<figure>
  <figcaption class="label">{caption}</figcaption>
  <button class="frame" onclick={() => (open = true)} aria-label="Enlarge figure">
    <img {src} alt={caption} />
  </button>
</figure>

{#if open}
  <button class="overlay" onclick={() => (open = false)} aria-label="Close figure">
    <img {src} alt={caption} />
  </button>
{/if}

<style>
  figure {
    margin: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    overflow: hidden;
  }

  figcaption {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }

  .frame {
    display: block;
    width: 100%;
    padding: 0;
    border: 0;
    background: #fff;
    cursor: zoom-in;
  }

  img {
    display: block;
    width: 100%;
    height: auto;
  }

  .overlay {
    position: fixed;
    inset: 0;
    z-index: 50;
    display: grid;
    place-items: center;
    padding: 32px;
    border: 0;
    background: rgba(0, 0, 0, 0.85);
    cursor: zoom-out;
  }

  .overlay img {
    width: auto;
    max-width: 100%;
    max-height: 100%;
    border-radius: 6px;
  }
</style>
