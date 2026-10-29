<script>
  import { chat, setModel } from '../lib/chat.svelte.js'
  import { ask } from '../lib/socket.js'

  let text = $state('')
  let textarea

  const disabled = $derived(!chat.connected || chat.busy)

  function submit() {
    if (ask(text)) text = ''
  }

  function onKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  // Grow the textarea with its content, up to the CSS max-height.
  $effect(() => {
    text
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = `${textarea.scrollHeight}px`
  })
</script>

<div class="wrap">
  <form
    class:disabled
    onsubmit={(e) => {
      e.preventDefault()
      submit()
    }}
  >
    <textarea
      bind:this={textarea}
      bind:value={text}
      onkeydown={onKeydown}
      rows="1"
      placeholder={chat.connected ? 'Ask about a ticker, forecast or market…' : 'Waiting for harness connection…'}
    ></textarea>
    <button type="submit" disabled={disabled || !text.trim()}>
      {chat.busy ? 'Running' : 'Send'}
    </button>
  </form>
  {#if chat.models.length > 1}
    <div class="model">
      <label class="label" for="model">Model</label>
      <select
        id="model"
        value={chat.model}
        onchange={(e) => setModel(e.currentTarget.value)}
        disabled={chat.busy}
      >
        {#each chat.models as m (m.id)}
          <option value={m.id}>{m.label}</option>
        {/each}
      </select>
    </div>
  {/if}
  <div class="label foot">Enter to send · Shift+Enter for new line · Forecasts are model output, not advice</div>
</div>

<style>
  .wrap {
    flex-shrink: 0;
    max-width: 820px;
    width: 100%;
    margin: 0 auto;
    padding: 12px 24px 18px;
  }

  form {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    padding: 10px 10px 10px 14px;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: var(--panel);
    transition: border-color 0.15s;
  }
  form:focus-within {
    border-color: var(--accent);
  }

  textarea {
    flex: 1;
    min-height: 24px;
    max-height: 200px;
    padding: 4px 0;
    border: 0;
    outline: none;
    resize: none;
    background: transparent;
    line-height: 1.5;
  }
  textarea::placeholder {
    color: var(--faint);
  }

  button {
    flex-shrink: 0;
    padding: 7px 14px;
    border: 0;
    border-radius: 6px;
    background: var(--accent);
    color: var(--on-accent);
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  button:disabled {
    background: var(--panel-2);
    color: var(--faint);
    cursor: not-allowed;
  }

  .model {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }
  select {
    padding: 3px 6px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--panel-2);
    font-family: var(--mono);
    font-size: 11px;
  }
  select:disabled {
    opacity: 0.5;
  }

  .foot {
    margin-top: 8px;
    text-align: center;
    font-size: 10px;
    color: var(--faint);
  }

  @media (max-width: 720px) {
    .wrap {
      padding: 10px 16px 14px;
    }
  }
</style>
