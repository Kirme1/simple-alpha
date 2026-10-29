<script>
  import { chat } from '../lib/chat.svelte.js'
  import { newChat } from '../lib/socket.js'
  import { theme, toggleTheme } from '../lib/theme.svelte.js'
</script>

<aside class="sidebar">
  <div class="brand">
    <img src="/favicon.svg" alt="" width="26" height="26" />
    <span>Simple Alpha</span>
  </div>

  <button class="new" onclick={newChat} disabled={chat.busy}>
    <span>+</span> New chat
  </button>

  <nav>
    <span class="label section">General</span>
    <span class="item active">Chat</span>
  </nav>

  <button class="theme" onclick={toggleTheme}>
    {theme.mode === 'light' ? '☾ Dark theme' : '☀ Light theme'}
  </button>

  <div class="status">
    <span class="dot" class:on={chat.connected}></span>
    <span class="label">{chat.connected ? 'Harness online' : 'Connecting…'}</span>
  </div>
</aside>

<style>
  aside {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 16px 12px;
    border-right: 1px solid var(--border);
    background: var(--chrome-side);
    min-height: 0;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 2px 6px;
    font-weight: 600;
  }

  .new {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--panel-2);
    font-family: var(--mono);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .new:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }
  .new:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  nav {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .section {
    padding: 0 8px 6px;
    color: var(--faint);
  }
  .item {
    padding: 7px 10px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--muted);
  }
  .item.active {
    background: var(--panel-2);
    color: var(--text);
  }

  .theme {
    margin-top: auto;
    padding: 7px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: transparent;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    text-align: left;
  }
  .theme:hover {
    border-color: var(--border-strong);
    color: var(--text);
  }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 8px;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--red);
    box-shadow: 0 0 0 3px var(--red-dim);
  }
  .dot.on {
    background: var(--green);
    box-shadow: 0 0 0 3px var(--green-dim);
  }
</style>
