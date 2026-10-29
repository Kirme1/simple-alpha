<script>
  import { onMount } from 'svelte'
  import { connect, newChat } from './lib/socket.js'
  import { chat } from './lib/chat.svelte.js'
  import Sidebar from './components/Sidebar.svelte'
  import MessageList from './components/MessageList.svelte'
  import Composer from './components/Composer.svelte'
  import EmptyState from './components/EmptyState.svelte'
  import ToolActivity from './components/ToolActivity.svelte'

  onMount(connect)
</script>

<div class="shell">
  <Sidebar />

  <main>
    <header>
      <span class="label">Analysis</span>
      <span class="sep">/</span>
      <span class="label current">Chat</span>
      <span class="spacer"></span>
      <span class="label hint">Each question is answered independently</span>
      <span class="mobile">
        <span class="dot" class:on={chat.connected} title={chat.connected ? 'Online' : 'Offline'}></span>
        <button class="label" onclick={newChat} disabled={chat.busy}>New</button>
      </span>
    </header>

    {#if chat.messages.length === 0}
      <EmptyState />
    {:else}
      <MessageList />
    {/if}

    <Composer />
  </main>

  <ToolActivity />
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr) 320px;
    height: 100%;
  }

  main {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
  }

  header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 24px;
    height: 52px;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border);
    background: var(--chrome);
  }

  .sep {
    color: var(--faint);
  }
  .current {
    color: var(--text);
  }
  .spacer {
    flex: 1;
  }
  .hint {
    color: var(--faint);
    font-size: 10px;
  }

  .mobile {
    display: none;
    align-items: center;
    gap: 12px;
  }
  .mobile button {
    padding: 4px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--panel-2);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--red);
  }
  .dot.on {
    background: var(--green);
  }

  @media (max-width: 1100px) {
    .shell {
      grid-template-columns: 200px minmax(0, 1fr);
    }
    .shell :global(.activity) {
      display: none;
    }
  }

  @media (max-width: 720px) {
    .shell {
      grid-template-columns: minmax(0, 1fr);
    }
    .shell :global(.sidebar) {
      display: none;
    }
    header {
      padding: 0 16px;
    }
    .hint {
      display: none;
    }
    .mobile {
      display: flex;
    }
  }
</style>
