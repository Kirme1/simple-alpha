<script>
  import { tick } from 'svelte'
  import { chat } from '../lib/chat.svelte.js'
  import Message from './Message.svelte'

  let scroller

  // Keep the newest content in view whenever messages, tools or figures change.
  $effect(() => {
    JSON.stringify(chat.messages.map((m) => [m.text, m.tools?.length, m.figures?.length, m.pending]))
    tick().then(() => scroller?.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' }))
  })
</script>

<div class="scroller" bind:this={scroller}>
  <div class="list">
    {#each chat.messages as message (message.id)}
      <Message {message} />
    {/each}
  </div>
</div>

<style>
  .scroller {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }

  .list {
    max-width: 820px;
    margin: 0 auto;
    padding: 28px 24px 12px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  @media (max-width: 720px) {
    .list {
      padding: 20px 16px 8px;
    }
  }
</style>
