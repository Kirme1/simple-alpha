import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      // Forward the chat socket to the Python harness (llm_harness/ws_server.py)
      '/ws': { target: 'ws://127.0.0.1:8002', ws: true },
    },
  },
})
