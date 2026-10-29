import { addQuery, chat, clearChat, handleEvent, setConnected } from './chat.svelte.js'

// Same-origin /ws is proxied to the harness by Vite; override with VITE_WS_URL if needed.
const WS_URL =
  import.meta.env.VITE_WS_URL ??
  `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`

let socket = null
let retryDelay = 1000

export function connect() {
  socket = new WebSocket(WS_URL)

  socket.addEventListener('open', () => {
    retryDelay = 1000
    setConnected(true)
  })

  socket.addEventListener('message', (e) => {
    try {
      handleEvent(JSON.parse(e.data))
    } catch (err) {
      console.error('Bad message from harness', err)
    }
  })

  socket.addEventListener('close', () => {
    setConnected(false)
    setTimeout(connect, retryDelay)
    retryDelay = Math.min(retryDelay * 2, 10000)
  })
}

// Adds the question to the chat and sends it. Returns false if it could not be sent.
export function ask(text) {
  text = text.trim()
  if (!text || chat.busy || socket?.readyState !== WebSocket.OPEN) return false
  addQuery(text)
  socket.send(JSON.stringify({ type: 'query', text, model: chat.model }))
  return true
}

// Clears the chat and tells the harness to forget the conversation too.
export function newChat() {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'reset' }))
  clearChat()
}
