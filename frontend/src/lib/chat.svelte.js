// Shared app state. Socket events are applied here so components stay presentational.

export const chat = $state({
  messages: [],
  activity: [],
  connected: false,
  busy: false,
  models: [],
  model: null,
})

let nextId = 1

function now() {
  return new Date().toLocaleTimeString([], { hour12: false })
}

function log(text, kind = 'info') {
  chat.activity.push({ id: nextId++, time: now(), text, kind })
}

function current() {
  const last = chat.messages.at(-1)
  return last?.role === 'assistant' && last.pending ? last : null
}

export function formatArgs(args) {
  return Object.values(args ?? {})
    .filter((v) => v !== '' && v !== null && v !== undefined)
    .join(', ')
}

export function addQuery(text) {
  chat.messages.push({ id: nextId++, role: 'user', text })
  chat.messages.push({
    id: nextId++,
    role: 'assistant',
    text: '',
    tools: [],
    figures: [],
    pending: true,
    error: null,
  })
  chat.busy = true
  log(`QUERY  ${text.slice(0, 60)}`)
}

export function clearChat() {
  chat.messages = []
  chat.activity = []
}

export function setModel(id) {
  chat.model = id
  try {
    localStorage.setItem('model', id)
  } catch {
    // storage unavailable: the choice just lasts for this session
  }
}

export function setConnected(connected) {
  if (chat.connected === connected) return
  chat.connected = connected
  log(connected ? 'CONNECTED TO HARNESS' : 'CONNECTION LOST', connected ? 'ok' : 'error')
  if (!connected) {
    const msg = current()
    if (msg) {
      msg.pending = false
      msg.error = 'Connection to the harness was lost.'
    }
    chat.busy = false
  }
}

export function handleEvent(event) {
  const msg = current()

  switch (event.type) {
    case 'models': {
      chat.models = event.models
      let saved = null
      try {
        saved = localStorage.getItem('model')
      } catch {
        // ignore
      }
      const ids = event.models.map((m) => m.id)
      chat.model = ids.includes(chat.model) ? chat.model : ids.includes(saved) ? saved : event.default
      break
    }

    case 'status':
      log('MODEL THINKING')
      break

    case 'tool_start':
      msg?.tools.push({ id: event.id, name: event.name, arguments: event.arguments, status: 'running' })
      log(`CALL   ${event.name}(${formatArgs(event.arguments)})`, 'tool')
      break

    case 'tool_end': {
      const tool = msg?.tools.find((t) => t.id === event.id)
      if (tool) tool.status = event.is_error ? 'error' : 'done'
      log(`${event.is_error ? 'FAIL' : 'DONE'}   ${event.name}`, event.is_error ? 'error' : 'ok')
      break
    }

    case 'image':
      msg?.figures.push({ type: 'image', tool: event.tool, arguments: event.arguments, mime: event.mime, data: event.data })
      log(`FIGURE ${event.tool}`, 'ok')
      break

    case 'chart':
      msg?.figures.push({ type: 'chart', tool: event.tool, arguments: event.arguments, chart: event.chart })
      log(`CHART  ${event.chart.kind} ${event.chart.ticker}`, 'ok')
      break

    case 'answer':
      if (msg) {
        msg.text = event.text
        msg.pending = false
      }
      chat.busy = false
      log('ANSWER RECEIVED', 'ok')
      break

    case 'error':
      if (msg) {
        msg.error = event.message
        msg.pending = false
      }
      chat.busy = false
      log(`ERROR  ${event.message}`, 'error')
      break
  }
}
