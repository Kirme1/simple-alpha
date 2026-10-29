// Light/dark theme. The choice is remembered in localStorage and falls back to the OS preference.

const KEY = 'theme'

function stored() {
  try {
    const value = localStorage.getItem(KEY)
    return value === 'light' || value === 'dark' ? value : null
  } catch {
    return null
  }
}

function preferred() {
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export const theme = $state({ mode: 'dark' })

function apply(mode) {
  theme.mode = mode
  document.documentElement.dataset.theme = mode
}

export function initTheme() {
  apply(stored() ?? preferred())
}

export function toggleTheme() {
  const mode = theme.mode === 'light' ? 'dark' : 'light'
  apply(mode)
  try {
    localStorage.setItem(KEY, mode)
  } catch {
    // storage unavailable: the choice just lasts for this session
  }
}
