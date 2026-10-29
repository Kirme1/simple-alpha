// Resolves the app's CSS variables to concrete colours for the chart libraries, which
// can't read var(--x) themselves. Call it inside an effect that depends on theme.mode.

export function palette() {
  const style = getComputedStyle(document.documentElement)
  const get = (name) => style.getPropertyValue(name).trim()
  return {
    text: get('--text'),
    muted: get('--muted'),
    faint: get('--faint'),
    border: get('--border'),
    borderStrong: get('--border-strong'),
    panel: get('--panel'),
    accent: get('--accent'),
    green: get('--green'),
    red: get('--red'),
    mono: get('--mono'),
  }
}

// '#rrggbb' -> 'rgba(r, g, b, alpha)'. Other formats are returned unchanged.
export function withAlpha(color, alpha) {
  const match = /^#([0-9a-f]{6})$/i.exec(color)
  if (!match) return color
  const n = parseInt(match[1], 16)
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}

export function pct(value, digits = 1) {
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}%`
}
