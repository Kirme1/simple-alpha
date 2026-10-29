<script>
  import * as Plot from '@observablehq/plot'
  import { theme } from '../lib/theme.svelte.js'
  import { palette } from '../lib/chartTheme.js'

  let { chart: data } = $props()

  let featureHost
  let attentionHost
  let width = $state(0)

  const attention = $derived(
    data.attention.map((d) => ({ date: new Date(`${d.date}T00:00:00Z`), weight: d.weight })),
  )
  const recent = $derived(attention.slice(-data.recency_days))
  const wide = $derived(width >= 640)

  $effect(() => {
    theme.mode // re-render with the new colours when the theme changes
    if (!width) return
    const c = palette()
    const panelWidth = wide ? Math.floor((width - 24) / 2) : width
    const base = {
      width: panelWidth,
      marginTop: 24,
      marginRight: 36,
      style: { background: 'transparent', color: c.muted, fontFamily: c.mono, fontSize: '11px', overflow: 'visible' },
    }

    const longest = Math.max(...data.features.map((f) => f.name.length))
    const features = Plot.plot({
      ...base,
      height: data.features.length * 24 + 44,
      marginLeft: Math.min(longest * 6.8 + 12, panelWidth * 0.5),
      x: { label: 'Variable selection weight (%)', percent: true, ticks: 4, grid: true },
      y: { label: null },
      marks: [
        Plot.barX(data.features, { y: 'name', x: 'weight', fill: c.accent, sort: { y: '-x' }, rx: 2 }),
        Plot.text(data.features, {
          y: 'name',
          x: 'weight',
          text: (d) => `${(d.weight * 100).toFixed(1)}%`,
          textAnchor: 'start',
          dx: 5,
          fill: c.text,
        }),
      ],
    })

    const attn = Plot.plot({
      ...base,
      height: data.features.length * 24 + 44,
      marginLeft: 44,
      x: { type: 'utc', label: null, ticks: 4, tickFormat: '%b %d' },
      y: { label: 'Attention weight (%)', percent: true, grid: true, ticks: 4, nice: true },
      marks: [
        Plot.areaY(attention, { x: 'date', y: 'weight', fill: c.accent, fillOpacity: 0.18, curve: 'step' }),
        Plot.areaY(recent, { x: 'date', y: 'weight', fill: c.accent, fillOpacity: 0.5, curve: 'step' }),
        Plot.lineY(attention, { x: 'date', y: 'weight', stroke: c.accent, curve: 'step' }),
        Plot.tip(
          attention,
          Plot.pointerX({
            x: 'date',
            y: 'weight',
            fill: c.panel,
            stroke: c.borderStrong,
            title: (d) => `${d.date.toISOString().slice(0, 10)}\nAttention ${(d.weight * 100).toFixed(2)}%`,
          }),
        ),
      ],
    })

    featureHost.replaceChildren(features)
    attentionHost.replaceChildren(attn)
    return () => {
      features.remove()
      attn.remove()
    }
  })
</script>

<div class="panels" class:wide bind:clientWidth={width}>
  <section>
    <div class="label">Top drivers</div>
    <div bind:this={featureHost}></div>
  </section>
  <section>
    <div class="label">
      Attention ·
      <span class="strong">last {data.recency_days} sessions {(data.recency_score * 100).toFixed(1)}%</span>
    </div>
    <div bind:this={attentionHost}></div>
  </section>
</div>

<style>
  .panels {
    display: grid;
    gap: 16px 24px;
    padding: 12px 12px 12px;
  }
  .panels.wide {
    grid-template-columns: 1fr 1fr;
  }
  .strong {
    color: var(--text);
  }
</style>
