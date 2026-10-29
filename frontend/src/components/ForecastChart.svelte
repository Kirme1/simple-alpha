<script>
  import * as Plot from '@observablehq/plot'
  import { theme } from '../lib/theme.svelte.js'
  import { palette, pct } from '../lib/chartTheme.js'

  let { chart: data } = $props()

  let host
  let width = $state(0)

  const rows = $derived(
    data.days.map((day, i) => ({ day, q10: data.q10[i], q50: data.q50[i], q90: data.q90[i] })),
  )
  const last = $derived(rows.at(-1))

  $effect(() => {
    theme.mode // re-render with the new colours when the theme changes
    if (!width) return
    const c = palette()
    const milestones = rows.filter((r) => [5, 10, 15, 20].includes(r.day))

    const plot = Plot.plot({
      width,
      height: 280,
      marginTop: 28,
      marginLeft: 48,
      marginRight: 28,
      style: { background: 'transparent', color: c.muted, fontFamily: c.mono, fontSize: '11px', overflow: 'visible' },
      x: { label: 'Trading days ahead', ticks: [1, 5, 10, 15, 20], tickFormat: 'd', labelAnchor: 'center' },
      y: { label: 'Excess return vs SPY (%)', grid: true, nice: true, tickFormat: (d) => `${d}%` },
      marks: [
        Plot.ruleY([0], { stroke: c.faint, strokeDasharray: '3,3' }),
        Plot.areaY(rows, { x: 'day', y1: 'q10', y2: 'q90', fill: c.accent, fillOpacity: 0.15 }),
        Plot.lineY(rows, { x: 'day', y: 'q90', stroke: c.accent, strokeOpacity: 0.4 }),
        Plot.lineY(rows, { x: 'day', y: 'q10', stroke: c.accent, strokeOpacity: 0.4 }),
        Plot.lineY(rows, { x: 'day', y: 'q50', stroke: c.accent, strokeWidth: 2 }),
        Plot.dot(milestones, { x: 'day', y: 'q50', fill: c.accent, stroke: c.panel, r: 4 }),
        Plot.text(milestones, {
          x: 'day',
          y: 'q50',
          text: (d) => pct(d.q50),
          dy: -14,
          fill: c.text,
          stroke: c.panel,
          strokeWidth: 3,
          paintOrder: 'stroke',
          fontWeight: 600,
        }),
        Plot.tip(
          rows,
          Plot.pointerX({
            x: 'day',
            y: 'q50',
            fill: c.panel,
            stroke: c.borderStrong,
            title: (d) => `Day ${d.day}\nMedian ${pct(d.q50, 2)}\nRange ${pct(d.q10, 2)} to ${pct(d.q90, 2)}`,
          }),
        ),
      ],
    })

    host.replaceChildren(plot)
    return () => plot.remove()
  })
</script>

<div class="summary">
  <div>
    <span class="label">20-day median</span>
    <strong class:up={last.q50 >= 0} class:down={last.q50 < 0}>{pct(last.q50, 2)}</strong>
  </div>
  <div>
    <span class="label">80% range</span>
    <strong>{pct(last.q10, 1)} to {pct(last.q90, 1)}</strong>
  </div>
  <div>
    <span class="label">As of</span>
    <strong>{data.as_of}</strong>
  </div>
</div>
<div class="plot" bind:clientWidth={width}>
  <div bind:this={host}></div>
</div>

<style>
  .summary {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 28px;
    padding: 12px 14px 0;
  }
  .summary div {
    display: flex;
    flex-direction: column;
  }
  .summary strong {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
  }
  .up {
    color: var(--green);
  }
  .down {
    color: var(--red);
  }
  .plot {
    width: 100%;
    padding: 4px 12px 12px;
  }
</style>
