<script>
  import { createChart, CandlestickSeries, HistogramSeries, CrosshairMode } from 'lightweight-charts'
  import { theme } from '../lib/theme.svelte.js'
  import { palette, withAlpha, pct } from '../lib/chartTheme.js'

  let { chart: data } = $props()

  let el
  let hover = $state(null)

  const shown = $derived(hover ?? data.candles.at(-1))
  const change = $derived(shown ? (shown.close / shown.open - 1) * 100 : 0)

  $effect(() => {
    theme.mode // re-create the chart with the new colours when the theme changes
    const c = palette()

    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { color: c.panel },
        textColor: c.muted,
        fontFamily: c.mono,
        fontSize: 11,
      },
      grid: { vertLines: { color: c.border }, horzLines: { color: c.border } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: c.faint, labelBackgroundColor: c.accent },
        horzLine: { color: c.faint, labelBackgroundColor: c.accent },
      },
      // The chart sits inside a scrolling chat, so the wheel scrolls the page instead of zooming.
      handleScroll: { mouseWheel: false },
      handleScale: { mouseWheel: false },
    })

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: c.green,
      downColor: c.red,
      borderVisible: false,
      wickUpColor: c.green,
      wickDownColor: c.red,
    })
    candles.setData(data.candles)
    // leave room below for the volume bars and the TradingView attribution logo
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.3 } })

    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      lastValueVisible: false,
      priceLineVisible: false,
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.78, bottom: 0.12 } })
    volume.setData(
      data.candles.map((d) => ({
        time: d.time,
        value: d.volume,
        color: withAlpha(d.close >= d.open ? c.green : c.red, 0.35),
      })),
    )

    chart.timeScale().fitContent()

    chart.subscribeCrosshairMove((param) => {
      const bar = param.seriesData.get(candles)
      hover = bar ? { ...bar, volume: param.seriesData.get(volume)?.value } : null
    })

    return () => chart.remove()
  })
</script>

<div class="legend">
  <span class="label">{shown?.time ?? ''}</span>
  {#if shown}
    <span>O <b>{shown.open.toFixed(2)}</b></span>
    <span>H <b>{shown.high.toFixed(2)}</b></span>
    <span>L <b>{shown.low.toFixed(2)}</b></span>
    <span>C <b>{shown.close.toFixed(2)}</b></span>
    <span class:up={change >= 0} class:down={change < 0}>{pct(change, 2)}</span>
  {/if}
</div>
<div class="chart" bind:this={el}></div>

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    padding: 10px 12px 0;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
  }
  .legend b {
    color: var(--text);
    font-weight: 500;
  }
  .up {
    color: var(--green);
  }
  .down {
    color: var(--red);
  }
  .chart {
    height: 320px;
  }
</style>
