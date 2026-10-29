<script>
  import { formatArgs } from '../lib/chat.svelte.js'
  import PriceChart from './PriceChart.svelte'
  import ForecastChart from './ForecastChart.svelte'
  import ReasoningChart from './ReasoningChart.svelte'

  let { item } = $props()

  const titles = { price: 'Price history', forecast: 'Forecast', reasoning: 'Model reasoning' }
  const caption = $derived(`${item.chart.ticker} · ${titles[item.chart.kind] ?? item.tool}`)
</script>

<figure>
  <figcaption class="label">{caption}</figcaption>
  {#if item.chart.kind === 'price'}
    <PriceChart chart={item.chart} />
  {:else if item.chart.kind === 'forecast'}
    <ForecastChart chart={item.chart} />
  {:else if item.chart.kind === 'reasoning'}
    <ReasoningChart chart={item.chart} />
  {:else}
    <div class="unknown label">Unsupported chart type: {item.chart.kind}</div>
  {/if}
</figure>

<style>
  figure {
    margin: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    overflow: hidden;
  }
  figcaption {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .unknown {
    padding: 16px 12px;
  }
</style>
